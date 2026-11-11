"""Casos de borda da preparação e do recomendador (plano §5.3).

Uso:
    python -m pytest tests/test_data_preparation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app.data_preparation import ColunasAusentesError, DataPreparation  # noqa: E402


@pytest.fixture(scope='module')
def artefatos():
    return joblib.load(RAIZ / 'app' / 'models' / 'artifacts.pkl')


@pytest.fixture(scope='module')
def amostra():
    caminho = RAIZ / 'tests' / 'fixtures' / 'amostra_notebook.csv'
    if not caminho.exists():
        pytest.skip('fixture ausente — rode tests/gerar_fixture.py')
    df = pd.read_csv(caminho, low_memory=False)
    return df.drop(columns=['price_estimate_esperado'])


@pytest.fixture(scope='module')
def cliente():
    from app.api import app
    with TestClient(app) as c:
        yield c


# --- preparação -------------------------------------------------------------

def test_linhas_invalidas_sao_marcadas_nao_removidas(artefatos, amostra):
    """100 entram, 100 saem; as inválidas vêm rotuladas (plano §4.3)."""
    lote = amostra.head(100).copy().reset_index(drop=True)
    lote.loc[0, 'price'] = 0          # viola price <= 0
    lote.loc[1, 'bathrooms'] = 50     # viola bathrooms > 10
    lote.loc[2, 'minimum_nights'] = 400

    prep = DataPreparation(artefatos)
    X = prep.transform(lote)

    assert len(X) == 100, 'a correspondência 1:1 foi quebrada'
    invalidas = prep.ultimo_status.str.startswith('invalido')
    assert invalidas.sum() == 3
    assert 'price <= 0' in prep.ultimo_status.iloc[0]
    assert 'bathrooms > 10' in prep.ultimo_status.iloc[1]
    assert 'minimum_nights > 365' in prep.ultimo_status.iloc[2]
    assert (prep.ultimo_status.iloc[3:] == 'ok').all()


def test_coluna_obrigatoria_ausente_nomeia_o_que_falta(artefatos, amostra):
    """Erro explícito, não KeyError no meio do pipeline (plano §3.4)."""
    lote = amostra.head(5).drop(columns=['accommodates', 'latitude'])
    prep = DataPreparation(artefatos)

    with pytest.raises(ColunasAusentesError) as exc:
        prep.transform(lote)
    assert 'accommodates' in str(exc.value)
    assert 'latitude' in str(exc.value)


def test_categoria_nunca_vista_vira_menos_um(artefatos, amostra):
    """Bairro novo não quebra: vai para −1 (regra da Etapa 09)."""
    lote = amostra.head(10).copy().reset_index(drop=True)
    lote.loc[0, 'neighbourhood_cleansed'] = 'Bairro Inexistente'
    lote.loc[1, 'property_type'] = 'Castelo'

    prep = DataPreparation(artefatos)
    X = prep.transform(lote)

    assert X.loc[0, 'neighbourhood_cleansed'] == -1.0
    assert len(X) == 10
    assert X.isna().sum().sum() == 0


def test_nulos_sao_imputados_por_mecanismo(artefatos, amostra):
    """MNAR → 0, MAR → −1, MCAR → mediana do treino."""
    lote = amostra.head(20).copy().reset_index(drop=True)
    lote.loc[0, 'security_deposit'] = np.nan      # MNAR
    lote.loc[0, 'review_scores_rating'] = np.nan  # MAR
    lote.loc[0, 'bathrooms'] = np.nan             # MCAR

    prep = DataPreparation(artefatos)
    X = prep.transform(lote)

    assert X.loc[0, 'security_deposit'] == 0.0
    assert X.loc[0, 'review_scores_rating'] == -1.0
    assert X.loc[0, 'bathrooms'] == artefatos['medianas_mcar']['bathrooms']


def test_lote_de_uma_linha(artefatos, amostra):
    """O caso mínimo não quebra."""
    prep = DataPreparation(artefatos)
    X = prep.transform(amostra.head(1))
    assert X.shape == (1, 27)


# --- API --------------------------------------------------------------------

def test_health(cliente):
    r = cliente.get('/health')
    assert r.status_code == 200
    d = r.json()
    assert d['status'] == 'ok'
    assert d['model_loaded'] is True
    assert d['n_features'] == 27
    assert 0.20 < d['mape_validado'] < 0.25


def test_predict_devolve_tudo_que_recebeu(cliente, amostra):
    lote = amostra.head(50).replace({np.nan: None}).to_dict(orient='records')
    r = cliente.post('/predict', json={'listings': lote})

    assert r.status_code == 200
    d = r.json()
    assert d['n_input'] == 50
    assert len(d['results']) == 50, 'correspondência 1:1 quebrada'
    assert all('price_estimate' in x for x in d['results'])


def test_predict_zonas_exaustivas_e_piso_respeitado(cliente, amostra):
    """Cada anúncio em exatamente uma zona; nenhum uplift abaixo do piso."""
    lote = amostra.head(200).replace({np.nan: None}).to_dict(orient='records')
    d = cliente.post('/predict', json={'listings': lote}).json()
    res = pd.DataFrame(d['results'])

    validos = res[res.status == 'ok']
    zonas_esperadas = {'subprecificado', 'zona_morta', 'superprecificado',
                       'abaixo_do_piso'}
    assert set(validos.zona.unique()) <= zonas_esperadas
    assert validos.zona.notna().all(), 'anúncio válido sem zona'

    # Piso: nada abaixo de €75 pode gerar recomendação (plano §4.5)
    abaixo = validos[validos.price < d['piso_recomendacao']]
    assert (abaixo.uplift.fillna(0) == 0).all()
    assert (abaixo.zona != 'subprecificado').all()

    # uplift > 0 somente na zona subprecificado
    assert (validos[validos.zona != 'subprecificado'].uplift.fillna(0) == 0).all()
    subs = validos[validos.zona == 'subprecificado']
    if len(subs):
        assert (subs.uplift > 0).all()


def test_predict_banda_coerente(cliente, amostra):
    """L < estimate < U, com a largura do MAPE validado."""
    lote = amostra.head(30).replace({np.nan: None}).to_dict(orient='records')
    d = cliente.post('/predict', json={'listings': lote}).json()
    res = pd.DataFrame(d['results'])
    v = res[res.status == 'ok']

    assert (v.L < v.price_estimate).all()
    assert (v.price_estimate < v.U).all()
    largura = (v.U - v.L) / v.price_estimate
    assert np.allclose(largura, 2 * d['mape'], atol=0.01)


def test_predict_linha_invalida_vem_com_nulos(cliente, amostra):
    lote = amostra.head(10).replace({np.nan: None}).to_dict(orient='records')
    lote[0]['price'] = 0
    d = cliente.post('/predict', json={'listings': lote}).json()

    assert d['n_input'] == 10
    assert d['n_predicted'] == 9
    r0 = d['results'][0]
    assert r0['price_estimate'] is None
    assert r0['zona'] is None
    assert 'invalido' in r0['status']


def test_predict_sem_price_e_rejeitado(cliente, amostra):
    """`price` é obrigatório (decisão 5 do plano)."""
    lote = amostra.head(3).replace({np.nan: None}).to_dict(orient='records')
    del lote[1]['price']
    r = cliente.post('/predict', json={'listings': lote})
    assert r.status_code == 422
    assert 'price' in r.text


def test_predict_coluna_faltante_responde_422(cliente, amostra):
    lote = (amostra.head(3).drop(columns=['accommodates'])
            .replace({np.nan: None}).to_dict(orient='records'))
    r = cliente.post('/predict', json={'listings': lote})
    assert r.status_code == 422
    assert 'accommodates' in r.text


def test_predict_aceita_price_com_cifrao(cliente, amostra):
    """`price` no formato cru do InsideAirbnb ("$198.00") é aceito.

    Regressão: a fixture tem `price` já numérico, então o formato cru —
    que é o que um cliente real envia — não era exercitado. Causava HTTP 500.
    """
    lote = amostra.head(10).replace({np.nan: None}).to_dict(orient='records')
    for item in lote:
        item['price'] = f'${float(item["price"]):.2f}'

    r = cliente.post('/predict', json={'listings': lote})
    assert r.status_code == 200, f'formato monetário rejeitado: {r.text[:200]}'

    res = pd.DataFrame(r.json()['results'])
    assert res.price_estimate.notna().all()
    assert res.zona.notna().all(), 'zona não calculada com price em formato cru'


def test_predict_escala_em_euro(cliente, amostra):
    """`price_estimate` em euro, não em log (plano §11)."""
    lote = amostra.head(50).replace({np.nan: None}).to_dict(orient='records')
    d = cliente.post('/predict', json={'listings': lote}).json()
    est = pd.DataFrame(d['results']).price_estimate.dropna()

    assert est.min() > 10, 'parece escala logarítmica'
    assert 50 < est.median() < 400
