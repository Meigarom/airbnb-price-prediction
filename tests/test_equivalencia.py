"""Teste de equivalência: pipeline do notebook ↔ classe de produção.

O item mais importante do plano de deploy (§5). Em produção não há segunda
chance de conferir: um erro de transcrição numa feature produz predições
plausíveis e erradas, indefinidamente.

A fixture foi gerada por `tests/gerar_fixture.py`, que transcreve o pipeline
de forma INDEPENDENTE da classe `DataPreparation`. Comparar a classe consigo
mesma não provaria nada.

Critério: diferença < €0,01 por anúncio. Não é "próximo" — é idêntico.

Uso:
    python -m pytest tests/test_equivalencia.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app.data_preparation import DataPreparation  # noqa: E402

TOLERANCIA_EUR = 0.01


@pytest.fixture(scope='module')
def artefatos():
    return joblib.load(RAIZ / 'app' / 'models' / 'artifacts.pkl')


@pytest.fixture(scope='module')
def modelo():
    return joblib.load(RAIZ / 'app' / 'models' / 'model.pkl')


@pytest.fixture(scope='module')
def amostra():
    caminho = RAIZ / 'tests' / 'fixtures' / 'amostra_notebook.csv'
    if not caminho.exists():
        pytest.skip('fixture ausente — rode tests/gerar_fixture.py')
    return pd.read_csv(caminho, low_memory=False)


def test_equivalencia_predicoes(artefatos, modelo, amostra):
    """O caminho de produção reproduz EXATAMENTE a referência."""
    esperado = amostra['price_estimate_esperado'].values
    cru = amostra.drop(columns=['price_estimate_esperado'])

    prep = DataPreparation(artefatos)
    X = prep.transform(cru)
    obtido = np.expm1(modelo.predict(X))

    diff = np.abs(obtido - esperado)
    pior = int(np.argmax(diff))

    assert diff.max() < TOLERANCIA_EUR, (
        f'{int((diff >= TOLERANCIA_EUR).sum())} de {len(diff)} anúncios divergem.\n'
        f'  pior caso: id={cru.iloc[pior]["id"]} '
        f'esperado=€{esperado[pior]:.4f} obtido=€{obtido[pior]:.4f} '
        f'(diferença €{diff.max():.4f})\n'
        f'  diferença mediana: €{np.median(diff):.6f}'
    )


def test_matriz_de_features(artefatos, amostra):
    """27 colunas, ordem exata do treino, sem nulos."""
    prep = DataPreparation(artefatos)
    X = prep.transform(amostra.drop(columns=['price_estimate_esperado']))

    assert list(X.columns) == list(artefatos['features']), 'ordem das features divergiu'
    assert X.shape == (len(amostra), 27), f'shape inesperado: {X.shape}'
    assert X.isna().sum().sum() == 0, 'nulos após a preparação'
    assert np.isfinite(X.values).all(), 'inf na matriz de features'


def test_escala_em_euro(artefatos, modelo, amostra):
    """`price_estimate` em euro — `expm1` aplicado, não valores em log."""
    prep = DataPreparation(artefatos)
    X = prep.transform(amostra.drop(columns=['price_estimate_esperado']))
    pred = np.expm1(modelo.predict(X))

    assert pred.min() > 10, f'mínimo €{pred.min():.2f} — parece escala logarítmica'
    assert 50 < np.median(pred) < 400, f'mediana €{np.median(pred):.2f} fora do plausível'


def test_correspondencia_um_para_um(artefatos, amostra):
    """Tantas linhas na saída quantas na entrada (plano §4.3)."""
    cru = amostra.drop(columns=['price_estimate_esperado'])
    prep = DataPreparation(artefatos)
    X = prep.transform(cru)

    assert len(X) == len(cru)
    assert len(prep.ultimo_status) == len(cru)


def test_classe_nunca_aprende():
    """Nenhuma chamada a `.fit(` em data_preparation.py (plano §11).

    Examina só o CÓDIGO: comentários e docstrings mencionam `fit_transform`
    justamente para explicar que a classe não o usa.
    """
    import ast
    import io
    import tokenize

    caminho = RAIZ / 'app' / 'data_preparation.py'
    fonte = caminho.read_text(encoding='utf-8')

    # Remove comentários; docstrings saem como nós Expr na checagem de AST
    sem_comentarios = []
    for tok in tokenize.generate_tokens(io.StringIO(fonte).readline):
        if tok.type != tokenize.COMMENT:
            sem_comentarios.append(tok.string)
    codigo = ' '.join(sem_comentarios)

    arvore = ast.parse(fonte)
    chamadas = [
        no.func.attr for no in ast.walk(arvore)
        if isinstance(no, ast.Call) and isinstance(no.func, ast.Attribute)
    ]
    proibidas = {'fit', 'fit_transform', 'fit_predict'}
    encontradas = sorted(set(chamadas) & proibidas)

    assert not encontradas, (
        f'a classe de produção não pode treinar nada, mas chama: {encontradas}'
    )
    assert 'fit_transform(' not in codigo
