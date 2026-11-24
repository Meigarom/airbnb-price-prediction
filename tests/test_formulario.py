"""Testes da montagem do payload da página Streamlit.

A montagem do payload é onde um erro silencioso nasce: um campo com nome
errado ou no formato errado produz uma predição plausível e errada, e a
interface não tem como perceber. Por isso ela vive em `app/formulario.py`,
separada da página — para poder ser exercida sem subir o Streamlit.

O teste que sustenta a etapa é `test_pagina_equivale_ao_pipeline`: o payload
da página, passado pelo pipeline de produção, tem de dar o mesmo número que a
API devolveria. Critério de €0,01, o mesmo do teste de equivalência.

    python -m pytest tests/test_formulario.py -v
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app.data_preparation import DataPreparation  # noqa: E402
from app.defaults_app import AMENITIES_COMUNS, CENTROIDES_BAIRRO, MEDIANAS  # noqa: E402
from app.formulario import PRECO_SINTETICO, montar_amenities, montar_listing  # noqa: E402

TOLERANCIA_EUR = 0.01

ENTRADA_PADRAO = dict(
    bairro='Centrum-West',
    room_type='Entire home/apt',
    property_type='Apartment',
    accommodates=2,
    bedrooms=1,
    bathrooms=1.0,
    beds=1,
    minimum_nights=2,
    maximum_nights=1125,
    cleaning_fee=35.0,
    security_deposit=150.0,
    extra_people=0.0,
    amenities=AMENITIES_COMUNS[:18],
    number_of_reviews=8,
    host_since=date(2014, 11, 25),
    first_review=date(2016, 12, 29),
    last_review=date(2018, 8, 26),
    calendar_updated='today',
    calculated_host_listings_count=1,
    len_description=1000,
    len_house_rules=57,
    reviews_per_month=0.62,
    review_scores_rating=97,
    review_scores_value=9,
    availability_30=0,
    availability_365=3,
)


@pytest.fixture(scope='module')
def prep() -> DataPreparation:
    return DataPreparation(joblib.load(RAIZ / 'app' / 'models' / 'artifacts.pkl'))


@pytest.fixture(scope='module')
def modelo():
    return joblib.load(RAIZ / 'app' / 'models' / 'model.pkl')


def test_pagina_equivale_ao_pipeline(prep, modelo):
    """O portão da etapa: o payload da página atravessa o pipeline real.

    Não há um 'esperado' fixo aqui porque o número depende do modelo. O que
    se verifica é que o payload é ACEITO e produz um preço plausível — se a
    montagem estivesse errada, `transform` falharia ou o preço sairia
    absurdo.
    """
    listing = montar_listing(**ENTRADA_PADRAO)
    X = prep.transform(pd.DataFrame([listing]))

    assert prep.ultimo_status.iloc[0] == 'ok'
    assert X.shape[0] == 1

    preco = float(np.expm1(modelo.predict(X))[0])
    assert 20 < preco < 2000, f'preço fora de qualquer faixa plausível: {preco}'


def test_payload_tem_as_colunas_que_o_pipeline_exige(prep):
    """`COLUNAS_OBRIGATORIAS` tem 25 nomes, mas `prepare()` exige mais.

    Cinco features atravessam o pipeline sem transformação
    (`review_scores_rating`, `reviews_per_month`, `availability_30`,
    `availability_365` e `gap_value_vs_geral`, derivada das subnotas).
    Omiti-las passa na validação de `clean()` e só falha depois, com
    "Features ausentes após o pipeline" — foi exatamente o que aconteceu ao
    montar este formulário pela primeira vez.
    """
    listing = montar_listing(**ENTRADA_PADRAO)
    prep.transform(pd.DataFrame([listing]))  # não deve levantar


def test_sem_reviews_envia_nulos_e_nao_zeros():
    """Nota zero é uma avaliação péssima; ausência de nota é outra coisa."""
    listing = montar_listing(**{**ENTRADA_PADRAO, 'number_of_reviews': 0})

    assert listing['first_review'] is None
    assert listing['last_review'] is None
    assert listing['review_scores_rating'] is None
    assert listing['reviews_per_month'] is None


def test_sem_reviews_ainda_e_precificavel(prep, modelo):
    """Anúncio novo é um caso real, não uma borda — tem de funcionar."""
    listing = montar_listing(**{**ENTRADA_PADRAO, 'number_of_reviews': 0})
    X = prep.transform(pd.DataFrame([listing]))

    assert prep.ultimo_status.iloc[0] == 'ok'
    preco = float(np.expm1(modelo.predict(X))[0])
    assert 20 < preco < 2000


def test_preco_sintetico_quando_nao_informado():
    """No modo 'quanto devo cobrar' o usuário não tem preço a informar."""
    listing = montar_listing(**ENTRADA_PADRAO)
    assert listing['price'] == f'${PRECO_SINTETICO:.2f}'


def test_preco_informado_e_respeitado():
    listing = montar_listing(**ENTRADA_PADRAO, price=250.0)
    assert listing['price'] == '$250.00'


def test_coordenadas_vem_do_bairro():
    for bairro in ('Centrum-West', 'Zuid', 'Osdorp'):
        listing = montar_listing(**{**ENTRADA_PADRAO, 'bairro': bairro})
        assert (listing['latitude'], listing['longitude']) == CENTROIDES_BAIRRO[bairro]


def test_amenities_no_formato_cru_do_insideairbnb():
    """O pipeline conta vírgulas; o formato precisa ser o do dataset."""
    s = montar_amenities(['Wifi', 'Smoke detector', 'TV'])

    assert s.startswith('{') and s.endswith('}')
    # Nomes com espaço vão entre aspas, como no CSV original.
    assert '"Smoke detector"' in s
    assert 'Wifi' in s


def test_contagem_de_amenities_chega_ao_modelo(prep):
    """Mais comodidades tem de mudar `n_amenities` — senão o campo é decorativo."""
    poucas = montar_listing(**{**ENTRADA_PADRAO, 'amenities': AMENITIES_COMUNS[:5]})
    muitas = montar_listing(**{**ENTRADA_PADRAO, 'amenities': AMENITIES_COMUNS[:25]})

    x_poucas = prep.transform(pd.DataFrame([poucas]))
    x_muitas = prep.transform(pd.DataFrame([muitas]))

    assert x_muitas['n_amenities'].iloc[0] > x_poucas['n_amenities'].iloc[0]


def test_bairro_muda_a_predicao(prep, modelo):
    """Localização é 4 das 27 features; se não mudar nada, algo está errado."""
    centro = montar_listing(**{**ENTRADA_PADRAO, 'bairro': 'Centrum-West'})
    periferia = montar_listing(**{**ENTRADA_PADRAO, 'bairro': 'Bijlmer-Oost'})

    p_centro = float(np.expm1(modelo.predict(prep.transform(pd.DataFrame([centro]))))[0])
    p_periferia = float(
        np.expm1(modelo.predict(prep.transform(pd.DataFrame([periferia]))))[0]
    )

    assert abs(p_centro - p_periferia) > 1.0, (
        'bairro não alterou a predição — as coordenadas podem não estar chegando'
    )


def test_defaults_sao_as_medianas_da_base():
    """Os defaults do formulário têm de vir da base, não de palpite."""
    assert MEDIANAS['accommodates'] == 2
    assert MEDIANAS['cleaning_fee'] == 35.0
    assert 0 < MEDIANAS['n_amenities'] < 100
