"""Monta o payload de 25 colunas a partir do que o formulário coletou.

Separado de `app.py` de propósito: esta é a parte testável. A montagem do
payload é onde um erro silencioso nasce — um campo com nome errado ou no
formato errado produz uma predição plausível e errada, e a interface não tem
como perceber. Isolando-a aqui, o teste de equivalência (§7, passo 6) pode
exercê-la sem subir o Streamlit.

Plano: docs/plans/11_streamlit.md §2 e §3
"""

from __future__ import annotations

from datetime import date

from app.data_preparation import SUBNOTAS as _SUBNOTAS
from app.defaults_app import CENTROIDES_BAIRRO, MEDIANAS

# `price` existe no contrato da API mas NÃO é feature: o modelo não o usa para
# prever. A API o usa só para classificar a zona. No modo "quanto devo cobrar"
# o usuário não tem preço a informar, e este valor serve apenas para passar
# pela regra de validade `price > 0` — a zona resultante é DESCARTADA pela
# página, porque seria calculada contra um número inventado (plano §2.1).
PRECO_SINTETICO = 100.0

# `id` é identificador, não característica. A API o devolve intacto.
ID_SINTETICO = 999999999


def montar_amenities(selecionadas: list[str]) -> str:
    """Remonta a string crua do InsideAirbnb: `{Wifi,"Smoke detector",...}`.

    O modelo usa apenas a contagem, mas a string precisa estar no formato
    esperado porque `feature_engineering` conta as vírgulas.
    """
    itens = [f'"{a}"' if ' ' in a or '/' in a else a for a in selecionadas]
    return '{' + ','.join(itens) + '}'


def montar_listing(
    *,
    bairro: str,
    room_type: str,
    property_type: str,
    accommodates: int,
    bedrooms: int,
    bathrooms: float,
    beds: int,
    minimum_nights: int,
    maximum_nights: int,
    cleaning_fee: float,
    security_deposit: float,
    extra_people: float,
    amenities: list[str],
    number_of_reviews: int,
    host_since: date,
    first_review: date | None,
    last_review: date | None,
    calendar_updated: str,
    calculated_host_listings_count: int,
    len_description: int,
    len_house_rules: int,
    reviews_per_month: float,
    review_scores_rating: int,
    review_scores_value: int,
    availability_30: int,
    availability_365: int,
    price: float | None = None,
) -> dict:
    """Devolve o dicionário de entrada que a API exige.

    São 30 campos, não os 25 de `COLUNAS_OBRIGATORIAS`. Aquela lista é o que
    `clean()` valida; `prepare()` exige cinco features adicionais que
    atravessam o pipeline sem transformação — `review_scores_rating`,
    `reviews_per_month`, `availability_30`, `availability_365` e
    `gap_value_vs_geral`, esta última derivada das subnotas de review.
    Omiti-las faz `prepare()` falhar com "Features ausentes após o pipeline".

    `price=None` significa modo "quanto devo cobrar": usa o preço sintético.
    """
    latitude, longitude = CENTROIDES_BAIRRO[bairro]

    # Sem review, as datas precisam ser nulas e não uma data qualquer: o
    # pipeline trata ausência de review como mecanismo MNAR, e inventar uma
    # data mudaria a feature `dias_desde_ultima_review` de "nunca" para um
    # número — que é uma afirmação diferente.
    sem_reviews = number_of_reviews == 0

    return {
        'id': ID_SINTETICO,
        'price': f'${price if price is not None else PRECO_SINTETICO:.2f}',
        # -- imóvel --
        'neighbourhood_cleansed': bairro,
        'room_type': room_type,
        'property_type': property_type,
        'accommodates': accommodates,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'beds': beds,
        'latitude': latitude,
        'longitude': longitude,
        # -- regras e taxas --
        'minimum_nights': minimum_nights,
        'maximum_nights': maximum_nights,
        'cleaning_fee': f'${cleaning_fee:.2f}',
        'security_deposit': f'${security_deposit:.2f}',
        'extra_people': f'${extra_people:.2f}',
        # -- histórico --
        'number_of_reviews': number_of_reviews,
        'first_review': None if sem_reviews else str(first_review),
        'last_review': None if sem_reviews else str(last_review),
        'host_since': str(host_since),
        'calendar_updated': calendar_updated,
        'calculated_host_listings_count': calculated_host_listings_count,
        # Sem review não há nota: None, e não 0 — uma nota zero é uma
        # avaliação péssima, ausência de nota é outra coisa. O pipeline
        # distingue as duas (imputação por mecanismo, Etapa 06).
        'reviews_per_month': None if sem_reviews else reviews_per_month,
        'review_scores_rating': None if sem_reviews else review_scores_rating,
        # As subnotas alimentam `gap_value_vs_geral`. A página pede só
        # `value`; as demais recebem a nota geral convertida para 0–10, que
        # é a suposição mais neutra (gap zero nas que não foram perguntadas).
        **(
            {s: None for s in _SUBNOTAS}
            if sem_reviews
            else {
                **{s: round(review_scores_rating / 10) for s in _SUBNOTAS},
                'review_scores_value': review_scores_value,
            }
        ),
        # Disponibilidade no calendário. NÃO é ocupação — `availability_365`
        # alto significa agenda aberta, não imóvel vazio (CLAUDE.md §4).
        'availability_30': availability_30,
        'availability_365': availability_365,
        # -- textos --
        # O modelo usa o COMPRIMENTO de `house_rules`, não o conteúdo. A
        # página pede um número, e aqui ele vira texto do tamanho pedido:
        # fazer o usuário escrever 57 caracteres de regra não acrescentaria
        # nada ao que o modelo enxerga.
        'amenities': montar_amenities(amenities),
        'description': 'x' * len_description,
        'house_rules': 'x' * len_house_rules,
    }


def valores_iniciais() -> dict:
    """Defaults do formulário — as medianas da base."""
    return dict(MEDIANAS)
