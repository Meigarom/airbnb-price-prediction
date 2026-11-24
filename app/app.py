"""Protótipo visual do recomendador de preço — Airbnb Amsterdam.

Roda LOCALMENTE e chama a API na AWS. Dois propósitos (plano §Contexto):

  1. observar o deploy funcionando de forma visual
  2. mostrar o modelo às áreas de negócio

Não é produto para anfitriões reais. Mas justamente porque será mostrada a
pessoas que decidem com base nela, os avisos sobre a banda de erro, o piso de
€75 e a safra de 2018 são mais importantes aqui, não menos.

    streamlit run app/app.py

Exige credencial AWS: a Function URL usa autenticação IAM.

Plano: docs/plans/11_streamlit.md
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import cliente_aws
from app.defaults_app import (
    AMENITIES_COMUNS,
    CALENDAR_UPDATED,
    CENTROIDES_BAIRRO,
    DATA_REFERENCIA,
    DATAS_MEDIANAS,
    MEDIANAS,
    PROPERTY_TYPES,
    ROOM_TYPES,
)
from app.formulario import montar_listing

st.set_page_config(page_title='Precificação de Diárias — Amsterdam',
                   page_icon='🏠', layout='wide')

REF = date.fromisoformat(DATA_REFERENCIA)


def _data(rotulo: str, chave: str, ajuda: str) -> date:
    """Seletor de data limitado à safra do dataset.

    `max_value=REF` não é detalhe: sem ele o usuário escolheria uma data de
    2026 e produziria uma idade negativa — um valor que nenhuma validação do
    pipeline captura e que o modelo nunca viu no treino.
    """
    return st.date_input(
        rotulo,
        value=date.fromisoformat(DATAS_MEDIANAS[chave]),
        min_value=date(2008, 1, 1),
        max_value=REF,
        help=ajuda,
    )


# ===========================================================================
# Barra lateral — estado da API e modo de uso
# ===========================================================================

with st.sidebar:
    st.header('API na AWS')

    if st.button('Verificar conexão', use_container_width=True):
        st.session_state.pop('saude', None)

    if 'saude' not in st.session_state:
        try:
            with st.spinner('consultando /health...'):
                st.session_state['saude'] = cliente_aws.health()
        except cliente_aws.ErroAPI as exc:
            st.session_state['saude'] = None
            st.session_state['erro_saude'] = str(exc)

    saude_lat = st.session_state.get('saude')
    if saude_lat:
        saude, lat_saude = saude_lat
        st.success(f"modelo {saude['model_version']} carregado")
        st.caption(
            f"MAPE {saude['mape_validado'] * 100:.2f}% · "
            f"{saude['n_features']} features · {lat_saude:.1f}s"
        )
        for aviso in saude.get('avisos') or []:
            st.caption(f'aviso: {aviso}')
    else:
        st.error(st.session_state.get('erro_saude', 'API indisponível'))
        st.caption('A página precisa da API para funcionar.')

    st.divider()
    st.header('Modo')
    modo = st.radio(
        'O que você quer saber?',
        ['Quanto devo cobrar?', 'Meu preço está certo?'],
        help=(
            'O modelo não usa o preço para prever — ele o usa apenas para '
            'classificar a zona. No primeiro modo não há preço a informar, '
            'e a zona não é exibida porque não significaria nada.'
        ),
    )
    comparar = modo == 'Meu preço está certo?'

    preco_atual = None
    if comparar:
        preco_atual = st.number_input(
            'Seu preço hoje (€/noite)', min_value=1.0, max_value=2000.0,
            value=150.0, step=5.0,
        )

# ===========================================================================
# Cabeçalho
# ===========================================================================

st.title('Precificação de Diárias — Airbnb Amsterdam')
st.caption(
    'Protótipo. O modelo erra **±22,21% em média** — a faixa importa mais que '
    'o número. Dados de **dezembro de 2018**; Amsterdam restringiu aluguel de '
    'curta duração depois dessa safra, então os valores não refletem o '
    'mercado atual.'
)

# ===========================================================================
# Formulário — 30 campos distribuídos em abas
# ===========================================================================

aba_imovel, aba_regras, aba_hist, aba_textos = st.tabs(
    ['Imóvel', 'Regras e taxas', 'Histórico', 'Textos e comodidades']
)

with aba_imovel:
    c1, c2 = st.columns(2)
    with c1:
        bairro = st.selectbox('Bairro', sorted(CENTROIDES_BAIRRO),
                              index=sorted(CENTROIDES_BAIRRO).index('Centrum-West'))
        room_type = st.selectbox('Tipo de anúncio', ROOM_TYPES)
        property_type = st.selectbox('Tipo de imóvel', PROPERTY_TYPES,
                                     index=PROPERTY_TYPES.index('Apartment'))
    with c2:
        accommodates = st.number_input('Hóspedes', 1, 16,
                                       int(MEDIANAS['accommodates']))
        bedrooms = st.number_input('Quartos', 0, 10, int(MEDIANAS['bedrooms']))
        bathrooms = st.number_input('Banheiros', 0.0, 10.0,
                                    float(MEDIANAS['bathrooms']), step=0.5)
        beds = st.number_input('Camas', 1, 20, int(MEDIANAS['beds']))

    lat, lon = CENTROIDES_BAIRRO[bairro]
    st.caption(
        f'Coordenadas assumidas: {lat}, {lon} — o **centroide do bairro**. '
        'Dois imóveis no mesmo bairro recebem a mesma distância ao centro, '
        'então a página é menos sensível à localização que o modelo é capaz.'
    )

with aba_regras:
    c1, c2 = st.columns(2)
    with c1:
        minimum_nights = st.number_input('Mínimo de noites', 1, 365,
                                         int(MEDIANAS['minimum_nights']))
        maximum_nights = st.number_input('Máximo de noites', 1, 1125,
                                         int(MEDIANAS['maximum_nights']))
        availability_30 = st.slider(
            'Dias disponíveis nos próximos 30', 0, 30,
            int(MEDIANAS.get('availability_30', 0)),
            help='Disponibilidade no calendário — NÃO é ocupação.',
        )
        availability_365 = st.slider('Dias disponíveis no ano', 0, 365, 3)
    with c2:
        cleaning_fee = st.number_input('Taxa de limpeza (€)', 0.0, 500.0,
                                       float(MEDIANAS['cleaning_fee']), step=5.0)
        security_deposit = st.number_input('Depósito de segurança (€)', 0.0, 5000.0,
                                           float(MEDIANAS['security_deposit']),
                                           step=50.0)
        extra_people = st.number_input('Cobrança por hóspede extra (€)', 0.0, 200.0,
                                       float(MEDIANAS['extra_people']), step=5.0)
        calendar_updated = st.selectbox('Calendário atualizado', CALENDAR_UPDATED)

with aba_hist:
    c1, c2 = st.columns(2)
    with c1:
        number_of_reviews = st.number_input('Número de avaliações', 0, 1000,
                                            int(MEDIANAS['number_of_reviews']))
        reviews_per_month = st.number_input('Avaliações por mês', 0.0, 15.0,
                                            0.62, step=0.1)
        calculated_host_listings_count = st.number_input(
            'Anúncios deste anfitrião', 1, 100,
            int(MEDIANAS['calculated_host_listings_count']),
        )
    with c2:
        review_scores_rating = st.slider(
            'Nota geral (0–100)', 20, 100, 97,
            help='Escala 0–100 nesta safra do dataset, não 0–5.',
        )
        review_scores_value = st.slider('Nota de custo-benefício (0–10)', 2, 10, 9)

    st.caption(
        f'As datas abaixo viram **tempo decorrido** medido contra '
        f'**{REF.isoformat()}**, a data de referência do dataset — não contra hoje.'
    )
    c3, c4, c5 = st.columns(3)
    with c3:
        host_since = _data('Anfitrião desde', 'host_since',
                           'Vira a feature host_experiencia_dias.')
    with c4:
        first_review = _data('Primeira avaliação', 'first_review',
                             'Ignorada se não houver avaliações.')
    with c5:
        last_review = _data('Última avaliação', 'last_review',
                            'Ignorada se não houver avaliações.')

    if number_of_reviews == 0:
        st.info(
            'Sem avaliações: as datas e notas acima serão enviadas como '
            '**nulas**. Nota zero seria uma avaliação péssima; ausência de '
            'nota é outra coisa, e o pipeline distingue as duas.'
        )

with aba_textos:
    amenities = st.multiselect(
        'Comodidades', AMENITIES_COMUNS,
        default=AMENITIES_COMUNS[: int(MEDIANAS['n_amenities'])],
        help='O modelo usa apenas a CONTAGEM, não quais são.',
    )
    st.caption(f'{len(amenities)} comodidades — a mediana da base é '
               f'{int(MEDIANAS["n_amenities"])}.')

    c1, c2 = st.columns(2)
    with c1:
        len_description = st.number_input(
            'Tamanho da descrição (caracteres)', 0, 1000,
            int(MEDIANAS['len_description']),
        )
    with c2:
        len_house_rules = st.number_input(
            'Tamanho das regras da casa (caracteres)', 0, 5000,
            int(MEDIANAS['len_house_rules']),
        )
    st.caption(
        'O modelo usa o **comprimento** desses textos, não o conteúdo — por '
        'isso a página pede um número em vez do texto.'
    )

# ===========================================================================
# Chamada e resultado
# ===========================================================================

st.divider()

if st.button('Calcular preço justo', type='primary', use_container_width=True):
    listing = montar_listing(
        bairro=bairro, room_type=room_type, property_type=property_type,
        accommodates=accommodates, bedrooms=bedrooms, bathrooms=bathrooms,
        beds=beds, minimum_nights=minimum_nights, maximum_nights=maximum_nights,
        cleaning_fee=cleaning_fee, security_deposit=security_deposit,
        extra_people=extra_people, amenities=amenities,
        number_of_reviews=number_of_reviews, host_since=host_since,
        first_review=first_review, last_review=last_review,
        calendar_updated=calendar_updated,
        calculated_host_listings_count=calculated_host_listings_count,
        len_description=len_description, len_house_rules=len_house_rules,
        reviews_per_month=reviews_per_month,
        review_scores_rating=review_scores_rating,
        review_scores_value=review_scores_value,
        availability_30=availability_30, availability_365=availability_365,
        price=preco_atual,
    )

    try:
        with st.spinner('consultando a Lambda na AWS...'):
            resposta, latencia = cliente_aws.predict([listing])
    except cliente_aws.ErroAPI as exc:
        st.error(str(exc))
        st.stop()

    r = resposta['results'][0]
    if r.get('price_estimate') is None:
        st.error(f"A API não conseguiu precificar: {r.get('status')}")
        st.stop()

    est, inf, sup = r['price_estimate'], r['L'], r['U']
    piso = resposta['piso_recomendacao']

    # A faixa recebe peso visual igual ao do ponto: destacar só o número
    # seria enganoso com um erro médio de 22% (plano §5).
    c1, c2, c3 = st.columns(3)
    c1.metric('Preço justo estimado', f'€ {est:,.0f}'.replace(',', '.'))
    c2.metric('Mínimo da faixa', f'€ {inf:,.0f}'.replace(',', '.'))
    c3.metric('Máximo da faixa', f'€ {sup:,.0f}'.replace(',', '.'))

    st.progress(0.5, text=(
        f'A estimativa é € {est:.0f}, mas o valor real está, tipicamente, '
        f'entre € {inf:.0f} e € {sup:.0f} — o modelo erra '
        f'±{resposta["mape"] * 100:.1f}% em média.'
    ))

    if est < piso:
        st.warning(
            f'**Estimativa abaixo do piso de € {piso:.0f}.** Nessa faixa o '
            'modelo erra 38,5% e tende a prever ~29% acima do real. Trate '
            'este número como indicação fraca, não como recomendação.'
        )

    if comparar:
        zona = r['zona']
        if zona == 'subprecificado':
            st.success(
                f'**Subprecificado.** Seu preço (€ {preco_atual:.0f}) está '
                f'abaixo do mínimo da faixa. Subir para € {inf:.0f} '
                f'representaria **€ {r["uplift"]:.0f} a mais por noite**.'
            )
        elif zona == 'superprecificado':
            st.info(
                f'**Acima da faixa.** Seu preço (€ {preco_atual:.0f}) supera '
                f'€ {sup:.0f}. O modelo apenas sinaliza: baixar preço depende '
                'de ocupação, que não é medida aqui.'
            )
        elif zona == 'abaixo_do_piso':
            st.warning(
                f'**Abaixo do piso de € {piso:.0f}.** Nenhuma recomendação é '
                'emitida nessa faixa — o erro do modelo é grande demais.'
            )
        else:
            st.info(
                f'**Dentro da faixa.** Seu preço (€ {preco_atual:.0f}) cabe '
                f'entre € {inf:.0f} e € {sup:.0f}. A diferença em relação à '
                'estimativa é menor que o erro do modelo, então não há o que '
                'recomendar.'
            )
    else:
        st.caption(
            'A zona do recomendador não é exibida neste modo: ela seria '
            'calculada contra um preço que você não informou.'
        )

    # Procedência — é o que torna visível que o número veio da AWS, e não de
    # um cálculo local (plano §5.2).
    with st.expander('De onde veio esta resposta'):
        saude_atual = st.session_state.get('saude')
        st.markdown(
            f'- **Endpoint:** `{cliente_aws.FUNCTION_URL}`\n'
            f'- **Região:** {cliente_aws.REGIAO} · autenticação SigV4 (AWS_IAM)\n'
            f'- **Modelo:** {resposta["model_version"]}'
            + (f' · MAPE {saude_atual[0]["mape_validado"] * 100:.2f}%'
               if saude_atual else '')
            + f'\n- **Latência:** {latencia:.2f}s'
            + ('  (cold start — a imagem de 1,3 GB foi carregada agora)'
               if latencia > 5 else '  (container morno)')
            + f'\n- **Campos enviados:** {len(listing)}'
        )
        st.json(listing, expanded=False)
