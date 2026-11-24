"""Valores extraídos da base, usados como defaults da página Streamlit.

Gerado por `python -m app.gerar_defaults` — nenhum número aqui foi
escolhido à mão. Regenerar se a base mudar.

Por que este módulo existe separado de `app.py`: a página precisa de
medianas e centroides, e calculá-los no carregamento exigiria ler os 75 MB
do CSV a cada execução. Congelá-los aqui deixa a página instantânea e sem
dependência do arquivo de dados.

Plano: docs/plans/11_streamlit.md §3
"""

from __future__ import annotations

# Centroide (mediana de latitude/longitude) de cada bairro.
# O usuário escolhe o bairro e as coordenadas vêm daqui — pedir lat/long a
# um anfitrião não é razoável, e o modelo usa 4 features derivadas delas.
# Perda registrada: dois imóveis no mesmo bairro recebem a mesma distância
# ao centro, então a página é menos sensível à localização que o modelo.
CENTROIDES_BAIRRO: dict[str, tuple[float, float]] = {
    'Bijlmer-Centrum': (52.31809, 4.94875),
    'Bijlmer-Oost': (52.32156, 4.97722),
    'Bos en Lommer': (52.37808, 4.85207),
    'Buitenveldert - Zuidas': (52.33003, 4.87231),
    'Centrum-Oost': (52.3648, 4.90267),
    'Centrum-West': (52.3754, 4.88665),
    'De Aker - Nieuw Sloten': (52.34607, 4.797),
    'De Baarsjes - Oud-West': (52.36614, 4.86207),
    'De Pijp - Rivierenbuurt': (52.35233, 4.89712),
    'Gaasperdam - Driemond': (52.29984, 4.98034),
    'Geuzenveld - Slotermeer': (52.37853, 4.82297),
    'IJburg - Zeeburgereiland': (52.35646, 4.99468),
    'Noord-Oost': (52.39644, 4.952),
    'Noord-West': (52.41007, 4.90134),
    'Oostelijk Havengebied - Indische Buurt': (52.36446, 4.93802),
    'Osdorp': (52.35597, 4.79842),
    'Oud-Noord': (52.39126, 4.91635),
    'Oud-Oost': (52.35728, 4.91887),
    'Slotervaart': (52.35885, 4.83752),
    'Watergraafsmeer': (52.3507, 4.93114),
    'Westerpark': (52.38267, 4.87417),
    'Zuid': (52.3517, 4.85929),
}

# Categorias que o modelo conhece. Um `property_type` fora desta lista vira
# "Outro" no encoder — por isso a página oferece apenas estas.
ROOM_TYPES: list[str] = ['Entire home/apt', 'Private room', 'Shared room']

PROPERTY_TYPES: list[str] = ['Apartment', 'Bed and breakfast', 'Boat', 'Condominium', 'House', 'Houseboat', 'Loft', 'Townhouse']

# As comodidades mais frequentes da base. O modelo usa apenas a CONTAGEM
# (`n_amenities`), não quais são — a multisseleção existe para o usuário
# chegar a uma contagem realista sem digitar a string crua do InsideAirbnb.
AMENITIES_COMUNS: list[str] = [
    'Wifi',
    'Heating',
    'Essentials',
    'Kitchen',
    'Smoke detector',
    'TV',
    'Washer',
    'Hangers',
    'Laptop friendly workspace',
    'Iron',
    'Shampoo',
    'Hair dryer',
    'Hot water',
    'Dryer',
    'Family/kid friendly',
    'Fire extinguisher',
    'Cable TV',
    'Carbon monoxide detector',
    'Internet',
    'First aid kit',
    'Host greets you',
    'Refrigerator',
    'Private entrance',
    'Bed linens',
    'Dishes and silverware',
    'Paid parking off premises',
    'Coffee maker',
    'Oven',
    'Stove',
    'Cooking basics',
]

# Medianas da base, valor inicial de cada campo numérico.
MEDIANAS: dict[str, float] = {
    'accommodates': 2,
    'bedrooms': 1,
    'bathrooms': 1.0,
    'beds': 1,
    'minimum_nights': 2,
    'maximum_nights': 1125,
    'cleaning_fee': 35.0,
    'security_deposit': 150.0,
    'extra_people': 0.0,
    'number_of_reviews': 8,
    'calculated_host_listings_count': 1,
    'len_description': 1000,
    'len_house_rules': 57,
    'n_amenities': 18,
}

# Datas medianas. Viram features de tempo decorrido.
DATAS_MEDIANAS: dict[str, str] = {
    'host_since': '2014-11-25',
    'first_review': '2016-12-29',
    'last_review': '2018-08-26',
}

# Valores que `calendar_updated` assume na base, em texto livre.
CALENDAR_UPDATED: list[str] = ['today', 'a week ago', '2 months ago', '2 weeks ago', '3 months ago', '5 months ago', '4 months ago', '3 weeks ago', 'yesterday', '4 weeks ago', '3 days ago', '5 weeks ago', '6 weeks ago', '4 days ago']

# Data de referência do dataset. TODA feature de tempo é medida contra ela,
# não contra hoje: usar a data atual produziria idades de ~8 anos a mais e
# jogaria o anúncio para fora do domínio em que o modelo foi treinado.
DATA_REFERENCIA = '2018-12-06'
