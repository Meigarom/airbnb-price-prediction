"""Gera `app/defaults_app.py` a partir da base.

A página Streamlit precisa de centroides de bairro, medianas e listas de
categorias. Calculá-los no carregamento exigiria ler os 75 MB do CSV a cada
execução; escrevê-los à mão abriria espaço para números inventados.

Este script resolve os dois: extrai tudo da base e emite um módulo Python com
os valores congelados. Rode-o quando a base mudar.

    python -m app.gerar_defaults

Plano: docs/plans/11_streamlit.md §3
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CSV = RAIZ / 'data' / 'listings_details.csv'
SAIDA = RAIZ / 'app' / 'defaults_app.py'

N_AMENITIES_LISTADAS = 30


def _monetario(s: pd.Series) -> pd.Series:
    """`"$1,234.00"` -> 1234.0. O cifrão é herança do schema; a moeda é euro."""
    return pd.to_numeric(
        s.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce'
    )


def main() -> None:
    df = pd.read_csv(CSV, low_memory=False)
    artefatos = joblib.load(RAIZ / 'app' / 'models' / 'artifacts.pkl')

    centroides = (
        df.groupby('neighbourhood_cleansed')[['latitude', 'longitude']]
        .median()
        .round(5)
    )

    # `translation missing: ...` é sujeira do dataset, não uma comodidade.
    listas = (
        df.amenities.fillna('')
        .str.strip('{}')
        .str.replace('"', '', regex=False)
        .str.split(',')
    )
    contagem = Counter(x.strip() for lst in listas for x in lst if x.strip())
    amenities = [
        nome for nome, _ in contagem.most_common(N_AMENITIES_LISTADAS + 10)
        if not nome.startswith('translation missing')
    ][:N_AMENITIES_LISTADAS]

    medianas = {
        'accommodates': int(df.accommodates.median()),
        'bedrooms': int(df.bedrooms.median()),
        'bathrooms': float(df.bathrooms.median()),
        'beds': int(df.beds.median()),
        'minimum_nights': int(df.minimum_nights.median()),
        # 9999 é o sentinela de "sem máximo" e distorceria a mediana.
        'maximum_nights': int(df.maximum_nights.replace(9999, np.nan).median()),
        'cleaning_fee': float(_monetario(df.cleaning_fee).median()),
        'security_deposit': float(_monetario(df.security_deposit).median()),
        'extra_people': float(_monetario(df.extra_people).median()),
        'number_of_reviews': int(df.number_of_reviews.median()),
        'calculated_host_listings_count': int(
            df.calculated_host_listings_count.median()
        ),
        'len_description': int(df.description.fillna('').str.len().median()),
        'len_house_rules': int(df.house_rules.fillna('').str.len().median()),
        'n_amenities': int(df.amenities.str.count(',').add(1).median()),
    }

    datas = {
        c: str(pd.to_datetime(df[c], errors='coerce').median().date())
        for c in ('host_since', 'first_review', 'last_review')
    }
    calendario = list(df.calendar_updated.value_counts().head(14).index)

    print(f'base: {len(df):,} anúncios'.replace(',', '.'))
    print(f'  {len(centroides)} bairros, {len(amenities)} comodidades')
    print(f'escrevendo {SAIDA.relative_to(RAIZ)}')

    _escrever(SAIDA, centroides, artefatos, amenities, medianas, datas, calendario)
    print('pronto')


def _escrever(destino, centroides, artefatos, amenities, medianas, datas, calendario):
    p = []
    p.append('"""Valores extraídos da base, usados como defaults da página Streamlit.\n\n')
    p.append('Gerado por `python -m app.gerar_defaults` — nenhum número aqui foi\n')
    p.append('escolhido à mão. Regenerar se a base mudar.\n\n')
    p.append('Por que este módulo existe separado de `app.py`: a página precisa de\n')
    p.append('medianas e centroides, e calculá-los no carregamento exigiria ler os 75 MB\n')
    p.append('do CSV a cada execução. Congelá-los aqui deixa a página instantânea e sem\n')
    p.append('dependência do arquivo de dados.\n\n')
    p.append('Plano: docs/plans/11_streamlit.md §3\n"""\n\n')
    p.append('from __future__ import annotations\n\n')

    p.append('# Centroide (mediana de latitude/longitude) de cada bairro.\n')
    p.append('# O usuário escolhe o bairro e as coordenadas vêm daqui — pedir lat/long a\n')
    p.append('# um anfitrião não é razoável, e o modelo usa 4 features derivadas delas.\n')
    p.append('# Perda registrada: dois imóveis no mesmo bairro recebem a mesma distância\n')
    p.append('# ao centro, então a página é menos sensível à localização que o modelo.\n')
    p.append('CENTROIDES_BAIRRO: dict[str, tuple[float, float]] = {\n')
    for nome, linha in centroides.sort_index().iterrows():
        p.append(f'    {nome!r}: ({float(linha.latitude)}, {float(linha.longitude)}),\n')
    p.append('}\n\n')

    p.append('# Categorias que o modelo conhece. Um `property_type` fora desta lista vira\n')
    p.append('# "Outro" no encoder — por isso a página oferece apenas estas.\n')
    p.append(f'ROOM_TYPES: list[str] = {sorted(artefatos["mapas_ordinal"]["room_type"])!r}\n\n')
    p.append(f'PROPERTY_TYPES: list[str] = {sorted(artefatos["categorias_mantidas"])!r}\n\n')

    p.append('# As comodidades mais frequentes da base. O modelo usa apenas a CONTAGEM\n')
    p.append('# (`n_amenities`), não quais são — a multisseleção existe para o usuário\n')
    p.append('# chegar a uma contagem realista sem digitar a string crua do InsideAirbnb.\n')
    p.append('AMENITIES_COMUNS: list[str] = [\n')
    for nome in amenities:
        p.append(f'    {nome!r},\n')
    p.append(']\n\n')

    p.append('# Medianas da base, valor inicial de cada campo numérico.\n')
    p.append('MEDIANAS: dict[str, float] = {\n')
    for chave, valor in medianas.items():
        p.append(f'    {chave!r}: {valor},\n')
    p.append('}\n\n')

    p.append('# Datas medianas. Viram features de tempo decorrido.\n')
    p.append('DATAS_MEDIANAS: dict[str, str] = {\n')
    for chave, valor in datas.items():
        p.append(f'    {chave!r}: {valor!r},\n')
    p.append('}\n\n')

    p.append('# Valores que `calendar_updated` assume na base, em texto livre.\n')
    p.append(f'CALENDAR_UPDATED: list[str] = {calendario!r}\n\n')

    p.append('# Data de referência do dataset. TODA feature de tempo é medida contra ela,\n')
    p.append('# não contra hoje: usar a data atual produziria idades de ~8 anos a mais e\n')
    p.append('# jogaria o anúncio para fora do domínio em que o modelo foi treinado.\n')
    p.append(f'DATA_REFERENCIA = {artefatos["data_referencia"]!r}\n')

    destino.write_text(''.join(p))


if __name__ == '__main__':
    main()
