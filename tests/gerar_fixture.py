"""Gera a fixture de referência do teste de equivalência (plano §5.1).

A referência é construída reproduzindo o pipeline do notebook de forma
INDEPENDENTE da classe `DataPreparation` — código transcrito aqui, não
importado de `app/`. Se os dois caminhos convergirem, a reimplementação em
`app/data_preparation.py` é fiel; se divergirem, há erro de transcrição.

Comparar a classe consigo mesma não provaria nada. É por isso que este
arquivo duplica a lógica em vez de importá-la.

Uso:
    python tests/gerar_fixture.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

N_AMOSTRA = 300
SEMENTE = 2024          # diferente de 42: amostra não ligada a nenhum split


def main() -> None:
    artifacts = joblib.load(RAIZ / 'app' / 'models' / 'artifacts.pkl')
    modelo = joblib.load(RAIZ / 'app' / 'models' / 'model.pkl')
    ref = pd.Timestamp(artifacts['data_referencia'])
    cats = set(artifacts['categorias_mantidas'])

    df = pd.read_csv(RAIZ / 'data' / 'listings_details.csv', low_memory=False)

    # --- Etapa 02 (transcrição independente) -------------------------------
    for col in ['price', 'security_deposit', 'cleaning_fee', 'extra_people']:
        df[col] = (df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
                   .replace({'nan': np.nan}).astype(float))
    for col in ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
                'is_location_exact', 'instant_bookable',
                'require_guest_profile_picture', 'require_guest_phone_verification']:
        df[col] = df[col].map({'f': 0, 't': 1})

    valido = ~((df.bathrooms > 10) | (df.minimum_nights > 365)
               | (df.maximum_nights == 9999) | (df.price <= 0)).fillna(False)
    df = df[valido].reset_index(drop=True)

    amostra = df.sample(N_AMOSTRA, random_state=SEMENTE).reset_index(drop=True)
    d = amostra.copy()

    # --- Etapa 03 (transcrição independente) -------------------------------
    for col in ['host_since', 'first_review', 'last_review']:
        d[col] = pd.to_datetime(d[col], errors='coerce')
    d['host_experiencia_dias'] = (ref - d['host_since']).dt.days
    d['dias_desde_primeira_review'] = (ref - d['first_review']).dt.days
    d['dias_desde_ultima_review'] = (ref - d['last_review']).dt.days

    def _cal(t):
        if pd.isna(t):
            return np.nan
        t = str(t).lower().strip()
        if t == 'today':
            return 0
        if t == 'yesterday':
            return 1
        if t in ('a week ago', 'one week ago'):
            return 7
        if 'never' in t:
            return np.nan
        m = re.match(r'(\d+)\s+(day|week|month)s?\s+ago', t)
        return int(m.group(1)) * {'day': 1, 'week': 7, 'month': 30}[m.group(2)] if m else np.nan

    d['calendar_atualizado_dias'] = d['calendar_updated'].apply(_cal)

    def _hav(la1, lo1, la2, lo2):
        la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return 6371.0 * 2 * np.arcsin(np.sqrt(a))

    for nome, (la, lo) in artifacts['polos_geograficos'].items():
        d[nome] = _hav(d['latitude'], d['longitude'], la, lo)

    def _amen(t):
        if pd.isna(t):
            return set()
        it = re.findall(r'"([^"]+)"|([^{},]+)', str(t))
        return {(a or b).strip() for a, b in it if (a or b).strip()}

    d['n_amenities'] = d['amenities'].apply(_amen).apply(len)

    d['taxa_limpeza_por_noite'] = d['cleaning_fee'] / d['minimum_nights'].replace(0, np.nan)

    subnotas = ['review_scores_accuracy', 'review_scores_cleanliness',
                'review_scores_checkin', 'review_scores_communication',
                'review_scores_location', 'review_scores_value']
    d['review_score_medio'] = d[subnotas].mean(axis=1)
    d['gap_value_vs_geral'] = d['review_scores_value'] - d['review_score_medio']

    d['len_house_rules'] = d['house_rules'].fillna('').astype(str).str.len()

    d['property_type_agrupado'] = d['property_type'].where(
        d['property_type'].isin(cats), 'Other')

    # --- Etapa 06 (transcrição independente) -------------------------------
    X = d.copy()
    for c in artifacts['cols_mnar_zero']:
        X[c] = X[c].fillna(0.0)
    for c in artifacts['cols_mar_sentinela']:
        X[c] = X[c].fillna(-1.0)
    for c, mapa in artifacts['mapas_ordinal'].items():
        X[c] = X[c].map(mapa).fillna(-1).astype(float)
    for c, med in artifacts['medianas_mcar'].items():
        X[c] = X[c].fillna(med)
    X = X[artifacts['features']].astype(float)

    assert X.isna().sum().sum() == 0, 'nulos na referência'

    pred = np.expm1(modelo.predict(X))

    fixture = amostra.copy()
    fixture['price_estimate_esperado'] = pred

    destino = RAIZ / 'tests' / 'fixtures' / 'amostra_notebook.csv'
    destino.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(destino, index=False)

    print(f'fixture: {len(fixture)} anúncios -> {destino.relative_to(RAIZ)}')
    print(f'  price_estimate: €{pred.min():.2f} a €{pred.max():.2f} '
          f'(mediana €{np.median(pred):.2f})')
    print(f'  MAPE da amostra: '
          f'{np.mean(np.abs(amostra["price"].values - pred) / amostra["price"].values)*100:.2f}%')


if __name__ == '__main__':
    main()
