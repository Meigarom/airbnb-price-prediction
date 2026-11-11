"""Treino e serialização do modelo de precificação — Airbnb Amsterdam.

Lê a base crua, reproduz o pipeline validado no notebook, treina o LightGBM
na base completa e grava dois artefatos:

    app/models/model.pkl      — o estimador
    app/models/artifacts.pkl  — os parâmetros aprendidos no treino

Por que dois: a preparação depende de 5 grupos de parâmetros aprendidos
(categorias de property_type, medianas de imputação, mapas ordinais, ordem
das features). Se produção os recalcular a partir do lote recebido, as
predições divergem SILENCIOSAMENTE do que foi validado (plano §2.1).

Uso:
    python -m app.train

Plano: docs/plans/13_deploy_producao.md §2
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data_preparation import (  # noqa: E402
    COLS_FLAGS,
    COLS_MONETARIAS,
    DataPreparation,
    REGRAS_VALIDADE,
)

RAIZ = Path(__file__).resolve().parent.parent
CSV_BRUTO = RAIZ / 'data' / 'listings_details.csv'
DIR_MODELOS = RAIZ / 'app' / 'models'

VERSAO = 'v1'
RANDOM_STATE = 42
DATA_REF = '2018-12-06'          # data do scrape (plano §2.2)

# Hiperparâmetros vencedores do Optuna — 120 trials, Etapa 08
PARAMS_LGBM = {
    'objective': 'regression',
    'n_estimators': 900,
    'learning_rate': 0.019833393362041398,
    'num_leaves': 53,
    'min_child_samples': 5,
    'subsample': 0.858247631044562,
    'subsample_freq': 4,
    'colsample_bytree': 0.5191979454584352,
    'reg_alpha': 0.11438965529570752,
    'reg_lambda': 1.7093305272785042e-08,
}

# As 27 features da Etapa 05 — ORDEM EXATA (LightGBM é sensível a ela)
FEATURES = [
    'accommodates', 'bathrooms', 'bedrooms', 'room_type',
    'latitude', 'longitude', 'dist_centro_km', 'dist_museumplein_km',
    'neighbourhood_cleansed', 'n_amenities', 'host_experiencia_dias',
    'minimum_nights', 'maximum_nights', 'cleaning_fee', 'taxa_limpeza_por_noite',
    'security_deposit', 'extra_people', 'calendar_atualizado_dias',
    'review_scores_rating', 'gap_value_vs_geral', 'number_of_reviews',
    'reviews_per_month', 'dias_desde_primeira_review', 'dias_desde_ultima_review',
    'availability_30', 'availability_365', 'len_house_rules',
]

POLOS = {
    'dist_centro_km': (52.3731, 4.8922),        # Dam Square
    'dist_museumplein_km': (52.3579, 4.8816),   # Museumplein
}

# Blocos de imputação por mecanismo (Análise 3.4)
COLS_MNAR_ZERO = ['security_deposit', 'cleaning_fee', 'taxa_limpeza_por_noite']
COLS_MAR_SENTINELA = ['review_scores_rating', 'gap_value_vs_geral',
                      'reviews_per_month', 'dias_desde_primeira_review',
                      'dias_desde_ultima_review']
COLS_MCAR = ['bathrooms', 'bedrooms', 'host_experiencia_dias',
             'calendar_atualizado_dias']
COLS_CATEGORICAS = ['room_type', 'neighbourhood_cleansed']

# Métricas validadas (Seção CV — 5 folds, out-of-sample, 19.981 anúncios)
MAPE_CV = 0.2221
MAPE_CV_DESVIO = 0.0013
MAPE_POR_FOLD = [0.2202, 0.2217, 0.2215, 0.2236, 0.2233]
MAPE_HOLDOUT = 0.2193
PISO_RECOMENDACAO = 75.0


def carregar_e_limpar() -> pd.DataFrame:
    """Carga + Etapa 02, removendo as linhas inválidas (como no treino original)."""
    df = pd.read_csv(CSV_BRUTO, low_memory=False)
    print(f'  CSV bruto: {len(df):,} linhas x {df.shape[1]} colunas'.replace(',', '.'))

    for col in COLS_MONETARIAS:
        if col in df.columns and df[col].dtype == object:
            df[col] = (df[col].astype(str)
                       .str.replace(r'[\$,]', '', regex=True)
                       .replace({'nan': np.nan}).astype(float))
    for col in COLS_FLAGS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'f': 0, 't': 1})

    # No TREINO as regras removem linhas (em produção apenas marcam)
    n0 = len(df)
    for descricao, regra in REGRAS_VALIDADE:
        viola = regra(df).fillna(False)
        if viola.any():
            print(f'    removendo {int(viola.sum()):>3} linhas: {descricao}')
        df = df[~viola]
    print(f'  após limpeza: {len(df):,} linhas (−{n0 - len(df)})'.replace(',', '.'))
    return df.reset_index(drop=True)


def remover_precos_inconsistentes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove anúncios com preço absurdo face aos comparáveis (notebook cél. 72).

    Regra da Etapa 04 (EDA), não da 02: descarta quem cobra mais de 5x ou menos
    de 1/4 da mediana do seu grupo (bairro, room_type, accommodates), apenas em
    grupos com massa crítica (>= 10 anúncios).

    É EXCLUSIVA DO TREINO. Depende de `price`, e em produção não se descarta um
    anúncio por ele ser caro — lá a regra não faz sentido e não é aplicada.

    DIVERGÊNCIA CONHECIDA vs o notebook (2 linhas em 19.981, 0,01%): lá as
    medianas de grupo foram calculadas apenas sobre o treino, porque o hold-out
    já estava separado naquele ponto; aqui usam a base inteira, já que o modelo
    de produção treina em tudo. O efeito são 44 remoções em vez de 42.
    """
    grupo = ['neighbourhood_cleansed', 'room_type', 'accommodates']
    ref = df.groupby(grupo, observed=True)['price'].transform('median')
    n_grupo = df.groupby(grupo, observed=True)['price'].transform('size')
    razao = df['price'] / ref

    confiavel = n_grupo >= 10
    remover = confiavel & ((razao > 5.00) | (razao < 0.25))
    print(f'    removendo {int(remover.sum()):>3} linhas: preço > 5x ou < 0,25x '
          f'os comparáveis (grupos com n >= 10)')
    return df[~remover.fillna(False)].reset_index(drop=True)


def aprender_artefatos(df: pd.DataFrame) -> dict:
    """Aprende os 5 grupos de parâmetros que produção precisa reutilizar."""
    # 1. CATEGORIAS_MANTIDAS: corte de 1% em property_type ('Boat' por decisão
    #    de negócio da Etapa 03)
    freq = df['property_type'].value_counts(normalize=True)
    categorias = sorted(set(freq[freq >= 0.01].index) | {'Boat'})
    print(f'  categorias_mantidas: {len(categorias)} níveis')

    # 2. Mapas ordinais das categóricas
    mapas = {}
    for c in COLS_CATEGORICAS:
        niveis = sorted(df[c].dropna().unique())
        mapas[c] = {v: i for i, v in enumerate(niveis)}
        print(f'  mapa ordinal {c}: {len(niveis)} níveis')

    artefatos = {
        'features': FEATURES,
        'categorias_mantidas': categorias,
        'mapas_ordinal': mapas,
        'cols_mnar_zero': COLS_MNAR_ZERO,
        'cols_mar_sentinela': COLS_MAR_SENTINELA,
        'cols_mcar': COLS_MCAR,
        'medianas_mcar': {},        # preenchido após o feature engineering
        'medianas_mar': {},         # rastreabilidade (caminho linear)
        'mape_validado': MAPE_CV,
        'mape_desvio': MAPE_CV_DESVIO,
        'mape_por_fold': MAPE_POR_FOLD,
        'mape_holdout': MAPE_HOLDOUT,
        'piso_recomendacao': PISO_RECOMENDACAO,
        'data_referencia': DATA_REF,
        'polos_geograficos': POLOS,
        'versao': VERSAO,
        'params_modelo': PARAMS_LGBM,
        'ambiente': {
            'python': platform.python_version(),
            'lightgbm': lightgbm.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
        },
    }
    return artefatos


def main() -> None:
    print('=' * 70)
    print(f'TREINO — modelo de precificação {VERSAO}')
    print('=' * 70)

    print('\n[1/5] Carga e limpeza (Etapas 02 e 04)')
    df = carregar_e_limpar()
    df = remover_precos_inconsistentes(df)
    print(f'  base de treino: {len(df):,} linhas'.replace(',', '.'))

    print('\n[2/5] Aprendendo parâmetros do treino')
    artefatos = aprender_artefatos(df)

    print('\n[3/5] Feature engineering (Etapa 03)')
    # A classe é usada aqui mesmo: garante que treino e produção percorram
    # EXATAMENTE o mesmo código — a única defesa real contra divergência.
    prep = DataPreparation(artefatos)
    d = prep.clean(df)
    d = prep.feature_engineering(d)
    print(f'  {d.shape[1]} colunas após as features')

    # As medianas MCAR só podem ser calculadas depois que as features existem
    artefatos['medianas_mcar'] = {c: float(d[c].median()) for c in COLS_MCAR}
    artefatos['medianas_mar'] = {c: float(d[c].median()) for c in COLS_MAR_SENTINELA
                                 if c in d.columns}
    print('  medianas MCAR aprendidas:')
    for c, v in artefatos['medianas_mcar'].items():
        print(f'    {c:<28} {v:>10.2f}')

    print('\n[4/5] Preparação e treino (Etapas 06 + 07/08)')
    prep = DataPreparation(artefatos)          # recriado com as medianas
    X = prep.prepare(d.drop(columns=['_status']))
    y_log = np.log1p(df['price'].values)
    print(f'  X: {X.shape[0]:,} x {X.shape[1]}'.replace(',', '.'))
    assert X.isna().sum().sum() == 0, 'nulos na matriz de treino'
    assert list(X.columns) == FEATURES, 'ordem das features divergiu'

    modelo = LGBMRegressor(**PARAMS_LGBM, random_state=RANDOM_STATE,
                           n_jobs=-1, verbose=-1)
    modelo.fit(X, y_log)
    artefatos['n_treino'] = int(len(X))
    artefatos['treinado_em'] = datetime.now(timezone.utc).isoformat()
    print(f'  modelo treinado em {len(X):,} anúncios'.replace(',', '.'))

    # Sanidade: o erro IN-SAMPLE é otimista por construção (15,6% na Etapa 10).
    # Serve só para detectar um treino quebrado, não para avaliar o modelo.
    pred = np.expm1(modelo.predict(X))
    real = df['price'].values
    mape_in = float(np.mean(np.abs(real - pred) / real))
    print(f'  MAPE in-sample: {mape_in*100:.2f}% '
          f'(otimista; o honesto é {MAPE_CV*100:.2f}% da CV)')
    assert mape_in < MAPE_CV, 'MAPE in-sample pior que o da CV — treino suspeito'

    print('\n[5/5] Serialização')
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, DIR_MODELOS / 'model.pkl')
    joblib.dump(artefatos, DIR_MODELOS / 'artifacts.pkl')
    for nome in ['model.pkl', 'artifacts.pkl']:
        kb = (DIR_MODELOS / nome).stat().st_size / 1024
        print(f'  {nome:<16} {kb:>8.0f} KB')

    # MLflow: registro histórico, NÃO o caminho de serving (plano §6.2).
    # Backend SQLite: o MLflow 3.x descontinuou o file store ('./mlruns'), que
    # entrou em modo de manutenção e lança exceção.
    try:
        import mlflow
        RAIZ.joinpath('mlruns').mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f'sqlite:///{RAIZ / "mlruns" / "mlflow.db"}')
        mlflow.set_experiment('airbnb-amsterdam-precificacao')
        with mlflow.start_run(run_name=f'treino-{VERSAO}'):
            mlflow.log_params({**PARAMS_LGBM,
                               'n_treino': artefatos['n_treino'],
                               'n_features': len(FEATURES),
                               'data_referencia': DATA_REF})
            mlflow.log_metrics({'mape_cv': MAPE_CV, 'mape_cv_desvio': MAPE_CV_DESVIO,
                                'mape_holdout': MAPE_HOLDOUT, 'mape_in_sample': mape_in,
                                **{f'mape_fold_{i}': m
                                   for i, m in enumerate(MAPE_POR_FOLD, 1)}})
            mlflow.set_tags({'versao': VERSAO,
                             'dataset': 'InsideAirbnb Amsterdam 2018-12',
                             'modelo': 'LightGBM',
                             **{f'lib_{k}': v
                                for k, v in artefatos['ambiente'].items()}})
            mlflow.log_artifact(str(DIR_MODELOS / 'model.pkl'))
            mlflow.log_artifact(str(DIR_MODELOS / 'artifacts.pkl'))
        print('  MLflow: run registrada em mlruns/')
    except Exception as exc:                                  # noqa: BLE001
        print(f'  MLflow indisponível ({type(exc).__name__}) — pickles gravados')

    print('\n' + '=' * 70)
    print(f'  OK — {VERSAO} | {artefatos["n_treino"]:,} anúncios | '
          f'MAPE honesto {MAPE_CV*100:.2f}%'.replace(',', '.'))
    print('=' * 70)


if __name__ == '__main__':
    main()
