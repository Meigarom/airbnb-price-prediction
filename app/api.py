"""API de precificação de diárias — Airbnb Amsterdam.

Recebe anúncios crus do InsideAirbnb e devolve o preço justo estimado, a
banda de confiança e a zona do recomendador.

    uvicorn app.api:app --reload --port 8000

Documentação interativa em /docs.

Plano: docs/plans/13_deploy_producao.md §4
"""

from __future__ import annotations

import platform
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.data_preparation import ColunasAusentesError, DataPreparation
from app.schemas import HealthResponse, PredictRequest, PredictResponse

RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ / 'app' / 'models'

# Estado carregado uma única vez, no startup (plano §4.6): um LightGBM de 900
# árvores por requisição inviabilizaria a latência.
ESTADO: dict = {'modelo': None, 'prep': None, 'artefatos': None, 'avisos': []}


def _conferir_ambiente(artefatos: dict) -> list[str]:
    """Compara as versões de lib com as do treino (plano §2.1).

    Pickle é sensível a versão: um modelo carregado com lightgbm diferente
    pode falhar silenciosamente. Avisa, não bloqueia — bloquear impediria
    uma atualização de patch legítima.
    """
    atual = {
        'python': platform.python_version(),
        'lightgbm': lightgbm.__version__,
        'pandas': pd.__version__,
        'numpy': np.__version__,
    }
    treino = artefatos.get('ambiente', {})
    return [
        f'{lib}: treino={treino[lib]} runtime={atual[lib]}'
        for lib in atual
        if lib in treino and treino[lib] != atual[lib]
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        artefatos = joblib.load(DIR_MODELOS / 'artifacts.pkl')
        ESTADO['modelo'] = joblib.load(DIR_MODELOS / 'model.pkl')
        ESTADO['artefatos'] = artefatos
        ESTADO['prep'] = DataPreparation(artefatos)
        ESTADO['avisos'] = _conferir_ambiente(artefatos)

        print(f"modelo {artefatos['versao']} carregado — "
              f"{artefatos['n_treino']:,} anúncios, "
              f"MAPE {artefatos['mape_validado']*100:.2f}%".replace(',', '.'))
        for aviso in ESTADO['avisos']:
            print(f'  AVISO divergência de ambiente — {aviso}')
    except FileNotFoundError:
        print(f'ERRO: artefatos não encontrados em {DIR_MODELOS}. '
              'Rode `python -m app.train`.')
    yield
    ESTADO.clear()


app = FastAPI(
    title='Precificação de Diárias — Airbnb Amsterdam',
    description=(
        'Estima o preço justo de anúncios e classifica a zona de precificação.\n\n'
        'O modelo opera com MAPE de 22,21% (validado out-of-sample em 19.981 '
        'anúncios). Anúncios abaixo de €75 recebem predição mas **não** '
        'recomendação: naquela faixa o erro é de 38,5% com viés de −28,7%.'
    ),
    version='1.0.0',
    lifespan=lifespan,
)


@app.get('/health', response_model=HealthResponse, tags=['operação'])
def health() -> HealthResponse:
    """Estado do serviço e metadados do modelo carregado."""
    art = ESTADO.get('artefatos')
    if art is None:
        return HealthResponse(status='degradado', model_version='nenhum',
                              model_loaded=False,
                              avisos=['artefatos não carregados — rode app/train.py'])
    return HealthResponse(
        status='ok',
        model_version=art['versao'],
        model_loaded=True,
        mape_validado=art['mape_validado'],
        mape_desvio=art['mape_desvio'],
        n_features=len(art['features']),
        n_treino=art['n_treino'],
        treinado_em=art['treinado_em'],
        data_referencia=art['data_referencia'],
        piso_recomendacao=art['piso_recomendacao'],
        avisos=ESTADO.get('avisos', []),
    )


@app.post('/predict', response_model=PredictResponse, tags=['predição'])
def predict(req: PredictRequest) -> PredictResponse:
    """Estima o preço justo de um lote de anúncios.

    Devolve **tantas linhas quantas recebeu** (§4.3). Anúncios que violam uma
    regra de validade voltam com os campos de predição nulos e `status`
    descrevendo o motivo.
    """
    if ESTADO.get('modelo') is None:
        raise HTTPException(503, 'Modelo não carregado. Rode `python -m app.train`.')

    art = ESTADO['artefatos']
    prep: DataPreparation = ESTADO['prep']
    mape = art['mape_validado']
    piso = art['piso_recomendacao']

    entrada = pd.DataFrame(req.listings)
    n_input = len(entrada)

    # `price` pode chegar como "$198.00" (formato cru do InsideAirbnb) ou já
    # numérico. A zona do recomendador precisa dele como float; `clean()` faz
    # essa conversão internamente, mas aqui comparamos contra a entrada ORIGINAL.
    preco_num = pd.to_numeric(
        entrada['price'].astype(str).str.replace(r'[\$,]', '', regex=True),
        errors='coerce',
    )

    try:
        X = prep.transform(entrada)
    except ColunasAusentesError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(500, f'Falha na preparação: {type(exc).__name__}: {exc}') from exc

    status = prep.ultimo_status.reset_index(drop=True)
    valido = ~status.str.startswith('invalido')

    saida = entrada.copy().reset_index(drop=True)
    for col in ['price_estimate', 'L', 'U', 'uplift']:
        saida[col] = np.nan
    saida['zona'] = None
    saida['status'] = status.values

    if valido.any():
        # A predição roda em todas as linhas (a matriz está completa), mas só
        # as válidas recebem resultado: um anúncio com price <= 0 não tem zona.
        pred = np.expm1(ESTADO['modelo'].predict(X))
        idx = saida.index[valido]

        est = pred[valido.values]
        inf = est * (1 - mape)
        sup = est * (1 + mape)
        preco = preco_num.loc[idx].values

        zona = np.where(
            preco < inf,
            np.where(preco >= piso, 'subprecificado', 'abaixo_do_piso'),
            np.where(preco > sup, 'superprecificado', 'zona_morta'),
        )
        uplift = np.where(zona == 'subprecificado', inf - preco, 0.0)

        saida.loc[idx, 'price_estimate'] = np.round(est, 2)
        saida.loc[idx, 'L'] = np.round(inf, 2)
        saida.loc[idx, 'U'] = np.round(sup, 2)
        saida.loc[idx, 'zona'] = zona
        saida.loc[idx, 'uplift'] = np.round(uplift, 2)

    # NaN não é JSON válido; None é.
    registros = saida.replace({np.nan: None}).to_dict(orient='records')

    return PredictResponse(
        model_version=art['versao'],
        mape=mape,
        piso_recomendacao=piso,
        n_input=n_input,
        n_predicted=int(valido.sum()),
        results=registros,
    )
