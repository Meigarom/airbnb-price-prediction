"""Contratos de entrada e saída da API (Pydantic).

Separados de `api.py` para manter os endpoints legíveis e os contratos de
dados explícitos.

Plano: docs/plans/13_deploy_producao.md §4.2
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Lote de anúncios crus do InsideAirbnb.

    Cada item é um dicionário livre: a base tem 96 colunas e enumerá-las no
    schema criaria um contrato frágil, que quebra a cada safra nova do
    InsideAirbnb. A validação real é feita pela `DataPreparation`, que exige
    as colunas obrigatórias e nomeia as que faltam.
    """

    listings: list[dict[str, Any]] = Field(
        ...,
        description='Anúncios crus. `price` é obrigatório em cada um.',
        min_length=1,
        max_length=10_000,
    )

    @field_validator('listings')
    @classmethod
    def _exige_price(cls, v: list[dict]) -> list[dict]:
        """`price` é obrigatório (decisão 5 do plano, §4.4).

        Nenhuma das 27 features usa `price` — ele serve para a regra de
        validade e para classificar a zona do recomendador. Sem ele não há
        recomendação, só predição, e esta versão não atende esse caso.
        """
        sem_price = [i for i, item in enumerate(v) if 'price' not in item]
        if sem_price:
            amostra = sem_price[:5]
            sufixo = '...' if len(sem_price) > 5 else ''
            raise ValueError(
                f"{len(sem_price)} anúncio(s) sem o campo obrigatório 'price'. "
                f'Índices: {amostra}{sufixo}'
            )
        return v


class PredictResult(BaseModel):
    """Um anúncio da resposta: a entrada original + as colunas de predição.

    Os campos de predição são nulos quando o anúncio viola uma regra de
    validade — a correspondência 1:1 com a entrada é preservada (§4.3).
    """

    price_estimate: float | None = Field(None, description='Preço justo estimado (€)')
    L: float | None = Field(None, description='Piso da banda: estimate × (1 − MAPE)')
    U: float | None = Field(None, description='Teto da banda: estimate × (1 + MAPE)')
    zona: str | None = Field(
        None,
        description='subprecificado | zona_morta | superprecificado | abaixo_do_piso',
    )
    uplift: float | None = Field(
        None, description='€/diária a ganhar; > 0 só em subprecificado'
    )
    status: str = Field(..., description="'ok' ou 'invalido: <motivo>'")

    model_config = {'extra': 'allow'}      # carrega as colunas originais


class PredictResponse(BaseModel):
    model_version: str
    mape: float = Field(..., description='MAPE validado; largura da banda')
    piso_recomendacao: float
    n_input: int
    n_predicted: int = Field(..., description='Quantos receberam predição')
    results: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    mape_validado: float | None = None
    mape_desvio: float | None = None
    n_features: int | None = None
    n_treino: int | None = None
    treinado_em: str | None = None
    data_referencia: str | None = None
    piso_recomendacao: float | None = None
    avisos: list[str] = Field(default_factory=list)
