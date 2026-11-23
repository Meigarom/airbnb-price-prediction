"""Métricas de drift emitidas via EMF (CloudWatch Embedded Metric Format).

O contexto que justifica este módulo: **em produção não há hold-out**, então a
degradação do MAPE é invisível. É a pendência mais séria herdada do plano 13.
Nenhuma métrica aqui mede MAPE — isso exigiria preços reais posteriores, que
só existem semanas depois da predição. O que estas três detectam é **mudança
de regime**, que é o sintoma observável de drift.

Por que EMF e não `PutMetricData`:

    PutMetricData é uma chamada HTTP à API do CloudWatch. Numa Lambda, isso
    soma latência a toda requisição e pode falhar — e aí é preciso decidir se
    a falha de uma métrica derruba a predição (não deve). O EMF evita as duas
    coisas: basta imprimir um JSON no stdout, e o agente do Lambda extrai as
    métricas do log de forma assíncrona. Custo na requisição: um print.

Os três valores são emitidos POR REQUISIÇÃO, e os alarmes agregam com
`Average` sobre 24 h (ou 1 h, para inválidos). Isso significa que a média é
por requisição, não por anúncio: um lote de 1 anúncio pesa tanto quanto um de
1.000. É uma escolha consciente — a alternativa (emitir um valor por anúncio)
multiplicaria o volume de log pelo tamanho do lote, e o que se quer detectar
é deslocamento de regime, não a estatística exata da base.

Baselines medidos na base completa de 19.981 anúncios (seção de CV do
notebook), não estimados:

    mediana de price_estimate    €128      alarme: ±15%
    taxa de subprecificados      11,6%     alarme: +5 p.p.
    taxa de inválidos            0,04%     alarme: >2%

Plano: docs/plans/14_deploy_aws.md §3.5
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

NAMESPACE = os.environ.get('METRIC_NAMESPACE', 'Precificacao/Airbnb')

# Os nomes precisam casar EXATAMENTE com os dos alarmes já criados pelo
# Terraform (terraform/cloudwatch.tf). Um nome divergente não gera erro: a
# métrica é publicada num lugar que nenhum alarme observa, e o alarme fica
# em INSUFFICIENT_DATA para sempre — falha silenciosa, cara de descobrir.
METRICA_MEDIANA = 'PriceEstimateMediana'
METRICA_SUBPRECIFICADO = 'TaxaSubprecificado'
METRICA_INVALIDOS = 'TaxaInvalidos'


def emitir(
    price_estimate: np.ndarray | None,
    zonas: np.ndarray | None,
    n_input: int,
    n_invalido: int,
) -> dict | None:
    """Imprime um registro EMF no stdout com as três métricas de drift.

    Devolve o registro emitido (para teste) ou None se nada foi emitido.

    NUNCA levanta exceção. Uma falha ao medir não pode derrubar a predição:
    a métrica é observabilidade, não o produto.
    """
    try:
        metricas: list[dict] = []
        valores: dict = {}

        # Taxa de inválidos: sempre mensurável, mesmo quando nenhum anúncio
        # é válido — e é justamente aí que ela importa mais.
        if n_input > 0:
            valores[METRICA_INVALIDOS] = round(n_invalido / n_input, 6)
            metricas.append({'Name': METRICA_INVALIDOS, 'Unit': 'None'})

        # As outras duas só existem se houve predição. Emitir 0 num lote
        # totalmente inválido envenenaria a média com um valor que não
        # significa "mediana zero", e sim "não houve medição".
        if price_estimate is not None and len(price_estimate) > 0:
            valores[METRICA_MEDIANA] = round(float(np.median(price_estimate)), 2)
            metricas.append({'Name': METRICA_MEDIANA, 'Unit': 'None'})

            if zonas is not None and len(zonas) > 0:
                taxa = float(np.mean(zonas == 'subprecificado'))
                valores[METRICA_SUBPRECIFICADO] = round(taxa, 6)
                metricas.append({'Name': METRICA_SUBPRECIFICADO, 'Unit': 'None'})

        if not metricas:
            return None

        registro = {
            '_aws': {
                'Timestamp': int(time.time() * 1000),
                'CloudWatchMetrics': [
                    {
                        'Namespace': NAMESPACE,
                        # Dimensions vazio de propósito: os alarmes consultam
                        # as métricas SEM dimensão. Acrescentar uma aqui
                        # publicaria numa série diferente da observada.
                        'Dimensions': [[]],
                        'Metrics': metricas,
                    }
                ],
            },
            **valores,
            # Campos fora de `Metrics` viram propriedades do log, pesquisáveis
            # no Log Insights mas sem custo de métrica. Úteis no diagnóstico:
            # uma mediana estranha num lote de 2 anúncios não é drift.
            'n_input': n_input,
            'n_invalido': n_invalido,
        }

        print(json.dumps(registro), file=sys.stdout, flush=True)
        return registro

    # Amplo de propósito, e sem relançar: medir não pode derrubar a predição.
    except Exception as exc:
        print(f'AVISO: falha ao emitir métricas ({type(exc).__name__}: {exc})')
        return None
