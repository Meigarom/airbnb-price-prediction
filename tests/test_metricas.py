"""Testes das métricas de drift (app/metricas.py).

O que importa verificar aqui não é o valor em si — é o CONTRATO com os
alarmes que já existem no CloudWatch. Um nome de métrica ou namespace
divergente não produz erro em lugar nenhum: a métrica é publicada numa série
que nenhum alarme observa, o alarme fica em INSUFFICIENT_DATA para sempre, e
a falha só aparece quando alguém se pergunta por que o alarme nunca disparou.

Uso:
    python -m pytest tests/test_metricas.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app import metricas  # noqa: E402

# Os valores que os alarmes do Terraform consultam (terraform/cloudwatch.tf).
# Se este teste quebrar porque alguém renomeou uma métrica, o alarme
# correspondente precisa ser renomeado junto.
NAMESPACE_ESPERADO = 'Precificacao/Airbnb'
METRICAS_ESPERADAS = {'PriceEstimateMediana', 'TaxaSubprecificado', 'TaxaInvalidos'}


def _capturar(capsys) -> dict:
    """Lê o registro EMF que `emitir` imprimiu no stdout."""
    saida = capsys.readouterr().out
    for linha in saida.splitlines():
        if linha.startswith('{') and '_aws' in linha:
            return json.loads(linha)
    raise AssertionError(f'nenhum registro EMF no stdout:\n{saida}')


def test_namespace_casa_com_o_dos_alarmes(capsys):
    metricas.emitir(np.array([100.0, 200.0]), np.array(['zona_morta'] * 2), 2, 0)
    emf = _capturar(capsys)
    ns = emf['_aws']['CloudWatchMetrics'][0]['Namespace']
    assert ns == NAMESPACE_ESPERADO


def test_nomes_das_metricas_casam_com_os_alarmes(capsys):
    metricas.emitir(np.array([100.0]), np.array(['subprecificado']), 1, 0)
    emf = _capturar(capsys)
    nomes = {m['Name'] for m in emf['_aws']['CloudWatchMetrics'][0]['Metrics']}
    assert nomes == METRICAS_ESPERADAS


def test_sem_dimensoes():
    """Os alarmes consultam as métricas SEM dimensão.

    Acrescentar uma dimensão publicaria numa série diferente da observada —
    outro modo de falhar em silêncio.
    """
    reg = metricas.emitir(np.array([100.0]), np.array(['zona_morta']), 1, 0)
    assert reg['_aws']['CloudWatchMetrics'][0]['Dimensions'] == [[]]


def test_mediana_e_taxa_conferem():
    est = np.array([100.0, 200.0, 300.0])
    zonas = np.array(['subprecificado', 'zona_morta', 'zona_morta'])
    reg = metricas.emitir(est, zonas, n_input=3, n_invalido=0)

    assert reg['PriceEstimateMediana'] == 200.0
    assert reg['TaxaSubprecificado'] == pytest.approx(1 / 3, abs=1e-6)
    assert reg['TaxaInvalidos'] == 0.0


def test_lote_totalmente_invalido_nao_emite_mediana():
    """O caso que mais importa.

    Emitir mediana 0 num lote sem predição envenenaria a média de 24 h com um
    valor que não significa "mediana zero", e sim "não houve medição" — e
    poderia disparar o alarme de mediana-baixa por um motivo falso.
    """
    reg = metricas.emitir(None, None, n_input=5, n_invalido=5)

    assert reg['TaxaInvalidos'] == 1.0
    assert 'PriceEstimateMediana' not in reg
    assert 'TaxaSubprecificado' not in reg

    nomes = {m['Name'] for m in reg['_aws']['CloudWatchMetrics'][0]['Metrics']}
    assert nomes == {'TaxaInvalidos'}


def test_lote_vazio_nao_emite_nada():
    assert metricas.emitir(None, None, n_input=0, n_invalido=0) is None


@pytest.mark.parametrize(
    'kwargs',
    [
        {'price_estimate': 'não é array', 'zonas': 123, 'n_input': -5, 'n_invalido': None},
        {'price_estimate': np.array(['a']), 'zonas': None, 'n_input': 1, 'n_invalido': 0},
    ],
)
def test_nunca_levanta_excecao(kwargs):
    """Medir não pode derrubar a predição.

    A métrica é observabilidade, não o produto: qualquer falha aqui deve virar
    um aviso no log, nunca um 500 para quem pediu um preço.
    """
    assert metricas.emitir(**kwargs) is None


def test_emf_e_json_valido_em_uma_linha(capsys):
    """O agente do Lambda extrai as métricas lendo UMA linha de log.

    Um JSON quebrado em várias linhas não é reconhecido, e as métricas somem
    sem erro algum.
    """
    metricas.emitir(np.array([123.45]), np.array(['zona_morta']), 1, 0)
    saida = capsys.readouterr().out.strip()

    linhas = [c for c in saida.splitlines() if c.startswith('{')]
    assert len(linhas) == 1
    json.loads(linhas[0])
