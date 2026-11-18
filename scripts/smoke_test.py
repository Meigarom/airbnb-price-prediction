"""Smoke test pós-deploy: confere que a Lambda está servindo o modelo.

Separado do workflow de propósito. Embutido no YAML, o script fica com três
níveis de aspas aninhadas (YAML → shell → python) — e um erro de escape só
aparece em produção, no pior momento possível.

Verifica o CONTEÚDO, não só o HTTP 200: a função pode responder 200 com o
modelo não carregado, e é exatamente esse o caso que o teste precisa pegar.

Uso:
    python scripts/smoke_test.py '<json do /health>'
"""

from __future__ import annotations

import json
import sys

MAPE_ESPERADO = 0.2221
N_FEATURES_ESPERADO = 27


def main() -> int:
    bruto = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()

    try:
        r = json.loads(bruto)
    except json.JSONDecodeError:
        print(f'FALHOU: resposta não é JSON.\n{bruto[:300]}', file=sys.stderr)
        return 1

    erros = []

    if r.get('status') != 'ok':
        erros.append(f"status={r.get('status')!r}, esperado 'ok'")

    if r.get('model_loaded') is not True:
        erros.append(f"model_loaded={r.get('model_loaded')!r}, esperado True")

    # Se o MAPE divergir, a imagem está servindo OUTRO modelo — o deploy
    # tecnicamente funcionou, mas entregou a coisa errada.
    mape = r.get('mape_validado')
    if mape is None or abs(mape - MAPE_ESPERADO) > 1e-6:
        erros.append(f'mape_validado={mape!r}, esperado {MAPE_ESPERADO}')

    if r.get('n_features') != N_FEATURES_ESPERADO:
        erros.append(f"n_features={r.get('n_features')!r}, esperado {N_FEATURES_ESPERADO}")

    if erros:
        print('SMOKE TEST FALHOU:', file=sys.stderr)
        for e in erros:
            print(f'  - {e}', file=sys.stderr)
        return 1

    print(f"OK — MAPE {mape:.4f}, {r['n_features']} features, modelo {r.get('model_version')}")

    # Avisos não derrubam o deploy, mas precisam ficar visíveis no log.
    for aviso in r.get('avisos') or []:
        print(f'  aviso: {aviso}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
