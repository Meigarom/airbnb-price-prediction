"""Cliente da Function URL, com assinatura SigV4.

A Function URL está configurada com `authorization_type = AWS_IAM`: toda
requisição precisa ser assinada. Uma URL pública exporia o modelo a qualquer
um que a descobrisse.

`requests` não assina SigV4 sozinha. Aqui a assinatura é feita por `botocore`
— as credenciais vêm da cadeia padrão da AWS (perfil, variáveis de ambiente,
role), e nenhuma chave fica no código.

Plano: docs/plans/11_streamlit.md §6
"""

from __future__ import annotations

import json
import os
import time

import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import NoCredentialsError, ProfileNotFound
from botocore.session import Session

FUNCTION_URL = os.environ.get(
    'FUNCTION_URL',
    'https://nzv7criq6nbyrhchpexdfqjcge0pjzmn.lambda-url.eu-west-1.on.aws',
).rstrip('/')

REGIAO = os.environ.get('AWS_REGION', 'eu-west-1')
PERFIL = os.environ.get('AWS_PROFILE', 'airbnb')
SERVICO = 'lambda'

# Cold start de uma imagem de 1,3 GB leva ~25 s na primeira invocação; morno
# responde em ~0,8 s. O timeout precisa acomodar o primeiro caso.
TIMEOUT_S = 120


class ErroAPI(RuntimeError):
    """Falha ao chamar a API — com mensagem destinada à tela, não ao log."""


def _credenciais():
    try:
        sessao = Session(profile=PERFIL)
        cred = sessao.get_credentials()
    except ProfileNotFound as exc:
        raise ErroAPI(
            f'Perfil AWS "{PERFIL}" não encontrado em ~/.aws/credentials.\n'
            'Configure com `aws configure --profile airbnb` ou ajuste a '
            'variável AWS_PROFILE.'
        ) from exc

    if cred is None:
        raise ErroAPI(
            'Nenhuma credencial AWS encontrada. A Function URL exige '
            'autenticação IAM — configure com `aws configure --profile airbnb`.'
        )
    return cred.get_frozen_credentials()


def chamar(rota: str, payload: dict | None = None) -> tuple[dict, float]:
    """Chama a Function URL e devolve `(resposta, latência_em_segundos)`.

    A latência é devolvida junto porque a página a exibe: é o que distingue
    um cold start de uma requisição morna, e torna visível que a resposta
    veio da nuvem (plano §5.2).
    """
    url = f'{FUNCTION_URL}/{rota.lstrip("/")}'
    corpo = json.dumps(payload) if payload is not None else None
    metodo = 'POST' if payload is not None else 'GET'

    req = AWSRequest(
        method=metodo,
        url=url,
        data=corpo,
        headers={'Content-Type': 'application/json'},
    )

    try:
        SigV4Auth(_credenciais(), SERVICO, REGIAO).add_auth(req)
    except NoCredentialsError as exc:
        raise ErroAPI('Credenciais AWS inválidas ou expiradas.') from exc

    inicio = time.perf_counter()
    try:
        resp = requests.request(
            metodo, url, data=corpo, headers=dict(req.headers), timeout=TIMEOUT_S
        )
    except requests.exceptions.Timeout as exc:
        raise ErroAPI(
            f'A API não respondeu em {TIMEOUT_S}s. Um cold start leva ~25s; '
            'além disso, algo está errado.'
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ErroAPI(f'Falha de rede: {type(exc).__name__}.') from exc
    latencia = time.perf_counter() - inicio

    if resp.status_code == 403:
        # Causa quase sempre de permissão, e a mensagem crua da AWS não ajuda
        # quem está olhando a tela.
        raise ErroAPI(
            'A AWS recusou a requisição (403). A credencial precisa de '
            '`lambda:InvokeFunctionUrl` E `lambda:InvokeFunction`.'
        )
    if resp.status_code != 200:
        raise ErroAPI(f'A API respondeu HTTP {resp.status_code}: {resp.text[:300]}')

    try:
        return resp.json(), latencia
    except json.JSONDecodeError as exc:
        raise ErroAPI(f'Resposta não é JSON: {resp.text[:200]}') from exc


def health() -> tuple[dict, float]:
    return chamar('/health')


def predict(listings: list[dict]) -> tuple[dict, float]:
    return chamar('/predict', {'listings': listings})
