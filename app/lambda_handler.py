"""Adaptador entre o AWS Lambda e a aplicação FastAPI.

O Mangum traduz eventos do Lambda (Function URL, API Gateway) para o
protocolo ASGI que o FastAPI entende, e a resposta de volta.

`lifespan='on'` é essencial: é o que faz o `lifespan` da API rodar, carregando
modelo e artefatos **uma vez por container** em vez de a cada invocação. Um
LightGBM de 900 árvores carregado por requisição inviabilizaria a latência
(plano 13 §4.6).

Plano: docs/plans/14_deploy_aws.md §6.2
"""

from mangum import Mangum

from app.api import app

handler = Mangum(app, lifespan='on')
