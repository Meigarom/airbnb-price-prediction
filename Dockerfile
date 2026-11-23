# Imagem da Lambda — precificação de diárias, Airbnb Amsterdam
#
# arm64 (Graviton2): a máquina de build é Apple Silicon, então o build é
# nativo em vez de emulado, e o Lambda cobra ~20% menos (plano 14 §2.1).
# A arquitetura é fixada AQUI e no Terraform — se divergirem, o deploy falha
# em vez de quebrar em runtime.
#
# Build:
#   docker build --platform linux/arm64 -t precificacao-airbnb:v1 .
#
# Teste local (emula o runtime do Lambda na porta 9000):
#   docker run --rm -p 9000:8080 precificacao-airbnb:v1
#
# Plano: docs/plans/14_deploy_aws.md §6

# Python 3.12 e não 3.11: a imagem do 3.11 roda Amazon Linux 2 com glibc
# 2.26, e numpy/scipy modernos só publicam wheels manylinux_2_28 (glibc
# >= 2.28). Sem isso o pip tentaria compilar do fonte e o build falharia.
# A 3.12 traz Amazon Linux 2023 com glibc 2.34.
FROM --platform=linux/arm64 public.ecr.aws/lambda/python:3.12

# LightGBM precisa de OpenMP em runtime; a imagem base não o traz.
RUN dnf install -y libgomp && dnf clean all

# A imagem base traz pip 24.0, antigo demais para reconhecer os wheels
# manylinux_2_28 de numpy/scipy — sem esta atualização o pip tenta compilar
# do fonte e o build falha por falta de toolchain.
RUN pip install --no-cache-dir --upgrade pip

# Dependências primeiro: esta camada só é reconstruída quando o
# requirements muda, não a cada alteração de código.
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# Código da aplicação
COPY app/__init__.py          ${LAMBDA_TASK_ROOT}/app/
COPY app/data_preparation.py  ${LAMBDA_TASK_ROOT}/app/
COPY app/schemas.py           ${LAMBDA_TASK_ROOT}/app/
COPY app/metricas.py          ${LAMBDA_TASK_ROOT}/app/
COPY app/api.py               ${LAMBDA_TASK_ROOT}/app/
COPY app/lambda_handler.py    ${LAMBDA_TASK_ROOT}/app/

# Artefatos EMBUTIDOS na imagem (decisão 1 do plano, §3.2).
# Garante que a imagem em produção contenha exatamente o modelo que passou
# nos testes — baixá-los do S3 em runtime permitiria divergência silenciosa.
COPY app/models/model.pkl      ${LAMBDA_TASK_ROOT}/app/models/
COPY app/models/artifacts.pkl  ${LAMBDA_TASK_ROOT}/app/models/

CMD ["app.lambda_handler.handler"]
