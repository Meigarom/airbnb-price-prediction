# Plano — Deploy na AWS (Airbnb Amsterdam)

## Contexto

O plano 13 entregou os componentes de produção, validados localmente:

| Componente | Estado |
|---|---|
| `app/data_preparation.py` | classe `DataPreparation`, 96 colunas → 27 features |
| `app/train.py` | treino + serialização + MLflow |
| `app/api.py` | FastAPI com `/health`, `/predict`, `/docs` |
| `app/models/` | `model.pkl` (5,4 MB) + `artifacts.pkl` (4 KB) |
| `tests/` | **19 testes**, incluindo equivalência com critério de €0,01 |

**Este plano leva isso para a AWS**, com o código no GitHub.

### O perfil medido — é ele que define a arquitetura

| Métrica | Valor | Implicação |
|---|---|---|
| Cold start (load dos pickles) | **1,11 s** | aceitável para serverless |
| Latência, lote de 1 | 22 ms | — |
| Latência, lote de 15 | **17 ms** | o modelo não é o gargalo |
| Latência, lote de 1.000 | **55 ms** | escala quase de graça |
| Modelo em memória | ~5 MB | cabe em qualquer runtime |
| Imagem com LightGBM | ~400 MB | **passa do limite de 250 MB do Lambda zip** |

Aplicação **leve, sem estado, com tráfego intermitente**. Isso elimina metade das
opções antes de qualquer discussão de preferência.

### Pré-requisitos — nada disso está instalado

```bash
brew install docker terraform awscli
pip install nbstripout
```

E o projeto **ainda não é um repositório Git** (§4.1).

---

## 1. Arquitetura

```
┌─ GITHUB ──────────────────────────────────────────────────────┐
│  app/ · tests/ · docs/ · Dockerfile · terraform/ · workflows  │
│  SEM dados (163 MB) e SEM pickles (vão para o S3)             │
└───────────────────────────┬───────────────────────────────────┘
                            │ push na main
                            ▼
┌─ GITHUB ACTIONS (OIDC, sem chaves estáticas) ─────────────────┐
│  PR:    19 testes — equivalência de €0,01 é o portão          │
│  main:  baixa artefatos do S3 → build arm64 → ECR → Lambda    │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─ AWS (região eu-west-1) ──────────────────────────────────────┐
│                                                                │
│   ECR ─────────► Lambda (container, arm64/Graviton2, 1 GB)    │
│   imagem+pickles  ├─ FastAPI + Mangum (adapter ASGI)          │
│                   ├─ artefatos EMBUTIDOS na imagem            │
│                   └─ Function URL (AWS_IAM)                   │
│                              │                                 │
│                              ▼                                 │
│                        CloudWatch — logs + 3 métricas de drift│
│                                                                │
│   S3 (precificacao-airbnb-ams)                                │
│   ├─ raw/listings_details.csv      dados brutos (75 MB)       │
│   ├─ models/v1/{model,artifacts}.pkl   fonte do build         │
│   └─ mlflow/                        backend do MLflow         │
│                                                                │
│   SageMaker Training Job (opcional) — retreino sob demanda    │
└────────────────────────────────────────────────────────────────┘
```

### 1.1 Decisões validadas pelo usuário

| # | Decisão | Onde |
|---|---|---|
| 1 | **Artefatos embutidos na imagem** (rastreabilidade) | §3.2 |
| 2 | **Function URL + IAM** (sem API Gateway por ora) | §3.3 |
| 3 | **Notebook versionado limpo** com `nbstripout` | §4.2 |
| 4 | **Terraform** como infraestrutura como código | §5 |

---

## 2. Por que Lambda, e não as alternativas

| Opção | Veredito |
|---|---|
| **Lambda (imagem de container)** | **Escolhida** |
| Lambda (zip) | Inviável: LightGBM + pandas + numpy passam dos 250 MB |
| ECS Fargate | Paga 24/7 por algo que responde em 17 ms |
| EC2 | Exige gestão de instância, patching, autoscaling |
| SageMaker Endpoint | ~$50/mês **ocioso** — 50× o custo da Lambda neste perfil |

O ponto fraco é o **cold start de 1,11 s**. Para uso em lote é irrelevante. Se um SLA
interativo exigir, `provisioned_concurrency = 1` resolve por ~$12/mês — mas reintroduz
custo fixo, então só com motivo.

### 2.1 arm64 (Graviton2) — decisão técnica com ganho duplo

A máquina de desenvolvimento é **arm64** (Apple Silicon), e o Lambda suporta Graviton2:

| | x86_64 | **arm64** |
|---|---|---|
| Build local (M1) | emulação QEMU, lento e frágil | **nativo** |
| Preço do Lambda | referência | **~20% menor** |
| LightGBM / numpy | wheels disponíveis | wheels disponíveis |

Sem contrapartida relevante. O `Dockerfile` fixa `--platform linux/arm64` e o Terraform
declara `architectures = ["arm64"]` — se divergirem, a Lambda falha no deploy, não em
runtime.

---

## 3. Os componentes na AWS

### 3.1 ECR — registro da imagem

Repositório `precificacao-airbnb`, com:

- **Scan on push** — detecta CVEs nas dependências
- **Lifecycle policy** — mantém as 10 imagens mais recentes; sem isso o storage cresce
  indefinidamente
- **Tag imutável** — `v1-<git-sha>`, para que uma tag sempre aponte para o mesmo
  conteúdo

### 3.2 Artefatos embutidos na imagem (decisão 1)

| | **Na imagem (escolhido)** | Do S3 no cold start |
|---|---|---|
| Cold start | 1,11 s | +~0,5 s |
| Trocar modelo | rebuild + redeploy | só atualizar o S3 |
| **Rastreabilidade** | **imagem ≡ modelo** | imagem e modelo podem divergir |

Com 5,4 MB o custo é irrelevante, e a garantia é forte: **a imagem em produção contém
exatamente o modelo que passou nos testes**. É o mesmo princípio dos dois artefatos do
plano 13 — evitar que produção e validação se separem silenciosamente.

O S3 continua sendo a **fonte** dos artefatos (o build os baixa), não o caminho de
runtime.

### 3.3 Function URL + IAM (decisão 2)

```
https://<id>.lambda-url.eu-west-1.on.aws/
auth: AWS_IAM  (requisições assinadas com SigV4)
```

Sem API Gateway nesta versão. O que isso **não** entrega, e que o plano 13 já registrou
como pendência:

| Falta | Quando passa a importar |
|---|---|
| Rate limiting por consumidor | múltiplos clientes externos |
| Domínio customizado | API pública |
| Chaves de API | cobrança ou quotas por cliente |

Mitigação parcial e imediata: **`reserved_concurrency`** na Lambda limita o paralelismo
e, com isso, o custo máximo — independentemente do volume de chamadas.

Migrar para API Gateway depois é aditivo: a Lambda não muda.

### 3.4 S3 — dados, artefatos e MLflow

Bucket `precificacao-airbnb-ams`, versionamento **ativado**:

| Prefixo | Conteúdo | Observação |
|---|---|---|
| `raw/` | `listings_details.csv` (75 MB) | GitHub não é lugar para dados |
| `models/v1/` | `model.pkl`, `artifacts.pkl` | fonte do build; versionamento dá rollback |
| `mlflow/` | backend do MLflow | hoje é SQLite local (`mlruns/mlflow.db`) |

Bloqueio de acesso público em todos os prefixos. Lifecycle movendo `raw/` para
Infrequent Access após 90 dias.

### 3.5 CloudWatch — e o que fazer sobre drift

Logs vêm de graça. O que **não** vem, e é a pendência mais séria do plano 13: **em
produção não há hold-out, então a degradação do MAPE é invisível**.

Três métricas customizadas, emitidas pela própria Lambda via EMF, são o que mais se
aproxima de detectar isso:

| Métrica | Baseline | Alarme |
|---|---|---|
| Mediana de `price_estimate` | ~€128 | desvio > 15% em 24 h |
| Proporção de `subprecificado` | 11,6% | desvio > 5 p.p. |
| Taxa de `status: invalido` | ~0,04% | > 2% (formato de entrada mudou) |

Nenhuma delas mede o MAPE — isso exigiria preços reais posteriores. Mas as três detectam
**mudança de regime**, que é o sintoma de drift.

Mais dois alarmes operacionais: taxa de erro 5xx > 1% e duração p99 > 5 s.

### 3.6 SageMaker Training Job — opcional, para o retreino

`train.py` roda em ~1 minuto localmente; não há necessidade hoje.

Mas a pendência de **gravidade alta** herdada do plano 13 é que **os dados são de
dezembro de 2018**, e Amsterdam restringiu aluguel de curta duração depois disso.
Quando a base for atualizada, um Training Job lendo de `s3://…/raw/` e gravando em
`s3://…/models/v2/` evita que o retreino dependa da máquina de alguém.

Fora do escopo desta entrega; registrado como caminho.

---

## 4. GitHub

### 4.1 O repositório ainda não existe

```bash
cd /Users/meigarom/repos/previsao-preco-aluguel-temporada
git init
git branch -M main
```

### 4.2 O que versionar

| Vai | Não vai | Por quê |
|---|---|---|
| `app/`, `tests/`, `docs/plans/` | `data/` (163 MB) | vai para o S3 |
| `Dockerfile`, `requirements.txt` | `app/models/*.pkl` | vêm do S3 no build |
| `terraform/`, `.github/workflows/` | `mlruns/`, `__pycache__`, `.pytest_cache` | efêmeros |
| `CLAUDE.md`, `README.md` | `.jupyter*`, `.DS_Store` | locais |

**O notebook (decisão 3):** versionado **limpo**, com `nbstripout` como filtro do Git.
São 207 células com saídas embutidas — o registro de todo o trabalho analítico, mas que
polui diffs e pesa. Os números permanecem preservados nos documentos de `docs/plans/`,
que são a fonte citável.

```bash
pip install nbstripout
nbstripout --install          # registra o filtro no .git/config
```

### 4.3 `.gitignore`

```
data/
app/models/*.pkl
mlruns/
.jupyter*
__pycache__/
.pytest_cache/
*.pyc
.DS_Store
.terraform/
*.tfstate
*.tfstate.*
.env
```

`*.tfstate` fora do Git é inegociável: o state pode conter valores sensíveis, e o
versionamento dele é o backend S3 (§5.2).

---

## 5. Terraform (decisão 4)

### 5.1 Estrutura

```
terraform/
├── main.tf          provider, backend, locals
├── ecr.tf           repositório + lifecycle policy
├── lambda.tf        função, role, Function URL, concorrência
├── s3.tf            bucket, versionamento, lifecycle
├── cloudwatch.tf    log group, métricas, alarmes
├── iam.tf           role da Lambda + role OIDC do GitHub
├── variables.tf     região, nome do projeto, tags
└── outputs.tf       URL da função, URI do ECR
```

### 5.2 Backend remoto

State em `s3://precificacao-airbnb-ams/terraform/state` com lock via DynamoDB. State
local num projeto com CI/CD é fonte garantida de conflito.

**Problema do ovo e da galinha:** o bucket do state precisa existir antes do Terraform
rodar. Resolvido com um bootstrap manual (`aws s3 mb`) documentado no README — uma vez,
nunca mais.

### 5.3 Os recursos

| Arquivo | Recursos |
|---|---|
| `ecr.tf` | `aws_ecr_repository`, `aws_ecr_lifecycle_policy` |
| `lambda.tf` | `aws_lambda_function` (package_type=Image, arm64, 1 GB, timeout 60 s), `aws_lambda_function_url` (AWS_IAM), `reserved_concurrent_executions` |
| `s3.tf` | `aws_s3_bucket` + versioning + public access block + lifecycle |
| `iam.tf` | role de execução da Lambda (CloudWatch + leitura do S3), role OIDC do GitHub (push ECR + update da Lambda) |
| `cloudwatch.tf` | log group (retenção 30 dias), 5 alarmes (§3.5) |

---

## 6. Dockerfile

```dockerfile
FROM --platform=linux/arm64 public.ecr.aws/lambda/python:3.11

COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY app/models/ ${LAMBDA_TASK_ROOT}/app/models/   # artefatos EMBUTIDOS (§3.2)

CMD ["app.lambda_handler.handler"]
```

### 6.1 `requirements-lambda.txt` — subconjunto do runtime

O `requirements.txt` atual tem 12 pacotes, mas **mlflow, pytest e httpx não são
necessários em runtime** — mlflow sozinho traz ~200 MB de dependências.

| Runtime | Desenvolvimento |
|---|---|
| pandas, numpy, scikit-learn | mlflow |
| lightgbm, joblib, scipy | pytest, httpx |
| fastapi, pydantic, mangum | uvicorn |

Separar reduz a imagem em ~40% e diminui a superfície de CVEs.

### 6.2 `app/lambda_handler.py` — o adaptador

```python
from mangum import Mangum
from app.api import app

handler = Mangum(app, lifespan='on')
```

`lifespan='on'` é essencial: é o que faz o carregamento dos pickles acontecer **uma vez
por container**, e não a cada invocação (plano 13 §4.6).

---

## 7. CI/CD — GitHub Actions

### 7.1 `.github/workflows/ci.yml` (em todo PR)

```
lint (ruff) → pytest (19 testes) → build de validação
```

O teste de equivalência (€0,01) é o portão: **falhou, não faz merge**. É a tradução em
CI do portão 4 do plano 13.

### 7.2 `.github/workflows/deploy.yml` (no merge para main)

```
OIDC → baixa artefatos do S3 → build arm64 → push ECR → update Lambda → smoke test
```

**OIDC em vez de chaves estáticas:** o workflow assume uma role via token de curta
duração. Nenhum segredo de longa duração no repositório.

O **smoke test** final chama `/health` na Function URL e confere `model_loaded: true` —
se falhar, o deploy é marcado como quebrado em vez de passar silenciosamente.

---

## 8. Custo estimado

Para **10 mil requisições/mês**, lotes de ~50 anúncios:

| Serviço | Custo/mês |
|---|---|
| Lambda (1 GB arm64, ~200 ms/req) | ~$0,08 |
| ECR (1 GB de imagens) | ~$0,10 |
| S3 (200 MB + requests) | ~$0,05 |
| CloudWatch (logs + 5 alarmes) | ~$0,60 |
| **Total** | **< $1,00/mês** |

Com `provisioned_concurrency = 1`: **+$12/mês**.
Um SageMaker Endpoint `ml.t3.medium` custaria **~$50/mês** ocioso.

---

## 9. Ordem de execução — 8 portões

1. **Git + `.gitignore` + `nbstripout`** — repositório local limpo
2. **Bootstrap AWS** — bucket do state, upload de dados e artefatos para o S3
3. **`Dockerfile` + `lambda_handler.py` + `requirements-lambda.txt`**, validados
   **localmente** com `docker run` antes de qualquer AWS
4. **Terraform** — `init`, `plan` revisado, `apply`
5. **Primeiro deploy manual** — build, push, update; valida a imagem antes do CI
6. **GitHub Actions** — CI no PR, deploy no merge, com OIDC
7. **Métricas e alarmes** de drift (§3.5)
8. **Validação end-to-end** — mesmo teste da seção de testes do notebook, agora
   contra a Function URL

O portão 3 vem antes do 4 deliberadamente: um erro no Dockerfile descoberto localmente
custa minutos; descoberto no deploy, custa um ciclo inteiro de CI.

---

## 10. Entregáveis

1. `.gitignore` + repositório Git inicializado + `nbstripout` configurado
2. `Dockerfile` (arm64) + `requirements-lambda.txt`
3. `app/lambda_handler.py` (adaptador Mangum)
4. `terraform/` — 8 arquivos (§5.1)
5. `.github/workflows/ci.yml` e `deploy.yml`
6. `scripts/bootstrap_aws.sh` — upload inicial para o S3
7. `README.md` — como fazer deploy, retreinar e consultar a API
8. Métricas e alarmes do CloudWatch

---

## 11. Verificação

- **Imagem reproduz o local:** `docker run` + `/predict` devolve os mesmos valores dos
  19 testes
- **Artefatos na imagem:** `docker run ... ls app/models/` mostra os dois pickles
- **Arquitetura correta:** `docker inspect` reporta `arm64`; Terraform declara o mesmo
- **Cold start medido:** primeira invocação após deploy registrada no CloudWatch
- **Function URL exige IAM:** chamada sem assinatura devolve 403
- **CI bloqueia regressão:** PR com equivalência quebrada não faz merge
- **State remoto:** `terraform state list` funciona de outra máquina
- **Sem segredos no Git:** `git log -p | grep -iE 'aws_secret|password'` → vazio
- **Smoke test pós-deploy:** `/health` com `model_loaded: true`

---

## 12. Riscos

| Risco | Gravidade | Mitigação |
|---|---|---|
| **Dados de dezembro/2018** — regulação de Amsterdam mudou | **ALTA** | Herdado do plano 13. Retreinar antes de tráfego real; a infra não resolve |
| **Viés de −28,7% no decil 1** vai para produção | **ALTA** | Piso de €75 já implementado; a API marca `abaixo_do_piso` sem `uplift` |
| **Sem medição de MAPE em produção** | **ALTA** | As 3 métricas de §3.5 detectam mudança de regime, não erro. Medir exigiria preços reais posteriores |
| Cold start de 1,11 s | MÉDIA | Aceitável em lote; provisioned concurrency se virar SLA |
| Mismatch de arquitetura arm64/x86 | MÉDIA | Fixado no Dockerfile **e** no Terraform; divergência falha no deploy |
| Custo por tráfego anômalo | MÉDIA | `reserved_concurrency` limita o teto |
| Sem rate limiting por consumidor | MÉDIA | Consequência da decisão 2; API Gateway quando houver cliente externo |
| Vazamento de credencial | MÉDIA | OIDC elimina chaves estáticas; `.gitignore` cobre `.tfstate` e `.env` |

---

## 13. Pendências

**Resolvidas antes da execução** (validadas pelo usuário):

1. Artefatos **embutidos na imagem** (§3.2) · 2. **Function URL + IAM** (§3.3) ·
3. Notebook **versionado limpo** (§4.2) · 4. **Terraform** (§5)

**Em aberto:**

1. **Região da AWS** — assumi `eu-west-1` (Irlanda) por proximidade com Amsterdam e
   latência. Trocar é mudar uma variável.
2. **Conta AWS e permissões** — o bootstrap exige credenciais com poder de criar S3,
   ECR, Lambda, IAM. Fora do que posso executar.
3. **Repositório GitHub remoto** — não existe. Precisa ser criado e configurado com a
   role OIDC.
4. **API Gateway** — adiado (§3.3). Necessário antes de exposição externa.
5. **Docker, Terraform e AWS CLI não instalados** — pré-requisito do portão 1.
