# Plano — Deploy em Produção (Airbnb Amsterdam)

## Contexto

As Etapas 01 a 10 e a Seção CV entregaram um modelo validado e um recomendador:

| Artefato | Valor |
|---|---|
| Modelo | LightGBM com `PARAMS_LGBM_TUNED` (10 hiperparâmetros, Etapa 08) |
| **MAPE honesto** | **22,21% ± 0,13 p.p.** (CV de 5 folds, out-of-sample, 19.981 anúncios) |
| Features | 27 (`FEATURES_05`) |
| Banda do recomendador | ±22,21% |
| Piso de recomendação | €75 |
| EV estimado | €127 mil/ano (adoção 50%, elasticidade 0,85) |

Tudo isso vive dentro de um notebook de 207 células. **Este plano transforma esse
conhecimento em código de produção.**

### Pré-requisitos verificados

| Pacote | Status |
|---|---|
| `fastapi` 0.141.1, `uvicorn` 0.53.0, `pydantic` 2.13.5 | Disponíveis |
| `lightgbm` 4.7.0, `pandas` 2.3.3, `numpy` 2.3.3, `sklearn` 1.7.2 | Disponíveis |
| `joblib` 1.5.2 | Disponível |
| **`mlflow`** | **AUSENTE — instalar** |

```bash
/Users/meigarom/.pyenv/versions/test-bench/bin/pip install mlflow
```

---

## 1. Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│  1. TREINO (offline, train.py)                              │
│     base completa (19.981) → LightGBM                       │
│     ├→ models/model.pkl        (o estimador)                │
│     └→ models/artifacts.pkl    (parâmetros aprendidos)      │
│     + registro no MLflow (versão, métricas, ambiente)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. data_preparation.py — classe DataPreparation            │
│     96 colunas cruas → 27 features numéricas, sem nulos     │
│     NUNCA aprende: só aplica os artefatos                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. api.py — FastAPI                                        │
│     JSON de anúncios → price_estimate + L + U + zona        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  4. tests/ — equivalência notebook ↔ produção               │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 Decisões validadas pelo usuário

| # | Decisão | Onde |
|---|---|---|
| 1 | **Dois artefatos:** `model.pkl` + `artifacts.pkl` | §2 |
| 2 | **FastAPI** como framework | §4 |
| 3 | **JSON** na entrada e na saída | §4.2 |
| 4 | **Linhas inválidas voltam com `price_estimate = null`** — correspondência 1:1 | §4.3 |
| 5 | **`price` é obrigatório** na entrada | §4.4 |
| 6 | **Resposta inclui `L`, `U` e `zona`** — o recomendador, não só a predição | §4.5 |
| 7 | **Teste de equivalência** notebook ↔ produção | §5 |
| 8 | **MLflow** para versionamento | §6 |
| 9 | Estrutura `app/` + `tests/` | §7 |

---

## 2. Componente 1 — Treino e serialização (`app/train.py`)

### 2.1 Por que dois artefatos, e por que isso é crítico

O pickle do modelo **não basta**. A preparação depende de **5 grupos de parâmetros
aprendidos no treino**:

| Parâmetro | Origem | O que quebra se recalculado em produção |
|---|---|---|
| `CATEGORIAS_MANTIDAS` (8 níveis) | Etapa 03 | corte de 1% sobre o lote recebido → níveis diferentes → encoding desalinhado |
| `medianas_mcar` (4 colunas) | Etapa 06 | imputação com estatística do lote, não do treino |
| `medianas_mar` (5 colunas) | Etapa 06 | idem |
| `mapas_ordinal` (2 colunas) | Etapa 06 | códigos trocados entre treino e produção |
| `FEATURES_05` (ordem exata) | Etapa 05 | LightGBM é sensível à ordem das colunas |

> Se a API recalcular qualquer um desses a partir do lote que recebeu, as predições
> divergem **silenciosamente** do que foi validado. Não há erro — há um número errado.

**`artifacts.pkl` carrega:**

```python
{
    'features': FEATURES_05,              # 27 nomes, ORDEM EXATA
    'categorias_mantidas': {...},         # 8 níveis de property_type
    'medianas_mcar': {...},               # 4 colunas
    'medianas_mar': {...},                # 5 colunas (não usadas no caminho árvore,
                                          #  mas registradas p/ rastreabilidade)
    'mapas_ordinal': {...},               # room_type, neighbourhood_cleansed
    'cols_mnar_zero': [...],              # 3 colunas
    'cols_mar_sentinela': [...],          # 5 colunas
    'cols_mcar': [...],                   # 4 colunas
    'mape_validado': 0.2221,              # a banda do recomendador
    'mape_desvio': 0.0013,
    'piso_recomendacao': 75.0,
    'data_referencia': '2018-12-06',      # DATA_REF das features temporais
    'polos_geograficos': {...},           # Dam Square, Museumplein
    'versao': 'v1',
    'treinado_em': '<timestamp>',
    'n_treino': 19981,
    'ambiente': {'python': ..., 'lightgbm': ..., 'pandas': ..., 'numpy': ...},
}
```

O campo `ambiente` existe porque **pickle é sensível à versão de biblioteca**. Um modelo
carregado com `lightgbm` diferente pode falhar silenciosamente — a API valida isso no
startup e registra um aviso se divergir.

### 2.2 `DATA_REF` — a decisão que o plano precisa travar

As features temporais (`host_experiencia_dias`, `dias_desde_primeira_review`,
`dias_desde_ultima_review`, `anuncio_ativo`) são calculadas contra
`DATA_REF = 2018-12-06` — a data do scrape.

Em produção há duas opções, e elas não são equivalentes:

| Opção | Comportamento | Consequência |
|---|---|---|
| **DATA_REF fixa (2018-12-06)** | features idênticas ao treino | um anúncio criado em 2026 teria `host_experiencia_dias` negativo |
| **`datetime.now()`** | features refletem o presente | distribuição diverge do treino: todo host fica ~8 anos mais velho |

**Decisão do plano: `DATA_REF` fixa, vinda do `artifacts.pkl`**, com a ressalva registrada.
É o que mantém produção equivalente ao que foi validado. Quando o modelo for retreinado com
dados atuais (§9), a `DATA_REF` acompanha.

A API **sinaliza** (campo `status`) anúncios cujas datas sejam posteriores à `DATA_REF`,
porque para eles a predição é extrapolação.

---

## 3. Componente 2 — `app/data_preparation.py`

### 3.1 O contrato da classe

```python
class DataPreparation:
    def __init__(self, artifacts: dict): ...          # recebe, nunca aprende

    def clean(self, df) -> pd.DataFrame: ...          # Etapa 02
    def feature_engineering(self, df) -> pd.DataFrame # Etapa 03
    def prepare(self, df) -> pd.DataFrame             # Etapa 06
    def transform(self, df) -> pd.DataFrame           # orquestra os três
```

**A classe nunca chama `fit`.** Todo parâmetro vem de `artifacts`. É a diferença entre
`fit_transform` e `transform` — e é o que garante que produção reproduza a validação.

### 3.2 Mapa de origem: notebook → método

| Método | Células do notebook | Operações |
|---|---|---|
| `clean()` | Etapa 02 (19–45) | parse monetário (4 cols: `price`, `security_deposit`, `cleaning_fee`, `extra_people`) · flags f/t → 0/1 (7 cols) · 4 regras de validade |
| `feature_engineering()` | Etapa 03 (46–60) | temporais (`DATA_REF`) · haversine (2 polos) · amenities (parse + 11 flags + contagem) · host (verificações, profissional, local) · razões estruturais · política de reserva · reviews e gaps · texto (`len`, contagens) · `property_type_agrupado` (usa `CATEGORIAS_MANTIDAS`) |
| `prepare()` | Etapa 06 (158–163) | MNAR → 0 (3 cols) · MAR → −1 (5 cols) · MCAR → mediana do treino (4 cols) · encoding ordinal (2 cols, desconhecida → −1) · recorte e ordenação das 27 |

### 3.3 As 4 regras de validade (não removem linhas em produção)

No notebook, estas regras **removiam linhas**. Em produção, elas **marcam** (decisão 4):

| Regra | Origem |
|---|---|
| `bathrooms > 10` | Etapa 02, cél. 30 |
| `minimum_nights > 365` | Etapa 02, cél. 30 |
| `maximum_nights == 9999` | Etapa 02, cél. 43 |
| `price <= 0` | Etapa 02, cél. 44 |

Uma linha que viola qualquer uma recebe `status = 'invalido: <regra>'` e
`price_estimate = null`. A correspondência 1:1 entre entrada e saída é preservada — em
produção, isso vale mais que a pureza do pipeline.

### 3.4 Validação de entrada

A classe verifica, antes de qualquer transformação, que as colunas necessárias existem.
Falta de coluna obrigatória → erro explícito nomeando quais, **não** `KeyError` no meio do
pipeline.

---

## 4. Componente 3 — `app/api.py` (FastAPI)

### 4.1 Endpoints

| Rota | Método | Função |
|---|---|---|
| `/health` | GET | status, versão do modelo, MAPE validado |
| `/predict` | POST | recebe anúncios, devolve predição + banda + zona |
| `/docs` | GET | Swagger automático (FastAPI) |

### 4.2 Formato (decisão 3)

**Entrada:**

```json
{"listings": [{"id": 123, "price": 150.0, "accommodates": 4, ...}, ...]}
```

**Saída** — entrada original **mais** as colunas de predição:

```json
{
  "model_version": "v1",
  "mape": 0.2221,
  "n_input": 100,
  "n_predicted": 97,
  "results": [
    {"id": 123, "price": 150.0, "...": "...",
     "price_estimate": 178.4, "L": 138.8, "U": 218.0,
     "zona": "subprecificado", "uplift": 0.0, "status": "ok"},
    {"id": 456, "price": 0.0, "...": "...",
     "price_estimate": null, "L": null, "U": null,
     "zona": null, "uplift": null, "status": "invalido: price <= 0"}
  ]
}
```

O cliente converte `results` em DataFrame com `pd.DataFrame(resposta['results'])`. Uma API
REST não devolve objetos Python — devolve JSON.

### 4.3 Linhas inválidas (decisão 4)

100 anúncios entram → **100 saem**. Os inválidos têm `price_estimate = null` e `status`
descrevendo o motivo. Isso evita que o cliente precise reconciliar índices.

### 4.4 `price` obrigatório (decisão 5)

Registrado com uma observação: **nenhuma das 27 features usa `price`**. Ele serve apenas
para a regra de validade e para classificar a zona do recomendador.

Consequência: o caso de uso "anúncio novo, ainda sem preço" **não é atendido** por esta
versão. É uma limitação consciente — um endpoint futuro `/estimate` poderia aceitar `price`
nulo e devolver só `price_estimate`, sem zona. Fica registrado como evolução, fora do
escopo.

### 4.5 O recomendador na resposta (decisão 6)

```
L = price_estimate × (1 − 0,2221)
U = price_estimate × (1 + 0,2221)
```

| Condição | `zona` | `uplift` |
|---|---|---|
| `price < L` e `price ≥ €75` | `subprecificado` | `L − price` |
| `L ≤ price ≤ U` | `zona_morta` | 0 |
| `price > U` | `superprecificado` | 0 |
| `price < L` e `price < €75` | `abaixo_do_piso` | 0 |

O piso de €75 é mantido: o MAPE no decil mais barato é **38,5%** com viés de **−28,7%**, e
nada em produção corrige isso.

### 4.6 Carga do modelo

Modelo e artefatos carregados **uma vez no startup** (lifespan do FastAPI), não por
requisição. Carregar um LightGBM de 900 árvores a cada chamada inviabilizaria a latência.

---

## 5. Componente 4 — Teste de equivalência (`tests/`)

**Decisão 7 do usuário**, e o item que considero mais importante do plano.

### 5.1 O teste

1. Exportar N anúncios do notebook com seu `price_estimate` conhecido (de `PRECOS_ESTIMADOS`)
2. Passar os mesmos anúncios **crus** pela `DataPreparation` + modelo
3. Comparar as predições

**Critério:** diferença absoluta < €0,01 por anúncio. Não é "próximo" — é **idêntico**. A
reimplementação deve ser exata; qualquer divergência indica erro de transcrição.

### 5.2 Por que isso não é opcional

O portão 3 da Seção CV validou o pipeline reimplementado antes de rodar os 5 folds, e
mediu 0,09 p.p. de divergência — confirmando que estava correto. O mesmo princípio se
aplica aqui, com uma diferença: **em produção não há segunda chance de conferir**. Um erro
de transcrição numa feature produz predições plausíveis e erradas, indefinidamente.

### 5.3 Testes adicionais

| Teste | Verifica |
|---|---|
| Lote com linha inválida | 100 entram, 100 saem, 3 com `null` |
| Coluna obrigatória ausente | erro explícito nomeando a coluna |
| Categoria nunca vista (bairro novo) | tratada como −1, sem exceção |
| Zonas exaustivas | todo anúncio válido em exatamente uma zona |
| Piso respeitado | nenhum `uplift > 0` abaixo de €75 |
| `expm1` aplicado | `price_estimate` em euro, não em log |

---

## 6. Versionamento com MLflow (decisão 8)

### 6.1 O que registrar

| Categoria | Itens |
|---|---|
| **Parâmetros** | os 10 de `PARAMS_LGBM_TUNED` · `n_treino` · `n_features` · `DATA_REF` |
| **Métricas** | `mape_cv` (0,2221) · `mape_desvio` (0,0013) · MAPE por fold (5) · MAPE do hold-out (0,2193) |
| **Artefatos** | `model.pkl` · `artifacts.pkl` · lista de features · `PRECOS_ESTIMADOS` (amostra para o teste de equivalência) |
| **Tags** | versão · dataset (`InsideAirbnb Amsterdam 2018-12`) · ambiente (versões de lib) |

### 6.2 Escopo

MLflow local (`mlruns/`), sem servidor remoto — suficiente para rastrear versões e comparar
retreinos. Se houver necessidade de registry compartilhado depois, a migração é trivial.

O `model.pkl`/`artifacts.pkl` em `app/models/` continuam sendo **a fonte que a API carrega**;
o MLflow é o registro histórico, não o caminho de serving. Isso mantém a API sem dependência
de MLflow em runtime.

---

## 7. Estrutura de pastas (decisão 9)

```
app/
├── __init__.py
├── train.py                  # treina na base completa, salva pickles, loga no MLflow
├── data_preparation.py       # classe DataPreparation
├── api.py                    # FastAPI
├── schemas.py                # modelos Pydantic (entrada/saída)
└── models/
    ├── model.pkl
    └── artifacts.pkl
tests/
├── test_equivalencia.py      # notebook ↔ produção (§5.1)
├── test_data_preparation.py  # casos de borda (§5.3)
└── fixtures/
    └── amostra_notebook.csv  # N anúncios + price_estimate conhecido
mlruns/                       # registro do MLflow
requirements.txt
```

`schemas.py` separa os modelos Pydantic da lógica da API — mantém `api.py` legível e os
contratos de dados explícitos.

---

## 8. Ordem de execução — 7 portões

1. **Instalar MLflow** + criar a estrutura de pastas
2. **`train.py`** — treinar na base completa, gerar `model.pkl` e `artifacts.pkl`, logar no
   MLflow
3. **`data_preparation.py`** — a classe, com os três métodos e a validação de entrada
4. **Teste de equivalência** — o portão crítico. Se falhar, **parar** e auditar antes de
   seguir
5. **`schemas.py` + `api.py`** — contratos Pydantic e endpoints
6. **Testes de borda** — lote inválido, coluna ausente, categoria nova, zonas
7. **Validação end-to-end** — subir a API, enviar um lote real, conferir a resposta

O portão 4 vem **antes** da API deliberadamente: não faz sentido construir a camada de
serviço sobre uma preparação que não foi provada equivalente.

---

## 9. Riscos e limitações registrados

| Risco | Gravidade | Mitigação |
|---|---|---|
| **Dados de dezembro/2018** — Amsterdam restringiu aluguel de curta duração depois | **ALTA** | Retreinar com dados atuais antes de tráfego real. O pipeline é o mesmo; só os artefatos mudam |
| **Viés do decil 1** (−28,7%, MAPE 38,5%) vai para produção | **ALTA** | Piso de €75 na zona; a API devolve `abaixo_do_piso` sem `uplift`. Mas `price_estimate` é devolvido — alguém pode usá-lo |
| **`DATA_REF` fixa** — anúncios recentes extrapolam | MÉDIA | `status` sinaliza datas posteriores à referência (§2.2) |
| **Pickle sensível a versão de biblioteca** | MÉDIA | `artifacts.pkl` carrega `ambiente`; a API valida no startup e avisa se divergir |
| **Divergência silenciosa de transcrição** | **ALTA** | Teste de equivalência com critério de €0,01 (§5) |
| **Sem monitoramento de drift** | MÉDIA | Fora do escopo. Registrado: em produção não há hold-out, e o MAPE pode degradar sem sinal |

---

## 10. Entregáveis

1. **`app/train.py`** — script de treino e serialização
2. **`app/data_preparation.py`** — classe `DataPreparation`
3. **`app/schemas.py`** — contratos Pydantic
4. **`app/api.py`** — FastAPI com `/health`, `/predict`, `/docs`
5. **`app/models/model.pkl`** e **`artifacts.pkl`**
6. **`tests/test_equivalencia.py`** — notebook ↔ produção, critério €0,01
7. **`tests/test_data_preparation.py`** — casos de borda
8. **`requirements.txt`** — dependências pinadas
9. **Registro MLflow** — parâmetros, métricas, artefatos, tags

---

## 11. Verificação

- **Equivalência:** diferença < €0,01 entre notebook e produção em N anúncios conhecidos
- **Correspondência 1:1:** `len(saída) == len(entrada)` sempre
- **Nenhum `fit` em produção:** busca textual por `.fit(` em `data_preparation.py` → zero
  ocorrências
- **Ordem das features:** `list(X.columns) == artifacts['features']` antes de cada `predict`
- **Escala correta:** `price_estimate` em euro (`expm1` aplicado); valores na ordem de
  dezenas/centenas, não unidades
- **Zonas exaustivas e exclusivas:** cada anúncio válido em exatamente uma
- **Piso respeitado:** `uplift == 0` para todo `price < €75`
- **Startup carrega uma vez:** modelo não recarregado por requisição
- **Ambiente conferido:** versões de lib comparadas com `artifacts['ambiente']`

---

## 12. Pendências

**Resolvidas antes da execução** (validadas pelo usuário):

1. Dois artefatos (`model.pkl` + `artifacts.pkl`) · 2. FastAPI · 3. JSON ·
4. Linhas inválidas com `null` · 5. `price` obrigatório · 6. Resposta com `L`, `U`, `zona` ·
7. Teste de equivalência · 8. MLflow · 9. Estrutura `app/` + `tests/`

**Em aberto:**

1. **Endpoint `/estimate` sem `price`** — atenderia o caso "anúncio novo, ainda sem preço",
   que é o de maior valor segundo a EDA (anúncios sem review cobram €139 vs €125 e têm
   CV 0,85). Fora do escopo desta versão (§4.4).
2. **Autenticação e rate limiting** — não planejados. Necessários antes de exposição
   externa.
3. **Containerização (Docker)** — não planejada. O deploy assume execução local via
   `uvicorn`.
4. **Retreino automatizado** — não planejado. `train.py` é manual.
5. **Monitoramento de drift** — em produção não há hold-out; a degradação do MAPE seria
   invisível. Um grupo de controle sem recomendação resolveria, mas está fora do escopo.
