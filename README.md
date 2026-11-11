# Precificação de Diárias — Airbnb Amsterdam

Modelo de machine learning que estima o **preço justo** da diária de anúncios do Airbnb
em Amsterdam e identifica imóveis subprecificados, servido por uma API REST.

**MAPE de 22,21%** (± 0,13 p.p.), medido out-of-sample em 19.981 anúncios por
cross-validation de 5 folds.

---

## O problema

O faturamento do Airbnb é uma fração de cada reserva (`take_rate × diária × noites`).
Quando o anfitrião precifica errado, ele e a plataforma perdem dinheiro: subprecificar
deixa receita na mesa, superprecificar encalha o anúncio.

O projeto trata o preço da diária como um problema de regressão e **conecta o erro do
modelo a um retorno financeiro esperado** — a pergunta que viabiliza o investimento não
é "qual o MAPE?", mas "quanto vale cada ponto percentual de precisão?".

## Resultado

| Métrica | Valor |
|---|---|
| **MAPE (out-of-sample)** | **22,21% ± 0,13 p.p.** |
| MAE | €36,70 |
| Anúncios precificados | 19.981 (100% da base) |
| Recomendações ativas | 2.314 (11,6%) |
| **Retorno estimado** | **~€127 mil/ano** (adoção 50%, elasticidade 0,85) |

Referências de contexto:

| Comparação | MAPE |
|---|---|
| Baseline (média do alvo) | 40,71% |
| Mediana dos comparáveis (regra de negócio) | 26,80% |
| **Modelo** | **22,21%** |
| Teto entre anúncios idênticos | 22,67% |
| Piso teórico da EDA | 20,80% |

O modelo opera **abaixo do erro irredutível entre anúncios idênticos** — 72,9% da base
tem um "gêmeo" com os mesmos 7 atributos, e entre eles os preços diferem 22,67%. Não há
margem relevante restante neste conjunto de dados.

---

## Arquitetura

```
dados brutos (96 colunas)
        │
        ▼
  DataPreparation  ──  limpeza + 46 features + imputação por mecanismo
        │
        ▼
   27 features  ──  selecionadas por RFECV sobre 71 candidatas
        │
        ▼
    LightGBM  ──  900 árvores, hiperparâmetros por Optuna (120 trials)
        │
        ▼
 price_estimate + banda de ±22,21% + zona do recomendador
```

### Zonas do recomendador

| Condição | Zona | Ação |
|---|---|---|
| `price < L` e `price ≥ €75` | **subprecificado** | recomenda subir até `L` |
| `L ≤ price ≤ U` | zona morta | não age — o desvio cabe no erro |
| `price > U` | superprecificado | apenas sinaliza |
| `price < L` e `price < €75` | abaixo do piso | diagnóstico, sem recomendação |

Onde `L = price_estimate × (1 − MAPE)` e `U = price_estimate × (1 + MAPE)`.

O **piso de €75** existe porque no decil mais barato o modelo erra 38,5% e prevê ~29%
acima do real. Recomendar aumento nessa faixa seria irresponsável.

---

## Estrutura

```
app/
├── data_preparation.py   classe DataPreparation — 96 colunas → 27 features
├── train.py              treino, serialização, registro no MLflow
├── api.py                FastAPI — /health, /predict, /docs
├── schemas.py            contratos Pydantic
└── models/               model.pkl + artifacts.pkl (não versionados)
tests/
├── test_equivalencia.py      notebook ↔ produção, critério de €0,01
├── test_data_preparation.py  casos de borda e contrato da API
└── gerar_fixture.py          gera a referência de equivalência
docs/plans/               14 documentos de planejamento, um por etapa
notebooks/                análise exploratória e modelagem
```

---

## Como usar

### Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Treinar

```bash
python -m app.train
```

Lê `data/listings_details.csv`, treina na base completa e grava `app/models/model.pkl`
e `app/models/artifacts.pkl`, registrando a execução no MLflow.

### Servir

```bash
uvicorn app.api:app --port 8000
```

Documentação interativa em `http://localhost:8000/docs`.

### Consultar

```python
import pandas as pd, requests

listings = pd.read_csv('data/listings_details.csv').head(10)
payload = {'listings': listings.where(pd.notna(listings), None).to_dict('records')}

r = requests.post('http://localhost:8000/predict', json=payload)
df = pd.DataFrame(r.json()['results'])

print(df[['id', 'price', 'price_estimate', 'L', 'U', 'zona', 'uplift']])
```

A resposta devolve **as colunas originais** mais seis de predição, com correspondência
1:1: 100 anúncios enviados, 100 recebidos. Anúncios que violam regras de validade voltam
com `price_estimate: null` e um `status` explicando o motivo.

### Testar

```bash
pytest tests/ -v
```

---

## Decisões de projeto que merecem destaque

**Dois artefatos, não um.** `model.pkl` carrega o estimador; `artifacts.pkl` carrega os
parâmetros aprendidos no treino — categorias de `property_type`, medianas de imputação,
mapas de encoding, ordem exata das features. Se produção recalculasse qualquer um deles
a partir do lote recebido, as predições divergiriam **silenciosamente** do que foi
validado.

**A classe de preparação nunca aprende.** `DataPreparation` só aplica. Um teste
verifica, via AST, que não existe nenhuma chamada a `.fit()` no módulo.

**Teste de equivalência com critério de €0,01.** A fixture de referência é gerada por um
caminho independente, que transcreve o pipeline sem importar a classe — comparar a classe
consigo mesma não provaria nada. Diferença máxima medida: **€0,0049**.

**O erro não é uniforme.** O MAPE global de 22,21% esconde variação de 2,5×: 38,5% no
decil mais barato, 15,2% no centro, 28,3% no mais caro. O viés é monotônico — o modelo
puxa os baratos para cima e os caros para baixo (regressão à média). O piso de €75 é a
mitigação; o problema em si não foi resolvido.

---

## Limitações conhecidas

| Limitação | Impacto |
|---|---|
| **Dados de dezembro/2018** | Amsterdam restringiu aluguel de curta duração depois dessa safra. **Retreinar antes de uso real.** |
| **Viés de −28,7% no decil 1** | Contornado pelo piso de €75, não corrigido. 1.007 anúncios (5%) ficam sem recomendação. |
| **Sem medição de erro em produção** | Não há hold-out contínuo; a degradação do MAPE seria invisível sem monitoramento de drift. |
| **`price` obrigatório na API** | O caso "anúncio novo, ainda sem preço" não é atendido nesta versão. |
| **Taxa de adoção não medida** | É o parâmetro mais sensível do retorno estimado (amplitude de 75 p.p. na análise de sensibilidade). |

---

## Documentação

Cada etapa tem um documento de planejamento em [`docs/plans/`](docs/plans/), escrito
antes da implementação e com as decisões registradas:

| Etapa | Documento |
|---|---|
| 02 — Data Quality | [`02_data_quality.md`](docs/plans/02_data_quality.md) |
| 03 — Feature Engineering | [`03_feature_engineering.md`](docs/plans/03_feature_engineering.md) |
| 04 — Análise Exploratória | [`04_eda.md`](docs/plans/04_eda.md) |
| 05 — Feature Selection | [`05_feature_selection.md`](docs/plans/05_feature_selection.md) |
| 06 — Data Preparation | [`06_data_preparation.md`](docs/plans/06_data_preparation.md) |
| 07 — Machine Learning | [`07_machine_learning.md`](docs/plans/07_machine_learning.md) |
| 08 — Fine Tuning | [`08_fine_tuning.md`](docs/plans/08_fine_tuning.md) |
| 09 — Métricas e Recomendador | [`09_metricas_recomendador.md`](docs/plans/09_metricas_recomendador.md) |
| 10 — Resultado Final | [`10_resultado_final.md`](docs/plans/10_resultado_final.md) |
| CV — Precificação da base completa | [`12_cross_validation_base_completa.md`](docs/plans/12_cross_validation_base_completa.md) |
| Deploy — Produção | [`13_deploy_producao.md`](docs/plans/13_deploy_producao.md) |
| Deploy — AWS | [`14_deploy_aws.md`](docs/plans/14_deploy_aws.md) |

---

## Stack

Python 3.11 · pandas · scikit-learn · LightGBM · Optuna · FastAPI · MLflow · pytest

**Dados:** [InsideAirbnb](http://insideairbnb.com/) — Amsterdam, dezembro de 2018
(20.030 anúncios, 96 colunas). Preços em euro.

---

## Licença

GPL-3.0 — ver [LICENSE](LICENSE).
