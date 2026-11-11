# Plano — Etapa 07: Machine Learning (Airbnb Amsterdam)

## Contexto

Entramos nesta etapa com os **dois datasets** produzidos pela Etapa 06, cada um dedicado a
uma família de algoritmo:

| Dataset | Treino | Validação | Features | Destino |
|---|---|---|---|---|
| **`df06_arvore`** | 14.382 × 29 | 3.596 × 29 | **27** (`FEATURES_06_ARVORE`) | Random Forest, XGBoost, LightGBM |
| **`df06_linear`** | 14.382 × 51 | 3.596 × 51 | **49** (`FEATURES_06_LINEAR`) | Linear Regression, Regressão Polinomial |

Mais `ARTEFATOS_06` (17 chaves) — todo parâmetro aprendido no treino, insumo obrigatório da
Etapa 09.

**Isolamento do teste** (CLAUDE.md §5.3): `test_raw.csv` (2.003 linhas) permanece
**fechado**. Esta etapa escolhe o modelo campeão usando **apenas** treino e validação. O
hold-out é aberto uma única vez, na Etapa 09.

**Orientação de métrica** (CLAUDE.md §1 e §7): comparação por **MAPE**, com MAE e RMSE de
apoio. O `fit` é em `log1p(price)`, mas **todo MAPE reportado é em escala original, após
`expm1`**. Sem exceção — MAPE sobre o log é outra métrica, não comparável com o número que
o negócio lê na Etapa 10.

### Expectativa calibrada antes de começar

A Análise 2.9 (Etapa 04) mediu um **piso de dispersão intragrupo de 20,8%** (MAD%,
controlando bairro × room_type × accommodates, cobertura 91,3%). O Random Forest da Etapa 06
já entrega **23,71%**.

> **A margem de ganho desta etapa é de ~3 p.p., não de 10.** Se XGBoost ou LightGBM
> chegarem a 21–22%, é um resultado muito bom. Se algum modelo marcar **abaixo de 19%**, a
> primeira hipótese é **vazamento**, não sucesso — e a investigação precede qualquer
> celebração.

---

## 1. Bloqueio a resolver antes de começar

Verificação executada no venv `test-bench`:

| Pacote | Status |
|---|---|
| `scikit-learn` 1.7.2 | Disponível |
| `lightgbm` | **AUSENTE** |
| `xgboost` | **AUSENTE** |
| `optuna` | **AUSENTE** (requisito da Etapa 08) |

Quatro dos seis modelos exigidos pelo CLAUDE.md §7 dependem dos dois primeiros.

**Ação do usuário:**

```bash
/Users/meigarom/.pyenv/versions/test-bench/bin/pip install lightgbm xgboost optuna
```

Depois é necessário **reiniciar o kernel** do JupyterLab para que os pacotes sejam
importáveis — e o restart apaga todo o estado acumulado (`df06_*`, `ARTEFATOS_06`, `df05`,
`df04`).

### 1.1 Opção C — comentar o Feature Selection e ler as features de uma variável

**Decisão do usuário.** As células caras da Etapa 05 são **comentadas** e substituídas por
uma única variável com a lista de features já selecionadas. Após o restart, a Etapa 05
apenas lê a variável e filtra o dataset, sem reexecutar nenhum seletor.

Custo evitado, medido nas execuções reais:

| Célula | Operação | Tempo |
|---|---|---|
| 147 | `permutation_importance` (71 cols × 5 repeats) | ~32 s |
| 149 | Null Importance v2 (50 shuffles × permutação) | ~8 min |
| 150 | RFECV (5 folds, step=5) | ~24 min |
| 151 | Boruta (`max_iter=50`) | ~2,2 min |
| 152 | Estabilidade (3 seeds de RFECV) | ~35 min |
| | **Total evitado** | **~70 min** |

**O que a variável preserva** — a lista completa, tal como a etapa a produziu:

```python
# FEATURES_05 — resultado travado da Etapa 05 (não reexecutar os seletores).
# Origem: RFECV (corte de 26 por +0,5 p.p.) + blindagem de neighbourhood_cleansed.
# Validado: MAPE 23,71% em valid_05; N_CORTE=26 estável nas seeds 42/7/2024.
FEATURES_05 = [
    'accommodates', 'bathrooms', 'bedrooms', 'room_type',
    'latitude', 'longitude', 'dist_centro_km', 'dist_museumplein_km',
    'neighbourhood_cleansed', 'n_amenities', 'host_experiencia_dias',
    'minimum_nights', 'maximum_nights', 'cleaning_fee',
    'taxa_limpeza_por_noite', 'security_deposit', 'extra_people',
    'calendar_atualizado_dias', 'review_scores_rating', 'gap_value_vs_geral',
    'number_of_reviews', 'reviews_per_month', 'dias_desde_primeira_review',
    'dias_desde_ultima_review', 'availability_30', 'availability_365',
    'len_house_rules',
]
df05 = df04[['id', TARGET] + FEATURES_05].copy()
```

**Regras da operação, para não perder rastreabilidade:**

1. O código original é **comentado, não apagado** — a auditoria de como as 27 features
   foram escolhidas continua legível no notebook.
2. As células de **markdown** da Etapa 05 (cabeçalho e consolidação) ficam **intactas**:
   elas contêm os resultados, as tabelas e as decisões de blindagem.
3. A nova célula declara explicitamente que é um **resultado travado** e aponta a origem.
4. As células 153–154 (matriz de consenso e validação final) dependem de variáveis que só
   as células lentas produzem (`ARVORES`, `NULL_IMP_B`, `RANK_RFECV`, `BORUTA_CONF`,
   `freq`). Serão **comentadas junto**, com as tabelas já registradas no markdown de
   consolidação.
5. **Teste de aceitação obrigatório:** após o restart e a reexecução, verificar
   `df05.shape == (17978, 29)` e `set(FEATURES_05)` idêntico ao registrado. Se divergir,
   algo na cadeia 02→04 mudou e a comparabilidade com os 23,71% está perdida.

### 1.2 Ordem de reexecução após o restart

Etapas 01 → 02 → 03 → 04 (completas) → **05 reduzida à variável** → 06 (completa) → 07.

A Etapa 06 **precisa** rodar inteira: ela aprende os artefatos (medianas, mapas de encoder,
centro e escala do `RobustScaler`) que a Etapa 07 consome e a Etapa 09 reaplica.

---

## 2. Baselines — o chão contra o qual tudo é medido

O CLAUDE.md §7 pede "Baseline (média)". A medição revelou que a leitura importa, porque
`price` tem **assimetria 5,92**:

| Baseline | Predição | MAPE | Papel |
|---|---|---|---|
| **Média do `log1p`, revertida com `expm1`** | €130,7 | **40,71%** | **Oficial** (decisão do usuário) |
| Média em euro | €148,8 | 48,48% | Referência — o pior dos três |
| Mediana em euro | €125,0 | 38,79% | Piso trivial |
| **Mediana de `GROUP_COLS`** | variável | **~28,22%** | **Chão de negócio** |

**Decisão do usuário: a média do `log1p` é o baseline oficial** — é o que respeita a
transformação do alvo usada em todos os modelos. Os outros três são calculados e
reportados.

Duas observações que o relatório precisa conter:

**Por que a média em euro é a pior.** Com assimetria 5,92, a média (€148,8) é puxada pelos
anúncios de €1.000+. Como o MAPE é erro **percentual**, sobrepredizer um anúncio de €60 em
€88 custa 147% de erro naquele ponto — e a cauda arrasta a média para longe da massa.

**A mediana de `GROUP_COLS` é o número que importa para o negócio.** É exatamente a regra
da Etapa 01 (CLAUDE.md §3.1): a mediana LOO dos comparáveis. Ela mede **28,22%**, e é
contra ela que o modelo precisa provar valor — não contra 40,71%. Um modelo que não bate a
mediana dos comparáveis não justifica existir, porque a Etapa 01 já entrega essa
estimativa sem nenhum treino.

---

## 3. Qual dataset para qual modelo

A Etapa 06 mediu o efeito cruzado, e o resultado orienta esta alocação:

| Modelo | Dataset | Justificativa (medida na Etapa 06) |
|---|---|---|
| Baseline | — | Sem features |
| **Linear Regression** | `df06_linear` | **+1,10 p.p.** de ganho (25,88% → 24,78%) |
| **Regressão Polinomial** | `df06_linear` | Exige escala comum |
| **Random Forest** | `df06_arvore` | 27 colunas vs 49; importâncias legíveis |
| **XGBoost** | `df06_arvore` | idem |
| **LightGBM** | `df06_arvore` | idem |

Registro honesto, herdado da Etapa 06: o RF foi **indiferente** entre os datasets (23,69%
no linear vs 23,71% na árvore — diferença de 0,02 p.p., que é ruído). A alocação das
árvores ao `df06_arvore` se justifica por **economia e legibilidade**, não por MAPE.

---

## 4. A Regressão Polinomial — o problema técnico da etapa

Grau 2 sobre as 49 colunas do `df06_linear` geraria **1.274 termos** para 14.382 linhas de
treino. Três consequências: multicolinearidade severa (cada termo é produto de duas colunas
já correlacionadas), sobreajuste garantido sem regularização, e risco de matriz
mal-condicionada.

**Decisões do usuário, ambas adotadas:**

### 4.1 Grau 2 apenas nas 25 numéricas

As 24 dummies ficam fora da expansão polinomial e entram **lineares** no modelo. Razão
conceitual: interações entre dummies de bairro são **estruturalmente nulas** — um anúncio
está em exatamente um bairro, então `bairro_A × bairro_B = 0` para toda linha. Gerar esses
termos produziria centenas de colunas constantes em zero.

Contagem: 25 numéricas → 25 lineares + 25 quadráticas + 300 interações = **350 termos**,
mais as 24 dummies = **374 colunas**. Contra 1.274 da versão ingênua.

### 4.2 `PolynomialFeatures(degree=2)` + `Ridge`, com `alpha` por CV

`Ridge` em vez de `LinearRegression` porque sem penalização L2 o modelo com 374 colunas
correlacionadas é inútil — os coeficientes explodem e a predição não generaliza.

`alpha` escolhido por `RidgeCV` sobre um grid logarítmico (`np.logspace(-3, 3, 13)`), com
CV **interna ao treino**. É a **única exceção** à regra de não-tuning desta etapa (§6), e
existe porque sem ela o modelo não é avaliável.

O `alpha` escolhido vai no relatório.

---

## 5. Protocolo de avaliação

**Decisões do usuário, ambas adotadas.**

### 5.1 5-fold CV no treino para rankear; `valid_06` para confirmar o campeão

Avaliar só em `valid_06` daria um número por modelo, sem noção de variância — e se LightGBM
der 22,8% e XGBoost 23,1%, não haveria como saber se a diferença é real ou ruído de
partição.

Protocolo adotado:

1. **`KFold(n_splits=5, shuffle=True, random_state=42)` sobre o treino**, com scorer de MAPE
   em euro (via `make_scorer` customizado, porque o
   `neg_mean_absolute_percentage_error` do sklearn operaria sobre o log).
2. Cada modelo reporta **média ± desvio** do MAPE nos 5 folds. O ranking usa a média; o
   desvio diz se a diferença entre dois modelos é maior que o ruído.
3. **`valid_06` é usada uma única vez por modelo**, como confirmação — não para escolher.

Custo: ~5× mais fits. Com RF a ~3,6 s por fit, irrelevante.

### 5.2 MAPE por faixa de preço — o teste que o CLAUDE.md não pede

A Análise 2.9 mostrou que o erro percentual é muito pior nos anúncios baratos, e é
justamente lá que está a massa acionável do EV da Etapa 01 (2.833 anúncios subprecificados,
uplift mediano de €14/diária).

Um modelo com MAPE global de 23% pode ter **40% no primeiro decil** — e seria péssimo para
o recomendador da Etapa 09, que atua exatamente na subprecificação. O MAPE global esconde
isso.

Entregável: **MAPE por decil de `price`** para os 3 melhores modelos, com a contagem de
anúncios por decil. Se o campeão global tiver erro muito pior no decil 1 que o
vice-campeão, isso precisa pesar na escolha.

---

## 6. Parâmetros — default ou modestos e fixos

**Decisão do usuário.** Nenhum tuning nesta etapa. O ajuste de hiperparâmetros é a Etapa 08
(Optuna + CV); antecipá-lo aqui tornaria a comparação injusta — o modelo que eu ajustasse
mais ganharia por atenção, não por mérito.

| Modelo | Parâmetros |
|---|---|
| Linear Regression | default |
| Polinomial | `degree=2` nas numéricas + `RidgeCV` (**única exceção**, §4.2) |
| Random Forest | `n_estimators=100`, `random_state=42` |
| XGBoost | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `random_state=42` |
| LightGBM | `n_estimators=100`, `num_leaves=31`, `learning_rate=0.1`, `random_state=42` |

`random_state=42` em tudo que for estocástico. Os parâmetros usados vão no relatório, para
que a Etapa 08 saiba de onde partiu.

---

## 7. Ordem de execução — 8 portões

1. **Instalação + restart + revalidação** — Opção C aplicada (§1.1), conferir
   `df05.shape == (17978, 29)` e `FEATURES_05` idêntico ao travado
2. **Baselines** — média do log (oficial), média, mediana, mediana de `GROUP_COLS` (§2)
3. **Linear Regression** — `df06_linear`
4. **Regressão Polinomial** — grau 2 nas 25 numéricas + `RidgeCV` (§4)
5. **Random Forest** — `df06_arvore`
6. **XGBoost** — `df06_arvore`
7. **LightGBM** — `df06_arvore`
8. **Comparação consolidada** — tabela, gráfico, MAPE por decil, escolha do campeão,
   análise de resíduos

Cada modelo é um portão, com a saída na tela antes do próximo — Regra de Trabalho nº 2 do
CLAUDE.md. São **8 portões**, não 1.

---

## 8. Entregáveis

1. **Tabela comparativa** — 6 modelos × (MAPE CV média ± desvio, MAPE validação, MAE, RMSE,
   tempo de treino, dataset usado)
2. **Gráfico de barras** do MAPE com barras de erro (desvio dos 5 folds)
3. **MAPE por decil de `price`** para os 3 melhores — o teste de §5.2
4. **`MODELO_CAMPEAO`** e **`DATASET_CAMPEAO`** registrados para a Etapa 08
5. **Análise de resíduos do campeão** — distribuição do erro, viés sistemático por faixa de
   preço, e verificação de super/subpredição
6. **Premissas registradas** — parâmetros de cada modelo, protocolo de CV, `alpha` do Ridge,
   e o que explicitamente fica para a Etapa 08

---

## 9. Verificação

- **Isolamento do teste:** nenhuma célula desta etapa lê `test_raw.csv` — busca textual
- **MAPE em escala original:** toda métrica passa por `expm1`; `assert 0 < mape < 1`
- **Faixa esperada:** MAPE do campeão entre **20% e 30%**. Abaixo de 19% dispara
  investigação de vazamento antes de qualquer conclusão
- **Dataset correto por modelo:** `assert` de que lineares usaram `df06_linear` e árvores
  `df06_arvore`
- **Alinhamento de colunas:** treino e validação com as mesmas colunas, na mesma ordem
- **Sem tuning disfarçado:** os parâmetros de §6 conferidos contra o que foi efetivamente
  passado a cada estimador
- **Reprodutibilidade pós-restart:** `FEATURES_05` e `df05.shape` idênticos ao registrado
  antes do restart (§1.1, item 5)

---

## 10. Pendências

**Resolvidas antes da execução** (validadas pelo usuário em conversa):

1. **Opção C** para o restart — comentar o Feature Selection, manter `FEATURES_05` como
   variável travada (§1.1). Evita ~70 min de reexecução.
2. **Baseline oficial:** média do `log1p`; mediana de `GROUP_COLS` calculada como chão de
   negócio (§2).
3. **Polinomial:** grau 2 apenas nas 25 numéricas + `PolynomialFeatures` com `RidgeCV`
   (§4.1, §4.2).
4. **Protocolo:** 5-fold CV no treino para rankear, `valid_06` para confirmar; mais MAPE por
   faixa de preço (§5.1, §5.2).
5. **Parâmetros:** default ou modestos e fixos; `alpha` do Ridge é a única exceção (§6).

**Em aberto:**

1. **Grau da polinomial além de 2** — não planejado. Se o grau 2 ficar muito abaixo das
   árvores, grau 3 sobre um subconjunto pequeno (top-10 features) seria o próximo teste,
   mas provavelmente não vale o custo.
2. **Simetrização de `review_scores_rating`** (assimetria −3,55, herdada da Etapa 06).
   Reavaliar se a Linear Regression e a Polinomial ficarem muito atrás das árvores.
3. **Critério de desempate para o campeão** — se dois modelos empatarem dentro do desvio da
   CV, a proposta é escolher pelo **MAPE no decil 1** (§5.2), por ser onde está o valor de
   negócio. Confirmar quando o caso aparecer.
