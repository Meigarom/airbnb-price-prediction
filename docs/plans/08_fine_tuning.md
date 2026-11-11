# Plano — Etapa 08: Fine Tuning (Airbnb Amsterdam)

## Contexto

Entramos nesta etapa com o resultado da Etapa 07: seis modelos comparados por MAPE, e um
**empate técnico** no topo.

| Modelo | MAPE CV (5 folds) | MAPE valid_06 | Dataset | Fit |
|---|---|---|---|---|
| **LightGBM** | **23,00% ± 0,36** | **22,63%** | `df06_arvore` | 4,7 s |
| **XGBoost** | 23,11% ± 0,35 | 22,67% | `df06_arvore` | **1,7 s** |
| Regressão Polinomial | 23,64% ± 0,30 | 23,34% | `df06_linear` | 8,0 s |
| Random Forest | 24,10% ± 0,38 | 23,71% | `df06_arvore` | 13,9 s |
| Linear Regression | 25,00% ± 0,86 | 24,78% | `df06_linear` | 0,1 s |

Referências: baseline oficial 40,71% · **chão de negócio 28,22%** · piso da EDA 20,80%.

**Isolamento do teste** (CLAUDE.md §5.3): `test_raw.csv` (2.003 linhas) permanece
**fechado**. A Seção 5.3 inclui a Etapa 08 explicitamente — o hold-out é aberto uma única
vez, na Etapa 09.

**Orientação de métrica** (CLAUDE.md §1): o `fit` é em `log1p(price)`, mas **todo MAPE
reportado é em escala original, após `expm1`**. Sem exceção.

### Pré-requisito: estado do kernel

O kernel foi reiniciado ao fim da Etapa 07 (para interromper um `groupby` travado durante
um diagnóstico), e o estado em memória foi perdido. O notebook mantém todas as células.

**Antes do portão 1:** executar o notebook até a célula 172 (Run All). Com a Opção C
aplicada na Etapa 05 (`docs/plans/07_machine_learning.md` §1.1), os ~75 min dos seletores
não são reexecutados.

---

## 1. A expectativa precisa estar escrita ANTES de rodar

Esta seção é a mais importante do plano. Sem ela, qualquer resultado desta etapa será lido
errado.

Quando se investigou a viabilidade de um MAPE de 10–15%, três medições delimitaram o teto
do dataset:

| Referência | MAPE | O que é |
|---|---|---|
| LightGBM atual (valid_06) | **22,63%** | o ponto de partida |
| Piso da EDA (Análise 2.9) | 20,80% | MAD% intragrupo, cobertura 91,3% |
| Oráculo que **vaza o alvo** | 24,23% | mediana do próprio conjunto de avaliação |
| **MAPE irredutível entre gêmeos exatos** | **22,67%** | o limite físico |

A última linha é a decisiva. **72,9% dos anúncios de Amsterdam têm um "gêmeo"** — outro
anúncio com valores idênticos em bairro, tipo, capacidade, quartos, banheiros, camas, tipo
de propriedade e mínimo de noites. Entre esses gêmeos, os preços diferem **22,67%** em
média.

> Dois apartamentos indistinguíveis nas features disponíveis cobram preços diferentes.
> Nenhum ajuste de hiperparâmetro pode prever essa diferença, porque **a informação que a
> causa não está no dataset** (qualidade interna, vista, andar, estratégia do anfitrião,
> momento do calendário).

**O modelo atual (22,63%) já opera nesse limite — e supera o oráculo que trapaceia.**

### 1.1 Margem realista e critério de alarme

| | |
|---|---|
| **Ganho realista desta etapa** | **0,5 a 1,5 p.p.** |
| Resultado excelente | ~21,5% |
| Resultado neutro (aceitável) | 22,5–23,0% |
| **Gatilho de investigação** | **MAPE < 19%** |

Um MAPE abaixo de 19% **não é sucesso** — é sinal de vazamento, e a investigação precede
qualquer conclusão. Registrar isso agora evita que o resultado seja lido como fracasso
quando for apenas o teto do dado, ou como triunfo quando for um bug.

---

## 2. Escopo — os dois modelos do empate técnico

**Decisão do usuário: tunar LightGBM e XGBoost.**

O CLAUDE.md §7 autoriza com o plural em *"melhor(es) modelo(s)"*, e a decisão se justifica
porque a diferença entre os dois (0,11 p.p.) está **dentro do ruído da CV** (±0,36). O
tuning pode reordená-los, e descobrir isso é informação, não desperdício.

Ambos operam sobre **`df06_arvore`** (27 features, encoding ordinal, sem escala).

A Regressão Polinomial (3º lugar, 23,64%) **não é tunada**: seu único hiperparâmetro
relevante (`alpha`) já foi escolhido por `RidgeCV` na Etapa 07, e o grau 2 é uma decisão
estrutural, não de ajuste.

---

## 3. Protocolo anti-vazamento — o risco central do Optuna

O perigo desta etapa é sutil e não aparece em nenhuma métrica interna do estudo: **muitos
trials otimizando contra a mesma partição acabam ajustando o modelo àquela partição.** É
overfitting ao conjunto de avaliação.

Três camadas de proteção:

**3.1 O Optuna otimiza o MAPE da CV de 5 folds no TREINO — nunca `valid_06`.**
Mesmo `KFold(n_splits=5, shuffle=True, random_state=42)` da Etapa 07, para que o número
seja diretamente comparável com os 23,00%.

**3.2 `valid_06` é consultada UMA ÚNICA VEZ por modelo**, depois que o estudo termina, só
para confirmar que o ganho da CV se transferiu. Se a CV melhorar e a validação piorar, isso
é overfitting ao fold — e **precisa ser reportado, não escondido**.

**3.3 `test_raw.csv` permanece fechado** (CLAUDE.md §5.3).

### 3.4 Trava de honestidade: número de trials fixado antes

**Decisão do usuário: 120 trials por modelo.** O número é fixado **antes** de ver qualquer
resultado e **não será aumentado depois**.

Rodar mais trials porque "não melhorou" é o equivalente estatístico a repetir um teste até
dar significância — cada rodada extra aumenta a chance de encontrar, por acaso, uma
combinação que vai bem naquela CV específica e não generaliza.

Configuração: `TPESampler(seed=42)` para reprodutibilidade e `MedianPruner` para abortar
trials ruins cedo.

---

## 4. Espaços de busca

### 4.1 LightGBM

| Parâmetro | Faixa | Papel |
|---|---|---|
| `objective` | `{'regression', 'mae'}` | **ver §5** |
| `n_estimators` | 200–2000 | com `lr` baixo, precisa de mais árvores |
| `learning_rate` | 0,01–0,3 (log) | trade-off central com `n_estimators` |
| `num_leaves` | 15–255 | capacidade — principal risco de overfit no LGBM |
| `min_child_samples` | 5–100 | freia folhas minúsculas |
| `subsample` | 0,6–1,0 | bagging de linhas |
| `subsample_freq` | 1–7 | frequência do bagging |
| `colsample_bytree` | 0,5–1,0 | amostragem de features |
| `reg_alpha` | 1e-8–10 (log) | L1 |
| `reg_lambda` | 1e-8–10 (log) | L2 |

### 4.2 XGBoost

| Parâmetro | Faixa | Papel |
|---|---|---|
| `objective` | `{'reg:squarederror', 'reg:absoluteerror'}` | **ver §5** |
| `n_estimators` | 200–2000 | idem |
| `learning_rate` | 0,01–0,3 (log) | idem |
| `max_depth` | 3–12 | capacidade (XGB cresce por profundidade, não por folhas) |
| `min_child_weight` | 1–20 | peso mínimo por folha |
| `subsample` | 0,6–1,0 | bagging |
| `colsample_bytree` | 0,5–1,0 | amostragem de features |
| `gamma` | 1e-8–10 (log) | ganho mínimo para abrir um split |
| `reg_alpha` / `reg_lambda` | 1e-8–10 (log) | L1 / L2 |

---

## 5. `objective='mae'` no espaço de busca — a mudança de maior potencial

**Decisão do usuário: incluir.**

O `objective` default dos dois modelos é **L2** (erro quadrático), que penaliza desvios
grandes de forma desproporcional. Mas a métrica do projeto é **MAPE**, um erro
**percentual**.

O desalinhamento é concreto. Com L2 no alvo em log, um anúncio de €1.000 com erro de €200
contribui com um resíduo grande no espaço logarítmico, puxando o ajuste; um anúncio de €60
com erro de €20 — **33% de MAPE**, muito pior para a métrica — contribui pouco. O modelo é
otimizado para a coisa errada.

Treinar com **L1** (`mae` / `reg:absoluteerror`) no alvo em log aproxima a função de perda
da métrica real, porque o erro absoluto em log é aproximadamente o erro percentual em
escala original.

Isso pode valer mais que todos os hiperparâmetros juntos — e tem relação direta com o viés
documentado em §6: L2 é exatamente o que produz regressão à média.

Entra como parâmetro **categórico**, para que o Optuna decida com base em evidência em vez
de assumirmos a resposta.

---

## 6. Reavaliação do viés por faixa (diagnóstico obrigatório)

A Etapa 07 revelou um **viés monotônico** no LightGBM, invisível no MAPE global:

| Decil | Faixa € | MAPE | Erro % mediano |
|---|---|---|---|
| D1 | 19–75 | **40,87%** | **−29,81%** |
| D5 | 116–125 | 15,01% | −0,05% |
| D10 | 252–1.050 | 28,53% | **+26,70%** |

(erro negativo = o modelo prevê **acima** do real)

Globalmente o modelo parece não-viesado (erro mediano −0,31%; 49,6% vs 50,4% de
sub/superpredição). **O viés só aparece ao desagregar.**

Isso importa para o negócio: o recomendador da Etapa 09 só atua na **subprecificação**
(CLAUDE.md §3.3). Como o modelo prevê sistematicamente **acima** do real nos anúncios
baratos, ele marcará como "subprecificado" anúncios do D1 que apenas são baratos —
inflando a contagem e superestimando o EV. Registrado como **E09-6**.

**Esta etapa mede se o tuning melhora, piora ou é neutro quanto a esse viés**, e inclui o
MAPE do decil 1 na tabela final. A hipótese a testar: se o Optuna escolher `objective='mae'`,
o viés de regressão à média deve diminuir, porque L1 não penaliza desvios grandes de forma
quadrática.

> **Nota sobre o critério de escolha:** o campeão é escolhido pelo **MAPE da CV puro**
> (decisão do usuário, §7). O viés por faixa é **diagnóstico**, não critério de seleção —
> mas se o campeão tunado piorar o D1, isso vai registrado como alerta para a Etapa 09.

---

## 7. Critério de escolha do campeão

**Decisão do usuário: MAPE da CV puro.**

O campeão é o modelo com menor MAPE médio na CV de 5 folds do treino. Sem ponderação por
decil, sem critério composto.

Dois registros que acompanham essa decisão:

1. **Se a diferença entre os dois tunados ficar dentro do ruído** (como aconteceu na
   Etapa 07, 0,11 p.p. contra ±0,36), o desvio da CV é reportado e o empate é declarado
   como tal. Na dúvida entre dois modelos estatisticamente indistinguíveis, o XGBoost tem
   a vantagem prática de ser **3× mais rápido no fit**.
2. **Se o tuning não melhorar a CV**, o resultado honesto é **manter os parâmetros da
   Etapa 07** e registrar que o modelo já estava no teto. Isso é um resultado legítimo —
   ver §1.1.

---

## 8. Ordem de execução — 7 portões

1. **Revalidação do estado pós-restart** — `df05.shape == (17978, 29)`, `FEATURES_05`
   íntegro, `df06_*` e `ARTEFATOS_06` presentes, bibliotecas importáveis
2. **Setup do estudo** — scorer de MAPE em euro, `CV_08` idêntico ao da Etapa 07,
   `TPESampler(seed=42)`, `MedianPruner`
3. **Optuna no LightGBM** — 120 trials (célula isolada)
4. **Optuna no XGBoost** — 120 trials (célula isolada)
5. **Comparação** — os dois tunados vs os dois da Etapa 07, em CV; confirmação em
   `valid_06` (uma vez por modelo)
6. **Diagnóstico de viés** — MAPE por decil e erro % mediano por faixa do campeão tunado,
   comparados com a Etapa 07
7. **Registro e consolidação** — `MELHORES_PARAMS`, `MODELO_FINAL`, `PREMISSAS_08`

Cada item é uma pausa para validação, conforme a Regra de Trabalho nº 2 do CLAUDE.md. São
**7 portões**, não 1.

---

## 9. Entregáveis

1. **`MELHORES_PARAMS`** — dicionário por modelo com os hiperparâmetros vencedores; insumo
   direto da Etapa 09
2. **`MODELO_FINAL`** e `DATASET_FINAL` — o campeão tunado, pronto para o retreino da
   Etapa 09
3. **Tabela de ganho** — Etapa 07 vs Etapa 08, em CV e em validação, para os dois modelos,
   com o delta em p.p.
4. **Histórico de otimização** — curva de convergência e **importância dos
   hiperparâmetros** (`optuna.importance.get_param_importances`), que diz quais parâmetros
   realmente moveram a métrica. Informação reutilizável: se `num_leaves` dominar, sabemos
   onde focar num futuro ajuste.
5. **Diagnóstico de viés** — MAPE por decil e erro % mediano por faixa, comparados com a
   Etapa 07 (§6)
6. **`PREMISSAS_08`** — espaços de busca, 120 trials, sampler, pruner, protocolo de CV, e
   o `objective` escolhido por cada modelo

---

## 10. Verificação

- **Isolamento do teste:** nenhuma célula desta etapa lê `test_raw.csv` — busca textual
- **Optuna não viu `valid_06`:** o objetivo do estudo usa exclusivamente `CV_08` sobre
  `X_arv_tr`; conferido por inspeção da função-objetivo
- **MAPE em escala original:** toda métrica passa por `expm1`; `assert 0 < mape < 1`
- **Gatilho de vazamento:** `assert mape_campeao > 0.19` — abaixo disso, parar e investigar
  (§1.1)
- **Trials fixos:** 120 por modelo, conferido em `len(study.trials)`; não aumentado após
  ver o resultado (§3.4)
- **Reprodutibilidade:** `TPESampler(seed=42)`; reexecutar o estudo deve dar o mesmo
  resultado
- **Dataset correto:** ambos os modelos em `df06_arvore`, com as 27 colunas de
  `FEATURES_06_ARVORE`
- **Ganho transferido:** se o MAPE da CV melhorar e o de `valid_06` piorar, registrar como
  overfitting ao fold (§3.2)

---

## 11. Pendências

**Resolvidas antes da execução** (validadas pelo usuário em conversa):

1. **Tunar os dois modelos** — LightGBM e XGBoost (§2).
2. **`objective='mae'` incluído** no espaço de busca de ambos (§5).
3. **120 trials por modelo**, fixados antes e não aumentáveis (§3.4).
4. **Critério de escolha: MAPE da CV puro** (§7). O viés por faixa é diagnóstico, não
   critério.

**Em aberto:**

1. **Se o tuning piorar o viés do decil 1** — decisão de como tratar fica para a Etapa 09,
   junto com E09-6. As duas rotas já mapeadas: calibrar a predição por faixa, ou impor um
   piso de preço na recomendação.
2. **Retreino com treino+validação** — o CLAUDE.md §7 (Etapa 09) pede que o modelo final
   seja retreinado com treino e validação juntos, usando os melhores params desta etapa.
   Isso acontece **na Etapa 09**, não aqui — esta etapa entrega apenas os parâmetros.
3. **Ensemble dos dois modelos** — não planejado. Se os dois tunados ficarem empatados e
   com erros descorrelacionados, uma média simples poderia ganhar ~0,2–0,3 p.p. Fica
   registrado como possibilidade, fora do escopo do CLAUDE.md §7.
