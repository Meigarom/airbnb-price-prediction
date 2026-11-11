# Plano — Etapa 05: Feature Selection (Airbnb Amsterdam)

## Contexto

Entramos nesta etapa com `df04`: **17.978 linhas × 73 colunas**, granularidade de 1 linha
por `listing_id`, resultado da Etapa 04 (EDA). Dessas 73 colunas, **1 é chave** (`id`),
**1 é o alvo** (`price`) e **71 são features candidatas** — a lista `FEATURES_04`.

A Etapa 04 já removeu 12 colunas por motivos **técnicos** (vazamento, redundância com
ρ ≥ 0,88, identidade linear exata). O que resta aqui é uma pergunta diferente, e é
importante não confundir as duas:

> A Etapa 04 perguntou *"esta coluna é tecnicamente defeituosa?"*.
> A Etapa 05 pergunta *"esta coluna **contribui** para prever `price`?"*.

A primeira pergunta é independente do alvo e do modelo. A segunda não: depende de qual
modelo, medida com qual métrica, sobre quais dados. Por isso toda a etapa é construída em
torno de uma comparação contra baselines explícitos, e não de um ranking isolado.

**Isolamento do teste** (regra travada em CLAUDE.md, Seção 5.3): os 10% separados na
Etapa 02 (`data/processed/test_raw.csv`, 2.003 linhas) **não entram nesta etapa**. A
seleção de features é exatamente o tipo de decisão que a Seção 5.2 do CLAUDE.md descreve
como contaminável — "quais colunas dropar" está na lista literal. Toda medição aqui usa
uma partição interna de `df04`.

**Orientação de métrica** (CLAUDE.md, Seção 1): o alvo do `fit` é `log1p(price)`, mas
**todo MAPE reportado nesta etapa é calculado em escala original, após `expm1`**. Sem
exceção. Um MAPE calculado sobre o log é uma métrica diferente, não comparável com o
número que o negócio vai ler na Etapa 10.

---

## 1. O conflito de ordem entre as Etapas 05 e 06 — e a solução adotada

O CLAUDE.md (Seção 7) coloca Feature Selection (05) **antes** de Data Preparation (06).
Mas os seletores desta etapa exigem uma matriz numérica, completa e sem nulos. E é
justamente a Etapa 06 que faz encoding e imputação.

O estado real de `FEATURES_04` confirma o problema:

| Situação | Contagem |
|---|---|
| Colunas categóricas (object/category) | **5** |
| Colunas numéricas | 66 |
| Colunas com ao menos um nulo | **28** |

As 5 categóricas: `neighbourhood_cleansed` (22 níveis), `property_type_agrupado` (9),
`host_response_time` (4), `cancellation_policy` (4), `room_type` (3).

**Decisão validada pelo usuário: manter a ordem 05 → 06**, usando uma **preparação mínima
descartável** interna à Etapa 05.

### 1.1 O que "descartável" significa aqui

A preparação mínima existe **apenas para viabilizar os seletores** e é destruída ao fim da
etapa. Ela não é herdada pela Etapa 06, não define encoder final, não define estratégia de
imputação final, e não produz nenhum objeto persistido.

O entregável da Etapa 05 é a **lista de nomes de colunas** (`FEATURES_05`) e o dataset
`df05` recortado de `df04` — **com os valores originais, nulos inclusive**. A Etapa 06
recebe dados crus, exatamente como receberia se a preparação mínima nunca tivesse existido.

Isso é registrado explicitamente porque é o principal risco de erro desta etapa: deixar a
imputação improvisada vazar para o pipeline definitivo. A Seção 15 traz a verificação
programática que prova a não-contaminação.

### 1.2 Regras da preparação mínima

Cada escolha abaixo é derivada de um achado da Análise 3.4 (padrão de faltantes), não de
conveniência:

| Item | Tratamento na 05 | Motivo |
|---|---|---|
| **Categóricas** | Ordinal encoding (códigos inteiros) | Árvores lidam bem com ordinal. One-hot de `neighbourhood_cleansed` criaria 22 colunas, fragmentando a importância do bairro em 22 pedaços fracos — o que o mataria injustamente nos seletores. O bairro é `GROUP_COLS`; distorcê-lo é inaceitável. |
| **`security_deposit`, `cleaning_fee`, `taxa_limpeza_por_noite`** (30,9% / 18,0% / 18,0%) | Preenche **0** | **MNAR confirmado na 3.4**: 100% dos nulos têm a flag correspondente (`tem_deposito` / `tem_taxa_limpeza`) = 0. O nulo significa "não cobro", não "desconhecido". Imputar a mediana transformaria "não cobro" em "cobro o valor mediano" — erro grave, já registrado como E06-3. |
| **`host_response_time`** (47,3%) | Categoria explícita `"sem_resposta"` | **MNAR confirmado na 3.4**: quem não responde tem perfil distinto (superhost 8,8% vs 22,3%; 0,34 vs 0,91 review/mês; p = 4,94e-45). A ausência é informação, não lacuna. Registrado como E06-4. |
| **Bloco de reviews** (~13%: 7 `review_scores_*`, 2 `gap_*`, 3 de datas/frequência) | Sentinela **−1** | **MAR confirmado na 3.4**: 98,8–100% de coincidência com `review_scores_rating`, e a coluna `sem_reviews` já marca o padrão de forma limpa. A sentinela mantém os ausentes separáveis por um único split da árvore, sem inventar um score que o anúncio nunca recebeu. Registrado como E06-2. |
| **Estruturais** (`camas_por_quarto` 5,7%, `pessoas_por_quarto` 5,6%, e 9 colunas ≤ 0,05%) | Mediana do **treino-05** | **MCAR**, volume irrelevante. A mediana é calculada só no treino-05 e aplicada à validação-05 — mesmo com preparação descartável, não se aprende parâmetro fora do treino. |

---

## 2. Estrutura de validação

Nenhuma decisão de seleção pode ser medida no mesmo dado em que foi tomada. `df04` é
dividido em duas partes:

```python
RANDOM_STATE = 42
train_05, valid_05 = train_test_split(
    df04, test_size=0.20, random_state=RANDOM_STATE, shuffle=True,
)
```

| Partição | Linhas (aprox.) | Papel |
|---|---|---|
| `train_05` | 14.382 | Ajuste dos modelos, cálculo de importâncias, execução dos seletores |
| `valid_05` | 3.596 | **Única fonte de MAPE** para comparar conjuntos de features |
| `test_raw.csv` | 2.003 | **Fechado.** Não é lido nesta etapa. |

Regras que atravessam todos os blocos:

- Toda mediana, todo encoder, todo `fit` usa **só `train_05`**.
- Todo número de performance citado numa decisão vem de **`valid_05`**.
- Toda operação estocástica recebe `random_state=42`, exceto o teste de estabilidade da
  Seção 12, que varia a seed de propósito.

**Nota crítica sobre o RFECV:** ele faz cross-validation **interna a `train_05`**, sem
nunca tocar `valid_05`. Isso preserva `valid_05` limpa para a comparação final do Bloco 8.
Se o RFECV escolhesse o número de features olhando `valid_05`, a comparação final estaria
contaminada — o conjunto teria sido otimizado no mesmo dado em que é julgado.

---

## 3. Restrição de ferramental (levantamento feito no ambiente real)

Uma verificação no kernel revelou duas restrições que alteram o desenho original:

| Pacote | Status | Consequência |
|---|---|---|
| `boruta` | **Disponível** | Testado e validado (ver 3.1) |
| `scikit-learn` 1.7.2 | Disponível | Base de tudo |
| `joblib` | Disponível | Paraleliza os blocos internos (Seção 4) |
| `lightgbm` | **AUSENTE** | O plano original previa LightGBM como modelo-sonda |
| `shap` | Ausente | Não era requisito; não será usado |

### 3.1 Validação funcional do Boruta

`BorutaPy` é um pacote sem manutenção ativa e com incompatibilidades conhecidas frente ao
numpy 2.x (o ambiente roda numpy 2.3.3). Foi testado antes de constar deste plano.

**Resultado:** funciona, mas com uma armadilha registrada. Num teste sintético onde uma
única feature de seis explicava o alvo (importância RF = 0,999), o Boruta com
`max_iter=5` rejeitou **todas as seis**. Com `max_iter=60`, acertou exatamente a feature
correta. O Boruta precisa de iterações para acumular significância estatística — um
`max_iter` baixo não produz um resultado "aproximado", produz um resultado **errado no
sentido conservador**. Fica travado: **`max_iter` nunca abaixo de 50**.

### 3.2 Modelo-sonda: RandomForest, não LightGBM

Com `lightgbm` ausente, o modelo-sonda desta etapa é
`RandomForestRegressor(n_estimators=100, random_state=42)`.

A troca é aceitável e não compromete a etapa, por três razões: o Boruta já usa RF
internamente (consistência entre os métodos); o objetivo aqui é **ordenar** features, não
maximizar MAPE — a escolha do campeão é a Etapa 07; e a Etapa 07 continua livre para testar
LightGBM e XGBoost, que são requisito do CLAUDE.md.

**Ação para o usuário:** instalar `lightgbm` e `xgboost` antes da Etapa 07. Não é
bloqueador para a Etapa 05.

---

## 4. Orçamento de tempo e estratégia de paralelização

Todos os números desta seção foram **medidos no ambiente real** (Apple M1 Pro, 10 núcleos,
16 GB), não estimados.

### 4.1 Custo unitário

Um RF de 100 árvores sobre 14.382 × 71 leva **8,3 s** com `n_jobs=-1`. Daí:

| Bloco | Configuração | Tempo projetado |
|---|---|---|
| Null importance | 50 shuffles do `y` | **~7 min** |
| RFECV | 5 folds, `step=5` (~15 treinos/fold) | **~10 min** |
| RFECV | 5 folds, `step=1` (71 treinos/fold) | ~49 min ⚠️ descartado |
| Boruta | `max_iter=50`, base completa | **~14 min** |

**Decisão sobre o RFECV:** `step=5` em vez de `step=1` — 10 min contra 49 min. O RFECV
interpola a curva entre os pontos avaliados; se o ótimo cair numa vizinhança ambígua,
refina-se com `step=1` **apenas naquela faixa**.

### 4.2 Por que NÃO paralelizar os seletores entre si

A pergunta foi levantada e medida. O resultado não justifica a mudança.

Os seletores **já são paralelos por dentro** (`n_jobs=-1` usa os 10 núcleos). Paralelizar
os seletores entre si não adiciona núcleos — apenas redistribui os mesmos. O que torna
algum ganho possível é uma ineficiência medida no escalonamento do RF:

| `n_jobs` | Tempo (1 RF) | Speedup | Eficiência |
|---|---|---|---|
| 1 | 63,5 s | 1,00× | 100% |
| 2 | 36,0 s | 1,77× | 89% |
| 5 | 23,2 s | 2,74× | 55% |
| 10 | 18,7 s | 3,40× | **34%** |

De 1 para 10 núcleos, apenas **3,40×** de ganho (lei de Amdahl: bootstrap e agregação do RF
não paralelizam). Nos 10 núcleos, dois terços da capacidade ficam ociosos.

Medindo o ganho real de rodar os 3 seletores em paralelo, com cargas nas proporções reais:

| Modo | Total |
|---|---|
| Sequencial, 10 núcleos cada | 137,7 s |
| 3 processos, ~3 núcleos cada | 103,5 s |

**Ganho: 1,33×** — nos tempos reais, 31 min → ~23 min. Economiza ~8 min. Note que em
paralelo cada seletor fica individualmente **mais lento** (o Boruta proxy passou de 62 s
para 101 s, com 3 núcleos em vez de 10); o ganho vem só da sobreposição, e o gargalo passa
a ser a tarefa mais lenta.

**Três razões para rejeitar essa paralelização, apesar do ganho:**

1. **Viola a Regra nº 2 do CLAUDE.md.** Desenvolvimento incremental exige ver a saída de
   cada bloco antes do próximo. Uma célula paralela devolve os três resultados juntos,
   ~23 min depois, sem output intermediário — `joblib` com backend `loky` não propaga
   `print` dos subprocessos.
2. **Inviabiliza o descarte condicional do Boruta** (Seção 10). Em paralelo não há como
   cronometrá-lo isoladamente: se o bloco todo demora, não se sabe quanto foi dele.
3. **Depuração opaca.** Boruta com numpy 2.x é o componente frágil do plano. Falha dentro
   de subprocesso devolve traceback serializado e truncado.

### 4.3 Onde a paralelização É aplicada — dois pontos

O paralelismo entra onde as tarefas são **independentes e homogêneas**, e onde a eficiência
é muito melhor que 1,33×:

**(a) Null importance — os 50 shuffles do `y`.** Perfeitamente independentes: cada um
embaralha o alvo, treina, coleta importâncias. Nada compartilhado entre iterações.

```python
from joblib import Parallel, delayed

def _importancia_sob_ruido(X, y, seed):
    """Um shuffle do alvo + 1 RF. Devolve o vetor de importâncias sob ruído puro."""
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y)
    m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    m.fit(X, y_shuffled)
    return m.feature_importances_

# n_jobs=10 nas tarefas externas; n_jobs=1 no RF interno (evita oversubscription)
nulas = Parallel(n_jobs=10, backend='loky')(
    delayed(_importancia_sob_ruido)(X_tr, y_tr, s) for s in range(50)
)
dist_nula = np.vstack(nulas)   # (50, 71)
```

Regra de ouro registrada: **`n_jobs` alto no laço externo, `n_jobs=1` no estimador
interno**. O inverso (ou ambos altos) causa *oversubscription* — mais threads que núcleos,
com perda de desempenho por troca de contexto.

**(b) Teste de estabilidade — as 3 seeds.** Também independentes por construção. Roda as 3
execuções do Null Importance simultaneamente, cada uma com ~3 núcleos.

```python
def _seleciona_com_seed(X, y, seed):
    """Pipeline de Null Importance completo para uma seed. Devolve set de features."""
    ...
    return set(features_aprovadas)

conjuntos = Parallel(n_jobs=3, backend='loky')(
    delayed(_seleciona_com_seed)(X_tr, y_tr, s) for s in (42, 7, 2024)
)
estaveis = set.intersection(*conjuntos)
```

**Resultado da estratégia:** os blocos permanecem sequenciais entre si — preservando a
auditabilidade célula a célula exigida pela Regra nº 2 — e o paralelismo atua dentro dos
dois blocos onde é eficiente, reduzindo o custo total mais do que os 1,33× que a
paralelização entre seletores ofereceria.

### 4.4 Ordem das células lentas

**Null importance (7 min) → RFECV (10 min) → Boruta (14 min).**

Deliberadamente com o Boruta por último: se ele estourar o tempo, é removido **sem
comprometer nada do que veio antes**. Cada um roda em célula isolada, cronometrada, com o
tempo impresso.

---

## 5. Bloco 1 — Baselines

Antes de qualquer seleção, os números contra os quais tudo será comparado. Três:

1. **RF com todas as 71 features** — a âncora do critério de corte (Seção 10).
2. **Mediana de `GROUP_COLS`** — a mesma regra da Etapa 01. Se o modelo com 71 features não
   bater a mediana dos comparáveis, há algo errado no pipeline, e nenhuma feature selection
   conserta isso.
3. **`GROUP_COLS` apenas como features** — o piso: quanto 3 colunas já entregam sozinhas.

Métricas: MAPE (escala original, após `expm1`), MAE e RMSE.

**Referência da EDA:** a Análise 2.9 mediu um piso de dispersão intragrupo de **20,8%**
(MAD%, controlando bairro × room_type × accommodates, cobertura 91,3%). O MAPE do baseline
deve pousar na faixa de **20–30%**. Um valor muito abaixo disso é sinal de vazamento, não
de sucesso — e deve disparar investigação antes de prosseguir.

---

## 6. Bloco 2 — Diagnóstico barato (não seletor)

Roda em segundos e serve para confrontar os seletores caros depois. **Nenhuma feature é
descartada com base neste bloco** — decisão validada pelo usuário.

- **Mutual information** (`mutual_info_regression`) — mede dependência estatística geral,
  não apenas monótona. Captura relação em U que o Spearman da Análise 2.1 perde.
  `minimum_nights` é o candidato natural: mínimos muito baixos e muito altos são perfis de
  preço distintos.
- **mRMR** (*minimum Redundancy Maximum Relevance*) — a única medida que enxerga
  redundância **e** relevância ao mesmo tempo. A Etapa 04 cortou em ρ ≥ 0,88; os pares na
  faixa 0,70–0,88 sobreviveram, e o mRMR os sinaliza.

Ambos entram no ranking consolidado como colunas informativas, para leitura cruzada com os
seletores.

---

## 7. Bloco 3 — Importância por árvores (três lentes)

Uma única medida de importância engana. Três, comparadas:

| Medida | O que mede | Viés conhecido |
|---|---|---|
| **Impurity (MDI)** | Redução de impureza acumulada nos splits | Enviesada a favor de alta cardinalidade e de features contínuas |
| **Split count** | Quantas vezes a coluna foi usada | Revela features muito consultadas com pouco ganho |
| **Permutation importance** | Quanto o MAPE **piora em `valid_05`** ao embaralhar a coluna | A única medida **fora da amostra**; a mais confiável |

A permutação é calculada em `valid_05` com `n_repeats=5`, reportando média e desvio.

**O caso a caçar:** feature bem colocada em MDI e nula em permutação — o modelo a usou para
decorar o treino, e ela não generaliza. Sem a terceira lente, passaria despercebida.

---

## 8. Bloco 4 — Null importance (seletor principal) · ~7 min, paralelizado

Embaralha o **`y`** 50 vezes, coletando a distribuição de importância de cada feature sob
ruído puro. Mantém as que superam seu **próprio percentil 95** da distribuição nula.

Mesma lógica estatística do Boruta — comparar contra ruído — com três vantagens: funciona
com qualquer modelo, é paralelizável (Seção 4.3a), e o custo é controlado diretamente pelo
número de shuffles.

**Entregável:** para cada feature, a importância real, o percentil 95 da distribuição nula,
e o **score** (razão entre eles). Features com score < 1 não se distinguem de ruído.

A distinção conceitual importa: a permutation importance do Bloco 3 embaralha **uma coluna
por vez** (quanto essa coluna contribui); o null importance embaralha **o alvo** (qual
importância surgiria por acaso). São perguntas diferentes e complementares.

---

## 9. Bloco 5 — Boruta (confirmação secundária) · ~14 min, descartável

`BorutaPy` + RF, `max_iter=50` (piso da Seção 3.1), `n_estimators=100`, **base completa,
sem amostragem** — decisão validada pelo usuário.

**Mecanismo:** para cada feature real, cria-se uma cópia embaralhada (*shadow feature*).
Uma feature só é confirmada se bater consistentemente a **melhor** sombra, com correção
para testes múltiplos. Saída em três classes: **confirmadas**, **tentativas**
(inconclusivas ao fim das iterações) e **rejeitadas**.

**Condição de descarte (validada pelo usuário):** se exceder ~20 min, o Boruta é **removido
da etapa** e o fato é registrado. O Null Importance já cumpre o papel de comparação contra
ruído, então nada fica descoberto. É por isso que ele é a última das células lentas
(Seção 4.4).

**Expectativa a alinhar antes de rodar:** o Boruta é *all-relevant*, não *minimal-optimal*
— busca tudo que carrega sinal, não o menor conjunto suficiente. Em dados tabulares ricos,
confirma a grande maioria. **Se confirmar 60 das 71, isso não é falha do método.** É
evidência de que a Etapa 04 já removeu o que era removível por critério técnico, e de que a
decisão restante é de **parcimônia** — o que é o papel do Bloco 6.

---

## 10. Bloco 6 — RFECV (define o conjunto) · ~10 min

`RFECV` com RF, **5 folds**, **`step=5`** (Seção 4.1), scoring de MAPE em escala original.

Nota de implementação: o `neg_mean_absolute_percentage_error` do sklearn operaria sobre o
log, produzindo uma métrica diferente da do projeto. É preciso um `make_scorer` customizado
que aplique `expm1` antes de calcular o MAPE.

Este bloco **substitui a curva de parcimônia** da versão anterior do plano (decisão
validada pelo usuário) e responde a pergunta real da etapa:

> Não *"quais features importam"*, mas **"qual o menor conjunto que não perde performance"**.

A vantagem sobre a curva de parcimônia é usar **k folds em vez de partição única** — a
decisão sobre o número de features fica muito mais robusta pelo mesmo trabalho conceitual.

**Critério de corte (validado pelo usuário):**

> O **menor** conjunto cujo MAPE fique dentro de **+0,5 p.p.** do baseline de 71 features.

O critério é numérico e explícito, fixado **antes** de ver a curva — para que a leitura do
resultado não seja retroajustada. O RFECV também reporta seu próprio ótimo; se os dois
divergirem, prevalece o critério de +0,5 p.p., que é mais parcimonioso.

---

## 11. Bloco 7 — Consenso e blindagem de negócio

### 11.1 Matriz de consenso

Cruzamento dos três seletores (Null Importance × Boruta × RFECV):

| | Alta importância | Baixa importância |
|---|---|---|
| **Aprovada pelos seletores** | Consenso: manter | **Investigar** — sinal fraco mas real |
| **Rejeitada pelos seletores** | **Investigar** — provável artefato | Consenso: candidata a descarte |

Os quadrantes de discordância são inspecionados **caso a caso**, com o número na tela. Não
há regra automática para eles.

### 11.2 Blindagem (validada pelo usuário)

Três travas onde o compromisso de projeto prevalece sobre o veredito estatístico:

**`GROUP_COLS` são intocáveis.** `neighbourhood_cleansed`, `room_type` e `accommodates`
sustentam o cálculo de `ref_price` e de EV das Etapas 01, 09 e 10 (CLAUDE.md, Seção 3.1).
Permanecem em `FEATURES_05` **independentemente** do veredito dos seletores. Se algum
seletor as rejeitar, isso é um **achado a reportar** — provável indicação de que o sinal
está sendo capturado por uma feature correlata — e não uma instrução a obedecer.

**Os dois `gap_*` recebem análise dedicada.** `gap_location_vs_geral` e
`gap_value_vs_geral` são a hipótese H5, central para o recomendador da Etapa 09 (E09-4). Se
caírem por importância baixa, o número precisa ser explícito e a decisão consciente — não
uma remoção silenciosa no meio de uma lista.

**`host_is_superhost` é reavaliada antes de qualquer descarte.** A EDA mostrou o caso mais
claro de paradoxo de Simpson do projeto: efeito marginal **zero** (mediana €125 em ambos os
grupos) e **+7,7%** sob controle de bairro × tipo × capacidade, positivo em 64,2% das 190
células comparáveis. Se um seletor a rejeitar, a primeira hipótese é que o seletor está
reproduzindo o mesmo mascaramento — inspeção obrigatória antes do descarte.

---

## 12. Bloco 8 — Teste de estabilidade · paralelizado

Todo o pipeline de Null Importance é repetido com **3 seeds** (42, 7, 2024), executadas
**em paralelo** (Seção 4.3b), medindo quantas features aparecem nas três.

Seleção que muda conforme a seed não é seleção, é ruído — por isso `FEATURES_05` é a
**interseção** das três execuções, não o resultado de uma.

**Escopo do teste:** aplicado ao Null Importance, que é o seletor principal. Boruta e RFECV
rodam uma única vez, com `random_state=42`.

**Limitação registrada:** o RFECV é quem fixa o **número** final de features, e não medir
sua estabilidade deixa esse número sem intervalo de confiança. A paralelização da
Seção 4.3b reduz o custo o suficiente para que, se o resultado do Bloco 6 parecer sensível,
o RFECV possa ser repetido em 3 seeds como análise adicional (~30 min).

---

## 13. Bloco 9 — Validação final

Comparação lado a lado em `valid_05`, com MAPE, MAE e RMSE:

| Conjunto | Papel |
|---|---|
| 71 features (baseline) | Referência |
| **`FEATURES_05`** (selecionado) | O candidato |
| Top-10 | Sanidade — quanto se perde no extremo da parcimônia |
| `GROUP_COLS` apenas | Piso — o que 3 colunas já entregam |

A última linha é a mais informativa para o negócio: mostra quanto das ~68 features
adicionais realmente compra desempenho sobre os três atributos óbvios.

---

## 14. Ordem de execução

1. **Preparação mínima descartável** — encoding ordinal + imputação por regra (Seção 1.2)
2. **Partição `train_05` / `valid_05`** + verificação de isolamento do teste
3. **Baselines** — três números de referência (Bloco 1)
4. **Diagnóstico barato** — mutual information + mRMR (Bloco 2)
5. **Importâncias MDI e split count** (Bloco 3, parte 1)
6. **Permutation importance** em `valid_05` (Bloco 3, parte 2)
7. **Null importance** (Bloco 4 — ~7 min, isolada, **paralelizada**)
8. **RFECV** (Bloco 6 — ~10 min, isolada)
9. **Boruta** (Bloco 5 — ~14 min, isolada, cronometrada, **descartável**)
10. **Matriz de consenso** (Bloco 7.1)
11. **Blindagem de negócio** + inspeção das discordâncias (Bloco 7.2)
12. **Teste de estabilidade** — 3 seeds (Bloco 8, **paralelizado**)
13. **Validação final** do conjunto + construção de `df05` (Bloco 9)
14. **Consolidação** — tabela de decisão e recomendações para a Etapa 06

Cada item é uma pausa natural para validação, conforme a Regra de Trabalho nº 2 do
CLAUDE.md. São **14 portões**, não 1.

---

## 15. Entregáveis

1. **`df05`** — `id` + `price` + features selecionadas, **com os valores originais de
   `df04`** (nulos preservados; a preparação mínima não vaza para cá).
2. **`FEATURES_05`** — a lista final de nomes, insumo direto da Etapa 06.
3. **Ranking completo das 71** — mutual information, mRMR, MDI, split count, permutation
   importance, null importance score e veredito do Boruta, numa tabela única.
4. **Tabela de decisão** — `feature | seletores | importância | decisão | motivo`,
   incluindo explicitamente as mantidas por razão de negócio **contra** o veredito
   estatístico.
5. **Curva do RFECV** com o ponto de corte marcado.
6. **Registro de premissas** da preparação mínima descartável, com o aviso de que a Etapa
   06 não deve herdá-la.

---

## 16. Verificação

- **Granularidade:** `len(df05) == df05['id'].nunique()` permanece verdadeiro
- **Isolamento do teste:** nenhuma célula desta etapa lê `test_raw.csv` — verificável por
  busca textual no notebook
- **`id` fora das features:** `'id' not in FEATURES_05` e `TARGET not in FEATURES_05` —
  `assert` explícito, não inspeção visual
- **Preparação não vazou:** `df05[cols].isna().sum()` idêntico a `df04[cols].isna().sum()`
  nas colunas mantidas — comparação programática, é a prova de que a imputação descartável
  ficou contida na etapa
- **Sem `fit` fora do treino:** toda mediana e todo encoder derivam de `train_05`
- **MAPE em escala original:** toda métrica reportada passa por `expm1` antes do cálculo —
  conferido por asserção sobre a ordem de grandeza (MAPE entre 0 e 1, não centenas)
- **Subconjunto estrito:** `set(FEATURES_05).issubset(set(FEATURES_04))` — a Etapa 05 só
  remove, nunca cria features
- **Blindagem aplicada:** `GROUP_COLS` ⊂ `FEATURES_05` — `assert` explícito
- **Sem oversubscription:** nos blocos paralelos, `n_jobs` alto no laço externo e
  `n_jobs=1` no estimador interno (Seção 4.3)

---

## 17. Pendências para validação

**Resolvidas antes da execução** (validadas pelo usuário em conversa):

1. **Ordem 05 → 06 mantida**, com preparação mínima descartável (Seção 1).
2. **Critério de corte** de +0,5 p.p. sobre o baseline (Seção 10).
3. **Blindagem** de `GROUP_COLS` e dos `gap_*` contra os seletores (Seção 11.2).
4. **Null importance adicionado** como seletor principal (Seção 8).
5. **Boruta mantido sem amostragem**, removível se exceder ~20 min (Seção 9).
6. **RFECV substitui a curva de parcimônia** (Seção 10).
7. **mRMR e mutual information como diagnóstico**, não como seletores (Seção 6).
8. **Teste de disponibilidade em produção: descartado** a pedido do usuário. Fica
   registrado o que ele mediria: as colunas `host_response_time` (47,3% nulo) e o bloco de
   reviews são ausentes justamente nos **anúncios novos** — o segmento que a EDA apontou
   como o de maior valor para o recomendador (cobram €139 vs €125, CV 0,85 vs 0,58). Se a
   Etapa 09 ou 11 revelar falha nesse segmento, este é o primeiro lugar a investigar.
9. **Paralelização interna** nos Blocos 4 e 8, com os seletores sequenciais entre si
   (Seção 4).

**Em aberto, a decidir com o resultado na tela:**

1. **O que fazer se os seletores aprovarem quase tudo** (cenário provável, Seção 9). A
   proposta é deixar o RFECV decidir sozinho o corte, tratando Null Importance e Boruta
   como filtros de segurança e não como seletores principais.
2. **O que fazer se nenhum subconjunto ficar dentro de +0,5 p.p.** — nesse caso a resposta
   honesta é **não selecionar**, entregando `FEATURES_05 == FEATURES_04` e documentando que
   a EDA já havia extraído toda a redução defensável. É um resultado legítimo, e é preciso
   combinar antes que ele não será tratado como fracasso.
3. **Instalação de `lightgbm` e `xgboost`** antes da Etapa 07 (Seção 3.2). Não bloqueia a
   Etapa 05, mas bloqueia a 07.
