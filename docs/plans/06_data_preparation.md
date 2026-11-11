# Plano — Etapa 06: Data Preparation (Airbnb Amsterdam)

## Contexto

Entramos nesta etapa com `df05`: **17.978 linhas × 29 colunas**, granularidade de 1 linha
por `listing_id`, resultado da Etapa 05 (Feature Selection). São **1 chave** (`id`),
**1 alvo** (`price`) e **27 features** (`FEATURES_05`).

`df05` carrega os **valores originais** de `df04` — a preparação mínima descartável da
Etapa 05 ficou contida naquela etapa e não vazou (verificado: 23.227 nulos em 12 colunas
preservados). Esta etapa é, portanto, a **primeira e única** preparação definitiva.

**Isolamento do teste** (CLAUDE.md §5.3): `test_raw.csv` (2.003 linhas) permanece
**fechado**. Nenhuma célula desta etapa o lê. A reaplicação do pipeline ao teste é
responsabilidade da Etapa 09 (decisão do usuário — ver §2).

**Orientação de métrica** (CLAUDE.md §1): o alvo do `fit` é `log1p(price)`, mas **todo
MAPE reportado é em escala original, após `expm1`**. Sem exceção.

---

## 1. O problema central — dois clientes com exigências opostas

O CLAUDE.md §7 exige seis modelos na Etapa 07: baseline, Linear Regression, Regressão
Polinomial, Random Forest, XGBoost e LightGBM. Eles **não querem os mesmos dados**:

| Requisito | Linear / Polinomial | Árvores (RF, XGB, LGBM) |
|---|---|---|
| Escala comum | **Obrigatória** — coeficientes e regularização dependem dela | Irrelevante — splits são invariantes a transformação monótona |
| One-hot do bairro (22 níveis) | Necessário — ordinal criaria ordem numérica falsa | **Prejudicial** — fragmenta o sinal em 22 colunas fracas |
| Sentinela `-1` para nulo | **Veneno** — vira um valor real na reta de regressão | Ideal — separável por um único split |
| Cauda longa (assimetria alta) | Distorce o coeficiente | Indiferente |

Preparar um único dataset forçaria uma escolha que prejudica metade dos modelos — e
invalidaria a comparação da Etapa 07, porque a Linear Regression competiria em
desvantagem artificial.

**Decisão validada pelo usuário: produzir DOIS datasets.**

| Dataset | Destino | Encoding | Escala | Nulos (bloco MAR) |
|---|---|---|---|---|
| **`df06_linear`** | Linear Regression, Regressão Polinomial | One-hot (`drop_first`) | `RobustScaler` + `log1p` | Mediana + flag |
| **`df06_arvore`** | Random Forest, XGBoost, LightGBM | Ordinal | **Nenhuma** | Sentinela `-1` |

As **regras de imputação de mecanismo MNAR e MCAR são compartilhadas** (§5); só encoding,
escala e o tratamento do bloco MAR divergem. O trabalho extra é pequeho e evita sabotar
metade da Etapa 07.

---

## 2. Escopo — código inline, sem funções (decisão do usuário)

A proposta inicial era entregar uma função `preparar_dados(df_cru, artefatos=None)`
encapsulando Etapas 02 + 03 + 06, para garantir que a Etapa 09 reaplicasse exatamente o
mesmo tratamento ao hold-out cru.

**Decisão do usuário: NÃO.** A preparação é escrita de forma **linear nas células**, e a
preparação dos dados de teste fica **concentrada na Etapa 09**.

Consequência registrada, para a Etapa 09 não ser pega de surpresa: o `test_raw.csv` está
**cru**, com as 96 colunas originais. A Etapa 09 precisará reaplicar, na mesma ordem e com
os mesmos parâmetros (CLAUDE.md §5.4):

1. Toda a limpeza da Etapa 02 (parse monetário, flags f/t, drops)
2. Todo o feature engineering da Etapa 03 (46 features)
3. O recorte das 27 colunas de `FEATURES_05`
4. Toda a preparação desta etapa, usando os **artefatos aprendidos aqui**

Por isso, o §7 desta etapa exige que todo parâmetro aprendido seja **persistido num
dicionário explícito** (`ARTEFATOS_06`), com os valores impressos. Sem função
reutilizável, o dicionário é a única garantia contra divergência silenciosa entre treino e
teste — que invalida a métrica final tanto quanto vazamento.

---

## 3. Partição treino/validação — ANTES de qualquer `fit` (decisão do usuário)

**Primeira célula de código da etapa**, antes de qualquer imputador, encoder ou scaler:

```python
RANDOM_STATE = 42
train_06, valid_06 = train_test_split(
    df05, test_size=0.20, random_state=RANDOM_STATE, shuffle=True,
)
```

| Partição | Linhas | Papel |
|---|---|---|
| `train_06` | 14.382 | **Único** lugar de onde parâmetros são aprendidos |
| `valid_06` | 3.596 | Avaliação; recebe apenas `transform` |
| `test_raw.csv` | 2.003 | **Fechado** até a Etapa 09 |

Por que a ordem importa: calcular uma mediana, um IQR ou as categorias de um encoder sobre
`df05` **inteiro** contamina a validação — mesmo com o teste fechado. O parâmetro carrega
informação das linhas de validação, e a métrica fica otimista sem deixar rastro.

### 3.1 Semente 42 e 80/20 — e o viés que isso carrega

**Decisão validada pelo usuário: mesma semente (42) e mesma proporção (80/20) da Etapa 05.**

Isso significa reutilizar a **mesma partição** em que o RFECV escolheu as 27 features.
Registro honesto da consequência:

> O MAPE de validação desta etapa e da Etapa 07 carrega **viés otimista leve**, porque as
> 27 colunas foram selecionadas usando (indiretamente) estes mesmos dados. Não é vazamento
> grosseiro — a validação nunca entrou no `fit` de um modelo — mas a CV do RFECV rodou
> dentro deste treino e o critério de corte comparou curvas deste split.

Três razões pelas quais o viés é aceitável:

1. **Evidência direta de que a seleção não é artefato do split.** O teste de estabilidade
   da Etapa 05 deu N_CORTE = 26 nas sementes 42, 7 e 2024, com as 26 features no núcleo
   estável das três. Se a seleção mudasse com a semente, trocar seria obrigatório.
2. **Comparabilidade.** Mantém o MAPE de **23,71%** da Etapa 05 diretamente comparável com
   o da Etapa 07 — permitindo isolar o efeito da preparação. Trocar a semente tornaria
   impossível distinguir "a preparação piorou" de "este split é mais difícil".
3. **O CLAUDE.md §5 já resolve estruturalmente.** O `test_raw.csv` está fechado desde a
   Etapa 02 e nunca participou de nenhuma decisão. **Ele é o juiz imparcial**; a validação
   serve para comparar alternativas durante o desenvolvimento.

---

## 4. Escala — uma regra do CLAUDE.md que se resolve sozinha

O CLAUDE.md §7 pede *"padronização (standardization) para variáveis Normais; rescala
(min-max/robust) para não-Normais"*.

**Teste executado (Shapiro-Wilk, amostra de 5.000 quando n > 5.000) sobre as 25 numéricas:**

> **Zero são Normais.** Todas com p < 0,001.

Assimetrias de **−3,52 a +25,67**; curtose de até **1.000,58**. A regra, portanto, aplica-se
sem ambiguidade: **nenhum `StandardScaler`; rescala para tudo**. O teste vai no notebook —
a decisão é documentada com evidência, não afirmada.

### 4.1 Por que `RobustScaler` e não Min-Max

Os outliers inviabilizam Min-Max:

| Coluna | p50 | p99 | máx | máx/p99 |
|---|---|---|---|---|
| `minimum_nights` | 2 | 20 | **365** | **18,2×** |
| `security_deposit` | 150 | 1.500 | 4.736 | 3,2× |
| `maximum_nights` | 1.125 | 1.125 | 1.825 | 1,6× |

Min-Max usa mínimo e máximo observados. Em `minimum_nights`, 99% dos anúncios seriam
comprimidos no intervalo [0; 0,05] para acomodar um único anúncio de 365 noites — o
intervalo útil colapsa.

**`RobustScaler`** usa mediana e IQR: `(x − mediana) / IQR`. É insensível à cauda, porque
nem a mediana nem o IQR se movem com valores extremos. Aplicado **apenas ao
`df06_linear`** (árvores não precisam de escala).

---

## 5. Imputação — três mecanismos, três tratamentos

Das 27 features, **12 têm nulos**:

| Coluna | % nulo | Mecanismo |
|---|---|---|
| `security_deposit` | 30,89% | MNAR |
| `cleaning_fee` | 18,03% | MNAR |
| `taxa_limpeza_por_noite` | 18,03% | MNAR |
| `gap_value_vs_geral` | 13,18% | MAR |
| `review_scores_rating` | 13,10% | MAR |
| `reviews_per_month` | 11,94% | MAR |
| `dias_desde_primeira_review` | 11,94% | MAR |
| `dias_desde_ultima_review` | 11,94% | MAR |
| `bathrooms`, `bedrooms`, `calendar_atualizado_dias` | 0,04% | MCAR |
| `host_experiencia_dias` | 0,02% | MCAR |

**Achado favorável, a registrar:** `host_response_time` (47,3% de nulos — o maior problema
de imputação identificado na EDA) **foi eliminada pelo RFECV**. O pior caso saiu de cena
por si só.

### 5.1 MNAR → imputar **0** (ambos os datasets)

Reconfirmado nesta etapa, sobre `df05`:

```
security_deposit   nulos = 5.554 | flag tem_deposito = 1 entre os nulos:     0
cleaning_fee       nulos = 3.242 | flag tem_taxa_limpeza = 1 entre os nulos: 0
```

**100% dos nulos têm a flag correspondente em 0.** O nulo significa **"não cobro"**, não
"desconhecido". Imputar a mediana transformaria "não cobro" em "cobro €150" — erro grave,
já registrado como **E06-3** na Análise 3.4.

### 5.2 MAR (bloco de reviews) → divergente por dataset

O mecanismo é MAR: 98,8–100% de coincidência com `review_scores_rating`, e a ausência é
explicada por uma variável observável (o anúncio não tem reviews). Registrado como
**E06-2**.

- **`df06_arvore`**: sentinela **−1**. A árvore separa os "sem review" com um único split,
  sem necessidade de flag auxiliar. É o tratamento mais limpo.
- **`df06_linear`**: **mediana do treino + flag binária `sem_reviews_imputado`**. A
  sentinela seria veneno num modelo linear — `−1` entraria na reta como um valor real,
  distorcendo o coeficiente. A flag preserva a informação da ausência sem falsear a escala.

Nota conceitual: um anúncio novo **não tem score ruim** — tem ausência de score. Confundir
as duas coisas é o erro que a flag evita.

### 5.3 MCAR → mediana do treino (ambos)

Quatro colunas, ≤ 0,04% (7 linhas). Mecanismo MCAR, volume irrelevante. Mediana aprendida
**só em `train_06`**.

---

## 6. `log1p` nas colunas de assimetria extrema (decisão do usuário)

**Decisão validada: aplicar `log1p`** nas colunas de assimetria > 3 (sem teste comparativo
prévio).

### 6.1 O que a transformação faz

`log1p(x) = log(1 + x)` comprime a cauda direita **preservando a ordem**:

| Valor original | `log1p` |
|---|---|
| 1 | 0,69 |
| 2 | 1,10 |
| 20 | 3,04 |
| 365 | 5,90 |

Em `minimum_nights`, a distância entre 1 e 2 noites (0,41 em log) passa a ser da mesma
ordem da distância entre 20 e 365 (2,86). No espaço original, a segunda é **345× maior**.
O termo `1+` existe para que `x = 0` seja válido (`log(0)` é indefinido; `log1p(0) = 0`).

### 6.2 Correção de contagem — são **7** colunas, não 6

Um levantamento anterior citou 6 colunas. A contagem correta, medida sobre `df05`:

| Coluna | Assimetria | mín | máx | `log1p`? |
|---|---|---|---|---|
| `minimum_nights` | **+25,67** | 1 | 365 | **Sim** |
| `security_deposit` | +6,59 | 0 | 4.736 | **Sim** |
| `number_of_reviews` | +5,34 | 0 | 695 | **Sim** |
| `bathrooms` | +4,97 | 0 | 10 | **Sim** |
| `extra_people` | +4,46 | 0 | 280 | **Sim** |
| `taxa_limpeza_por_noite` | +4,12 | 0 | 300 | **Sim** |
| `review_scores_rating` | **−3,52** | 20 | 100 | **NÃO** — ver 6.3 |

Todas têm mínimo ≥ 0, então `log1p` é matematicamente válida (não gera `NaN` nem `-inf`).

### 6.3 Por que `review_scores_rating` fica fora

Sua assimetria é **negativa** (−3,52): a cauda é à **esquerda** (concentração perto de 100,
poucos anúncios com nota baixa). A `log1p` comprime a cauda **direita** — aplicá-la aqui
não corrigiria nada e **agravaria** a assimetria, comprimindo ainda mais o lado onde a
massa já está.

Se houver interesse em simetrizar esta coluna, a transformação correta seria uma reflexão
(`log1p(100 − x)`) ou Yeo-Johnson. **Fica fora do escopo desta etapa**, registrado como
pendência — a coluna entra sem transformação.

**Resultado: `log1p` em 6 colunas** (as de assimetria positiva > 3), aplicada **apenas ao
`df06_linear`**. Árvores são invariantes a transformação monótona; aplicá-la lá seria
trabalho sem efeito.

### 6.4 Custo registrado — interpretabilidade

`log1p` destrói a leitura direta do coeficiente: *"cada noite adicional de mínimo custa
€X"* vira *"cada unidade de log-noite custa €X"*, que não significa nada para um anfitrião.
Num projeto cujo entregável final é um recomendador de preço (Etapa 11), isso é um custo
real. Como o `df06_linear` serve à comparação da Etapa 07 — e não ao recomendador, que usará
o melhor modelo (provavelmente uma árvore) — o custo é aceitável. Registrado para que a
Etapa 11 não tente interpretar coeficientes em escala logarítmica.

---

## 7. Encoding

As **2 únicas categóricas** de `FEATURES_05`: `room_type` (3 níveis) e
`neighbourhood_cleansed` (22 níveis).

| Dataset | Tratamento | Colunas resultantes |
|---|---|---|
| `df06_linear` | One-hot com `drop_first=True` | 25 numéricas + 2 + 21 = **48** |
| `df06_arvore` | Ordinal (códigos inteiros) | **27** |

O `drop_first` evita a *dummy variable trap*: com todas as 22 dummies mais o intercepto, a
matriz fica singular e os coeficientes ficam indeterminados.

### 7.1 Target encoding do bairro — avaliado e **descartado**

Foi considerado (reduziria 21 colunas a 1) e rejeitado por risco de vazamento. A
distribuição de frequências:

| Bairro | Anúncios |
|---|---|
| `De Baarsjes - Oud-West` (maior) | 3.161 |
| `Osdorp` | 145 |
| `De Aker - Nieuw Sloten` | 126 |
| `Gaasperdam - Driemond` | 116 |
| `Bijlmer-Centrum` | 99 |
| `Bijlmer-Oost` (menor) | **90** |

**2 bairros com menos de 100 anúncios; 6 com menos de 200.** Nessas células, a média do
alvo é calculada sobre poucas observações e carrega o próprio `price` das linhas — vaza o
alvo mesmo com smoothing e CV interna.

O risco não compensa: `dist_centro_km` e `dist_museumplein_km` já capturam a geografia de
forma contínua. Foi exatamente por isso que o RFECV **eliminou** `neighbourhood_cleansed` —
ela só retornou por blindagem de `GROUP_COLS` (CLAUDE.md §3.1).

---

## 8. Registro de artefatos aprendidos (`ARTEFATOS_06`)

Como a etapa não produz função reutilizável (§2), todo parâmetro aprendido em `train_06`
vai para um dicionário explícito, **com os valores impressos no notebook**:

```python
ARTEFATOS_06 = {
    'medianas_mcar':    {...},   # 4 colunas MCAR
    'medianas_mar':     {...},   # 5 colunas do bloco de reviews (p/ df06_linear)
    'cols_mnar_zero':   [...],   # 3 colunas -> 0
    'cols_log1p':       [...],   # 6 colunas
    'categorias_onehot': {...},  # níveis vistos no treino, na ordem
    'robust_center':    {...},   # mediana por coluna
    'robust_scale':     {...},   # IQR por coluna
}
```

Este dicionário é o **insumo obrigatório da Etapa 09**. Sem ele, a reaplicação ao hold-out
seria uma reimplementação — e divergência entre treino e teste invalida a métrica final
tanto quanto vazamento.

---

## 9. Verificação anti-vazamento

Além dos asserts usuais de shape e granularidade, três testes específicos:

**9.1 Nenhum parâmetro aprendido fora do treino.** Para cada mediana em `ARTEFATOS_06`,
confirmar programaticamente que é igual à mediana de `train_06` e **diferente** da de
`valid_06` (quando os valores divergirem) — prova de que o `transform` da validação usou o
parâmetro do treino.

**9.2 Nenhuma categoria nova na validação.** Verificar que `valid_06` não tem nível de
`neighbourhood_cleansed` ou `room_type` ausente do treino. Se tiver, o one-hot gera coluna
faltante e o shape divergeria silenciosamente.

**9.3 Idempotência das transformações.** Reaplicar `log1p` ou o scaler a um dado já
transformado deve ser detectado — as células são escritas de forma que a reexecução parta
sempre de `train_06`/`valid_06` crus, nunca do resultado anterior.

**9.4 Teste fechado.** Busca textual no notebook confirmando que nenhuma célula desta
etapa lê `test_raw.csv`.

**9.5 Sem nulos na saída.** `df06_linear` e `df06_arvore` com `isna().sum().sum() == 0`.

---

## 10. Ordem de execução — 9 portões

1. **Partição `train_06` / `valid_06`** — primeira célula, antes de qualquer `fit` (§3)
2. **Diagnóstico das 27 colunas** — Shapiro-Wilk, nulos por mecanismo, cardinalidade (§4, §5)
3. **Imputação MNAR → 0** + reconfirmação de que todo nulo tem flag = 0 (§5.1)
4. **Imputação MAR e MCAR** — sentinela para árvore, mediana + flag para linear (§5.2, §5.3)
5. **`log1p` nas 6 colunas** de assimetria positiva > 3 (§6)
6. **Encoding** — one-hot para linear, ordinal para árvore (§7)
7. **`RobustScaler`** no `df06_linear` (§4.1)
8. **Montagem de `df06_linear`, `df06_arvore` e `ARTEFATOS_06`** (§8)
9. **Verificação anti-vazamento** + consolidação (§9)

Cada item é uma pausa para validação, conforme a Regra de Trabalho nº 2 do CLAUDE.md. São
**9 portões**, não 1.

---

## 11. Entregáveis

1. **`df06_linear`** — ~48 colunas, one-hot + `log1p` + `RobustScaler`, nulos imputados com
   flag. Destino: Linear Regression e Regressão Polinomial.
2. **`df06_arvore`** — 27 colunas, ordinal, sem escala, nulos como sentinela `−1`. Destino:
   Random Forest, XGBoost, LightGBM.
3. **`ARTEFATOS_06`** — dicionário com todo parâmetro aprendido no treino; insumo
   obrigatório da Etapa 09.
4. **Tabela de decisões** — coluna | mecanismo | tratamento (linear) | tratamento (árvore) |
   motivo.
5. **Relatório de normalidade** — Shapiro-Wilk das 25 numéricas, justificando a ausência de
   `StandardScaler`.
6. **Premissas registradas** — semente, proporção, viés otimista da validação reutilizada.

---

## 12. Verificação

- **Granularidade:** `len(df06_*) == df06_*['id'].nunique()`
- **Isolamento do teste:** nenhuma leitura de `test_raw.csv` — busca textual
- **Sem nulos:** `isna().sum().sum() == 0` nos dois datasets
- **`id` e `price` fora das features:** `assert` explícito
- **Sem `fit` fora do treino:** cada parâmetro de `ARTEFATOS_06` conferido contra `train_06`
- **`GROUP_COLS` presentes:** `neighbourhood_cleansed`, `room_type` e `accommodates`
  sobrevivem em ambos os datasets (CLAUDE.md §3.1)
- **Alinhamento de colunas:** `list(train.columns) == list(valid.columns)` nos dois datasets
- **MAPE em escala original:** qualquer métrica de sanidade passa por `expm1`

---

## 13. Pendências

**Resolvidas antes da execução** (validadas pelo usuário em conversa):

1. **Dois datasets** (`df06_linear` + `df06_arvore`), um por família de algoritmo (§1).
2. **Código inline, sem funções**; preparação do teste concentrada na Etapa 09 (§2).
3. **Partição antes de qualquer `fit`**, semente **42**, split **80/20** (§3).
4. **`log1p`** nas colunas de assimetria > 3 (§6), com a correção de contagem: **6**
   colunas recebem a transformação, não 7 — `review_scores_rating` tem assimetria
   **negativa** e fica fora (§6.3).

**Em aberto:**

1. **Simetrização de `review_scores_rating`** (assimetria −3,52). Exigiria reflexão
   (`log1p(100 − x)`) ou Yeo-Johnson. Fora do escopo desta etapa; a coluna entra sem
   transformação. Reavaliar se a Linear Regression da Etapa 07 ficar muito abaixo das
   árvores.
2. **`lightgbm` e `xgboost` não instalados** — requisito do CLAUDE.md §7 para a Etapa 07.
   Não bloqueia a Etapa 06, bloqueia a 07.
3. **Grau da Regressão Polinomial** — com 48 colunas, grau 2 gera ~1.200 termos de
   interação. Decisão da Etapa 07, mas registrada aqui porque pode exigir um recorte de
   features específico para esse modelo.
