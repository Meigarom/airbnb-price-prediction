# Plano — Etapa 09: Métricas e Recomendador (Airbnb Amsterdam)

## Contexto

Esta é a etapa em que o **hold-out é aberto pela primeira e única vez**. Até aqui, todas as
decisões — limpeza, features, seleção, preparação, modelo, hiperparâmetros — foram tomadas
sem que `test_raw.csv` participasse de nenhuma delas (CLAUDE.md §5.3).

Entrada:

| Artefato | Conteúdo |
|---|---|
| `MODELO_FINAL` | LightGBM tunado — `PARAMS_LGBM_TUNED` (10 hiperparâmetros) |
| `DATASET_FINAL` | `df06_arvore` (27 features de `FEATURES_06_ARVORE`) |
| `ARTEFATOS_06` | 17 chaves — medianas, mapas ordinais, centro/escala do `RobustScaler` |
| `CATEGORIAS_MANTIDAS` | 8 níveis de `property_type` aprendidos no treino |
| `train_06` / `valid_06` | 14.382 / 3.596 linhas, para o retreino conjunto |
| `test_raw.csv` | **2.003 × 96, CRU** — verificado: `id` único |

Desempenho até aqui, em `valid_06`:

| | MAPE |
|---|---|
| Chão de negócio (mediana `GROUP_COLS`) | 28,22% |
| LightGBM default (Etapa 07) | 22,63% |
| **LightGBM tunado (Etapa 08)** | **22,14%** |
| Teto entre gêmeos exatos | 22,67% |
| Piso da EDA (Análise 2.9) | 20,80% |

---

## 1. O que o CLAUDE.md exige desta etapa

A Seção 7 (Etapa 09) pede quatro coisas, em ordem:

1. **Retreinar** o melhor modelo (melhores params do Optuna) com **treino + validação juntos**
2. **Reaplicar a `test_raw.csv`** toda a limpeza da Etapa 02 e todo o feature engineering da
   Etapa 03, na mesma ordem e com os mesmos parâmetros (§5.4)
3. **Avaliar no teste**
4. **Construir o recomendador** que gera o preço justo por anúncio

A Seção 3 (estratégia travada) governa o recomendador: banda de MAPE (§3.2), atuação
exclusiva na subprecificação (§3.3), e a fórmula de EV (§3.4).

---

## 2. O risco central: reaplicar o pipeline sem divergência

O `test_raw.csv` está **cru** — 96 colunas originais, sem limpeza e sem features. A Etapa 06
não produziu função reutilizável (decisão do usuário), então **a reaplicação é manual**, e é
aqui que mora o perigo:

> Divergência de tratamento entre treino e teste **invalida a métrica final tanto quanto
> vazamento** (CLAUDE.md §5.4). Não aparece em nenhum erro — produz um número plausível e
> errado.

### 2.1 Inventário dos parâmetros aprendidos no treino

Um levantamento das células das Etapas 02 (19–45) e 03 (46–60) identificou **toda operação
que aprende algo dos dados**. Estas são as que exigem reutilizar o valor do treino, nunca
recalcular no teste:

| Origem | Parâmetro | Onde está | Risco se recalculado |
|---|---|---|---|
| Etapa 03, cél. 56 | **`CATEGORIAS_MANTIDAS`** (8 níveis de `property_type`) | em memória, verificado | O teste tem 2.003 linhas: categorias com ≥1% lá podem diferir das do treino → níveis distintos → **one-hot/ordinal desalinhado** |
| Etapa 06 | `medianas_mcar` (4 colunas) | `ARTEFATOS_06` | imputa com a mediana do teste → vazamento |
| Etapa 06 | `medianas_mar` (5 colunas) | `ARTEFATOS_06` | idem |
| Etapa 06 | `mapas_ordinal` (2 categóricas) | `ARTEFATOS_06` | códigos trocados entre treino e teste |
| Etapa 06 | `robust_center` / `robust_scale` (25 colunas) | `ARTEFATOS_06` | escala diferente (irrelevante p/ árvore, mas registrar) |
| Etapa 03 | `maximum_nights.clip(upper=365)` | constante fixa | nenhum — é constante, não aprendido |
| Etapa 01 | `LAT_CENTRO` / `LON_CENTRO` | constantes do notebook | nenhum — coordenadas fixas |

**`CATEGORIAS_MANTIDAS` é o item de maior risco**, porque o agrupamento de `property_type`
usa um corte de frequência (`<1% → "Other"`). Recalcular esse corte sobre as 2.003 linhas do
teste produziria um conjunto de níveis diferente — erro silencioso que desalinha o encoding.

### 2.2 Protocolo de reaplicação

A reaplicação segue **exatamente a ordem das células originais**, lendo os parâmetros dos
objetos já em memória:

```
test_raw.csv (2.003 × 96)
  → limpeza Etapa 02 (cél. 19-45): parse monetário, flags f/t, drops de coluna
  → feature engineering Etapa 03 (cél. 46-60): blocos A-H, usando CATEGORIAS_MANTIDAS
  → recorte FEATURES_05 (27 colunas)
  → preparação Etapa 06: imputação por mecanismo + encoding ordinal, via ARTEFATOS_06
  → X_test (2.003 × 27)
```

**Nota sobre linhas removidas:** a Etapa 02 removeu linhas (`market` fora de Amsterdam,
`price == 0`). No teste, a mesma regra se aplica — mas **cada remoção deve ser contada e
justificada**, porque reduzir o hold-out afeta a representatividade da métrica final. Se
mais de ~2% das linhas saírem, isso vai no relatório como ressalva.

### 2.3 Testes de aceitação da reaplicação

Antes de qualquer predição:

- `X_test.shape[1] == 27` e `list(X_test.columns) == FEATURES_06_ARVORE` (ordem idêntica)
- `X_test.isna().sum().sum() == 0`
- Nenhum nível de `room_type` / `neighbourhood_cleansed` / `property_type_agrupado` no teste
  ausente dos mapas do treino — se houver, tratar como categoria desconhecida (código −1),
  **não** criar nível novo
- As medianas usadas na imputação são **idênticas** às de `ARTEFATOS_06` (comparação
  programática, não visual)
- `X_test` tem **0 linhas em comum** com treino/validação por `id`

---

## 3. Retreino com treino + validação

O CLAUDE.md §7 pede o retreino com as duas partições juntas: **17.978 linhas** (14.382 +
3.596), com os `PARAMS_LGBM_TUNED` da Etapa 08.

Racional: os hiperparâmetros foram escolhidos por CV no treino, e `valid_06` já cumpriu seu
papel (confirmar que o ganho transferiu). Treinar no conjunto completo aproveita 25% mais
dados sem reabrir nenhuma decisão.

**Consequência a registrar:** depois deste retreino, **não existe mais conjunto de
validação**. Qualquer ajuste feito olhando o resultado do teste seria vazamento retroativo.
O número do teste é final — é o que vai para a Etapa 10.

---

## 4. Avaliação no hold-out

### 4.1 Métricas

MAPE (principal), MAE e RMSE, em **escala original** após `expm1`.

### 4.2 Comparação obrigatória

| Referência | MAPE esperado |
|---|---|
| MAPE em `valid_06` (Etapa 08) | 22,14% |
| **MAPE no teste** | **a medir** |
| Chão de negócio no teste (mediana `GROUP_COLS`) | ~28% |

A pergunta que esta etapa responde: **o 22,14% se sustenta em dados que nunca participaram
de nenhuma decisão?**

### 4.3 Faixa esperada e gatilhos

| Resultado no teste | Leitura |
|---|---|
| 21,5–23,5% | **Esperado.** Confirma o pipeline |
| 23,5–25% | Degradação moderada — investigar, provável diferença de distribuição |
| > 25% | Degradação séria — suspeitar de divergência na reaplicação (§2) |
| **< 19%** | **Suspeitar de vazamento**, não celebrar |

O teto entre gêmeos exatos (22,67%) e o piso da EDA (20,80%) continuam valendo como
referência física.

### 4.4 MAPE por decil no teste

Repetir o diagnóstico das Etapas 07 e 08 **no hold-out**, porque é o número que o
recomendador vai herdar. A expectativa, pelo padrão já medido: ~40% no D1, ~15% no D5,
~27% no D10.

---

## 5. O problema E09-6 — o viés que o tuning não corrigiu

Este é o item mais importante da etapa, e precisa ser resolvido **antes** do recomendador.

A Etapa 08 mediu o viés por faixa no LightGBM tunado:

| Decil | Faixa € | Erro % mediano |
|---|---|---|
| D1 | 19–75 | **−29,51%** |
| D5 | 116–125 | −0,73% |
| D10 | 252–1.050 | **+25,26%** |

(negativo = o modelo prevê **acima** do real)

O viés é **monotônico e estrutural** — 120 trials de Optuna não o moveram (D1 melhorou
apenas 0,30 p.p.). A causa é a soma de perda L2 com o teto de informação do dataset.

### 5.1 Por que isso corrompe o EV

O recomendador só atua na **subprecificação** (CLAUDE.md §3.3): recomenda subir quando
`pᵢ < Lᵢ`, onde `Lᵢ = ŷᵢ·(1 − MAPE)`.

Como o modelo prevê **sistematicamente acima do real** nos anúncios baratos, `ŷᵢ` fica
inflado no D1 — e anúncios que apenas *são baratos* passam a ser classificados como
subprecificados. O EV calculado sobre essas recomendações é **superestimado**, e o
recomendador aconselharia aumentos que o mercado pode não sustentar.

### 5.2 Diagnóstico obrigatório

Medir, no teste, **quantos subprecificados há por decil**. Se a concentração no D1 for muito
maior que nos demais, o viés está contaminando a recomendação — e o número precisa estar no
relatório, não escondido no agregado.

### 5.3 Tratamento — a decidir com o número na tela

Três rotas, em ordem de preferência:

1. **Piso de preço na recomendação.** Não recomendar para anúncios abaixo de um limiar (ex.:
   p10 ou €75). Simples, defensável, e remove justamente a faixa onde o erro é 40%. Custo:
   abandona a faixa com mais subprecificados aparentes.
2. **Calibração por faixa.** Ajustar a predição por um fator estimado **no treino** (mediana
   do real / mediana do predito, por decil). Corrige o viés sem descartar anúncios, mas
   adiciona um passo de calibração ao pipeline que a Etapa 11 também terá de reproduzir.
3. **Banda assimétrica.** Usar um MAPE maior no piso (`Lᵢ`) para os decis baixos, tornando a
   recomendação mais conservadora onde o erro é maior.

**A escolha fica para a execução**, com o diagnóstico de §5.2 na tela. O que o plano trava é
que **alguma** rota será aplicada e justificada — não seguir adiante com o viés conhecido e
não tratado.

---

## 6. O recomendador (CLAUDE.md §3)

### 6.1 Construção

Para cada anúncio do teste:

```
ŷᵢ   = predição do modelo (preço justo)
Lᵢ   = ŷᵢ · (1 − MAPE_teste)        # piso da banda
Uᵢ   = ŷᵢ · (1 + MAPE_teste)        # teto da banda
```

Zonas (§3.2):

| Condição | Zona | Ação |
|---|---|---|
| `pᵢ < Lᵢ` | **subprecificado** | recomenda subir até `Lᵢ` (ponto conservador) |
| `Lᵢ ≤ pᵢ ≤ Uᵢ` | zona morta | não age — o desvio cabe no erro |
| `pᵢ > Uᵢ` | superprecificado | **apenas sinaliza**, sem ação (§3.3) |

O `MAPE` da banda é o **MAPE real do teste**, não o da validação — é o erro honesto do
modelo.

### 6.2 Entregável por anúncio

Tabela com `id`, `price` atual, `ŷ` (preço justo), banda `[L, U]`, zona, e `uplift = L − p`
para os subprecificados. O `id` foi preservado desde a Etapa 04 exatamente para isso.

### 6.3 O que NÃO entra aqui

O cálculo de EV é a **Etapa 10** (re-execução da Etapa 01 com o MAPE real e as predições do
modelo). Esta etapa entrega o recomendador e a contagem de zonas; a conversão em euros fica
para a próxima, com código inline e sem funções (CLAUDE.md §4 das regras de trabalho).

---

## 7. Ordem de execução — 8 portões

1. **Reaplicação da Etapa 02 ao teste** — limpeza, com contagem de linhas removidas
2. **Reaplicação da Etapa 03 ao teste** — features, usando `CATEGORIAS_MANTIDAS`
3. **Recorte + preparação Etapa 06** — 27 colunas via `ARTEFATOS_06`, com os testes de
   aceitação de §2.3
4. **Retreino** do `MODELO_FINAL` com treino + validação (17.978 linhas)
5. **Avaliação no hold-out** — MAPE/MAE/RMSE + comparação com `valid_06` e com o chão de
   negócio
6. **Diagnóstico E09-6** — MAPE por decil e subprecificados por decil no teste
7. **Tratamento do viés** — aplicar e justificar uma das rotas de §5.3
8. **Recomendador** — bandas, zonas, tabela por anúncio, consolidação

Cada item é uma pausa para validação, conforme a Regra de Trabalho nº 2. São **8 portões**,
não 1.

---

## 8. Entregáveis

1. **`X_test`** (2.003 × 27) — o hold-out processado, com o relatório de linhas removidas
2. **`MODELO_FINAL_RETREINADO`** — LightGBM com `PARAMS_LGBM_TUNED` sobre 17.978 linhas
3. **`METRICAS_TESTE`** — MAPE, MAE, RMSE no hold-out, comparados com `valid_06`
4. **`DECIS_TESTE`** — MAPE e viés por faixa de preço no teste
5. **`RECOMENDADOR`** — tabela por anúncio: `id`, `price`, `preco_justo`, `L`, `U`, `zona`,
   `uplift`
6. **Contagem de zonas** — subprecificados / zona morta / superprecificados, total e por
   decil
7. **Decisão sobre E09-6** — rota escolhida, com o número que a justificou
8. **`PREMISSAS_09`** — MAPE usado na banda, regra de tratamento do viés, linhas removidas

---

## 9. Verificação

- **Hold-out aberto uma única vez:** nenhuma célula anterior à Etapa 09 lê `test_raw.csv` —
  busca textual no notebook
- **Sem `fit` no teste:** toda imputação, encoder e escala usa parâmetros de `ARTEFATOS_06`;
  nenhum `fit_transform` sobre `X_test`
- **`CATEGORIAS_MANTIDAS` reutilizada**, não recalculada — `assert` explícito
- **Alinhamento de colunas:** `list(X_test.columns) == FEATURES_06_ARVORE`
- **Sem nulos:** `X_test.isna().sum().sum() == 0`
- **Sem sobreposição:** `set(test.id) & set(train_06.id) == ∅`
- **MAPE em escala original:** `assert 0 < mape < 1`
- **Gatilho de vazamento:** `mape_teste > 0.19`; abaixo disso, parar e investigar
- **Zonas exaustivas e exclusivas:** cada anúncio em exatamente uma zona; soma = total
- **Só subprecificados geram uplift:** `uplift > 0` apenas na zona subprecificada (§3.3)

---

## 10. Pendências

**Resolvidas antes da execução:**

1. O hold-out foi **verificado apenas em dimensão** (2.003 × 96, `id` único) para dimensionar
   este plano. Nenhuma estatística do alvo foi inspecionada — o teste segue fechado até o
   portão 1.
2. **`CATEGORIAS_MANTIDAS` está em memória** (8 níveis, verificado) e será reutilizada, não
   recalculada.

**Em aberto, a decidir com o resultado na tela:**

1. **Rota de tratamento de E09-6** (§5.3) — piso de preço, calibração por faixa ou banda
   assimétrica. Depende do diagnóstico do portão 6.
2. **Se o MAPE do teste degradar acima de 25%** — a hipótese principal passa a ser
   divergência na reaplicação (§2), e o portão 3 precisa ser reauditado antes de aceitar o
   número.
3. **Se mais de ~2% das linhas do teste forem removidas** na limpeza, registrar como
   ressalva de representatividade.
4. **XGBoost não tunado** (herdado da Etapa 08) — se o resultado do teste decepcionar, tunar
   o XGBoost é a alavanca não explorada. Não é escopo desta etapa.
