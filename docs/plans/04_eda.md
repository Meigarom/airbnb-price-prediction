# Plano — Etapa 04: Análise Exploratória de Dados (Airbnb Amsterdam)

## Contexto

Entramos nesta etapa com `df03`: **18.020 linhas × 83 colunas**, granularidade de 1 linha
por `listing_id`, resultado da Etapa 03 (Feature Engineering). Dessas 83 colunas, 46 são
features novas criadas na etapa anterior.

> **Nota de revisão (pendências resolvidas antes da execução).** Uma inspeção prévia sobre
> a versão anterior de `df03` (18.022 × 86) revelou quatro achados. Três deles eram fatos,
> não hipóteses, e foram corrigidos **na origem** antes de iniciar a EDA:
>
> | Achado | Ação tomada | Onde |
> |---|---|---|
> | `price == 0` em 2 anúncios | 2 linhas removidas | Etapa 02, célula nova |
> | `host_experiencia_anos` (ρ = 1,0 com `host_experiencia_dias`) | Não é mais criada | Etapa 03, Bloco C |
> | `host_portfolio_size` (ρ = 1,0 com `calculated_host_listings_count`) | Não é mais criada | Etapa 03, Bloco D |
> | `dist_estacao_central_km` (ρ = 0,958 com `dist_centro_km`, VIF 44,3) | Não é mais criada | Etapa 03, Bloco A |
>
> As três colunas foram eliminadas no ponto de criação, e não por um drop posterior: um
> drop na Etapa 02 não funcionaria, porque as colunas só passam a existir na Etapa 03.
> O quarto achado (assimetria do alvo) permanece como análise, na Seção 3.2.
>
> Efeito: 18.022 × 86 → **18.020 × 83**. O VIF do par geográfico remanescente caiu de
> 74,6 / 44,3 / 9,8 para **2,66**.

**Isolamento do teste** (regra travada em CLAUDE.md, Seção 5.3): os 10% separados na
Etapa 02 (`data/processed/test_raw.csv`, 2.003 linhas) **não entram nesta EDA**. Toda
análise é feita sobre o conjunto de treino. Olhar o teste aqui contaminaria as decisões de
seleção e preparação das Etapas 05 e 06 — é vazamento por via analítica, mais sutil que o
vazamento de coluna, e igualmente fatal.

O objetivo do projeto (CLAUDE.md, Seção 1) é prever a diária (`price`) minimizando
**MAPE**. Isso dá à EDA uma orientação específica que atravessa todo este plano:

> O MAPE é um **erro percentual**. Errar €20 num anúncio de €60 custa 33% de MAPE;
> errar os mesmos €20 num anúncio de €400 custa 5%. Portanto, a EDA precisa dar atenção
> desproporcional à **cauda inferior de preços** — é lá que o modelo mais pode destruir
> a métrica, e é lá que está a massa de anúncios acionáveis do EV da Etapa 01.

---

## 1. O problema das 86 colunas — estratégia de agrupamento

Analisar 86 colunas numa sequência linear de gráficos é ilegível: são ~86 histogramas,
~86 scatterplots e uma matriz de correlação 86×86 onde nada é distinguível.

**Solução adotada:** todas as análises são organizadas em **7 grupos temáticos**, fechados
e mutuamente exclusivos. Cada grupo é uma subseção do notebook com no máximo ~15 colunas,
executada e validada de forma independente, e fechada por uma célula de markdown com as
conclusões daquele grupo.

A regra de leitura é: **você avalia um grupo por vez, não o dataset inteiro**.

### 1.1 Os 7 grupos

| # | Grupo | Nº cols | Conteúdo |
|---|---|---|---|
| **G0** | **Alvo** | 1 | `price` |
| **G1** | **Capacidade e estrutura física** | 12 | `accommodates`, `bathrooms`, `bedrooms`, `beds`, `guests_included`, `pessoas_por_quarto`, `pessoas_por_cama`, `banheiros_por_pessoa`, `camas_por_quarto`, `tem_multiplos_banheiros`, `room_type`, `property_type_agrupado` |
| **G2** | **Geografia** | 6 | `latitude`, `longitude`, `dist_centro_km`, `dist_museumplein_km`, `dist_centro_bin`, `neighbourhood_cleansed` |
| **G3** | **Amenities e qualidade do imóvel** | 13 | `n_amenities`, `n_amenities_luxo`, e as 11 flags `tem_*` de amenities |
| **G4** | **Host e operação** | 10 | `host_is_superhost`, `host_identity_verified`, `host_response_time`, `host_listings_count`, `calculated_host_listings_count`, `host_eh_profissional`, `host_eh_local`, `host_tem_id_verificado`, `n_host_verifications`, `host_experiencia_dias` |
| **G5** | **Política de reserva e custos** | 16 | `minimum_nights`, `maximum_nights`, `maximum_nights_cap`, `sem_limite_maximo`, `min_nights_bin`, `permite_estadia_curta`, `cancellation_policy`, `instant_bookable`, `is_location_exact`, `cleaning_fee`, `taxa_limpeza_por_noite`, `tem_taxa_limpeza`, `security_deposit`, `tem_deposito`, `extra_people`, `calendar_atualizado_dias` |
| **G6** | **Reviews e reputação** | 17 | as 7 `review_scores_*`, `review_score_medio`, `gap_location_vs_geral`, `gap_value_vs_geral`, `number_of_reviews`, `reviews_per_month`, `reviews_por_mes_ativo`, `sem_reviews`, `dias_desde_primeira_review`, `dias_desde_ultima_review`, `anuncio_ativo` |
| **G7** | **Disponibilidade e texto** | 7 | `availability_30/60/90/365`, `len_description`, `len_house_rules`, `n_campos_texto_preenchidos` |

Total: 1 (alvo) + 12 + 6 + 13 + 10 + 16 + 17 + 7 = **82** colunas, sendo **81 explicativas**
+ o alvo `price` (G0), + `id` (excluída de toda análise) = 83. ✅

> **Verificado na execução (célula 62 do notebook):** a checagem programática confirma
> **81 colunas explicativas**. O número 82 acima inclui o alvo `price`.

**Por que estes recortes:** cada grupo corresponde a uma **pergunta de negócio distinta**
("o imóvel é grande?", "fica bem localizado?", "é bem equipado?", "quem opera é
profissional?", "quais as regras?", "tem boa reputação?"). Isso faz com que as conclusões
de cada bloco sejam diretamente traduzíveis para o recomendador da Etapa 09.

### 1.2 Convenção de tipo dentro de cada grupo

Dentro de cada grupo, as colunas se dividem em três tratamentos visuais distintos:

- **Contínuas** (52 no total) → histograma + boxplot
- **Binárias 0/1** (25 no total) → barra de frequência + boxplot de `price` por classe
- **Categóricas** (7 no total) → barra de frequência + boxplot de `price` por categoria

**Limite de leitura:** no máximo **4 painéis por célula**. É o mesmo padrão já adotado na
Etapa 02 com os boxplots categóricos, que funcionou bem.

---

## 2. Achados prévios — o que já foi resolvido e o que resta

Uma inspeção rápida sobre a versão anterior de `df03` revelou quatro questões concretas.
Três já foram corrigidas na origem (ver nota de revisão no Contexto); ficam registradas
aqui porque o racional continua valendo e orienta o que a EDA ainda precisa verificar.

### 2.1 `price == 0` em 2 anúncios — **resolvido**

`price.min()` era 0, com 2 registros. Isto era crítico:

> O MAPE é `|y - ŷ| / |y|`. Com `y = 0`, o termo é **divisão por zero** → `inf`. Dois
> registros bastam para tornar a métrica-objetivo do projeto inteiro incalculável.

Os 2 registros foram removidos na Etapa 02. `price` mínimo agora é **€ 8,00**.

**O que resta para a Análise 1.1:** há 6 registros com `price < 20` (abaixo de qualquer
custo operacional plausível em Amsterdam) e 22 com `price > 1000`. O tratamento desses
extremos (piso em €20? winsorize em p1/p99?) é decisão a tomar com o gráfico na tela —
é a pendência nº 1 da Seção 9.

### 2.2 Duas duplicatas exatas (ρ = 1,0000) — **resolvido**

| Par | ρ Pearson | Natureza | Ação |
|---|---|---|---|
| `host_experiencia_dias` × `host_experiencia_anos` | **1,0000** | Mesma variável, escala diferente (`/365,25`) | `_anos` não é mais criada |
| `calculated_host_listings_count` × `host_portfolio_size` | **1,0000** | Cópia literal, criada no Bloco D | `host_portfolio_size` não é mais criada |

Eram redundâncias perfeitas — zero informação adicional. Em ambos os casos foi mantida a
coluna de origem, e a derivada deixou de ser criada no Bloco C / Bloco D da Etapa 03.

### 2.3 Multicolinearidade forte (ρ > 0,80)

| Par | ρ Pearson | Situação |
|---|---|---|
| ~~`dist_centro_km` × `dist_estacao_central_km`~~ | ~~0,9577~~ | **Resolvido** — ver 2.3.1 |
| `maximum_nights` × `maximum_nights_cap` | 0,9581 | Aberto — Análise 3.2 |
| `reviews_per_month` × `reviews_por_mes_ativo` | 0,8922 | Aberto — Análise 3.2 |
| `host_listings_count` × `calculated_host_listings_count` | 0,8743 | Aberto — Análise 3.2 |
| `review_scores_rating` × `review_score_medio` | 0,8239 | Aberto — Análise 3.2 |
| `availability_30` × `availability_90` | 0,8061 | Aberto — Análise 3.2 |

Os cinco pares abertos seguem para a Análise 3.2, onde a decisão de descarte é tomada com
o ranking de associação com o alvo já em mãos.

#### 2.3.1 As 3 distâncias geográficas — **resolvido, com teste**

A hipótese de "múltiplos polos" do plano da Etapa 03 (Bloco A) **não se sustentou**: os
três pontos de referência são vizinhos entre si (Dam ↔ Centraal = 0,86 km;
Dam ↔ Museumplein = 1,84 km), num raio urbano onde os anúncios chegam a 11 km.

**VIF antes:** `dist_centro_km` 74,6 · `dist_estacao_central_km` 44,3 ·
`dist_museumplein_km` 9,8 — duas delas muito acima do limiar de alarme (VIF > 10).

**Poder preditivo** (R², RandomForest CV=3, alvo `log1p(price)`):

| Conjunto | Colunas | R² |
|---|---|---|
| só `dist_centro` | 1 | 0,055 |
| centro + centraal | 2 | 0,074 |
| **centro + museumplein** | **2** | **0,087** |
| as três | 3 | 0,093 |
| `latitude` + `longitude` puros | 2 | **0,112** |
| lat/lon + as três | 5 | 0,110 |

**Decisão:** mantidas `dist_centro_km` + `dist_museumplein_km`; `dist_estacao_central_km`
deixou de ser criada. Centraal era a mais redundante (a 0,86 km do Dam, VIF 44,3), e
trocá-la por Museumplein no par **melhora** o R² (0,074 → 0,087). A terceira distância
agregava +0,006 de R² ao custo de multicolinearidade severa. O VIF do par remanescente é
**2,66**.

> **Achado a registrar para a Etapa 05:** `latitude` + `longitude` sozinhas (R² = 0,112)
> **superam as três distâncias juntas** (0,093), e acrescentar as distâncias ao lat/lon
> piora ligeiramente (0,110). A árvore extrai da geometria bruta mais do que qualquer
> distância radial — faz sentido, porque Amsterdam não é radial: o IJ corta a cidade, os
> canais formam anéis, e bairros equidistantes do Dam têm valores muito diferentes.
>
> As distâncias foram mantidas mesmo assim por serem **interpretáveis para o negócio** —
> "a 1,2 km do centro" é comunicável ao anfitrião no recomendador da Etapa 09, "latitude
> 52,37" não é. O Boruta (Etapa 05) é o portão formal dessa decisão.

### 2.4 Assimetria extrema do alvo

`price.skew() = 22,68`. Após `log1p`: **0,44** — praticamente simétrico.

Isto antecipa a decisão mais importante da Etapa 06/07: **treinar em `log(price)`**.
A EDA precisa produzir a evidência visual dessa escolha (Análise 1.2), não apenas o número.

---

## 3. Análise Univariada

**Objetivo:** entender cada variável isoladamente — distribuição, faltantes, outliers,
variabilidade. Responde: *"esta coluna tem informação aproveitável?"*

### 3.1 Análise 1.1 — O alvo `price` (G0)

Executada **primeiro e sozinha**, porque condiciona todo o resto.

| Tarefa | Saída esperada |
|---|---|
| `describe()` completo + percentis p1, p5, p95, p99 | Tabela |
| Histograma de `price` (escala linear) | Gráfico — evidencia a cauda |
| Boxplot de `price` | Gráfico — magnitude dos outliers |
| Contagem de `price == 0`, `< 20`, `> 500`, `> 1000` | Tabela de corte |
| **Decisão documentada** sobre remoção/winsorize | Markdown + célula de código |

### 3.2 Análise 1.2 — Transformação do alvo

| Tarefa | Saída esperada |
|---|---|
| Histograma de `log1p(price)` lado a lado com `price` | 2 painéis |
| Q-Q plot de `price` vs. `log1p(price)` | 2 painéis |
| `skew()` e `kurtosis()` antes/depois | Tabela |
| Teste de normalidade (D'Agostino/Shapiro em amostra) | p-valor |

**Entregável:** recomendação explícita e justificada sobre treinar em `log(price)`,
registrada como premissa para a Etapa 07. Se adotada, lembrar que o MAPE deve ser
calculado **na escala original** (após `expm1`), nunca na escala log.

### 3.3 Análise 1.3 a 1.9 — Distribuição por grupo (G1 a G7)

Uma subseção por grupo, cada uma seguindo o mesmo roteiro fixo:

1. **Tabela descritiva do grupo** — `describe().T` das contínuas + `value_counts(normalize=True)` das categóricas/binárias, com `% nulos` e `nunique` em coluna.
2. **Painéis de distribuição** — histograma + boxplot para contínuas; barra de frequência para binárias/categóricas. Máximo 4 painéis por célula.
3. **Célula de markdown "Observações do grupo Gx"** — 3 a 5 bullets, em português, apontando: colunas quase-constantes, caudas pesadas, nulos relevantes, candidatas a descarte.

**Critérios objetivos de sinalização** (aplicados uniformemente aos 7 grupos):

| Critério | Limiar | Ação sugerida |
|---|---|---|
| Quase-constância | >95% numa única categoria/valor | Candidata a descarte |
| Nulos altos | >30% | Decidir imputação vs. descarte na Etapa 06 |
| Cauda extrema | p99 > 10× mediana | Candidata a clip/log na Etapa 06 |
| Variância nula | `nunique() == 1` | Descarte imediato |

**Pontos de atenção já conhecidos, por grupo:**

- **G1:** `pessoas_por_quarto` e `camas_por_quarto` têm 5,6%/5,7% de nulos — vêm da proteção contra divisão por zero (`bedrooms == 0`, ou seja, estúdios). Confirmar que a interpretação "estúdio" está correta e decidir se vira flag `eh_estudio` em vez de `NaN`.
- **G2:** verificar se `latitude`/`longitude` têm pontos fora da bounding box de Amsterdam.
- **G3:** `tem_elevator` (9,7%) e `tem_step_free_access` (6,3%) ficaram abaixo do piso de 10% do plano da Etapa 03 e foram mantidas por argumento de negócio. Este é o momento de olhar a distribuição e confirmar ou reverter.
- **G5:** `security_deposit` tem 30,9% de nulos — o maior do dataset após a limpeza. `taxa_limpeza_por_noite` tem 18,1%.
- **G6:** as 7 `review_scores_*` compartilham o mesmo padrão de ~13,2% de nulos (anúncios sem review). Confirmar que o padrão de nulos é idêntico entre elas — se for, um único indicador `sem_reviews` já captura toda a informação de ausência.
- **G7:** `availability_365` — reforçar em markdown que **não é ocupação** (CLAUDE.md Seção 4). Risco de uso indevido nas etapas seguintes.

---

## 4. Análise Bivariada

**Objetivo:** relação de cada explicativa com o alvo `price`. Responde: *"esta coluna
ajuda a prever preço?"*

**Nota metodológica:** todas as análises bivariadas usam **`price` na escala definida na
Análise 1.2** (provavelmente `log1p`). Correlações medidas na escala original são
dominadas pelos 22 outliers acima de €1.000 e não refletem a relação na faixa central —
justamente a faixa que importa para o MAPE.

### 4.1 Análise 2.1 — Ranking geral de associação com o alvo

Executada primeiro, para dar o mapa antes do detalhe.

| Tarefa | Saída esperada |
|---|---|
| Spearman de todas as contínuas vs. `price`, ordenado | Tabela + gráfico de barras horizontais |
| Pearson (em `log1p(price)`) das mesmas, lado a lado | Coluna adicional na tabela |
| **Divergência Pearson vs. Spearman** | Coluna `delta` — divergência alta sinaliza relação não-linear |
| Para binárias: diferença de mediana de `price` entre classes 0 e 1 + Mann-Whitney | Tabela ordenada por efeito |
| Para categóricas: razão de correlação (η²) ou Kruskal-Wallis | Tabela ordenada |

**Entregável:** um ranking único de todas as 81 explicativas por força de associação com
o alvo — o insumo direto da Etapa 05 (Feature Selection) e a primeira visão consolidada
que caberá numa tela só.

### 4.2 Análise 2.2 a 2.8 — Detalhe por grupo (G1 a G7)

Uma subseção por grupo, com roteiro fixo:

1. **Contínuas** → scatterplot vs. `price` com linha de tendência (LOWESS), máx. 4 por célula. Onde houver sobreposição densa (18k pontos), usar `hexbin` ou amostragem.
2. **Binárias** → boxplot de `price` por classe, 2 por célula (padrão da Etapa 02).
3. **Categóricas** → boxplot de `price` por categoria, ordenado por mediana, com contagem por categoria anotada.
4. **Markdown "Observações bivariadas do grupo Gx"** — força da relação, forma (linear/não-linear/degrau), presença de interação aparente.

**Hipóteses de negócio a testar explicitamente** (cada uma vira um bullet respondido):

| # | Hipótese | Grupo |
|---|---|---|
| H1 | Preço cai monotonicamente com a distância ao centro | G2 |
| H2 | `room_type = Entire home/apt` tem preço mediano significativamente maior que `Private room` | G1 |
| H3 | Superhosts cobram mais que não-superhosts, controlando por capacidade | G4 |
| H4 | Anfitriões profissionais (`host_eh_profissional`) precificam de forma mais homogênea (menor variância) | G4 |
| H5 | `gap_value_vs_geral` negativo associa-se a preços mais altos (hóspede achou caro) | G6 |
| H6 | `n_amenities` tem relação positiva mas com retorno decrescente | G3 |
| H7 | `Boat` tem dinâmica de preço própria, distinta de `Apartment` | G1 |
| H8 | Anúncios sem review (`sem_reviews`) têm distribuição de preço distinta | G6 |

**H5 merece destaque:** é a hipótese diretamente ligada ao objetivo do projeto. Se
confirmada, `gap_value_vs_geral` vira candidata forte tanto a feature quanto a sinal de
validação do recomendador na Etapa 09.

### 4.3 Análise 2.9 — Erro percentual por faixa de preço (análise dedicada ao MAPE)

Análise específica que não existe numa EDA genérica, mas é essencial aqui.

| Tarefa | Saída esperada |
|---|---|
| Distribuição de anúncios por decil de `price` | Tabela + gráfico |
| Para cada decil: mediana, IQR e **coeficiente de variação** de `price` | Tabela |
| Identificar em quais faixas a dispersão intra-grupo é maior | Markdown |

**Racional:** o CV por faixa é uma estimativa do **piso de MAPE alcançável** em cada
faixa. Se a faixa de €50–80 tem CV de 40%, nenhum modelo vai alcançar MAPE de 10% ali com
as features disponíveis. Este número calibra a expectativa da Etapa 07 e alimenta os
cenários (otimista/base/pessimista) que a Etapa 10 vai reconciliar com a Etapa 01.

---

## 5. Análise Multivariada

**Objetivo:** relação das explicativas **entre si** — detectar autocorrelação e
redundância. Responde: *"quais colunas dizem a mesma coisa?"*

### 5.1 Análise 3.1 — Correlação intra-grupo

Uma matriz de correlação **por grupo** (7 matrizes de no máximo 17×17), em vez de uma
matriz 81×81 ilegível.

| Tarefa | Saída esperada |
|---|---|
| Heatmap de Spearman por grupo, com anotação numérica | 7 heatmaps legíveis |
| Máscara triangular superior | Evita leitura duplicada |
| Markdown por grupo apontando pares com \|ρ\| > 0,80 | Lista de candidatos a descarte |

**Racional do recorte por grupo:** variáveis do mesmo grupo temático são, por construção,
as mais propensas a redundância (7 notas de review, 4 janelas de disponibilidade, as
distâncias geográficas). É lá que a multicolinearidade se concentra, e é lá que a matriz é
legível.

### 5.2 Análise 3.2 — Correlação inter-grupo e consolidação

| Tarefa | Saída esperada |
|---|---|
| Lista única de **todos** os pares com \|ρ\| > 0,80 no dataset inteiro (sem plotar a matriz cheia) | Tabela ordenada por \|ρ\| |
| Clusterização hierárquica das features por distância `1 - |ρ|`, com dendrograma | 1 gráfico |
| **VIF** (Variance Inflation Factor) das contínuas | Tabela ordenada |
| Heatmap consolidado apenas das **top-25 features** do ranking da Análise 2.1 | 1 heatmap legível |

**Entregável central:** uma **lista de descarte por redundância**, com o par, o ρ e qual
das duas manter — e o porquê (preferir a de maior correlação com o alvo, ou a de
interpretação de negócio mais direta).

**Casos já conhecidos a resolver aqui:** os cinco pares em aberto da Seção 2.3 (ρ > 0,80).
Os casos de ρ = 1,0 (Seção 2.2) e o par geográfico (Seção 2.3.1) já foram resolvidos na
origem, antes da execução desta etapa.

**Nota sobre o dendrograma:** é a forma mais compacta de enxergar redundância em 81
features — clusters apertados são grupos de colunas dizendo a mesma coisa. Cabe numa
imagem e responde a pergunta "o que é redundante?" melhor que qualquer matriz.

### 5.3 Análise 3.3 — Interações e efeitos condicionais

| Tarefa | Saída esperada |
|---|---|
| `price` mediano por `neighbourhood_cleansed` × `room_type` | Heatmap (22×3) |
| `price` mediano por `dist_centro_bin` × `accommodates` | Heatmap |
| `price` mediano por `property_type_agrupado` × `room_type` | Heatmap |
| Mapa geográfico (scatter lat/lon colorido por `price` mediano) | 1 gráfico |

**Racional:** o preço justo do recomendador (Etapa 09) é, na prática, uma função de
`(bairro, tipo de quarto, capacidade)` — exatamente o `GROUP_COLS` da metodologia travada
da Etapa 01 (CLAUDE.md, Seção 3.1). Estes heatmaps são a validação visual de que aquele
agrupamento é o correto, e mostram onde há células com poucos anúncios (onde o fallback
hierárquico da Etapa 01 entra em ação).

### 5.4 Análise 3.4 — Padrão de dados faltantes

| Tarefa | Saída esperada |
|---|---|
| Matriz de nulos (`missingno.matrix` ou heatmap booleano) | 1 gráfico |
| Correlação entre indicadores de nulidade | 1 heatmap |
| Tabela: coluna, % nulo, **hipótese de mecanismo** (MCAR/MAR/MNAR) | Tabela |

**Racional:** nulos correlacionados indicam causa comum. As 7 `review_scores_*` com
~13,2% devem ser o mesmo conjunto de anúncios (os sem review) — se confirmado, é
**MAR explicável**, e a decisão de imputação da Etapa 06 é direta. Já `security_deposit`
(30,9%) provavelmente é **MNAR** ("não cobro depósito" ≠ "não informei"), e imputar pela
mediana ali seria um erro.

---

## 6. Ordem de execução

1. **Análise 1.1** — alvo `price` + decisão sobre `price == 0` (**bloqueador — primeiro de tudo**)
2. **Análise 1.2** — transformação log do alvo
3. **Análises 1.3 a 1.9** — univariada por grupo (G1 → G7)
4. **Análise 2.1** — ranking geral de associação com o alvo
5. **Análises 2.2 a 2.8** — bivariada por grupo (G1 → G7)
6. **Análise 2.9** — dispersão por faixa de preço (piso de MAPE)
7. **Análise 3.1** — correlação intra-grupo
8. **Análise 3.2** — redundância consolidada + VIF + dendrograma
9. **Análise 3.3** — interações e mapa geográfico
10. **Análise 3.4** — padrão de faltantes
11. **Consolidação** — relatório de conclusões e lista de recomendações para a Etapa 05

Cada item é uma pausa natural para validação, conforme a Regra de Trabalho nº 2 do
CLAUDE.md. São **11 portões**, não 1.

---

## 7. Entregáveis

Ao fim da etapa, quatro artefatos concretos:

1. **`df04`** — o dataset pós-EDA. Difere de `df03` apenas pelas remoções **justificadas
   e documentadas** que a própria EDA venha a motivar. A EDA não é etapa de transformação:
   a seleção de features é a Etapa 05, a preparação é a Etapa 06.
2. **Ranking de associação com o alvo** — as 81 explicativas ordenadas, insumo direto do
   Boruta na Etapa 05.
3. **Lista de redundâncias** — pares com \|ρ\| > 0,80 e qual manter, com justificativa.
4. **Registro de premissas atualizado** — decisão sobre escala do alvo, tratamento de
   outliers de `price`, e estimativa de piso de MAPE por faixa.

---

## 8. Verificação

- **Granularidade:** `len(df04) == df04['id'].nunique()` permanece verdadeiro
- **Isolamento do teste:** nenhuma célula desta etapa lê `test_raw.csv` — verificável por busca no notebook
- **Rastreabilidade de linhas:** toda remoção de linha tem contagem antes/depois impressa e justificativa em markdown
- **Sem `price == 0`:** `(df04['price'] == 0).sum() == 0` — pré-condição para o MAPE ser calculável (já garantido na Etapa 02; reconfirmar)
- **Cobertura:** as 81 colunas explicativas aparecem em exatamente um grupo (soma = 81 + o alvo em G0 = 82, sem interseção) — checagem programática, não visual
- **Legibilidade:** nenhuma célula gera mais de 4 painéis; nenhum heatmap excede 25×25

---

## 9. Pendências para validação

**Resolvidas antes da execução** (ver nota de revisão no Contexto): `price == 0`,
as duplicatas exatas de ρ = 1,0 e as 3 distâncias geográficas (reduzidas a 2).

**Em aberto:**

1. **Tratamento dos demais outliers de `price`** — os 2 registros com `price == 0` já
   saíram. Resta decidir o que fazer com os 6 registros abaixo de €20 e os 22 acima de
   €1.000: deixar como estão, aplicar piso/teto, ou winsorizar em p1/p99 como já é feito
   na Etapa 01? Há um argumento para alinhar com a Etapa 01 — mas isso remove do treino
   justamente os anúncios de luxo, um segmento real do mercado. Decisão da Análise 1.1,
   com o gráfico na tela.
2. **Escala do alvo** — a Análise 1.2 vai produzir a evidência, mas a decisão de treinar
   em `log(price)` tem consequência para a Etapa 07 inteira e merece validação explícita.
3. **Profundidade da Análise 3.3** — os heatmaps de interação podem se estender
   bastante (qualquer par de categóricas é um heatmap possível). O plano propõe 3 fixos,
   escolhidos por aderência ao `GROUP_COLS` da Etapa 01.
