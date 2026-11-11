# Etapa 02 — Data Quality (Airbnb Amsterdam)

## Contexto

A Etapa 01 (retorno financeiro / EV(MAPE)) já está implementada em
`notebooks/price-prediction.ipynb`. Antes de avançar para Feature Engineering (Etapa 03)
e EDA (Etapa 04), o CLAUDE.md exige uma Etapa 02 de Data Quality: entender tipos,
faltantes, ranges, duplicatas e consistência de cada fonte de dado, limpar `price` e as
"pegadinhas" já conhecidas (`review_scores_rating` em escala 0-100, `availability_365`
não é ocupação), e documentar cada decisão de limpeza.

**Decisão de modelagem:** o problema de ML é tratado como uma regressão estática — uma
**foto** dos anúncios do Airbnb, usada para avaliar subprecificação/superprecificação —
e **não** incorpora a dimensão temporal. Por isso, `calendar.csv` (granularidade
listing × dia, com ~365 linhas por anúncio) **fica fora do escopo desta etapa e de todo
o projeto**: não será carregado, diagnosticado, limpo, agregado nem unido às demais
tabelas. Isso também elimina a necessidade de qualquer "foto temporal futura"
(disponibilidade/preço futuro) como feature.

A tabela final resultante da Etapa 02 deve ter granularidade **1 linha por
`listing_id`** — o mesmo nível de `listings_details.csv` — pronta para Feature
Engineering (Etapa 03) e EDA (Etapa 04). Um levantamento read-only já foi feito nos
demais CSVs (ver Seção 1) para embasar as decisões abaixo.

> **Estado deste documento.** O plano foi executado e depois revisado. Duas mudanças
> importantes entraram depois da redação original: a separação do **hold-out de teste
> (10%)** logo após a carga dos dados — decisão estratégica, travada na **Seção 5 do
> CLAUDE.md** e apenas referenciada aqui (Seção 2.7) — e a **remoção de linhas inválidas**
> (`maximum_nights = 9999` e `price == 0`, Seção 2.8). O dataset de saída da etapa é
> **18.020 × 83**.

---

## 1. Estrutura de dados levantada (resumo)

| Arquivo | Linhas × Colunas | Granularidade | Chave p/ listing | Papel no projeto |
|---|---|---|---|---|
| `listings_details.csv` | 20.030 × 96 | 1 anúncio | `id` (PK) | **Base principal** — única fonte de features |
| `listings.csv` | 20.030 × 16 | 1 anúncio | `id` | Só validação cruzada (price, neighbourhood) — não entra no merge |
| `reviews.csv` | 431.830 × 2 | 1 review (resumo) | `listing_id` | Não usar — subconjunto de reviews_details.csv |
| `reviews_details.csv` | 431.830 × 6 | 1 review | `listing_id` | Agregar → 1 linha/listing, depois merge |
| `neighbourhoods.csv` | 22 × 2 | 1 bairro | via string `neighbourhood` | Só validação cruzada dos 22 bairros — não entra no merge |

`calendar.csv` (7.310.950 × 4, listing × dia) **excluído do projeto** por decisão de
modelagem estática (sem variável temporal) — não consta mais desta tabela nem de
nenhuma etapa seguinte do plano.

Achados-chave que orientam as decisões de limpeza:
- `id` de `listings_details.csv` é PK confirmada (20.030 únicos, sem duplicatas), e é o
  mesmo universo de `id` em `listings.csv`.
- `price` (e demais colunas monetárias) vêm como string `"$150.00"` — herança de schema,
  valores em euro (regra CLAUDE.md Seção 1).
- `reviews.csv` é subconjunto exato de `reviews_details.csv` (mesma info, sem
  reviewer/texto) — só `reviews_details.csv` será usado.
- `neighbourhood_cleansed` (22 valores, 0% nulo) é a coluna de bairro confiável;
  `neighbourhood` (cru, 44 valores, 8,25% nulo) e `neighbourhood_group_cleansed`
  (100% nulo) serão descartadas.
- `review_scores_rating`: confirmado 0-100 (não 0-5) — manter unidade original, apenas
  documentar.
- `availability_365`: confirmado ser disponibilidade **futura** no calendário no momento
  do scrape, não ocupação passada. Independente da exclusão de `calendar.csv`, essa
  coluna continua presente em `listings_details.csv` — a "pegadinha" é documentada e a
  coluna **nunca é usada como proxy de ocupação** (reforça a regra já travada na Seção 4
  do CLAUDE.md, que usa o modelo InsideAirbnb via `reviews_per_month`, não o calendário).
- `market`: 19.981 "Amsterdam" + 49 registros divergentes (42 NaN, 5 "Other
  (International)", 1 "Barcelona", 1 "Vilnius").

---

## 2. Decisões fechadas

1. **`calendar.csv` fora do escopo do projeto.** Modelagem é uma regressão estática
   (foto dos anúncios), sem dimensão temporal — a planilha não é carregada em nenhuma
   etapa. Nenhuma feature de disponibilidade/preço futuro será criada.
2. **Registros fora de `market == 'Amsterdam'` (49 linhas):** inspecionar
   `neighbourhood_cleansed`/`latitude`/`longitude` desses registros antes de decidir.
   Dropar apenas os que caírem fora da bounding box geográfica de Amsterdam; os que
   caírem dentro são ruído de cadastro no campo `market` e são mantidos.
3. **`security_deposit` / `cleaning_fee` nulos (18-31%):** manter como `NaN` após o
   parse de string→float. Não assumir a hipótese "NaN = sem cobrança" nesta etapa —
   decisão de imputação (se houver) fica para Etapa 03/06.
4. **Persistência do dataset limpo:** manter apenas em memória (`df_clean` ou nome
   equivalente) no mesmo notebook `price-prediction.ipynb` — a Etapa 03 continua nele,
   sem necessidade de exportar para disco.
5. **Agregação de `reviews_details.csv`:** aprovada conforme proposta na Seção 3.1
   abaixo (contagem e datas de review), com checagem cruzada contra as colunas
   equivalentes já existentes em `listings_details.csv` (`number_of_reviews`,
   `first_review`, `last_review`).
6. **Granularidade final obrigatória:** o dataset resultante da Etapa 02 deve ter
   **exatamente 1 linha por `listing_id`** (mesmo universo de 20.030 anúncios, salvo os
   eventuais descartados na decisão 2 e as linhas inválidas da decisão 8). Validado
   explicitamente no Bloco 4.
7. **Hold-out de teste separado sobre os dados BRUTOS** — decisão estratégica do projeto,
   travada na Seção 5 do CLAUDE.md. Ver Seção 2.7 para o que esta etapa executa.
8. **Remoção de linhas com valores inválidos** — `maximum_nights = 9999` e `price == 0`.
   Ver Seção 2.8.

### 2.7 Hold-out de teste — ver CLAUDE.md, Seção 5

A separação de 10% para teste é **decisão estratégica do projeto**, não desta etapa: ela
governa também as Etapas 06 (fit só no treino), 09 (reaplicação do pipeline ao teste) e 10.
Por isso está travada na **Seção 5 do CLAUDE.md**, que é a fonte de verdade.

O que esta etapa executa: o split é a **primeira ação** após a carga de
`listings_details.csv`, antes de qualquer limpeza — 20.030 → 18.027 treino / 2.003 teste.
Da célula do split em diante, todo o pipeline opera sobre `df_train`
(`df_final = df_train.copy()`).

### 2.8 Remoção de linhas com valores inválidos

Dois cortes de linha, ambos com contagem antes/depois impressa no notebook.

**a) `maximum_nights == 9999` — 1 linha.**
Valor sentinela isolado, fora do domínio plausível (27 anos de estadia máxima). Distinto do
`1125` que aparece em 51% dos registros — esse é o default do Airbnb para "sem limite
prático" e é tratado como feature na Etapa 03, não removido.

**b) `price == 0` — 2 linhas. Bloqueador da métrica-objetivo.**

Este corte não é cosmético:

> O MAPE é `|y - ŷ| / |y|`. Com `y = 0`, o termo é uma **divisão por zero** → `inf`. Dois
> registros bastam para tornar o MAPE — a métrica que define o sucesso do projeto inteiro,
> conforme CLAUDE.md Seção 1 — **incalculável para todo o dataset**.

Além do problema aritmético, uma diária de € 0,00 não representa oferta real de mercado:
é erro de cadastro ou anúncio inativo, não um imóvel que alguém aluga de graça.

```python
df_final = df_final[df_final['price'] > 0].copy()
```

Resultado: **18.022 → 18.020 linhas**. O `price` mínimo passa de € 0,00 para **€ 8,00**.

> **O que ficou de fora deste corte, deliberadamente.** Restam 6 anúncios abaixo de € 20 e
> 22 acima de € 1.000. Não foram tocados aqui: são valores extremos, mas aritmeticamente
> válidos e possivelmente legítimos (quartos compartilhados baratos, imóveis de luxo). A
> decisão sobre winsorize ou piso/teto exige ver a distribuição e pertence à **Etapa 04
> (EDA), Análise 1.1** — Data Quality remove o que está *quebrado*, não o que é *raro*.

---

## 3. Estrutura da implementação no notebook

Nova seção "## Etapa 02 — Data Quality" em `notebooks/price-prediction.ipynb`, após a
Etapa 01. Desenvolvimento incremental (poucas células por vez, com validação de saída),
via MCP Jupyter. Nesta etapa, **funções são permitidas** (diferente da Etapa 01).

### Bloco 1 — Diagnóstico de qualidade por arquivo
Uma célula (ou poucas) por arquivo, reproduzindo shape, dtypes, `% nulos` por coluna
(ordenado), duplicatas e ranges — cada sub-bloco fecha com 2-3 linhas de markdown
"Observação". Ordem: `neighbourhoods.csv` → `listings.csv` (comparação cruzada com a
base) → `reviews.csv` vs `reviews_details.csv` → `listings_details.csv` (a base, por
último por ser a mais rica). `calendar.csv` não entra nesta lista — fora de escopo
(decisão 1 da Seção 2).

### Bloco 2 — Limpeza de `listings_details.csv`
Sub-blocos, cada um pequeno e validável:
1. Drop de colunas 100% nulas: `medium_url`, `host_acceptance_rate`, `xl_picture_url`,
   `neighbourhood_group_cleansed`, `thumbnail_url`.
2. Drop de colunas quase-100% nulas: `license`, `square_feet`, `monthly_price`,
   `weekly_price` — documentar que seriam features valiosas mas sem cobertura confiável.
3. Colunas de texto livre 30-55% nulas (`notes`, `host_about`, `interaction`, `access`,
   `house_rules`, `neighborhood_overview`, `transit`): manter intactas (viram features
   simples só na Etapa 03), sem parsing de NLP aqui.
4. Parse de colunas monetárias (`price`, `security_deposit`, `cleaning_fee`,
   `extra_people`) via função reutilizável `parse_money(series)`
   (`str.replace('[\$,]', '', regex=True).astype(float)`); nulos de `security_deposit`/
   `cleaning_fee` mantidos como NaN (decisão 3 da Seção 2). Validar `price` limpo contra
   `listings.csv['price']` (deve bater 1:1 por `id`) como teste de aceitação.
5. `host_response_rate` (`rstrip('%')` + `/100`) e `host_response_time` (mantida como
   categórica, sem encoding ainda); nulos mantidos.
6. `review_scores_rating` e as 6 colunas `review_scores_*`: reprint do describe
   confirmando escala 0-100; documentar explicitamente para não ser reescalada
   erroneamente depois; nulos (~13%) mantidos como NaN.
7. `availability_365`: célula de documentação (sem limpeza de valor) reforçando que não
   é ocupação — mantida na base (não é dropada), mas com aviso explícito de uso proibido
   como proxy de ocupação.
8. `calendar_updated`: documentar que é texto relativo (não data ISO); parsing adiado
   para Etapa 03 se a feature "recência de atualização" for julgada relevante. (Coluna
   de `listings_details.csv`, não confundir com o arquivo `calendar.csv` — que está fora
   de escopo.)
9. `market`: célula de inspeção geográfica dos 49 registros divergentes → drop
   condicional conforme decisão 2 da Seção 2, com contagem final de linhas removidas.
10. Drop de colunas constantes (`experiences_offered`, `country_code`, `country`,
    `has_availability`, `requires_license`, `jurisdiction_names`,
    `is_business_travel_ready`), com checagem `nunique() == 1` antes de dropar.
11. Flags binárias f/t → mapear para `{0,1}` (checando antes que só existam `{'f','t',NaN}`).
12. Bairro: dropar `neighbourhood` (cru) — manter só `neighbourhood_cleansed` como fonte
    de verdade; documentar o porquê. `property_type` e `amenities` mantidos crus nesta
    etapa (agrupamento/parsing ficam para Etapa 03/06).
13. **Remoção de linhas inválidas** (decisão 8 / Seção 2.8), ao fim do bloco — depois do
    parse monetário, que é o que torna a checagem `price == 0` possível:
    - `maximum_nights == 9999` → 1 linha
    - `price == 0` → 2 linhas

### Bloco 3 — Agregação da tabela auxiliar ao nível de anúncio
Markdown explicando por que `reviews_details.csv` (1 linha/review) precisa ser agregada
antes do merge (evitar explosão de linhas) — é a única tabela auxiliar restante no
projeto após a exclusão de `calendar.csv`.

#### 3.1 `reviews_details.csv` — granularidade de entrada e de saída

**Entrada:** 431.830 linhas, granularidade **1 linha por review** (`listing_id`, `id`
único de review, `date`, `reviewer_id`, `reviewer_name`, `comments`).

**Saída:** 1 linha por `listing_id` — mesma granularidade de `listings_details.csv`,
pronta para `merge` por chave única.

**Passo a passo da agregação:**
1. `groupby('listing_id').agg(...)` único, calculando:
   - `reviews_count_total` = `size()` — contagem de reviews por listing.
   - `first_review_date_agg` = `min('date')`, `last_review_date_agg` = `max('date')`.
2. Célula de checagem cruzada: comparar `reviews_count_total` contra
   `number_of_reviews` (já existente em `listings_details.csv`), e
   `first_review_date_agg`/`last_review_date_agg` contra `first_review`/`last_review` —
   decidir se as colunas agregadas são redundantes (e nesse caso não entram no merge
   final, evitando duplicidade de informação) ou se há divergências que justifiquem
   mantê-las como checagem de qualidade adicional.
3. `comments` (texto livre de review) não é trazido para o nível de anúncio nesta etapa
   — fora de escopo de Data Quality/preço.
4. ~12% dos listings não têm nenhuma review — o `left join` no Bloco 4 gera `NaN` nessas
   colunas agregadas para esses casos; não imputar, é categoria de negócio válida (mesmo
   raciocínio de `review_scores_rating`).
5. Célula de validação explícita: `reviews_agg.shape[0]` deve ser ≤ 20.030 (só listings
   com ao menos 1 review aparecem no agregado, já que é um `groupby` sem `reindex`) e
   `reviews_agg['listing_id'].is_unique == True` — confirma 1 linha por listing antes do
   merge.

`neighbourhoods.csv` e `listings.csv` continuam usadas só como validação cruzada
(Bloco 1 e Bloco 2.4) — não entram no merge final.

### Bloco 4 — Merge final e validação de granularidade
1. `listings_details.csv` (limpo, Bloco 2) **left join** com `reviews_agg` (Bloco 3.1)
   por `id`==`listing_id`.
2. Validação: `shape` antes/depois do merge — **o número de linhas não deve mudar**
   (permanece 20.030, ou o total após eventual drop de `market` na decisão 2), já que o
   lado direito (`reviews_agg`) tem no máximo 1 linha por `listing_id`.
3. Checagem final e obrigatória de granularidade (decisão 6 da Seção 2):
   `df_clean['id'].is_unique == True` e `len(df_clean) == df_clean['id'].nunique()` —
   confirma explicitamente **1 linha por `listing_id`** no dataset final.
4. Tabela-resumo de nulos pós-merge, comparada com a do Bloco 1 (antes/depois).

### Bloco 5 — Consolidação e entregável
1. Markdown com tabela-resumo de decisões de limpeza (coluna | ação | motivo) — serve de
   changelog de Data Quality para a Etapa 03.
2. `df_clean.info()` final + contagem de colunas mantidas vs. dropadas.
3. `df_clean` mantido em memória (decisão 4 da Seção 2) para a Etapa 03 continuar no
   mesmo notebook. Granularidade final documentada explicitamente em markdown: **1 linha
   por `listing_id`, sem dimensão temporal**.
4. Fechamento com o checklist da Seção 8 do CLAUDE.md e pedido explícito de validação
   antes de iniciar a Etapa 03.

---

## 4. Verificação

- Cada bloco/sub-bloco é executado via MCP Jupyter (`insert_execute_code_cell` /
  `execute_cell`) e a saída é mostrada ao usuário antes de avançar, conforme Regra de
  Trabalho nº 2 do CLAUDE.md.
- Teste de aceitação explícito no Bloco 2.4: `price` limpo de `listings_details.csv`
  deve bater 1:1 com `listings.csv['price']` por `id` (checagem já validada no
  levantamento prévio, será reproduzida no notebook).
- Teste de aceitação no Bloco 4: nenhuma duplicação de linha após o merge (shape antes
  == shape depois em número de linhas) e `id.duplicated().sum() == 0`.
- Teste de aceitação final (Bloco 4/5): `len(df_clean) == df_clean['id'].nunique()` —
  confirma a granularidade obrigatória de 1 linha por `listing_id` exigida pela decisão
  de modelagem estática.
- **Pré-condição do MAPE:** `(df_final['price'] == 0).sum() == 0` — sem isso, a
  métrica-objetivo do projeto é incalculável (Seção 2.8b).
- **Isolamento do teste:** nenhuma célula posterior ao split lê `df_test` ou
  `test_raw.csv`. O pipeline opera exclusivamente sobre `df_train` (CLAUDE.md, Seção 5.3).
- Ao final de cada bloco macro (1 a 5), pausa para validação explícita do usuário antes
  de seguir, conforme Regra de Trabalho nº 1 e nº 2 do CLAUDE.md — Data Quality inteira
  não é escrita de uma vez.

---

## 5. Resultado da execução

| Momento | Linhas × Colunas | Causa |
|---|---|---|
| `listings_details.csv` (bruto) | 20.030 × 96 | — |
| Após hold-out de teste (10%) | 18.027 × 96 | CLAUDE.md, Seção 5 |
| Após limpeza de colunas | 18.023 × 88 | 8 colunas dropadas + linhas com `bathrooms > 10` / `minimum_nights > 365` |
| Após drop de categóricas desbalanceadas | 18.023 × 83 | 5 colunas quase-constantes |
| Após `maximum_nights == 9999` | 18.022 × 83 | Seção 2.8a |
| **Após `price == 0`** | **18.020 × 83** | **Seção 2.8b** |

Granularidade final confirmada: **1 linha por `listing_id`**, sem dimensão temporal.
`price` mínimo: **€ 8,00**. Dataset entregue em memória (`df_final`) para a Etapa 03.
