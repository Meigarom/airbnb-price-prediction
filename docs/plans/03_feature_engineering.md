# Plano — Etapa 03: Feature Engineering (Airbnb Amsterdam)

## Contexto

Entramos nesta etapa com o dataset limpo da Etapa 02: **18.020 linhas × 83 colunas**,
granularidade de 1 linha por `listing_id`, modelagem estática (sem dimensão temporal).

> **Nota sobre a contagem de linhas.** O plano original citava 20.025 linhas, número
> anterior a duas mudanças posteriores na Etapa 02: a separação do **hold-out de teste
> (10%)** sobre os dados crus, que levou o treino a 18.027 linhas, e a remoção de linhas
> inválidas (`maximum_nights = 9999` e, depois, `price == 0`). O resultado desta etapa
> é **18.020 × 83**.

> **Estado deste documento.** O plano foi executado e depois revisado. As seções abaixo
> refletem o que **de fato existe** no notebook, não a proposta original. Onde uma feature
> planejada foi descartada, o registro foi mantido com a justificativa — o histórico da
> decisão vale mais que um plano limpo. As alterações estão consolidadas na **Seção 9**.

O objetivo do projeto (CLAUDE.md, Seção 1) é prever a **diária** (`price`) como problema
de regressão, minimizando **MAPE**. O valor de negócio vem de identificar anúncios
subprecificados — ou seja, o modelo precisa acertar bem na **faixa central de preços**,
onde está a massa de anúncios acionáveis, e não apenas nos extremos.

Isso orienta duas decisões que atravessam todo este plano:

1. **Features que explicam capacidade e qualidade do imóvel** têm prioridade, porque são
   o que o anfitrião de fato precifica.
2. **Nenhuma feature pode derivar de `price`**, direta ou indiretamente — sob pena de
   vazamento que inflaria as métricas de treino e quebraria o modelo em produção.

---

## 1. Regra crítica — vazamento de dados (data leakage)

Antes de listar as features, é preciso marcar o que **não** pode ser criado.

### 1.1 Colunas proibidas como feature

| Coluna | Por que vaza |
|---|---|
| `weekly_price` | É `price × 7` com desconto — contém o alvo |
| `monthly_price` | Mesma lógica, escala mensal |
| `price_por_pessoa` | `price / accommodates` — o alvo está no numerador |
| Qualquer razão com `price` | Idem |

`weekly_price` (85,8% nulo) e `monthly_price` (92,2% nulo) devem ser **dropadas** logo no
início da etapa. O baixo preenchimento já as inviabilizaria; o vazamento as elimina em
definitivo.

> **Nota sobre `price_por_pessoa`:** é uma métrica excelente para **EDA** (Etapa 04) e
> para interpretar o resultado do recomendador (Etapa 09). Não entra como feature de
> treino. Se for calculada, deve ficar num DataFrame separado, nunca em `X`.

### 1.2 Colunas de scrape (constantes, sem informação)

`last_scraped` e `calendar_last_scraped` têm apenas 2 valores (2018-12-06 e 2018-12-07).
Servem apenas como **data de referência** para calcular idades — não viram feature.

---

## 2. Features planejadas

### Bloco A — Geográficas

O sinal mais forte que validei na inspeção prévia. A mediana de preço cai de **€160**
(<1km do centro) para **€95** (>8km), com correlação de Spearman **-0,251**.

| Feature | Fórmula | Justificativa de negócio |
|---|---|---|
| `dist_centro_km` | Haversine até Dam Square (52.3731, 4.8922) | Proximidade ao centro histórico é o principal driver de preço em Amsterdam |
| `dist_museumplein_km` | Haversine até Museumplein (52.3579, 4.8816) | Polo turístico (Rijksmuseum, Van Gogh) — demanda alta e sustentada |
| `dist_centro_bin` | Faixas: <1, 1-2, 2-3, 3-5, 5-8, >8 km | Captura não-linearidade; útil para modelos lineares |

**Racional da escolha de múltiplos polos:** Amsterdam não é monocêntrica para fins de
turismo. Um anúncio a 3km do Dam mas a 500m do Museumplein tem valor de mercado
diferente de um a 3km de ambos.

`latitude` e `longitude` permanecem no dataset — modelos de árvore conseguem extrair
interações espaciais que as distâncias não capturam (ex.: bairros ao norte do IJ, que
exigem balsa).

#### A.1 `dist_estacao_central_km` — descartada (revisão pós-execução)

O plano original previa uma terceira distância, até Amsterdam Centraal. Ela **não é mais
criada**. A hipótese de "múltiplos polos" não se sustentou para esse ponto: os três
referenciais são vizinhos entre si num raio urbano onde os anúncios chegam a 11 km.

| Par de referenciais | Distância |
|---|---|
| Dam Square ↔ Centraal | **0,86 km** |
| Dam Square ↔ Museumplein | 1,84 km |
| Centraal ↔ Museumplein | 2,68 km |

**Multicolinearidade.** `dist_estacao_central_km` tinha ρ Pearson de **+0,958** com
`dist_centro_km`. VIF das três: `dist_centro_km` 74,6 · `dist_estacao_central_km` 44,3 ·
`dist_museumplein_km` 9,8 — duas muito acima do limiar usual de alarme (VIF > 10).

**Poder preditivo** (R², RandomForest CV=3, alvo `log1p(price)`):

| Conjunto | Colunas | R² |
|---|---|---|
| só `dist_centro` | 1 | 0,055 |
| centro + centraal | 2 | 0,074 |
| **centro + museumplein** | **2** | **0,087** |
| as três | 3 | 0,093 |
| `latitude` + `longitude` puros | 2 | **0,112** |
| lat/lon + as três | 5 | 0,110 |

**Decisão:** mantido o par centro + Museumplein. Centraal era a mais redundante, e
trocá-la por Museumplein no par **melhora** o R² (0,074 → 0,087). A terceira distância
agregava apenas +0,006 de R² ao custo de multicolinearidade severa. VIF do par
remanescente: **2,66**.

> **Achado para a Etapa 05.** `latitude` + `longitude` sozinhas (R² = 0,112) **superam as
> três distâncias juntas** (0,093), e acrescentar as distâncias ao lat/lon piora
> ligeiramente (0,110). A árvore extrai da geometria bruta mais do que qualquer distância
> radial — coerente com a cidade não ser radial: o IJ a corta, os canais formam anéis, e
> bairros equidistantes do Dam têm valores muito diferentes.
>
> As distâncias foram mantidas mesmo assim por serem **interpretáveis para o negócio** —
> "a 1,2 km do centro" é comunicável ao anfitrião no recomendador da Etapa 09, "latitude
> 52,37" não é. O Boruta (Etapa 05) é o portão formal dessa decisão.

---

### Bloco B — Amenities

`amenities` é uma string no formato `{Wifi,Heating,"Smoke detector",...}` com
**121 amenities distintas**. Correlação da contagem com preço: **+0,213**.

#### B.1 Contagem

| Feature | Justificativa |
|---|---|
| `n_amenities` | Proxy direto de completude/luxo do imóvel |

#### B.2 Flags binárias selecionadas

Não vale criar 121 dummies — a maioria é ruído ou quase-constante. Seleção por dois
critérios: **incidência entre 10% e 90%** (tem variância) e **relevância de negócio**.

Incidências abaixo conforme **medidas na execução** (sobre as 18.020 linhas de treino):

| Flag | Incidência | Justificativa |
|---|---|---|
| `tem_tv` | 78,0% | Conforto básico esperado |
| `tem_washer` | 75,9% | Sinaliza estadia longa viável |
| `tem_dryer` | 44,6% | Diferencial em apartamentos europeus |
| `tem_family_kid_friendly` | 44,4% | Abre segmento de famílias (ticket maior) |
| `tem_cable_tv` | 39,5% | Indicador de imóvel mais equipado |
| `tem_private_entrance` | 27,2% | Privacidade — valor claro para casais |
| `tem_dishwasher` | 21,1% | Forte indicador de imóvel de padrão alto |
| `tem_patio_balcony` | 16,1% | Diferencial relevante e escasso |
| `tem_free_parking` | 11,7% | Escasso e caríssimo em Amsterdam — alto valor |
| `tem_elevator` | **9,7%** | Crítico: prédios holandeses são estreitos e altos |
| `tem_step_free_access` | **6,3%** | Acessibilidade — nicho com baixa concorrência |

> **Nota — duas flags abaixo do piso de 10%.** As estimativas prévias de `tem_elevator`
> (~15%) e `tem_step_free_access` (12,9%) não se confirmaram: na medição real ficaram em
> 9,7% e 6,3%, abaixo do critério de incidência ≥10% declarado acima. Ambas foram
> **mantidas** por relevância de negócio — elevador é decisivo em prédios holandeses e
> acessibilidade é um nicho de baixa concorrência — e nenhuma é rara a ponto de inviabilizar
> a estimativa (a menor delas cobre ~1.135 anúncios). O Boruta (Etapa 05) decide.

**Excluídas por quase-constância** (>90% ou <5%): `Wifi` (97,2%), `Heating` (96,6%),
`Essentials` (95,9%), `Kitchen` (91,3%). Não discriminam — mesmo problema das colunas
que removemos na Etapa 02.

**Excluídas por serem lixo de ETL:** `translation missing: en.hosting_amenity_49` e
`_50` (24% e 18,7%). São falhas de internacionalização do Airbnb, não amenities reais.

#### B.3 Agregado de qualidade

| Feature | Composição | Justificativa |
|---|---|---|
| `n_amenities_luxo` | Soma de: dishwasher, dryer, elevator, patio/balcony, free parking, private entrance | Índice composto de padrão do imóvel — mais robusto que flags isoladas |

---

### Bloco C — Temporais (derivadas de datas)

Data de referência: `2018-12-06` (data do scrape).

| Feature | Fórmula | Justificativa |
|---|---|---|
| `host_experiencia_dias` | `ref - host_since` | Anfitrião veterano tende a precificar melhor e ter operação profissional |
| `dias_desde_primeira_review` | `ref - first_review` | Idade efetiva do anúncio no mercado |
| `dias_desde_ultima_review` | `ref - last_review` | **Proxy de atividade** — anúncio sem review recente pode estar inativo |
| `anuncio_ativo` | `dias_desde_ultima_review <= 180` | Flag binária: separa operação viva de anúncio dormente |
| `calendar_atualizado_dias` | Parse de `calendar_updated` (texto relativo) | Anfitrião que atualiza calendário está engajado |

**Sobre `calendar_updated`:** é texto relativo (`"today"`, `"a week ago"`,
`"2 months ago"`). Requer parser dedicado que converta para dias. Os 2.996 registros
com `"today"` (15%) indicam gestão ativa — sinal relevante de profissionalização.

**Sobre nulos:** `first_review`/`last_review` têm 2.162 nulos (12,0%) — são anúncios sem
nenhuma review. Não imputar: criar flag `sem_reviews` e deixar `NaN`. É categoria de
negócio legítima (anúncio novo), não dado faltante.

#### C.1 `host_experiencia_anos` — descartada (revisão pós-execução)

Previa-se uma versão em anos (`host_experiencia_dias / 365.25`) como "versão interpretável
para análise". Ela **não é mais criada**: por ser uma transformação linear por constante,
tem **ρ = 1,0000** com `host_experiencia_dias` — é a mesma variável noutra unidade, com
zero informação adicional para o modelo.

A conversão para anos é trivial na hora de comunicar um resultado (`dias / 365,25`), e não
justifica carregar uma coluna redundante pela matriz de correlação da EDA e pelo Boruta.

---

### Bloco D — Host / Operação

| Feature | Fórmula | Justificativa |
|---|---|---|
| `n_host_verifications` | Contagem da lista em `host_verifications` | Proxy de confiabilidade — mais verificações, mais credibilidade |
| `host_tem_id_verificado` | `'government_id' in host_verifications` | Verificação forte, distinta de e-mail/telefone |
| `host_eh_profissional` | `calculated_host_listings_count > 1` | **Separa anfitrião casual de gestora** — modelos de precificação distintos |
| `host_eh_local` | `host_location` contém "Amsterdam" | Anfitrião local vs. investidor remoto — afeta gestão e precificação |

**Racional de `host_eh_profissional`:** é possivelmente a feature mais importante deste
bloco. Gestoras profissionais usam precificação dinâmica e têm estrutura de custo
diferente; anfitriões casuais precificam por intuição. O modelo se beneficia de saber
com qual perfil está lidando. Incidência medida: **21,1%** dos anúncios.

#### D.1 `host_portfolio_size` — descartada (revisão pós-execução)

Era uma **cópia literal** de `calculated_host_listings_count` (ρ = 1,0000), renomeada.
A coluna original já está no dataset e cumpre o mesmo papel — "escala da operação" — então
a duplicata apenas inflava a contagem de features. **Não é mais criada.**

#### D.2 `host_response_rate_num` — criada e depois removida

A conversão de texto para numérico foi implementada como planejado, mas a feature foi
removida na validação de fim de etapa: **47,4% de nulos** (8.534 anúncios), a maior lacuna
do conjunto de longe.

A causa é estrutural, não um defeito de dados: o Airbnb só exibe taxa de resposta para
anfitriões com histórico de mensagens suficiente. Imputar metade da coluna introduziria
mais ruído que sinal, e a informação de "não tem taxa exibida" já é parcialmente capturada
por `host_response_time` (categórica, mantida). Com a saída dela, a maior lacuna do dataset
caiu de 47,4% para **18,1%** (`taxa_limpeza_por_noite`).

---

### Bloco E — Estrutura do imóvel (razões)

Razões entre colunas existentes capturam **eficiência de uso do espaço** — algo que o
valor absoluto de cada coluna não expressa.

| Feature | Fórmula | Justificativa |
|---|---|---|
| `pessoas_por_quarto` | `accommodates / bedrooms` | Alta razão = acomodação apertada/hostel; baixa = conforto |
| `pessoas_por_cama` | `accommodates / beds` | Idem, granularidade de cama |
| `banheiros_por_pessoa` | `bathrooms / accommodates` | Forte indicador de padrão |
| `camas_por_quarto` | `beds / bedrooms` | Distingue suíte de dormitório compartilhado |
| `tem_multiplos_banheiros` | `bathrooms > 1` | Salto de categoria no mercado |

**Cuidado com divisão por zero:** `bedrooms` e `beds` têm valor 0 em alguns registros
(mín. = 0 na tabela descritiva). Usar `.replace(0, np.nan)` antes da divisão para gerar
`NaN` em vez de `inf`.

---

### Bloco F — Política de reserva

| Feature | Fórmula | Justificativa |
|---|---|---|
| `sem_limite_maximo` | `maximum_nights >= 1125` | **Sentinela em 51% dos dados** (ver análise Etapa 02) |
| `maximum_nights_cap` | `maximum_nights.clip(upper=365)` | Torna a variável contínua interpretável |
| `min_nights_bin` | Faixas: 1, 2-3, 4-7, 8-30, >30 | Captura regimes distintos (fim de semana, semana, mensal) |
| `permite_estadia_curta` | `minimum_nights <= 3` | Segmento turístico de alta rotatividade |
| `taxa_limpeza_por_noite` | `cleaning_fee / minimum_nights` | Custo real percebido pelo hóspede na estadia mínima |
| `tem_taxa_limpeza` | `cleaning_fee.notna() & > 0` | Nulo aqui significa "sem cobrança", não dado faltante |
| `tem_deposito` | `security_deposit.notna() & > 0` | Idem |

**Sobre `maximum_nights`:** conforme análise da Etapa 02, o valor `1125` aparece em
**51,06%** dos registros — é o default do Airbnb para "sem limite prático", não uma
decisão do anfitrião. Tratar como contínuo distorce a distribuição (bimodal). A dupla
flag + clip preserva o sinal sem a distorção.

`taxa_limpeza_por_noite` não vaza: `cleaning_fee` é cobrança separada da diária, não
derivada de `price`.

---

### Bloco G — Reviews e reputação

| Feature | Fórmula | Justificativa |
|---|---|---|
| `sem_reviews` | `number_of_reviews == 0` | Anúncio novo — sem histórico de reputação |
| `review_score_medio` | Média das 6 subnotas `review_scores_*` | Consolida dimensões correlacionadas |
| `gap_location_vs_geral` | `review_scores_location - review_score_medio` | **Isola valor da localização** percebido pelo hóspede |
| `gap_value_vs_geral` | `review_scores_value - review_score_medio` | Nota de "value" baixa relativa sugere **superprecificação** |
| `reviews_por_mes_ativo` | `number_of_reviews / meses_desde_first_review` | Velocidade de ocupação normalizada |

**Racional de `gap_value_vs_geral`:** é a feature conceitualmente mais alinhada ao
objetivo do projeto. Um anúncio com notas altas em tudo, exceto em "value for money",
está sinalizando que os hóspedes o consideram caro — o inverso do que buscamos. Pode
ajudar o modelo a calibrar a faixa de preço justo.

**Atenção:** `review_scores_rating` está em escala **0–100**, enquanto as 6 subnotas
estão em **0–10** (confirmado na Etapa 02).

**Como foi resolvido na execução:** em vez de normalizar, `review_scores_rating` foi
**excluída da média** — `review_score_medio` é a média das 6 subnotas em 0–10, todas na
mesma escala. Reescalar a nota geral para juntá-la às subnotas seria redundante: ela já é,
em essência, um agregado das mesmas dimensões. `review_scores_rating` permanece no dataset
como coluna independente, na escala original.

---

### Bloco H — Texto livre (features simples)

Não faremos NLP nesta etapa. Extraímos apenas **sinais de esforço do anfitrião** —
preencher esses campos é opcional, então preenchê-los indica engajamento.

| Feature | Fórmula | Justificativa |
|---|---|---|
| `len_description` | Nº de caracteres | Proxy de detalhamento |
| `len_house_rules` | Nº de caracteres | Regras extensas = anfitrião criterioso |
| `n_campos_texto_preenchidos` | Contagem de not-null entre 10 campos de texto | **Índice de completude do anúncio** |

Campos considerados (10): `summary`, `space`, `description`, `neighborhood_overview`,
`notes`, `transit`, `access`, `interaction`, `house_rules`, `host_about`.

#### H.1 `tem_descricao_completa` — criada e depois removida

`description.notna()` foi implementada, mas removida na validação de fim de etapa:
**99,4% de incidência**. Praticamente todo anúncio tem descrição preenchida, então a flag
não discrimina nada — é o mesmo critério de quase-constância que eliminou `Wifi` (97,2%) e
`Heating` (96,6%) no Bloco B, e as colunas categóricas desbalanceadas na Etapa 02.

O sinal que ela pretendia capturar ("esforço do anfitrião") sobrevive em `len_description`
e `n_campos_texto_preenchidos`, ambas contínuas e com variância real.

---

## 3. Colunas a descartar nesta etapa

| Coluna | Motivo |
|---|---|
| `weekly_price`, `monthly_price` | **Vazamento** + 86%/92% nulos |
| `square_feet` | 98% nulo — inviável mesmo sendo valiosa |
| `license` | 100% nulo |
| URLs (`listing_url`, `picture_url`, `host_url`, `host_thumbnail_url`, `host_picture_url`) | Identificadores sem valor preditivo |
| `host_name`, `name` | Texto livre identificador; risco de overfitting |
| `last_scraped`, `calendar_last_scraped` | Constantes (2 valores) |
| `country`, `country_code`, `state`, `smart_location` | Constantes — todos Amsterdam/NL |
| `experiences_offered`, `has_availability`, `requires_license`, `is_business_travel_ready`, `jurisdiction_names` | Constantes (verificadas na Etapa 02) |
| `neighbourhood` (cru) | 44 valores inconsistentes; `neighbourhood_cleansed` é a fonte de verdade |
| `zipcode` | 6.870 valores únicos, 4,3% nulo, formato inconsistente — alta cardinalidade sem ganho sobre `neighbourhood_cleansed` |
| Colunas de texto originais | Substituídas pelas features do Bloco H |

---

## 4. Tratamento de `property_type`

Sinalizado na Etapa 02: 31 categorias, com `Apartment` em 77,8%.

**Estratégia:** agrupar categorias com <1% de incidência em `"Other"`, preservando as
que têm massa crítica.

**Resultado da execução:** 30 categorias → **9**.

| Categoria | Anúncios | % |
|---|---|---|
| Apartment | 14.040 | 77,91 |
| House | 1.364 | 7,57 |
| Townhouse | 579 | 3,21 |
| Other | 454 | 2,52 |
| Bed and breakfast | 411 | 2,28 |
| Boat | 345 | 1,91 |
| Loft | 336 | 1,86 |
| Condominium | 290 | 1,61 |
| Houseboat | 201 | 1,12 |

`Boat` e `Houseboat` merecem destaque: são característicos de Amsterdam (casas-barco nos
canais) e têm dinâmica de preço própria. Ambos sobreviveram ao corte de 1% por mérito
próprio, sem necessidade de exceção manual — juntos somam 546 anúncios (3,03%).
`Serviced apartment`, citada no plano original como categoria a preservar, ficou abaixo do
limiar e caiu em `"Other"`.

---

## 5. Ordem de execução

1. Drop das colunas de vazamento (`weekly_price`, `monthly_price`) — **primeiro de tudo**
2. Bloco C (temporais) — precisa das datas antes de dropá-las
3. Bloco A (geográficas)
4. Bloco B (amenities)
5. Bloco D (host)
6. Bloco E (estrutura)
7. Bloco F (política de reserva)
8. Bloco G (reviews)
9. Bloco H (texto) — antes de dropar as colunas de texto
10. Agrupamento de `property_type`
11. Drop final das colunas originais consumidas
12. Validação: shape, nulos, `inf`, e **checagem explícita de correlação com `price`**
13. **Remoção pós-validação** — features criadas que a própria validação reprovou
    (`host_response_rate_num`, `tem_descricao_completa`; ver D.2 e H.1)

---

## 6. Verificação

- **Granularidade:** `len(df) == df['id'].nunique()` permanece verdadeiro
- **Sem infinitos:** `np.isinf(df.select_dtypes('number')).sum().sum() == 0`
- **Anti-vazamento (crítico):** correlação de Spearman de cada nova feature com `price`.
  Qualquer feature com |ρ| > 0,95 deve ser investigada como suspeita de vazamento
- **Nulos controlados:** features novas não devem introduzir nulos além dos herdados
- **Documentação:** cada feature registrada em markdown com fórmula e justificativa

### 6.1 Resultado da verificação

Todos os critérios foram atendidos:

| Critério | Resultado |
|---|---|
| Granularidade | 18.020 → 18.020 linhas · 1 linha por `listing_id` ✅ |
| Valores infinitos | 0 ✅ |
| Anti-vazamento | Nenhuma feature com \|ρ\| > 0,95 ✅ |
| Nulos | Todos herdados das colunas de origem — nenhum introduzido ✅ |

**Top 8 correlações de Spearman com `price`:**

| Feature | ρ |
|---|---|
| `banheiros_por_pessoa` | **−0,395** |
| `taxa_limpeza_por_noite` | +0,281 |
| `dist_centro_km` | −0,250 |
| `tem_tv` | +0,239 |
| `tem_multiplos_banheiros` | +0,233 |
| `dist_museumplein_km` | −0,232 |
| `tem_family_kid_friendly` | +0,232 |
| `tem_dryer` | +0,226 |

A feature mais associada ao alvo é `banheiros_por_pessoa` (Bloco E) — razão que não existia
no dataset original e que o feature engineering criou. Confirma a hipótese do Bloco E: a
**eficiência de uso do espaço** carrega mais sinal de preço que os valores absolutos de
`bathrooms` ou `accommodates` isolados.

Nenhum ρ ultrapassa 0,40 em módulo, o que é o esperado e saudável: preço de imóvel é
multifatorial, e uma correlação isolada muito alta seria mais suspeita de vazamento que
promissora.

**Nulos remanescentes** (todos herdados, nenhum criado pelo feature engineering):

| Feature | Nulos | % | Origem |
|---|---|---|---|
| `taxa_limpeza_por_noite` | 3.262 | 18,1% | `cleaning_fee` ausente |
| `gap_location_vs_geral` | 2.388 | 13,3% | Anúncio sem review |
| `gap_value_vs_geral` | 2.387 | 13,2% | Idem |
| `review_score_medio` | 2.374 | 13,2% | Idem |
| `reviews_por_mes_ativo` | 2.168 | 12,0% | Idem |
| `dias_desde_primeira_review` | 2.162 | 12,0% | Idem |
| `dias_desde_ultima_review` | 2.162 | 12,0% | Idem |
| `camas_por_quarto` | 1.023 | 5,7% | `bedrooms = 0` (estúdio) |
| `pessoas_por_quarto` | 1.017 | 5,6% | Idem |
| Demais | ≤ 10 | ≤ 0,1% | — |

---

## 7. Resumo quantitativo

| Bloco | Planejado | **Final** | Diferença |
|---|---|---|---|
| A — Geográficas | 4 | **3** | −1 (`dist_estacao_central_km`) |
| B — Amenities | 13 | **13** | — |
| C — Temporais | 6 | **5** | −1 (`host_experiencia_anos`) |
| D — Host | 6 | **4** | −2 (`host_portfolio_size`, `host_response_rate_num`) |
| E — Estrutura | 5 | **5** | — |
| F — Política de reserva | 7 | **7** | — |
| G — Reviews | 5 | **5** | — |
| H — Texto | 4 | **3** | −1 (`tem_descricao_completa`) |
| Agrupamento | 1 | **1** | — |
| **Total** | ~50 | **46** | **−5** |

**Dataset resultante: 18.020 linhas × 83 colunas.**

A seleção final fica para a **Etapa 05 (Feature Selection)**, com Boruta + importância
por árvores. Aqui o objetivo é **gerar candidatas com fundamento de negócio**, não
decidir o conjunto final.

> As 5 features eliminadas não foram "erros de planejamento" — quatro delas só puderam ser
> avaliadas depois de calculadas (nulos reais, incidência real, colinearidade real), e uma
> (`host_portfolio_size`) era redundância que passou despercebida no plano. Registrá-las
> aqui com o motivo é o que permite não recriá-las por engano na Etapa 06.

---

## 8. Pendências — status final

As três pendências levantadas antes da execução, e como ficaram:

1. **Ponto de referência geográfico** — ✅ **Resolvido.** Dam Square confirmado como
   referência principal, com Museumplein como segundo polo. Amsterdam Centraal foi
   testada e **descartada** por colinearidade (ver A.1). O teste empírico substituiu o
   argumento: o par centro + Museumplein prevê melhor (R² 0,087) que centro + Centraal
   (0,074).
2. **Corte de `property_type`** — ✅ **Resolvido.** Limiar de 1% mantido. Resultou em 9
   categorias, com `Boat` e `Houseboat` sobrevivendo por mérito próprio (ver Seção 4).
3. **Bloco H / NLP** — ⏸️ **Adiado, sem escopo definido.** Mantido o tratamento simples
   (contagem de caracteres e de campos preenchidos). TF-IDF ou embeddings sobre
   `description`/`house_rules` continuam sendo uma possibilidade de ganho, mas exigem
   etapa própria e não foram avaliados. Fica registrado como oportunidade, não como
   pendência bloqueante.

---

## 9. Registro de alterações pós-execução

Consolidação das mudanças feitas depois da execução inicial, para rastreabilidade.

### 9.1 Features que deixaram de ser criadas (removidas na origem)

| Feature | Bloco | Motivo | Evidência |
|---|---|---|---|
| `dist_estacao_central_km` | A | Colinear com `dist_centro_km` | ρ = 0,958 · VIF 44,3 · R² do par piora |
| `host_experiencia_anos` | C | Duplicata exata de `host_experiencia_dias` | ρ = 1,0000 |
| `host_portfolio_size` | D | Cópia literal de `calculated_host_listings_count` | ρ = 1,0000 |

Estas três foram eliminadas **no ponto de criação**, e não por um drop posterior. O motivo
é prático: um drop na Etapa 02 não funcionaria, porque as colunas só passam a existir na
Etapa 03 — com `errors='ignore'` o drop seria silenciosamente inócuo e a Etapa 03 as
recriaria logo em seguida.

### 9.2 Features criadas e removidas na validação

| Feature | Bloco | Motivo |
|---|---|---|
| `host_response_rate_num` | D | 47,4% de nulos — maior lacuna do conjunto |
| `tem_descricao_completa` | H | 99,4% de incidência — quase-constante |

Diferente das anteriores, estas **precisam ser calculadas para serem avaliadas**: só depois
de criá-las se conhece o percentual real de nulos e de incidência. Por isso permanecem no
fluxo e saem numa célula dedicada ao fim da etapa (passo 13 da Seção 5).

### 9.3 Mudança na contagem de linhas

| Momento | Linhas | Causa |
|---|---|---|
| Plano original | 20.025 | Dataset completo pós-Etapa 02 |
| Após hold-out de teste (10%) | 18.027 | Separação de `test_raw.csv` sobre dados crus |
| Após remoção de `price == 0` | **18.020** | 2 anúncios com diária € 0,00 |

> **Débito registrado para a Etapa 09** (regra travada em CLAUDE.md, Seção 5.4).
> `test_raw.csv` está **cru**. Toda a limpeza da Etapa 02 e todo o feature engineering
> desta etapa precisam ser reaplicados a ele, na mesma ordem e com os mesmos parâmetros,
> antes da avaliação final.
