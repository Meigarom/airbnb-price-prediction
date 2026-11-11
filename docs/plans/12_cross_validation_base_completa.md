# Plano — Precificação da base completa via Cross-Validation (Airbnb Amsterdam)

## Contexto

As Etapas 01 a 10 entregaram um modelo validado: **LightGBM tunado, MAPE de 21,93%** medido
num hold-out de 2.003 anúncios. Mas a extensão da Etapa 10 revelou um problema ao tentar
precificar a base inteira:

| Regime | n | MAPE | Zona morta | Subprecificados |
|---|---|---|---|---|
| in-sample (vistos no fit) | 17.978 | **15,63%** | 75,6% | 6,2% |
| out-of-sample (hold-out) | 2.003 | **21,93%** | 63,0% | 12,3% |

O modelo **memorizou parcialmente** o preço dos anúncios de treino. A predição fica próxima
do preço real, o desvio cai dentro da banda, e o anúncio é classificado como "zona morta" —
sem recomendação. Memorização não produz recomendações, produz silêncio.

Consequência: o EV calculado sobre a base completa (€109.372) é um **artefato** de medir num
conjunto majoritariamente memorizado, não uma estimativa de produção. O número honesto
(€351.044) depende de **extrapolar ×10** o hold-out.

> **O que este plano resolve:** gerar `price_estimate` **out-of-sample para todos os
> anúncios**, eliminando a extrapolação, produzindo 5 medições independentes de MAPE em vez
> de uma, e permitindo reconstruir o recomendador sobre a base inteira.

---

## 1. A estratégia

Cross-validation de **k=5 folds** sobre a base completa. Cada anúncio recebe uma predição de
um modelo que **nunca o viu**.

```
base limpa (19.981 anúncios)
  │
  ├─ fold 1: treino 80% → modelo₁ → prediz os 20% separados  ─┐
  ├─ fold 2: treino 80% → modelo₂ → prediz os 20% separados   │
  ├─ fold 3: treino 80% → modelo₃ → prediz os 20% separados   ├→ price_estimate
  ├─ fold 4: treino 80% → modelo₄ → prediz os 20% separados   │   (100% da base,
  └─ fold 5: treino 80% → modelo₅ → prediz os 20% separados  ─┘    out-of-sample)
```

As 5 partições de teste são **disjuntas** e sua união é a base inteira — propriedade do
`KFold`. Ao fim, `price_estimate` está preenchido para todos, e nenhum valor veio de um
modelo que viu aquele anúncio.

### 1.1 Decisões validadas pelo usuário

| # | Decisão | Onde |
|---|---|---|
| 1 | **k=5 com 20% de teste por fold** | §1 — a especificação original dizia 10%, o que com k=5 cobriria apenas 50% da base |
| 2 | **As 27 features de `FEATURES_05` ficam FIXAS** | §4.1 |
| 3 | **Opção A: limpeza fora, pipeline dentro** | §3 |
| 4 | **`price_estimate` SUBSTITUI `preco_justo` no recomendador** — é a mesma coluna, renomeada | §5 |
| 5 | **A banda passa a usar o MAPE médio da CV** | §6 |

---

## 2. O que recalcular por fold — o ponto crítico

A especificação original dizia *"manter todas as etapas, exceto os valores de encoding"*.
Está correto no princípio, mas a lista real é maior. Um levantamento das Etapas 02, 03 e 06
identificou **todo parâmetro aprendido dos dados**:

### 2.1 RECALCULA a cada fold (aprendido do treino do fold)

| Parâmetro | Origem | O que é | Risco se fixado |
|---|---|---|---|
| `CATEGORIAS_MANTIDAS` | Etapa 03 | níveis de `property_type` com ≥1% de frequência | vazamento: o corte veria dados de teste |
| `medianas_mcar` | Etapa 06 | 4 colunas: `bathrooms`, `bedrooms`, `host_experiencia_dias`, `calendar_atualizado_dias` | imputação com estatística do teste |
| `medianas_mar` | Etapa 06 | 5 colunas do bloco de reviews | idem |
| `mapas_ordinal` | Etapa 06 | códigos de `room_type` (3) e `neighbourhood_cleansed` (22) | códigos inconsistentes entre folds |

### 2.2 NÃO recalcula (constantes, não aprendidas)

`DATA_REF` (2018-12-06) · `POLOS` geográficos (Dam Square, Museumplein) ·
`FLAGS_AMENITIES` (11 flags) · bins de `pd.cut` · `clip(upper=365)` ·
`PARAMS_LGBM_TUNED` (10 hiperparâmetros da Etapa 08) · as 4 regras de remoção de linha.

Recalcular essas seria inofensivo mas inútil — o resultado é idêntico.

### 2.3 `robust_center` / `robust_scale` — não se aplica

O `RobustScaler` existe apenas no `df06_linear`. O modelo final é LightGBM sobre
`df06_arvore`, que **não usa escala** (árvores são invariantes a transformação monótona).
Fora do escopo deste plano.

---

## 3. Arquitetura — por que a limpeza fica fora do pipeline

**Decisão do usuário: Opção A.**

O `Pipeline` do sklearn exige que `transform` **preserve o número de linhas**. Mas as Etapas
02 e 03 removem linhas:

| Regra | Origem |
|---|---|
| `bathrooms > 10` | Etapa 02, cél. 30 |
| `minimum_nights > 365` | Etapa 02, cél. 30 |
| `maximum_nights == 9999` | Etapa 02, cél. 43 |
| `price <= 0` | Etapa 02, cél. 44 |

Juntas removem **49 linhas** de 20.030 (0,24%).

**Por que aplicá-las antes dos folds não causa vazamento:** as quatro são **constantes
fixas**, não parâmetros estimados. Nenhuma depende da distribuição dos dados — são regras de
sanidade (100,5 banheiros é impossível; 1.001 noites mínimas não faz sentido; `price = 0`
inviabiliza o MAPE). Aplicá-las à base inteira antes de particionar é equivalente a
aplicá-las dentro de cada fold.

### 3.1 A divisão de responsabilidades

```
FORA do pipeline (uma vez, sobre a base inteira):
  ├─ carga de listings_details.csv (20.030)
  ├─ Etapa 02: parse monetário, flags f/t, drops de coluna, 4 remoções de linha
  └─ Etapa 03: as 46 features — todas DETERMINÍSTICAS POR LINHA
       (exceção: property_type_agrupado, que vai para dentro do pipeline)
  → base_limpa: 19.981 × (features + id + price)

DENTRO do pipeline (refeito a cada fold):
  ├─ property_type_agrupado        ← CATEGORIAS_MANTIDAS do treino do fold
  ├─ imputação MNAR → 0            (constante, mas fica junto por coesão)
  ├─ imputação MAR → sentinela −1   (constante)
  ├─ imputação MCAR → mediana       ← aprendida no treino do fold
  ├─ encoding ordinal               ← mapas aprendidos no treino do fold
  └─ LGBMRegressor(**PARAMS_LGBM_TUNED)
```

**Verificação obrigatória:** confirmar que as 46 features da Etapa 03 são de fato
determinísticas por linha. A auditoria da Etapa 09 já estabeleceu isso — haversine, parse de
amenities, razões estruturais, `len()` de texto, diferenças de data. `property_type_agrupado`
é a única exceção, e por isso migra para dentro do pipeline.

---

## 4. Premissas e seus custos

### 4.1 As 27 features ficam fixas (decisão do usuário)

O purismo exigiria rodar o RFECV dentro de cada fold — ele também aprende dos dados.
Rejeitado por custo: a Etapa 05 levou ~70 min de seletores, e ×5 folds seria ~6 horas.

**O que sustenta a decisão:** o teste de estabilidade da Etapa 05 mediu o RFECV em 3 seeds
(42, 7, 2024) e obteve **N_CORTE = 26 nas três**, com as 26 features no núcleo estável e
índice de Jaccard de 0,742. A seleção não é artefato de uma partição.

**O custo, declarado:** `FEATURES_05` foi escolhido olhando 17.978 anúncios que, nos folds,
estarão parcialmente no conjunto de teste. Há um **viés otimista residual** — pequeno, porque
a seleção é estável, mas real. Vai registrado nas premissas, não escondido.

### 4.2 Os hiperparâmetros ficam fixos

`PARAMS_LGBM_TUNED` veio de 120 trials de Optuna sobre o treino da Etapa 08 (~2h). Mesmo
raciocínio: tunar por fold custaria ~10h e o ganho seria marginal.

Mesmo viés residual, mesma declaração.

### 4.3 O MAPE da CV não é comparável com os 21,93%

Cada fold treina em **~15.985** linhas contra 17.978 do modelo final — 11% menos dados.
Menos treino tende a dar MAPE ligeiramente pior.

São perguntas diferentes:

| Métrica | Pergunta que responde |
|---|---|
| MAPE do hold-out (21,93%) | qual o erro **deste modelo** em dados novos? |
| **MAPE médio da CV ± desvio** | qual o erro **esperado** de um modelo assim, e com quanta variação? |

A segunda é a que importa para precificar a base, porque traz o intervalo de confiança que
hoje falta — e é por isso que ela passa a definir a banda (§6).

### 4.4 O modelo de produção não é nenhum dos 5

Os 5 modelos dos folds servem para **gerar `price_estimate` honesto e medir o erro**. Para
precificar um anúncio **novo**, usa-se um modelo treinado na base completa — e o MAPE médio
da CV é a estimativa honesta do erro dele.

Isso precisa estar explícito no relatório, senão a pergunta "qual dos cinco está em
produção?" fica sem resposta.

---

## 5. O recomendador reconstruído sobre a base completa

**Decisão do usuário: `price_estimate` substitui `preco_justo`.** É a mesma coluna
conceitual — o preço justo estimado pelo modelo — apenas renomeada. Não há mudança de
semântica, só de origem: antes vinha do hold-out (2.003 anúncios), agora vem da CV (19.981).

### 5.1 A tabela base

| Coluna | Conteúdo |
|---|---|
| `id` | identificação do anúncio |
| `price` | preço real praticado |
| **`price_estimate`** | **preço estimado out-of-sample** (ex-`preco_justo`) |
| `fold` | rastreabilidade — qual modelo gerou a predição |
| `erro_pct` | `(price − price_estimate) / price` |

### 5.2 O recomendador completo

Sobre essa base, aplicam-se as mesmas regras travadas da Etapa 09 (CLAUDE.md §3.2, §3.3):

```
L = price_estimate × (1 − MAPE_CV)      # piso da banda
U = price_estimate × (1 + MAPE_CV)      # teto da banda
```

| Condição | Zona | Ação |
|---|---|---|
| `price < L` e `price ≥ €75` | **subprecificado** | recomenda subir até `L` |
| `L ≤ price ≤ U` | zona morta | não age |
| `price > U` | superprecificado | apenas sinaliza (§3.3) |
| `price < L` e `price < €75` | abaixo do piso | diagnóstico, sem ação |

O **piso de €75** da Etapa 09 é mantido: o MAPE no decil mais barato era 37,5% com viés de
−29,2%, e nada neste plano corrige isso.

### 5.3 O ganho concreto

| | Etapa 09/10 | Com este plano |
|---|---|---|
| Base do recomendador | 2.003 (hold-out) | **19.981 (toda a base)** |
| EV | extrapolado ×10 (€351.044) | **medido diretamente** |
| Predições contaminadas | 17.978 in-sample | **nenhuma** |
| Medições de MAPE | 1 | **5, com desvio** |

O EV deixa de depender de uma premissa de representatividade e passa a ser contado anúncio
por anúncio.

---

## 6. A banda passa a usar o MAPE médio da CV

**Decisão do usuário.** A banda de confiança deixa de ser ±21,93% (hold-out) e passa a
±`MAPE_CV_MEDIO`.

### 6.1 Por que isso é coerente

Cada `price_estimate` vem de um dos 5 modelos dos folds. O erro esperado dessas predições é
exatamente o que a CV mede — e não o erro do modelo final da Etapa 09, que foi treinado em
mais dados e avaliado em outro conjunto. Usar o MAPE da CV alinha a banda com a origem real
das predições.

Além disso, a CV entrega o que uma medição única não pode: o **desvio entre folds**. Se os 5
folds derem 21,9% / 22,4% / 21,1% / 23,0% / 22,2%, sabemos que a estimativa é estável. Se
derem 18% a 27%, a banda de ponto único era frágil e isso precisa aparecer.

### 6.2 Consequências a registrar

**A banda provavelmente ficará ligeiramente mais larga.** Cada fold treina com 11% menos
dados, então o MAPE médio da CV deve ser igual ou um pouco maior que 21,93%. Banda mais larga
→ zona morta maior → **menos subprecificados** → EV menor. Direção esperada, magnitude a
medir.

**O número reportado ao negócio muda de origem.** Qualquer material que cite "±21,93%" deve
ser atualizado. A Etapa 11 (Streamlit) deve exibir a faixa de confiança baseada no MAPE da
CV, não no do hold-out.

**O hold-out perde o papel de fonte da banda, mas não de validação.** Os 21,93% continuam
sendo a evidência mais limpa de que o pipeline não sobreajustou — medidos num conjunto que
nunca participou de nenhuma decisão. Ele passa a servir como **teste de sanidade** da CV
(§8): se o MAPE médio divergir mais de 2 p.p. dos 21,93%, há erro na reimplementação.

### 6.3 Variante a reportar, não a adotar

Vale calcular — e **apenas reportar** — o EV com banda de `MAPE_CV_MEDIO + 1 desvio`, como
cenário conservador. Banda mais larga significa menos recomendações, todas com margem maior
de segurança. Não é a configuração padrão, mas dá ao negócio a noção de quanto o EV se move
com a incerteza da própria estimativa de erro.

---

## 7. Ordem de execução — 8 portões

1. **Preparação fora do pipeline** — Etapas 02 e 03 sobre a base completa (19.981 linhas),
   com verificação de que as features são determinísticas por linha
2. **Construção do pipeline** — transformers customizados para os 4 parâmetros de §2.1 +
   LightGBM
3. **Validação do pipeline num fold** — confirmar que reproduz aproximadamente os 21,93%
   antes de rodar os 5
4. **Loop dos 5 folds** — treino, transform do separado, predição, preenchimento de
   `price_estimate`, MAPE do fold
5. **Consolidação da tabela** — `PRECOS_ESTIMADOS`, MAPE médio ± desvio, MAPE por fold
6. **MAPE por decil sobre a base completa** — 19.981 predições honestas, não 2.003
7. **Recomendador reconstruído** (§5) com a banda do MAPE da CV (§6)
8. **EV sem extrapolação** + comparação com €351.044 (extrapolado) e €109.372 (contaminado)

Cada portão é uma pausa para validação, conforme a Regra de Trabalho nº 2 do CLAUDE.md.

---

## 8. Entregáveis

1. **`PRECOS_ESTIMADOS`** — `id`, `price`, `price_estimate`, `fold`, `erro_pct` para 19.981
   anúncios
2. **`MAPE_POR_FOLD`** — 5 valores + média + desvio; `MAPE_CV_MEDIO` é a nova banda
3. **`PIPELINE_CV`** — o objeto sklearn, reutilizável
4. **`RECOMENDADOR_COMPLETO`** — zonas, banda e uplift sobre 19.981 anúncios (§5.2)
5. **MAPE por decil** sobre a base completa, comparado com o do hold-out
6. **EV medido** (sem extrapolação), com as variantes: banda = MAPE médio, e banda = MAPE
   médio + 1σ (§6.3)
7. **`PREMISSAS_CV`** — features fixas, hiperparâmetros fixos, viés residual, origem da banda

---

## 9. Verificação

- **Cobertura total:** `price_estimate.isna().sum() == 0` e `id` único — todos os 19.981
  anúncios com predição
- **Partições disjuntas:** soma dos 5 conjuntos de teste = 19.981, sem interseção
- **Nenhuma predição in-sample:** para cada fold, `set(id_teste) ∩ set(id_treino) == ∅`
- **Parâmetros recalculados:** os 4 itens de §2.1 têm valores **diferentes** entre pelo menos
  dois folds (se idênticos nos 5, provavelmente estão sendo fixados por engano)
- **MAPE em escala original:** `expm1` antes do cálculo; `assert 0 < mape < 1` por fold
- **Gatilho de vazamento:** MAPE médio **> 19%**; abaixo disso, investigar antes de concluir
- **Sanidade vs hold-out:** MAPE médio da CV dentro de **±2 p.p.** dos 21,93%. Divergência
  maior indica erro na reimplementação do pipeline, não propriedade dos dados
- **Banda coerente:** `L < U` em todas as linhas; banda construída com `MAPE_CV_MEDIO`
- **Zonas exaustivas e exclusivas:** cada anúncio em exatamente uma zona
- **Só subprecificados geram uplift** (CLAUDE.md §3.3); abaixo do piso de €75 não entra no EV
- **`price > 0`** em todas as linhas (garantido pela limpeza)

---

## 10. Riscos da implementação

| Risco | Mitigação |
|---|---|
| **Divergência entre o pipeline e as células originais** — reimplementar as Etapas 02/03/06 em transformers pode introduzir diferenças silenciosas | Portão 3: validar um fold contra os 21,93% antes de rodar os 5. Divergência > 2 p.p. = parar e auditar |
| **Categoria nunca vista no treino do fold** — com 22 bairros e 20% de teste, um bairro raro pode faltar | Mapa ordinal devolve **−1** para desconhecida (regra da Etapa 09). Contar ocorrências por fold |
| **`CATEGORIAS_MANTIDAS` instável entre folds** | Registrar os níveis de cada fold. A Etapa 09 mediu que recalcular no hold-out dava os mesmos 8 níveis — variação deve ser nula ou mínima |
| **Banda mais larga reduz o EV** (§6.2) | Esperado e correto. Reportar o delta vs a banda de 21,93% para que a mudança seja rastreável, não silenciosa |
| **Tempo de execução** — 5 fits de LightGBM com 900 árvores | ~17 s por fit na Etapa 09; o loop deve ficar em poucos minutos |

---

## 11. Pendências

**Resolvidas antes da execução** (validadas pelo usuário):

1. **k=5 com 20%** de teste por fold — cobertura de 100% (§1)
2. **27 features fixas**, com o viés residual declarado (§4.1)
3. **Opção A** — limpeza fora, pipeline dentro (§3)
4. **`price_estimate` substitui `preco_justo`** — mesma coluna, renomeada; o recomendador é
   reconstruído sobre os 19.981 (§5)
5. **A banda usa `MAPE_CV_MEDIO`**, não os 21,93% do hold-out (§6)

**Em aberto:**

1. **Retreinar com dados atuais** — o `price` é uma foto de dezembro de 2018, e Amsterdam
   restringiu aluguel de curta duração depois dessa safra. Pendência herdada da Etapa 10,
   anterior a qualquer uso real.
2. **O viés residual de E09-6** (−29,2% no D1) segue contornado pelo piso de €75, não
   corrigido. Este plano não o resolve — apenas dá uma base maior para medi-lo.
3. **`avg_length_of_stay = 3`** nunca foi conferido para Amsterdam (nota da Etapa 01 que
   atravessou dez etapas). Vale ±30% no EV.
