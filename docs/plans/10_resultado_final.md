# Plano — Etapa 10: Resultado Final (Airbnb Amsterdam)

## Contexto

Esta etapa fecha o círculo do projeto. A Etapa 01 estimou quanto a solução **poderia** valer,
usando a mediana LOO dos comparáveis e um grid hipotético de MAPE. Agora temos o modelo
real, o MAPE real e as predições reais — e a pergunta é direta:

> **O retorno projetado no início se confirma?**

O CLAUDE.md §7 (Etapa 10) pede: re-rodar a Etapa 01 substituindo a mediana LOO pelas
predições do modelo e usando o **MAPE REAL** do teste, comparar o retorno realizado com a
estimativa inicial e reportar o **delta**. Mesma regra da Etapa 01: **código inline, sem
funções** (Regra de Trabalho nº 4).

### Entrada

| Artefato | Conteúdo |
|---|---|
| `MAPE_TESTE` | **21,93%** — o MAPE real do hold-out |
| `RECOMENDADOR` | 2.003 linhas: `id`, `price`, `preco_justo`, `L`, `U`, `zona`, `uplift` |
| `PREMISSAS` | take_rate 0,15 · review_rate 0,50 · estadia 3 · cap 0,70 · n_min 30 |
| `PREMISSAS_09` | piso de €75, rota de tratamento de E09-6 |
| `df_ev_mape` | a curva `EV(MAPE)` da Etapa 01 (8 pontos do grid) |
| `df_cenarios` | otimista 10% · base 20% · pessimista 30% |
| `dte` | hold-out processado, com `reviews_per_month` para o modelo de ocupação |

### A referência da Etapa 01

| MAPE | Subprecificados | Uplift médio | EV |
|---|---|---|---|
| 10% | 7.354 | €24,53 | €1.498.725 |
| **20%** | **4.795** | **€20,00** | **€785.808** |
| 25% | 3.592 | €18,52 | €540.326 |
| 30% | 2.730 | €16,15 | €354.944 |

O MAPE real (21,93%) cai **entre o cenário base (20%) e o pessimista (30%)**, mais próximo
do base. Por interpolação da curva, a expectativa de EV fica em torno de €690–700 mil.

---

## 1. O problema de escala — e como resolvê-lo

Este é o ponto que exige decisão antes de escrever qualquer código.

**A Etapa 01 rodou sobre a base inteira: 20.030 anúncios.** A Etapa 10 só pode usar o
hold-out: **2.003 anúncios — exatamente 10,0% da base.**

Comparar €785.808 (Etapa 01, 20.030 anúncios) com o EV calculado sobre 2.003 anúncios seria
comparar grandezas de escalas diferentes e concluir, falsamente, que o projeto perdeu 90%
do valor.

### 1.1 Proposta: reportar os três números, sempre juntos

| Número | Base | Papel |
|---|---|---|
| **EV observado** | 2.003 anúncios do hold-out | o que foi **medido**, sem extrapolação |
| **EV extrapolado** | × 10 (hold-out → base) | comparável com a Etapa 01 |
| **EV da Etapa 01** | 20.030 anúncios | a estimativa inicial |

O **delta** que o CLAUDE.md pede é entre o *extrapolado* e o da Etapa 01 — são as duas
grandezas na mesma escala.

### 1.2 A premissa da extrapolação, declarada

Multiplicar por 10 assume que **o hold-out é representativo da base**. Isso é defensável:
foi separado aleatoriamente com `random_state=42` sobre os dados brutos (CLAUDE.md §5.1), e
a Etapa 09 confirmou que nenhuma linha foi perdida na reaplicação.

Mas é uma premissa, não um fato, e vai registrada como tal. Um teste de sanidade: comparar a
mediana de `price` e a distribuição de `room_type` entre hold-out e treino. Se divergirem
materialmente, a extrapolação ganha uma ressalva.

**Alternativa considerada e descartada:** rodar o recomendador sobre a base inteira
(20.030). Daria um EV diretamente comparável, mas **17.978 desses anúncios foram vistos no
treino** — as predições neles são otimistas por construção, e o EV resultante seria
inflado. O hold-out existe justamente para evitar isso.

---

## 2. Duas mudanças metodológicas em relação à Etapa 01

A Etapa 01 e a Etapa 10 **não calculam a mesma coisa**, e as diferenças precisam estar
explícitas no relatório — senão o delta é lido como "o modelo melhorou/piorou" quando parte
dele vem de mudança de método.

### 2.1 O preço de referência muda de origem

| | Etapa 01 | Etapa 10 |
|---|---|---|
| `ŷᵢ` | mediana LOO dos comparáveis + fallback hierárquico | **predição do LightGBM tunado** |
| MAPE | grid hipotético (5% a 40%) | **21,93%**, medido no hold-out |
| Winsorize de `price` | p1/p99 antes do cálculo | **não se aplica** — o modelo já foi treinado em dados limpos |

A mediana LOO usava `GROUP_COLS` (3 atributos). O modelo usa 27 features. São estimadores
diferentes do mesmo conceito ("preço justo"), e é exatamente essa substituição que a etapa
mede.

### 2.2 O piso de €75 (novo, vindo da Etapa 09)

A Etapa 01 não tinha piso: todo anúncio subprecificado gerava EV. A Etapa 09 introduziu um
**piso de €75** para tratar E09-6 (o modelo prevê ~29% acima do real no decil mais barato).

Consequência: **93 anúncios do hold-out (27% dos subprecificados) não geram EV**, por
decisão deliberada. Isso reduz o EV medido — e é uma redução *correta*, porque remove
recomendações que o modelo não sustenta.

**Proposta:** reportar o EV **com e sem** o piso. A diferença quantifica o custo da decisão
conservadora da Etapa 09, e deixa o leitor julgar o trade-off.

---

## 3. O modelo de ocupação (CLAUDE.md §4) — reaplicado ao hold-out

Mesma fórmula, mesmos parâmetros da Etapa 01:

```
bookings_per_month = reviews_per_month / 0.50
nights_per_month   = bookings_per_month * 3
noites_reservadas  = min(nights_per_month * 12, 365 * 0.70)
```

**Cuidado obrigatório:** `reviews_per_month` ausente → `noites_reservadas = 0` (conservador,
não gera EV). No hold-out são **244 anúncios (12,2%)** nessa condição — número a registrar,
como a Etapa 01 fez.

Note a interação com o piso: os sem-review tendem a ser anúncios novos, e a EDA mostrou que
eles cobram mais (€139 vs €125). Não há sobreposição óbvia com o piso de €75, mas vale medir
quantos anúncios são excluídos do EV por **cada** motivo, e quantos por ambos.

---

## 4. A fórmula do EV (CLAUDE.md §3.4)

```
EV = take_rate × Σ_{i : zona = subprecificado}  uplift_i × noites_reservadas_i
```

Com `take_rate = 0,15`, `uplift_i = Lᵢ − pᵢ` (já calculado em `RECOMENDADOR`), e a soma
restrita aos **subprecificados elegíveis** — superprecificados não entram (§3.3), e os
abaixo do piso não entram (Etapa 09).

`taxa_adocao` e `desconto_elasticidade` seguem em 1,0, como na Etapa 01. Se forem alterados,
entram nas premissas — mas a proposta é mantê-los, para que o delta isole o efeito do
modelo.

---

## 5. Entregável central: a tabela de delta

| Métrica | Etapa 01 (estimado) | Etapa 10 (realizado) | Delta |
|---|---|---|---|
| MAPE | 20% (cenário base) | **21,93%** | +1,93 p.p. |
| Origem de `ŷ` | mediana LOO | LightGBM tunado | — |
| Subprecificados | 4.795 (24,0%) | a medir × 10 | — |
| Uplift médio | €20,00 | a medir | — |
| **EV** | **€785.808** | a medir × 10 | **a medir** |

Mais a **curva `EV(MAPE)` da Etapa 01 com o ponto real marcado** — o gráfico que mostra
onde o projeto efetivamente pousou na curva que ele próprio projetou.

### 5.1 O que esperar

O MAPE real (21,93%) é **pior** que o cenário base (20%), então a expectativa é um EV
**abaixo** de €785.808 por esse efeito. Mas três fatores empurram em direções diferentes:

| Fator | Direção no EV |
|---|---|
| MAPE 21,93% vs 20% (banda mais larga) | ↓ |
| Piso de €75 (93 anúncios excluídos) | ↓ |
| Modelo com 27 features vs mediana de 3 | **?** — pode ir para qualquer lado |

O terceiro é a incógnita real. Um estimador melhor de preço justo identifica a
subprecificação com mais precisão — pode achar *mais* casos verdadeiros ou *menos* falsos
positivos. **O plano não prevê o sinal**, e é isso que a etapa descobre.

---

## 6. Análise de sensibilidade (obrigatória, CLAUDE.md §7)

A Etapa 01 exigiu sensibilidade; a Etapa 10 repete com os números reais. Variar um
parâmetro por vez, mantendo os outros:

| Parâmetro | Faixa | Por quê |
|---|---|---|
| `take_rate` | 0,10 / 0,15 / 0,20 | o EV é linear nele — fator direto |
| `avg_length_of_stay` | 2 / 3 / 4 | a Etapa 01 marcou "CONFERIR p/ Amsterdam" |
| `occupancy_cap` | 0,60 / 0,70 / 0,80 | teto do modelo InsideAirbnb |
| `taxa_adocao` | 0,25 / 0,50 / 1,00 | nem todo anfitrião aceita a recomendação |
| **`piso_recomendacao`** | 0 / 75 / 100 | **novo** — quantifica a decisão da Etapa 09 |

O último é o mais informativo para o negócio: mostra em euros o que custa a cautela com o
decil barato.

---

## 7. Ordem de execução — 6 portões

1. **Representatividade do hold-out** — comparar `price` e `room_type` contra o treino, para
   sustentar (ou ressalvar) a extrapolação de §1.2
2. **Modelo de ocupação** reaplicado ao hold-out, com a contagem de sem-review
3. **EV realizado** — com e sem o piso de €75, observado e extrapolado
4. **Tabela de delta** vs Etapa 01 + curva `EV(MAPE)` com o ponto real marcado
5. **Análise de sensibilidade** — 5 parâmetros
6. **Consolidação** — premissas, ressalvas e o fechamento do projeto

**Código inline, sem funções** (CLAUDE.md, Regra nº 4). Cada portão é uma pausa para
validação — são **6 portões**, não 1.

---

## 8. Entregáveis

1. **`EV_REALIZADO`** — observado (2.003) e extrapolado (×10), com e sem o piso
2. **`DELTA_EV`** — tabela comparativa com a Etapa 01, item por item
3. **Curva `EV(MAPE)`** da Etapa 01 com o ponto real (21,93%) marcado
4. **`SENSIBILIDADE_10`** — EV por variação dos 5 parâmetros
5. **Contagem de exclusões** — quantos anúncios não geram EV por sem-review, por piso, e
   por ambos
6. **`PREMISSAS_10`** — todas as premissas, incluindo a de extrapolação, explicitadas
7. **Fechamento do projeto** — o que o MAPE de 21,93% significa em euros, e as três
   diferenças metodológicas vs a Etapa 01

---

## 9. Verificação

- **Código inline:** nenhuma `def` nas células desta etapa (CLAUDE.md Regra nº 4)
- **Só subprecificados geram EV:** soma restrita à zona `subprecificado` (§3.3)
- **Piso respeitado:** `subprecificado_abaixo_do_piso` não entra no EV
- **Sem-review não gera EV:** `noites_reservadas == 0` onde `reviews_per_month` é nulo
- **Escala declarada:** todo número de EV vem rotulado como *observado* ou *extrapolado* —
  nunca um valor solto
- **Premissas idênticas às da Etapa 01** (take_rate, review_rate, estadia, cap), para que o
  delta isole o efeito do modelo; qualquer mudança vai registrada
- **MAPE da banda = 21,93%** (o real do hold-out), não o da validação
- **Reprodutibilidade:** o EV recalculado a partir de `RECOMENDADOR` bate com o uplift total
  de €4.392/diária já reportado na Etapa 09

---

## 10. Pendências

**Em aberto, a decidir antes ou durante a execução:**

1. **A extrapolação ×10** é a abordagem proposta (§1.1). Se preferir reportar **apenas** o EV
   observado sobre 2.003 anúncios, sem extrapolar, o delta vs a Etapa 01 deixa de ser
   calculável na mesma escala — e o CLAUDE.md pede esse delta. Confirmar a abordagem.
2. **`avg_length_of_stay = 3`** — a Etapa 01 registrou "CONFERIR o valor configurado p/
   Amsterdam" e isso nunca foi conferido. O InsideAirbnb usa valores por cidade. Fica na
   sensibilidade (2/3/4), mas o número correto mudaria o EV proporcionalmente.
3. **`taxa_adocao = 1,0`** assume que 100% dos anfitriões aceitam a recomendação — otimista
   para um número de negócio. A sensibilidade cobre 0,25 e 0,50; vale discutir qual usar
   como cenário principal no fechamento.
4. **O viés residual de E09-6** foi contornado pelo piso, não corrigido. Se a Etapa 11
   expuser o recomendador a anúncios abaixo de €75, o tratamento precisa ser revisto.
