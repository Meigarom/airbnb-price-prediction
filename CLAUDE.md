# CLAUDE.md — Projeto de Precificação de Diárias (Airbnb Amsterdam · Host Growth)

Este arquivo é a fonte de verdade para o Claude Code executar o projeto. Leia por
inteiro antes de qualquer ação. As **Regras de Trabalho (Seção 2)** têm prioridade
sobre qualquer conveniência de implementação.

---

## 1. Contexto e objetivo

O faturamento do Airbnb é uma fração de cada reserva (`take_rate × diária × noites`).
Quando o anfitrião precifica errado, ele e a plataforma perdem dinheiro:
subprecificar deixa receita na mesa, superprecificar encalha o anúncio. O projeto
trata o preço da diária como um **problema de regressão** e usa a métrica **MAPE**
como meta de desenvolvimento do modelo.

O objetivo central, e o que viabiliza o projeto, é **conectar o MAPE a um retorno
financeiro esperado (EV)**: qual valor a solução gera em cada nível de erro do
modelo. Esse cálculo é a Etapa 01 e é re-executado com números reais na Etapa 10.

**Dataset:** InsideAirbnb — Amsterdam. Preços em **euro** (o campo `price` vem como
texto com `"$"` por herança do schema; os valores são locais, em euro).

**Base principal:** `listings_details.csv` (~90 colunas). **Não usar** `listings.csv`
(resumido, sem tamanho do imóvel). Demais arquivos: `calendar.csv`, `review.csv`,
`review_details.csv`, `neighbourhood.csv`, `neighbourhood.geojson`.

---

## 2. Regras de trabalho (COMO você deve operar) — PRIORIDADE MÁXIMA

1. **Uma etapa por vez, com portão de validação.** Execute as etapas da Seção 7 em
   ordem. Ao fim de CADA etapa, PARE, apresente o resultado e peça minha validação
   explícita antes de começar a próxima. Não emende etapas.

2. **Desenvolvimento incremental dentro da etapa.** Não escreva a etapa inteira de
   uma vez. Adicione o código ao notebook **em blocos pequenos (uma célula ou poucas
   células por vez)**, execute, me mostre a saída e só então avance para o próximo
   bloco. Eu preciso acompanhar o desenvolvimento passo a passo.

3. **Edição e execução via MCP.** Toda manipulação de célula (criar, editar, executar,
   ler saída) deve ser feita pelo MCP de Jupyter. **Se não houver MCP instalado, NÃO
   improvise** editando o `.ipynb` como JSON: pare e me entregue o checklist de
   instalação da Seção 6, apontando exatamente o que eu preciso fazer.

4. **Etapa 01 sem funções — código inline nas células.** No cálculo do potencial de
   retorno financeiro (Etapa 01 e a re-execução na Etapa 10), **não crie funções
   Python** (`def`/helpers). Escreva toda a lógica de forma linear, dentro das
   células do notebook indicado, para facilitar a auditoria célula a célula.
   Idiomas de pandas com `lambda` inline são aceitos; o que não quero é lógica de
   negócio abstraída em funções. Nas demais etapas (02–11), funções são permitidas e
   bem-vindas.

5. **Idioma e unidades.** Comentários, markdown e prints em **português (BR)**.
    Nomes e variáveis em inglês. Moeda em **euro (€)**. Nada de reintroduzir `"$"`.

6. **Registro de premissas.** Todo número financeiro depende de premissas
   (take rate, ocupação, grid de MAPE). Mantenha um dicionário/tabela de premissas
   **visível no topo da Etapa 01** e reutilize-o — nunca deixe uma premissa
   "escondida" no meio de uma conta.

---

## 3. Estratégia de negócio travada (metodologia do retorno financeiro)

Estas decisões estão fechadas. **Não as reabra sem me perguntar.**

**3.1 Preço de referência = mediana Leave-One-Out dos comparáveis.**
Para cada anúncio *i*, o preço justo `ŷᵢ` é a **mediana** do `price` dos anúncios do
mesmo grupo `(neighbourhood_cleansed, room_type, accommodates)`, **excluindo o próprio
anúncio** (Leave-One-Out), para o anúncio não votar no próprio preço.

- Antes de calcular, faça **clip/winsorize** de `price` nos extremos (ex.: p1 e p99)
  para remover €5 e €9.000.
- **Fallback hierárquico** para células com menos de `n_min` anúncios (default
  `n_min = 30`): suba de nível na ordem
  `(bairro, room_type, accommodates)` → `(bairro, room_type)` →
  `(room_type, accommodates)` → `(room_type)` → cidade inteira.
  Guarde de qual nível veio a referência (indicador de confiança).
- Implementação LOO exata e inline (exemplo, adaptável):
  ```python
  df['ref_price'] = df.groupby(GROUP_COLS)['price'].transform(
      lambda s: pd.Series(
          [s.drop(idx).median() for idx in s.index], index=s.index
      )
  )
  ```
  Para grupos acima de `n_min`, a mediana LOO ≈ mediana do grupo; a correção LOO
  importa principalmente nas células pequenas/fallback.

**3.2 MAPE define os limites de variação do preço estimado.**
A referência vira uma banda: `Lᵢ = ŷᵢ·(1 − MAPE)` (piso) e `Uᵢ = ŷᵢ·(1 + MAPE)`
(teto). Zonas:
- `pᵢ < Lᵢ` → **subprecificado** → recomenda subir até `Lᵢ` (ponto conservador).
- `Lᵢ ≤ pᵢ ≤ Uᵢ` → **zona morta** → não age (o desvio cabe dentro do erro).
- `pᵢ > Uᵢ` → **superprecificado** → apenas **sinalizado**, sem ação.

**3.3 Só atuamos na subprecificação.**
Superprecificados são identificados e contados (para diagnóstico), mas **não geram
recomendação nem entram no EV**, porque o valor de baixar preço depende de ocupação,
e não temos ocupação observada confiável. Ficamos apenas com o vetor defensável.

**3.4 Fórmula do retorno (EV).**
```
EV(MAPE) = take_rate × Σ_{i : pᵢ < Lᵢ}  (Lᵢ − pᵢ) × noites_reservadas_i
```
- `noites_reservadas_i` vem do **modelo de ocupação do InsideAirbnb** (Seção 4).
- Opcionalmente aplique `taxa_de_adoção` e `desconto_elasticidade ∈ (0,1]` como
  fatores de realismo; se aplicar, registre-os nas premissas.

**3.5 Efeito do MAPE (por que a curva é monótona decrescente).**
MAPE menor → banda mais estreita → (a) mais anúncios saem da zona morta (cobertura↑)
e (b) recomendação mais próxima da referência (magnitude↑). Ambos empurram o EV para
cima. Em `MAPE = 0` captura-se o teto teórico; o MAPE corrói esse teto até zerar.

---

## 4. Modelo de ocupação do InsideAirbnb (Tarefa obrigatória)

Substitua qualquer premissa fixa de ocupação pela estimativa do **modelo do
InsideAirbnb** ("San Francisco Model"). Estime as **noites reservadas por ano** por
anúncio assim:

```
review_rate         = 0.50    # ~50% dos hóspedes deixam review (padrão InsideAirbnb)
avg_length_of_stay  = 3       # noites por reserva; CONFERIR o valor configurado p/ Amsterdam
occupancy_cap       = 0.70    # teto de ocupação do modelo InsideAirbnb

bookings_per_month  = reviews_per_month / review_rate
nights_per_month    = bookings_per_month * avg_length_of_stay
noites_reservadas   = min(nights_per_month * 12, 365 * occupancy_cap)
```

Cuidados:
- `reviews_per_month` ausente/NaN (anúncio sem reviews) → tratar como ocupação
  desconhecida. Default: `noites_reservadas = 0` (conservador, não gera EV). Registre
  quantos anúncios caem nesse caso.
- Todos os parâmetros (`review_rate`, `avg_length_of_stay`, `occupancy_cap`) entram no
  registro de premissas e na análise de sensibilidade.
- **Não** use `365 − availability_365` como ocupação — `availability_365` é
  disponibilidade no calendário, não ocupação.

---

## 5. Separação treino/teste travada (hold-out)

Esta decisão está fechada e atravessa todas as etapas. **Não a reabra sem me perguntar.**

**5.1 O teste é separado sobre os dados BRUTOS, antes de qualquer tratamento.**
10% de `listings_details.csv` são isolados **imediatamente após a carga**, antes de
qualquer limpeza, imputação, remoção de linha ou feature engineering.

```python
RANDOM_STATE = 42
TEST_SIZE = 0.10
df_train, df_test = train_test_split(
    df_details, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True,
)
```

| Conjunto | Linhas | Arquivo |
|---|---|---|
| Original | 20.030 | `data/listings_details.csv` |
| Treino | **18.027** | `data/processed/train_raw.csv` |
| Teste (hold-out) | **2.003** | `data/processed/test_raw.csv` |

**5.2 Por que antes da limpeza, e não depois.**
Separar sobre os dados brutos garante **isolamento total**. Se a separação acontecesse
depois da limpeza, toda decisão tomada olhando o dataset completo — quais colunas dropar,
onde cortar outliers, qual mediana imputar, quais features criar — teria sido informada
também pelas linhas de teste. É vazamento sutil: não aparece em nenhuma coluna, mas infla
a métrica final sem deixar rastro. A avaliação da Etapa 09 só é honesta se o teste **nunca
participou de nenhuma decisão**, nem estatística nem analítica.

**5.3 Regra de uso — o teste fica congelado até a Etapa 09.**
- Das Etapas 02 a 08, **todo o pipeline opera exclusivamente sobre `df_train`**.
- Nenhuma célula dessas etapas lê `df_test` ou `test_raw.csv` — nem para "dar uma olhada",
  nem para conferir distribuição, nem para EDA. Olhar o teste na EDA contamina as decisões
  de seleção (Etapa 05) e preparação (Etapa 06) tanto quanto usá-lo no `fit`.
- O teste é aberto **uma única vez**, na Etapa 09, para a avaliação final.

**5.4 Reaplicação do pipeline ao teste (obrigatória na Etapa 09).**
`test_raw.csv` está **cru**: tem as 96 colunas originais, sem limpeza e sem features.
Antes da avaliação, toda a limpeza da Etapa 02 e todo o feature engineering da Etapa 03
precisam ser reaplicados a ele **na mesma ordem e com os mesmos parâmetros** do treino.

Consequência prática para a Etapa 06: qualquer transformação que aprenda parâmetros dos
dados (encoder, scaler, imputação) faz `fit` **só no treino** e `transform` no teste —
nunca `fit_transform` no teste. Divergência de tratamento entre treino e teste invalida a
métrica final tanto quanto vazamento.

> Consolidar o pipeline numa função reutilizável é a forma natural de garantir que treino
> e teste recebam exatamente o mesmo tratamento, sem divergência por reimplementação.

---

## 6. Setup do MCP de Jupyter (fazer se não houver MCP)

No início da sessão, **verifique se há ferramentas de MCP de notebook disponíveis**
(criar/editar/executar células, ler kernel). Se sim, siga direto para a Etapa 01. Se
**não**, pare e me entregue este checklist — estas são as tarefas que EU (usuário)
preciso executar; aponte cada uma claramente:

Recomendado: **Datalayer `jupyter-mcp-server`** (conexão viva com o kernel do
JupyterLab; permite executar células, ler saídas e corrigir erros em loop).

**Tarefas do usuário (você, Claude Code, me instrui e aguarda):**
1. Criar um venv limpo para o projeto e ativá-lo.
2. Instalar as dependências:
   ```bash
   pip install jupyterlab jupyter-collaboration jupyter-mcp-server
   ```
   (a extensão `jupyter_collaboration` é necessária para editar/executar células via
   MCP.)
3. Subir o JupyterLab com um token fixo, na raiz do projeto:
   ```bash
   jupyter lab --port 8888 --IdentityProvider.token MEU_TOKEN
   ```
4. Deixar o notebook-alvo aberto no JupyterLab (para o kernel estar vivo).
5. Registrar o MCP no Claude Code criando `.mcp.json` na raiz do projeto:
   ```json
   {
     "mcpServers": {
       "jupyter": {
         "command": "jupyter-mcp-server",
         "env": {
           "SERVER_URL": "http://localhost:8888",
           "TOKEN": "MEU_TOKEN",
           "DOCUMENT_ID": "notebooks/01_retorno_financeiro.ipynb"
         }
       }
     }
   }
   ```
   (Ajuste `command`/variáveis conforme a documentação da versão instalada — confirme
   os nomes exatos das variáveis na doc do pacote antes de eu rodar.)
6. Reiniciar o Claude Code para carregar o MCP.

Depois que eu confirmar que o MCP subiu, valide a conexão listando/lendo o notebook
antes de escrever qualquer célula. Se o WebSocket cair (restart do JupyterLab),
reconecte antes de continuar.

---

## 7. Etapas do projeto (executar em ordem, uma por vez, com validação ao fim)

### Etapa 01 — Cálculo do Potencial de Retorno Financeiro
Aplicar a metodologia da Seção 3 + ocupação da Seção 4. **Código inline, sem funções.**
No `01_retorno_financeiro.ipynb`, de forma incremental:
1. Registro de premissas (take_rate, review_rate, avg_length_of_stay, occupancy_cap,
   grid de MAPE, GROUP_COLS, n_min, bounds de winsorize).
2. Carregar `listings_details.csv` e limpeza mínima **apenas** das colunas usadas aqui
   (`price` → float; `neighbourhood_cleansed`, `room_type`, `accommodates`,
   `reviews_per_month`).
3. Estimar `noites_reservadas` (modelo InsideAirbnb).
4. Calcular `ref_price` (mediana LOO + fallback).
5. **Varredura de MAPE**: para cada MAPE do grid, classificar zonas, isolar
   subprecificados, `uplift = Lᵢ − pᵢ`, `EV = take_rate × Σ(uplift × noites)`.
6. Entregáveis: **curva `EV(MAPE)`**, **3 cenários** (otimista / base / pessimista,
   ancorados em níveis plausíveis de MAPE), **registro de premissas** e **análise de
   sensibilidade** (variar take_rate, `avg_length_of_stay`, `occupancy_cap` e adoção,
   mostrando como o EV se move). Este é o número que viabiliza o projeto.

### Etapa 02 — Data Quality
**Primeiro de tudo: separar o hold-out de teste** (Seção 5) — logo após a carga de
`listings_details.csv`, antes de qualquer limpeza. Da separação em diante, tudo opera
sobre `df_train`.

Depois: estatística descritiva para qualidade dos dados — tipos, faltantes, ranges,
duplicatas, consistência. Limpar `price` (`$`→float), tratar as pegadinhas da safra
(`review_scores_rating` está em escala 0–100, não 0–5; `availability_365` não é
ocupação). Documentar cada decisão de limpeza.

### Etapa 03 — Feature Engineering
Gerar features a partir das existentes: densidade de amenities, idade do anúncio via
`host_since`, distância ao centro via `latitude`/`longitude`, preço por pessoa
(`price/accommodates`), flags (`host_is_superhost`, `instant_bookable`), etc.

### Etapa 04 — Análise Exploratória (EDA)
1) **Univariada:** faltantes, outliers, variabilidade, distribuição de cada variável.
2) **Bivariada:** relação de cada explicativa com o alvo `price`.
3) **Multivariada:** matriz de correlação entre explicativas para detectar
   autocorrelação.

### Etapa 05 — Feature Selection
Combinar **Boruta** + importância por árvores para escolher o conjunto final de
features. Reportar o racional da seleção.

### Etapa 06 — Data Preparation
Encoders para categóricas; **padronização** (standardization) para variáveis Normais;
**rescala** (min-max/robust) para não-Normais. **Sem vazamento**: `fit` só no treino,
`transform` em validação/teste.

### Etapa 07 — Machine Learning
Treinar e comparar por **MAPE** (com MAE/RMSE de apoio):
1) Baseline (média), 2) Linear Regression, 3) Regressão Polinomial, 4) Random Forest,
5) XGBoost, 6) LightGBM.

### Etapa 08 — Fine Tuning
**Optuna + Cross-Validation** no(s) melhor(es) modelo(s) da Etapa 07.

### Etapa 09 — Métricas e Recomendador
Retreinar o melhor modelo (melhores params do Optuna) com treino+validação juntos.
**Antes de avaliar:** reaplicar a `test_raw.csv` toda a limpeza da Etapa 02 e todo o
feature engineering da Etapa 03, na mesma ordem e com os mesmos parâmetros (Seção 5.4) —
o hold-out está cru. Avaliar no **teste**. Construir o recomendador que gera o **preço
justo por anúncio**.

### Etapa 10 — Resultado Final
Re-rodar a **Etapa 01** substituindo a mediana LOO pelas predições do modelo e usando
o **MAPE REAL** do teste. Comparar o retorno realizado com a estimativa inicial e
reportar o **delta**. Mesma regra da Etapa 01: **código inline, sem funções.**

### Etapa 11 — Apresentação Final
Página em **Streamlit** (`app/app.py`) expondo o modelo como produto de dados: o
usuário insere as características do imóvel e recebe o **preço justo recomendado** mais
a **faixa de confiança** (banda de MAPE).

---

## 8. Checklist antes de fechar cada etapa
- [ ] Código adicionado de forma incremental e executado via MCP.
- [ ] Saídas revisadas comigo.
- [ ] Premissas/decisões registradas no notebook.

Leia todas as informações, conclua os checklists e espere o próximo comando para iniciar.