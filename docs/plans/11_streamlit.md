# Plano — Etapa 11: Apresentação Final (Streamlit)

## Contexto

As dez etapas anteriores produziram um modelo com **MAPE de 22,21%** servido por uma
API na AWS, validada de ponta a ponta no portão 8. O que falta é a camada que o
CLAUDE.md descreve como **produto de dados**: uma página onde o usuário insere as
características do imóvel e recebe o preço justo recomendado com a faixa de confiança.

A diferença em relação a tudo que veio antes é o público. Até aqui o consumidor era
um cliente programático, que monta um JSON com 96 colunas. Agora é uma pessoa.

### Para que esta página serve, e para que não serve

É um **protótipo**, com dois propósitos declarados:

1. **Observar o deploy na AWS funcionando de forma visual** — ver o preço aparecer na
   tela é uma validação diferente de ler `{"price_estimate": 147.77}` num terminal.
2. **Mostrar o modelo às áreas de negócio** — dar a quem não lê JSON uma forma de
   experimentar o recomendador.

Não é um produto para anfitriões reais, não vai para o Streamlit Community Cloud e
**roda apenas localmente**. Isso não a dispensa de ser honesta: justamente porque
será mostrada a pessoas que decidem com base nela, os avisos sobre a banda de erro, o
piso de €75 e a safra de 2018 (§5) são mais importantes aqui, não menos.

---

## 1. Decisões travadas pelo usuário

| # | Decisão | Consequência |
|---|---|---|
| 1 | **Formulário com os 25 campos obrigatórios** | fiel ao contrato da API, sem suposição escondida |
| 2 | **A página chama a API na AWS** | exercita a infraestrutura real; exige credencial AWS local |

Ambas foram escolhidas sobre alternativas mais simples, e as duas têm um custo que
este plano registra em vez de disfarçar (§6).

---

## 2. O problema central: 25 campos, nem todos perguntáveis

`DataPreparation.validar_colunas` exige **25 colunas**. Elas não são homogêneas — e
tratar todas como "campo de formulário" produziria uma página inutilizável:

| Grupo | Campos | Como aparecem na interface |
|---|---|---|
| **O anfitrião sabe de cabeça** (9) | `accommodates`, `bedrooms`, `bathrooms`, `beds`, `room_type`, `property_type`, `neighbourhood_cleansed`, `minimum_nights`, `maximum_nights` | controles diretos |
| **O anfitrião sabe consultando** (4) | `cleaning_fee`, `security_deposit`, `extra_people`, `calculated_host_listings_count` | campos numéricos com default |
| **Histórico do anúncio** (5) | `number_of_reviews`, `first_review`, `last_review`, `host_since`, `calendar_updated` | seção recolhida, com defaults coerentes |
| **Texto livre** (3) | `amenities`, `description`, `house_rules` | multisseleção e áreas de texto |
| **Localização** (2) | `latitude`, `longitude` | **derivadas do bairro** (§3.2) |
| **Não aplicáveis** (2) | `id`, `price` | **sintéticos** (§3.1) |

### 2.1 O paradoxo do `price`

A API exige `price` — e é exatamente o que o usuário quer descobrir. Não é
inconsistência do modelo: `price` **não é feature**. O modelo não o usa para prever;
a API o usa para classificar a **zona** (subprecificado / zona morta / superprecificado),
comparando o preço atual contra a banda.

Isso dá dois modos de uso legítimos, e a página oferece os dois:

| Modo | O usuário informa | A página mostra |
|---|---|---|
| **"Quanto devo cobrar?"** | nada sobre preço | preço justo + banda |
| **"Meu preço está certo?"** | o preço atual | preço justo + banda + **zona e recomendação** |

No primeiro modo a página envia um `price` sintético e **descarta** a zona devolvida,
porque ela não significaria nada. Essa supressão é deliberada e precisa estar
explícita no código — devolver uma zona calculada contra um preço inventado seria
pior que não devolver nenhuma.

---

## 3. Campos derivados, não perguntados

### 3.1 `id` e `price`
`id` é identificador, não característica: a página gera um valor sintético.
`price`, no modo "quanto devo cobrar", recebe um placeholder cujo único papel é passar
pela regra de validade `price > 0`.

### 3.2 `latitude` / `longitude` a partir do bairro
Pedir coordenadas a um anfitrião não é razoável, e o modelo usa `latitude`,
`longitude`, `dist_centro_km` e `dist_museumplein_km` — quatro das 27 features.

Solução: **centroide do bairro**, calculado da base e embutido na página. O usuário
escolhe entre os 22 bairros e as coordenadas vêm junto. Perde-se a variação
intrabairro; a alternativa (um mapa) é escopo maior que o da etapa.

Registrar essa perda importa: dois imóveis no mesmo bairro recebem a mesma distância
ao centro, então a página é menos sensível à localização do que o modelo é capaz.

### 3.3 `amenities`
O formato cru é `{Internet,Wifi,"Smoke detector",...}`. O modelo usa apenas
`n_amenities` — a contagem. A interface oferece uma multisseleção com as comodidades
mais frequentes da base e remonta a string nesse formato.

### 3.4 Datas
`host_since`, `first_review` e `last_review` viram features de **tempo decorrido**
(`host_experiencia_dias`, `dias_desde_ultima_review`), medidas contra
`data_referencia = 2018-12-06`. A página expõe seletores de data com defaults
medianos e explica a referência — sem isso, um usuário escolheria "hoje" (2026) e
produziria um valor negativo que nenhuma validação captura.

---

## 4. Estrutura da página

```
app/app.py

┌─ Cabeçalho ───────────────────────────────────────────────┐
│  Precificação de Diárias — Airbnb Amsterdam               │
│  MAPE 22,21% · modelo v1 · dados de dez/2018              │
├─ Barra lateral ───────────────────────────────────────────┤
│  Estado da API (/health)                                  │
│  Modo: "Quanto cobrar?" | "Meu preço está certo?"         │
├─ Formulário (abas) ───────────────────────────────────────┤
│  [Imóvel] [Regras e taxas] [Histórico] [Textos]           │
├─ Resultado ───────────────────────────────────────────────┤
│  PREÇO JUSTO: €XXX                                        │
│  Faixa: €L — €U  (±22,21%)                                │
│  [modo 2] Zona + recomendação                             │
│  Campos assumidos por padrão (transparência)              │
└───────────────────────────────────────────────────────────┘
```

As abas existem para que os 25 campos não apareçam como uma lista de 25 — os nove que
importam ficam na primeira aba, com defaults razoáveis no resto.

---

## 5. Comunicação do resultado — o ponto mais delicado

A tentação é mostrar **€147** em destaque. Seria enganoso: o modelo erra 22,21% em
média, então a informação honesta é a **faixa**, não o ponto.

| Elemento | Tratamento |
|---|---|
| Preço justo | grande, mas acompanhado da faixa com igual peso visual |
| Faixa `L — U` | sempre visível, nunca em nota de rodapé |
| MAPE | citado como "o modelo erra ±22% em média" |
| Piso de €75 | se o preço estimado cair abaixo, **avisa** que a recomendação é menos confiável |

### 5.1 O aviso do piso
Abaixo de €75 o modelo erra 38,5% e prevê ~29% acima do real. Na API isso vira a zona
`abaixo_do_piso`; na página precisa virar **texto legível**, porque é onde o usuário
poderia tomar a pior decisão com base na ferramenta.

### 5.2 Tornar visível que a resposta veio da AWS

O propósito 1 é observar o deploy funcionando. Uma página que simplesmente mostra um
número não cumpre isso — seria indistinguível de um cálculo local.

Então a página exibe, junto do resultado, a procedência da resposta:

| Elemento | Por quê |
|---|---|
| Endereço da Function URL | deixa explícito que saiu da máquina |
| Tag da imagem (`v1-<sha>`) | liga o número ao commit que o gerou |
| Latência da chamada | distingue cold start (~25 s) de requisição morna (~0,8 s) |
| Estado do `/health` na barra lateral | mostra o modelo carregado antes de qualquer envio |

É isso que transforma a página de "uma calculadora" em "uma janela para o deploy".

### 5.3 A safra dos dados
Os dados são de **dezembro de 2018**. Um anfitrião em 2026 que visse "€147" sem esse
contexto tomaria uma decisão com base num mercado que mudou — Amsterdam restringiu
aluguel de curta duração depois dessa safra. O aviso vai no cabeçalho, não escondido.

---

## 6. Custos das duas decisões travadas

Ambas foram escolha sua; o plano registra o que elas implicam. Como a página roda
**só localmente** e serve de protótipo, os dois custos são menores do que eu havia
estimado antes de saber disso.

**Credencial AWS obrigatória (decisão 2).** A Function URL usa `AWS_IAM`, então a
máquina que abrir a página precisa de `~/.aws/credentials` com permissão de
invocação. Como só roda na sua máquina — onde o perfil `airbnb` já está configurado e
foi usado no portão 8 — isso deixa de ser obstáculo e vira exatamente o que se quer:
**a página exercita a infraestrutura real**, que é o propósito 1. A assinatura SigV4
usará `botocore`, que não está instalado e entra no `requirements.txt`.

Consequência a registrar: se outra pessoa da área de negócio quiser abrir a página na
própria máquina, vai precisar de credencial AWS. Para uma demonstração conduzida por
você, não é problema; para distribuir, seria.

**25 campos (decisão 1).** O formulário fica longo, e numa demonstração ao vivo isso
custa tempo de tela. Mitigo com abas e defaults medianos: os nove campos que o
anfitrião sabe de cabeça ficam na primeira aba, e dá para gerar uma predição mexendo
só neles. Os outros dezesseis ficam disponíveis para quem quiser explorar a
sensibilidade do modelo — o que, num protótipo destinado a mostrar como a coisa
funciona, é uma vantagem e não um defeito.

## 7. Ordem de execução

| # | Tarefa | Validação |
|---|---|---|
| 1 | Instalar `streamlit` e `botocore` | importam |
| 2 | Extrair centroides dos bairros e defaults medianos da base | tabela conferida |
| 3 | `app/app.py` — formulário e montagem do payload | payload passa em `validar_colunas` |
| 4 | Cliente SigV4 da Function URL | `/health` responde |
| 5 | Camada de resultado (faixa, zona, avisos) | casos de borda cobertos |
| 6 | Teste: a página produz o mesmo número da API direta | diferença < €0,01 |
| 7 | Documentar no README | — |

O passo 6 é o portão: se a página divergir da API, o problema está na montagem do
payload, e isso precisa aparecer antes de a etapa fechar.

---

## 8. Entregáveis

1. `app/app.py` — a página
2. `app/defaults_app.py` — centroides e medianas extraídos da base
3. `streamlit` e `botocore` no `requirements.txt`
4. Seção no README com instruções de uso
5. Teste de equivalência página ↔ API
