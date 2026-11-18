# Parâmetros da infraestrutura.
#
# Os defaults refletem o ambiente já provisionado nos portões 1–3; mudá-los
# exige conferir se o recurso correspondente já existe na conta.

variable "aws_region" {
  description = "Região da AWS. eu-west-1 (Irlanda): dados de Amsterdam, latência baixa para a UE e todos os serviços usados disponíveis."
  type        = string
  default     = "eu-west-1"
}

variable "aws_profile" {
  description = "Perfil do ~/.aws/credentials usado localmente. No GitHub Actions a autenticação é por OIDC e este valor é ignorado."
  type        = string
  default     = "airbnb"
}

variable "project_name" {
  description = "Prefixo de nome de todos os recursos."
  type        = string
  default     = "precificacao-airbnb"
}

variable "environment" {
  description = "Ambiente, usado apenas como tag."
  type        = string
  default     = "prod"
}

variable "bucket_name" {
  description = "Bucket que guarda dados brutos, artefatos e o state do Terraform. Criado manualmente no portão 2 e adotado pelo Terraform via import."
  type        = string
  default     = "precificacao-airbnb-ams"
}

# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------

variable "lambda_memory_mb" {
  description = <<-EOT
    Memória da Lambda. 1024 MB não é sobre memória — o modelo ocupa ~5 MB.
    É sobre CPU: o Lambda aloca CPU proporcional à memória, e abaixo de 1 GB
    o carregamento dos pickles (1,11 s medido) degrada. Mais memória custa
    mais por ms mas termina antes; 1 GB é o ponto medido como razoável.
  EOT
  type        = number
  default     = 1024
}

variable "lambda_timeout_s" {
  description = "Timeout. Um lote de 1.000 anúncios respondeu em 55 ms; 60 s é folga de três ordens de grandeza, e existe para cortar um travamento patológico, não para acomodar carga."
  type        = number
  default     = 60
}

variable "lambda_reserved_concurrency" {
  description = <<-EOT
    Teto de execuções simultâneas. É a mitigação de custo da decisão de usar
    Function URL sem API Gateway (§3.3 do plano): sem rate limiting por
    consumidor, este número é o que limita o gasto máximo, independentemente
    do volume de chamadas. -1 desativa a reserva.

    DEFAULT -1 POR IMPOSIÇÃO DA CONTA: a cota total desta conta é de 10
    execuções simultâneas (contas novas começam em 10, não em 1.000), e a AWS
    exige que pelo menos 10 permaneçam NÃO reservadas. Qualquer reserva é
    rejeitada com InvalidParameterValueException.

    Na prática não se perde o controle de custo: com teto de 10 na conta
    inteira, a própria cota já é o limite que a reserva buscava. Ao pedir
    aumento da cota (Service Quotas → Lambda → Concurrent executions), voltar
    este valor para 10 passa a fazer sentido.
  EOT
  type        = number
  default     = -1
}

variable "image_tag" {
  description = <<-EOT
    Tag da imagem no ECR que a Lambda executa, imutável por política do
    repositório.

    NÃO é "v1": aquela tag ficou permanentemente presa a um manifesto OCI
    image index (multi-plataforma), que o Lambda recusa. Como a política de
    imutabilidade impede sobrescrever uma tag — e é desejável que impeça —, a
    imagem corrigida foi publicada sob outra tag.

    O build precisa de `--provenance=false`: sem isso o BuildKit anexa uma
    attestation e promove o manifesto a índice, mesmo com uma só plataforma.
  EOT
  type        = string
  default     = "v1-9dbb1e2-docker"
}

# ---------------------------------------------------------------------------
# GitHub OIDC
# ---------------------------------------------------------------------------

variable "github_repo" {
  description = "Repositório <owner>/<nome> autorizado a assumir a role de deploy via OIDC."
  type        = string
  default     = "Meigarom/airbnb-price-prediction"
}

variable "github_branch" {
  description = "Único branch autorizado a fazer deploy. Restringe a confiança OIDC: um PR de fork não consegue assumir a role."
  type        = string
  default     = "main"
}

variable "create_github_oidc_provider" {
  description = <<-EOT
    Cria o provedor OIDC do GitHub. Só pode existir UM por conta AWS.

    DESLIGADO por ora: o usuário airbnb-deploy não tem iam:TagOpenIDConnectProvider,
    e o provider da AWS sempre envia tags na criação — o pedido volta com
    AccessDenied. É pré-requisito apenas do portão 6 (GitHub Actions); o deploy
    manual do portão 5 não depende dele.
  EOT
  type        = bool
  default     = false
}

# ---------------------------------------------------------------------------
# Alarmes (§3.5) — baselines medidos na base completa de 19.981 anúncios
# ---------------------------------------------------------------------------

variable "alarm_email" {
  description = "E-mail que recebe os alarmes. Vazio desativa as notificações; os alarmes continuam existindo e visíveis no console."
  type        = string
  default     = ""
}

variable "baseline_mediana_preco" {
  description = "Mediana de price_estimate na base completa (€). Referência do alarme de drift."
  type        = number
  default     = 128
}

variable "baseline_taxa_subprecificado" {
  description = "Proporção de anúncios classificados como subprecificados na base completa (2.314 / 19.981)."
  type        = number
  default     = 0.116
}
