# A função e sua Function URL.
#
# Plano: docs/plans/14_deploy_aws.md §3.3 e §5.3

resource "aws_lambda_function" "api" {
  function_name = local.nome
  role          = aws_iam_role.lambda.arn

  # Imagem de container, não zip: LightGBM + pandas + numpy + scipy passam
  # dos 250 MB do pacote zip (§2 do plano). A imagem construída tem 1,31 GB,
  # bem dentro do limite de 10 GB.
  package_type = "Image"
  image_uri    = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"

  # arm64 declarado em DOIS lugares: aqui e no Dockerfile. Se divergirem, o
  # deploy falha explicitamente em vez de a função quebrar em runtime.
  architectures = ["arm64"]

  memory_size = var.lambda_memory_mb
  timeout     = var.lambda_timeout_s

  # Teto de custo (§3.3): sem API Gateway não há rate limiting por consumidor,
  # então este número é o que limita o gasto máximo.
  reserved_concurrent_executions = var.lambda_reserved_concurrency

  environment {
    variables = {
      # Namespace das métricas de drift emitidas via EMF.
      METRIC_NAMESPACE = local.metric_namespace
      # Silencia o aviso de pickle salvo em Python 3.11 e carregado em 3.12
      # (pendência registrada; o teste de equivalência de €0,01 passa dentro
      # do container, então a divergência é só de versão, não de resultado).
      PYTHONWARNINGS = "ignore::UserWarning"
    }
  }

  # O log group é criado pelo Terraform (cloudwatch.tf) com retenção
  # definida. Sem esta dependência, a primeira invocação criaria o group
  # automaticamente com retenção INFINITA, e o Terraform então falharia ao
  # tentar criar um recurso que já existe.
  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_iam_role_policy.lambda,
  ]

  lifecycle {
    # O CI atualiza a imagem a cada deploy (UpdateFunctionCode). Sem isto, o
    # próximo `terraform apply` reverteria a função para a tag de `var.image_tag`,
    # desfazendo silenciosamente o último deploy.
    ignore_changes = [image_uri]
  }
}

resource "aws_lambda_function_url" "api" {
  function_name = aws_lambda_function.api.function_name

  # AWS_IAM, não NONE: toda requisição precisa ser assinada com SigV4. Uma
  # Function URL pública exporia o modelo a qualquer um que descobrisse a URL.
  authorization_type = "AWS_IAM"

  cors {
    allow_origins = ["*"]
    allow_methods = ["POST", "GET"]
    allow_headers = ["content-type", "authorization", "x-amz-date", "x-amz-security-token", "x-amz-content-sha256"]
    max_age       = 3600
  }
}
