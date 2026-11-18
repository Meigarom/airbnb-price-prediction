# Duas identidades distintas, com privilégio mínimo para cada uma:
#
#   1. a role de EXECUÇÃO da Lambda — o que o código pode fazer em runtime
#   2. a role de DEPLOY do GitHub   — o que o CI pode fazer na conta
#
# Separá-las importa: a Lambda não precisa de permissão para publicar imagens,
# e o CI não precisa de permissão para escrever logs da função.
#
# Plano: docs/plans/14_deploy_aws.md §5.3

# ===========================================================================
# 1. Role de execução da Lambda
# ===========================================================================

data "aws_iam_policy_document" "lambda_confianca" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${local.nome}-lambda-exec"
  assume_role_policy = data.aws_iam_policy_document.lambda_confianca.json
}

data "aws_iam_policy_document" "lambda_permissoes" {
  # Logs. Escopo restrito ao log group DESTA função: a política gerenciada
  # AWSLambdaBasicExecutionRole daria acesso a todos os log groups da conta.
  statement {
    sid    = "EscreverLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.lambda.arn}:*"]
  }

  # Métricas de drift via EMF (§3.5). O CloudWatch não permite restringir
  # PutMetricData por recurso — a condição de namespace é o único escopo
  # possível, e impede que a função escreva em métricas de outros sistemas.
  statement {
    sid       = "PublicarMetricasDeDrift"
    effect    = "Allow"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = [local.metric_namespace]
    }
  }

  # Leitura dos artefatos no S3. NÃO é usada no caminho de runtime — os
  # pickles estão embutidos na imagem (decisão 1, §3.2). Existe para o caso
  # de diagnóstico e para um eventual retreino; é leitura, e só do prefixo
  # de modelos.
  statement {
    sid       = "LerArtefatos"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.dados.arn}/models/*"]
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${local.nome}-lambda-policy"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda_permissoes.json
}

# ===========================================================================
# 2. Role de deploy do GitHub Actions (OIDC)
# ===========================================================================
#
# OIDC em vez de chaves estáticas: o workflow troca um token de curta duração
# emitido pelo GitHub por credenciais temporárias da AWS. Não existe segredo
# de longa duração no repositório para vazar.

resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 1 : 0

  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]

  # A lista de thumbprints é legado: desde 2023 a AWS valida a cadeia de
  # certificados do GitHub por conta própria e ignora este campo para este
  # provedor. Mantido porque a API ainda o exige.
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

locals {
  github_oidc_arn = var.create_github_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : "arn:aws:iam::${local.account_id}:oidc-provider/token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "github_confianca" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # ESTA condição é o controle de acesso real. Sem ela, QUALQUER repositório
    # do GitHub — de qualquer pessoa no mundo — poderia assumir esta role.
    # `sub` é restrito a um repositório e a um branch específicos, então um
    # PR vindo de fork não consegue fazer deploy.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/${var.github_branch}"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "${local.nome}-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_confianca.json

  # O CI não deve precisar de sessões longas. 1 h cobre build + push + deploy
  # com folga.
  max_session_duration = 3600
}

data "aws_iam_policy_document" "github_permissoes" {
  # Token de login no ECR. Esta ação não aceita restrição por recurso.
  statement {
    sid       = "AutenticarNoECR"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  # Push da imagem — restrito a ESTE repositório do ECR.
  statement {
    sid    = "PublicarImagem"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:DescribeImages",
    ]
    resources = [aws_ecr_repository.api.arn]
  }

  # Atualizar a função para a nova imagem. Note o que NÃO está aqui:
  # CreateFunction e DeleteFunction. O CI troca a imagem de uma função que o
  # Terraform criou; ele não cria nem apaga infraestrutura.
  statement {
    sid    = "AtualizarLambda"
    effect = "Allow"
    actions = [
      "lambda:UpdateFunctionCode",
      "lambda:GetFunction",
      "lambda:GetFunctionConfiguration",
      # O smoke test descobre a Function URL em vez de recebê-la fixa: assim
      # o workflow não precisa ser editado se a URL mudar.
      "lambda:GetFunctionUrlConfig",
      "lambda:PublishVersion",
      "lambda:UpdateFunctionConfiguration",
    ]
    resources = [aws_lambda_function.api.arn]
  }

  # Smoke test pós-deploy: o workflow chama /health na Function URL, e a
  # chamada precisa ser assinada (auth AWS_IAM).
  statement {
    sid       = "InvocarParaSmokeTest"
    effect    = "Allow"
    actions   = ["lambda:InvokeFunctionUrl"]
    resources = [aws_lambda_function.api.arn]
  }

  # Baixar os artefatos do S3 para o build (os pickles não estão no Git).
  statement {
    sid       = "BaixarArtefatos"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.dados.arn}/models/*"]
  }

  statement {
    sid       = "ListarArtefatos"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.dados.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["models/*"]
    }
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  name   = "${local.nome}-github-deploy-policy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_permissoes.json
}
