# Bucket de dados brutos, artefatos e state do Terraform.
#
# ATENÇÃO: este bucket foi criado manualmente no portão 2 (o backend do
# Terraform precisava existir antes do primeiro `init` — problema do ovo e da
# galinha, §5.2 do plano). Para o Terraform passar a gerenciá-lo sem recriá-lo,
# ele é ADOTADO via bloco `import`, não criado. Sem isso, o `apply` falharia
# com BucketAlreadyOwnedByYou.
#
# Plano: docs/plans/14_deploy_aws.md §3.4

import {
  to = aws_s3_bucket.dados
  id = var.bucket_name
}

resource "aws_s3_bucket" "dados" {
  bucket = var.bucket_name

  # O bucket guarda o state do próprio Terraform. Um `destroy` que o apagasse
  # destruiria o registro do que foi destruído.
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "dados" {
  bucket = aws_s3_bucket.dados.id

  # Versionamento é o que dá rollback de modelo: `models/v1/model.pkl`
  # sobrescrito continua recuperável pela versão anterior. Também é o
  # mecanismo de recuperação do state do Terraform.
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "dados" {
  bucket = aws_s3_bucket.dados.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dados" {
  bucket = aws_s3_bucket.dados.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    # Reduz o custo de chamadas de KMS; com SSE-S3 não se aplica, mas o
    # parâmetro é inofensivo e explicita a intenção.
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "dados" {
  bucket = aws_s3_bucket.dados.id

  # O CSV de 75 MB é lido no treino, não em runtime. Depois de 90 dias sem
  # acesso, Infrequent Access custa ~45% menos por GB.
  rule {
    id     = "raw-para-infrequent-access"
    status = "Enabled"

    filter { prefix = "raw/" }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
  }

  # Versões antigas do state e dos artefatos: mantém 90 dias de histórico
  # para rollback, depois expira. Sem esta regra o versionamento acumula
  # indefinidamente cada `apply`.
  rule {
    id     = "expirar-versoes-antigas"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}
