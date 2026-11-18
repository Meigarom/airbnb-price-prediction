# Provider, backend remoto e valores derivados.
#
# Plano: docs/plans/14_deploy_aws.md §5

terraform {
  required_version = ">= 1.11"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # State no S3, não local. Num projeto com CI/CD o state local é fonte
  # garantida de conflito: a máquina local e o runner do GitHub Actions
  # teriam visões divergentes da infraestrutura.
  #
  # `use_lockfile` usa o lock NATIVO do S3 (condicional de escrita), disponível
  # desde o Terraform 1.11. O plano original previa DynamoDB; não é mais
  # necessário, e dispensa uma tabela e sua política de acesso.
  #
  # O bucket foi criado manualmente no portão 2 (problema do ovo e da galinha:
  # o backend precisa existir antes do primeiro `init`).
  backend "s3" {
    bucket       = "precificacao-airbnb-ams"
    key          = "terraform/state/terraform.tfstate"
    region       = "eu-west-1"
    profile      = "airbnb"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  # Tags aplicadas a todo recurso que as suporte — sem isso, rastrear custo
  # por projeto no Cost Explorer vira arqueologia.
  default_tags {
    tags = local.tags
  }
}

data "aws_caller_identity" "atual" {}

locals {
  account_id = data.aws_caller_identity.atual.account_id

  # Nome base reutilizado por todos os recursos, para que o prefixo seja
  # consistente e um `terraform destroy` não deixe órfãos de nome divergente.
  nome = var.project_name

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
