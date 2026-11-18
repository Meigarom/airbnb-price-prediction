# Registro da imagem de container da Lambda.
#
# Plano: docs/plans/14_deploy_aws.md §3.1

resource "aws_ecr_repository" "api" {
  name = local.nome

  # Tag imutável: uma tag sempre aponta para o mesmo conteúdo. Sem isso,
  # `v1-abc123` poderia ser sobrescrita e a rastreabilidade entre imagem e
  # commit — a razão de embutir os artefatos na imagem (§3.2) — se perderia.
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    # Varre CVEs das dependências a cada push. A imagem carrega pandas,
    # numpy, scipy e LightGBM; a superfície não é desprezível.
    scan_on_push = true
  }

  # O repositório guarda imagens, não state. Permitir a destruição com
  # imagens dentro evita que um `destroy` fique preso.
  force_delete = true
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  # Cada imagem tem ~1,3 GB. Sem expiração, o storage cresce a cada deploy
  # indefinidamente — a 10 ¢/GB/mês, 50 deploys custariam mais que todo o
  # resto da infraestrutura somada.
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Mantém as 10 imagens mais recentes"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      },
    ]
  })
}
