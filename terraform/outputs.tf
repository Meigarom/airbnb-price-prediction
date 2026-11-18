# Valores necessários para o deploy manual (portão 5) e para configurar o
# GitHub Actions (portão 6).

output "function_url" {
  description = "Endpoint da API. Exige assinatura SigV4 (auth AWS_IAM)."
  value       = aws_lambda_function_url.api.function_url
}

output "ecr_repository_url" {
  description = "URI do repositório ECR, destino do `docker push`."
  value       = aws_ecr_repository.api.repository_url
}

output "lambda_function_name" {
  description = "Nome da função, usado pelo `aws lambda update-function-code`."
  value       = aws_lambda_function.api.function_name
}

output "github_deploy_role_arn" {
  description = "ARN a configurar no GitHub Actions como `role-to-assume` (portão 6)."
  value       = aws_iam_role.github_deploy.arn
}

output "log_group_name" {
  description = "Log group da Lambda, para `aws logs tail`."
  value       = aws_cloudwatch_log_group.lambda.name
}

# Os comandos do primeiro deploy manual, já com os valores preenchidos — para
# não haver transcrição manual de URI de ECR, que é fonte clássica de erro.
output "comandos_deploy_manual" {
  description = "Sequência do portão 5: login, tag, push."
  value       = <<-EOT

    # 1. Autenticar o Docker no ECR
    aws ecr get-login-password --region ${var.aws_region} --profile ${var.aws_profile} \
      | docker login --username AWS --password-stdin ${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com

    # 2. Marcar a imagem local com a tag de release
    docker tag precificacao-airbnb:v1 ${aws_ecr_repository.api.repository_url}:${var.image_tag}

    # 3. Publicar
    docker push ${aws_ecr_repository.api.repository_url}:${var.image_tag}

    # 4. Apontar a Lambda para a nova imagem
    aws lambda update-function-code \
      --function-name ${aws_lambda_function.api.function_name} \
      --image-uri ${aws_ecr_repository.api.repository_url}:${var.image_tag} \
      --region ${var.aws_region} --profile ${var.aws_profile}

  EOT
}
