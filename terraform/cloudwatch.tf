# Logs, métricas de drift e alarmes.
#
# O contexto que justifica esta seção: em produção NÃO HÁ hold-out, então a
# degradação do MAPE é invisível — é a pendência mais séria herdada do plano
# 13. Nenhum alarme aqui mede MAPE; isso exigiria preços reais posteriores.
# O que estes alarmes detectam é MUDANÇA DE REGIME, que é o sintoma de drift.
#
# Os baselines vêm da base completa de 19.981 anúncios (seção de CV do
# notebook), não de estimativa.
#
# Plano: docs/plans/14_deploy_aws.md §3.5

locals {
  metric_namespace = "Precificacao/Airbnb"

  # Dois dos três alarmes de drift dependem de métricas que a própria Lambda
  # emite via EMF — código que ainda NÃO existe em api.py (portão 7). Enquanto
  # não existir, eles ficam em estado INSUFFICIENT_DATA, que é honesto: não há
  # dado, e o alarme diz isso em vez de fingir que está tudo bem.
  alarmes_ativos = var.alarm_email != ""
}

resource "aws_cloudwatch_log_group" "lambda" {
  name = "/aws/lambda/${local.nome}"

  # 30 dias. Logs de uma API sem estado perdem utilidade diagnóstica rápido,
  # e retenção infinita (o default quando o group é criado implicitamente)
  # é a maior fonte de custo surpresa do CloudWatch.
  retention_in_days = 30
}

# ---------------------------------------------------------------------------
# Notificação
# ---------------------------------------------------------------------------

resource "aws_sns_topic" "alarmes" {
  count = local.alarmes_ativos ? 1 : 0
  name  = "${local.nome}-alarmes"
}

resource "aws_sns_topic_subscription" "email" {
  count = local.alarmes_ativos ? 1 : 0

  topic_arn = aws_sns_topic.alarmes[0].arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

locals {
  acoes_alarme = local.alarmes_ativos ? [aws_sns_topic.alarmes[0].arn] : []
}

# ---------------------------------------------------------------------------
# Alarmes operacionais — métricas nativas do Lambda, funcionam desde já
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "taxa_de_erro" {
  alarm_name        = "${local.nome}-taxa-de-erro"
  alarm_description = "Mais de 1% das invocações falhou em 5 minutos."

  comparison_operator = "GreaterThanThreshold"
  threshold           = 1
  evaluation_periods  = 1

  # `treat_missing_data = notBreaching`: sem tráfego não há erro. O default
  # (`missing`) deixaria o alarme oscilar em toda janela ociosa.
  treat_missing_data = "notBreaching"
  alarm_actions      = local.acoes_alarme
  ok_actions         = local.acoes_alarme

  metric_query {
    id = "taxa"
    # IF evita a divisão por zero de uma janela sem tráfego. MAX([x, 1]) não
    # serve: metric math não aceita array literal nesta posição.
    expression  = "IF(invocacoes > 0, 100 * erros / invocacoes, 0)"
    label       = "Taxa de erro (%)"
    return_data = true
  }

  metric_query {
    id = "erros"
    metric {
      namespace   = "AWS/Lambda"
      metric_name = "Errors"
      period      = 300
      stat        = "Sum"
      dimensions  = { FunctionName = aws_lambda_function.api.function_name }
    }
  }

  metric_query {
    id = "invocacoes"
    metric {
      namespace   = "AWS/Lambda"
      metric_name = "Invocations"
      period      = 300
      stat        = "Sum"
      dimensions  = { FunctionName = aws_lambda_function.api.function_name }
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "duracao_p99" {
  alarm_name        = "${local.nome}-duracao-p99"
  alarm_description = "p99 acima de 5 s. Um lote de 1.000 anúncios respondeu em 55 ms e o cold start é 1,11 s — 5 s indica patologia, não carga."

  namespace           = "AWS/Lambda"
  metric_name         = "Duration"
  dimensions          = { FunctionName = aws_lambda_function.api.function_name }
  extended_statistic  = "p99"
  period              = 300
  evaluation_periods  = 2
  comparison_operator = "GreaterThanThreshold"
  threshold           = 5000 # ms

  treat_missing_data = "notBreaching"
  alarm_actions      = local.acoes_alarme
  ok_actions         = local.acoes_alarme
}

resource "aws_cloudwatch_metric_alarm" "throttles" {
  alarm_name        = "${local.nome}-throttles"
  alarm_description = "A Lambda esta sendo estrangulada: a demanda passou do teto de concorrencia da conta (10 execucoes simultaneas). Nao e erro de codigo; e limite de cota. Resolve-se pedindo aumento em Service Quotas, nao mexendo no codigo."

  namespace           = "AWS/Lambda"
  metric_name         = "Throttles"
  dimensions          = { FunctionName = aws_lambda_function.api.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0

  treat_missing_data = "notBreaching"
  alarm_actions      = local.acoes_alarme
  ok_actions         = local.acoes_alarme
}

# ---------------------------------------------------------------------------
# Alarmes de drift — dependem das métricas EMF do portão 7
# ---------------------------------------------------------------------------

# Dois alarmes para a mediana: a banda é de DOIS lados, e um alarme do
# CloudWatch só compara numa direção. Preço subindo e preço caindo são
# sintomas diferentes, e separá-los também torna a notificação mais legível.
#
# Limiar estático, não ANOMALY_DETECTION_BAND: a detecção de anomalia precisa
# de histórico para treinar um modelo, e esta métrica NUNCA foi emitida (o
# código EMF é do portão 7). Pedir banda sobre série inexistente é o que
# produziu o ValidationError no primeiro apply. Com o baseline medido em
# 19.981 anúncios, o limiar fixo de ±15% é mais defensável de qualquer forma —
# é um número que veio de medição, não de um modelo sem dado.

resource "aws_cloudwatch_metric_alarm" "drift_mediana_preco_alta" {
  alarm_name        = "${local.nome}-drift-mediana-preco-alta"
  alarm_description = "Mediana de price_estimate acima de EUR ${format("%.0f", var.baseline_mediana_preco * 1.15)} (baseline EUR ${var.baseline_mediana_preco} +15%) em 24 h. Sintoma de mudanca de regime na entrada ou no mercado."

  namespace           = local.metric_namespace
  metric_name         = "PriceEstimateMediana"
  statistic           = "Average"
  period              = 86400
  evaluation_periods  = 1
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.baseline_mediana_preco * 1.15

  treat_missing_data = "missing"
  alarm_actions      = local.acoes_alarme
}

resource "aws_cloudwatch_metric_alarm" "drift_mediana_preco_baixa" {
  alarm_name        = "${local.nome}-drift-mediana-preco-baixa"
  alarm_description = "Mediana de price_estimate abaixo de EUR ${format("%.0f", var.baseline_mediana_preco * 0.85)} (baseline EUR ${var.baseline_mediana_preco} -15%)."

  namespace           = local.metric_namespace
  metric_name         = "PriceEstimateMediana"
  statistic           = "Average"
  period              = 86400
  evaluation_periods  = 1
  comparison_operator = "LessThanThreshold"
  threshold           = var.baseline_mediana_preco * 0.85

  treat_missing_data = "missing"
  alarm_actions      = local.acoes_alarme
}

resource "aws_cloudwatch_metric_alarm" "drift_taxa_subprecificado" {
  alarm_name        = "${local.nome}-drift-taxa-subprecificado"
  alarm_description = "Proporção de subprecificados acima de ${format("%.1f", (var.baseline_taxa_subprecificado + 0.05) * 100)}% (baseline ${format("%.1f", var.baseline_taxa_subprecificado * 100)}% + 5 p.p.). Mais recomendações que o esperado sugere que o modelo deslocou, não que o mercado mudou."

  namespace           = local.metric_namespace
  metric_name         = "TaxaSubprecificado"
  statistic           = "Average"
  period              = 86400
  evaluation_periods  = 1
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.baseline_taxa_subprecificado + 0.05

  treat_missing_data = "missing"
  alarm_actions      = local.acoes_alarme
}

resource "aws_cloudwatch_metric_alarm" "taxa_invalidos" {
  alarm_name        = "${local.nome}-taxa-invalidos"
  alarm_description = "Mais de 2% das entradas rejeitadas por regras de validade (baseline 0,04%). O sintoma típico é mudança no formato de entrada do cliente."

  namespace           = local.metric_namespace
  metric_name         = "TaxaInvalidos"
  statistic           = "Average"
  period              = 3600
  evaluation_periods  = 2
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0.02

  treat_missing_data = "missing"
  alarm_actions      = local.acoes_alarme
}
