resource "aws_sns_topic" "alarms" {
  name = "${local.name}-alarms"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alarm_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# The worker logs JSON lines with event="analysis_failed" (structlog) on permanent failures.
resource "aws_cloudwatch_log_metric_filter" "analysis_failed" {
  name           = "${local.name}-analysis-failed"
  log_group_name = aws_cloudwatch_log_group.app["worker"].name
  pattern        = "{ $.event = \"analysis_failed\" }"
  metric_transformation {
    name      = "AnalysisFailed"
    namespace = "TerraSentinel"
    value     = "1"
    unit      = "Count"
  }
}

resource "aws_cloudwatch_log_metric_filter" "analysis_seconds" {
  name           = "${local.name}-analysis-seconds"
  log_group_name = aws_cloudwatch_log_group.app["worker"].name
  pattern        = "{ $.event = \"analysis_succeeded\" }"
  metric_transformation {
    name      = "AnalysisSeconds"
    namespace = "TerraSentinel"
    value     = "$.seconds"
    unit      = "Seconds"
  }
}

resource "aws_cloudwatch_metric_alarm" "analysis_failures" {
  alarm_name          = "${local.name}-analysis-failures"
  namespace           = "TerraSentinel"
  metric_name         = "AnalysisFailed"
  statistic           = "Sum"
  period              = 900
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "${local.name}-api-5xx"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  dimensions          = { LoadBalancer = aws_lb.main.arn_suffix }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 10
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_metric_alarm" "worker_memory" {
  alarm_name          = "${local.name}-worker-memory"
  namespace           = "AWS/ECS"
  metric_name         = "MemoryUtilization"
  dimensions          = { ClusterName = aws_ecs_cluster.main.name, ServiceName = aws_ecs_service.worker.name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 90
  comparison_operator = "GreaterThanOrEqualToThreshold"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}
