# ──────────────────────────────────────────────────────────────────────────
# Alarms → SNS → email.
#
# First-time setup: after this applies, AWS sends a "confirm subscription"
# email to var.alarm_email. That email contains a link — CLICK IT to start
# receiving alerts. Until the subscription is confirmed, alarms fire but
# don't reach anyone.
# ──────────────────────────────────────────────────────────────────────────

resource "aws_sns_topic" "alerts" {
  name              = "anna-chat-${var.env}-alerts"
  kms_master_key_id = module.kms_logs.key_arn
  tags              = local.tags
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# ── Lambda error rate ────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "chat_errors" {
  alarm_name          = "anna-chat-${var.env}-chat-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 3
  treat_missing_data  = "notBreaching"
  alarm_description   = "Chat Lambda returned >3 errors in 5 minutes. Check /aws/lambda/anna-chat-${var.env}-chat logs."

  dimensions = {
    FunctionName = module.lambda_chat.function_name
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
  tags          = local.tags
}

resource "aws_cloudwatch_metric_alarm" "attachments_errors" {
  alarm_name          = "anna-chat-${var.env}-attachments-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 3
  treat_missing_data  = "notBreaching"
  alarm_description   = "Attachments Lambda returned >3 errors in 5 minutes."

  dimensions = {
    FunctionName = module.lambda_attachments.function_name
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
  tags          = local.tags
}

resource "aws_cloudwatch_metric_alarm" "extract_errors" {
  alarm_name          = "anna-chat-${var.env}-extract-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 600
  statistic           = "Sum"
  threshold           = 5
  treat_missing_data  = "notBreaching"
  alarm_description   = "Extract Lambda returned >5 errors in 10 minutes — attachments may be stuck in 'extracting'."

  dimensions = {
    FunctionName = module.lambda_extract.function_name
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
  tags          = local.tags
}

# ── API Gateway 5xx ──────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "anna-chat-${var.env}-api-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  metric_name         = "5xx"
  namespace           = "AWS/ApiGateway"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  treat_missing_data  = "notBreaching"
  alarm_description   = "API Gateway returned >5 5xx responses in 5 minutes."

  dimensions = {
    ApiId = module.api.api_id
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
  tags          = local.tags
}

# ── WAF blocks ───────────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "waf_blocks" {
  alarm_name          = "anna-chat-${var.env}-waf-blocks"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  metric_name         = "BlockedRequests"
  namespace           = "AWS/WAFV2"
  period              = 300
  statistic           = "Sum"
  threshold           = 50
  treat_missing_data  = "notBreaching"
  alarm_description   = ">50 WAF blocks in 5 minutes — possible attack in progress."

  dimensions = {
    WebACL = "anna-chat-${var.env}-spa-acl"
    Region = var.aws_region
    Rule   = "ALL"
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
  tags          = local.tags
}

# ── DynamoDB throttles (rare but worth catching) ─────────────────────────

resource "aws_cloudwatch_metric_alarm" "ddb_throttles" {
  alarm_name          = "anna-chat-${var.env}-ddb-throttles"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  metric_name         = "UserErrors"
  namespace           = "AWS/DynamoDB"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  treat_missing_data  = "notBreaching"
  alarm_description   = ">5 DDB user errors in 5 min across the account (throttles, validation, missing-key, etc.)."

  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
  tags          = local.tags
}

# ── Billing anomaly (account-wide) ───────────────────────────────────────

# Near-real-time Bedrock runaway detection (budgets are lagging by 24-48h).
# A normal day has maybe 100-500 invocations across all users. 500 invocations
# in a single 5-minute window almost certainly indicates a recursive bug,
# prompt injection loop, or a compromised user session hammering the API.
resource "aws_cloudwatch_metric_alarm" "bedrock_runaway" {
  alarm_name          = "anna-chat-${var.env}-bedrock-runaway"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  metric_name         = "Invocations"
  namespace           = "AWS/Bedrock"
  period              = 300
  statistic           = "Sum"
  threshold           = 500
  treat_missing_data  = "notBreaching"
  alarm_description   = "Bedrock invocations > 500 in 5 min. Likely a runaway loop. Check /aws/lambda/anna-chat-${var.env}-chat logs; consider attaching the bedrock-kill-switch policy manually while diagnosing."

  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
  tags          = local.tags
}

resource "aws_cloudwatch_metric_alarm" "billing_anomaly" {
  alarm_name          = "anna-chat-${var.env}-billing-over-100usd"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 21600 # 6h (billing metric granularity)
  statistic           = "Maximum"
  threshold           = 100
  treat_missing_data  = "notBreaching"
  alarm_description   = "AWS estimated monthly charges crossed $100. Review spend — unexpected traffic or a misconfigured Bedrock loop can rack up fast."

  dimensions = {
    Currency = "USD"
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
  tags          = local.tags
}

# ── Outputs ──────────────────────────────────────────────────────────────

output "alerts_topic_arn" {
  description = "SNS topic receiving alarm notifications. Manually add subscriptions here if you want more than one destination."
  value       = aws_sns_topic.alerts.arn
}
