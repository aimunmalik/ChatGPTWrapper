# ──────────────────────────────────────────────────────────────────────────
# Tier C #2: Bedrock spend hard-cap.
#
# Two defense layers, because AWS Budgets has 24-48h data delay:
#
#   Layer 1 (near-real-time, minutes): CloudWatch alarm in monitoring.tf
#   fires on Bedrock Invocations > threshold in 5 min → SNS email. Catches
#   runaway loops (recursive assistant-replies, prompt injection triggering
#   many expensive invokes, etc.) before they run up a $1k bill overnight.
#
#   Layer 2 (this file, slower but enforced): AWS Budget tracks month-to-
#   date Bedrock spend. Notifies at 50/80/100% of the monthly cap, and at
#   120% a Budget Action attaches a deny policy to the chat Lambda's role
#   to hard-stop further InvokeModel calls.
#
# If the kill switch fires you'll know: users will see the chat feature
# return an AccessDenied error, and you'll get an SNS email. To recover,
# detach the kill-switch policy manually (or raise the budget limit and
# re-apply Terraform — the budget action reverts on re-apply).
# ──────────────────────────────────────────────────────────────────────────

resource "aws_budgets_budget" "bedrock" {
  name         = "anna-chat-${var.env}-bedrock-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.bedrock_monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  time_period_start = "2026-01-01_00:00"

  cost_filter {
    name   = "Service"
    values = ["Amazon Bedrock"]
  }

  # Warn at 50%
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alarm_email]
  }

  # Warn at 80%
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alarm_email]
  }

  # Warn at 100% (over budget)
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alarm_email]
  }
}

# Role that the Budgets service assumes to attach the kill-switch policy.
resource "aws_iam_role" "budget_executor" {
  name = "anna-chat-${var.env}-budget-executor"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "budgets.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "budget_executor" {
  name = "attach-kill-switch"
  role = aws_iam_role.budget_executor.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "iam:AttachRolePolicy",
        "iam:DetachRolePolicy",
      ]
      Resource = [
        module.lambda_chat.role_arn,
      ]
    }]
  })
}

# The kill-switch policy itself — denies Bedrock InvokeModel(*) globally.
resource "aws_iam_policy" "bedrock_kill_switch" {
  name        = "anna-chat-${var.env}-bedrock-kill-switch"
  description = "Attached to the chat Lambda role by AWS Budgets when Bedrock spend exceeds 120% of budget. Detach manually to restore service."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Deny"
      Action = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
      ]
      Resource = "*"
    }]
  })

  tags = local.tags
}

resource "aws_budgets_budget_action" "bedrock_kill_switch" {
  budget_name        = aws_budgets_budget.bedrock.name
  action_type        = "APPLY_IAM_POLICY"
  approval_model     = "AUTOMATIC"
  notification_type  = "ACTUAL"
  execution_role_arn = aws_iam_role.budget_executor.arn

  action_threshold {
    action_threshold_type  = "PERCENTAGE"
    action_threshold_value = 120
  }

  definition {
    iam_action_definition {
      policy_arn = aws_iam_policy.bedrock_kill_switch.arn
      roles      = [module.lambda_chat.role_name]
    }
  }

  subscriber {
    address           = var.alarm_email
    subscription_type = "EMAIL"
  }
}
