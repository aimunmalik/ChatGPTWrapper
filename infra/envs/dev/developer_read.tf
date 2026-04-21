# ──────────────────────────────────────────────────────────────────────────
# Developer read-only policy
#
# Grants the developer IAM user logs / ddb-read / lambda-read / textract-read
# on anna-chat-* resources so we can tail CloudWatch when Lambdas misbehave
# without falling back to DevTools round-trips. Scoped tight enough that a
# compromise of the dev user can't touch prod resources or write anything.
#
# NOT attached from Terraform — see note below the policy resource. Attach
# manually via the AWS console or:
#   aws iam attach-user-policy --user-name <dev-user> --policy-arn <arn>
# ──────────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "developer_read" {
  # CloudWatch Logs: read events from every anna-chat-dev-* log group. This
  # is the big one — it's what `aws logs tail` + `filter-log-events` need.
  statement {
    sid    = "LogsRead"
    effect = "Allow"
    actions = [
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
      "logs:FilterLogEvents",
      "logs:GetLogEvents",
      "logs:StartQuery",
      "logs:StopQuery",
      "logs:GetQueryResults",
    ]
    resources = [
      "arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/lambda/anna-chat-${var.env}-*",
      "arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/lambda/anna-chat-${var.env}-*:*",
      "arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/apigateway/anna-chat-${var.env}-*",
      "arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/apigateway/anna-chat-${var.env}-*:*",
    ]
  }

  # DDB: read-only on anna-chat-dev-* tables so we can inspect conversation
  # / message / kb state when debugging. No write actions.
  statement {
    sid    = "DynamoDBRead"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:Query",
      "dynamodb:Scan",
      "dynamodb:DescribeTable",
      "dynamodb:ListTables",
    ]
    resources = [
      "arn:aws:dynamodb:${var.aws_region}:${local.account_id}:table/anna-chat-${var.env}-*",
      "arn:aws:dynamodb:${var.aws_region}:${local.account_id}:table/anna-chat-${var.env}-*/index/*",
    ]
  }

  # Lambda read: see which version is deployed, describe config, etc.
  statement {
    sid    = "LambdaRead"
    effect = "Allow"
    actions = [
      "lambda:GetFunction",
      "lambda:GetFunctionConfiguration",
      "lambda:ListFunctions",
      "lambda:ListVersionsByFunction",
    ]
    resources = [
      "arn:aws:lambda:${var.aws_region}:${local.account_id}:function:anna-chat-${var.env}-*",
    ]
  }
}

resource "aws_iam_policy" "developer_read" {
  name        = "anna-chat-${var.env}-developer-read"
  description = "Read-only access to anna-chat-${var.env} CloudWatch Logs, DDB, and Lambda metadata for debugging."
  policy      = data.aws_iam_policy_document.developer_read.json

  tags = local.tags
}

# NOTE: We intentionally do NOT attach this policy from Terraform. The CI
# deploy role was scoped in Tier C to remove iam:Attach*Policy — exactly
# so a compromised PR couldn't elevate a user to admin. Attach manually:
#
#   aws iam attach-user-policy \
#     --user-name lucent-dev \
#     --policy-arn arn:aws:iam::<account>:policy/anna-chat-dev-developer-read
#
# Or click through the AWS console → IAM → Users → lucent-dev → Add
# permissions → Attach policies → anna-chat-dev-developer-read.

output "developer_read_policy_arn" {
  description = "ARN of the developer read-only policy. Attach to a developer IAM user manually (CI role can't attach by design)."
  value       = aws_iam_policy.developer_read.arn
}
