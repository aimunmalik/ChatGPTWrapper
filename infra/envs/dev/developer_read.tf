# ──────────────────────────────────────────────────────────────────────────
# Developer read-only policy
#
# Grants the developer IAM user logs / ddb-read / lambda-read / textract-read
# on anna-chat-* resources so we can tail CloudWatch when Lambdas misbehave
# without falling back to DevTools round-trips. Scoped tight enough that a
# compromise of the dev user can't touch prod resources or write anything.
#
# Attach to the user by name via var.developer_iam_user_name (tfvars). Leave
# blank to skip attaching — the policy still exists for future use.
# ──────────────────────────────────────────────────────────────────────────

variable "developer_iam_user_name" {
  description = "IAM user to grant read-only observability on anna-chat-* resources. Empty string disables attachment."
  type        = string
  default     = "lucent-dev"
}

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

# Attach to the developer user. Skipped entirely when the var is blank so
# the policy can exist without requiring any particular user to exist.
resource "aws_iam_user_policy_attachment" "developer_read" {
  count      = var.developer_iam_user_name == "" ? 0 : 1
  user       = var.developer_iam_user_name
  policy_arn = aws_iam_policy.developer_read.arn
}
