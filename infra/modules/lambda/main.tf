locals {
  has_vpc = length(var.vpc_subnet_ids) > 0
}

resource "aws_cloudwatch_log_group" "this" {
  name              = "/aws/lambda/${var.function_name}"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.logs_kms_key_arn
  tags              = var.tags
}

resource "aws_iam_role" "this" {
  name = "${var.function_name}-role"
  path = "/service-role/"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "basic_execution" {
  role       = aws_iam_role.this.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "vpc_execution" {
  count      = local.has_vpc ? 1 : 0
  role       = aws_iam_role.this.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

data "aws_iam_policy_document" "inline" {
  dynamic "statement" {
    for_each = length(var.dynamodb_table_arns) > 0 ? [1] : []
    content {
      sid    = "DynamoDBAccess"
      effect = "Allow"
      actions = [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query",
        "dynamodb:BatchWriteItem",
        "dynamodb:BatchGetItem",
        "dynamodb:DescribeTable",
      ]
      resources = concat(
        var.dynamodb_table_arns,
        [for arn in var.dynamodb_table_arns : "${arn}/index/*"],
      )
    }
  }

  dynamic "statement" {
    for_each = length(var.kms_key_arns) > 0 ? [1] : []
    content {
      sid    = "KmsAccess"
      effect = "Allow"
      actions = [
        "kms:Decrypt",
        "kms:Encrypt",
        "kms:GenerateDataKey",
        "kms:DescribeKey",
      ]
      resources = var.kms_key_arns
    }
  }

  dynamic "statement" {
    for_each = length(var.bedrock_model_arns) > 0 ? [1] : []
    content {
      sid    = "BedrockInvoke"
      effect = "Allow"
      actions = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
      ]
      resources = var.bedrock_model_arns
    }
  }
}

resource "aws_iam_role_policy" "inline" {
  count  = length(data.aws_iam_policy_document.inline.statement) > 0 ? 1 : 0
  name   = "${var.function_name}-inline"
  role   = aws_iam_role.this.id
  policy = data.aws_iam_policy_document.inline.json
}

resource "aws_security_group" "this" {
  count       = local.has_vpc ? 1 : 0
  name        = "${var.function_name}-sg"
  description = "SG for Lambda ${var.function_name}; outbound 443 to VPC only."
  vpc_id      = var.vpc_id

  tags = merge(var.tags, { Name = "${var.function_name}-sg" })
}

resource "aws_vpc_security_group_egress_rule" "https_out" {
  count             = local.has_vpc ? 1 : 0
  security_group_id = aws_security_group.this[0].id
  description       = "HTTPS outbound (VPC endpoints; gateway endpoints resolve to public IPs but route table keeps traffic on AWS backbone; no NAT means internet is unreachable regardless)."
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_lambda_function" "this" {
  function_name = var.function_name
  role          = aws_iam_role.this.arn
  handler       = var.handler
  runtime       = var.runtime

  filename         = var.zip_path
  source_code_hash = filebase64sha256(var.zip_path)

  timeout     = var.timeout_seconds
  memory_size = var.memory_mb

  architectures = ["x86_64"]

  environment {
    variables = var.environment_variables
  }

  dynamic "vpc_config" {
    for_each = local.has_vpc ? [1] : []
    content {
      subnet_ids         = var.vpc_subnet_ids
      security_group_ids = [aws_security_group.this[0].id]
    }
  }

  tracing_config {
    mode = "PassThrough"
  }

  tags = var.tags

  depends_on = [
    aws_cloudwatch_log_group.this,
    aws_iam_role_policy_attachment.basic_execution,
    aws_iam_role_policy_attachment.vpc_execution,
  ]
}

# ──────────────────────────────────────────────────────────────────────────
# Optional Lambda Function URL (for response streaming — API Gateway HTTP
# APIs don't support Lambda response streaming, Function URLs do).
# ──────────────────────────────────────────────────────────────────────────

resource "aws_lambda_function_url" "this" {
  count              = var.function_url_enabled ? 1 : 0
  function_name      = aws_lambda_function.this.function_name
  authorization_type = "AWS_IAM"
  invoke_mode        = var.function_url_invoke_mode
}

# Lambda permission for CloudFront OAC access is added from the env root
# module (infra/envs/*/edge.tf) to avoid a dependency cycle between the
# lambda and edge modules.
