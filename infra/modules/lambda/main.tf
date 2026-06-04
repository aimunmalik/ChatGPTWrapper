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
        # Scan is needed by the KB repo (list_docs + scan_all_chunks for RAG
        # retrieval). Still scoped to the specific table ARNs passed in, so
        # this doesn't widen which tables each Lambda can reach.
        "dynamodb:Scan",
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
      # S3/DDB server-side encryption flows use GenerateDataKey on the CMK
      # and Decrypt on retrieval; direct Encrypt/ReEncrypt aren't needed.
      # Keeping the action set minimal reduces the blast radius if the
      # function role is ever compromised.
      sid    = "KmsAccess"
      effect = "Allow"
      actions = [
        "kms:Decrypt",
        "kms:GenerateDataKey*",
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

  dynamic "statement" {
    for_each = length(var.s3_bucket_arns) > 0 ? [1] : []
    content {
      sid    = "S3ObjectAccess"
      effect = "Allow"
      actions = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:AbortMultipartUpload",
      ]
      resources = [for arn in var.s3_bucket_arns : "${arn}/*"]
    }
  }

  dynamic "statement" {
    for_each = length(var.s3_bucket_arns) > 0 ? [1] : []
    content {
      sid    = "S3BucketList"
      effect = "Allow"
      actions = [
        "s3:ListBucket",
        "s3:GetBucketLocation",
      ]
      resources = var.s3_bucket_arns
    }
  }

  dynamic "statement" {
    for_each = var.textract_enabled ? [1] : []
    content {
      sid    = "TextractAccess"
      effect = "Allow"
      actions = [
        "textract:DetectDocumentText",
        "textract:StartDocumentTextDetection",
        "textract:GetDocumentTextDetection",
      ]
      resources = ["*"]
    }
  }

  dynamic "statement" {
    for_each = length(var.lambda_invoke_function_arns) > 0 ? [1] : []
    content {
      # Async fan-out (InvocationType=Event) to a worker Lambda. Scoped to
      # the specific worker ARNs the caller passes — never wildcard.
      sid    = "LambdaInvoke"
      effect = "Allow"
      actions = [
        "lambda:InvokeFunction",
      ]
      resources = var.lambda_invoke_function_arns
    }
  }

  dynamic "statement" {
    for_each = length(var.cognito_user_pool_arn) > 0 ? [1] : []
    content {
      # Admin user-management actions for the Praxis admin panel. Scoped to
      # the single user pool ARN passed in — never wildcard. Action set
      # mirrors the routes in docs/ADMIN_USERS_CONTRACT.md (list, invite,
      # group toggle, enable/disable, force sign-out).
      sid    = "CognitoAdminUserManagement"
      effect = "Allow"
      actions = [
        "cognito-idp:ListUsers",
        "cognito-idp:ListUsersInGroup",
        "cognito-idp:AdminGetUser",
        "cognito-idp:AdminCreateUser",
        "cognito-idp:AdminAddUserToGroup",
        "cognito-idp:AdminRemoveUserFromGroup",
        "cognito-idp:AdminEnableUser",
        "cognito-idp:AdminDisableUser",
        "cognito-idp:AdminUserGlobalSignOut",
        "cognito-idp:AdminListGroupsForUser",
      ]
      resources = [var.cognito_user_pool_arn]
    }
  }

  dynamic "statement" {
    for_each = var.dlq_enabled ? [1] : []
    content {
      # Lets the function deliver failed async (Event) invocations to its own
      # dead-letter queue. Scoped to this function's DLQ ARN only.
      sid       = "DlqSendMessage"
      effect    = "Allow"
      actions   = ["sqs:SendMessage"]
      resources = [aws_sqs_queue.dlq[0].arn]
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

# ──────────────────────────────────────────────────────────────────────────
# Optional dead-letter queue for async (InvocationType=Event) failures.
# The Lambda service retries an async invoke up to maximum_retry_attempts, then
# drops the event silently — for fire-and-forget workers (translate, chat) that
# means a failed job just vanishes. Capturing it in a DLQ + alarming on depth
# (alarm lives in the env's monitoring.tf) turns that into a visible, replayable
# signal.
# ──────────────────────────────────────────────────────────────────────────

resource "aws_sqs_queue" "dlq" {
  count                     = var.dlq_enabled ? 1 : 0
  name                      = "${var.function_name}-dlq"
  message_retention_seconds = var.dlq_message_retention_seconds

  # Async event payloads can carry user content (PHI). Encrypt at rest with the
  # provided CMK; fall back to SSE-SQS (AWS-managed) only when no CMK is passed.
  # Only one of kms_master_key_id / sqs_managed_sse_enabled may be set, so the
  # SSE flag is left unset (null) whenever a CMK is provided.
  kms_master_key_id                 = var.dlq_kms_key_arn
  kms_data_key_reuse_period_seconds = var.dlq_kms_key_arn != null ? 300 : null
  sqs_managed_sse_enabled           = var.dlq_kms_key_arn == null ? true : null

  tags = merge(var.tags, { Name = "${var.function_name}-dlq" })
}

resource "aws_lambda_function_event_invoke_config" "this" {
  count                        = var.dlq_enabled ? 1 : 0
  function_name                = aws_lambda_function.this.function_name
  maximum_retry_attempts       = 2
  maximum_event_age_in_seconds = 3600

  destination_config {
    on_failure {
      destination = aws_sqs_queue.dlq[0].arn
    }
  }
}
