locals {
  lambda_zip_path = "${path.module}/../../../backend/lambda.zip"

  lambda_env = {
    COGNITO_USER_POOL_ID       = module.cognito.user_pool_id
    COGNITO_SPA_CLIENT_ID      = module.cognito.spa_client_id
    CONVERSATIONS_TABLE        = module.dynamodb.conversations_table_name
    MESSAGES_TABLE             = module.dynamodb.messages_table_name
    ATTACHMENTS_TABLE          = module.attachments.table_name
    ATTACHMENTS_BUCKET         = module.attachments.bucket_name
    ATTACHMENTS_MAX_SIZE_BYTES = "52428800"
    ATTACHMENTS_MAX_TEXT_BYTES = "512000"
    BEDROCK_MODEL_ID           = var.bedrock_model_id
    MESSAGE_TTL_DAYS           = tostring(var.message_ttl_days)
  }

  bedrock_model_arns = [
    "arn:aws:bedrock:*::foundation-model/anthropic.claude-*",
    "arn:aws:bedrock:*:${local.account_id}:inference-profile/*",
  ]
}

module "lambda_chat" {
  source = "../../modules/lambda"

  function_name   = "anna-chat-${var.env}-chat"
  handler         = "anna_chat.handlers.chat.handler"
  zip_path        = local.lambda_zip_path
  timeout_seconds = 60
  memory_mb       = 1024

  environment_variables = merge(local.lambda_env, {
    AWS_LAMBDA_LOG_FORMAT = "JSON"
  })

  log_retention_days = var.log_retention_days
  logs_kms_key_arn   = module.kms_logs.key_arn

  vpc_id         = module.network.vpc_id
  vpc_cidr       = module.network.vpc_cidr
  vpc_subnet_ids = module.network.private_subnet_ids

  dynamodb_table_arns = [
    module.dynamodb.conversations_table_arn,
    module.dynamodb.messages_table_arn,
    module.attachments.table_arn,
  ]
  kms_key_arns       = [module.kms_dynamodb.key_arn]
  bedrock_model_arns = local.bedrock_model_arns

  tags = local.tags
}

module "lambda_conversations" {
  source = "../../modules/lambda"

  function_name   = "anna-chat-${var.env}-conversations"
  handler         = "anna_chat.handlers.conversations.handler"
  zip_path        = local.lambda_zip_path
  timeout_seconds = 15
  memory_mb       = 512

  environment_variables = merge(local.lambda_env, {
    AWS_LAMBDA_LOG_FORMAT = "JSON"
  })

  log_retention_days = var.log_retention_days
  logs_kms_key_arn   = module.kms_logs.key_arn

  vpc_id         = module.network.vpc_id
  vpc_cidr       = module.network.vpc_cidr
  vpc_subnet_ids = module.network.private_subnet_ids

  dynamodb_table_arns = [
    module.dynamodb.conversations_table_arn,
    module.dynamodb.messages_table_arn,
  ]
  kms_key_arns = [module.kms_dynamodb.key_arn]

  tags = local.tags
}

# Streaming chat Lambda was deliberately removed in Phase 6b cleanup.
# Python Lambda does not support native response streaming (Node.js does).
# To re-enable: rewrite as Node.js OR Python + Lambda Web Adapter, then
# re-add this module block + the CloudFront /api/chat-stream behavior in
# edge.tf, and add an aws_lambda_permission for CloudFront OAC.

# ──────────────────────────────────────────────────────────────────────────
# Phase 6c: attachments pipeline
# ──────────────────────────────────────────────────────────────────────────

# Extraction Lambda runs async off S3 ObjectCreated events. Has access to
# the attachments bucket (GetObject), attachments DDB table (RW incl. GSI),
# both the S3 and DynamoDB CMKs (Decrypt), and Textract for PDFs/images.
module "lambda_extract" {
  source = "../../modules/lambda"

  function_name   = "anna-chat-${var.env}-extract"
  handler         = "anna_chat.handlers.extract.handler"
  zip_path        = local.lambda_zip_path
  timeout_seconds = 300
  memory_mb       = 2048

  environment_variables = merge(local.lambda_env, {
    AWS_LAMBDA_LOG_FORMAT = "JSON"
  })

  log_retention_days = var.log_retention_days
  logs_kms_key_arn   = module.kms_logs.key_arn

  vpc_id         = module.network.vpc_id
  vpc_cidr       = module.network.vpc_cidr
  vpc_subnet_ids = module.network.private_subnet_ids

  dynamodb_table_arns = [module.attachments.table_arn]
  kms_key_arns = [
    module.kms_s3.key_arn,
    module.kms_dynamodb.key_arn,
  ]
  s3_bucket_arns   = [module.attachments.bucket_arn]
  textract_enabled = true

  tags = local.tags
}

module "attachments" {
  source = "../../modules/attachments"

  env                  = var.env
  kms_key_arn          = module.kms_s3.key_arn
  dynamodb_kms_key_arn = module.kms_dynamodb.key_arn
  cors_allow_origins   = var.cors_allow_origins
  extract_lambda_arn   = module.lambda_extract.function_arn
  deletion_protection  = var.env == "prod"

  tags = local.tags
}

# Attachments CRUD handler — presigns POSTs, lists, deletes. Needs the
# conversations table for ownership checks and the attachments table for
# metadata. S3 access is scoped to the attachments bucket.
module "lambda_attachments" {
  source = "../../modules/lambda"

  function_name   = "anna-chat-${var.env}-attachments"
  handler         = "anna_chat.handlers.attachments.handler"
  zip_path        = local.lambda_zip_path
  timeout_seconds = 15
  memory_mb       = 512

  environment_variables = merge(local.lambda_env, {
    AWS_LAMBDA_LOG_FORMAT = "JSON"
  })

  log_retention_days = var.log_retention_days
  logs_kms_key_arn   = module.kms_logs.key_arn

  vpc_id         = module.network.vpc_id
  vpc_cidr       = module.network.vpc_cidr
  vpc_subnet_ids = module.network.private_subnet_ids

  dynamodb_table_arns = [
    module.attachments.table_arn,
    module.dynamodb.conversations_table_arn,
  ]
  kms_key_arns   = [module.kms_dynamodb.key_arn]
  s3_bucket_arns = [module.attachments.bucket_arn]

  tags = local.tags
}

module "api" {
  source = "../../modules/api"

  env                   = var.env
  cognito_user_pool_id  = module.cognito.user_pool_id
  cognito_spa_client_id = module.cognito.spa_client_id

  cors_allow_origins = var.cors_allow_origins

  log_retention_days = var.log_retention_days
  logs_kms_key_arn   = module.kms_logs.key_arn

  routes = {
    "POST /chat" = {
      lambda_function_name = module.lambda_chat.function_name
      lambda_invoke_arn    = module.lambda_chat.invoke_arn
    }
    "GET /conversations" = {
      lambda_function_name = module.lambda_conversations.function_name
      lambda_invoke_arn    = module.lambda_conversations.invoke_arn
    }
    "GET /conversations/{conversationId}/messages" = {
      lambda_function_name = module.lambda_conversations.function_name
      lambda_invoke_arn    = module.lambda_conversations.invoke_arn
    }
    "DELETE /conversations/{conversationId}" = {
      lambda_function_name = module.lambda_conversations.function_name
      lambda_invoke_arn    = module.lambda_conversations.invoke_arn
    }
    "POST /attachments/presigned-upload" = {
      lambda_function_name = module.lambda_attachments.function_name
      lambda_invoke_arn    = module.lambda_attachments.invoke_arn
    }
    "GET /conversations/{conversationId}/attachments" = {
      lambda_function_name = module.lambda_attachments.function_name
      lambda_invoke_arn    = module.lambda_attachments.invoke_arn
    }
    "DELETE /attachments/{attachmentId}" = {
      lambda_function_name = module.lambda_attachments.function_name
      lambda_invoke_arn    = module.lambda_attachments.invoke_arn
    }
  }

  tags = local.tags
}
