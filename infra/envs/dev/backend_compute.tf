locals {
  lambda_zip_path = "${path.module}/../../../backend/lambda.zip"

  lambda_env = {
    COGNITO_USER_POOL_ID  = module.cognito.user_pool_id
    COGNITO_SPA_CLIENT_ID = module.cognito.spa_client_id
    CONVERSATIONS_TABLE   = module.dynamodb.conversations_table_name
    MESSAGES_TABLE        = module.dynamodb.messages_table_name
    BEDROCK_MODEL_ID      = var.bedrock_model_id
    MESSAGE_TTL_DAYS      = tostring(var.message_ttl_days)
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
  }

  tags = local.tags
}
