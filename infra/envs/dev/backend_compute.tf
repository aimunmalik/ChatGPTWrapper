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
    PROMPTS_TABLE              = module.prompts.table_name
    KB_TABLE                   = module.kb.table_name
    KB_BUCKET                  = module.kb.bucket_name
    KB_MAX_SIZE_BYTES          = "104857600"
  }

  bedrock_model_arns = [
    "arn:aws:bedrock:*::foundation-model/anthropic.claude-*",
    "arn:aws:bedrock:*:${local.account_id}:inference-profile/*",
    "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0",
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
    module.kb.table_arn,
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

  # Empty → let the module create the account-level GuardDuty detector. If a
  # detector is already enabled in this region (e.g. via AWS Organizations
  # delegated admin), set this to its ID to skip creation.
  guardduty_detector_id = ""

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
  # Needs the DDB CMK (table encryption) AND the S3 CMK (SSE-KMS on the
  # attachments bucket). S3 uses the caller's credentials — via the
  # presigned URL, those are this Lambda's — to call kms:GenerateDataKey.
  kms_key_arns   = [module.kms_dynamodb.key_arn, module.kms_s3.key_arn]
  s3_bucket_arns = [module.attachments.bucket_arn]

  tags = local.tags
}

# ──────────────────────────────────────────────────────────────────────────
# Phase 6d: per-user prompt library
# ──────────────────────────────────────────────────────────────────────────

module "prompts" {
  source = "../../modules/prompts"

  env                 = var.env
  kms_key_arn         = module.kms_dynamodb.key_arn
  deletion_protection = var.env == "prod"

  tags = local.tags
}

# Prompts CRUD handler — full create/list/update/delete for a user's prompt
# library. Only needs the prompts table and the DDB CMK.
module "lambda_prompts" {
  source = "../../modules/lambda"

  function_name   = "anna-chat-${var.env}-prompts"
  handler         = "anna_chat.handlers.prompts.handler"
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

  dynamodb_table_arns = [module.prompts.table_arn]
  kms_key_arns        = [module.kms_dynamodb.key_arn]

  tags = local.tags
}

# ──────────────────────────────────────────────────────────────────────────
# Phase 7: knowledge base / RAG
# ──────────────────────────────────────────────────────────────────────────

# Ingestion Lambda runs async off S3 ObjectCreated events in the kb/ prefix.
# Needs lots of memory + a long timeout because it extracts text (Textract
# for PDFs), chunks, embeds via Titan, and writes chunks to DDB for the
# whole document in one invocation. Has access to the KB bucket (GetObject),
# KB DDB table (RW), both CMKs (Decrypt), Textract, and Bedrock invoke on
# both Claude (harmless — inherited from the shared list) and Titan
# embeddings (required).
module "lambda_kb_ingest" {
  source = "../../modules/lambda"

  function_name   = "anna-chat-${var.env}-kb-ingest"
  handler         = "anna_chat.handlers.kb_ingest.handler"
  zip_path        = local.lambda_zip_path
  timeout_seconds = 600
  memory_mb       = 3072

  environment_variables = merge(local.lambda_env, {
    AWS_LAMBDA_LOG_FORMAT = "JSON"
  })

  log_retention_days = var.log_retention_days
  logs_kms_key_arn   = module.kms_logs.key_arn

  vpc_id         = module.network.vpc_id
  vpc_cidr       = module.network.vpc_cidr
  vpc_subnet_ids = module.network.private_subnet_ids

  dynamodb_table_arns = [module.kb.table_arn]
  kms_key_arns = [
    module.kms_dynamodb.key_arn,
    module.kms_s3.key_arn,
  ]
  s3_bucket_arns     = [module.kb.bucket_arn]
  textract_enabled   = true
  bedrock_model_arns = local.bedrock_model_arns

  tags = local.tags
}

module "kb" {
  source = "../../modules/kb"

  env                  = var.env
  kms_key_arn          = module.kms_s3.key_arn
  dynamodb_kms_key_arn = module.kms_dynamodb.key_arn
  cors_allow_origins   = var.cors_allow_origins
  ingest_lambda_arn    = module.lambda_kb_ingest.function_arn
  deletion_protection  = var.env == "prod"

  tags = local.tags
}

# Admin CRUD handler — presigns uploads, lists docs, deletes. Needs the KB
# table (RW for metadata + chunk deletion), both CMKs (DDB + S3 via
# presigned URLs that use this Lambda's credentials), and the KB bucket for
# object deletes. No Textract and no Bedrock — those belong to the ingest
# lambda.
module "lambda_kb" {
  source = "../../modules/lambda"

  function_name   = "anna-chat-${var.env}-kb"
  handler         = "anna_chat.handlers.kb.handler"
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

  dynamodb_table_arns = [module.kb.table_arn]
  kms_key_arns        = [module.kms_dynamodb.key_arn, module.kms_s3.key_arn]
  s3_bucket_arns      = [module.kb.bucket_arn]

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
    "POST /prompts" = {
      lambda_function_name = module.lambda_prompts.function_name
      lambda_invoke_arn    = module.lambda_prompts.invoke_arn
    }
    "GET /prompts" = {
      lambda_function_name = module.lambda_prompts.function_name
      lambda_invoke_arn    = module.lambda_prompts.invoke_arn
    }
    "PUT /prompts/{promptId}" = {
      lambda_function_name = module.lambda_prompts.function_name
      lambda_invoke_arn    = module.lambda_prompts.invoke_arn
    }
    "DELETE /prompts/{promptId}" = {
      lambda_function_name = module.lambda_prompts.function_name
      lambda_invoke_arn    = module.lambda_prompts.invoke_arn
    }
    "POST /kb/presigned-upload" = {
      lambda_function_name = module.lambda_kb.function_name
      lambda_invoke_arn    = module.lambda_kb.invoke_arn
    }
    "GET /kb/documents" = {
      lambda_function_name = module.lambda_kb.function_name
      lambda_invoke_arn    = module.lambda_kb.invoke_arn
    }
    "DELETE /kb/documents/{kbDocId}" = {
      lambda_function_name = module.lambda_kb.function_name
      lambda_invoke_arn    = module.lambda_kb.invoke_arn
    }
    # Clinicians open KB source PDFs from chat replies via this route.
    # Not admin-gated on the handler side — any authenticated user can
    # pull a short-lived presigned GET URL.
    "GET /kb/documents/{kbDocId}/download" = {
      lambda_function_name = module.lambda_kb.function_name
      lambda_invoke_arn    = module.lambda_kb.invoke_arn
    }
  }

  tags = local.tags
}
