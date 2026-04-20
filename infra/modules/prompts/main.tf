locals {
  name_prefix = "anna-chat-${var.env}"
  table_name  = "${local.name_prefix}-prompts"
}

# ──────────────────────────────────────────────────────────────────────────
# DynamoDB table for per-user prompt templates
# ──────────────────────────────────────────────────────────────────────────

resource "aws_dynamodb_table" "prompts" {
  name         = local.table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "userId"
  range_key    = "promptId"

  attribute {
    name = "userId"
    type = "S"
  }

  attribute {
    name = "promptId"
    type = "S"
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = var.kms_key_arn
  }

  point_in_time_recovery {
    enabled = true
  }

  deletion_protection_enabled = var.deletion_protection

  tags = merge(var.tags, { Name = local.table_name })
}
