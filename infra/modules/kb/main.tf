data "aws_caller_identity" "current" {}

locals {
  name_prefix = "anna-chat-${var.env}"
  bucket_name = "${local.name_prefix}-kb-${data.aws_caller_identity.current.account_id}"
  table_name  = "${local.name_prefix}-kb"
}

# ──────────────────────────────────────────────────────────────────────────
# S3 bucket for knowledge-base source documents
#
# Modeled on modules/attachments but with two deliberate differences:
#   1. No Object Lock — KB content is editable admin-managed reference
#      material, not audit-immutable user uploads.
#   2. Longer lifecycle (365d current / 60d noncurrent) — clinical protocols,
#      training modules, and research are kept much longer than ephemeral
#      chat attachments.
# ──────────────────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "kb" {
  bucket = local.bucket_name

  tags = merge(var.tags, { Name = local.bucket_name })
}

resource "aws_s3_bucket_versioning" "kb" {
  bucket = aws_s3_bucket.kb.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "kb" {
  bucket = aws_s3_bucket.kb.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = var.kms_key_arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "kb" {
  bucket = aws_s3_bucket.kb.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "kb" {
  bucket = aws_s3_bucket.kb.id

  rule {
    id     = "expire-kb-docs"
    status = "Enabled"

    filter {}

    expiration {
      days = 365
    }

    noncurrent_version_expiration {
      noncurrent_days = 60
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

data "aws_iam_policy_document" "kb_bucket" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions   = ["s3:*"]
    resources = [aws_s3_bucket.kb.arn, "${aws_s3_bucket.kb.arn}/*"]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "kb" {
  bucket = aws_s3_bucket.kb.id
  policy = data.aws_iam_policy_document.kb_bucket.json
}

resource "aws_s3_bucket_cors_configuration" "kb" {
  bucket = aws_s3_bucket.kb.id

  cors_rule {
    allowed_origins = var.cors_allow_origins
    allowed_methods = ["POST", "PUT"]
    # Same explicit allowlist used on the attachments bucket after the
    # Phase 6c security hardening. Keep in sync if you add new SDK-generated
    # headers.
    allowed_headers = [
      "content-type",
      "authorization",
      "x-amz-date",
      "x-amz-content-sha256",
      "x-amz-security-token",
      "x-amz-user-agent",
    ]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# ──────────────────────────────────────────────────────────────────────────
# S3 → ingestion Lambda notification
# ──────────────────────────────────────────────────────────────────────────

resource "aws_lambda_permission" "s3_invoke_kb_ingest" {
  statement_id  = "AllowS3InvokeKbIngest"
  action        = "lambda:InvokeFunction"
  function_name = var.ingest_lambda_arn
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.kb.arn
}

resource "aws_s3_bucket_notification" "kb" {
  bucket = aws_s3_bucket.kb.id

  lambda_function {
    lambda_function_arn = var.ingest_lambda_arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "kb/"
  }

  depends_on = [aws_lambda_permission.s3_invoke_kb_ingest]
}

# ──────────────────────────────────────────────────────────────────────────
# DynamoDB table for KB document metadata + chunks
#
# Single table stores META items (sk = "META") alongside chunk items
# (sk = "chunk#NNNN"). See docs/KB_CONTRACT.md for the schema.
# ──────────────────────────────────────────────────────────────────────────

resource "aws_dynamodb_table" "kb" {
  name         = local.table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "kbDocId"
  range_key    = "sk"

  attribute {
    name = "kbDocId"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = var.dynamodb_kms_key_arn
  }

  point_in_time_recovery {
    enabled = true
  }

  deletion_protection_enabled = var.deletion_protection

  tags = merge(var.tags, { Name = local.table_name })
}
