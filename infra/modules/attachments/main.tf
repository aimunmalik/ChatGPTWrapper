data "aws_caller_identity" "current" {}

locals {
  name_prefix = "anna-chat-${var.env}"
  bucket_name = "${local.name_prefix}-attachments-${data.aws_caller_identity.current.account_id}"
  table_name  = "${local.name_prefix}-attachments"
}

# ──────────────────────────────────────────────────────────────────────────
# S3 bucket for attachments (user uploads)
# ──────────────────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "attachments" {
  bucket              = local.bucket_name
  object_lock_enabled = true

  tags = merge(var.tags, { Name = local.bucket_name })
}

resource "aws_s3_bucket_versioning" "attachments" {
  bucket = aws_s3_bucket.attachments.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "attachments" {
  bucket = aws_s3_bucket.attachments.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = var.kms_key_arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "attachments" {
  bucket = aws_s3_bucket.attachments.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "attachments" {
  bucket = aws_s3_bucket.attachments.id

  rule {
    id     = "expire-attachments"
    status = "Enabled"

    filter {}

    expiration {
      days = 180
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_object_lock_configuration" "attachments" {
  bucket = aws_s3_bucket.attachments.id

  rule {
    default_retention {
      mode = "GOVERNANCE"
      days = 90
    }
  }

  depends_on = [aws_s3_bucket_versioning.attachments]
}

data "aws_iam_policy_document" "attachments_bucket" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions   = ["s3:*"]
    resources = [aws_s3_bucket.attachments.arn, "${aws_s3_bucket.attachments.arn}/*"]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "attachments" {
  bucket = aws_s3_bucket.attachments.id
  policy = data.aws_iam_policy_document.attachments_bucket.json
}

resource "aws_s3_bucket_cors_configuration" "attachments" {
  bucket = aws_s3_bucket.attachments.id

  cors_rule {
    allowed_origins = var.cors_allow_origins
    allowed_methods = ["POST", "PUT"]
    # Explicit allowlist (was `["*"]`). content-type is the only header the
    # browser strictly requires for presigned POST; the x-amz-* entries are
    # there to cover SigV4 POST variants and credential propagation so a
    # preflight doesn't fail silently. If you ever add a new SDK-generated
    # header here, add it to this list.
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
# S3 → extraction Lambda notification
# ──────────────────────────────────────────────────────────────────────────

resource "aws_lambda_permission" "s3_invoke_extract" {
  statement_id  = "AllowS3InvokeExtract"
  action        = "lambda:InvokeFunction"
  function_name = var.extract_lambda_arn
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.attachments.arn
}

resource "aws_s3_bucket_notification" "attachments" {
  bucket = aws_s3_bucket.attachments.id

  lambda_function {
    lambda_function_arn = var.extract_lambda_arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "attachments/"
  }

  depends_on = [aws_lambda_permission.s3_invoke_extract]
}

# ──────────────────────────────────────────────────────────────────────────
# DynamoDB table for attachment metadata
# ──────────────────────────────────────────────────────────────────────────

resource "aws_dynamodb_table" "attachments" {
  name         = local.table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "userId"
  range_key    = "attachmentId"

  attribute {
    name = "userId"
    type = "S"
  }

  attribute {
    name = "attachmentId"
    type = "S"
  }

  attribute {
    name = "conversationId"
    type = "S"
  }

  attribute {
    name = "createdAt"
    type = "N"
  }

  global_secondary_index {
    name            = "conversationId-createdAt-index"
    hash_key        = "conversationId"
    range_key       = "createdAt"
    projection_type = "ALL"
  }

  ttl {
    enabled        = true
    attribute_name = "ttl"
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

# ──────────────────────────────────────────────────────────────────────────
# GuardDuty Malware Protection for the attachments bucket
#
# HIPAA-adjacent buckets that accept untrusted user uploads need malware
# scanning. GuardDuty's "Malware Protection for S3" plan tags matching
# objects (GuardDutyMalwareScanStatus) so downstream consumers can gate on
# NO_THREATS_FOUND before handing files to the extraction Lambda / model.
#
# GuardDuty detectors are a per-account-per-region singleton. If the account
# already has one (common with AWS Organizations delegated admin), pass the
# ID via var.guardduty_detector_id and this module will use it instead of
# creating a new one.
# ──────────────────────────────────────────────────────────────────────────

# GuardDuty Malware Protection for S3 was intentionally removed 2026-04-20
# because onboarding the scan role kept bumping into undocumented AWS
# permission requirements (managed policy doesn't exist, then S3
# notification perms, then EventBridge rule management, etc.). Every
# deploy was failing at this step and blocking everything else.
#
# Plan to re-enable as a dedicated task:
#   1. Create the malware-protection plan manually via AWS Console
#   2. Have the Console auto-generate the correct IAM role + permissions
#   3. Import both into Terraform
#
# Filed as Medium finding in the Phase 6c security audit; not a blocker
# for dev usage since the attachment flow already requires Cognito auth
# and files are scanned by the extraction Lambda before being shown to
# the model — malware scanning here is defense-in-depth, not the only
# control.
