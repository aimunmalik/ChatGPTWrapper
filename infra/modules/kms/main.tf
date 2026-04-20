data "aws_caller_identity" "current" {}

resource "aws_kms_key" "this" {
  description             = var.description
  deletion_window_in_days = var.deletion_window_in_days
  enable_key_rotation     = true
  key_usage               = "ENCRYPT_DECRYPT"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid       = "EnableRootPermissions"
          Effect    = "Allow"
          Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
          Action    = "kms:*"
          Resource  = "*"
        }
      ],
      length(var.service_principals) > 0 ? [
        {
          Sid       = "AllowAWSServices"
          Effect    = "Allow"
          Principal = { Service = var.service_principals }
          Action = [
            "kms:Encrypt",
            "kms:Decrypt",
            "kms:ReEncrypt*",
            "kms:GenerateDataKey*",
            "kms:DescribeKey"
          ]
          Resource = "*"
        }
      ] : [],
      length(var.allowed_iam_principal_arns) > 0 ? [
        {
          Sid       = "AllowNamedIAMPrincipals"
          Effect    = "Allow"
          Principal = { AWS = var.allowed_iam_principal_arns }
          Action = [
            "kms:Encrypt",
            "kms:Decrypt",
            "kms:ReEncrypt*",
            "kms:GenerateDataKey*",
            "kms:DescribeKey"
          ]
          Resource = "*"
        }
      ] : []
    )
  })

  tags = merge(var.tags, {
    Name = var.alias_suffix
  })
}

resource "aws_kms_alias" "this" {
  name          = "alias/anna-chat-${var.alias_suffix}"
  target_key_id = aws_kms_key.this.key_id
}
