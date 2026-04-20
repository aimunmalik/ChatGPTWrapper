data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

locals {
  github_repo = "${var.github_org}/${var.github_repo}"
}

data "aws_iam_policy_document" "github_actions_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:${local.github_repo}:ref:refs/heads/main",
        "repo:${local.github_repo}:pull_request",
      ]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "anna-chat-${var.env}-github-actions"
  description        = "Assumed by GitHub Actions workflows in ${local.github_repo} to deploy anna-chat ${var.env}."
  assume_role_policy = data.aws_iam_policy_document.github_actions_trust.json
  max_session_duration = 3600

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "github_actions_admin" {
  # Phase 4 MVP: broad permissions for simplicity. Scope down in Phase 5 (prod hardening)
  # — the role only needs Lambda/ApiGateway/S3/CloudFront/DynamoDB/IAM/KMS/Cognito/WAF/EC2/Logs.
  role       = aws_iam_role.github_actions.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

# Separate policy for the Terraform state backend so it's explicit in what CI
# touches (even while the role is currently AdminAccess).
resource "aws_iam_role_policy" "github_actions_tfstate" {
  name = "tfstate-access"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
        ]
        Resource = [
          "arn:aws:s3:::anna-chat-tfstate-${local.account_id}",
          "arn:aws:s3:::anna-chat-tfstate-${local.account_id}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:DeleteItem",
        ]
        Resource = "arn:aws:dynamodb:${var.aws_region}:${local.account_id}:table/anna-chat-tfstate-lock"
      },
    ]
  })
}
