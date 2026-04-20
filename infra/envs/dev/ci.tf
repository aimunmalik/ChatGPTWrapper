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
      # Trust only workflow runs from the main branch. We deliberately do NOT
      # trust `pull_request` here — doing so means any PR (including from a
      # contributor whose patch modifies the workflow) could assume a role
      # that currently holds AdministratorAccess. The PR-time `terraform-plan`
      # job was removed from .github/workflows/ci.yml as a consequence; the
      # deploy workflow still runs plan+apply on merge to main.
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:${local.github_repo}:ref:refs/heads/main",
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
  # SECURITY TODO (Phase 5 — prod hardening): replace AdministratorAccess with
  # a scoped policy. The role only needs Lambda/ApiGateway/S3/CloudFront/
  # DynamoDB/IAM/KMS/Cognito/WAF/EC2/Logs/GuardDuty. Kept as AdminAccess for
  # now because authoring the scoped policy is a non-trivial effort and the
  # trust policy was narrowed (main branch only, no PR trust) in this PR to
  # shrink the blast radius in the meantime.
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
