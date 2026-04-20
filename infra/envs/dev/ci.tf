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

# Tier C #1: swapped AdministratorAccess for PowerUserAccess + a scoped IAM
# inline policy. PowerUser covers every service Terraform touches
# (Lambda/APIGW/S3/CloudFront/DDB/KMS/Cognito/WAF/VPC/Logs/Bedrock/Budgets/
# SNS/ACM/GuardDuty) but excludes IAM write, Organizations, and Billing. The
# inline policy below re-grants just the IAM actions Terraform needs, scoped
# to anna-chat-* resources so a compromised PR can't create arbitrary roles
# or escalate to admin.
resource "aws_iam_role_policy_attachment" "github_actions_poweruser" {
  role       = aws_iam_role.github_actions.name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

resource "aws_iam_role_policy" "github_actions_iam_scoped" {
  name = "iam-scoped"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "RolesScopedToAnnaChatPrefix"
        Effect = "Allow"
        Action = [
          "iam:CreateRole",
          "iam:DeleteRole",
          "iam:UpdateRole",
          "iam:UpdateRoleDescription",
          "iam:UpdateAssumeRolePolicy",
          "iam:PutRolePolicy",
          "iam:DeleteRolePolicy",
          "iam:AttachRolePolicy",
          "iam:DetachRolePolicy",
          "iam:TagRole",
          "iam:UntagRole",
          "iam:PassRole",
          "iam:ListAttachedRolePolicies",
          "iam:ListRolePolicies",
          "iam:GetRole",
          "iam:GetRolePolicy",
        ]
        Resource = [
          "arn:aws:iam::${local.account_id}:role/anna-chat-*",
          "arn:aws:iam::${local.account_id}:role/service-role/anna-chat-*",
          "arn:aws:iam::${local.account_id}:role/aws-service-role/*",
        ]
      },
      {
        Sid    = "PoliciesScopedToAnnaChatPrefix"
        Effect = "Allow"
        Action = [
          "iam:CreatePolicy",
          "iam:DeletePolicy",
          "iam:CreatePolicyVersion",
          "iam:DeletePolicyVersion",
          "iam:SetDefaultPolicyVersion",
          "iam:TagPolicy",
          "iam:UntagPolicy",
          "iam:GetPolicy",
          "iam:GetPolicyVersion",
          "iam:ListPolicyVersions",
        ]
        Resource = [
          "arn:aws:iam::${local.account_id}:policy/anna-chat-*",
        ]
      },
      {
        Sid      = "ServiceLinkedRoleCreation"
        Effect   = "Allow"
        Action   = "iam:CreateServiceLinkedRole"
        Resource = "arn:aws:iam::${local.account_id}:role/aws-service-role/*"
      },
      {
        Sid    = "OIDCProviderForBootstrap"
        Effect = "Allow"
        Action = [
          "iam:GetOpenIDConnectProvider",
          "iam:ListOpenIDConnectProviders",
          "iam:TagOpenIDConnectProvider",
          "iam:UntagOpenIDConnectProvider",
        ]
        Resource = "arn:aws:iam::${local.account_id}:oidc-provider/*"
      },
      {
        Sid    = "ReadOnlyAcrossAccount"
        Effect = "Allow"
        Action = [
          "iam:List*",
          "iam:Get*",
          "iam:SimulatePrincipalPolicy",
          "iam:SimulateCustomPolicy",
        ]
        Resource = "*"
      },
    ]
  })
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
