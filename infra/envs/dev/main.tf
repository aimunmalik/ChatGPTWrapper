data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id

  tags = {
    Project     = "anna-chat"
    Environment = var.env
    ManagedBy   = "Terraform"
    PHIScope    = var.env == "prod" ? "possible" : "none"
  }
}

module "kms_dynamodb" {
  source       = "../../modules/kms"
  alias_suffix = "${var.env}-dynamodb"
  description  = "Encrypts anna-chat ${var.env} DynamoDB tables."
  tags         = local.tags
}

module "kms_logs" {
  source             = "../../modules/kms"
  alias_suffix       = "${var.env}-logs"
  description        = "Encrypts anna-chat ${var.env} CloudWatch log groups."
  service_principals = ["logs.${var.aws_region}.amazonaws.com"]
  tags               = local.tags
}

module "kms_secrets" {
  source       = "../../modules/kms"
  alias_suffix = "${var.env}-secrets"
  description  = "Encrypts anna-chat ${var.env} Secrets Manager entries."
  tags         = local.tags
}

module "kms_s3" {
  source             = "../../modules/kms"
  alias_suffix       = "${var.env}-s3"
  description        = "Encrypts anna-chat ${var.env} S3 buckets (SPA assets, future attachments)."
  service_principals = ["s3.amazonaws.com"]
  tags               = local.tags
}

module "network" {
  source             = "../../modules/network"
  env                = var.env
  aws_region         = var.aws_region
  vpc_cidr           = var.vpc_cidr
  flow_logs_enabled  = var.flow_logs_enabled
  logs_kms_key_arn   = module.kms_logs.key_arn
  log_retention_days = var.log_retention_days
  tags               = local.tags
}

module "cognito" {
  source                 = "../../modules/cognito"
  env                    = var.env
  domain_suffix          = var.cognito_domain_suffix
  callback_urls          = var.cognito_callback_urls
  logout_urls            = var.cognito_logout_urls
  deletion_protection    = var.env == "prod"
  # Tier B — flip dev from AUDIT (observe-only) to ENFORCED so impossible-
  # travel, credential-stuffing, and compromised-password detections actually
  # BLOCK risky sign-ins rather than just logging them. Extra ~$0.05/MAU +
  # $0.005/event — negligible for internal use.
  advanced_security_mode = "ENFORCED"
  tags                   = local.tags
}

module "dynamodb" {
  source              = "../../modules/dynamodb"
  env                 = var.env
  kms_key_arn         = module.kms_dynamodb.key_arn
  deletion_protection = var.env == "prod"
  tags                = local.tags
}
