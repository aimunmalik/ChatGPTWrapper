output "account_id" {
  description = "AWS account ID this environment is deployed to."
  value       = local.account_id
}

output "aws_region" {
  description = "AWS region."
  value       = var.aws_region
}

output "kms_key_arns" {
  description = "KMS key ARNs by domain."
  value = {
    dynamodb = module.kms_dynamodb.key_arn
    logs     = module.kms_logs.key_arn
    secrets  = module.kms_secrets.key_arn
    s3       = module.kms_s3.key_arn
  }
}

output "vpc_id" {
  description = "VPC ID. Lambda gets attached here in Phase 2."
  value       = module.network.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs for Lambda."
  value       = module.network.private_subnet_ids
}

output "vpc_endpoints_security_group_id" {
  description = "Security group Lambda SGs must be allowed to reach on 443."
  value       = module.network.vpc_endpoints_security_group_id
}

output "cognito" {
  description = "Cognito user pool details for the frontend and backend."
  value = {
    user_pool_id  = module.cognito.user_pool_id
    user_pool_arn = module.cognito.user_pool_arn
    spa_client_id = module.cognito.spa_client_id
    hosted_ui_url = module.cognito.user_pool_domain_url
    jwks_uri      = module.cognito.jwks_uri
    issuer        = module.cognito.issuer
  }
}

output "cognito_user_pool_id" {
  description = "Flat accessor for the user pool ID (convenient for CLI use)."
  value       = module.cognito.user_pool_id
}

output "cognito_spa_client_id" {
  description = "Flat accessor for the SPA client ID."
  value       = module.cognito.spa_client_id
}

output "dynamodb" {
  description = "DynamoDB table names."
  value = {
    conversations = module.dynamodb.conversations_table_name
    messages      = module.dynamodb.messages_table_name
  }
}
