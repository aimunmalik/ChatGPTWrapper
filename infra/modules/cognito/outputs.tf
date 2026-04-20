output "user_pool_id" {
  description = "Cognito user pool ID."
  value       = aws_cognito_user_pool.this.id
}

output "user_pool_arn" {
  description = "Cognito user pool ARN."
  value       = aws_cognito_user_pool.this.arn
}

output "user_pool_endpoint" {
  description = "Cognito user pool endpoint (for JWKS)."
  value       = aws_cognito_user_pool.this.endpoint
}

output "user_pool_domain" {
  description = "Cognito hosted UI domain (without the .auth.region.amazoncognito.com suffix)."
  value       = aws_cognito_user_pool_domain.this.domain
}

output "user_pool_domain_url" {
  description = "Full URL of the Cognito hosted UI."
  value       = "https://${aws_cognito_user_pool_domain.this.domain}.auth.${data.aws_region.current.name}.amazoncognito.com"
}

output "spa_client_id" {
  description = "Cognito app client ID used by the SPA."
  value       = aws_cognito_user_pool_client.spa.id
}

output "jwks_uri" {
  description = "JWKS URI for verifying JWTs issued by this user pool."
  value       = "https://cognito-idp.${data.aws_region.current.name}.amazonaws.com/${aws_cognito_user_pool.this.id}/.well-known/jwks.json"
}

output "issuer" {
  description = "OIDC issuer URL. Used by Lambda to verify JWTs."
  value       = "https://cognito-idp.${data.aws_region.current.name}.amazonaws.com/${aws_cognito_user_pool.this.id}"
}

data "aws_region" "current" {}
