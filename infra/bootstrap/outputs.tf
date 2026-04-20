output "tf_state_bucket" {
  description = "Name of the S3 bucket holding Terraform state. Copy into every env's backend.tf."
  value       = aws_s3_bucket.tf_state.id
}

output "tf_lock_table" {
  description = "Name of the DynamoDB table used for Terraform state locking."
  value       = aws_dynamodb_table.tf_lock.name
}

output "github_oidc_arn" {
  description = "ARN of the GitHub Actions OIDC provider. Used in IAM role trust policies for CI."
  value       = aws_iam_openid_connect_provider.github_actions.arn
}

output "account_id" {
  description = "AWS account ID this bootstrap ran in."
  value       = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  description = "AWS region the state backend lives in."
  value       = var.aws_region
}
