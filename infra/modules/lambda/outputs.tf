output "function_name" {
  description = "Name of the Lambda function."
  value       = aws_lambda_function.this.function_name
}

output "function_arn" {
  description = "ARN of the Lambda function."
  value       = aws_lambda_function.this.arn
}

output "invoke_arn" {
  description = "Invoke ARN (used by API Gateway integrations)."
  value       = aws_lambda_function.this.invoke_arn
}

output "role_arn" {
  description = "IAM role ARN the Lambda assumes."
  value       = aws_iam_role.this.arn
}

output "role_name" {
  description = "IAM role name."
  value       = aws_iam_role.this.name
}

output "log_group_name" {
  description = "CloudWatch log group for the function."
  value       = aws_cloudwatch_log_group.this.name
}

output "function_url" {
  description = "Lambda Function URL (null when not enabled). Format: https://<id>.lambda-url.<region>.on.aws/"
  value       = var.function_url_enabled ? aws_lambda_function_url.this[0].function_url : null
}

output "function_url_domain" {
  description = "Domain-only form of the Function URL (for use as a CloudFront origin)."
  value       = var.function_url_enabled ? replace(replace(aws_lambda_function_url.this[0].function_url, "https://", ""), "/", "") : null
}
