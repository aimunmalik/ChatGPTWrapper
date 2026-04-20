output "table_name" {
  description = "Name of the prompts DynamoDB table."
  value       = aws_dynamodb_table.prompts.name
}

output "table_arn" {
  description = "ARN of the prompts DynamoDB table."
  value       = aws_dynamodb_table.prompts.arn
}
