output "bucket_name" {
  description = "Name of the knowledge-base S3 bucket."
  value       = aws_s3_bucket.kb.id
}

output "bucket_arn" {
  description = "ARN of the knowledge-base S3 bucket."
  value       = aws_s3_bucket.kb.arn
}

output "table_name" {
  description = "Name of the knowledge-base DynamoDB table."
  value       = aws_dynamodb_table.kb.name
}

output "table_arn" {
  description = "ARN of the knowledge-base DynamoDB table."
  value       = aws_dynamodb_table.kb.arn
}
