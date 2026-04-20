output "bucket_name" {
  description = "Name of the attachments S3 bucket."
  value       = aws_s3_bucket.attachments.id
}

output "bucket_arn" {
  description = "ARN of the attachments S3 bucket."
  value       = aws_s3_bucket.attachments.arn
}

output "table_name" {
  description = "Name of the attachments DynamoDB table."
  value       = aws_dynamodb_table.attachments.name
}

output "table_arn" {
  description = "ARN of the attachments DynamoDB table."
  value       = aws_dynamodb_table.attachments.arn
}

output "table_gsi_arn" {
  description = "ARN of the conversationId-createdAt GSI on the attachments table."
  value       = "${aws_dynamodb_table.attachments.arn}/index/conversationId-createdAt-index"
}
