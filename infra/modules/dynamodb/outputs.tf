output "conversations_table_name" {
  description = "Name of the conversations table."
  value       = aws_dynamodb_table.conversations.name
}

output "conversations_table_arn" {
  description = "ARN of the conversations table."
  value       = aws_dynamodb_table.conversations.arn
}

output "messages_table_name" {
  description = "Name of the messages table."
  value       = aws_dynamodb_table.messages.name
}

output "messages_table_arn" {
  description = "ARN of the messages table."
  value       = aws_dynamodb_table.messages.arn
}

output "messages_user_index_arn" {
  description = "ARN of the userId-sortKey GSI on messages."
  value       = "${aws_dynamodb_table.messages.arn}/index/userId-sortKey-index"
}
