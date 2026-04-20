output "api_id" {
  description = "API Gateway HTTP API ID."
  value       = aws_apigatewayv2_api.this.id
}

output "api_endpoint" {
  description = "Invoke URL for the API (e.g. https://abc123.execute-api.us-east-1.amazonaws.com)."
  value       = aws_apigatewayv2_api.this.api_endpoint
}

output "authorizer_id" {
  description = "JWT authorizer ID."
  value       = aws_apigatewayv2_authorizer.cognito_jwt.id
}
