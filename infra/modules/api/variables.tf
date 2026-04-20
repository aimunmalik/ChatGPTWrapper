variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "cognito_user_pool_id" {
  description = "Cognito user pool that issues JWTs used to authorize requests."
  type        = string
}

variable "cognito_spa_client_id" {
  description = "Cognito SPA app client ID; checked as the JWT audience."
  type        = string
}

variable "routes" {
  description = <<-EOT
    Map from route key (e.g. "POST /chat") to the Lambda it targets.
    Each value is an object with:
      lambda_function_name = string
      lambda_invoke_arn    = string
  EOT
  type = map(object({
    lambda_function_name = string
    lambda_invoke_arn    = string
  }))
}

variable "cors_allow_origins" {
  description = "Origins allowed by CORS. Include both local dev and the CloudFront domain."
  type        = list(string)
  default     = ["http://localhost:5173"]
}

variable "log_retention_days" {
  description = "API Gateway access-log retention."
  type        = number
  default     = 30
}

variable "logs_kms_key_arn" {
  description = "KMS CMK ARN for encrypting API Gateway access logs."
  type        = string
  default     = null
}

variable "tags" {
  description = "Tags applied to API Gateway resources."
  type        = map(string)
  default     = {}
}
