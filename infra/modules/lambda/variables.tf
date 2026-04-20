variable "function_name" {
  description = "Lambda function name. Also used as prefix for role, SG, log group."
  type        = string
}

variable "handler" {
  description = "Handler in dotted module.attr form, e.g. 'anna_chat.handlers.chat.handler'."
  type        = string
}

variable "runtime" {
  description = "Lambda runtime."
  type        = string
  default     = "python3.12"
}

variable "zip_path" {
  description = "Path to the Lambda deployment zip on the machine running Terraform."
  type        = string
}

variable "timeout_seconds" {
  description = "Function timeout. Default 30s is enough for typical chat turns."
  type        = number
  default     = 30
}

variable "memory_mb" {
  description = "Function memory. More memory also gets more CPU."
  type        = number
  default     = 1024
}

variable "environment_variables" {
  description = "Env vars passed to the Lambda."
  type        = map(string)
  default     = {}
}

variable "log_retention_days" {
  description = "CloudWatch log retention."
  type        = number
  default     = 30
}

variable "logs_kms_key_arn" {
  description = "KMS CMK ARN encrypting the log group. Pass null to use AWS-managed encryption."
  type        = string
  default     = null
}

variable "vpc_id" {
  description = "VPC ID to attach the Lambda to. Leave empty to run outside any VPC."
  type        = string
  default     = ""
}

variable "vpc_cidr" {
  description = "CIDR block of the VPC. Used in the Lambda SG egress rule."
  type        = string
  default     = ""
}

variable "vpc_subnet_ids" {
  description = "Private subnet IDs for the Lambda. Empty list = no VPC attachment."
  type        = list(string)
  default     = []
}

variable "dynamodb_table_arns" {
  description = "ARNs of DynamoDB tables the Lambda can read/write. GSI access is included."
  type        = list(string)
  default     = []
}

variable "kms_key_arns" {
  description = "ARNs of KMS CMKs the Lambda can use."
  type        = list(string)
  default     = []
}

variable "bedrock_model_arns" {
  description = "ARNs (or wildcards) of Bedrock models the Lambda can invoke."
  type        = list(string)
  default     = []
}

variable "s3_bucket_arns" {
  description = "ARNs of S3 buckets the Lambda can read, write, and delete objects in. Object-level actions are scoped to <arn>/*."
  type        = list(string)
  default     = []
}

variable "textract_enabled" {
  description = "Whether to grant Textract document-text detection permissions to the Lambda."
  type        = bool
  default     = false
}

variable "function_url_enabled" {
  description = "Whether to create a Lambda Function URL. Needed for response streaming; otherwise Lambda is invoked via API Gateway."
  type        = bool
  default     = false
}

variable "function_url_invoke_mode" {
  description = "Function URL invoke mode. Use RESPONSE_STREAM for streaming Lambdas, BUFFERED otherwise."
  type        = string
  default     = "BUFFERED"

  validation {
    condition     = contains(["BUFFERED", "RESPONSE_STREAM"], var.function_url_invoke_mode)
    error_message = "Must be BUFFERED or RESPONSE_STREAM."
  }
}

variable "function_url_cors_origins" {
  description = "Origins allowed in the Function URL CORS config. Include both local dev and the deployed SPA origin."
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags applied to resources created by this module."
  type        = map(string)
  default     = {}
}
