locals {
  cloudfront_url = module.edge.cloudfront_url

  # Pin the CSP connect-src for S3 to exactly the attachments bucket rather
  # than `*.s3.amazonaws.com`, which would let an XSS payload reach any
  # S3 bucket the browser can reach. Both hostname forms are valid for the
  # same bucket (virtual-hosted–style, global and regional), so we list both.
  _attachments_bucket = module.attachments.bucket_name
  _attachments_host_1 = "https://${local._attachments_bucket}.s3.amazonaws.com"
  _attachments_host_2 = "https://${local._attachments_bucket}.s3.${var.aws_region}.amazonaws.com"

  csp_connect_extra = join(" ", [
    "https://cognito-idp.${var.aws_region}.amazonaws.com",
    "https://${module.cognito.user_pool_domain}.auth.${var.aws_region}.amazoncognito.com",
    "https://*.execute-api.${var.aws_region}.amazonaws.com",
    # Attachments bucket uploads (presigned POST from browser → S3).
    local._attachments_host_1,
    local._attachments_host_2,
  ])
}

module "edge" {
  source = "../../modules/edge"

  env                 = var.env
  price_class         = var.cloudfront_price_class
  rate_limit_per_5min = var.waf_rate_limit
  csp_connect_extra   = local.csp_connect_extra

  # Streaming path (/api/chat-stream) is disabled. Python Lambda doesn't
  # support native response streaming; to re-enable, pass the function URL
  # domain here and add a lambda_chat_stream module call in backend_compute.tf.
  # chat_stream_origin_domain = module.lambda_chat_stream.function_url_domain

  tags = local.tags
}
