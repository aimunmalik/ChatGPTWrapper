locals {
  cloudfront_url = module.edge.cloudfront_url

  csp_connect_extra = join(" ", [
    "https://cognito-idp.${var.aws_region}.amazonaws.com",
    "https://${module.cognito.user_pool_domain}.auth.${var.aws_region}.amazoncognito.com",
    "https://*.execute-api.${var.aws_region}.amazonaws.com",
  ])
}

module "edge" {
  source = "../../modules/edge"

  env                 = var.env
  price_class         = var.cloudfront_price_class
  rate_limit_per_5min = var.waf_rate_limit
  csp_connect_extra   = local.csp_connect_extra

  tags = local.tags
}
