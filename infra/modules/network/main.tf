data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  az_count     = 2
  azs          = slice(data.aws_availability_zones.available.names, 0, local.az_count)
  name_prefix  = "anna-chat-${var.env}"
  subnet_cidrs = [for i in range(local.az_count) : cidrsubnet(var.vpc_cidr, 4, i)]
}

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(var.tags, {
    Name = "${local.name_prefix}-vpc"
  })
}

resource "aws_subnet" "private" {
  count                   = local.az_count
  vpc_id                  = aws_vpc.this.id
  cidr_block              = local.subnet_cidrs[count.index]
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = false

  tags = merge(var.tags, {
    Name = "${local.name_prefix}-private-${local.azs[count.index]}"
    Tier = "private"
  })
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.this.id

  tags = merge(var.tags, {
    Name = "${local.name_prefix}-private-rt"
  })
}

resource "aws_route_table_association" "private" {
  count          = local.az_count
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

resource "aws_security_group" "vpc_endpoints" {
  name        = "${local.name_prefix}-vpce-sg"
  description = "Allows VPC endpoint access on 443 from within the VPC."
  vpc_id      = aws_vpc.this.id

  tags = merge(var.tags, {
    Name = "${local.name_prefix}-vpce-sg"
  })
}

resource "aws_vpc_security_group_ingress_rule" "vpce_from_vpc" {
  security_group_id = aws_security_group.vpc_endpoints.id
  description       = "HTTPS from within VPC"
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = aws_vpc.this.cidr_block
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.this.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]

  tags = merge(var.tags, {
    Name = "${local.name_prefix}-s3-vpce"
  })
}

resource "aws_vpc_endpoint" "dynamodb" {
  vpc_id            = aws_vpc.this.id
  service_name      = "com.amazonaws.${var.aws_region}.dynamodb"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]

  tags = merge(var.tags, {
    Name = "${local.name_prefix}-ddb-vpce"
  })
}

locals {
  interface_endpoint_services = [
    "bedrock-runtime",
    "kms",
    "secretsmanager",
    "logs",
    "sts",
    "ssm",
    "monitoring",
    # Textract is the document-text extractor used by BOTH the attachments
    # handler and the KB ingestion Lambda (PDFs → Textract → text). Without
    # this endpoint, any Textract call from a VPC-resident Lambda hangs
    # until the Lambda times out — the VPC has no NAT gateway by design
    # (HIPAA: no egress), so DNS resolves but the socket never connects.
    "textract",
  ]
}

resource "aws_vpc_endpoint" "interface" {
  for_each            = toset(local.interface_endpoint_services)
  vpc_id              = aws_vpc.this.id
  service_name        = "com.amazonaws.${var.aws_region}.${each.value}"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true

  tags = merge(var.tags, {
    Name = "${local.name_prefix}-${each.value}-vpce"
  })
}

resource "aws_flow_log" "vpc" {
  count                    = var.flow_logs_enabled ? 1 : 0
  vpc_id                   = aws_vpc.this.id
  traffic_type             = "ALL"
  log_destination_type     = "cloud-watch-logs"
  log_destination          = aws_cloudwatch_log_group.flow_logs[0].arn
  iam_role_arn             = aws_iam_role.flow_logs[0].arn
  max_aggregation_interval = 60
}

resource "aws_cloudwatch_log_group" "flow_logs" {
  count             = var.flow_logs_enabled ? 1 : 0
  name              = "/aws/vpc/${local.name_prefix}/flowlogs"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.logs_kms_key_arn
}

resource "aws_iam_role" "flow_logs" {
  count = var.flow_logs_enabled ? 1 : 0
  name  = "${local.name_prefix}-vpc-flow-logs"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "vpc-flow-logs.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "flow_logs" {
  count = var.flow_logs_enabled ? 1 : 0
  name  = "flow-logs-write"
  role  = aws_iam_role.flow_logs[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams",
      ]
      Resource = "${aws_cloudwatch_log_group.flow_logs[0].arn}:*"
    }]
  })
}
