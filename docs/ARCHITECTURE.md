# Architecture

## Overview

anna-chat is a HIPAA-compliant web chat application. A clinician logs in through AWS Cognito, sends a message to a Python Lambda, which invokes Claude on Amazon Bedrock and streams the response back. Conversation history is stored encrypted in DynamoDB. Everything PHI-adjacent stays inside the AWS account covered by ANNA's Business Associate Agreement with AWS — no PHI traverses a third-party API.

## Diagram

```
                            ┌─────────────────────────────────────────┐
                            │          End user (browser)             │
                            └───────────────────┬─────────────────────┘
                                                │ HTTPS (TLS 1.2+)
                                                ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │ CloudFront  ◄── AWS WAF (managed rules + rate limit)    │
                  │    │                                                    │
                  │    ├─► S3 (private, OAC)  ── React SPA (static assets)  │
                  │    │                                                    │
                  │    └─► Origin: Lambda Function URL (chat streaming)     │
                  │        Origin: API Gateway HTTP API (CRUD)              │
                  └─────────────────────────────────────────────────────────┘
                                                │
                    ┌───────────────────────────┼────────────────────────────┐
                    ▼                           ▼                            ▼
          ┌──────────────────┐       ┌──────────────────────┐    ┌──────────────────────┐
          │ Cognito User Pool│       │ API Gateway HTTP API │    │ Lambda Function URL  │
          │ (MFA required,   │──JWT──┤ JWT Authorizer       │    │ (response streaming) │
          │  Hosted UI)      │       │                      │    │ IAM auth + JWT verify│
          └──────────────────┘       └──────────┬───────────┘    └──────────┬───────────┘
                                                │                           │
                                                ▼                           ▼
                                ┌───────────────────────────────────────────────────────┐
                                │                 VPC (private subnets)                 │
                                │                                                       │
                                │   Lambda: conversations-crud    Lambda: chat-stream   │
                                │         │                              │              │
                                │         └──────────┬───────────────────┤              │
                                │                    │                   │              │
                                │        VPC endpoints (no public egress)│              │
                                │   ┌────────────────┼───────────────────┼───────────┐  │
                                │   ▼                ▼                   ▼           ▼  │
                                │ DynamoDB         KMS              Bedrock      CW Logs│
                                │ (CMK-encrypted,  (CMK for all     Runtime      (CMK,  │
                                │  PITR on)         services)       (Claude      no PHI)│
                                │   │                                Sonnet 4.6          │
                                │   ├── Conversations  (PK=userId)   + Opus 4.7)        │
                                │   └── Messages       (PK=convId)   + Guardrails       │
                                └───────────────────────────────────────────────────────┘

     ┌─────────────── Cross-cutting (every account) ───────────────┐
     │ CloudTrail (org trail, KMS, log file validation)            │
     │ AWS Config + HIPAA conformance pack                         │
     │ GuardDuty · Security Hub · Access Analyzer                  │
     │ Secrets Manager (CMK) · SSM Param Store                     │
     │ AWS Organizations: dev / staging / prod (PHI only in prod)  │
     └─────────────────────────────────────────────────────────────┘
```

## Data flow: a single chat turn

1. Browser loads the React SPA from CloudFront.
2. User clicks **Sign in**, gets redirected to the Cognito Hosted UI, authenticates with password + TOTP.
3. Cognito returns an OIDC ID token + access token.
4. User types a message, frontend `POST`s to the **chat-stream Lambda Function URL** with the access token.
5. Lambda verifies the JWT signature against the Cognito JWKS, extracts `sub` (user id).
6. Lambda reads recent turns for this conversation from DynamoDB (KMS-decrypted transparently).
7. Lambda calls `bedrock-runtime.invoke_model_with_response_stream` against Claude Sonnet 4.6 through a VPC endpoint — traffic never leaves the AWS network.
8. Lambda streams the tokens back to the browser via Lambda response streaming.
9. On stream completion, Lambda writes the assistant turn to DynamoDB with a `ttl` 90 days out.
10. CloudWatch captures metadata only (userId, conversationId, token counts, latency) — **never the message content**.

## Why each service

### Compute: AWS Lambda
Serverless fits the traffic profile (bursty, user-count in the dozens). No servers to patch. Each handler has a narrow IAM role. Response streaming via Function URL gives us token-by-token delivery without the operational overhead of WebSockets.

### Model: Amazon Bedrock
Bedrock is a HIPAA-eligible AWS service. Using it keeps PHI inside the AWS BAA boundary — the alternative, calling Anthropic's public API directly, would require a separate BAA with Anthropic and route PHI through the public internet. Bedrock also gives us a VPC endpoint so Lambda-to-model traffic stays private.

Bedrock **invocation logging is disabled** in prod. Those logs would capture raw prompts and completions (i.e. PHI) — keeping them encrypted is possible but the safer posture is to not write them at all.

### Auth: Amazon Cognito User Pool
Managed identity, OIDC-compliant, HIPAA-eligible. We use the Hosted UI (no custom login form to secure), require TOTP MFA, and disable self-signup so an ANNA admin must invite each user.

### Storage: DynamoDB
Two tables:
- **Conversations** — `PK=userId, SK=conversationId`, metadata only (title, createdAt)
- **Messages** — `PK=conversationId, SK=timestamp#messageId`, with `ttl` attribute for the 90-day retention window

Both encrypted with a customer-managed KMS key. Point-in-time recovery enabled. DynamoDB is HIPAA-eligible and gives us predictable per-request performance.

### Network: VPC with endpoints, no NAT
Lambda runs in private subnets. Outbound traffic to AWS services (Bedrock, DynamoDB, KMS, CloudWatch) goes through **VPC endpoints** — no NAT gateway, no internet egress at all. This gives us cost savings and a much cleaner posture: even if a Lambda were compromised, it has no route to the public internet.

### Edge: CloudFront + WAF
CloudFront is the TLS terminator, the static SPA origin (S3 with Origin Access Control), and the front door for the API. WAF gives us AWS Managed Rules (OWASP), rate limiting per IP, and geo-restriction if we ever need it.

### Audit: CloudTrail + Config + GuardDuty
CloudTrail captures every AWS API call. AWS Config with the HIPAA conformance pack alerts on drift from HIPAA-recommended posture. GuardDuty watches for anomalous activity. All three are standard for a HIPAA environment.

## HIPAA-specific design decisions

### PHI may exist in — and only in — these places

1. Messages stored in DynamoDB (encrypted at rest with CMK)
2. In-flight request/response bodies (encrypted in transit with TLS 1.2+)
3. Lambda memory during processing (ephemeral, zeroed on container recycle)
4. CloudTrail data events for S3 (if we enable them for attachments later)

PHI must **NOT** exist in:
- CloudWatch Logs (metadata only — user ids, conversation ids, token counts, latency)
- Bedrock model-invocation logs (disabled)
- X-Ray traces (if enabled, redact all request/response bodies)
- Terraform state (no PHI references)
- Frontend analytics, error tracking, or any third-party JS

### Keys
One customer-managed KMS key per data domain (DDB, Logs, Secrets, S3). Automatic annual rotation. Key policies restrict decrypt to the specific Lambda roles.

### Retention
- Messages: 90-day TTL, user-configurable
- CloudTrail: 7 years (HIPAA-aligned)
- CloudWatch Logs: 90 days
- DynamoDB backups (PITR): 35 days
- Lambda environment: no PHI in environment variables

### Access
- No long-term IAM user access keys for humans. IAM Identity Center only.
- MFA required for all human access (Cognito for app users, Identity Center for AWS console).
- Least-privilege IAM roles per Lambda, per CI job.
- Quarterly access review documented in `docs/OPS.md`.

## Environments

| Env | Purpose | Data |
|---|---|---|
| `dev` | Development, smoke tests | Synthetic only. **No PHI.** |
| `prod` | Production | Real PHI. Strictest posture. |

Both live in the same AWS account for now (separated by Terraform workspaces + resource naming prefix). Path to split: AWS Organizations with a dedicated prod account, SCPs preventing dev-account access to prod resources. Tracked as a future improvement in `docs/OPS.md`.

## What's explicitly out of scope

- Self-service signup (admin-invite only)
- Social login (Google, Microsoft)
- File uploads / attachments (can be added with a separate S3 bucket + presigned URLs later)
- Multi-region DR (can be added with DynamoDB global tables and cross-region Lambda replicas)
- Custom domain name on day one (CloudFront default domain is fine until we're ready)
- Email/SMS notifications to users (not part of the chat flow)
