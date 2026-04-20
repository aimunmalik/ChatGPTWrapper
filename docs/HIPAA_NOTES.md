# HIPAA posture

This document is the operational reference for how anna-chat handles Protected Health Information (PHI). It exists to make HIPAA decisions explicit and auditable.

## Scope

anna-chat is a workforce-facing clinical tool. Users are ANNA Health staff. The application processes PHI in the form of free-text conversations with the Claude model. Patient identifiers may appear in messages.

**Data classification for this system:** all user messages and assistant responses are assumed to contain PHI. All operational controls below are written with that assumption.

## Business Associate Agreements (BAAs)

| Party | Status | Covers | Where filed |
|---|---|---|---|
| AWS | ⏳ to sign | All HIPAA-eligible AWS services (incl. Bedrock, Cognito, DynamoDB, Lambda, etc.) | AWS Artifact → Agreements → AWS BAA |
| Anthropic (direct) | N/A | *Not required*. Claude is accessed through Bedrock, covered by the AWS BAA. | — |

**Action:** the AWS BAA must be signed in AWS Artifact before any production deploy. Keep a copy in ANNA's compliance record system.

## Services in use and HIPAA eligibility

All services listed are on the [AWS HIPAA-eligible services list](https://aws.amazon.com/compliance/hipaa-eligible-services-reference/) as of 2026.

| Service | Role | HIPAA-eligible |
|---|---|---|
| CloudFront | Edge / CDN | Yes |
| AWS WAF | Web app firewall | Yes |
| S3 | SPA static hosting, future attachments | Yes |
| Route 53 | DNS | Yes |
| ACM | TLS certificates | Yes |
| Cognito User Pools | Identity | Yes |
| API Gateway (HTTP APIs) | CRUD API | Yes |
| Lambda | Compute | Yes |
| Bedrock | LLM (Claude) | Yes |
| DynamoDB | Storage | Yes |
| KMS | Encryption keys | Yes |
| VPC, VPC endpoints | Network isolation | Yes |
| CloudTrail | Audit logs | Yes |
| CloudWatch Logs + Metrics | Operational logs | Yes |
| AWS Config | Posture drift | Yes |
| GuardDuty, Security Hub | Threat detection | Yes |
| Secrets Manager | Secrets | Yes |
| IAM Identity Center | Human access | Yes |

Any service not on this list **must not** be added to the stack without a compliance review.

## Encryption

**In transit.** TLS 1.2+ on every hop exposed outside the VPC (CloudFront → browser, CloudFront → origin). Internal AWS-to-AWS traffic uses AWS's network-layer encryption. No plaintext protocols anywhere.

**At rest.** Customer-managed KMS keys (CMKs) for every data store. One key per domain:

| CMK alias | Protects |
|---|---|
| `alias/anna-chat-dynamodb` | DynamoDB tables |
| `alias/anna-chat-logs` | CloudWatch log groups, CloudTrail |
| `alias/anna-chat-secrets` | Secrets Manager entries |
| `alias/anna-chat-s3` | S3 buckets (SPA + future attachments) |

All CMKs: annual rotation enabled, deletion window 30 days, key policies that restrict `Decrypt` to named IAM roles.

## Logging rules (critical)

**PHI must never be written to any log.** Logs are searchable, indexable, retained, and often read by humans who don't have a clinical need.

Logging rules for the application:

1. **Bedrock model invocation logging: OFF.** Those logs would contain full prompts and completions.
2. **CloudWatch application logs:** metadata only. Allowed fields:
   - `userId` (Cognito sub)
   - `conversationId`
   - `messageId`
   - `inputTokens`, `outputTokens`
   - `latencyMs`
   - `model`
   - `errorCode` (but not error messages if they echo user input)
3. **X-Ray:** if enabled, all request and response bodies must be redacted.
4. **CloudTrail:** management events always on. Data events for S3 only on buckets holding PHI, with CMK encryption.
5. **Structured logging enforced** — developers write `logger.info(..., extra={"userId": ...})`, never `logger.info(f"user said: {message}")`.

A unit test (`backend/tests/test_no_phi_in_logs.py`, to be written in Phase 2) asserts the logger redacts message content.

## Access controls

- **No long-lived IAM user access keys** for humans. Ever.
- **IAM Identity Center** is the single sign-on for AWS console/CLI access.
- **MFA required** for every human principal: TOTP on Identity Center, TOTP on Cognito.
- **Least-privilege IAM roles** per Lambda, per CI job. No wildcard `Resource: "*"` on `kms:Decrypt` or `dynamodb:*`.
- **Quarterly access review** — documented in `docs/OPS.md`, performed by the ANNA HIPAA Security Officer.

## Retention

| What | Retention | Why |
|---|---|---|
| Messages in DynamoDB | 90 days (user-configurable) via TTL | Minimizes PHI exposure; clinicians can extend per conversation |
| CloudTrail events | 7 years | HIPAA audit trail aligned with industry norms |
| CloudWatch Logs | 90 days | Long enough for incident investigation, short enough to limit exposure |
| DynamoDB PITR snapshots | 35 days | AWS max, supports rollback |
| Bedrock invocation logs | N/A (disabled) | Would contain PHI |

Deletion: a user deletion request (HIPAA right-to-amend is narrow, but ANNA policy may grant deletion) is handled by deleting all rows for `userId` from both tables. PITR copies age out within 35 days. CloudTrail retains the deletion action itself, not the deleted content.

## Backup and recovery

- **DynamoDB PITR:** enabled on both tables, 35-day window.
- **CloudTrail:** logs shipped to a dedicated S3 bucket with Object Lock.
- **Lambda code / IaC:** stored in Git, mirrored to GitHub and an AWS CodeCommit replica (optional).

## Incident response

See [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) for the breach checklist. Key numbers:
- Breach notification deadline: 60 days from discovery (HIPAA §164.404).
- Internal discovery-to-triage target: 24 hours (measured from first GuardDuty or CloudWatch alarm).

## Required ANNA-side work (not code)

These are organizational, not technical, but the system is non-compliant without them:

- [ ] Written **HIPAA Security Risk Assessment** for this application (§164.308(a)(1)(ii)(A))
- [ ] **Policies & procedures** covering workforce access, incident response, device/media controls
- [ ] **Workforce training** for every user who will access anna-chat (§164.308(a)(5))
- [ ] Designated **HIPAA Security Officer** and **Privacy Officer** on record
- [ ] **Sanctions policy** for workforce members who violate procedures
- [ ] **Business continuity / contingency plan** (§164.308(a)(7))

## Compliance posture statement

anna-chat implements the technical and physical safeguards required under the HIPAA Security Rule as interpreted by AWS's HIPAA Reference Architecture. Administrative safeguards (policies, training, risk assessment, workforce sanctions, incident response execution) are the responsibility of ANNA Health and are tracked in ANNA's compliance record system, not in this repository.
