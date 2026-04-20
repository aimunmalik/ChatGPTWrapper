# HIPAA Security Risk Assessment — Praxis

**Regulation:** 45 CFR §164.308(a)(1)(ii)(A) — Security Management Process: Risk Analysis
**Covered entity:** ANNA Health (Allied Network for Neurodevelopmental Advancement)
**System under assessment:** Praxis — internal clinical AI assistant (codebase: `anna-chat`)
**Document owner:** HIPAA Security Officer
**Effective date:** `[DATE]`
**Review cadence:** Annual + on material change

---

## 1. Scope

This assessment covers the Praxis application end-to-end: the React SPA served from CloudFront, authentication via Cognito, the Python Lambda backend, the Amazon Bedrock model invocation path, the DynamoDB persistence layer, and all cross-cutting AWS services (CloudTrail, GuardDuty, Config, KMS, CloudWatch) that support it.

Out of scope: ANNA's corporate network, EMR, billing systems, and workforce endpoints (laptops, mobile devices) except where they directly interact with Praxis.

## 2. Methodology

Assessment follows the structure recommended by NIST SP 800-30r1 and the HHS OCR *Guidance on Risk Analysis*:

1. Characterize the system and data flows.
2. Inventory PHI.
3. Identify threats and vulnerabilities.
4. Map existing controls.
5. Determine likelihood and impact.
6. Rate residual risk (Low / Medium / High).
7. Define action items, owners, target dates.

Ratings use a 3x3 impact × likelihood matrix; residual risk is the rating *after* existing controls are applied.

## 3. System description

Praxis is a Claude-based clinical assistant for ANNA workforce members. Architecture summary (full detail in [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)):

- **Edge:** CloudFront + AWS WAF (managed rules, rate limiting).
- **Auth:** Cognito User Pool, Hosted UI, TOTP MFA required, no self-signup.
- **API:** API Gateway (HTTP APIs) with JWT authorizer; Lambda Function URL for streaming.
- **Compute:** Python Lambda in private VPC subnets, no internet egress, per-handler IAM roles.
- **Model:** Amazon Bedrock (Claude Sonnet 4.6 / Opus 4.7) accessed via VPC endpoint. **Bedrock invocation logging is disabled** to prevent PHI in logs.
- **Storage:** DynamoDB tables (Conversations, Messages), CMK-encrypted, PITR enabled, 90-day TTL on messages.
- **Crypto:** Customer-managed KMS keys per data domain (DDB, Logs, Secrets, S3). Annual rotation.
- **Audit:** CloudTrail (7-year retention, Object-Lock S3), AWS Config with HIPAA conformance pack, GuardDuty, Security Hub, Access Analyzer.

## 4. Data flow and PHI inventory

**Assumption:** Every message a user types and every assistant response is treated as PHI.

| # | PHI location | State | Protection |
|---|---|---|---|
| 1 | Browser memory (in-session) | In use | TLS 1.2+, session tokens 1h |
| 2 | TLS tunnel CloudFront ↔ Lambda | In transit | TLS 1.2+ |
| 3 | Lambda memory | In use | Ephemeral, zeroed on container recycle |
| 4 | Bedrock VPC endpoint traffic | In transit | AWS internal network encryption |
| 5 | DynamoDB (Messages, Conversations) | At rest | CMK (`alias/anna-chat-dynamodb`), PITR |
| 6 | DynamoDB PITR snapshots | At rest | Same CMK, 35-day window |
| 7 | Future S3 attachments (Phase 6+) | At rest | CMK (`alias/anna-chat-s3`), Object Lock, presigned URLs |

**PHI is explicitly NOT written to:** CloudWatch Logs, Bedrock invocation logs (disabled), X-Ray traces, Terraform state, Lambda env vars, frontend analytics.

## 5. Threats, vulnerabilities, controls, residual risk

| # | Threat | Vulnerability | Existing controls | Likelihood | Impact | Residual |
|---|---|---|---|---|---|---|
| T1 | Insider misuse — workforce member accesses or exports PHI without clinical need | Over-broad role, curiosity, malicious intent | Cognito unique user IDs; role groups (`admins`, `users`); CloudTrail records every API call; DynamoDB read metrics; quarterly access review; sanctions policy; workforce training | Medium | High | **Medium** |
| T2 | Account compromise — stolen credentials, phishing | User reuses password, token theft via malware, lost laptop with cached tokens | TOTP MFA required; 1-hour access tokens; Cognito risk-based auth; GuardDuty anomalous-login detection; MDM on endpoints (`[ANNA CONTACT]`); incident response runbook | Medium | High | **Medium** |
| T3 | Attachment malware (Phase 6+) — malicious file uploaded as clinical attachment | Uploaded file processed or forwarded to another user | Attachments scoped to issuer (presigned GET/PUT with user-bound keys); no in-browser execution of uploaded content; S3 Object Lock; future malware scan step (planned; see action items) | Low | Medium | **Low** |
| T4 | Data exfiltration via logs — a developer accidentally writes PHI into CloudWatch | `logger.info(f"user said: {msg}")` instead of structured metadata | Structured-logging policy; unit test `test_no_phi_in_logs.py` asserts redaction; code review; Bedrock invocation logging OFF; X-Ray body redaction required | Medium | High | **Medium** |
| T5 | Misconfigured S3 — public bucket, overly broad ACL | Drift from IaC, console changes, missing Block Public Access | AWS Config HIPAA conformance pack flags drift; Access Analyzer; S3 Block Public Access at account level; CMK-required bucket policies; CloudTrail S3 data events on PHI buckets | Low | High | **Low** |
| T6 | Denial of service — volumetric or application-layer attack against Praxis | Exhausted Lambda concurrency, overloaded API Gateway, cost blowout | AWS WAF rate limits per IP; CloudFront absorbs L3/L4 volumetrics; Lambda reserved concurrency ceiling; CloudWatch alarms on 5xx spike; budget alarms on unexpected spend | Medium | Medium | **Low** |
| T7 | Supplier / vendor compromise — AWS, Anthropic (via Bedrock), or another BA has a breach | Third-party security incident cascades to ANNA PHI | AWS BAA in force (see [BAA_TRACKER.md](BAA_TRACKER.md)); no direct Anthropic API use (all via Bedrock under AWS BAA); CMKs held in ANNA account — AWS cannot decrypt without our key policy; vendor-list quarterly review | Low | High | **Medium** |
| T8 | KMS key misuse — `Decrypt` called outside intended scope | Over-broad IAM policy, role assumption chain | Per-domain CMKs; key policies restrict `Decrypt` to named Lambda roles; no wildcard `kms:*`; CloudTrail data events on KMS; automatic annual rotation | Low | High | **Medium** |
| T9 | Ransomware on PHI data store | Credential compromise leading to destructive write | DynamoDB PITR (35 days); CloudTrail Object Lock S3 bucket (immutable); IaC in Git; deny-by-default IAM | Low | High | **Medium** |

### Matrix legend

| | Low impact | Medium impact | High impact |
|---|---|---|---|
| **High likelihood** | Medium | High | High |
| **Medium likelihood** | Low | Medium | High |
| **Low likelihood** | Low | Low | Medium |

## 6. Risk determination summary

| Residual | Count | IDs |
|---|---|---|
| High | 0 | — |
| Medium | 5 | T1, T2, T4, T7, T8, T9 |
| Low | 3 | T3, T5, T6 |

No residual High risks remain, assuming every listed control is actually implemented and operating. Any gap surfaced during audit moves the corresponding item back up.

## 7. Action items

| # | Action | Owner | Target date |
|---|---|---|---|
| A1 | Name HIPAA Security Officer and Privacy Officer on record with ANNA HR | `[ANNA CONTACT]` | `[DATE]` |
| A2 | Sign AWS BAA in AWS Artifact and file copy in compliance record system | `[ANNA CONTACT]` | Before prod deploy |
| A3 | Deliver workforce training per [WORKFORCE_TRAINING.md](POLICIES/WORKFORCE_TRAINING.md) to every Praxis user | `[ANNA CONTACT]` | Before access granted |
| A4 | Implement `test_no_phi_in_logs.py` unit test in Phase 2 backend | Engineering | Phase 2 close |
| A5 | Define attachment malware-scan step for Phase 6+ (Lambda + ClamAV or third-party) | Engineering | Phase 6 design |
| A6 | Establish quarterly access review process and calendar invite | HIPAA Security Officer | `[DATE]` |
| A7 | Tabletop an incident response drill | HIPAA Security Officer | Annually from `[DATE]` |
| A8 | Multi-region DR design (currently single-region, Tier C) | Engineering | When business requires |

## 8. Review cadence

- **Annual:** full re-assessment on the anniversary of the effective date.
- **On material change:** new AWS service added, vendor added/removed, new user role, after any reportable incident, or HHS guidance update.
- Review evidence (minutes, updated matrix, sign-off) filed in the compliance record system.

## 9. Signatures

| Role | Name | Signature | Date |
|---|---|---|---|
| HIPAA Security Officer | `[NAME]`, `[TITLE]` | ____________________ | `[DATE]` |
| HIPAA Privacy Officer | `[NAME]`, `[TITLE]` | ____________________ | `[DATE]` |

Next scheduled review: `[DATE + 1 year]`.
