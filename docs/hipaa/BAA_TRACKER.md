# Business Associate Agreement (BAA) Register — ANNA Health

**Regulation:** 45 CFR §164.308(b)(1), §164.502(e), §164.504(e) (Business Associate Contracts)
**Document owner:** HIPAA Privacy Officer
**Effective date:** `[DATE]`
**Review cadence:** Quarterly

---

## 1. Purpose

HIPAA requires a signed Business Associate Agreement with any vendor who creates, receives, maintains, or transmits PHI on ANNA's behalf. This register lists every such vendor, the BAA status, and where the signed copy is filed.

Vendors who **only** touch ANNA's non-PHI systems do not need a BAA, but they are still listed here for completeness if they sit close to the PHI boundary.

## 2. Register

| # | Vendor | Service / Role | PHI access? | BAA required? | BAA signed? | Date signed | Document location |
|---|---|---|---|---|---|---|---|
| 1 | **AWS (Amazon Web Services)** | All HIPAA-eligible services backing Praxis: Bedrock, Lambda, DynamoDB, Cognito, CloudFront, WAF, S3, KMS, CloudTrail, CloudWatch, Config, GuardDuty, Security Hub, VPC, Secrets Manager, IAM Identity Center, API Gateway, Route 53, ACM | **Yes** (at rest and in transit) | **Yes** | `[Y/N]` | `[DATE]` | AWS Artifact → Agreements → AWS BAA; copy filed in `[ANNA COMPLIANCE RECORD SYSTEM]` |
| 2 | Anthropic (direct API) | — | No | **No** | n/a | n/a | **Not required.** Claude is accessed via Amazon Bedrock, which is covered by the AWS BAA. ANNA does not use the Anthropic direct API with PHI. |
| 3 | `[EMAIL PROVIDER]` (e.g. Google Workspace, Microsoft 365) | ANNA workforce email | Possibly (if clinical email is sent) | Yes | `[Y/N]` | `[DATE]` | `[LOCATION]` |
| 4 | `[EMR VENDOR]` (if and when Praxis integrates with it) | Electronic medical record | **Yes** | Yes | `[Y/N]` | `[DATE]` | `[LOCATION]` |
| 5 | `[MDM VENDOR]` (mobile device management) | Workforce laptops / mobile endpoints that may cache Praxis tokens or display PHI | Indirect (device-level) | Yes (for MDM that can read device content) | `[Y/N]` | `[DATE]` | `[LOCATION]` |
| 6 | `[BACKUP / ARCHIVE VENDOR]` (if any beyond AWS-native) | Off-AWS backup of audit trail or PHI | **Yes** | Yes | `[Y/N]` | `[DATE]` | `[LOCATION]` |
| 7 | `[SIEM / LOG AGGREGATOR]` (if any beyond CloudWatch) | Security log aggregation | Metadata-only (no PHI should leave logs per [HIPAA_NOTES.md](../HIPAA_NOTES.md)) | Yes, if vendor cannot guarantee no-PHI scope | `[Y/N]` | `[DATE]` | `[LOCATION]` |
| 8 | `[HELP DESK / SUPPORT VENDOR]` (if outsourced) | Tier-1 user support, possibly observing screens with PHI | **Yes** | Yes | `[Y/N]` | `[DATE]` | `[LOCATION]` |
| 9 | `[CYBER INSURANCE CARRIER]` | Insurance claims processing post-incident | Possibly (claim investigation) | Yes, if they receive PHI in claims | `[Y/N]` | `[DATE]` | `[LOCATION]` |
| 10 | `[EXTERNAL COUNSEL — BREACH NOTIFICATION]` | Legal representation on HIPAA matters | Possibly (case review) | Attorney-client privilege typically; BAA still advised | `[Y/N]` | `[DATE]` | `[LOCATION]` |

> Add rows for every additional vendor that handles PHI or sits in the PHI blast radius. Remove rows that do not apply to ANNA's actual stack.

## 3. Process for adding a new vendor

Before a new vendor is engaged to handle PHI on ANNA's behalf:

1. Business owner submits vendor details to the Privacy Officer.
2. Privacy Officer confirms whether the vendor is a Business Associate under §160.103.
3. If yes, the Privacy Officer requests the vendor's standard BAA (or issues ANNA's template).
4. Legal counsel reviews.
5. Authorized ANNA signatory executes the BAA.
6. Signed copy filed in the compliance record system. Row added to this register.
7. No data flow begins until Step 6 is complete.

## 4. Quarterly review

The Privacy Officer, every quarter:

- Confirms each "BAA signed? = Y" row has a signed copy on file that is still in force.
- Adds any new vendor onboarded during the quarter.
- Removes any vendor that has been offboarded (retains historical row; marks as inactive).
- Re-verifies that vendors marked "No BAA required" still do not touch PHI.

Review log entries are filed in the compliance record system with date and reviewer.

## 5. Termination

When a BA relationship ends, the Privacy Officer confirms in writing that the vendor has:

- Returned or destroyed all ANNA PHI in their possession, and
- Confirmed destruction in writing, and
- Destroyed any backup copies within a reasonable timeline.

If any PHI cannot be returned or destroyed, the BA must extend the BAA's protections to that retained PHI indefinitely.

## 6. Signatures

| Role | Name | Signature | Date |
|---|---|---|---|
| HIPAA Privacy Officer | `[NAME]`, `[TITLE]` | ____________________ | `[DATE]` |
| HIPAA Security Officer | `[NAME]`, `[TITLE]` | ____________________ | `[DATE]` |

Effective date: `[DATE]`. Next review: `[DATE + 3 months]` (quarterly).
