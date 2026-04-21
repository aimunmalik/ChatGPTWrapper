# `<PRODUCT-NAME>` — Architecture

> **How to use this doc:** this is a template based on the ANNA Standard Stack
> (see `anna-chat` for the canonical reference implementation). Replace
> `<angle-bracketed>` placeholders with product-specific values. Delete
> sections that genuinely don't apply. If a section needs meaningful change,
> stop and ask whether you're deviating from the standard stack for a good
> reason.

---

## 1. What this is

`<PRODUCT-NAME>` is an internal, HIPAA-compliant `<one-sentence description>`.
Deployed to AWS account `<account-id>` in region `<region>`, under an active
AWS BAA. Primary users: `<who uses this>`. Admin users: `<who administers>`.

---

## 2. Layer-by-layer

### Frontend — CloudFront + S3
- React 18 + TypeScript + Vite SPA, statically built, uploaded to a private S3
  bucket on every deploy.
- CloudFront with OAC fronts the bucket. Bucket itself is BPA-locked, no
  direct public read.
- WAF attached: AWS managed rule groups (Common, KnownBad, IpReputation) +
  per-IP rate limit.
- CSP injected at CloudFront with enumerated `connect-src` origins — no
  wildcards. Hosts listed: API Gateway, Cognito, explicitly named S3 buckets.
- HSTS preload, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`.
- Custom domain `<subdomain>.<root-domain>` via ACM (DNS-validated) attached
  to the distribution.
- Auth state: `react-oidc-context` talking to Cognito Hosted UI, PKCE flow.

### Edge
- Single CloudFront distribution.
- Assets under `/assets/*`: `Cache-Control: public, max-age=31536000, immutable`.
- Root HTML: `no-cache, must-revalidate`. CI invalidates on every deploy.

### API — API Gateway HTTP API + JWT authorizer
- HTTP API (not REST) — cheaper, lower latency, native CORS, native JWT auth.
- Cognito User Pool JWT authorizer on every route; signature verified inside
  API Gateway, not in Lambda code.
- CORS `allowed_origins` is an enumerated list of real origins — never `*`.
- One Lambda integration per route; no monolithic handler.

### Compute — Lambda (Python 3.12, x86_64)
Functions (copy this table, edit to reflect this product):

| Function | Route(s) | Purpose |
|---|---|---|
| `<product>-<env>-<name>` | `<METHOD> /path` | `<what it does>` |
| … | … | … |

All Lambdas are VPC-resident in private subnets. No NAT, no IGW. Egress only
through VPC endpoints.

### VPC — air-gapped by design
- 2 private subnets across 2 AZs.
- Gateway endpoints: S3, DynamoDB.
- Interface endpoints (by service actually used): bedrock-runtime, kms,
  secretsmanager, logs, sts, ssm, monitoring, textract, `<others>`.
- Single SG for all endpoints accepts 443 from the Lambda SG.
- Lambda SG egress: 443 to 0.0.0.0/0 (DDB gateway endpoint needs public IP
  ranges internally — route table redirects).

### Storage

**DynamoDB** (all SSE-KMS, PITR enabled, PAY_PER_REQUEST, deletion
protection on in prod):

| Table | PK | SK | TTL | Notes |
|---|---|---|---|---|
| `<product>-<env>-<entity>` | `<pk>` | `<sk>` | `<days or none>` | `<notes>` |
| … | … | … | … | … |

**S3** (SSE-KMS, versioning on, BPA locked, TLS-only bucket policy):

| Bucket | Purpose | Lifecycle |
|---|---|---|
| `<product>-<env>-spa-<account-id>` | SPA assets | n/a |
| `<product>-<env>-<domain>-<account-id>` | `<what>` | `<retention>` |

**KMS** — four customer-managed keys (not AWS-managed), each with rotation
enabled and a tight policy:

- `kms_dynamodb` — all DDB tables
- `kms_s3` — all S3 buckets
- `kms_logs` — CloudWatch log groups
- `kms_secrets` — Secrets Manager

Domain-split so a compromise of one key can't cross-decrypt.

### Auth — Cognito
- User Pool with MFA (TOTP) **ENFORCED**.
- Advanced security features: **ENFORCED**.
- Hosted UI + PKCE flow.
- Admin group: `admins`. Checked server-side via `http.require_admin()` for
  all privileged routes.
- **No self-signup.** Admins create users via the Cognito console.

### AI — Bedrock (delete this section if no LLM)
- `<model id(s)>` selectable per turn, whitelisted set enforced server-side.
- Embeddings: `<model id>`, `<dimension>`-dim, normalized.
- Client config: `read_timeout=25s`, `max_attempts=2` — sized to stay under
  API Gateway's 30s hard integration cap.
- AWS Budget with a Budget Action that disables Bedrock model access if
  spend crosses `<threshold>` — hard kill switch.

---

## 3. Security posture

- **Structured JSON logging with PHI redaction at the formatter.** Fields
  stripped globally: `content`, `message`, `body`, `prompt`, `text`,
  `completion`, `messages`. CloudWatch never sees chat content / uploaded
  document content / user-supplied prompts.
- **Tracebacks are never logged.** Library exceptions (openpyxl, python-docx,
  pypdf) embed user content in their messages; formatter records only the
  exception class name + AWS error code for ClientError.
- **defusedxml**: `defuse_stdlib()` called at process startup to harden
  stdlib XML parsers against XXE/billion-laughs.
- **Zip-bomb defense**: OOXML extractors reject files where uncompressed
  total > 250MB or per-entry compression ratio > 100:1.
- **Presigned-upload constraints** enforced server-side: MIME whitelist,
  max size, S3 key prefix (`Conditions` on the presigned POST).
- **WAF** (managed rules + rate limit) + **Cognito advanced security** +
  **Bedrock budget kill-switch** = layered defense.

---

## 4. CI/CD — GitHub Actions via OIDC

- `ci.yml` runs on PR: backend `pytest` + `ruff`, frontend `tsc -b` +
  `npm run build`.
- `deploy-<env>.yml` runs on push to main:
  1. Assumes a GitHub-Actions IAM role via OIDC (no long-lived keys).
  2. Builds Lambda zip with `--platform manylinux2014_x86_64`.
  3. `terraform apply`.
  4. Syncs SPA to S3 with correct cache-control headers per path.
  5. Invalidates CloudFront `/*`.
- GitHub Actions role is `PowerUserAccess` + inline policy scoped to
  resources prefixed `<product>-*`. **No `iam:Attach*Policy`** — preserves
  the "compromised PR can't elevate" invariant.
- PR-time `terraform plan` is **disabled** — requires trusting pull_request
  tokens with the apply role, which is a privilege-escalation path.
- Deploy role trust policy pins `sub` to `refs/heads/main` only.

---

## 5. Infrastructure-as-Code — Terraform

Structure:
```
infra/
  bootstrap/           # one-time state backend creation
  envs/
    <env>/
      *.tf             # composes modules, per-env tfvars
      backend.hcl      # state bucket/table refs
  modules/
    network/           # VPC + endpoints + SGs
    kms/               # single domain-keyed CMK
    cognito/           # user pool + hosted UI + groups
    lambda/            # python lambda wrapper (VPC + IAM + log group)
    api/               # HTTP API + JWT authorizer + routes var
    edge/              # CloudFront + OAC + WAF + CSP + custom domain
    dynamodb/          # standard-hardened table
    <product-modules>/ # anything product-specific
```

- State in an S3 bucket (versioned, SSE-KMS) + DDB lock table, bootstrapped
  once in `infra/bootstrap/`.
- **No secrets in `.tf`**. `terraform.tfvars` is `.gitignore`d. Sensitive
  inputs come from GitHub Actions variables at apply time.
- Terraform version pinned in `.github/workflows/deploy-<env>.yml`.

---

## 6. Observability

- CloudWatch log groups per Lambda, retention 30 days, encrypted with
  `kms_logs`.
- Structured JSON logs with standard fields: `userId`, `<product-specific
  correlation ids>`, `latencyMs`, `<domain metrics>` — all scalar, never
  content.
- Metric filters + CloudWatch alarms on: 5xx rate, Bedrock spend (if
  applicable), WAF-blocked request rate, Lambda throttles, Lambda error
  rate.
- SNS topic `<product>-<env>-alarms` with email subscription for paging.

---

## 7. Data flow (primary use case)

Replace this section with a diagram or numbered walkthrough of the product's
main request lifecycle. For a chat/RAG product, see Praxis `ARCHITECTURE.md`
Phase 7 section for a working template.

1. `<entry point>` → `<what happens>` → `<where data lands>`.
2. …

---

## 8. Data retention

| Data class | Storage | TTL | Reason |
|---|---|---|---|
| `<e.g. messages>` | DDB | 90 days | `<reason>` |
| `<e.g. uploads>` | S3 | 90 days via lifecycle | `<reason>` |
| CloudWatch logs | CW | 30 days | cost + minimum-necessary |
| CloudTrail | dedicated S3 bucket (Object Lock) | 7 years | compliance |

---

## 9. The HIPAA-valid checklist (also see `HIPAA_CHECKLIST.md`)

1. AWS BAA active, covering every service used.
2. CMK (not AWS-managed) encryption, split by domain.
3. TLS 1.2+ enforced at every edge + internal hop.
4. Cognito MFA ENFORCED + advanced security ENFORCED.
5. CloudTrail to dedicated log bucket, 7-year retention.
6. App logs strip PHI at the formatter layer.
7. Per-user JWT auth; admin group for privileged routes; IAM least-privilege
   scoped per Lambda.
8. Compute VPC-isolated with no public egress; AWS API via VPC endpoints.
9. Data retention via TTL + lifecycle rules; Object Lock where applicable;
   PITR on DDB.
10. Bedrock budget kill-switch (if LLM-backed).

---

## 10. Known deviations from the standard stack

> List anything this product does that departs from the ANNA Standard Stack,
> and why. If this section is empty, you're on the golden path. If it's long,
> treat each entry as a risk worth revisiting.

- `<none yet>`
