# `docs/templates/` — Starter docs for the next HIPAA-covered product

This directory contains the five documents every new ANNA Health internal
product should have on day one. They are generalized from the Praxis
(anna-chat) build and assume the **ANNA Standard Stack**.

## How to use

When starting a new product:

```
# From the new product's repo root
mkdir -p docs
cp /path/to/anna-chat/docs/templates/*.md docs/
rm docs/README.md   # the one you're reading now is meta
```

Then search-and-replace in the copied files:

| Placeholder | Replace with |
|---|---|
| `<PRODUCT-NAME>` | The new product's name |
| `<product>` | lowercase slug (e.g. `praxis`, `intake-triage`) |
| `<env>` | environment name (usually `dev`) |
| `<account-id>` | AWS account ID |
| `<region>` | AWS region (usually `us-east-1`) |
| `<pool-id>` | Cognito user pool ID (fill in after first deploy) |
| `<app-domain>` | the product's public domain |

Also fill in all `___` and `<angle-bracketed>` spots that are
product-specific.

## The five documents

| File | Audience | Purpose |
|---|---|---|
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Engineers, security review, onboarding new devs | Layer-by-layer what's deployed, why, and how the request flows through it |
| [`HIPAA_CHECKLIST.md`](./HIPAA_CHECKLIST.md) | HIPAA Security Officer, auditors, compliance review | The 45 CFR § 164 safeguards crosswalk. Reviewed annually |
| [`DISASTER_RECOVERY.md`](./DISASTER_RECOVERY.md) | On-call, primary admin | What to do when data/service is lost. Run a drill quarterly |
| [`SECURITY_RUNBOOK.md`](./SECURITY_RUNBOOK.md) | On-call, HIPAA officer | What to do when *something/someone is attacking you*. Includes breach decision tree |
| [`ONBOARDING.md`](./ONBOARDING.md) | Whoever administers the product's user base | Add / promote / demote / offboard users. The 4pm Friday playbook |

## Give this to Claude Code

When starting a new product with an AI assistant, a useful prompt:

> Build a new AWS-hosted internal HIPAA-compliant product called
> `<name>`. Use the ANNA Standard Stack documented in the
> `anna-chat` repo. Copy the terraform modules, backend
> scaffolding (`logging_config.py`, `http.py`, `settings.py`,
> `ddb.py`, `build.py`), and frontend scaffolding (auth wiring,
> `api/client.ts`, CommandPalette, Layout, theme tokens) verbatim.
> Preserve the security posture: CMK split by domain, VPC-only
> Lambda, CloudWatch PHI-redaction formatter, WAF + CSP + Cognito
> MFA, GitHub Actions via OIDC with resource-scoped inline policy,
> budget kill-switch. Also copy and fill in
> `docs/templates/{ARCHITECTURE,HIPAA_CHECKLIST,DISASTER_RECOVERY,SECURITY_RUNBOOK,ONBOARDING}.md`
> for this product.
>
> Product-specific scope:
> - Domain model: `<entities + relationships>`
> - Routes: `<verbs + paths>`
> - Retention policy: `<per-entity TTL>`
> - Non-standard integrations: `<none / list>`
> - Primary users: `<who>`
> - Admin users: `<who>`

## Updating these templates

When a real incident or product teaches you something that belongs in
the template (not just this product), update the template here. The
next product inherits the lesson.
