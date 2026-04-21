# `<PRODUCT-NAME>` — HIPAA Compliance Checklist

> **How to use this doc:** this is a living checklist. Review it at launch,
> after any architecture change, and annually regardless. Every `[ ]` item
> should become `[x]` with a date and evidence link before a new product
> goes live.

**Covered Entity:** ANNA Health (Allied Network for Neurodevelopmental Advancement)
**HIPAA Officer:** Aimun Malik — aimun@annaautismcare.com
**Product:** `<PRODUCT-NAME>`
**Environment:** `<dev / staging / prod>`
**Last reviewed:** `<YYYY-MM-DD>`
**Next review due:** `<YYYY-MM-DD>` (max annual)

---

## 1. Administrative safeguards (45 CFR § 164.308)

### 1.1 Security Management Process
- [ ] Written risk analysis exists for this product (threats, vulnerabilities,
      likelihood, impact). Date: `___`. Filed at: `___`.
- [ ] Risk management plan documents how identified risks are being reduced
      to "reasonable and appropriate."
- [ ] Sanction policy for workforce violations exists and has been
      acknowledged by all workforce members with access.
- [ ] Information system activity review (log review) cadence defined:
      `<e.g. weekly>`. Responsible party: `<name>`.

### 1.2 Assigned Security Responsibility
- [ ] HIPAA Security Officer designated. Name: `<name>`, contact: `<email>`.

### 1.3 Workforce Security
- [ ] Authorization/supervision procedures for workforce members with access
      to PHI via this product.
- [ ] Workforce clearance procedure (background checks where applicable).
- [ ] Termination procedure — see `ONBOARDING.md` for the specific steps
      when a user leaves.

### 1.4 Information Access Management
- [ ] Access authorization policy documented (who approves new access).
- [ ] Access establishment and modification procedures documented.
- [ ] **Minimum-necessary** access applied — no user has more access than
      their role requires.

### 1.5 Security Awareness and Training
- [ ] All workforce members with product access have completed HIPAA training.
      Tracking sheet: `___`.
- [ ] Periodic security reminders sent (phishing, password hygiene, MFA).
- [ ] Protection from malicious software: documented and enforced.
- [ ] Log-in monitoring: Cognito advanced security is ENFORCED (captures
      anomalous sign-ins).
- [ ] Password management: Cognito-enforced policy (length, complexity, MFA
      TOTP). Document the enforced policy in this file.

### 1.6 Security Incident Procedures
- [ ] Incident response runbook exists → see `SECURITY_RUNBOOK.md`.
- [ ] Workforce knows how to report a suspected incident (who to email/call).

### 1.7 Contingency Plan
- [ ] Data backup plan documented → DDB PITR + S3 versioning (covered by
      `DISASTER_RECOVERY.md`).
- [ ] Disaster recovery plan documented → `DISASTER_RECOVERY.md`.
- [ ] Emergency mode operation plan (how to operate during a system failure
      without losing PHI).
- [ ] Testing and revision procedures: quarterly DR drill on calendar.
      Last drill: `___`. Next: `___`.
- [ ] Applications and data criticality analysis done.

### 1.8 Evaluation
- [ ] Periodic technical + non-technical evaluation scheduled. Date last
      done: `___`. Next due: `___`.

### 1.9 Business Associate Contracts (§ 164.308(b))
- [ ] AWS BAA active and filed. Location: `___`. Renewal date: `___`.
- [ ] Anthropic BAA (if applicable via Bedrock). Bedrock is covered under
      the AWS BAA; verify in AWS BAA scope doc.
- [ ] BAA for any other third-party that touches PHI. List every one:

| Vendor | Service | BAA filed | Date |
|---|---|---|---|
| AWS | `<services used>` | Yes | `___` |
| `<other>` | `<what>` | `<yes/no>` | `___` |

---

## 2. Physical safeguards (45 CFR § 164.310)

Most physical safeguards are inherited from AWS (their responsibility under
the BAA). Document the inheritance:

- [ ] Facility access controls: AWS responsible (SOC 2, ISO 27001).
- [ ] Workstation use policy: workforce-facing policy documents which
      workstations/laptops may access PHI + required configuration
      (full-disk encryption, auto-lock, MFA).
- [ ] Device and media controls: policy for portable devices that ever
      cache PHI.

---

## 3. Technical safeguards (45 CFR § 164.312)

### 3.1 Access Control
- [x] Unique user identification: Cognito user per person.
- [x] Emergency access procedure: break-glass admin account documented.
      Where: `___`.
- [x] Automatic logoff: Cognito session duration `<hours>`; refresh token
      expiry `<days>`.
- [x] Encryption and decryption: SSE-KMS with customer-managed keys on
      every DDB table and S3 bucket.

### 3.2 Audit Controls
- [x] CloudTrail enabled, delivering to a dedicated log bucket with Object
      Lock + 7-year retention.
- [x] CloudWatch application logs structured as JSON; field-level PHI
      redaction at the formatter. Log group: `/aws/lambda/<product>-<env>-*`.
- [x] Downstream analysis: logs searchable via CloudWatch Logs Insights.
- [ ] Log review cadence defined and honored — see 1.1.

### 3.3 Integrity
- [x] DynamoDB PITR enabled on every table.
- [x] S3 Versioning enabled on every bucket.
- [x] S3 Object Lock (governance mode) on `<PHI-storing buckets>`.
- [x] No in-place PHI modification without audit trail (CloudTrail covers
      API-level mutations; application-level change history via
      `updatedAt`/`createdAt` fields).

### 3.4 Person or Entity Authentication
- [x] Cognito User Pool with MFA (TOTP) **ENFORCED**.
- [x] Advanced Security Features **ENFORCED** (anomaly detection,
      compromised-credentials check).
- [x] JWT-based session with signature verified by API Gateway.

### 3.5 Transmission Security
- [x] TLS 1.2+ enforced:
  - CloudFront min protocol: TLSv1.2_2021
  - API Gateway HTTP API: TLS by default
  - S3 bucket policy: `aws:SecureTransport = true` deny for false
- [x] No cross-region or cross-account replication of PHI to non-BAA
      environments.

---

## 4. Organizational requirements (45 CFR § 164.314)

- [x] BAA with AWS in effect.
- [x] BAA with `<any other vendor>` in effect where applicable.
- [ ] BAA coverage includes the specific services used (check the AWS
      BAA service list includes every service listed in `ARCHITECTURE.md`
      section 2).

---

## 5. Policies and procedures + documentation (§ 164.316)

- [ ] All policies referenced here exist in writing, are version-controlled,
      and are available to workforce members.
- [ ] Retention: all policies + procedures retained for 6 years from
      creation or last effective date, whichever is later.

---

## 6. Breach Notification Rule (§ 164.400–414)

- [ ] Breach notification procedure documented. Includes: internal
      escalation path, timing (60-day max to affected individuals),
      HHS reporting threshold (500+ individuals → immediate).
- [ ] Log review (1.1) designed to detect potential breaches.
- [ ] Template breach notification letters on file.

---

## 7. Product-specific checks

These are the things that go wrong at the code layer specifically for this
product. Port from `anna-chat` and add product-specific ones.

- [x] `JsonFormatter.PHI_FIELDS` includes every field name that could carry
      chat content, user input, uploaded document content, prompt text, or
      LLM completion.
- [x] No `logger.info(f"...{user_input}...")` formatted strings in handler
      code — content always goes in structured fields so it can be stripped
      (verified via grep in CI).
- [x] Every IAM policy is resource-scoped to `<product>-<env>-*` prefix.
      No `Resource: "*"` outside describe/list on AWS-managed metadata.
- [x] No raw Python `float` written into DDB anywhere — all writes pass
      through a `float→Decimal` sanitizer. See `_floats_to_decimal` in
      `anna_chat/ddb.py` for the reference pattern.
- [x] Every Lambda that's in VPC has a VPC interface endpoint for every
      AWS service it calls. Specifically verified for: `<list services>`.
- [ ] `<product-specific check 1>`
- [ ] `<product-specific check 2>`

---

## 8. Annual review checklist

To be completed once per year at `<month>`:

- [ ] Re-read the full checklist above and confirm every `[x]` is still true.
- [ ] Re-run risk analysis.
- [ ] Review access list — remove any users who no longer need access.
      Use `ONBOARDING.md` termination procedure.
- [ ] Review BAA list — confirm none have expired or gone unsigned.
- [ ] Review CloudTrail + CloudWatch logs retention and actual storage size.
- [ ] Run DR drill (see `DISASTER_RECOVERY.md`).
- [ ] Simulate a tabletop incident (see `SECURITY_RUNBOOK.md`).
- [ ] Check for new services used in the product that may need BAA
      verification.
- [ ] Sign and date this document.

**Signed:** `<HIPAA Security Officer>` — `<YYYY-MM-DD>`

---

## 9. References

- 45 CFR Part 160 — General administrative requirements
- 45 CFR Part 162 — Administrative requirements (transaction codes)
- 45 CFR Part 164 — Security and privacy rules
- HHS Guidance on Risk Analysis Requirements: https://www.hhs.gov/hipaa/for-professionals/security/guidance/
- AWS HIPAA Compliance: https://aws.amazon.com/compliance/hipaa-compliance/
