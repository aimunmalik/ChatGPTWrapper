# `<PRODUCT-NAME>` — Security Runbook

> **How to use this doc:** security incident ≠ outage. An outage is handled
> in `DISASTER_RECOVERY.md`. This doc covers incidents where *someone or
> something might be acting against you*. Each scenario has: how to detect
> it, first 10 minutes, investigation, remediation, when to escalate.

**Product:** `<PRODUCT-NAME>`
**Security Officer:** `<name + contact>`
**HIPAA Officer:** Aimun Malik — aimun@annaautismcare.com
**On-call rotation:** `<who + how reached>`

---

## Severity levels

| Severity | Definition | Response |
|---|---|---|
| **SEV-1** | Active PHI exposure, credential compromise confirmed, service-wide unauthorized access | Page immediately, begin breach notification clock |
| **SEV-2** | Strong signal of attempted compromise, no confirmed PHI access | Investigate within 1 hour |
| **SEV-3** | Anomaly requiring review but no evidence of harm | Investigate within 1 business day |

When in doubt, treat as higher severity. Downgrading after investigation
is cheap. The 60-day HIPAA breach notification clock starts on
**discovery**, not confirmation.

---

## Scenario 1: Cognito credentials compromised

### Detection
- User reports unexpected login activity.
- Cognito Advanced Security flags anomalous sign-in (check Cognito
  console → Threat protection → Event history).
- Unusual chat/attachment activity attributed to one user.

### First 10 minutes (SEV-1)
1. **Disable the user account** (don't delete — we want to preserve
   evidence):
   ```
   aws cognito-idp admin-disable-user \
     --user-pool-id <pool-id> --username <username>
   ```
2. **Invalidate all sessions**:
   ```
   aws cognito-idp admin-user-global-sign-out \
     --user-pool-id <pool-id> --username <username>
   ```
3. Note the exact time of disable for the timeline.
4. Notify the user via a trusted out-of-band channel (phone, not email
   if email could also be compromised).

### Investigation
1. Pull CloudTrail + CloudWatch logs for everything the user did in the
   last 7 days:
   ```
   aws logs filter-log-events \
     --log-group-name /aws/lambda/<product>-<env>-chat \
     --filter-pattern '"userId":"<cognito-sub>"' \
     --start-time <epoch-ms>
   ```
2. Cross-reference Cognito sign-in history (Cognito console) for IP
   addresses and device fingerprints.
3. Identify what PHI, if any, was accessed. For chat products: list
   all conversationIds touched. For KB: all kb_download events.
4. Determine whether the attacker:
   - Read existing PHI (breach potential).
   - Sent messages / created content (data integrity concern).
   - Made admin-level changes (scope widens).

### Remediation
1. Rotate the user's password + re-enroll MFA from scratch (new TOTP
   device).
2. Re-enable the account only after the user has confirmed the new
   credentials and has reviewed the session activity with you.
3. Audit any IAM roles / admin-group membership — remove if
   compromised user was an admin.
4. If admin credentials were used to modify infrastructure (check
   CloudTrail for `iam:*`, `kms:*`, `cognito-idp:*` events by that
   user): expand scope — see Scenario 5.

### Escalation / breach notification
- If PHI was accessed and we cannot confirm the attacker does not still
  have it → **possible breach**. Consult HIPAA Officer. HHS breach
  notification clock is 60 days from discovery for individual notices.
- Document everything with timestamps in a new incident file:
  `docs/incidents/YYYY-MM-DD-<slug>.md`.

---

## Scenario 2: WAF alarm — rate-limit triggered, unusual traffic

### Detection
- CloudWatch alarm on WAF `BlockedRequests` metric.
- WAF logs show high-volume traffic from a small set of IPs / ASNs.
- App latency spike correlated with traffic spike.

### First 10 minutes (SEV-2)
1. Confirm in AWS WAF console: which rule fired, which IPs, what
   request patterns.
2. If rule is working as intended (blocks exceed allows by wide
   margin): no action required — WAF is doing its job.
3. If **legitimate** users are being blocked: temporarily increase
   the rate-limit threshold while investigating (edit WAF rule in
   Terraform and deploy).

### Investigation
- Is the traffic pattern targeted (repeated attempts on `/chat` or
  a specific path) vs. broad scraping?
- Geo / ASN origin. If from unusual country / cloud provider, likely
  bots/scanning.
- Any successful requests from those IPs in the window? Check API
  Gateway access logs.

### Remediation
- If targeted attack on a specific endpoint: add a specific WAF rule
  for that path with tighter limit or IP-set block.
- If bots: usually the managed rule groups handle it; no action
  beyond monitoring.
- If you added temporary allow rules during containment, **put a
  calendar reminder to remove them** within 7 days.

---

## Scenario 3: Bedrock / API spend alarm

### Detection
- AWS Budget threshold alarm.
- CloudWatch alarm on Bedrock invocation count or cost metric.

### First 10 minutes (SEV-2)
1. Check the **Budget Action** status — if configured to auto-disable
   Bedrock, it may have already fired. Verify via IAM (role attached
   to the action is applied = Bedrock disabled for your role).
2. Check the primary chat Lambda's recent invocations: sudden volume
   spike? Unusual user causing it?
3. Check CloudWatch for `chat_turn_complete` log volume per `userId`
   over the last hour.

### Investigation
- One user monopolizing the service? Their account compromised?
- Script/bot making repeated requests? Check user-agent patterns in
  API Gateway access logs.
- Intentional abuse by an insider? Escalate if so.

### Remediation
- If a specific user: disable their Cognito account temporarily;
  contact them.
- If no single user and spend is legitimate: raise the budget and
  the kill-switch threshold, after informing the owner.
- If the kill-switch fired: unblock by removing the deny policy the
  Budget Action attached, only after the root cause is handled.

---

## Scenario 4: Lambda 5xx rate spike (without a deploy)

### Detection
- CloudWatch alarm on Lambda Errors > baseline.
- 500 responses in the SPA from multiple users.

### First 10 minutes (SEV-2)
1. Pull recent error logs:
   ```
   aws logs tail /aws/lambda/<product>-<env>-<name> \
     --since 10m | grep ERROR
   ```
2. Classify the error:
   - `ClientError` + `awsErrorCode: AccessDeniedException` → IAM drift.
   - `ClientError` + `ThrottlingException` → downstream AWS service
     throttle.
   - `TypeError` / `AttributeError` → code bug shipped in a recent
     deploy (even if no deploy today, a warm container may be running
     old-ish code with a late-triggered path).
   - `ConnectTimeoutError` → VPC endpoint missing or network misroute.

### Remediation
- IAM drift: someone changed IAM manually. Re-apply Terraform.
- Downstream throttle: temporary; will self-resolve. Consider adding
  retry/backoff if missing.
- Code bug: see `DISASTER_RECOVERY.md` scenario 3 (rollback).
- VPC issue: add the missing endpoint and re-apply.

---

## Scenario 5: Admin credentials compromised / unexplained infra changes

### Detection
- CloudTrail shows `iam:*` / `kms:*` / `cognito-idp:*` / `s3:DeleteBucket`
  / `lambda:DeleteFunction` events that no one claims responsibility for.
- Terraform state shows drift from last known config.
- Someone was added to the `admins` Cognito group without an ONBOARDING
  record.

### First 10 minutes (**SEV-1** — this is serious)
1. **Freeze the blast radius.** Rotate every credential that could have
   made the change:
   - Root account password + MFA (use break-glass path if needed).
   - Every IAM user's access keys:
     ```
     aws iam list-access-keys --user-name <user>
     aws iam update-access-key --user-name <user> \
       --access-key-id <id> --status Inactive
     ```
   - Every GitHub Actions role trust relationship — tighten sub
     condition to known-good values.
2. **Enable CloudTrail log file validation** if not already, to prevent
   log tampering.
3. Snapshot the current state: export CloudTrail for the relevant
   window to a separate bucket for forensic use.

### Investigation
- Who had access to the credentials? Review the full CloudTrail event
  set for that identity in the last 30 days.
- Did they access PHI? (Check S3 GetObject, DDB Query/GetItem,
  CloudWatch logs:GetLogEvents on PHI-bearing log groups.)
- Did they exfiltrate anything? (Large `s3:GetObject` volume,
  `ecr:BatchGetImage` for containers, `codecommit:GitPull` etc.)
- Source IP? Cross-reference known office/personal IPs.

### Remediation
- Remove the attacker's artifacts (any IAM users/roles/policies/keys
  they created; any resources they spun up).
- Restore any deleted or modified infrastructure via Terraform.
- Verify CloudTrail, Object Lock, KMS key policies are intact —
  attackers often target these first to hide their trail.
- Post-incident: rotate KMS key material if any key policy was
  modified (even if reverted). KMS supports manual rotation.

### Escalation
- **This is almost certainly a breach-notification event** if any PHI
  was accessible during the window. Engage HIPAA Officer immediately.
- Consider legal counsel + cyber insurance if applicable.

---

## Scenario 6: Lost admin MFA / account lockout

### Detection
- Primary admin reports they've lost their MFA device and cannot log in.

### First 10 minutes (SEV-3)
1. Verify identity out-of-band (phone call, ideally video) — this is
   also a social-engineering vector.
2. If verified: use a **break-glass admin** account (which should exist,
   see `ONBOARDING.md`) to reset MFA for the primary admin via Cognito
   console.

### Recovery
1. Reset the admin's MFA in Cognito.
2. Admin re-enrolls MFA TOTP on a new device.
3. Admin changes password as an additional precaution.
4. Audit the last 24h of admin actions to confirm no malicious use
   during the lockout period.

**If there is no break-glass account:** this is a root-account
recovery scenario. Contact AWS Support; they will verify ownership
via account documents. Expect hours to days. Create a break-glass
account **now** if you don't have one — see `ONBOARDING.md`.

---

## Scenario 7: Uploaded attachment contains malicious content

### Detection
- User reports suspicious behavior after opening an uploaded file.
- A scan flags a file (if integrated with GuardDuty Malware
  Protection or equivalent).

### First 10 minutes (SEV-2)
1. **Quarantine the file** by deleting it from S3 via the console
   (Object Lock may prevent this — if so, the object stays and you
   proceed to blocking downloads).
2. Block further download: update the application code to skip this
   `attachmentId` or `kbDocId`, or disable the user's access.
3. Identify everyone who downloaded the file: check `attachment_download`
   or `kb_download` audit log lines for the file's ID.

### Investigation
- What is the malware type? (If integrated scanner, check the finding.)
- Did it trigger on the client side (browser viewer) or would it
  require manual execution?
- Which users downloaded it? Notify them out-of-band.

### Remediation
- Notify affected users to scan their machines.
- Remove the file from all locations.
- Review the upload pathway: how did a malicious file pass content-type
  / extension checks? Tighten if needed.

---

## Scenario 8: Unusual CloudTrail event pattern

Catch-all: anything flagged by `AWS Config`, `GuardDuty`, or manual
review that doesn't fit the above.

1. Classify severity based on actual / potential PHI exposure.
2. Follow the relevant scenario above.
3. If novel, document in `docs/incidents/` as a new runbook entry.

---

## Smoke test suite (post-incident validation)

After any SEV-1 or SEV-2 remediation:

```
# Frontend reachable
curl -I https://<app-domain>/ | head -1
# API 401 on unauthenticated
curl -s -o /dev/null -w "%{http_code}" https://<api-endpoint>/chat
# CloudTrail active
aws cloudtrail get-trail-status --name <trail-name>
# No new errors in the last 5 min
aws logs filter-log-events --log-group-name /aws/lambda/<product>-<env>-<name> \
  --start-time $(($(date +%s%3N) - 300000)) --filter-pattern ERROR
# WAF active
aws wafv2 list-web-acls --scope CLOUDFRONT
# Cognito pool intact
aws cognito-idp describe-user-pool --user-pool-id <pool-id> | head -20
```

---

## Breach notification decision tree

After any SEV-1 incident, answer in order:

1. **Was PHI accessed by an unauthorized person?**
   - No → not a reportable breach.
   - Yes → continue.
2. **Was the PHI encrypted to HHS standards (AES-256 at rest, TLS
   1.2+ in transit)?**
   - Yes, and the key was NOT compromised → **safe harbor**, not a
     reportable breach. Document the analysis.
   - No, or key was compromised → continue.
3. **Was the access low-probability-of-compromise under the 4-factor
   risk assessment?** (Nature/extent, unauthorized person, was it
   actually acquired/viewed, mitigation to which PHI is returned.)
   - Demonstrably low → document analysis, not reportable.
   - Otherwise → **reportable breach.**
4. **Notification deadlines:**
   - Affected individuals: 60 days from discovery.
   - HHS: 60 days from discovery if < 500 individuals; immediately
     if ≥ 500.
   - Media notice: if ≥ 500 in one state/jurisdiction.
   - Business associate (us) → covered entity (ANNA): as soon as
     practicable, within 60 days max.

Consult HIPAA Officer + legal counsel for any borderline case.
When in doubt, notify.

---

## Incident logging

Every SEV-1 or SEV-2 gets an entry in `docs/incidents/`:

```
docs/incidents/YYYY-MM-DD-<short-slug>.md
```

Template:

```markdown
# <Incident title>

- **Severity:** SEV-1/2/3
- **Discovered:** YYYY-MM-DD HH:MM UTC
- **Resolved:** YYYY-MM-DD HH:MM UTC
- **Reporter:** <name>
- **Responders:** <names>

## Summary

<one-paragraph what happened>

## Timeline
- HH:MM — <event>
- HH:MM — <event>

## Impact
- PHI exposed: <yes/no + details>
- Users affected: <count>
- Systems affected: <list>

## Root cause

<what actually caused it>

## Remediation

<what we did to stop it>

## Breach analysis

<4-factor assessment if PHI involved>

## Follow-up actions

- [ ] <preventive action>
- [ ] <preventive action>
```

Keep these forever — they are your best teacher.
