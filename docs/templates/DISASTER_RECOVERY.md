# `<PRODUCT-NAME>` — Disaster Recovery Runbook

> **How to use this doc:** every scenario has a **detection**, **containment**,
> and **recovery** section. Keep it practical — if the steps can't be followed
> by the on-call person at 2 AM with a phone and AWS console access, they're
> wrong. Run a drill quarterly.

**Scope:** `<PRODUCT-NAME>` in AWS account `<account-id>`, region `<region>`.
**Owner:** `<name + contact>`.
**Drill cadence:** quarterly. Last drill: `<date>`. Next drill: `<date>`.

---

## Recovery objectives

| Metric | Target | Reasoning |
|---|---|---|
| **RPO** (max acceptable data loss) | `<e.g. 5 minutes>` | DynamoDB PITR continuous; S3 Versioning preserves object history |
| **RTO** (max acceptable downtime) | `<e.g. 4 hours>` | Lambda+S3+CloudFront recovery is fast; DB restore-to-new-table is the long pole |

If real-world impact exceeds these targets, that's a learning for the next
drill — capture it in this doc.

---

## Backup posture (what we have)

- **DynamoDB**: Point-in-Time Recovery (PITR) enabled on every table.
  Continuous backups retain the last 35 days; restore to any second in
  that window.
- **S3**: Versioning enabled on every bucket. Non-current versions retained
  per each bucket's lifecycle (see `ARCHITECTURE.md` storage table).
  Object Lock on buckets that hold regulated content.
- **CloudTrail**: 7-year retention in a dedicated bucket with Object Lock
  (governance mode). Cannot be deleted during the lock window.
- **CloudWatch Logs**: 30-day retention per log group. If logs are needed
  longer than that, export to S3 proactively; expired logs are gone.
- **Lambda code**: every version is kept; `$LATEST` always points at the
  most recent deploy. Function zips also remain in S3 as deploy artifacts
  for as long as the CI workflow keeps them.
- **Cognito**: user pool backup is **not** a native feature. Export users
  via `aws cognito-idp list-users` + archive monthly (see annual review).
- **Terraform state**: S3 bucket versioned, SSE-KMS. Worst-case state-bucket
  recovery: from version history.

---

## Scenario 1: DynamoDB data corruption or accidental deletion

**Detection:**
- User report of missing / wrong data.
- CloudWatch alarm on unusual `DeleteItem` rate.
- Failed app read paths that previously worked.

**Containment (first 5 minutes):**
1. Identify the affected table(s). Record names + timestamps.
2. If a bad application version is writing corrupt data, **roll back**
   the Lambda first:
   ```
   aws lambda update-function-code \
     --function-name <product>-<env>-<name> \
     --s3-bucket <deploy-bucket> --s3-key <previous-zip-key>
   ```
   Or revert the Git commit and re-run the deploy workflow.
3. If deletion is ongoing, tighten IAM: remove `DeleteItem`/`BatchWriteItem`
   from the offending Lambda's policy temporarily.

**Recovery:**
1. Identify the **restore point** — the last second before corruption.
   CloudWatch + CloudTrail helps narrow this.
2. Restore to a **new table** (PITR cannot restore in-place):
   ```
   aws dynamodb restore-table-to-point-in-time \
     --source-table-name <product>-<env>-<table> \
     --target-table-name <product>-<env>-<table>-restore \
     --restore-date-time 2026-04-21T17:00:00Z \
     --use-latest-restorable-time
   ```
3. Validate the restored table (spot check a few known-good items).
4. **Cut over**: either rename by migration (app config swap) or copy
   restored data back into the original table via `BatchWriteItem` loop.
   If copying: disable writes to the original first, copy, re-enable.
5. Verify CloudWatch dashboard is healthy post-cutover.
6. Delete the `-restore` table after 30 days of known-good operation.

**Post-mortem:**
- Which Lambda/code path caused the corruption?
- How did it get past CI tests?
- Add a regression test so this class of bug cannot ship again.

---

## Scenario 2: S3 object deleted or overwritten

**Detection:**
- User report of missing attachment / KB doc.
- CloudWatch alarm on `DeleteObject` volume.
- App 404 on a known object key.

**Containment:**
1. Identify affected bucket + key(s).
2. If the cause is a bad Lambda, roll back as above.
3. Check whether Object Lock is on this bucket. If yes, the object is
   still there in its locked version — regardless of delete markers.

**Recovery:**
1. List versions:
   ```
   aws s3api list-object-versions \
     --bucket <bucket> --prefix <key-prefix>
   ```
2. If the deletion is a **delete marker**, remove it:
   ```
   aws s3api delete-object \
     --bucket <bucket> --key <key> \
     --version-id <delete-marker-version-id>
   ```
   The previous version becomes current again.
3. If the object was overwritten, restore a specific version:
   ```
   aws s3api copy-object \
     --bucket <bucket> --key <key> \
     --copy-source <bucket>/<key>?versionId=<version-id>
   ```
4. If a DDB record points at this object (e.g. an attachment row), verify
   the app now resolves it correctly. Fix metadata if needed.

**If you're past the non-current retention window**: the object is gone.
Document the gap and consider tightening retention policy next review.

---

## Scenario 3: Lambda deploy introduces a regression

**Detection:**
- Post-deploy spike in 5xx errors in CloudWatch metrics.
- CloudWatch alarm on Lambda error rate.
- User reports.

**Containment + recovery (one step):**
1. **Revert the Git commit** and push to main:
   ```
   git revert <bad-commit> && git push origin main
   ```
   CI re-deploys the previous known-good code within ~5 minutes.
2. Alternatively, roll back just the Lambda without a Git revert:
   ```
   aws lambda update-function-code \
     --function-name <product>-<env>-<name> \
     --zip-file fileb://backend/lambda.zip.<timestamp>
   ```
   (requires that you keep previous zip artifacts)
3. Verify error rate returns to baseline.
4. Leave the bad commit reverted until the underlying bug is fixed on
   a branch, tested, and re-deployed.

---

## Scenario 4: Cognito user pool misconfiguration

**Detection:**
- All users suddenly cannot log in.
- MFA prompts missing / broken.
- SAML/OIDC callback errors.

**Containment:**
1. **Do not delete the user pool.** Cognito user pools are effectively
   irreplaceable — the user directory and JWT signing key are tied to
   the pool ID, and user-facing auth integrations will all break if
   the ID changes.
2. Identify the misconfiguration via CloudTrail `cognito-idp:Update*`
   events in the last hour.
3. Revert the setting in the AWS console or via Terraform.

**Recovery:**
- If a Terraform change caused it: revert the commit, re-apply.
- If a console change caused it: reverse it manually, then bring
  Terraform state back in sync (`terraform apply` with the original
  config should show no diff once the console reversal is done).

**If the pool is accidentally deleted:**
- Within 7 days: AWS Support can (sometimes) restore a deleted pool.
  Open a support case immediately; every hour reduces the chance.
- After 7 days: gone. You'd need to stand up a new pool, migrate users
  via admin-create-user flow (they'll need to re-enroll MFA), update
  every referring system (frontend env vars, JWT authorizer config).
  This is a **multi-day outage** scenario — prevention is the only
  real answer.

---

## Scenario 5: Full region outage (AWS us-east-1 down)

**Detection:**
- AWS Health Dashboard shows the region is degraded/down.
- CloudFront still serves cached SPA pages; API calls fail.

**Containment:**
- Communicate expected outage to users via `<status page or email>`.
- There is no in-product failover today — accept the outage.

**Recovery:**
- When AWS restores the region, verify all services return: API
  responses, Lambda invocations, DDB reads, S3 GETs.
- Check CloudWatch for any stuck/orphaned operations.
- Run smoke tests (see `SECURITY_RUNBOOK.md` § smoke suite).

**Long-term uplift (if tolerance drops):** multi-region replication is
significant engineering — not in scope today. Document the gap.

---

## Scenario 6: CloudTrail tampering or disable attempt

**Detection:**
- CloudWatch alarm on `cloudtrail:StopLogging` or
  `cloudtrail:DeleteTrail` events.
- Gaps in audit log stream.

**Containment:**
1. This is almost certainly a compromised IAM credential. **Rotate
   immediately** — see `SECURITY_RUNBOOK.md` § credential compromise.
2. Re-enable CloudTrail logging via console or CLI.
3. Because the log bucket has Object Lock (governance mode), prior logs
   cannot be deleted — audit the CloudTrail records for the gap window
   to see what the attacker did.

**Recovery:**
- Follow full credential-compromise playbook.
- Verify Object Lock is still in place post-incident.
- Consider escalating to compliance/legal if PHI access occurred during
  the gap.

---

## Scenario 7: Terraform state corruption or lost

**Detection:**
- `terraform plan` shows a massive diff (wants to recreate half the
  stack).
- Deploy fails mid-apply and subsequent plans are inconsistent.

**Containment:**
1. **Do not run `terraform apply` blind.** Terraform will happily destroy
   resources it doesn't know about.
2. The state bucket has versioning enabled. Check state history:
   ```
   aws s3api list-object-versions --bucket <state-bucket> \
     --prefix envs/<env>/terraform.tfstate
   ```

**Recovery:**
1. Find the last known-good state version.
2. Download it:
   ```
   aws s3api get-object --bucket <state-bucket> \
     --key envs/<env>/terraform.tfstate \
     --version-id <version-id> terraform.tfstate.good
   ```
3. Upload it as the current version (this moves forward; it doesn't
   delete the current corrupt state — a safety net):
   ```
   aws s3 cp terraform.tfstate.good \
     s3://<state-bucket>/envs/<env>/terraform.tfstate
   ```
4. `terraform plan` locally to verify it matches reality.
5. If reality has drifted (someone made console changes), use
   `terraform import` to reconcile.

---

## Smoke test suite (run after every DR event)

Once recovery is done, run these to verify the product is healthy:

1. Frontend loads: `curl -I https://<app-domain>/` returns 200.
2. API returns: `curl -I https://<api-endpoint>/` returns the expected
   401 (unauthenticated).
3. Login: complete an end-to-end sign-in for a known test user.
4. Primary flow: execute the single most important user journey (e.g.
   chat + KB retrieval + source download for Praxis).
5. Admin flow: execute the main admin action (e.g. upload a KB doc).
6. CloudWatch: no new 5xx errors in the last 5 minutes across any
   Lambda.

---

## Quarterly drill template

Pick one scenario per quarter, run it end-to-end on a copy of prod (or
dev if copy-of-prod is impractical). Record:

- **Scenario:** `<which one>`
- **Date:** `___`
- **Participants:** `___`
- **Detection time** (from "scenario starts" to "someone notices"): `___`
- **Recovery time** (from detection to "fully restored"): `___`
- **RPO achieved** (actual data loss window): `___`
- **RTO achieved** (actual downtime window): `___`
- **What went well:**
- **What went badly:**
- **Changes to runbook:**

---

## Phone tree

When an incident happens during off-hours:

1. `<primary on-call name, phone, email>`
2. `<secondary escalation>`
3. HIPAA Security Officer: `<name, contact>`
4. AWS Support (if paid plan): `<case creation link>`
