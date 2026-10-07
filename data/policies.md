# Support Policies (synthetic knowledge base for RAG)

## Refunds & duplicate charges
Duplicate or erroneous charges are refunded to the original payment method within
5-7 business days once confirmed. Customers do not need to return anything. Goodwill
credits may be offered for repeated billing errors.

## Plan changes & proration
Customers can switch between monthly and annual billing at any time. Mid-cycle changes
are prorated: unused time on the old plan becomes a credit toward the new plan. Plan
upgrades take effect immediately; new seats can take up to 10 minutes to propagate.

## Invoices
Invoices are available under Billing > Invoices. A date-range filter and a
"combine into single PDF" export option are available for the selected range.

## Account & contact details
Contact details are changed under Settings > Account. A greyed-out save button usually
means a pending unverified change; verification by email is required. Account closure
triggers data deletion within 30 days; saved payment methods are removed immediately.

## Authentication
Password reset emails come from no-reply@example.com; customers should allowlist it.
Agents can trigger a manual reset link after identity verification.

## Rate limits (API)
Limits apply per-minute as well as monthly. HTTP 429 indicates the per-minute limit was
exceeded. Recommended remedy: exponential backoff and honoring the `Retry-After` and
rate-limit response headers.

## Single sign-on (SAML)
After rotating a SAML signing certificate, the new certificate must be re-uploaded under
Settings > SSO. SSO breaks org-wide until the metadata is refreshed.

## Mobile app known issues
A dashboard crash on Android 14 is a known issue; the interim workaround is clearing the
app cache. A permanent fix is in progress.
