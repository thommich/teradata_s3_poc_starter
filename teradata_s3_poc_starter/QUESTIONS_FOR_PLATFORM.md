# Questions to take to platform/DBA teams

## WRITE_NOS
- What Teradata/Vantage version are we running?
- Is Native Object Store / WRITE_NOS enabled for my environment?
- Do I have permission to invoke WRITE_NOS?
- Is there already an AUTHORIZATION object for our AWS account/bucket?
- If not, what Teradata AWS principal/account must the IAM role trust?
- What ExternalId / Vantage Site ID should be configured?
- Can Teradata network egress reach S3?
- Is our bucket policy/KMS policy compatible with Teradata's assumed role?

## TPT
- Is there an approved EC2/utility host with Teradata Tools & Utilities installed?
- Is the Teradata Access Module for S3 installed (`libs3axsmod.so`)?
- Which TTU/TPT version is approved?
- Can that host connect to Teradata and S3?
- May we use an EC2 instance profile for S3?
- Are there workload-management/session limits for FastExport/TPT exports?

## Glue
- Is there already a Glue Teradata Vantage connection?
- Can Glue subnets/security groups reach Teradata?
- Which Secrets Manager secret should hold/provide Teradata credentials?
- Which Glue IAM role may write to the POC prefix?
- Which Glue version is approved?
- Is the Glue 5.0+ Teradata Vantage NOS connector exposed in our account?