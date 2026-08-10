# TPT POC validation

## Fast fail checks

```bash
# Teradata documentation says invoking tbuild with no options displays version.
tbuild

# Find S3 access module:
find /opt/teradata -name 'libs3axsmod.so' 2>/dev/null | head

# On EC2, validate the instance profile:
aws sts get-caller-identity
aws s3 ls s3://<bucket>/<base-prefix>/
```

If TPT or libs3axsmod.so is missing, stop and record the prerequisite.

## Authentication

For S3, prefer the EC2 instance role. The S3 Access Module can use IAM roles
on EC2, avoiding explicit AWS access keys.

For Teradata credentials, use your organization's accepted TPT approach
(job variables, wallet, external auth where configured). Do not commit
passwords in the TPT script.

## First transfer

Fill the schema in `export_to_s3_template.tpt`, then run:

```bash
tbuild -f export_to_s3_template.tpt -j td_s3_poc_<timestamp>
```

Use a unique job name and S3 prefix.

## Throughput tuning after the baseline

The S3 Access Module has its own S3 connection/buffer controls. Do not tune
them before the baseline works. Afterward, one controlled experiment may
raise S3ConnectionCount while keeping the source query and EC2 size fixed.

TPT's raw S3 output is not the Parquet result used by WRITE_NOS/Glue.
Record raw transfer time separately, then add a conversion-to-Parquet step
for the end-to-end comparison.