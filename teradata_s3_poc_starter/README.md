# Teradata -> S3 POC starter kit

Goal: compare three extraction paths while keeping the dataset out of SageMaker:
1. Teradata WRITE_NOS -> S3 Parquet
2. TPT Export/FastExport + S3 Access Module -> S3 landing format
3. AWS Glue Teradata connector -> Spark -> S3 Parquet

## Fill these values first

- TD_HOST
- TD_DATABASE / source objects
- A representative SELECT query
- S3_BUCKET
- AWS_REGION
- S3 base prefix, e.g. `teradata-poc/<your-user>/`
- Teradata credentials or approved auth mechanism
- For WRITE_NOS: Teradata AUTHORIZATION object name, if available
- For Glue: Glue connection name + job IAM role
- For TPT: EC2 host with TPT + S3 Access Module installed

Never paste secrets into source-controlled files.

## Recommended benchmark design

Use the exact same logical source data for all three paths.

Run two stages:
- Smoke test: small enough to fail fast and validate auth/network/schema.
- Performance test: a representative slice large enough that connection/job startup
  overhead is small relative to transfer time.

For every run capture:
- method
- start/end UTC timestamps
- elapsed seconds
- expected source row count
- output row count (when easy to measure)
- S3 bytes
- S3 object count
- output format
- effective throughput = S3 GiB / elapsed seconds
- notes/errors

Use a unique S3 prefix for every run:
`.../<method>/run_YYYYMMDD_HHMMSS/`

This is especially important for WRITE_NOS.

## Morning order

### 0. Common checks
Run:
```bash
aws sts get-caller-identity
aws s3 ls s3://<BUCKET>/<BASE_PREFIX>/
```

From the Teradata client, run a trivial query and then the query in
`sql/00_common_source_checks.sql`.

### 1. WRITE_NOS -- test first
Why first: it is the cleanest architecture and already produces Parquet.

Open `sql/01_write_nos_poc.sql`.

Validation gates:
- Does WRITE_NOS exist / do you have execute privilege?
- Does an approved AUTHORIZATION object exist?
- Can Teradata reach S3?
- Can the role/credential behind AUTHORIZATION PutObject to your prefix?
- Does a small Parquet export succeed?

If auth is the only blocker, collect:
- S3 bucket/prefix
- target IAM role ARN (ideally dedicated; POC may reuse an approved role)
- Teradata/Vantage external ID / site ID required by your platform
- Teradata principal/account that AWS must trust

### 2. TPT -- validate installation before doing any setup
On the EC2/host intended to run TPT:
```bash
tbuild
find /opt/teradata -name 'libs3axsmod.so' 2>/dev/null | head
aws sts get-caller-identity
aws s3 ls s3://<BUCKET>/<BASE_PREFIX>/
```

If `tbuild` or the S3 access module is absent, mark the POC as
"platform prerequisite missing" rather than spending the morning installing
enterprise software without the platform team's guidance.

Use `tpt/export_to_s3_template.tpt` as the job skeleton. It intentionally
requires you to fill the schema for your chosen POC query.

Important: TPT + S3 Access Module lands TPT's output stream in S3; it is not
the direct Parquet writer in this comparison. Measure:
A) Teradata -> raw S3 landing time
B) raw S3 -> Parquet conversion time
C) A+B end-to-end

Prefer an EC2 instance profile/IAM role for S3 instead of embedding AWS keys.

### 3. Glue -- simplest AWS-managed fallback
Create/identify:
- AWS Glue Teradata connection
- Secrets Manager secret containing Teradata credentials
- Glue IAM role allowed to read that secret and write the POC S3 prefix
- networking path from the Glue job to Teradata

Use `glue/teradata_to_s3.py`.

Start with the built-in Teradata connector and the exact SELECT query.
Then, only if needed, test a parallel JDBC/table read separately.

### 4. Validate the S3 outputs
Run:
```bash
python benchmark/s3_inventory.py \
  --uri s3://<BUCKET>/<RUN_PREFIX>/
```

For Parquet outputs:
```bash
python benchmark/parquet_probe.py \
  --uri s3://<BUCKET>/<RUN_PREFIX>/
```

## Decision rule

Prefer WRITE_NOS if:
- it is allowed,
- auth/network are supportable,
- and throughput is competitive.

Prefer TPT when:
- raw bulk-export throughput materially beats the alternatives,
- and the extra Parquet conversion/operational component is acceptable.

Prefer Glue when:
- platform simplicity/governance outweighs any throughput penalty,
- or WRITE_NOS/TPT prerequisites cannot be obtained.

## Bonus check
If your account exposes AWS Glue 5.0+ "Teradata Vantage NOS", record that
availability. It is worth a later test, but it should not replace the ordinary
Glue candidate in the first three-way benchmark because it uses NOS/WRITE_NOS
under the hood.