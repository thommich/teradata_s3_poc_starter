# Glue POC setup checklist

1. In Secrets Manager, create or identify an approved secret containing
   the Teradata username/password required by the Glue Teradata connection.

2. In AWS Glue -> Data connections:
   - choose Teradata Vantage
   - configure the Teradata URL/host details
   - select the Secrets Manager secret
   - configure VPC/subnet/security group if your network requires it

3. Job IAM role needs, at minimum for this POC:
   - permission to read the selected secret
   - permission to write the destination S3 prefix
   - KMS permissions if the secret/bucket uses a customer-managed KMS key

4. Add the Teradata connection as an Additional network connection to the job.

5. Upload `teradata_to_s3.py` as the Glue job script.

6. Pass job parameters:
   --CONNECTION_NAME <name>
   --DEST_URI s3://<bucket>/<unique-prefix>/
   --QUERY "SELECT ..."

7. Benchmark the first built-in-connector run as the simple Glue baseline.

Parallel-read follow-up:
AWS Glue supports parallel JDBC reads using hashfield/hashexpression and
hashpartitions for JDBC-table reads. Treat this as a second tuning experiment;
do not silently change the extraction semantics in the initial comparison.