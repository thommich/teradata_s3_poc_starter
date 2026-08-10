-- ============================================================
-- WRITE_NOS POC
-- Replace placeholders before running.
-- Always use a fresh/unique output prefix for the POC.
-- ============================================================

-- Optional: if your platform team tells you to create an AssumeRole
-- AUTHORIZATION yourself and you have the required Teradata privilege:
--
-- CREATE AUTHORIZATION <AUTH_OBJECT>
-- USING
--   AUTHSERVICETYPE 'ASSUME_ROLE'
--   ROLENAME 'arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME>'
--   EXTERNALID '<VANTAGE_EXTERNAL_ID>'
--   DURATION_SECONDS '3600';
--
-- In many enterprise environments this object will be centrally managed.

SELECT COUNT(*) AS exported_rows
FROM WRITE_NOS
(
    ON
    (
        <PASTE_BENCHMARK_SELECT_HERE>
    )
    USING
        LOCATION('/s3/s3.amazonaws.com/<BUCKET>/<UNIQUE_RUN_PREFIX>/')
        AUTHORIZATION(<AUTH_OBJECT>)
        STOREDAS('PARQUET')
        COMPRESSION('SNAPPY')
) AS exported;

-- First POC: avoid PARTITION BY/HASH BY until the basic export is validated.
-- Then benchmark physical-layout tuning separately.