import sys
import time
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext

# Required Glue job args:
# --JOB_NAME
# --CONNECTION_NAME
# --DEST_URI
# --QUERY
#
# Example DEST_URI:
# s3://my-bucket/teradata-poc/glue/run_20260810_090000/

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "CONNECTION_NAME", "DEST_URI", "QUERY"],
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

t0 = time.time()

source = glueContext.create_dynamic_frame.from_options(
    connection_type="teradata",
    connection_options={
        "connectionName": args["CONNECTION_NAME"],
        "query": args["QUERY"],
    },
)

df = source.toDF()

# Do not call df.count() before the write for the timed throughput test:
# it can cause an extra source scan. Validate row count in a separate run if needed.
(
    df.write
      .mode("overwrite")
      .option("compression", "snappy")
      .parquet(args["DEST_URI"])
)

elapsed = time.time() - t0
print(f"POC_METRIC elapsed_seconds={elapsed:.3f}")
print(f"POC_METRIC destination={args['DEST_URI']}")

job.commit()