import argparse
import pyarrow.dataset as ds

parser = argparse.ArgumentParser()
parser.add_argument("--uri", required=True)
args = parser.parse_args()

dataset = ds.dataset(args.uri, format="parquet")
schema = dataset.schema
print("schema:")
print(schema)

# Metadata-only row count when supported by the dataset fragments.
rows = dataset.count_rows()
print(f"rows={rows}")