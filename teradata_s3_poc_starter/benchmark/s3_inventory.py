import argparse
from urllib.parse import urlparse
import boto3

def parse_s3(uri: str):
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError("URI must start with s3://")
    return p.netloc, p.path.lstrip("/")

parser = argparse.ArgumentParser()
parser.add_argument("--uri", required=True)
args = parser.parse_args()

bucket, prefix = parse_s3(args.uri)
s3 = boto3.client("s3")

count = 0
size = 0
token = None
while True:
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    if token:
        kwargs["ContinuationToken"] = token
    resp = s3.list_objects_v2(**kwargs)
    for obj in resp.get("Contents", []):
        count += 1
        size += obj["Size"]
    if not resp.get("IsTruncated"):
        break
    token = resp["NextContinuationToken"]

gib = size / (1024**3)
print(f"uri={args.uri}")
print(f"objects={count}")
print(f"bytes={size}")
print(f"GiB={gib:.6f}")