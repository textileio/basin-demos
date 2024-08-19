"""Initialize a bucket to store the raw data and labels for the pipeline."""

from os import getenv, listdir, path
from tempfile import TemporaryDirectory, TemporaryFile
from time import sleep

from datasets import load_dataset
from dotenv import load_dotenv
from polars import read_parquet

from .basin import (
    connect_to_basin,
    create_bucket,
    get_object,
    list_objects,
    write_object,
)
from .classify import process_df
from .fetch import chunk_dataset_to_parquet
from .util import get_numeric_suffix

load_dotenv()

# Define your S3 adapter connection details (or get from env variables)
host = getenv("S3_HOST") or "http://localhost:8014"
access_key = getenv("S3_ACCESS_KEY") or "S3EXAMPLEAK"
secret_key = getenv("S3_SECRET_KEY") or "S3EXAMPLESK"

# Set the chunk size to 10MB and the total data cap to 100MB
# CHUNK_SIZE = 10 * 1024 * 1024  # 10MB in bytes
# MAX_SIZE = 100 * 1024 * 1024  # 100MB in bytes
# For testing, we'll just use 10 KB and 20 KB
CHUNK_SIZE = getenv("CHUNK_SIZE") or 10 * 1024
MAX_SIZE = getenv("MAX_SIZE") or 20 * 1024

# Load the Grass dataset (in streaming mode so we can get only a subset)
dataset = load_dataset("OpenCo7/UpVoteWeb", split="train", streaming=True)

# Connect to the Basin S3 adapter
basin = connect_to_basin(host, access_key, secret_key)

# Create a bucket to store both the raw and processed data
bucket = create_bucket(basin)
print(f"Bucket created successfully: {bucket}")

# Write the raw dataset to parquet and store in Basin under the `raw` prefix
raw_key_prefix = "raw"
with TemporaryDirectory() as raw_dataset_dir:
    chunk_dataset_to_parquet(dataset, CHUNK_SIZE, MAX_SIZE, raw_dataset_dir)
    files = sorted(listdir(raw_dataset_dir), key=get_numeric_suffix)
    for file in files:
        with open(path.join(raw_dataset_dir, file), "rb") as f:
            write_object(basin, bucket, f"{raw_key_prefix}/{file}", f.read())
        print(f"Uploaded {file} to Basin at key: {file}")

# Get a list of objects in the bucket (i.e., original dataset has "raw/" prefix)
objects = list_objects(basin, bucket, f"{raw_key_prefix}/")

# Wait for 1 second before reading values
# TODO: why do we need to wait for resolving? or can we handle somehow in boto3
# (natively) vs trying to hit some Basin-specific API?
sleep(1)

# For each object, process the data through the ML pipeline and write the new
# results as a new object, which has a `topic` and `sentiment` label
processed_key_prefix = "processed"
with TemporaryFile() as f:
    for obj in objects:
        print(f"Processing object with key: {obj['Key']}")
        key = obj["Key"]
        value = get_object(basin, bucket, key)
        if value:
            df = read_parquet(value)
            processed = process_df(df)
            processed.write_parquet(f)
            f.seek(0)
            new_key = key.replace(raw_key_prefix, processed_key_prefix)
            write_object(basin, bucket, new_key, f.read())
        else:
            print(f"Failed to retrieve object with key: {key}")

processed_objects = list_objects(basin, bucket, f"{processed_key_prefix}/")

print("Processing complete.\n")
print("Check the Basin bucket for the processed data:")
print(f" - Bucket: {bucket}")
print(f" - Objects (at prefix): {processed_key_prefix}/")
print(f" - Example object: {processed_objects[0]['Key']}")
print(
    f"For example, with the CLI:\nadm os get --address {bucket} {processed_objects[0]['Key']} > processed.parquet"
)
