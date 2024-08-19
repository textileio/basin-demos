"""Basin S3 adapter connection and bucket interactions."""

import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from .util import get_numeric_suffix

# Enable debug logging for botocore
# boto3.set_stream_logger("botocore", level="DEBUG")


# Establish a connection to the Basin S3 adapter
def connect_to_basin(host, access_key, secret_key) -> boto3.client:
    try:
        # Create a session and S3 client
        session = boto3.session.Session()
        basin = session.client(
            "s3",
            endpoint_url=host,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                retries={"max_attempts": 3},
                s3={"addressing_style": "path"},
            ),
        )

        return basin

    except NoCredentialsError:
        print("Credentials not available")
    except PartialCredentialsError:
        print("Incomplete credentials provided")
    except Exception as e:
        print(f"An error occurred: {e}")


# Create a bucket in Basin
def create_bucket(client) -> str | None:
    try:
        # Create a bucket (the name doesn't matter)
        client.create_bucket(Bucket="foo")
        buckets = client.list_buckets()
        # Assume the last bucket is the one we just created
        bucket = buckets["Buckets"][-1]["Name"]

        return bucket

    except NoCredentialsError:
        print("Credentials not available")
    except PartialCredentialsError:
        print("Incomplete credentials provided")
    except Exception as e:
        print(f"An error occurred: {e}")


# Write an object to the store under the given key with a value
def write_object(client, bucket, key, value) -> None:
    try:
        client.put_object(Bucket=bucket, Key=key, Body=value)
    except Exception as e:
        print(f"An error occurred while writing data: {e}")


# List objects in the store, optionally, with a prefix
def list_objects(client, bucket, prefix=None) -> list | None:
    options = {"Bucket": bucket}
    if prefix is not None:
        options["Prefix"] = prefix
    try:
        response = client.list_objects(**options)
        if "Contents" in response:
            objects = response["Contents"]
            objects.sort(key=lambda x: get_numeric_suffix(x["Key"]))
            return objects
        else:
            print("No objects found in the bucket.")
            return None
    except Exception as e:
        print(f"An error occurred while listing objects: {e}")
        return None


# Get an object from the store by key
def get_object(client, bucket, key) -> dict | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except Exception as e:
        print(f"An error occurred while getting object: {e}")
        return None
