import os
import requests
import boto3
from botocore.config import Config

def download_file(url, local_filename):
    try:
        if os.path.exists(local_filename):
            return local_filename, None
        parent = os.path.dirname(local_filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename, None
    except Exception as e:
        return None, e

def upload_to_s3(local_file, bucket_name, object_name):
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=os.getenv('BUCKET_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('BUCKET_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('BUCKET_SECRET_ACCESS_KEY'),
            config=Config(signature_version='s3v4'),
        )
        s3.upload_file(local_file, bucket_name, object_name)
        base = os.getenv('PUBLIC_URL_BASE')
        if base:
            return f"{base.rstrip('/')}/{object_name}", None
        return f"{os.getenv('BUCKET_ENDPOINT_URL')}/{bucket_name}/{object_name}", None
    except Exception as e:
        return None, e
