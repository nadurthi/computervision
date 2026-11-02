# Step 01 - private S3 bucket for datasets, clips, labels, models and runs (us-east-1).
#
# Run through the AWS MCP server's run_script (async boto3 sandbox: `call_boto3`, top-level await).
# CLI equivalent, for reference:
#   aws s3api create-bucket --bucket vb-data-493280347481 --region us-east-1
#   aws s3api put-public-access-block --bucket vb-data-493280347481 --public-access-block-configuration \
#       BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
#   aws s3api put-bucket-versioning --bucket vb-data-493280347481 --versioning-configuration Status=Enabled
#   aws s3api put-bucket-lifecycle-configuration --bucket vb-data-493280347481 --lifecycle-configuration file://lifecycle.json
# Encryption: new buckets get SSE-S3 (AES256) by default; the script reads it back to confirm.
B = "vb-data-493280347481"
await call_boto3(service_name="s3", operation_name="CreateBucket", region_name="us-east-1", params={"Bucket": B})
await call_boto3(service_name="s3", operation_name="PutPublicAccessBlock", region_name="us-east-1", params={"Bucket": B, "PublicAccessBlockConfiguration": {"BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True}})
await call_boto3(service_name="s3", operation_name="PutBucketVersioning", region_name="us-east-1", params={"Bucket": B, "VersioningConfiguration": {"Status": "Enabled"}})
await call_boto3(service_name="s3", operation_name="PutBucketLifecycleConfiguration", region_name="us-east-1", params={"Bucket": B, "LifecycleConfiguration": {"Rules": [
  {"ID": "abort-partial-uploads", "Status": "Enabled", "Filter": {}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}},
  {"ID": "expire-old-versions", "Status": "Enabled", "Filter": {}, "NoncurrentVersionExpiration": {"NoncurrentDays": 30}}]}})
v = await call_boto3(service_name="s3", operation_name="GetBucketVersioning", region_name="us-east-1", params={"Bucket": B})
pab = await call_boto3(service_name="s3", operation_name="GetPublicAccessBlock", region_name="us-east-1", params={"Bucket": B})
enc = await call_boto3(service_name="s3", operation_name="GetBucketEncryption", region_name="us-east-1", params={"Bucket": B})
result = {"versioning": v.get("Status"), "public_access_block": pab.get("PublicAccessBlockConfiguration"),
          "encryption": enc["ServerSideEncryptionConfiguration"]["Rules"][0]["ApplyServerSideEncryptionByDefault"]}
result
