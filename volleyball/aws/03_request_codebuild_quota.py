# Step 03 - ask AWS to raise CodeBuild's concurrent-build quota from 0 to 1 (Linux Medium and Large).
#
# Why: this account's applied quota for every CodeBuild environment is 0 (AWS default is 1), so
# StartBuild fails with "Cannot have more than 0 builds in queue for the account". The request is
# free; AWS reviews it (often within hours, sometimes a day). Step 05 needs it.
# CLI equivalent, for reference:
#   aws service-quotas request-service-quota-increase --service-code codebuild --quota-code L-2DC20C30 --desired-value 1 --region us-east-1
#   aws service-quotas request-service-quota-increase --service-code codebuild --quota-code L-4DDC4A99 --desired-value 1 --region us-east-1
QUOTAS = {"L-2DC20C30": "Concurrently running builds for Linux/Medium environment",
          "L-4DDC4A99": "Concurrently running builds for Linux/Large environment"}
out = {}
for code, name in QUOTAS.items():
    try:
        r = await call_boto3(service_name="service-quotas", operation_name="RequestServiceQuotaIncrease",
                             region_name="us-east-1", params={"ServiceCode": "codebuild", "QuotaCode": code, "DesiredValue": 1})
        rq = r["RequestedQuota"]
        out[name] = {"id": rq["Id"], "status": rq["Status"], "case": rq.get("CaseId")}
    except Exception as e:
        out[name] = {"error": str(e)[:300]}
result = out
result
