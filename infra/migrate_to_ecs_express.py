#!/usr/bin/env python3
"""Migrate the Claude-on-Bedrock gateway from App Runner to ECS Express Mode.

App Runner is closed to new customers and in maintenance (KTLO); AWS recommends
Amazon ECS Express Mode as the successor. Express Mode has no CloudFormation/CDK
support yet, so this idempotent script drives the documented CLI/API flow:

  1. Create an ECR repo and build+push the gateway image (port 8080).
  2. Ensure three IAM roles:
       - ecsTaskExecutionRole                 (pull image + logs)
       - ecsInfrastructureRoleForExpressServices (ALB/networking provisioning)
       - claude-gateway-task-role             (Bedrock invoke + read the token secret)
  3. Create the Express Gateway service (Fargate + ALB + autoscaling).
  4. Wait until it's serving, print the new URL, and leave App Runner untouched
     so you can validate and switch Claude Desktop's base URL, then delete it.

Run:  AWS_PROFILE=<profile> uv run python infra/migrate_to_ecs_express.py
Requires Docker running (colima is fine).
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION_DEPLOY", "us-east-1")
ECR_REPO = "claude-bedrock-gateway"
SERVICE_NAME = "claude-bedrock-gateway"
CONTAINER_PORT = 8080
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK_ROLE_NAME = "claude-gateway-task-role"
EXEC_ROLE_NAME = "ecsTaskExecutionRole"
INFRA_ROLE_NAME = "ecsInfrastructureRoleForExpressServices"
# There is no List op for express services, so remember the ARN across runs.
ARN_FILE = os.path.join(REPO_ROOT, "infra", ".ecs_express_service_arn")

# Autoscale on ALB requests-per-target (right signal for an I/O-bound gateway;
# CPU stays low while tasks wait on Bedrock). Target = avg requests per task
# before adding capacity. Min 1 (no scale-to-zero), tune max/target for load.
SCALING = {
    "minTaskCount": 1,
    "maxTaskCount": 10,
    "autoScalingMetric": "REQUEST_COUNT_PER_TARGET",
    "autoScalingTargetValue": 50,
}

sts = boto3.client("sts", region_name=REGION)
iam = boto3.client("iam", region_name=REGION)
ecr = boto3.client("ecr", region_name=REGION)
ecs = boto3.client("ecs", region_name=REGION)
sm = boto3.client("secretsmanager", region_name=REGION)
ACCOUNT = sts.get_caller_identity()["Account"]


def sh(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def secret_arn() -> str:
    return next(
        s["ARN"]
        for s in sm.list_secrets(MaxResults=100)["SecretList"]
        if s["Name"].startswith("GatewayToken")
    )


def ensure_ecr_repo() -> str:
    try:
        ecr.create_repository(repositoryName=ECR_REPO)
        print(f"[ecr] created repo {ECR_REPO}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        print(f"[ecr] repo {ECR_REPO} exists")
    return f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}"


def build_and_push(repo_uri: str) -> str:
    tag = f"{repo_uri}:latest"
    sh(f"aws ecr get-login-password --region {REGION} "
       f"| docker login --username AWS --password-stdin {ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com")
    sh(f"docker build --platform linux/amd64 -t {tag} {REPO_ROOT}")
    sh(f"docker push {tag}")
    return tag


def ensure_role(name: str, trust_service: str, managed: list[str] | None = None,
                inline: dict | None = None) -> str:
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow",
                       "Principal": {"Service": trust_service},
                       "Action": "sts:AssumeRole"}],
    }
    try:
        arn = iam.create_role(RoleName=name,
                              AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        print(f"[iam] created {name}")
    except iam.exceptions.EntityAlreadyExistsException:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        print(f"[iam] {name} exists")
    for policy_arn in managed or []:
        iam.attach_role_policy(RoleName=name, PolicyArn=policy_arn)
    if inline:
        iam.put_role_policy(RoleName=name, PolicyName=f"{name}-inline",
                            PolicyDocument=json.dumps(inline))
    return arn


def ensure_roles() -> tuple[str, str, str]:
    exec_arn = ensure_role(
        EXEC_ROLE_NAME, "ecs-tasks.amazonaws.com",
        managed=["arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"],
    )
    infra_arn = ensure_role(
        INFRA_ROLE_NAME, "ecs.amazonaws.com",
        managed=["arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices"],
    )
    task_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow",
             "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
             "Resource": [
                 f"arn:aws:bedrock:{REGION}:{ACCOUNT}:inference-profile/us.anthropic.claude-*",
                 "arn:aws:bedrock:*::foundation-model/anthropic.claude-*",
             ]},
            {"Effect": "Allow", "Action": "secretsmanager:GetSecretValue",
             "Resource": secret_arn()},
        ],
    }
    task_arn = ensure_role(TASK_ROLE_NAME, "ecs-tasks.amazonaws.com", inline=task_policy)
    return exec_arn, infra_arn, task_arn


def create_service(image: str, exec_arn: str, infra_arn: str, task_arn: str) -> str:
    container = {
        "image": image,
        "containerPort": CONTAINER_PORT,
        "environment": [
            {"name": "AWS_REGION", "value": REGION},
            {"name": "GATEWAY_TOKEN_SECRET_ARN", "value": secret_arn()},
        ],
    }
    # Update in place if we've created it before (ARN remembered locally).
    if os.path.exists(ARN_FILE):
        arn = open(ARN_FILE).read().strip()
        print(f"[ecs] updating existing service {arn}")
        ecs.update_express_gateway_service(
            serviceArn=arn, primaryContainer=container, scalingTarget=SCALING,
        )
        return arn

    print("[ecs] creating express gateway service (retrying while IAM propagates)…")
    params = dict(
        serviceName=SERVICE_NAME,
        executionRoleArn=exec_arn,
        infrastructureRoleArn=infra_arn,
        taskRoleArn=task_arn,
        healthCheckPath="/health",
        primaryContainer=container,
        cpu="256",
        memory="512",
        scalingTarget=SCALING,
    )
    for _ in range(9):
        try:
            arn = ecs.create_express_gateway_service(**params)["service"]["serviceArn"]
            with open(ARN_FILE, "w") as f:
                f.write(arn)
            print(f"[ecs] created {arn}")
            return arn
        except ClientError as e:
            if "assume" in str(e).lower():  # IAM role not yet assumable
                time.sleep(10)
                continue
            raise
    raise SystemExit("create_express_gateway_service kept failing on role assumption")


def _endpoint(svc: dict) -> str | None:
    for cfg in svc.get("activeConfigurations", []):
        for ing in cfg.get("ingressPaths", []):
            if ing.get("endpoint"):
                return ing["endpoint"]
    return None


def wait_and_report(arn: str) -> None:
    print("[ecs] waiting for the service endpoint (provisioning ~3-5 min)…")
    for _ in range(40):
        svc = ecs.describe_express_gateway_service(serviceArn=arn)["service"]
        code = svc.get("status", {}).get("statusCode")
        ep = _endpoint(svc)
        print(f"  statusCode={code} endpoint={ep}")
        if ep:
            url = ep if ep.startswith("http") else f"https://{ep}"
            print(f"\nNEW GATEWAY URL: {url}")
            print("Test it:  GATEWAY_URL=%s AWS_PROFILE=<profile> uv run python test_gateway.py" % url)
            return
        time.sleep(15)
    print("Timed out waiting for endpoint; check the ECS console (service still provisioning).")


def main() -> None:
    repo_uri = ensure_ecr_repo()
    image = build_and_push(repo_uri)
    exec_arn, infra_arn, task_arn = ensure_roles()
    arn = create_service(image, exec_arn, infra_arn, task_arn)
    wait_and_report(arn)
    print("\nApp Runner is left running. Validate the new URL, point Claude Desktop at it,")
    print("then delete App Runner:  cd infra && pnpm cdk destroy")


if __name__ == "__main__":
    main()
