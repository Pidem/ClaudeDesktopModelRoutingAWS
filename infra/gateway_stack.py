"""App Runner service running the Claude-on-Bedrock gateway.

Builds the Dockerfile at the repo root into an image asset, runs it on App Runner
with an instance role scoped to Bedrock invoke on the Claude models we route to,
and injects a gateway bearer token stored in Secrets Manager.
"""
import os

import aws_cdk as cdk
from aws_cdk import aws_apprunner_alpha as apprunner
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

# Repo root (one level up from infra/) — where the Dockerfile lives.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GatewayStack(cdk.Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)

        # --- Gateway bearer token (Desktop -> Gateway) ------------------------
        token_secret = secretsmanager.Secret(
            self,
            "GatewayToken",
            description="Bearer token Claude Desktop presents to the gateway",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_punctuation=True,
                password_length=48,
            ),
        )

        # --- Instance role (Gateway -> Bedrock) -------------------------------
        instance_role = iam.Role(
            self,
            "InstanceRole",
            assumed_by=iam.ServicePrincipal("tasks.apprunner.amazonaws.com"),
        )
        instance_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    # The cross-region inference profiles we route to.
                    f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/us.anthropic.claude-*",
                    # The underlying foundation models the profiles fan out to
                    # (FM ARNs carry no account id; profiles span multiple regions).
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-*",
                ],
            )
        )
        token_secret.grant_read(instance_role)

        # --- Container image --------------------------------------------------
        image = ecr_assets.DockerImageAsset(
            self,
            "GatewayImage",
            directory=REPO_ROOT,
            platform=ecr_assets.Platform.LINUX_AMD64,
        )

        # --- App Runner service ----------------------------------------------
        service = apprunner.Service(
            self,
            "Service",
            source=apprunner.Source.from_asset(
                asset=image,
                image_configuration=apprunner.ImageConfiguration(
                    port=8080,
                    environment_variables={
                        "AWS_REGION": self.region,
                        "GATEWAY_TOKEN_SECRET_ARN": token_secret.secret_arn,
                    },
                ),
            ),
            instance_role=instance_role,
            cpu=apprunner.Cpu.QUARTER_VCPU,
            memory=apprunner.Memory.HALF_GB,
            health_check=apprunner.HealthCheck.http(path="/health"),
            auto_deployments_enabled=False,
        )

        cdk.CfnOutput(self, "ServiceUrl", value=f"https://{service.service_url}")
        cdk.CfnOutput(self, "GatewayTokenSecretArn", value=token_secret.secret_arn)
