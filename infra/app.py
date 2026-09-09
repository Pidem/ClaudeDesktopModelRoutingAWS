#!/usr/bin/env python3
"""CDK app: deploys the Claude-on-Bedrock gateway to AWS App Runner."""
import aws_cdk as cdk

from gateway_stack import GatewayStack

app = cdk.App()
GatewayStack(
    app,
    "ClaudeBedrockGateway",
    env=cdk.Environment(region="us-east-1"),
)
app.synth()
