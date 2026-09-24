# AWS Setup

What you need in AWS to run this project, and the practices it assumes.

## Prerequisites
- An AWS account with a non-root IAM identity (IAM Identity Center or an IAM user) protected by MFA.
- AWS CLI v2 (recent version, for `aws login`).
- Access to Anthropic Claude models in Amazon Bedrock (first use may require
  submitting Anthropic's use-case form in the Bedrock console).

## Region and model
- Region: `us-east-1`.
- Model: Claude Sonnet 4.6, called through the US inference profile
  `us.anthropic.claude-sonnet-4-6`. The `us.` profile keeps requests within US regions.

## Credentials
- Use short-lived credentials; do not create long-term access keys.
- Log in: `aws login --profile <your-profile>`
- Verify: `aws sts get-caller-identity --profile <your-profile>`

## Permissions
The identity running the agent needs permission to invoke the model through the
inference profile. Invoking through an inference profile requires permission on both
the profile and the underlying foundation model.

## Cost control
- Create an AWS Budgets monthly cost budget with email alerts before running simulations.
- Budgets alert but do not stop spending.

## Verify Bedrock access
```bash
aws bedrock-runtime converse \
  --model-id us.anthropic.claude-sonnet-4-6 \
  --messages '[{"role":"user","content":[{"text":"Reply with OK"}]}]' \
  --region us-east-1 \
  --profile <your-profile>
```