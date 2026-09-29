"""Runtime configuration read from the environment (.env in development), with
non-secret defaults so importing this module never requires a .env file."""
import os

from dotenv import load_dotenv

load_dotenv()

BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")