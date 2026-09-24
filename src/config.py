"""Runtime configuration read from environment variables (.env in development)."""
import os

from dotenv import load_dotenv

load_dotenv()

BEDROCK_REGION = os.environ["BEDROCK_REGION"]
BEDROCK_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]