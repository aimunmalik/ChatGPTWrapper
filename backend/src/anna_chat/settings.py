import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    aws_region: str
    cognito_user_pool_id: str
    cognito_spa_client_id: str
    conversations_table: str
    messages_table: str
    bedrock_model_id: str
    message_ttl_days: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            aws_region=os.environ["AWS_REGION"],
            cognito_user_pool_id=os.environ["COGNITO_USER_POOL_ID"],
            cognito_spa_client_id=os.environ["COGNITO_SPA_CLIENT_ID"],
            conversations_table=os.environ["CONVERSATIONS_TABLE"],
            messages_table=os.environ["MESSAGES_TABLE"],
            bedrock_model_id=os.environ.get(
                "BEDROCK_MODEL_ID",
                "us.anthropic.claude-sonnet-4-6",
            ),
            message_ttl_days=int(os.environ.get("MESSAGE_TTL_DAYS", "90")),
        )
