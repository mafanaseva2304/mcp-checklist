from typing import Dict, Optional

from pydantic import BaseModel, Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAiSettings(BaseModel):
    api_key: str = Field(default="EMPTY")
    api_base: str = Field()
    model_name: str = Field()
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[PositiveInt] = Field(default=2048)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    request_timeout: PositiveInt = Field(default=60)
    max_retries: int = Field(default=2, ge=0)
    streaming: bool = Field(default=False)


class AgentSettings(BaseModel):
    max_iterations: PositiveInt = Field(default=15)
    max_execution_time: Optional[PositiveInt] = Field(default=120)
    early_stopping_method: str = Field(default="generate")
    handle_parsing_errors: bool = Field(default=True)
    verbose: bool = Field(default=False)
    return_intermediate_steps: bool = Field(default=False)

    @field_validator("early_stopping_method")
    @classmethod
    def validate_stopping(cls, v: str) -> str:
        if v not in ("force", "generate"):
            raise ValueError(f"Должно быть 'force' или 'generate', получено '{v}'")
        return v


class McpServerSettings(BaseModel):
    url: str
    transport: str = Field(default="http")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"url должен быть http(s)://, получено '{v}'")
        return v


class McpSelfSettings(BaseModel):
    enabled: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")
    port: PositiveInt = Field(default=8001)
    name: str = Field(default="No_name")
    description: str = Field(default="No_name")
    tool_name: str = Field(default="No_name")
    tool_description: str = Field(default="No name")
    transport: str = Field(default="http")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    openai: OpenAiSettings
    agent: AgentSettings = Field(default_factory=AgentSettings)
    system_prompt: str = Field(default="Ты - агент, выполняй инструкции.")
    mcp_servers: Dict[str, McpServerSettings]
    mcp_self: McpSelfSettings = Field(default_factory=McpSelfSettings)

    @field_validator("system_prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("system_prompt не может быть пустым")
        return v.strip()

    @field_validator("mcp_servers")
    @classmethod
    def validate_servers(cls, v: dict) -> dict:
        if not v:
            raise ValueError("Нужен хотя бы один MCP-сервер")
        return v
