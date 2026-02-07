from typing import Optional, List, Any

from pydantic import BaseModel, Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
    
class OpenAiSettings(BaseModel):
    OPENAI__URL: str = Field(default="http://localhost/v1/")
    OPENAI__API_KEY: str = Field(default="")
    model: str = Field(default="facebook/opt-125m")

class ToolsSettings:
    url: str = Field(default="")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_nested_delimiter="__"
    )

    openai: OpenAiSettings = Field(default_factory=OpenAiSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)