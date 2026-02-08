from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, PositiveInt, computed_field, field_validator
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgreSettings(BaseModel):
    dbname: str = Field(default="pg")
    host: str = Field(default="localhost")
    user: str = Field(default="admin")
    password: str = Field(default="admin")
    port: PositiveInt = Field(default=5432)

    @property
    def dsn(self) -> str:
        return (
            f"dbname={self.dbname} "
            f"user={self.user} "
            f"password={self.password} "
            f"host={self.host} "
            f"port={self.port}"
        )


class QuerySettings(BaseModel):
    status_type_id: PositiveInt = Field(default=5)
    model_type_id: PositiveInt = Field(default=6)
    non_tested_model_types: List[PositiveInt] = Field(default=[8, 9, 12])
    filter_layers: bool = Field(default=True)
    linear_type_model: PositiveInt = Field(default=0)

    @computed_field
    @property
    def layer1_filter(self) -> str:
        return "%_L1_%" if self.filter_layers else ""

    @computed_field
    @property
    def layer2_filter(self) -> str:
        return "%_L2_%" if self.filter_layers else ""

    @computed_field
    @property
    def layer3_filter(self) -> str:
        return "%_L3_%" if self.filter_layers else ""


class ServiceSettings(BaseModel):
    limit: PositiveInt = Field(default=10)
    non_continuous_types: List[PositiveInt] = [2, 3, 16, 27]
    transport: str = Field(default="http")
    host: str = Field(default="0.0.0.0")
    port: PositiveInt = Field(default=8000)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_nested_delimiter="__"
    )

    pg: PostgreSettings = Field(default_factory=PostgreSettings)
    qr: QuerySettings = Field(default_factory=QuerySettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
