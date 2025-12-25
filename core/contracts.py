from pydantic import BaseModel, Field


class ToyAskRequest(BaseModel):
    mode: str = Field(description="ask|tldr")
    text: str


class ToyAskResponse(BaseModel):
    mode: str
    response: str
