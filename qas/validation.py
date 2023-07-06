from typing import List, Optional

from pydantic import BaseConfig, BaseModel, Extra

BaseConfig.arbitrary_types_allowed = True


class QueryRequest(BaseModel):
    query: str
    params: Optional[dict] = None

    def get_query(self):
        return self.query

    class Config:
        extra = Extra.forbid

