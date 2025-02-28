from typing import Any, Dict

from pydantic import BaseModel


class InputData(BaseModel):
   input_data: Dict[str, Any] = None
