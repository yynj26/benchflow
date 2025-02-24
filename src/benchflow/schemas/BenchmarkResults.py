from typing import Dict, Any, Union
from pydantic import BaseModel

MetricValue = Union[bool, int, float, str]

class BenchmarkResult(BaseModel):
    is_resolved: bool
    log: str
    metrics: Dict[str, MetricValue]
    other: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "is_resolved": True,
                "log": "finish test",
                "metrics": {
                    "metric1": True,
                    "metric2": 123,
                    "metric3": 3.1415,
                    "metric4": "OK"
                },
                "other": {
                    "extra_info": "extra info"
                }
            }
        }