from typing import Any, Dict, Union

from pydantic import BaseModel

MetricValue = Union[bool, int, float, str]

class BenchmarkResult(BaseModel):
    is_resolved: bool
    log: Dict[str, str]
    metrics: Dict[str, MetricValue]
    other: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "is_resolved": True,
                "log": {
                    "trace": "trace message",
                },
                "metrics": {
                    "metric1": True,
                    "metric2": 123,
                    "metric3": 3.1415,
                    "metric4": "OK"
                },
                "other": {
                    "extra_info": "extra info",
                    "error": "error message"
                }
            }
        }