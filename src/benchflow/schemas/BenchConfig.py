from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field, field_validator
import yaml

class BenchConfig(BaseModel):
    # List of required parameter names (no defaults provided)
    required: List[str] = Field(default_factory=list)
    # Optional parameters can be defined either as a list of single-key dicts or as a dict directly
    optional: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(default_factory=dict)

    @field_validator('optional', mode='before')
    def merge_optional(cls, v):
        """
        If 'optional' is provided as a list, merge each single-key dictionary into one dictionary.
        For example:
            [ { "param3": 1 }, { "param4": "string" } ]
        will be merged into:
            { "param3": 1, "param4": "string" }
        """
        if isinstance(v, list):
            merged = {}
            for item in v:
                if isinstance(item, dict):
                    merged.update(item)
                else:
                    raise ValueError("Each item in 'optional' must be a dictionary")
            return merged
        elif isinstance(v, dict):
            return v
        else:
            raise ValueError("'optional' must be a dictionary or a list of dictionaries")

    def get_params(self, runtime: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Merge runtime parameters with configuration defaults and return the final parameters.
        
        For each parameter in 'required', a value must be provided in the runtime dictionary;
        otherwise, a ValueError is raised.
        For each parameter in 'optional', if a runtime value is provided, it is used;
        otherwise, the default value from the configuration is used.
        """
        runtime = runtime or {}
        params = {}
        
        # Validate and fetch required parameters from runtime
        for key in self.required:
            if key in runtime and runtime[key] is not None:
                params[key] = runtime[key]
            else:
                raise ValueError(f"Missing required parameter: {key}")
        
        # For optional parameters, use runtime value if provided; otherwise, use the default value
        for key, default in self.optional.items():
            params[key] = runtime.get(key, default)
        return params

    def __init__(self, config_source: Union[str, Dict[str, Any]], **kwargs):
        """
        The constructor accepts either a YAML file path or a dictionary as the configuration source.
        Additional keyword arguments can override or add to the configuration.
        """
        if isinstance(config_source, str):
            with open(config_source, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            data = config_source
        else:
            raise ValueError("config_source must be a YAML file path or a dictionary")
        
        # Merge any additional keyword arguments into the configuration data
        data.update(kwargs)
        super().__init__(**data)