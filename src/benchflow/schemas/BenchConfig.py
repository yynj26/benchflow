from typing import Any, Dict, List, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class BenchConfig(BaseModel):
    # List of required parameter names (no defaults provided)
    required: List[str] = Field(default_factory=list)
    # Optional parameters can be used to define the parameters that are not required and the default values
    optional: List[Dict[str, Any]] = Field(default_factory=list)

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
        else:
            raise ValueError("'optional' must be a dictionary or a list of dictionaries")

    def get_params(self, runtime_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Merge runtime parameters with configuration defaults and return the final parameters.
        
        For each parameter in 'required', a value must be provided in the runtime dictionary;
        otherwise, a ValueError is raised.
        For each parameter in 'optional', if a runtime value is provided, it is used;
        otherwise, the default value from the configuration is used.
        """
        runtime_params = runtime_params or {}
        params = {}
        
        # Validate and fetch required parameters from runtime
        for key in self.required:
            if key in runtime_params and runtime_params[key] is not None:
                params[key] = runtime_params[key]
            else:
                raise ValueError(f"Missing required parameter: {key}")
        
        # For optional parameters, use runtime value if provided; otherwise, use the default value
        for key, default in self.optional.items():
            params[key] = runtime_params.get(key, default)
        return params

    def __init__(self, config_source: Union[str, Dict[str, Any], None], **kwargs):
        """
        The constructor accepts either a YAML file path or a dictionary as the configuration source.
        Additional keyword arguments can override or add to the configuration.
        """
        if isinstance(config_source, str):
            with open(config_source, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            data = config_source
        elif config_source is None:
            data = {
                "required": [],
                "optional": {}
            }
        else:
            raise ValueError("config_source must be a YAML file path or a dictionary")
        
        data.update(kwargs)
        super().__init__(**data)