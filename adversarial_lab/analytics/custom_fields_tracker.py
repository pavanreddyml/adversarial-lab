from typing import Dict, Any, Optional
from . import Tracker

import warnings


class CustomFieldsTracker(Tracker):
    _columns = {}
    _allowed_types = ["int", "str", "float", "bool", "json", "blob"]

    def __init__(self,
                 field_types: Optional[Dict[str, str]] = None,
                 field_values: Optional[Dict[str, Any]] = None,
                 ) -> None:
        super().__init__()
        self.warned = False
        self.field_types = field_types or {}
        
        for key, value in self.field_types.items():
            if value not in self._allowed_types:
                raise ValueError(f"Unsupported type '{value}' for field '{key}'. Allowed types are: {self._allowed_types}")
            
        self._columns.update(self.field_types)
        self.set_values(field_values)

    def pre_attack(self, *args, **kwargs):
        self.data = self.field_values.copy()

    def post_epoch(self,
                   *args,
                   **kwargs
                   ) -> None:
        self.data = self.field_values.copy()

    def serialize(self) -> Dict:
        data = self.data.copy()
        return data

    def reset_values(self) -> None:
        self.data = {}

    def set_values(self, values: Dict[str, Any]) -> None:
        values = values or {}

        self.data = {key: None for key in self.field_types.keys()}

        for key, value in values.items():
            if key not in self.field_types:
                if not self.warned:
                    warnings.warn(f"Field '{key}' is not defined in field_types. It will be ignored.")
                    self.warned = True
                continue
            
            expected_type = self.field_types[key]
            if expected_type == "int" and not isinstance(value, int):
                warnings.warn(f"Expected int for field '{key}', but got {type(value).__name__}. Setting NULL.")
                self.data[key] = None
            elif expected_type == "str" and not isinstance(value, str):
                warnings.warn(f"Expected str for field '{key}', but got {type(value).__name__}. Setting NULL.")
                self.data[key] = None
            elif expected_type == "float" and not isinstance(value, float):
                warnings.warn(f"Expected float for field '{key}', but got {type(value).__name__}. Setting NULL.")
                self.data[key] = None
            elif expected_type == "bool" and not isinstance(value, bool):
                warnings.warn(f"Expected bool for field '{key}', but got {type(value).__name__}. Setting NULL.")
                self.data[key] = None
            elif expected_type == "json" and not isinstance(value, (dict, list)):
                warnings.warn(f"Expected json for field '{key}', but got {type(value).__name__}. Setting NULL.")
                self.data[key] = None
            elif expected_type == "blob" and not isinstance(value, bytes):
                warnings.warn(f"Expected blob for field '{key}', but got {type(value).__name__}. Setting NULL.")
                self.data[key] = None

        self.field_values = self.data.copy()