
import os
import json
from copy import deepcopy
from typing import Any, Dict, List, Union

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

def _strip_noise(obj: Json) -> Json:
    """
    Recursively remove keys we don't want to show in 'raw_slim':
    - confidence
    - boundingRegions / bounding_regions
    - spans
    - polygon (if present under regions)
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            lk = k.lower()
            if lk in {"confidence", "boundingregions", "bounding_regions", "spans", "polygon"}:
                continue
            cleaned[k] = _strip_noise(v)
        return cleaned
    elif isinstance(obj, list):
        return [_strip_noise(x) for x in obj]
    else:
        return obj

def _field_value_to_python(field) -> Any:
    """
    Convert a DocumentField (v3.x) to a plain Python value.
    Handles nested types (dictionary/list/currency/address/date/selection/...). 
    """
    if field is None:
        return None

    vt = getattr(field, "value_type", None)

    # Base scalars
    if vt in ("string", "phoneNumber", "selectionMark", "countryRegion", "time"):
        return getattr(field, "value", None)
    if vt in ("int64", "integer"):
        return int(getattr(field, "value", None)) if field.value is not None else None
    if vt == "number" or vt == "float":
        return float(getattr(field, "value", None)) if field.value is not None else None
    if vt == "date":
        return str(getattr(field, "value", None))  # iso-date
    if vt == "boolean":
        return bool(getattr(field, "value", None)) if field.value is not None else None

    # Currency
    if vt == "currency":
        val = getattr(field, "value", None)
        if val:
            # value is a CurrencyValue(amount, symbol, code)
            amt = getattr(val, "amount", None)
            code = getattr(val, "code", None)
            sym = getattr(val, "symbol", None)
            if amt is None:
                return None
            return f"{code or sym or ''} {amt}".strip()
        return None

    # Address
    if vt == "address":
        val = getattr(field, "value", None)
        if not val:
            return None
        # Render a clean single-line address if possible
        parts = [
            getattr(val, "unit", None),
            getattr(val, "house_number", None),
            getattr(val, "house", None),
            getattr(val, "road", None),
            getattr(val, "street_address", None),
            getattr(val, "city_district", None),
            getattr(val, "city", None),
            getattr(val, "state_district", None),
            getattr(val, "state", None),
            getattr(val, "postal_code", None),
            getattr(val, "country_region", None),
        ]
        text = ", ".join([str(p) for p in parts if p])
        # fall back to field.content if structured parts are messy
        return text or getattr(field, "content", None)

    # Dictionary / List
    if vt == "dictionary":
        out = {}
        value_dict = getattr(field, "value", {}) or {}
        for k, v in value_dict.items():
            out[k] = _field_value_to_python(v)
        return out

    if vt == "list":
        value_list = getattr(field, "value", []) or []
        return [_field_value_to_python(v) for v in value_list]

    # Fallback: try direct value or content
    return getattr(field, "value", None) or getattr(field, "content", None)
