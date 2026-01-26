
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
            # print(lk)
            if lk in {"confidence", "boundingregions", "bounding_regions", "spans", "polygon","span", "offset", "length", 
                      "row_index","column_index","row_span","column_span","page_number", "angle", "width", "height", "unit", "height", "length"}:
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

    if vt in ("string", "phoneNumber", "selectionMark", "countryRegion", "time"):
        return getattr(field, "value", None)
    if vt in ("int64", "integer"):
        return int(getattr(field, "value", None)) if field.value is not None else None
    if vt == "number" or vt == "float":
        return float(getattr(field, "value", None)) if field.value is not None else None
    if vt == "date":
        return str(getattr(field, "value", None))  
    if vt == "boolean":
        return bool(getattr(field, "value", None)) if field.value is not None else None

    if vt == "currency":
        val = getattr(field, "value", None)
        if val:
            amt = getattr(val, "amount", None)
            code = getattr(val, "code", None)
            sym = getattr(val, "symbol", None)
            if amt is None:
                return None
            return f"{code or sym or ''} {amt}".strip()
        return None

    if vt == "address":
        val = getattr(field, "value", None)
        if not val:
            return None

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
        return text or getattr(field, "content", None)


    if vt == "dictionary":
        out = {}
        value_dict = getattr(field, "value", {}) or {}
        for k, v in value_dict.items():
            out[k] = _field_value_to_python(v)
        return out

    if vt == "list":
        value_list = getattr(field, "value", []) or []
        return [_field_value_to_python(v) for v in value_list]

    return getattr(field, "value", None) or getattr(field, "content", None)




###### NEW - Azure Document Intelligence result fetching

def fetch_results(results, doc_type: str) -> Dict[str, Any]:
    if not results:
        return {
            "status": "error",
            "message": "No results provided."
        }
    
    out = {}
    
    fields = {
        "prebuilt-invoice": {
            "fields": ["VendorName", "VendorAddress", "VendorAddressRecipient", "CustomerName", "CustomerId", "CustomerAddress", "CustomerAddressRecipient", "InvoiceId", "InvoiceDate", "InvoiceTotal", "DueDate", "PurchaseOrder", "BillingAddress", "BillingAddressRecipient", "ShippingAddress", "ShippingAddressRecipient","SubTotal", "TotalTax", "PreviousUnpaidBalance", "AmountDue", "ServiceStartDate", "ServiceEndDate", "ServiceAddress", "ServiceAddressRecipient", "RemittanceAddress", "RemittanceAddressRecipient", ],
            "items": ["Description", "Quantity", "Unit", "UnitPrice", "ProductCode", "Date", "Tax", "Amount", ]
            },
        "prebuilt-receipt":{
            "fields": ["MerchantName", "TransactionDate", "Subtotal", "TotalTax", "Tip", "Total"],
            "items": ["Description", "Quantity", "Price", "TotalPrice"]
            },
        "prebuilt-idDocument": {
            "fields": ["FirstName", "LastName", "DateOfBirth", "DocumentNumber", "DateOfExpiration", "Address", "Sex" , "CountryRegion", "Region"],
            "items" : []
            }
    }

    for idx, document in enumerate(results.documents):
        for field in fields[doc_type]["fields"]:
            if not document.fields.get(field) or document.fields.get(field).content.lower() in {"n/a", "na", "none", "null", ""}:
                        continue
            out[field] = document.fields.get(field, None).content.replace("\n", "") if document.fields.get(field) else None
    if document.fields.get("Items"):
        # print("Processing line items...", document.fields.get("Items"))
        for idx, itemval in enumerate(document.fields.get("Items").value):
            # print("Item value:", itemval.value)
            for keyys in itemval.value.keys():
                if keyys in fields[doc_type]["items"]:
                    # print("Processing item:", itemval.value)
                    if not itemval.value.get(keyys) or itemval.value.get(keyys).content.lower() in {"n/a", "na", "none", "null", ""}:
                        continue
                    out[f"Item_{idx+1}_{keyys}"] = itemval.value.get(keyys).content.replace("\n", "")
        
    return out
        
def format_output(output):
    """
    Formats agent output for UI rendering.
    - Dict outputs → vertical key:value lines
    - String outputs → returned as-is
    """
    if isinstance(output, dict):
        return "\n".join(f"{k}: {v}" for k, v in output.items())
    return output


import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
import io


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    filename = filename.lower()

    # PDF
    if filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    # Word
    if filename.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    # Image (OCR)
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img)
