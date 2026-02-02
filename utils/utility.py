import os
import math
import logging
import hashlib
import mimetypes
import numpy as np
from PIL import Image
from io import BytesIO
from docx import Document
from functools import lru_cache
from pdf2image import convert_from_bytes
from typing import Any, Dict, List, Union
from logging.handlers import RotatingFileHandler
from pdfminer.high_level import extract_text as pdf_text

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

EASYOCR_LANG_MAP = {
    "ch": "ch_sim",
    "jp": "ja",
    "ko": "ko",
    "arabic": "ar",
    "ar": "ar",
    "fa": "fa",
    "ur": "ur",
    "ta": "ta",
    "te": "te",
    "ml": "ml",
}

DOC_TYPE_FIELDS = {
    "prebuilt-invoice": {
        "fields": ["VendorName", "VendorAddress", "VendorAddressRecipient", "CustomerName", "CustomerId",
                   "CustomerAddress", "CustomerAddressRecipient", "InvoiceId", "InvoiceDate", "InvoiceTotal", "DueDate",
                   "PurchaseOrder", "BillingAddress", "BillingAddressRecipient", "ShippingAddress",
                   "ShippingAddressRecipient", "SubTotal", "TotalTax", "PreviousUnpaidBalance", "AmountDue",
                   "ServiceStartDate", "ServiceEndDate", "ServiceAddress", "ServiceAddressRecipient",
                   "RemittanceAddress", "RemittanceAddressRecipient", ],
        "items": ["Description", "Quantity", "Unit", "UnitPrice", "ProductCode", "Date", "Tax", "Amount", ],
        "mask": ["VendorName", "VendorAddress", "CustomerName", "CustomerAddress", "VendorAddressRecipient",
                 "CustomerAddressRecipient"]
    },
    "prebuilt-receipt": {
        "fields": ["MerchantName", "TransactionDate", "Subtotal", "TotalTax", "Tip", "Total"],
        "items": ["Description", "Quantity", "Price", "TotalPrice"],
        "mask": ["MerchantName"]
    },
    "prebuilt-idDocument": {
        "fields": ["FirstName", "LastName", "DateOfBirth", "DocumentNumber", "DateOfExpiration", "Address", "Sex",
                   "CountryRegion", "Region"],
        "items": [],
        "mask": []
    },
    "prebuilt-document": {
        "fields": [],
        "items": [],
        "mask": []
    },
}


#############logger
def setup_logger(
        name: str = "app",
        log_file: str = "app.log",
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,  # 5 MB
        backup_count: int = 3
) -> logging.Logger:
    """
    Creates or returns a named logger with:
    - Console output
    - Rotating log file
    - INFO-level default
    - Duplicate handler protection
    """
    logger = logging.getLogger(name)

    # If the logger already has handlers, just update level and return it
    # (prevents duplicates when importing across modules)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)

    # Ensure logs dir exists
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)

    # Formatter: timestamp, level, logger name, message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Console handler ----
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    # ---- Rotating file handler ----
    fh = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setFormatter(formatter)
    fh.setLevel(level)

    # Attach handlers
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


logger = setup_logger(name="Agentic-AI", log_file="agentic-ai.log")


#########noise from azure doc analysis
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
            if lk in {"confidence", "boundingregions", "bounding_regions", "spans", "polygon", "span", "offset",
                      "length",
                      "row_index", "column_index", "row_span", "column_span", "page_number", "angle", "width", "height",
                      "unit", "height", "length"}:
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

    if doc_type not in DOC_TYPE_FIELDS.keys():
        return {
            "status": "error",
            "message": "Unsupported document type for result fetching."
        }
    elif doc_type == "prebuilt-document":

        for kv_pair in results.key_value_pairs:
            key = kv_pair.key.content.strip() if kv_pair.key and kv_pair.key.content else None
            value = kv_pair.value.content.strip() if kv_pair.value and kv_pair.value.content else None

            if key:
                out[key] = value

        return out

    for idx, document in enumerate(results.documents):
        for field in DOC_TYPE_FIELDS[doc_type]["fields"]:
            if not document.fields.get(field) or document.fields.get(field).content.lower() in {"n/a", "na", "none",
                                                                                                "null", ""}:
                continue
            out[field] = document.fields.get(field, None).content.replace("\n", "") if document.fields.get(
                field) else None
    if document.fields.get("Items"):
        # print("Processing line items...", document.fields.get("Items"))
        for idx, itemval in enumerate(document.fields.get("Items").value):
            # print("Item value:", itemval.value)
            for keyys in itemval.value.keys():
                if keyys in DOC_TYPE_FIELDS[doc_type]["items"]:
                    # print("Processing item:", itemval.value)
                    if not itemval.value.get(keyys) or itemval.value.get(keyys).content.lower() in {"n/a", "na", "none",
                                                                                                    "null", ""}:
                        continue
                    out[f"Item_{idx + 1}_{keyys}"] = itemval.value.get(keyys).content.replace("\n", "")

    return out


#######ui support
def format_output(output):
    """
    Formats agent output for UI rendering.
    - Dict outputs → vertical key:value lines
    - String outputs → returned as-is
    """
    if isinstance(output, dict):
        return "\n".join(f"{k}: {v}" for k, v in output.items())
    return output


##################ocr helpers
try:
    import easyocr

    EASYOCR_AVAILABLE = True
    logger.info("Easyocr available")
except Exception:
    EASYOCR_AVAILABLE = False
    logger.info("Easyocr unavailable")

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
    logger.info("paddleocr available")

except Exception:
    PADDLE_AVAILABLE = False
    logger.info("paddleocr unavailable")


def to_easyocr_lang(lg: str) -> str:
    return EASYOCR_LANG_MAP.get(lg, lg)


@lru_cache(maxsize=16)
def get_paddle_reader(lang_code: str):
    if not PADDLE_AVAILABLE:
        return None
    try:
        return PaddleOCR(
            use_angle_cls=True,
            lang=lang_code,
            show_log=False,
            # use_gpu=True,
        )
    except Exception:
        return None


@lru_cache(maxsize=16)
def get_easyocr_reader(lang_code: str):
    if not EASYOCR_AVAILABLE:
        # print("not aval")
        return None
    try:
        return easyocr.Reader(
            [to_easyocr_lang(lang_code)],
            gpu=False,
            verbose=False
        )

    except Exception:
        # print("exceppppppt")
        return None


def _ocr_with_paddle(arr: np.ndarray, lang_code: str):
    reader = get_paddle_reader(lang_code)
    if reader is None:
        return [], []
    try:
        result = reader.ocr(arr, cls=True)
    except Exception:
        return [], []
    # format: [ [ [bbox], (text, conf) ], ... ]
    lines, confs = [], []
    if result and len(result) > 0:
        for ln in result[0]:
            txt = ln[1][0] if ln[1] and len(ln[1]) > 0 else ""
            cf = ln[1][1] if ln[1] and len(ln[1]) > 1 else 0.0
            if txt:
                lines.append(txt)
                confs.append(cf)
    return lines, confs


def _ocr_with_easyocr(arr: np.ndarray, lang_code: str):
    """Run EasyOCR and return (lines, confs)."""
    reader = get_easyocr_reader(lang_code)
    if reader is None:
        return [], []
    try:
        # result: [ (bbox, text, confidence), ... ]
        result = reader.readtext(arr)
    except Exception:
        return [], []
    lines, confs = [], []

    for item in result:
        if len(item) >= 3:
            txt = item[1]
            cf = float(item[2]) if item[2] is not None else 0.0
            if txt:
                lines.append(txt)
                confs.append(cf)
    return lines, confs


def _score_text(lines, confs):
    """Compute a composite score: avg_conf * log(length+1)."""
    text = " ".join(lines).strip()
    if not text:
        return "", -1.0
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    score = avg_conf * math.log(len(text) + 1)
    return text, score


def _ocr_best_text_from_image(img: Image.Image, langs) -> str:
    """
    Try Paddle first per language; if it fails or is weak, fallback to EasyOCR.
    Choose the best scored result across languages.
    """
    arr = np.array(img.convert("RGB"))
    best_text = ""
    best_score = -1.0

    # threshold below which we consider Paddles result weak and try EasyOCR
    WEAK_CONF_THRESHOLD = 0.40

    for lg in langs:
        # Paddle
        pl_lines, pl_confs = _ocr_with_paddle(arr, lg)
        pl_text, pl_score = _score_text(pl_lines, pl_confs)
        use_easyocr = False

        if not pl_text:
            use_easyocr = True
        else:
            avg_conf = (sum(pl_confs) / len(pl_confs)) if pl_confs else 0.0
            if avg_conf < WEAK_CONF_THRESHOLD:
                use_easyocr = True

        # Fallback
        if use_easyocr:
            ez_lines, ez_confs = _ocr_with_easyocr(arr, lg)
            ez_text, ez_score = _score_text(ez_lines, ez_confs)
            # pick the better one between Paddle and Easy
            if ez_score > pl_score:
                cand_text, cand_score = ez_text, ez_score
            else:
                cand_text, cand_score = pl_text, pl_score
        else:
            cand_text, cand_score = pl_text, pl_score

        # Track overall best across languages
        if cand_text and cand_score > best_score:
            best_text = cand_text
            best_score = cand_score

    return best_text


def extract_text_from_file(file_bytes: bytes, filename: str, langs) -> str:
    ext = os.path.splitext(filename)[1].lower()
    mime = mimetypes.guess_type(filename)[0] or ""

    if ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")

    if ext == ".docx":
        doc = Document(BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])

    if ext == ".pdf":
        # digital pdf
        try:
            text = pdf_text(BytesIO(file_bytes))
            if text and text.strip():
                return text
        except Exception:
            pass

        # scanned pdf → OCR per page
        pages = convert_from_bytes(file_bytes)
        ocr_text = []
        for page in pages:
            page_text = _ocr_best_text_from_image(page, langs=langs)
            ocr_text.append(page_text)
        return "\n".join(ocr_text).strip()

    if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
        img = Image.open(BytesIO(file_bytes))
        return _ocr_best_text_from_image(img, langs=langs)

    return ""


################doc type determine helper
def classify_text(text: str) -> str:
    t = text.lower().strip()

    invoice_terms = [
        "invoice", "tax invoice", "invoice no", "bill to", "gstin", "vat",
        "unit price", "qty", "po number", "subtotal"
    ]

    receipt_terms = [
        "receipt", "sales receipt", "pos", "terminal id", "tendered",
        "change", "thank you", "auth code", "cash", "card"
    ]

    inv_hits = sum(k in t for k in invoice_terms)
    rec_hits = sum(k in t for k in receipt_terms)

    if inv_hits > rec_hits and inv_hits > 0:
        return "INVOICE"
    if rec_hits > inv_hits and rec_hits > 0:
        return "RECEIPT"
    if len(t) > 20:
        return "GENERAL_DOCUMENT"

    return "GENERAL_DOCUMENT"


###################memmory helper
def _bytes_hash(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _file_fingerprint(f: dict):
    b = f.get("bytes")
    if not b:
        return None
    return {
        "filename": f.get("filename", "uploaded.bin"),
        "size": len(b),
        "sha256": _bytes_hash(b),
    }


