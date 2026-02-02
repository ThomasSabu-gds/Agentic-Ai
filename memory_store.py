import uuid
import time
import threading
from typing import Optional
from flask import Request, Response


class Memory:
    def __init__(self):
        self._store = {}
        self.COOKIE_NAME = "session_id"
        self._lock = threading.Lock()

    def set_doc(self, session_id: str, document_blob: str, masked_blob:str=None, sticky: bool = True, ttl_seconds: int = None):
        with self._lock:
            existing = self._store.get(session_id, {})
            meta = existing.get("meta", {})
            self._store[session_id] = {
                "document_blob": document_blob,
                "masked_blob": masked_blob or document_blob,
                "sticky": sticky,
                "created_at": time.time(),
                "ttl": ttl_seconds,
                "meta": meta, 
            }

    def has_doc(self, session_id: str) -> bool:
        with self._lock:
            data = self._store.get(session_id)
            if not data:
                return False
            if data.get("ttl"):
                if time.time() - data["created_at"] > data["ttl"]:
                    self._store.pop(session_id, None)
                    return False
            return bool(data.get("document_blob"))

    def get_doc(self, session_id: str, masked=False,  consume_if_single_use: bool = True) -> Optional[str]:
        with self._lock:
            data = self._store.get(session_id)
            if not data:
                return None
            # blob = data.get("document_blob")
            blob = data.get("masked_blob") if masked else data.get("document_blob")
            if consume_if_single_use and not data.get("sticky", True):
                self._store.pop(session_id, None)
            return blob

    def clear(self, session_id: str):
        with self._lock:
            self._store.pop(session_id, None)

    def set_meta(self, session_id: str, meta: dict):
        with self._lock:
            data = self._store.get(session_id)
            if not data:
                self._store[session_id] = {
                    "document_blob": None,
                    "sticky": True,
                    "created_at": time.time(),
                    "ttl": None,
                    "meta": meta or {}
                }
            else:
                data["meta"] = meta or {}

    def get_meta(self, session_id: str) -> dict:
        with self._lock:
            data = self._store.get(session_id)
            if not data:
                return {}
            return data.get("meta", {}) or {}

    def get_or_create_session_id(self, request: Request) -> str:
        sid = request.headers.get("X-Session-ID") or request.cookies.get(self.COOKIE_NAME)
        if sid:
            return sid
        return str(uuid.uuid4())

    def attach_session_cookie(self, response: Response, session_id: str):
        response.set_cookie(
            key=self.COOKIE_NAME,
            value=session_id,
            httponly=True,
            samesite="Lax",
            secure=False
        )
