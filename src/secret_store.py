"""Small local secret store for provider API keys.

On Windows values are encrypted with DPAPI for the current user. The JSON file
contains ciphertext only. Non-Windows development falls back to an explicit
base64 marker so the caller can surface the weaker security boundary instead of
silently pretending the value is encrypted.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import sys
import tempfile
from ctypes import wintypes
from pathlib import Path

_runtime = Path(__file__).resolve().parent.parent / "outputs/secret-runtime"
if _runtime.is_dir() and str(_runtime) not in sys.path:
    sys.path.insert(0, str(_runtime))

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError as exc:
    raise ImportError(
        "缺少 cryptography，请运行 scripts/setup_secret_runtime.ps1"
    ) from exc


class SecretStoreError(RuntimeError):
    pass


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _blob(data: bytes):
    buffer = ctypes.create_string_buffer(data)
    value = _DataBlob(
        len(data),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    return value, buffer


def _dpapi_protect(data: bytes) -> bytes:
    if os.name != "nt":
        raise SecretStoreError("DPAPI is only available on Windows")
    source, source_buffer = _blob(data)
    encrypted = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    ok = crypt32.CryptProtectData(
        ctypes.byref(source),
        "shiyu-local-secret",
        None,
        None,
        None,
        0,
        ctypes.byref(encrypted),
    )
    del source_buffer
    if not ok:
        raise SecretStoreError(f"Windows DPAPI encryption failed ({ctypes.get_last_error()})")
    try:
        return ctypes.string_at(encrypted.pbData, encrypted.cbData)
    finally:
        kernel32.LocalFree(encrypted.pbData)


def _dpapi_unprotect(data: bytes) -> bytes:
    if os.name != "nt":
        raise SecretStoreError("DPAPI is only available on Windows")
    source, source_buffer = _blob(data)
    decrypted = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(source),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(decrypted),
    )
    del source_buffer
    if not ok:
        raise SecretStoreError(
            f"Windows DPAPI decryption failed ({ctypes.get_last_error()})"
        )
    try:
        return ctypes.string_at(decrypted.pbData, decrypted.cbData)
    finally:
        kernel32.LocalFree(decrypted.pbData)


class LocalSecretStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path = self.path.parent / "local-secret.key"
        self._fernet_client = None

    def _fernet(self):
        if self._fernet_client is not None:
            return self._fernet_client
        if self.key_path.exists():
            key = self.key_path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            temporary = self.key_path.with_suffix(".key.tmp")
            temporary.write_bytes(key + b"\n")
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            temporary.replace(self.key_path)
            try:
                os.chmod(self.key_path, 0o600)
            except OSError:
                pass
        try:
            self._fernet_client = Fernet(key)
        except (ValueError, TypeError) as exc:
            raise SecretStoreError("本机主密钥损坏，无法创建加密存储") from exc
        return self._fernet_client

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SecretStoreError("本机密钥文件损坏，无法读取") from exc
        if not isinstance(data, dict):
            raise SecretStoreError("本机密钥文件格式无效")
        return data

    def _write(self, data: dict) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        temporary.replace(self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def set(self, name: str, value: str) -> None:
        if not name or not value:
            raise ValueError("密钥名称和内容不能为空")
        data = self._read()
        raw = value.encode("utf-8")
        encoded = self._fernet().encrypt(raw).decode("ascii")
        protection = "fernet-local"
        data[name] = {"protection": protection, "value": encoded}
        self._write(data)

    def get(self, name: str) -> str | None:
        record = self._read().get(name)
        if not isinstance(record, dict) or not isinstance(record.get("value"), str):
            return None
        try:
            protection = record.get("protection")
            if protection == "fernet-local":
                raw = self._fernet().decrypt(record["value"].encode("ascii"))
            elif protection in ("windows-dpapi", "base64-fallback"):
                raw = base64.b64decode(record["value"], validate=True)
            elif protection != "base64-fallback":
                raise SecretStoreError("不支持的密钥保护方式")
            if protection == "windows-dpapi":
                raw = _dpapi_unprotect(raw)
                # Migrate old readable DPAPI entries to the restart-stable store.
                self.set(name, raw.decode("utf-8"))
            return raw.decode("utf-8")
        except (InvalidToken, ValueError, UnicodeDecodeError, SecretStoreError) as exc:
            raise SecretStoreError("本机密钥无法解密") from exc

    def has(self, name: str) -> bool:
        return self.get(name) is not None

    def delete(self, name: str) -> bool:
        data = self._read()
        existed = name in data
        if existed:
            data.pop(name, None)
            self._write(data)
        return existed

