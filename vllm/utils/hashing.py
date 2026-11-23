# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import pickle
import struct
from _hashlib import HASH, UnsupportedDigestmodError
from collections.abc import Callable, Sequence
from typing import Any

import cbor2

try:
    # It is important that this remains an optional dependency.
    # It would not be allowed in environments with strict security controls,
    # so it's best not to have it installed when not in use.
    import xxhash as _xxhash

    if not hasattr(_xxhash, "xxh3_128_digest"):
        _xxhash = None
except ImportError:  # pragma: no cover
    _xxhash = None


def sha256(input: Any) -> bytes:
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does
    not use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        Bytes representing the SHA-256 hash of the serialized input.

    """
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(input_bytes).digest()


def sha256_cbor(input: Any) -> bytes:
    """Hash objects using CBOR serialization and SHA-256.

    This option is useful for non-Python-dependent serialization and hashing.

    Args:
        input: Object to be serialized and hashed. Supported types include
            basic Python types and complex structures like lists, tuples, and
            dictionaries.
            Custom classes must implement CBOR serialization methods.

    Returns:
        Bytes representing the SHA-256 hash of the CBOR serialized input.

    """
    input_bytes = cbor2.dumps(input, canonical=True)
    return hashlib.sha256(input_bytes).digest()


_CBOR_UINT16_BYTES: tuple[bytes, ...] = tuple(
    bytes((t,))
    if t < 24
    else bytes((0x18, t))
    if t < 256
    else struct.pack(">BH", 0x19, t)
    for t in range(65536)
)
_CBOR_UINT32_PACK = struct.Struct(">BI").pack
_CBOR_HDR_32_16 = b"\x83\x58\x20"


def _encode_cbor_block_plain_bytes(
    parent_hash: bytes,
    token_ids: Sequence[int],
) -> bytes | None:
    """Encode ``(parent_hash, tuple(token_ids), None)`` in canonical CBOR.

    Returns ``None`` if inputs fall outside the fast range so callers can fall
    back to ``cbor2.dumps(..., canonical=True)``.
    """
    n_toks = len(token_ids)
    if n_toks > 255:
        return None
    plen = len(parent_hash)
    if plen == 32 and n_toks < 24:
        parts: list[bytes] = [_CBOR_HDR_32_16, parent_hash, bytes((0x80 + n_toks,))]
    else:
        if plen < 24:
            p_hdr = bytes((0x40 + plen,))
        elif plen < 256:
            p_hdr = bytes((0x58, plen))
        else:
            return None
        t_hdr = bytes((0x80 + n_toks,)) if n_toks < 24 else bytes((0x98, n_toks))
        parts = [b"\x83", p_hdr, parent_hash, t_hdr]

    tbl = _CBOR_UINT16_BYTES
    pack_u32 = _CBOR_UINT32_PACK
    parts_append = parts.append
    for tok in token_ids:
        if type(tok) is not int:
            return None
        if 0 <= tok < 65536:
            parts_append(tbl[tok])
        elif 65536 <= tok <= 0xFFFFFFFF:
            parts_append(pack_u32(0x1A, tok))
        else:
            return None
    parts_append(b"\xf6")
    return b"".join(parts)


def sha256_cbor_block_plain(
    parent_hash: bytes,
    token_ids: Sequence[int],
) -> bytes | None:
    """Fast canonical CBOR + SHA-256 digest for plain ``extra_keys=None`` blocks."""
    encoded = _encode_cbor_block_plain_bytes(parent_hash, token_ids)
    if encoded is None:
        return None
    return hashlib.sha256(encoded).digest()


def _xxhash_digest(input_bytes: bytes) -> bytes:
    if _xxhash is None:
        raise ModuleNotFoundError(
            "xxhash is required for the 'xxhash' prefix caching hash algorithms. "
            "Install it via `pip install xxhash`."
        )
    return _xxhash.xxh3_128_digest(input_bytes)


def xxhash(input: Any) -> bytes:
    """Hash picklable objects using xxHash."""
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return _xxhash_digest(input_bytes)


def xxhash_cbor(input: Any) -> bytes:
    """Hash objects serialized with CBOR using xxHash."""
    input_bytes = cbor2.dumps(input, canonical=True)
    return _xxhash_digest(input_bytes)


def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes]:
    """Get a hash function by name, or raise an error if the function is not found.

    Args:
        hash_fn_name: Name of the hash function.

    Returns:
        A hash function.

    """
    if hash_fn_name == "sha256":
        return sha256
    if hash_fn_name == "sha256_cbor":
        return sha256_cbor
    if hash_fn_name == "xxhash":
        return xxhash
    if hash_fn_name == "xxhash_cbor":
        return xxhash_cbor

    raise ValueError(f"Unsupported hash function: {hash_fn_name}")


def safe_hash(data: bytes, usedforsecurity: bool = True) -> HASH:
    """Hash for configs, defaulting to md5 but falling back to sha256
    in FIPS constrained environments.

    Args:
        data: bytes
        usedforsecurity: Whether the hash is used for security purposes

    Returns:
        Hash object

    """
    try:
        return hashlib.md5(data, usedforsecurity=usedforsecurity)
    except (UnsupportedDigestmodError, ValueError):
        return hashlib.sha256(data)
