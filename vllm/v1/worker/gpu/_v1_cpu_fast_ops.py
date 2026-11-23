# SPDX-License-Identifier: Apache-2.0
"""Lightweight CPU C extension for fast Python list[int] -> int32 buffer writes."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sysconfig
from typing import Any

_C_SRC = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

static PyObject* copy_pylist_i32(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "copy_pylist_i32(list_obj, dst_ptr)");
        return NULL;
    }
    PyObject* list_obj = args[0];
    if (!PyList_CheckExact(list_obj)) {
        Py_RETURN_FALSE;
    }
    uintptr_t ptr_val = (uintptr_t)PyLong_AsUnsignedLongLong(args[1]);
    if (ptr_val == 0 && PyErr_Occurred()) {
        return NULL;
    }
    int32_t* out = (int32_t*)ptr_val;
    Py_ssize_t n = PyList_GET_SIZE(list_obj);
    PyObject** items = ((PyListObject*)list_obj)->ob_item;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* obj = items[i];
        if (!PyLong_CheckExact(obj)) {
            Py_RETURN_FALSE;
        }
        int overflow = 0;
        long val = PyLong_AsLongAndOverflow(obj, &overflow);
        if (overflow || val < INT32_MIN || val > INT32_MAX) {
            Py_RETURN_FALSE;
        }
        out[i] = (int32_t)val;
    }
    Py_RETURN_TRUE;
}

static PyMethodDef _Methods[] = {
    {"copy_pylist_i32", (PyCFunction)copy_pylist_i32, METH_FASTCALL, "Fast copy list[int] to int32*"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "_v1_cpu_fast_ops_ext",
    NULL,
    -1,
    _Methods
};

PyMODINIT_FUNC PyInit_%s(void) {
    return PyModule_Create(&_moduledef);
}
"""

_SRC_HASH = hashlib.sha256(_C_SRC.encode("utf-8")).hexdigest()[:12]
_EXT_NAME = f"v1_cpu_fast_ops_{_SRC_HASH}"
_CACHED_EXT: Any = None
_TRIED_BUILD = False


def _candidate_ext_dirs() -> list[str]:
    dirs: list[str] = []
    if d := os.environ.get("VLLM_GEMMA4_SM80_EXT_DIR"):
        dirs.append(d)
    if c := os.environ.get("VLLM_CACHE_ROOT"):
        dirs.append(os.path.join(c, "v1_sm80_ext"))
    dirs.append(
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "vllm_v1_sm80",
        )
    )
    return dirs


def _load_so_direct(so_path: str) -> Any | None:
    try:
        spec = importlib.util.spec_from_file_location(_EXT_NAME, so_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _find_prebuilt_so() -> str | None:
    so_name = f"{_EXT_NAME}.so"
    for d in _candidate_ext_dirs():
        candidate = os.path.join(d, so_name)
        if os.path.isfile(candidate):
            return candidate
    return None


def get_cpu_fast_ext(allow_compile: bool = True) -> Any | None:
    """Load prebuilt or compile CPU-only C extension for fast pylist packing."""
    global _CACHED_EXT, _TRIED_BUILD
    if _CACHED_EXT is not None:
        return _CACHED_EXT

    so_path = _find_prebuilt_so()
    if so_path is not None:
        _CACHED_EXT = _load_so_direct(so_path)
        if _CACHED_EXT is not None:
            return _CACHED_EXT

    if (
        not allow_compile
        or os.environ.get("VLLM_GEMMA4_SM80_NO_COMPILE") == "1"
        or _TRIED_BUILD
    ):
        return None
    _TRIED_BUILD = True

    gcc = shutil.which("gcc")
    if gcc is None:
        return None

    py_include = sysconfig.get_path("include")
    if not py_include or not os.path.isfile(os.path.join(py_include, "Python.h")):
        return None

    for build_dir in _candidate_ext_dirs():
        try:
            os.makedirs(build_dir, exist_ok=True)
            pid = os.getpid()
            c_path = os.path.join(build_dir, f".{_EXT_NAME}.{pid}.c")
            tmp_so = os.path.join(build_dir, f".{_EXT_NAME}.{pid}.tmp.so")
            final_so = os.path.join(build_dir, f"{_EXT_NAME}.so")
            with open(c_path, "w") as f:
                f.write(_C_SRC % _EXT_NAME)
            subprocess.check_call(
                [gcc, "-O3", "-shared", "-fPIC", f"-I{py_include}", c_path, "-o", tmp_so],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.replace(tmp_so, final_so)
            try:
                os.remove(c_path)
            except OSError:
                pass
            _CACHED_EXT = _load_so_direct(final_so)
            if _CACHED_EXT is not None:
                return _CACHED_EXT
        except Exception:
            continue
    return None


_COPY_PYLIST_I32_FN: Any = None
_COPY_FN_INIT = False


def get_copy_pylist_i32():
    global _COPY_PYLIST_I32_FN, _COPY_FN_INIT
    if not _COPY_FN_INIT:
        _COPY_FN_INIT = True
        ext = get_cpu_fast_ext(allow_compile=True)
        if ext is not None:
            _COPY_PYLIST_I32_FN = getattr(ext, "copy_pylist_i32", None)
    return _COPY_PYLIST_I32_FN
