"""Helpers for the JSON / form POST APIs (ecd, napr).

These sites expose their data through ``POST`` endpoints that return JSON. Scrapy is
HTML-first, so these small builders keep the spiders readable and apply the headers
the servers expect (``X-Requested-With`` etc.).
"""

import json
from urllib.parse import urlencode

import scrapy

_JSON_HEADERS = {
    "Content-Type": "application/json",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/plain, */*",
}

_FORM_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/plain, */*",
}


def json_post(url, payload, *, callback, meta=None, headers=None, **kwargs):
    """A ``scrapy.Request`` that POSTs a JSON body and expects JSON back (ecd)."""
    request_headers = {**_JSON_HEADERS, **(headers or {})}
    return scrapy.Request(
        url=url,
        method="POST",
        body=json.dumps(payload),
        headers=request_headers,
        callback=callback,
        meta=meta or {},
        dont_filter=True,
        **kwargs,
    )


def form_post(url, payload, *, callback, meta=None, headers=None, **kwargs):
    """A ``scrapy.Request`` that POSTs a urlencoded form body (napr)."""
    request_headers = {**_FORM_HEADERS, **(headers or {})}
    return scrapy.Request(
        url=url,
        method="POST",
        body=urlencode(payload),
        headers=request_headers,
        callback=callback,
        meta=meta or {},
        dont_filter=True,
        **kwargs,
    )


def loads_maybe_double(text: str):
    """Parse JSON that may be double-encoded.

    napr's ``/legal_search`` returns a JSON *string* whose contents are themselves
    JSON, so a single ``json.loads`` yields a ``str``; decode once more in that case.
    """
    data = json.loads(text)
    if isinstance(data, str):
        data = json.loads(data)
    return data
