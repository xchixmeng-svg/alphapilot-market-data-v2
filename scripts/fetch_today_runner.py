#!/usr/bin/env python3
"""Run the locked daily downloader with a narrow TPEx TLS fallback.

TPEx occasionally serves an incomplete certificate chain to GitHub-hosted runners.
We keep normal TLS verification everywhere. Only when Requests raises SSLError for
exactly www.tpex.org.tw do we retry that request with certificate verification
disabled. The existing fetch_today.py payload-date consensus, cross-source date
checks, normalization and non-empty data gates still run unchanged.

This wrapper is intentionally small so the underlying market-data normalization
and DATA_GATE semantics remain unchanged.
"""
from __future__ import annotations

import runpy
from urllib.parse import urlparse

import requests
import urllib3

_real_get = requests.Session.get


def _tpex_tls_compatible_get(self, url, *args, **kwargs):
    try:
        return _real_get(self, url, *args, **kwargs)
    except requests.exceptions.SSLError:
        host = (urlparse(str(url)).hostname or "").lower()
        if host != "www.tpex.org.tw":
            raise
        print(
            "[tls-fallback] TPEx certificate chain verification failed on runner; "
            "retrying exact official host with verify=False while retaining all data gates.",
            flush=True,
        )
        retry_kwargs = dict(kwargs)
        retry_kwargs["verify"] = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return _real_get(self, url, *args, **retry_kwargs)


requests.Session.get = _tpex_tls_compatible_get
runpy.run_path("scripts/fetch_today.py", run_name="__main__")
