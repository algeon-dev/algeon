#!/usr/bin/env python3
"""Logging S3 reverse-proxy for ranged-GET evidence.

Sits in front of the RustFS S3 endpoint and records, per request, the method,
object path, and any `Range:` header, then transparently forwards to RustFS.
This is the only reliable way to prove whether a `SELECT ... LIMIT N` over an
Iceberg table issues ranged GETs (footer-first, then row-group ranges) or a
full-object GET of the data file: RustFS's own logs report storage-shard reads
for the whole object regardless of the client Range header, so they cannot
distinguish the two.

Usage (point algeon-datafusion at the proxy instead of RustFS directly):

    fixture/iceberg-local/s3_range_probe.py --listen 9100 --upstream http://localhost:9000 \
        --log /tmp/s3_range_probe.jsonl &
    ALGEON_ICEBERG_S3_ENDPOINT=http://localhost:9100 ... <run the scan>

Each forwarded request is appended to the log file as one JSON object:

    {"method": "GET", "path": "/lakehouse/warehouse/.../nation/data/...parquet",
     "range": "bytes=0-65535", "status": 206}

A request with no `range` and a 200 over a `*/data/*.parquet` object is a
full-object data-file GET (the regression tracked for the LIMIT path); ranged
GETs (206 with a `bytes=...` range) are the desired footer-first behavior.

Diagnostic tool only; local, header-signed SigV4 traffic. Do NOT put it in
front of presigned-URL traffic: the log records the full request path including
the query string, which for presigned URLs carries the signature/credential.
It also buffers each response fully in memory before forwarding, so a full-object
GET of a large data file will spike its RSS.

Stdlib only; no third-party deps so it can live in the repo.
"""

import argparse
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://localhost:9000"
LOG_PATH = "/tmp/s3_range_probe.jsonl"
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def _record(entry: dict) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
        fh.flush()


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # silence default stderr access log
        pass

    def _proxy(self, method: str) -> None:
        body = None
        if "content-length" in (k.lower() for k in self.headers.keys()):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else None

        url = UPSTREAM.rstrip("/") + self.path
        fwd_headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        # AWS SigV4 signs the Host header. The client signed for *this* proxy's
        # host:port, so forward that exact value to the upstream rather than
        # letting urllib rewrite it to the upstream host (which would yield a
        # SignatureDoesNotMatch 403). RustFS validates the signature against the
        # Host it receives, so the signed and received Host must agree.
        client_host = self.headers.get("Host")
        if client_host:
            fwd_headers["Host"] = client_host
        req = urllib.request.Request(url, data=body, method=method, headers=fwd_headers)

        rng = self.headers.get("Range")
        try:
            resp = urllib.request.urlopen(req)  # noqa: S310 (trusted local upstream)
            status = resp.status
            payload = resp.read()
            resp_headers = resp.getheaders()
        except urllib.error.HTTPError as err:  # forward error responses verbatim
            status = err.code
            payload = err.read()
            resp_headers = list(err.headers.items())

        _record({"method": method, "path": self.path, "range": rng, "status": status})

        self.send_response(status)
        for k, v in resp_headers:
            if k.lower() in HOP_BY_HOP:
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_GET(self):
        self._proxy("GET")

    def do_HEAD(self):
        self._proxy("HEAD")

    def do_PUT(self):
        self._proxy("PUT")

    def do_POST(self):
        self._proxy("POST")

    def do_DELETE(self):
        self._proxy("DELETE")


def main() -> int:
    global UPSTREAM, LOG_PATH
    parser = argparse.ArgumentParser(description="Logging S3 reverse-proxy for ranged-GET evidence")
    parser.add_argument("--listen", type=int, default=9100, help="local port to listen on")
    parser.add_argument("--upstream", default=UPSTREAM, help="RustFS S3 endpoint to forward to")
    parser.add_argument("--log", default=LOG_PATH, help="JSONL access-log path")
    args = parser.parse_args()

    UPSTREAM = args.upstream
    LOG_PATH = args.log
    # Truncate the log so each run starts clean.
    open(LOG_PATH, "w", encoding="utf-8").close()

    server = ThreadingHTTPServer(("0.0.0.0", args.listen), ProxyHandler)
    print(f"[s3-range-probe] listening on :{args.listen} -> {UPSTREAM}, log={LOG_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
