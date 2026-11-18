"""HTTP front end: GET /v1/models, POST /v1/completions (prompt as token ids).
One request at a time; the engine object does the work on the GPU."""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = "qwen3.8-27b"


def make_handler(engine, lock):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, doc):
            body = json.dumps(doc).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/health"):
                self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n))
            except ValueError:
                return self._send(400, {"error": "bad json"})
            if self.path.rstrip("/") != "/v1/completions":
                return self._send(404, {"error": "only /v1/completions"})
            prompt = req.get("prompt")
            if not (isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt)):
                return self._send(400, {"error": "prompt must be a list of token ids"})
            max_tokens = int(req.get("max_tokens") or 16)
            t0 = time.monotonic()
            with lock:
                try:
                    out, finish = engine.generate(prompt, max_tokens)
                except Exception as e:  # report, keep serving
                    import traceback
                    traceback.print_exc()
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            dt = time.monotonic() - t0
            print(f"req prompt={len(prompt)} out={len(out)} {finish} {dt*1000:.0f} ms", flush=True)
            doc = {
                "id": "cmpl-0", "object": "text_completion", "created": int(time.time()), "model": MODEL_NAME,
                "choices": [{"index": 0, "text": "", "token_ids": out, "finish_reason": finish, "logprobs": None}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(out),
                          "total_tokens": len(prompt) + len(out)},
            }
            if req.get("stream"):
                return self._stream(doc)
            self._send(200, doc)

        def _stream(self, doc):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"data: " + json.dumps(doc).encode() + b"\n\ndata: [DONE]\n\n")
            self.close_connection = True

    return H


def serve(engine, port):
    srv = ThreadingHTTPServer(("0.0.0.0", port), make_handler(engine, threading.Lock()))
    print(f"listening on {port}", flush=True)
    srv.serve_forever()
