"""tests/store_server.py DIR PORT: a file store on 127.0.0.1 for GPU jobs in the container (which sees no host
directory): GET /NAME, PUT /NAME, GET / lists the names. Started by tests/store.sh."""
import http.server
import os
import shutil
import sys

D, PORT = sys.argv[1], int(sys.argv[2])


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _path(self):
        return os.path.join(D, os.path.basename(self.path))

    def do_GET(self):
        if self.path == "/":
            body = "\n".join(sorted(os.listdir(D))).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        p = self._path()
        if not os.path.isfile(p):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            return self.end_headers()
        self.send_response(200)
        self.send_header("Content-Length", str(os.path.getsize(p)))
        self.end_headers()
        with open(p, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1 << 24)

    def do_PUT(self):
        n = int(self.headers["Content-Length"])
        p = self._path()
        with open(p + ".tmp", "wb") as f:
            while n:
                b = self.rfile.read(min(n, 1 << 24))
                f.write(b)
                n -= len(b)
        os.replace(p + ".tmp", p)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


os.makedirs(D, exist_ok=True)
http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
