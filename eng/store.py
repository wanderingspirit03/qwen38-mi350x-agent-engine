"""Client of the development file store (tests/store.sh): torch objects by name over HTTP on the loopback."""
import io
import os
import urllib.error
import urllib.request

import torch

URL = os.environ.get("ENG_STORE", "http://127.0.0.1:8765")


def names():
    return urllib.request.urlopen(URL + "/").read().decode().split()


def save(name, obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    urllib.request.urlopen(urllib.request.Request(f"{URL}/{name}", data=buf.getvalue(), method="PUT")).read()


def load(name, default=None):
    try:
        data = urllib.request.urlopen(f"{URL}/{name}").read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return default
        raise
    return torch.load(io.BytesIO(data), weights_only=False)
