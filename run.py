#!/usr/bin/env python3
"""Start the stained glass window.

    python run.py              # real generation with mflux (Apple Silicon)
    python run.py --mock       # fake images, to test everything else
"""
import argparse
import socket

import uvicorn

from belleverriere.server import create_app


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except Exception:
        return "localhost"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="use a fake generator (no GPU needed)")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets your phone reach the control page")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    ip = lan_ip()
    print(f"\n  Control page:  http://{ip}:{a.port}/   (open on your phone)")
    print(f"  Window display: http://localhost:{a.port}/display\n")
    uvicorn.run(create_app("mock" if a.mock else "mflux"), host=a.host, port=a.port, log_level="warning")
