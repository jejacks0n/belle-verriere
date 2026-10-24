#!/usr/bin/env python3
"""Start Belle Verrière: python run.py [--host 0.0.0.0] [--port 8765]"""
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
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets your phone reach the control page")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    ip = lan_ip()
    print(f"\n  Control page:  http://{ip}:{a.port}/   (open on your phone)")
    print(f"  Window display: http://localhost:{a.port}/display\n")
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning",
                timeout_graceful_shutdown=2)  # don't wait on the Roku's held long-poll request
