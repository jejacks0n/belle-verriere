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


def serve(app, host: str, port: int):
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning",
                                           timeout_graceful_shutdown=2))
    stop = server.handle_exit

    def handle_exit(sig, frame):  # Ctrl-C / kill: release the Roku's held request first, then stop
        app.state.release_waiters()
        stop(sig, frame)

    server.handle_exit = handle_exit
    try:
        server.run()
    except KeyboardInterrupt:  # uvicorn re-raises Ctrl-C after a clean shutdown (uvicorn.run hides it too)
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets your phone reach the control page")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    ip = lan_ip()
    print(f"\n  Control page:  http://{ip}:{a.port}/   (open on your phone)")
    print(f"  Window display: http://localhost:{a.port}/display\n")
    serve(create_app(), a.host, a.port)
