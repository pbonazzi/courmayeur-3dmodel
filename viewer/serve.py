#!/usr/bin/env python3
"""Serve the project folder locally and open the terrain viewer in the browser.

    python3 viewer/serve.py            # http://localhost:8765/viewer/
    python3 viewer/serve.py --port 9000 --no-open
"""
import argparse, functools, http.server, os, threading, webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--no-open", action="store_true")
args = ap.parse_args()

Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
url = f"http://localhost:{args.port}/viewer/"
print(f"Viewer running at {url}  (Ctrl+C to stop)")
if not args.no_open:
    threading.Timer(0.5, webbrowser.open, [url]).start()
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
