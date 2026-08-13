#!/usr/bin/env python3
"""
Serve the course locally and open it in your browser.

    python serve.py

Why bother instead of double-clicking index.html?
Browsers block localStorage on file:// URLs, so your chapter-progress
marks would not be saved. Serving over http://localhost fixes that.
"""
import http.server
import os
import socketserver
import threading
import webbrowser

PORT = 8777
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "course")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        # Without this the browser serves style.css / course.js from memory cache
        # and your edits appear to do nothing. Local dev server: never cache.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the console quiet
        pass


def main():
    import sys

    open_browser = "--no-open" not in sys.argv
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/index.html"
        print(f"\n  Transformers From Scratch\n  Serving {ROOT}\n  -> {url}\n\n  Ctrl-C to stop.\n")
        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopped.")


if __name__ == "__main__":
    main()
