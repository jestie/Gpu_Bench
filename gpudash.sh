#!/usr/bin/env bash
# Launch GPU Dash and open it in the browser.
cd "$(dirname "$0")" && exec python3 server.py "$@"
