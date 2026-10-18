#!/usr/bin/env python3
"""GPU Dash: live GPU / Ollama dashboard. Stdlib only.

Samples nvidia-smi, /proc and the Ollama API once a second and streams the
results to the browser over Server-Sent Events.

Also runs a pass-through proxy in front of Ollama (port 11435 by default).
Point clients at the proxy and the dashboard sees every token as it streams,
which is how it shows live tokens per second.
"""
import http.client
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("GPU_DASH_PORT", 8090))
PROXY_PORT = int(os.environ.get("GPU_DASH_PROXY_PORT", 11435))
PROXY_BIND = os.environ.get("GPU_DASH_PROXY_BIND", "127.0.0.1")
# Not OLLAMA_HOST: clients may set that to the proxy, which would loop.
OLLAMA = os.environ.get("GPU_DASH_OLLAMA", "http://localhost:11434")
HISTORY = 120  # seconds of sparkline history
ROOT = Path(__file__).parent
BENCH_FILE = ROOT / "bench.json"
BENCH_PROMPT = ("Explain how a graphics card renders a frame, step by step, "
                "for a curious teenager. Be thorough.")
BENCH_TOKENS = 256

GPU_FIELDS = [
    "name", "driver_version", "vbios_version", "memory.total", "memory.used",
    "utilization.gpu", "utilization.memory", "temperature.gpu", "power.draw",
    "power.limit", "fan.speed", "clocks.sm", "clocks.mem", "clocks.max.sm",
    "clocks.max.mem", "pstate", "pcie.link.gen.current", "pcie.link.width.current",
]

state = {"static": {}, "live": {}, "history": {k: deque(maxlen=HISTORY) for k in
         ("gpu", "cpu", "power", "temp", "vram", "tps")}}
lock = threading.Lock()


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Tracker:
    """Counts streamed tokens across all in-flight requests."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ticks = deque()  # timestamps of recent tokens
        self.active = {}      # request id -> {model, start, tokens}
        self.done = deque(maxlen=50)
        self.next_id = 0

    def start(self, model):
        with self.lock:
            self.next_id += 1
            self.active[self.next_id] = {"model": model, "start": time.time(), "tokens": 0}
            return self.next_id

    def tick(self, rid):
        now = time.time()
        with self.lock:
            self.ticks.append(now)
            if rid in self.active:
                req = self.active[rid]
                req["tokens"] += 1
                req.setdefault("first", now)
                req["last"] = now

    def finish(self, rid, final=None):
        """final: Ollama's closing JSON (has exact eval counts), if seen."""
        with self.lock:
            req = self.active.pop(rid, None)
        if not req:
            return
        final = final or {}
        ev_n, ev_ns = final.get("eval_count"), final.get("eval_duration")
        pr_n, pr_ns = final.get("prompt_eval_count"), final.get("prompt_eval_duration")
        if ev_n and ev_ns:
            tps = ev_n / (ev_ns / 1e9)
        else:  # no stats (e.g. OpenAI-style stream): time first to last token
            ev_n = req["tokens"]
            secs = req.get("last", 0) - req.get("first", 0)
            tps = (ev_n - 1) / secs if ev_n > 2 and secs > 0 else None
        if not ev_n or not tps:
            return
        with self.lock:
            self.done.append({
                "model": req["model"], "tps": tps, "out": ev_n, "in": pr_n,
                "prompt_tps": pr_n / (pr_ns / 1e9) if pr_n and pr_ns else None,
                "at": time.time(),
            })

    def snapshot(self):
        now = time.time()
        with self.lock:
            while self.ticks and self.ticks[0] < now - 2:
                self.ticks.popleft()
            # Divide by the time the window actually covers, so a stream that
            # started half a second ago isn't averaged over a full two seconds.
            span = min(2, max(0.5, now - self.ticks[0])) if self.ticks else 2
            live_tps = len(self.ticks) / span
            active = [dict(v, elapsed=now - v["start"]) for v in self.active.values()]
            return {"live_tps": live_tps, "active": active, "done": list(self.done)}


tracker = Tracker()


def read_gpu():
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
    return dict(zip(GPU_FIELDS, (s.strip() for s in out.split(","))))


def read_gpu_procs():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    procs = []
    for line in out.strip().splitlines():
        pid, name, mem = (s.strip() for s in line.split(","))
        procs.append({"pid": pid, "name": name, "mem": num(mem)})
    return procs


def cuda_version():
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5).stdout
    if "CUDA Version:" in out:
        return out.split("CUDA Version:")[1].split()[0]
    return None


def cpu_times():
    with open("/proc/stat") as f:
        vals = [int(x) for x in f.readline().split()[1:]]
    idle = vals[3] + vals[4]
    return sum(vals), idle


def read_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0]) * 1024
    return info["MemTotal"], info["MemTotal"] - info["MemAvailable"]


def ollama(path):
    try:
        with urllib.request.urlopen(f"{OLLAMA}{path}", timeout=1) as r:
            return json.load(r)
    except Exception:
        return None


def disks():
    result = []
    for label, path in (("Linux /", "/"), ("Windows C:", "/mnt/c")):
        if os.path.exists(path):
            u = shutil.disk_usage(path)
            result.append({"label": label, "total": u.total, "free": u.free})
    return result


def sampler():
    g = read_gpu()
    cpu_name = "CPU"
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    with lock:
        state["static"] = {
            "name": g["name"], "driver": g["driver_version"], "vbios": g["vbios_version"],
            "cuda": cuda_version(), "vram_total": num(g["memory.total"]),
            "power_limit": num(g["power.limit"]), "max_sm": num(g["clocks.max.sm"]),
            "max_mem": num(g["clocks.max.mem"]), "cpu_name": cpu_name, "cpus": os.cpu_count(),
            "started": time.time(), "proxy": f"{PROXY_BIND}:{PROXY_PORT}",
        }
    prev_total, prev_idle = cpu_times()
    energy_wh = 0.0
    last = time.time()
    tick = 0
    tags = None
    while True:
        time.sleep(1)
        now = time.time()
        dt, last = now - last, now
        try:
            g = read_gpu()
            procs = read_gpu_procs()
        except Exception as e:
            with lock:
                state["live"] = {"error": str(e)}
            continue
        total, idle = cpu_times()
        cpu = 100 * (1 - (idle - prev_idle) / max(total - prev_total, 1))
        prev_total, prev_idle = total, idle
        ram_total, ram_used = read_meminfo()
        power = num(g["power.draw"]) or 0
        energy_wh += power * dt / 3600
        if tick % 10 == 0:
            tags = ollama("/api/tags")
        tick += 1
        ps = ollama("/api/ps")
        gen = tracker.snapshot()
        live = {
            "gpu": num(g["utilization.gpu"]), "mem_bus": num(g["utilization.memory"]),
            "vram_used": num(g["memory.used"]), "temp": num(g["temperature.gpu"]),
            "power": power, "fan": num(g["fan.speed"]), "sm": num(g["clocks.sm"]),
            "mem_clock": num(g["clocks.mem"]), "pstate": g["pstate"],
            "pcie": f"Gen{g['pcie.link.gen.current']} x{g['pcie.link.width.current']}",
            "cpu": round(cpu, 1), "ram_total": ram_total, "ram_used": ram_used,
            "energy_wh": energy_wh, "procs": procs, "disks": disks() if tick % 10 == 1 else None,
            "ollama_up": ps is not None,
            "loaded": (ps or {}).get("models", []),
            "installed": [{"name": m["name"], "size": m["size"],
                           "params": m["details"].get("parameter_size"),
                           "quant": m["details"].get("quantization_level"),
                           "family": m["details"].get("family")}
                          for m in (tags or {}).get("models", [])],
            "gen": gen, "bench": bench.snapshot(),
        }
        with lock:
            if live["disks"] is None:
                live["disks"] = state["live"].get("disks")
            state["live"] = live
            h = state["history"]
            h["gpu"].append(live["gpu"])
            h["cpu"].append(live["cpu"])
            h["power"].append(power)
            h["temp"].append(live["temp"])
            h["vram"].append(live["vram_used"])
            h["tps"].append(gen["live_tps"])


# ---------------------------------------------------------------- benchmark

class Bench:
    """Runs a fixed prompt against models one at a time and keeps the scores."""

    def __init__(self):
        self.lock = threading.Lock()
        self.queue = deque()
        self.current = None  # {model, phase, tokens}
        self.results = {}
        if BENCH_FILE.exists():
            try:
                self.results = json.loads(BENCH_FILE.read_text())
            except ValueError:
                pass
        threading.Thread(target=self._worker, daemon=True).start()

    def enqueue(self, models):
        with self.lock:
            for m in models:
                if m not in self.queue and (not self.current or self.current["model"] != m):
                    self.queue.append(m)

    def cancel(self):
        with self.lock:
            self.queue.clear()

    def snapshot(self):
        with self.lock:
            return {"results": self.results, "queue": list(self.queue),
                    "current": dict(self.current) if self.current else None}

    def _worker(self):
        while True:
            with self.lock:
                model = self.queue.popleft() if self.queue else None
                if model:
                    self.current = {"model": model, "phase": "loading", "tokens": 0}
            if not model:
                time.sleep(0.5)
                continue
            try:
                result = self._run(model)
            except Exception as e:
                result = {"error": str(e)[:200], "at": time.time()}
            with self.lock:
                self.results[model] = result
                self.current = None
                BENCH_FILE.write_text(json.dumps(self.results, indent=1))

    def _run(self, model):
        # Unload whatever else is resident so every model gets the same, cold start.
        for m in (ollama("/api/ps") or {}).get("models", []):
            if m["name"] != model:
                post_json("/api/generate", {"model": m["name"], "keep_alive": 0})
        body = {"model": model, "prompt": BENCH_PROMPT, "stream": True,
                "options": {"num_predict": BENCH_TOKENS, "temperature": 0, "seed": 1}}
        rid = tracker.start(model)
        powers, temps, final = [], [], {}
        try:
            conn, resp = open_ollama("POST", "/api/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"}, timeout=600)
            if resp.status != 200:
                raise RuntimeError(resp.read().decode(errors="replace"))
            while line := resp.readline():
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                if chunk.get("response") or chunk.get("thinking"):
                    tracker.tick(rid)
                    with lock:
                        live = state["live"]
                        powers.append(live.get("power") or 0)
                        temps.append(live.get("temp") or 0)
                    with self.lock:
                        self.current["phase"] = "generating"
                        self.current["tokens"] += 1
                if chunk.get("done"):
                    final = chunk
            conn.close()
        finally:
            tracker.finish(rid, final)
        ps = next((m for m in (ollama("/api/ps") or {}).get("models", []) if m["name"] == model), {})
        ev_n, ev_s = final.get("eval_count", 0), final.get("eval_duration", 0) / 1e9
        pr_n, pr_s = final.get("prompt_eval_count", 0), final.get("prompt_eval_duration", 0) / 1e9
        avg_w = sum(powers) / len(powers) if powers else None
        tps = ev_n / ev_s if ev_s else None
        return {
            "tps": tps, "prompt_tps": pr_n / pr_s if pr_s else None,
            "load_s": final.get("load_duration", 0) / 1e9,
            "watts": avg_w, "j_per_tok": avg_w / tps if avg_w and tps else None,
            "peak_temp": max(temps) if temps else None,
            "vram": ps.get("size_vram"), "size": ps.get("size"), "tokens": ev_n,
            "at": time.time(),
        }


def open_ollama(method, path, body=None, headers=None, timeout=600):
    u = urllib.parse.urlsplit(OLLAMA)
    conn = http.client.HTTPConnection(u.hostname or "localhost", u.port or 80, timeout=timeout)
    conn.request(method, path, body=body, headers=headers or {})
    return conn, conn.getresponse()


def post_json(path, payload):
    try:
        conn, resp = open_ollama("POST", path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"}, timeout=60)
        resp.read()
        conn.close()
    except Exception:
        pass


bench = Bench()


# ---------------------------------------------------------------- web servers

def snapshot():
    with lock:
        return json.dumps({"static": state["static"], "live": state["live"],
                           "history": {k: list(v) for k, v in state["history"].items()}})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            body = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(f"data: {snapshot()}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/bench":
            bench.enqueue(payload.get("models", []))
        elif self.path == "/bench/cancel":
            bench.cancel()
        else:
            self.send_error(404)
            return
        self.send_response(204)
        self.end_headers()


class ProxyHandler(BaseHTTPRequestHandler):
    """Forwards everything to Ollama, counting tokens on the way back."""

    def log_message(self, format, *args):
        pass

    def _forward(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "connection", "content-length")}
        model = None
        if body:
            try:
                model = json.loads(body).get("model")
            except (ValueError, AttributeError):
                pass
        gen = model and any(p in self.path for p in ("/api/generate", "/api/chat", "/v1/chat/completions", "/v1/completions"))
        try:
            conn, resp = open_ollama(self.command, self.path, body, headers)
        except OSError as e:
            self.send_error(502, f"Ollama unreachable: {e}")
            return
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        rid = tracker.start(model) if gen else None
        final = None
        try:
            while line := resp.readline():
                self.wfile.write(line)
                self.wfile.flush()
                if rid:
                    final = self._count(rid, line) or final
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()
            if rid:
                tracker.finish(rid, final)

    @staticmethod
    def _count(rid, line):
        """Tick for each content chunk; return Ollama's final stats if this is them."""
        text = line.strip()
        if text.startswith(b"data: "):  # OpenAI-style SSE
            text = text[6:]
            if text == b"[DONE]":
                return None
            try:
                chunk = json.loads(text)
                delta = chunk["choices"][0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning"):
                    tracker.tick(rid)
            except (ValueError, KeyError, IndexError):
                pass
            return None
        try:
            chunk = json.loads(text)
        except ValueError:
            return None
        if not isinstance(chunk, dict):
            return None
        msg = chunk.get("message") or {}
        if chunk.get("response") or chunk.get("thinking") or msg.get("content") or msg.get("thinking"):
            tracker.tick(rid)
        return chunk if chunk.get("done") else None

    do_GET = do_POST = do_DELETE = do_HEAD = do_PUT = _forward


def open_browser(url):
    # Under WSL, hand the URL to Windows so it opens in the normal desktop browser.
    if "microsoft" in os.uname().release.lower() and shutil.which("cmd.exe"):
        subprocess.Popen(["cmd.exe", "/c", "start", "", url], cwd="/mnt/c",
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        webbrowser.open(url)


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    threading.Thread(target=sampler, daemon=True).start()
    try:
        proxy = QuietServer((PROXY_BIND, PROXY_PORT), ProxyHandler)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        print(f"Ollama proxy on {PROXY_BIND}:{PROXY_PORT} -> {OLLAMA}")
    except OSError as e:
        print(f"Proxy not started ({e}); live tok/s will only show benchmarks")
    server = QuietServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"GPU Dash running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, open_browser, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
