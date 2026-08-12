#!/usr/bin/env python3
"""
LLM Fleet Monitor
=================
A lightweight, dependency-free web dashboard to monitor a fleet of LLM inference
nodes over SSH: live GPU / CPU / memory / temperature / power plus per-model
decode, prefill, and TTFT.

Everything host- and model-specific is loaded from a JSON config file at startup
(path via the CONFIG env var, default ./config.json). There is NO hardcoded
infrastructure in this file - point it at your OWN hardware via config.json.

Architecture
------------
Each node is polled over SSH on its own staggered background timer (remote SSH is
slow, so the cadence is gentle and never blocks the others). Each configured model
endpoint is scraped over HTTP (Prometheus /metrics + /v1/models) on its own timer.
The latest result is cached in memory; the browser polls /api/metrics on a short
cadence and reads that cache. Per-GPU temp/power history is kept for sparklines.

READ-ONLY. Every command here only QUERIES state (nvidia-smi query, /proc, hwmon,
HTTP GET on the model /metrics endpoints). Nothing restarts, reconfigures, or kills
anything. Unreachable nodes/models degrade to "stale"; a poller never crashes.

Requires only the Python 3.8+ standard library.
"""

import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# ----------------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------------
# All infrastructure comes from a JSON config file. See config.example.json and
# the README for the full schema. Load it once at startup and normalize defaults.

DEFAULTS = {
    "server": {
        "title": "Sparky Command Center",
        "subtitle": "DGX Spark fleet + GPU boxes - GPU / CPU / mem / temp / power + per-model decode, prefill, TTFT",
        "bind": "0.0.0.0",
        "port": 8890,
        "browser_refresh_ms": 2500,
        # Cumulative token-served tracker. vLLM/llama.cpp expose prompt/generation
        # token counters that RESET on server restart; when enabled we bank the
        # deltas into token_store so a restart never zeroes the running history.
        # Works for every configured model with no extra config. Set false to skip.
        "token_tracking": True,
        "token_store": "data/token_usage.json",
    },
    "ssh": {
        "default_key": "~/.ssh/id_ed25519",
        "connect_timeout": 8,
        # Extra `-o Key=Value` options applied to every ssh call. BatchMode=yes so
        # a node that would prompt for a password fails fast instead of hanging.
        "options": {
            "IdentitiesOnly": "yes",
            "BatchMode": "yes",
            "StrictHostKeyChecking": "accept-new",
        },
    },
    "defaults": {
        "poll_interval": 6.0,        # seconds between node polls
        "model_poll_interval": 6.0,  # seconds between model /metrics scrapes
        "temp_warn": 70,             # deg C -> yellow pill
        "temp_hot": 84,              # deg C -> red pill
    },
}

HIST_LEN = 60  # last N samples per GPU for the sparkline


def _deep_merge(base, override):
    """Return base updated by override (nested dicts merged, not replaced)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    """Load and normalize config.json. Builds the flat NODES + MODELS lists the
    pollers iterate over. Every node/model gets a stable key and inherits the
    global defaults where a per-item value is not set."""
    path = path or os.environ.get("CONFIG", "config.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Expand ${VAR} placeholders from the environment (e.g. auth tokens).
    # Leaves unknown vars as-is rather than failing — configs must stay runnable
    # on hosts that don't export every secret.
    def _expand(obj):
        if isinstance(obj, str):
            return re.sub(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
                lambda m: os.environ.get(m.group(1), m.group(0)),
                obj,
            )
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        return obj

    raw = _expand(raw)

    cfg = {
        "server": _deep_merge(DEFAULTS["server"], raw.get("server")),
        "ssh": _deep_merge(DEFAULTS["ssh"], raw.get("ssh")),
        "defaults": _deep_merge(DEFAULTS["defaults"], raw.get("defaults")),
    }
    cfg["ssh"]["default_key"] = os.path.expanduser(cfg["ssh"]["default_key"])

    d = cfg["defaults"]
    nodes = []
    models = []
    for ni, node in enumerate(raw.get("nodes", [])):
        if node.get("name", "").startswith("_"):  # allow "_comment" pseudo-nodes
            continue
        name = node.get("name") or node.get("host") or f"node{ni + 1}"
        key = node.get("key") or re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower() or f"node{ni + 1}"
        n = {
            "key": key,
            "name": name,
            "host": node["host"],
            "user": node.get("user", os.environ.get("USER", "root")),
            "port": node.get("port"),
            "ssh_key": os.path.expanduser(node["ssh_key"]) if node.get("ssh_key") else cfg["ssh"]["default_key"],
            "jump_host": node.get("jump_host"),
            "jump_user": node.get("jump_user"),
            "poll_interval": float(node.get("poll_interval", d["poll_interval"])),
            "temp_warn": node.get("temp_warn", d["temp_warn"]),
            "temp_hot": node.get("temp_hot", d["temp_hot"]),
        }
        nodes.append(n)
        for mi, m in enumerate(node.get("models", []) or []):
            models.append(_norm_model(m, node=key, node_name=name, idx=mi, defaults=d))

    # Fleet-wide model instances that are not tied to a single node (e.g. run two
    # instances across the cluster and watch both). Rendered in their own section.
    for mi, m in enumerate(raw.get("models", []) or []):
        models.append(_norm_model(m, node=None, node_name=None, idx=mi, defaults=d))

    # Optional fabric-switch panel. Omit the top-level "switch" block to hide it.
    sw_raw = raw.get("switch")
    switch = None
    if sw_raw and sw_raw.get("host"):
        switch = {
            "key": "switch",
            "name": sw_raw.get("name", "Fabric Switch"),
            "badge": sw_raw.get("badge", "RoCE FABRIC"),
            "host": sw_raw["host"],
            "user": sw_raw.get("user", "admin"),
            "port": sw_raw.get("port"),
            "ssh_key": os.path.expanduser(sw_raw["ssh_key"]) if sw_raw.get("ssh_key") else cfg["ssh"]["default_key"],
            "jump_host": sw_raw.get("jump_host"),
            "jump_user": sw_raw.get("jump_user"),
            "poll_interval": float(sw_raw.get("poll_interval", 8.0)),
            "ports": sw_raw.get("ports") or [],   # fabric interfaces to show; [] = auto-detect running
            "temp_warn": sw_raw.get("temp_warn", 55),
            "temp_hot": sw_raw.get("temp_hot", 70),
        }

    # Optional ComfyUI-style image/video lanes. `url` is the only required
    # field; everything else falls back so a minimal entry still renders.
    lanes = []
    for li, ln in enumerate(raw.get("comfy_lanes", []) or []):
        if not ln.get("url"):
            continue
        key = ln.get("key") or re.sub(r"[^a-zA-Z0-9]+", "-",
                                      ln.get("name") or f"lane{li + 1}").strip("-").lower()
        lanes.append({
            "key": key,
            "lane": ln.get("lane") or str(li + 1),
            "name": ln.get("name") or key,
            "host": ln.get("host", ""),
            "url": ln["url"].rstrip("/"),
        })

    cfg["nodes"] = nodes
    cfg["models"] = models
    cfg["switch"] = switch
    cfg["comfy_lanes"] = lanes
    return cfg


def _norm_model(m, node, node_name, idx, defaults):
    """Normalize one model definition. `node` is the owning node key, or None for
    a fleet-wide instance."""
    prefix = node or "fleet"
    mkey = m.get("key") or f"{prefix}:m{idx}"
    return {
        "key": mkey,
        "node": node,
        "node_name": node_name,
        "group": m.get("group") or ("Fleet models" if node is None else None),
        "label": m.get("label") or m.get("model") or mkey,
        "endpoint": m["endpoint"].rstrip("/"),
        "port": m.get("port"),
        "model": m.get("model"),   # optional served-alias to prefer on /v1/models
        "gpus": m.get("gpus"),      # optional human label, e.g. "GPU 0-1"
        "poll_interval": float(m.get("poll_interval", defaults["model_poll_interval"])),
    }


CFG = None  # populated in main()


# ----------------------------------------------------------------------------
# Shared state (one slice per node/model; each poller writes its own slice)
# ----------------------------------------------------------------------------
_lock = threading.Lock()
STATE = {"nodes": {}, "models": {}, "switch": None, "comfy": {}}
_hist = {}  # sparkline history, keyed e.g. "node:<nodekey>:<gpuindex>:temp"
_iface_prev = {}  # switch interface byte counters for throughput deltas: (swkey,port) -> (ts,rx,tx)
_port_rate = {}   # switch static port link rate cache: (swkey,port) -> "100Gbps"


def _push_hist(key, value):
    if value is None:
        return
    dq = _hist.setdefault(key, [])
    dq.append(round(float(value), 1))
    if len(dq) > HIST_LEN:
        del dq[: len(dq) - HIST_LEN]


def _run(cmd, timeout, cwd=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as e:  # noqa
        return 1, "", str(e)


def _num(v):
    try:
        return float(v)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Node poller (one SSH round-trip per node: GPUs + CPU temps + system memory)
# ----------------------------------------------------------------------------
# nvidia-smi query for every GPU, CPU temps from hwmon (AMD k10temp / Intel
# coretemp / ARM cpu_thermal / etc.), and system RAM from `free`. All READ-ONLY.
_NODE_REMOTE = (
    'nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,'
    'utilization.gpu,memory.used,memory.total,fan.speed,clocks.gr,clocks.mem '
    '--format=csv,noheader,nounits 2>/dev/null; '
    'echo PIPECPU; '
    'for h in /sys/class/hwmon/hwmon*; do n=$(cat "$h/name" 2>/dev/null); '
    'case "$n" in k10temp|coretemp|cpu_thermal|zenpower|nct6*) '
    'cat "$h"/temp*_input 2>/dev/null;; esac; done; '
    'echo PIPEMEM; '
    'free -m | awk "/^Mem:/{print \\$2, \\$3}"'
)


def _ssh_flags(node):
    """Build the shared ssh flag list for a node: identity, timeout, options,
    optional port, optional ProxyJump bastion. Flags must be SEPARATE argv items."""
    flags = ["-i", node["ssh_key"], "-o", f"ConnectTimeout={CFG['ssh']['connect_timeout']}"]
    for k, v in (CFG["ssh"].get("options") or {}).items():
        flags += ["-o", f"{k}={v}"]
    if node.get("port"):
        flags += ["-p", str(node["port"])]
    if node.get("jump_host"):
        jump = node["jump_host"]
        if node.get("jump_user"):
            jump = f"{node['jump_user']}@{jump}"
        flags += ["-J", jump]  # standard SSH ProxyJump / bastion
    return flags


def _node_cmd(node):
    return ["ssh"] + _ssh_flags(node) + [f"{node['user']}@{node['host']}", _NODE_REMOTE]


def poll_node(node):
    res = {
        "key": node["key"], "name": node["name"], "reachable": False,
        "ts": time.time(), "err": None,
        "gpus": [], "cpu_temp": None, "cpu_temps": [],
        "mem_total_mb": None, "mem_used_mb": None, "mem_pct": None,
        "temp_warn": node["temp_warn"], "temp_hot": node["temp_hot"],
    }
    timeout = max(10, node["poll_interval"] * 2 + 4)
    rc, out, err = _run(_node_cmd(node), timeout=timeout)
    if rc != 0 or not out.strip():
        res["err"] = (err or "no output").strip()[:140]
        return res
    try:
        gpu_part, _, rest = out.partition("PIPECPU")
        cpu_part, _, mem_part = rest.partition("PIPEMEM")

        for line in gpu_part.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 11:
                continue
            idx = int(f[0])
            fan = _num(f[8]) if f[8].replace(".", "").isdigit() else None
            g = {
                "index": idx,
                "name": re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", f[1]),
                "temp": _num(f[2]), "power": _num(f[3]), "power_limit": _num(f[4]),
                "util": _num(f[5]),
                "mem_used_mb": _num(f[6]), "mem_total_mb": _num(f[7]),
                "fan": fan, "gr_clock": _num(f[9]), "mem_clock": _num(f[10]),
            }
            if g["mem_total_mb"]:
                g["mem_pct"] = round((g["mem_used_mb"] or 0) / g["mem_total_mb"] * 100.0, 1)
            res["gpus"].append(g)
            _push_hist(f"node:{node['key']}:{idx}:temp", g["temp"])
            _push_hist(f"node:{node['key']}:{idx}:power", g["power"])

        cpu_vals = [int(x) / 1000.0 for x in cpu_part.strip().splitlines() if x.strip().isdigit()]
        if cpu_vals:
            res["cpu_temps"] = [round(v, 1) for v in cpu_vals]
            res["cpu_temp"] = res["cpu_temps"][0]

        ml = mem_part.strip().splitlines()
        if ml:
            parts = ml[0].split()
            if len(parts) >= 2:
                res["mem_total_mb"] = _num(parts[0])
                res["mem_used_mb"] = _num(parts[1])
                if res["mem_total_mb"]:
                    res["mem_pct"] = round(res["mem_used_mb"] / res["mem_total_mb"] * 100.0, 1)

        res["reachable"] = bool(res["gpus"]) or res["mem_total_mb"] is not None
        if not res["reachable"]:
            res["err"] = "no GPU/mem reading"
    except Exception as e:  # noqa
        res["err"] = f"parse: {e}"[:140]
    return res


# ----------------------------------------------------------------------------
# Optional fabric-switch poller (MikroTik / RouterOS over SSH). READ-ONLY.
# ----------------------------------------------------------------------------
# The switch is reached over plain SSH using the same key/jump-host mechanism as
# nodes. RouterOS runs the command passed as the SSH argv and prints the result.
# All commands only QUERY state (`print` / `monitor once`). Omit the config
# "switch" block entirely to disable this panel.
def _switch_ssh(sw, statement):
    return ["ssh"] + _ssh_flags(sw) + [f"{sw['user']}@{sw['host']}", statement]


def _strip_thousands(s):
    """RouterOS prints byte counters with space thousands separators."""
    return s.replace(" ", "")


def _parse_health(text):
    """Parse `/system health print` value rows: "  #  NAME  VALUE  TYPE"."""
    h = {}
    for line in text.splitlines():
        mm = re.match(r"\s*\d+\s+([a-z0-9\-]+)\s+([0-9.]+|ok|fail|critical|warning)\b", line)
        if mm:
            h[mm.group(1)] = mm.group(2)
    return h


def _parse_resource(text):
    """Parse `/system resource print` "key: value" rows."""
    r = {}
    for line in text.splitlines():
        mm = re.match(r"\s*([a-z0-9\-]+):\s+(.+?)\s*$", line)
        if mm:
            r[mm.group(1)] = mm.group(2).strip()
    return r


def _parse_iface_stats(text):
    """Best-effort parse of `/interface print stats` rows -> {name: (rx, tx)}.
    Byte counters use single-space thousands separators, columns use 2+ spaces."""
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or not s[0].isdigit():
            continue  # data rows start with an index number
        m = re.match(r"\d+\s+(?:[A-Z]{1,3}\s+)?([A-Za-z][\w\-]*)\s+(.*)$", s)
        if not m:
            continue
        cols = re.split(r"\s{2,}", m.group(2).strip())
        vals = [int(_strip_thousands(c)) for c in cols if _strip_thousands(c).isdigit()]
        if len(vals) >= 2:
            out[m.group(1)] = (vals[0], vals[1])
    return out


def poll_switch(sw):
    res = {"reachable": False, "name": sw["name"], "badge": sw["badge"],
           "ts": time.time(), "err": None, "health": {}, "resource": {},
           "ports": [], "total_bps": 0,
           "temp_warn": sw["temp_warn"], "temp_hot": sw["temp_hot"]}
    to = max(20, sw["poll_interval"] * 2 + 6)

    # 1) health (temps / fans / psu)
    rc, out, err = _run(_switch_ssh(sw, "/system health print"), timeout=to)
    if rc != 0 or "NAME" not in out:
        res["err"] = (err or out or "switch unreachable").strip()[:140]
        return res
    res["health"] = _parse_health(out)
    res["reachable"] = True

    # 2) resource (version / uptime / cpu)
    rc, out, err = _run(_switch_ssh(sw, "/system resource print"), timeout=to)
    if rc == 0 and "version" in out:
        res["resource"] = _parse_resource(out)

    # 3) interface byte counters -> live per-port throughput
    rc, out, err = _run(_switch_ssh(sw, "/interface print stats where running"), timeout=to)
    now = time.time()
    stats = _parse_iface_stats(out) if rc == 0 else {}
    names = sw["ports"] or list(stats.keys())
    ports = {}
    for p in names:
        d = stats.get(p)
        entry = {"name": p, "running": d is not None, "rx_bps": 0, "tx_bps": 0,
                 "rate": _port_rate.get((sw["key"], p))}
        if d:
            rx, tx = d
            prev = _iface_prev.get((sw["key"], p))
            if prev:
                dt = now - prev[0]
                if dt > 0:
                    entry["rx_bps"] = max(0, (rx - prev[1]) * 8 / dt)
                    entry["tx_bps"] = max(0, (tx - prev[2]) * 8 / dt)
            _iface_prev[(sw["key"], p)] = (now, rx, tx)
        ports[p] = entry

    # Fetch each running port's static link rate once (one monitor per cycle so we
    # never hammer the switch).
    for p in names:
        if ports[p]["running"] and _port_rate.get((sw["key"], p)) is None:
            rc2, out2, _ = _run(_switch_ssh(sw, f"/interface ethernet monitor {p} once"), timeout=to)
            if rc2 == 0:
                rm = re.search(r"\brate:\s*([0-9A-Za-z]+)", out2)
                if rm:
                    _port_rate[(sw["key"], p)] = rm.group(1)
                    ports[p]["rate"] = rm.group(1)
            break

    res["ports"] = [ports[p] for p in names]
    res["total_bps"] = sum(pp["rx_bps"] + pp["tx_bps"] for pp in res["ports"])
    return res


# ----------------------------------------------------------------------------
# Per-model inference-server poller (vLLM + llama.cpp Prometheus /metrics)
# ----------------------------------------------------------------------------
# Previous counter snapshots per model key -> (ts, prompt_tokens, gen_tokens,
# ttft_sum, ttft_count) so we can compute decode/prefill tok/s as a RATE.
_model_prev = {}


def _http_get(url, timeout=6):
    """Tiny stdlib GET. Returns (ok, text). Never raises (down/idle is normal)."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "llm-fleet-monitor"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, r.read().decode("utf-8", "replace")
    except Exception:  # noqa
        return False, ""


def _prom_parse(text):
    """Parse a Prometheus exposition text body into {metric_name: value}.
    Labels are stripped - for single-engine servers each base metric appears once,
    so the last value wins. Histogram _sum/_count keep their suffixes."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$", line)
        if not m:
            continue
        v = _num(m.group(3))
        if v is not None:
            out[m.group(1)] = v
    return out


def _model_id(endpoint, prefer=None, timeout=6):
    """Live model id from /v1/models. If `prefer` is set and served here, use it;
    otherwise fall back to data[0].id. Empty string if unavailable."""
    ok, body = _http_get(endpoint + "/v1/models", timeout=timeout)
    if not ok:
        return ""
    try:
        data = json.loads(body).get("data", [])
        ids = [str(x.get("id", "") or "") for x in data]
        if prefer and prefer in ids:
            return prefer
        if ids:
            return ids[0]
    except Exception:  # noqa
        pass
    return ""


def _kv_pct(val):
    """vLLM kv/gpu cache usage is a 0-1 fraction (despite '_perc'); llama.cpp
    kv_cache_usage_ratio is also 0-1. Normalize anything <=1 to a percentage."""
    if val is None:
        return None
    return round(val * 100.0, 1) if val <= 1.0 else round(val, 1)


# ----------------------------------------------------------------------------
# Cumulative token tracker (banks prompt/generation counters across restarts)
# ----------------------------------------------------------------------------
# vLLM (vllm:prompt_tokens_total / vllm:generation_tokens_total) and llama.cpp
# (llamacpp:prompt_tokens_total / llamacpp:tokens_predicted_total) expose
# monotonic token counters that RESET to 0 whenever the inference server
# restarts. poll_model already reads these for the tok/s rate; here we bank the
# deltas into a persistent JSON store so a restart never zeroes the running
# history, plus a per-day bucket that rolls over at local midnight.
TOKEN_STORE = None            # abs path, set in main() when token_tracking is on
_tokens = {}                  # model key -> banked record
_tokens_lock = threading.Lock()
_tokens_dirty = False
_tokens_last_save = 0.0
TOKEN_SAVE_EVERY = 25.0       # seconds; throttle disk writes


def _load_tokens():
    global _tokens
    if not TOKEN_STORE:
        return
    try:
        with open(TOKEN_STORE, "r", encoding="utf-8") as f:
            _tokens = json.load(f)
    except Exception:  # noqa - missing/corrupt store just starts fresh
        _tokens = {}


def _save_tokens(force=False):
    """Atomically persist the token store, throttled to TOKEN_SAVE_EVERY."""
    global _tokens_dirty, _tokens_last_save
    if not TOKEN_STORE:
        return
    now = time.time()
    with _tokens_lock:
        if not _tokens_dirty or (not force and now - _tokens_last_save < TOKEN_SAVE_EVERY):
            return
        snap = json.dumps(_tokens, indent=2)
        _tokens_dirty = False
        _tokens_last_save = now
    try:
        d = os.path.dirname(os.path.abspath(TOKEN_STORE))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = TOKEN_STORE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(snap)
        os.replace(tmp, TOKEN_STORE)
    except Exception:  # noqa - never let a write error take down the poller
        pass


def _bank_counter(rec, cur, f_total, f_last, f_today):
    """Accumulate one monotonic-with-resets counter into a total + today bucket."""
    last = rec.get(f_last)
    if last is None:
        # First observation: seed the total with the current running-session value
        # so the tracker shows real numbers immediately, but start today fresh
        # (we cannot attribute the pre-existing count to today).
        rec[f_total] = cur
        rec[f_last] = cur
        rec.setdefault(f_today, 0.0)
        return
    delta = (cur - last) if cur >= last else cur   # cur < last => server restarted
    rec[f_total] = rec.get(f_total, 0.0) + delta
    rec[f_today] = rec.get(f_today, 0.0) + delta
    rec[f_last] = cur


def bank_tokens(m, prompt_tok, gen_tok):
    """Bank a model's cumulative token counters. Called from poll_model. No-op
    when token tracking is disabled or the server exposes no token counters."""
    global _tokens_dirty
    if not TOKEN_STORE or (prompt_tok is None and gen_tok is None):
        return
    today = time.strftime("%Y-%m-%d", time.localtime())
    with _tokens_lock:
        rec = _tokens.setdefault(m["key"], {})
        rec["label"] = m["label"]
        rec["node"] = m.get("node_name") or m.get("node")
        rec["gpus"] = m.get("gpus")
        if rec.get("today_date") != today:      # roll the day bucket at midnight
            rec["today_date"] = today
            rec["today_prompt"] = 0.0
            rec["today_gen"] = 0.0
        if prompt_tok is not None:
            _bank_counter(rec, prompt_tok, "total_prompt", "last_prompt", "today_prompt")
        if gen_tok is not None:
            _bank_counter(rec, gen_tok, "total_gen", "last_gen", "today_gen")
        rec["total_tokens"] = rec.get("total_prompt", 0.0) + rec.get("total_gen", 0.0)
        rec["today_tokens"] = rec.get("today_prompt", 0.0) + rec.get("today_gen", 0.0)
        rec["ts"] = time.time()
        _tokens_dirty = True
    _save_tokens()


def tokens_snapshot():
    """Read-only view of the banked token store for the API/UI. Copies under the
    token lock, then reads liveness under the node lock - never both at once."""
    if not TOKEN_STORE:
        return {"enabled": False, "models": [], "total": 0, "today": 0}
    with _tokens_lock:
        recs = [dict(v, key=k) for k, v in _tokens.items()]
    with _lock:
        for r in recs:
            st = STATE["models"].get(r["key"]) or {}
            r["reachable"] = bool(st.get("reachable"))
    recs.sort(key=lambda r: r.get("total_tokens", 0), reverse=True)
    return {
        "enabled": True,
        "models": recs,
        "total": sum(r.get("total_tokens", 0) for r in recs),
        "today": sum(r.get("today_tokens", 0) for r in recs),
    }


def poll_model(m):
    """Scrape one model server's /metrics + /v1/models. Computes decode/prefill
    tok/s as deltas between polls. Handles both vLLM and llama.cpp metric names."""
    res = {
        "key": m["key"], "node": m["node"], "label": m["label"],
        "port": m["port"], "gpus": m.get("gpus"),
        "reachable": False, "engine": None, "model": None,
        "decode_tps": None, "prefill_tps": None, "ttft_ms": None,
        "kv_pct": None, "running": None, "waiting": None,
        "ts": time.time(), "err": None,
    }
    ok, body = _http_get(m["endpoint"] + "/metrics", timeout=6)
    if not ok or not body.strip():
        res["err"] = "down / no /metrics"
        return res
    p = _prom_parse(body)
    now = time.time()

    is_vllm = any(k.startswith("vllm:") for k in p)
    is_llama = any(k.startswith("llamacpp:") for k in p)

    prompt_tok = gen_tok = ttft_sum = ttft_cnt = None

    if is_vllm:
        res["engine"] = "vLLM"
        prompt_tok = p.get("vllm:prompt_tokens_total")
        gen_tok = p.get("vllm:generation_tokens_total")
        ttft_sum = p.get("vllm:time_to_first_token_seconds_sum")
        ttft_cnt = p.get("vllm:time_to_first_token_seconds_count")
        kv = p.get("vllm:kv_cache_usage_perc")
        if kv is None:
            kv = p.get("vllm:gpu_cache_usage_perc")
        res["kv_pct"] = _kv_pct(kv)
        if "vllm:num_requests_running" in p:
            res["running"] = int(p["vllm:num_requests_running"])
        if "vllm:num_requests_waiting" in p:
            res["waiting"] = int(p["vllm:num_requests_waiting"])
    elif is_llama:
        res["engine"] = "llama.cpp"
        prompt_tok = p.get("llamacpp:prompt_tokens_total")
        gen_tok = p.get("llamacpp:tokens_predicted_total")
        res["kv_pct"] = _kv_pct(p.get("llamacpp:kv_cache_usage_ratio"))
        if "llamacpp:requests_processing" in p:
            res["running"] = int(p["llamacpp:requests_processing"])
        if "llamacpp:requests_deferred" in p:
            res["waiting"] = int(p["llamacpp:requests_deferred"])
        # llama.cpp exposes instantaneous rates directly; use as a fallback.
        inst_pred = p.get("llamacpp:predicted_tokens_seconds")
        inst_proc = p.get("llamacpp:prompt_tokens_seconds")
        if inst_pred is not None:
            res["decode_tps"] = round(inst_pred, 1)
        if inst_proc is not None:
            res["prefill_tps"] = round(inst_proc, 1)
    else:
        res["err"] = "unknown /metrics format"
        return res

    res["reachable"] = True

    # bank cumulative token counters into the persistent tracker (survives the
    # inference server restarting, which zeroes these counters)
    bank_tokens(m, prompt_tok, gen_tok)

    # rate computation from the delta vs the previous poll
    prev = _model_prev.get(m["key"])
    if prev and prompt_tok is not None and gen_tok is not None:
        dt = now - prev[0]
        if dt > 0:
            d_gen = (gen_tok - prev[2]) if prev[2] is not None else 0
            d_prompt = (prompt_tok - prev[1]) if prev[1] is not None else 0
            if d_gen >= 0:
                res["decode_tps"] = round(d_gen / dt, 1)
            if d_prompt >= 0:
                res["prefill_tps"] = round(d_prompt / dt, 1)
            if ttft_sum is not None and ttft_cnt is not None and prev[4] is not None:
                d_cnt = ttft_cnt - prev[4]
                d_sum = ttft_sum - prev[3]
                if d_cnt > 0 and d_sum >= 0:
                    res["ttft_ms"] = round(d_sum / d_cnt * 1000.0, 1)
    _model_prev[m["key"]] = (now, prompt_tok, gen_tok, ttft_sum, ttft_cnt)

    # default idle-but-up rates to 0 so the UI shows a number, not a dash
    if res["decode_tps"] is None:
        res["decode_tps"] = 0.0
    if res["prefill_tps"] is None:
        res["prefill_tps"] = 0.0

    # cumulative TTFT avg as a fallback when no window data yet (vLLM only)
    if res["ttft_ms"] is None and ttft_sum is not None and ttft_cnt and ttft_cnt > 0:
        res["ttft_ms"] = round(ttft_sum / ttft_cnt * 1000.0, 1)

    mid = _model_id(m["endpoint"], prefer=m.get("model"), timeout=6)
    res["model"] = mid or m["label"]
    return res


# ----------------------------------------------------------------------------
# Background pollers (one thread per node/model, staggered starts)
# ----------------------------------------------------------------------------
def _node_loop(node, offset):
    time.sleep(offset)
    while True:
        try:
            r = poll_node(node)
        except Exception as e:  # noqa
            r = {"key": node["key"], "name": node["name"], "reachable": False,
                 "ts": time.time(), "err": str(e)[:140], "gpus": [],
                 "temp_warn": node["temp_warn"], "temp_hot": node["temp_hot"]}
        with _lock:
            STATE["nodes"][node["key"]] = r
        time.sleep(node["poll_interval"])


def _model_loop(m, offset):
    time.sleep(offset)
    while True:
        try:
            r = poll_model(m)
        except Exception as e:  # noqa
            r = {"key": m["key"], "node": m["node"], "label": m["label"],
                 "port": m["port"], "gpus": m.get("gpus"), "reachable": False,
                 "ts": time.time(), "err": str(e)[:140]}
        with _lock:
            STATE["models"][m["key"]] = r
        time.sleep(m["poll_interval"])


def _switch_loop(sw):
    time.sleep(0.5)
    while True:
        try:
            r = poll_switch(sw)
        except Exception as e:  # noqa
            r = {"reachable": False, "name": sw["name"], "badge": sw["badge"],
                 "ts": time.time(), "err": str(e)[:140], "health": {}, "resource": {},
                 "ports": [], "total_bps": 0,
                 "temp_warn": sw["temp_warn"], "temp_hot": sw["temp_hot"]}
        with _lock:
            STATE["switch"] = r
        time.sleep(sw["poll_interval"])


def poll_comfy(lane):
    """Scrape one ComfyUI-style image/video lane.

    /system_stats gives liveness + device VRAM; /queue gives running/pending.
    A lane that is down is a normal state, not an error. Polled server-side
    because ComfyUI sends no CORS headers, so a browser fetch cannot read it.
    """
    out = {"key": lane["key"], "lane": lane.get("lane") or lane["key"],
           "name": lane.get("name") or lane["key"], "host": lane.get("host", ""),
           "url": lane["url"], "reachable": False, "ts": time.time(),
           "vram_total": None, "vram_free": None, "vram_used": None,
           "version": None, "running": 0, "pending": 0, "busy": False}
    ok, body = _http_get(lane["url"] + "/system_stats", timeout=5)
    if not ok:
        return out
    try:
        d = json.loads(body)
    except Exception:
        return out
    out["reachable"] = True
    out["version"] = (d.get("system") or {}).get("comfyui_version")
    devs = d.get("devices") or []
    if devs:
        tot, free = devs[0].get("vram_total"), devs[0].get("vram_free")
        if isinstance(tot, (int, float)) and isinstance(free, (int, float)):
            out["vram_total"], out["vram_free"] = tot, free
            out["vram_used"] = max(0, tot - free)
    ok2, qbody = _http_get(lane["url"] + "/queue", timeout=5)
    if ok2:
        try:
            q = json.loads(qbody)
            out["running"] = len(q.get("queue_running") or [])
            out["pending"] = len(q.get("queue_pending") or [])
            out["busy"] = out["running"] > 0
        except Exception:
            pass
    return out


def _comfy_loop(lane, delay):
    """Poll a lane forever, latching a HIGH-WATER MARK across polls.

    Peak is the number that matters for capacity planning: these models load
    their components one at a time, so idle understates and the sum of the
    parts overstates. Only a real render shows the true ceiling, and it lands
    between polls, so it is latched here rather than sampled for. `peak_busy`
    latches only while a job is in flight, which keeps a noisy neighbour on the
    same box out of the model's number.
    """
    ttl = float((CFG.get("server") or {}).get("comfy_poll_seconds") or 4.0)
    time.sleep(delay)
    while True:
        try:
            r = poll_comfy(lane)
        except Exception as e:  # noqa - a down lane must never kill the poller
            r = {"key": lane["key"], "lane": lane.get("lane") or lane["key"],
                 "name": lane.get("name") or lane["key"], "host": lane.get("host", ""),
                 "url": lane["url"], "reachable": False, "ts": time.time(),
                 "err": str(e)[:140], "running": 0, "pending": 0, "busy": False}
        with _lock:
            prev = STATE["comfy"].get(lane["key"]) or {}
            for fld, only_busy in (("peak_used", False), ("peak_busy", True)):
                cur = r.get("vram_used")
                if only_busy and not r.get("busy"):
                    cur = None
                old = prev.get(fld)
                r[fld] = max(old or 0, cur) if cur is not None else old
            STATE["comfy"][lane["key"]] = r
        time.sleep(ttl)


def comfy_snapshot():
    with _lock:
        return {"lanes": [STATE["comfy"].get(l["key"], {})
                          for l in (CFG.get("comfy_lanes") or [])]}


def start_pollers():
    for i, node in enumerate(CFG["nodes"]):
        STATE["nodes"][node["key"]] = {
            "key": node["key"], "name": node["name"], "reachable": False,
            "ts": 0, "err": "warming up", "gpus": [],
            "temp_warn": node["temp_warn"], "temp_hot": node["temp_hot"]}
        threading.Thread(target=_node_loop, args=(node, i * 0.8), daemon=True).start()
    for i, m in enumerate(CFG["models"]):
        STATE["models"][m["key"]] = {
            "key": m["key"], "node": m["node"], "label": m["label"],
            "port": m["port"], "gpus": m.get("gpus"), "reachable": False,
            "ts": 0, "err": "warming up"}
        threading.Thread(target=_model_loop, args=(m, i * 0.5), daemon=True).start()
    if CFG.get("switch"):
        sw = CFG["switch"]
        STATE["switch"] = {"reachable": False, "name": sw["name"], "badge": sw["badge"],
                           "ts": 0, "err": "warming up", "health": {}, "resource": {},
                           "ports": [], "total_bps": 0,
                           "temp_warn": sw["temp_warn"], "temp_hot": sw["temp_hot"]}
        threading.Thread(target=_switch_loop, args=(sw,), daemon=True).start()
    for i, ln in enumerate(CFG.get("comfy_lanes") or []):
        STATE["comfy"][ln["key"]] = {
            "key": ln["key"], "lane": ln.get("lane") or ln["key"],
            "name": ln.get("name") or ln["key"], "host": ln.get("host", ""),
            "url": ln["url"], "reachable": False, "ts": 0,
            "err": "warming up", "running": 0, "pending": 0, "busy": False}
        threading.Thread(target=_comfy_loop, args=(ln, 0.6 + i * 0.4),
                         daemon=True).start()


# ----------------------------------------------------------------------------
# Aggregate / snapshot
# ----------------------------------------------------------------------------
def snapshot():
    with _lock:
        nodes = [dict(STATE["nodes"][n["key"]]) for n in CFG["nodes"]]
        models_by_node = {}
        fleet_models = []
        for m in CFG["models"]:
            st = STATE["models"].get(m["key"])
            if not st:
                continue
            item = dict(st)
            item["group"] = m.get("group")
            if m["node"]:
                models_by_node.setdefault(m["node"], []).append(item)
            else:
                fleet_models.append(item)
        switch = dict(STATE["switch"]) if STATE.get("switch") else None
        hist = {k: list(v) for k, v in _hist.items()}

    # attach each node's model perf cards
    for n in nodes:
        n["models"] = models_by_node.get(n["key"], [])

    # fleet aggregates
    gpu_count = 0
    total_power = 0.0
    hottest = {"unit": None, "temp": -1}
    all_ok = True
    for n in nodes:
        if not n.get("reachable"):
            all_ok = False
        for g in n.get("gpus", []):
            gpu_count += 1
            if g.get("power"):
                total_power += g["power"]
            if g.get("temp") is not None and g["temp"] > hottest["temp"]:
                hottest = {"unit": f"{n['name']} GPU{g['index']}", "temp": g["temp"]}

    return {
        "ts": time.time(),
        "title": CFG["server"]["title"],
        "subtitle": CFG["server"]["subtitle"],
        "browser_refresh_ms": CFG["server"]["browser_refresh_ms"],
        "switch": switch,
        "nodes": nodes,
        "fleet_models": fleet_models,
        "tokens": tokens_snapshot(),
        "history": hist,
        "agg": {
            "gpu_count": gpu_count,
            "total_power": round(total_power),
            "hottest_unit": hottest["unit"],
            "hottest_temp": hottest["temp"] if hottest["temp"] >= 0 else None,
            "all_ok": all_ok,
        },
    }


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence default logging
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_GET(self):
        if self.path.startswith("/api/metrics"):
            self._send(200, json.dumps(snapshot()), "application/json")
            return
        if self.path.startswith("/api/tokens"):
            self._send(200, json.dumps(tokens_snapshot()), "application/json")
            return
        if self.path.startswith("/api/comfy"):
            self._send(200, json.dumps(comfy_snapshot()), "application/json")
            return
        if self.path == "/healthz":
            self._send(200, "ok", "text/plain")
            return
        self._send(200, PAGE)


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sparky Command Center</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root{
    /* Burnt-orange on scorched brown. The neutrals carry a red/brown bias on
       purpose so nothing reads as generic grey next to the orange. Semantic
       green/yellow/red stay clearly separate in hue from the accent so a
       warning pill never gets mistaken for an accent. */
    --bg:#0c0805; --bg2:#150e08; --card:rgba(40,27,17,0.62); --border:rgba(224,132,42,0.20);
    --txt:#f7f1ea; --dim:#a6907c; --accent:#ff7a1a; --accent2:#ffb454;
    --green:#5ec46a; --yellow:#f0c419; --red:#ff5347;
    --display:'Futura','Avenir Next Condensed','Oswald','Segoe UI',sans-serif;
    --mono:'SF Mono','Menlo','JetBrains Mono',ui-monospace,monospace;
    /* Themeable extras. --accent-rgb exists so every rgba() tint follows the
       accent instead of hard-coding orange; --on-accent is the text colour that
       sits ON a filled accent button, which has to flip on light themes. */
    --accent-rgb:255,122,26; --on-accent:#1a0d04;
    --glow1:rgba(255,122,26,0.13); --glow2:rgba(140,60,20,0.16);
    --grad-a:#0a0603; --grad-b:#140d07;
  }

  /* ── Themes ──────────────────────────────────────────────────────────────
     Sunset is :root above (the default). Each theme below only redefines
     tokens, never component rules, so a new theme cannot break layout. */

  /* BREEZE — the cool blue original, rebuilt from scratch: the pre-Sunset CSS
     is not in any commit, so this is a reconstruction, not a restoration. */
  :root[data-theme="breeze"]{
    --bg:#06090f; --bg2:#0b1220; --card:rgba(18,30,48,0.62); --border:rgba(90,170,255,0.20);
    --txt:#eaf2fb; --dim:#8296ad; --accent:#38bdf8; --accent2:#7dd3fc;
    --green:#4ade80; --yellow:#fbbf24; --red:#f87171;
    --accent-rgb:56,189,248; --on-accent:#04121d;
    --glow1:rgba(56,189,248,0.13); --glow2:rgba(30,80,160,0.18);
    --grad-a:#04070d; --grad-b:#0a111e;
  }
  /* LIGHT — inverted. Neutrals carry a slight cool bias so white cards on a
     white page still separate, and the accent darkens to stay legible. */
  :root[data-theme="light"]{
    --bg:#f4f6f9; --bg2:#ffffff; --card:rgba(255,255,255,0.86); --border:rgba(15,35,60,0.14);
    --txt:#14202e; --dim:#5b6b7d; --accent:#0b6fd4; --accent2:#2f92f0;
    --green:#17914a; --yellow:#a86a00; --red:#cc2b2b;
    --accent-rgb:11,111,212; --on-accent:#ffffff;
    --glow1:rgba(11,111,212,0.07); --glow2:rgba(120,150,190,0.10);
    --grad-a:#eef2f7; --grad-b:#ffffff;
  }
  /* DARK — true neutral black. The accent goes silver so the UI reads as
     monochrome; semantic green/yellow/red stay saturated to carry all state. */
  :root[data-theme="dark"]{
    --bg:#050505; --bg2:#0e0e0e; --card:rgba(26,26,26,0.66); --border:rgba(255,255,255,0.13);
    --txt:#f2f2f2; --dim:#8c8c8c; --accent:#e6e6e6; --accent2:#b3b3b3;
    --green:#5ec46a; --yellow:#f0c419; --red:#ff5347;
    --accent-rgb:230,230,230; --on-accent:#0a0a0a;
    --glow1:rgba(255,255,255,0.05); --glow2:rgba(255,255,255,0.03);
    --grad-a:#000000; --grad-b:#0d0d0d;
  }
  /* MATRIX — phosphor green on black, and the display face drops to mono so
     the headings read like a terminal rather than a poster. */
  :root[data-theme="matrix"]{
    --bg:#000000; --bg2:#04120a; --card:rgba(6,26,15,0.66); --border:rgba(0,255,102,0.22);
    --txt:#ccffdd; --dim:#58a97a; --accent:#00ff66; --accent2:#7dffb0;
    --green:#00ff66; --yellow:#d7ff5a; --red:#ff4d4d;
    --accent-rgb:0,255,102; --on-accent:#001b0c;
    --glow1:rgba(0,255,102,0.10); --glow2:rgba(0,120,50,0.14);
    --grad-a:#000000; --grad-b:#031008;
    --display:'SF Mono','Menlo','JetBrains Mono',ui-monospace,monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{
    font-family:'Avenir Next','SF Pro Text','Segoe UI',system-ui,sans-serif;
    background:
      radial-gradient(1100px 600px at 12% -10%, var(--glow1), transparent 60%),
      radial-gradient(900px 500px at 90% 0%, var(--glow2), transparent 60%),
      linear-gradient(160deg,var(--grad-a) 0%,var(--grad-b) 100%);
    color:var(--txt); min-height:100vh; padding:24px 32px 60px;
    letter-spacing:0.01em;
  }
  header{
    display:flex;justify-content:space-between;align-items:center;
    margin-bottom:18px;padding-bottom:16px;border-bottom:1px solid var(--border);
    flex-wrap:wrap;gap:14px;
  }
  h1{
    font-size:28px;font-weight:800;letter-spacing:0.02em;
    background:linear-gradient(90deg,var(--accent) 0%,var(--accent2) 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  }
  .pulse{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--green);
         box-shadow:0 0 12px var(--green);margin-right:8px;animation:p 2s infinite}
  .pulse.bad{background:var(--red);box-shadow:0 0 12px var(--red)}
  @keyframes p{0%,100%{opacity:1}50%{opacity:0.4}}
  .meta{font-size:13px;color:var(--dim);text-align:right;line-height:1.6}
  .meta .ts{color:var(--accent);font-variant-numeric:tabular-nums}

  /* summary strip */
  .summary{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:26px}
  @media(max-width:760px){.summary{grid-template-columns:repeat(2,1fr)}}
  .scard{
    background:var(--card);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
    border:1px solid var(--border);border-radius:14px;padding:14px 18px;
    box-shadow:0 8px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .scard .k{font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:var(--dim)}
  .scard .v{font-size:30px;font-weight:800;font-variant-numeric:tabular-nums;margin-top:6px;line-height:1}
  .scard .v .u{font-size:14px;color:var(--dim);font-weight:600;margin-left:4px}
  .scard .v.neon{color:var(--accent)} .scard .v.violet{color:var(--accent2)}
  .scard .v.red{color:var(--red)} .scard .v.green{color:var(--green)}

  .section-h{font-size:13px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;
    color:var(--accent2);margin:8px 0 14px;display:flex;align-items:center;gap:10px}
  .section-h.acc{color:var(--accent)}
  .section-h .ln{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}

  h1,.section-h,.card h2,.scard .k,.modbar{font-family:var(--display)}
  .scard .v,.row .val,.meta .ts,.pill{font-family:var(--mono);font-variant-numeric:tabular-nums}

  /* -- Modules: collapse + rearrange --------------------------------------
     Each panel is wrapped in .mod with its own .modbar handle. Collapse and
     order are per-browser (localStorage) so the server stays stateless and a
     wrecked layout is fixed by clearing two keys. */
  .mod{display:block;margin-bottom:6px}
  .modbar{display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none;
    font-size:12px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;
    color:var(--accent2);padding:6px 8px;margin:6px 0 2px -8px;border-radius:8px;
    transition:background .15s}
  .modbar:hover{background:rgba(var(--accent-rgb),0.07)}
  .modbar .chev{display:inline-block;transition:transform .18s;font-size:11px;opacity:.85}
  .modbar .ln{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}
  .modbar .grip{display:none;cursor:grab;letter-spacing:-2px;color:var(--accent);
    opacity:.75;font-size:14px}
  .mod.collapsed .chev{transform:rotate(-90deg)}
  .mod.collapsed .mod-body{display:none}
  .mod.collapsed .modbar{opacity:.72}

  .ctl{font-family:var(--display);background:rgba(var(--accent-rgb),0.10);color:var(--accent);
    border:1px solid rgba(var(--accent-rgb),0.34);border-radius:8px;padding:6px 12px;
    font-size:11px;font-weight:700;letter-spacing:0.1em;cursor:pointer;
    transition:background .15s,color .15s}
  .ctl:hover{background:rgba(var(--accent-rgb),0.20)}
  .ctl:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .ctl.active{background:var(--accent);color:var(--on-accent);border-color:var(--accent)}

  body.rearranging .mod{border:1px dashed rgba(var(--accent-rgb),0.45);border-radius:12px;
    padding:8px 10px;margin-bottom:12px;background:rgba(var(--accent-rgb),0.03)}
  body.rearranging .modbar .grip{display:inline-block}
  body.rearranging .modbar{cursor:grab}
  body.rearranging .mod.dragging{opacity:.45}
  body.rearranging .mod.drop-target{border-color:var(--accent);
    box-shadow:0 0 0 2px rgba(var(--accent-rgb),0.3)}
  @media (prefers-reduced-motion:reduce){ .modbar,.modbar .chev,.ctl{transition:none} }
  .section-h .badge{font-size:10px;padding:2px 8px;border-radius:8px;background:rgba(94,234,212,0.12);
    color:var(--accent);border:1px solid rgba(94,234,212,0.3);letter-spacing:0.06em}
  .section-h .badge.off{background:rgba(248,113,113,0.15);color:var(--red);border-color:rgba(248,113,113,0.3)}

  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px;margin-bottom:22px}
  .grid.solo{grid-template-columns:1fr}
  .card{
    background:var(--card);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
    border:1px solid var(--border);border-radius:16px;padding:18px 20px;
    box-shadow:0 8px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    transition:border-color 0.3s;
  }
  .card:hover{border-color:rgba(120,160,220,0.35)}
  .card.stale{opacity:0.65;border-color:rgba(248,113,113,0.4)}
  .card h2{
    font-size:13px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
    color:var(--txt);margin-bottom:13px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  }
  .card h2 .badge{
    font-size:10px;padding:2px 8px;border-radius:8px;background:rgba(94,234,212,0.12);
    color:var(--accent);border:1px solid rgba(94,234,212,0.3);letter-spacing:0.06em;
  }
  .card h2 .badge.violet{background:rgba(167,139,250,0.12);color:var(--accent2);
    border-color:rgba(167,139,250,0.3)}
  .dot{width:9px;height:9px;border-radius:50%;display:inline-block}
  .dot.on{background:var(--green);box-shadow:0 0 9px var(--green)}
  .dot.off{background:var(--red);box-shadow:0 0 9px var(--red)}

  .row{display:flex;justify-content:space-between;align-items:baseline;padding:5px 0;font-size:13.5px}
  .row .label{color:var(--dim)} .row .val{font-weight:600;font-variant-numeric:tabular-nums}
  .big{font-size:38px;font-weight:800;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;margin:2px 0}
  .sub{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:0.12em}
  .stack{display:flex;gap:16px;align-items:flex-end;margin-bottom:10px;flex-wrap:wrap}
  .stack > div{flex:1;min-width:78px}

  .pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:10.5px;
    font-weight:700;letter-spacing:0.05em;text-transform:uppercase}
  .pill.green{background:rgba(52,211,153,0.15);color:var(--green);border:1px solid rgba(52,211,153,0.3)}
  .pill.yellow{background:rgba(251,191,36,0.15);color:var(--yellow);border:1px solid rgba(251,191,36,0.3)}
  .pill.red{background:rgba(248,113,113,0.15);color:var(--red);border:1px solid rgba(248,113,113,0.3)}
  .pill.muted{background:rgba(138,150,170,0.12);color:var(--dim);border:1px solid rgba(138,150,170,0.25)}

  .bar{height:6px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;margin:5px 0 2px}
  .bar > span{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));
    border-radius:4px;transition:width 0.4s}
  .bar.temp > span{background:linear-gradient(90deg,var(--yellow),var(--red))}

  .spark{margin-top:8px;height:34px;width:100%}
  .spark path{fill:none;stroke:var(--accent);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
  .spark .area{fill:url(#sparkfill);stroke:none}

  /* per-model inference perf module */
  .modcard{
    background:linear-gradient(150deg,rgba(94,234,212,0.06),rgba(167,139,250,0.05)),var(--card);
    border:1px solid rgba(94,234,212,0.22);
  }
  .perf{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0 6px}
  .perf .pc{background:rgba(255,255,255,0.03);border:1px solid var(--border);
    border-radius:10px;padding:9px 10px;text-align:center}
  .perf .pc .pk{font-size:9.5px;color:var(--dim);letter-spacing:0.1em;text-transform:uppercase}
  .perf .pc .pv{font-size:21px;font-weight:800;font-variant-numeric:tabular-nums;
    margin-top:3px;line-height:1;color:var(--accent)}
  .perf .pc .pv.violet{color:var(--accent2)}
  .perf .pc .pv .pu{font-size:11px;color:var(--dim);font-weight:600;margin-left:3px}
  .perf .pc.idle .pv{color:var(--dim)}

  /* fabric switch: fan grid + fabric-port throughput tiles */
  .swcard{
    background:linear-gradient(150deg,rgba(167,139,250,0.06),rgba(94,234,212,0.05)),var(--card);
    border:1px solid rgba(167,139,250,0.22);
  }
  .ports{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-top:6px}
  .port{background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:10px;padding:8px 10px}
  .port .pn{font-size:11px;color:var(--dim);letter-spacing:0.04em}
  .port .pr{font-size:14px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:2px}
  .port .pt{font-size:10px;color:var(--accent);margin-top:2px}
  .fans{display:grid;grid-template-columns:repeat(auto-fit,minmax(60px,1fr));gap:6px;margin-top:4px}
  .fan{background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:8px;
    padding:6px 4px;text-align:center}
  .fan .fk{font-size:9px;color:var(--dim)} .fan .fv{font-size:12px;font-weight:700;margin-top:2px}

  .freshline{font-size:10.5px;color:var(--dim);margin-top:10px;letter-spacing:0.04em;
    display:flex;justify-content:space-between;align-items:center}
  .freshline.stale{color:var(--red)}
  .err{color:var(--red);font-size:11.5px;margin-top:6px}
  .footer{margin-top:24px;text-align:center;color:var(--dim);font-size:11px;
    letter-spacing:0.1em;text-transform:uppercase}
</style>
</head>
<body>
<header>
  <div>
    <h1 id="title">⚡ Sparky Command Center</h1>
    <div style="font-size:12px;color:var(--dim);margin-top:4px;letter-spacing:0.08em">
      <span class="pulse" id="pulse"></span><span id="subtitle">read-only</span>
    </div>
  </div>
  <div class="meta">
    <div>updated <span class="ts" id="ts">-</span></div>
    <div style="font-size:11px;margin-top:2px" id="refreshnote">browser refresh 2.5s</div>
    <div style="margin-top:8px;display:flex;gap:6px;justify-content:flex-end;flex-wrap:wrap">
      <div id="theme-wrap">
        <button id="theme-btn" class="ctl" type="button" aria-haspopup="true" aria-expanded="false">&#9680; THEME</button>
        <div id="theme-menu" role="menu" aria-label="Colour theme">
          <button class="theme-opt" data-set="sunset" role="menuitemradio"><i style="--s1:#ff7a1a;--s2:#150e08"></i>SUNSET</button>
          <button class="theme-opt" data-set="breeze" role="menuitemradio"><i style="--s1:#38bdf8;--s2:#0b1220"></i>BREEZE</button>
          <button class="theme-opt" data-set="light"  role="menuitemradio"><i style="--s1:#0b6fd4;--s2:#f4f6f9"></i>LIGHT</button>
          <button class="theme-opt" data-set="dark"   role="menuitemradio"><i style="--s1:#e6e6e6;--s2:#050505"></i>DARK</button>
          <button class="theme-opt" data-set="matrix" role="menuitemradio"><i style="--s1:#00ff66;--s2:#000000"></i>MATRIX</button>
        </div>
      </div>
      <button id="rearrange-btn" class="ctl" type="button">&#8645; REARRANGE</button>
      <button id="expand-btn" class="ctl" type="button">&#9776; COLLAPSE ALL</button>
    </div>
  </div>
</header>

<svg width="0" height="0" style="position:absolute">
  <defs>
    <linearGradient id="sparkfill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.38"/>
      <stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>
    </linearGradient>
  </defs>
</svg>

<div class="summary" id="summary"></div>

<div id="modules">
  <section class="mod" data-mod="tokens">
    <div class="modbar"><span class="grip">&#8942;&#8942;</span><span class="chev">&#9662;</span>Token Tracker<span class="ln"></span></div>
    <div class="mod-body"><div id="token-tracker"></div></div>
  </section>
  <section class="mod" data-mod="switch">
    <div class="modbar"><span class="grip">&#8942;&#8942;</span><span class="chev">&#9662;</span>Fabric Switch<span class="ln"></span></div>
    <div class="mod-body"><div id="switch"></div></div>
  </section>
  <section class="mod" data-mod="nodes">
    <div class="modbar"><span class="grip">&#8942;&#8942;</span><span class="chev">&#9662;</span>Nodes<span class="ln"></span></div>
    <div class="mod-body"><div id="nodes"></div></div>
  </section>
  <section class="mod" data-mod="fleetmodels">
    <div class="modbar"><span class="grip">&#8942;&#8942;</span><span class="chev">&#9662;</span>Fleet Models<span class="ln"></span></div>
    <div class="mod-body"><div id="fleet-models"></div></div>
  </section>
  <section class="mod" data-mod="video" id="mod-video" hidden>
    <div class="modbar"><span class="grip">&#8942;&#8942;</span><span class="chev">&#9662;</span>Video Generation<span class="ln"></span></div>
    <div class="mod-body"><div class="grid" id="comfy-grid"></div></div>
  </section>
</div>

<div class="footer">READ-ONLY - polled over SSH + HTTP /metrics - never disturbs live inference</div>

<script>
function fmtTs(t){
  if(!t) return '-';
  return new Date(t*1000).toLocaleTimeString('en-US',{hour12:false});
}
function age(ts){ return ts? Math.max(0, Math.round(Date.now()/1000 - ts)) : null; }

function tempPill(t, warn, hot){
  if(t==null) return '<span class="pill muted">-</span>';
  const c = t>=hot?'red':t>=warn?'yellow':'green';
  return `<span class="pill ${c}">${t.toFixed(0)}°C</span>`;
}

function sparkline(values){
  if(!values || values.length < 2) return '';
  const W=300,H=34,pad=2;
  const max=Math.max(...values,1), min=Math.min(...values,0);
  const range=Math.max(max-min,1);
  const step=(W-pad*2)/(values.length-1);
  const pts=values.map((v,i)=>{
    const x=pad+i*step;
    const y=H-pad-((v-min)/range)*(H-pad*2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <path class="area" d="M ${pad},${H-pad} L ${pts.join(' L ')} L ${(W-pad).toFixed(1)},${H-pad} Z"/>
    <path d="M ${pts.join(' L ')}"/></svg>`;
}

function freshLine(ts, err){
  const a = age(ts);
  const stale = a==null || a>20;
  const txt = a==null ? 'no data' : (a+'s ago');
  return `<div class="freshline ${stale?'stale':''}">
    <span>${err? '⚠ '+escH(err) : (stale?'STALE':'live')}</span>
    <span>updated ${txt}</span></div>`;
}

function escH(s){
  return String(s==null?'':s).replace(/[&<>"']/g,
    c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderGpu(node, g, hist){
  const sp = hist['node:'+node.key+':'+g.index+':temp'];
  const pl = g.power_limit||350;
  const warn = node.temp_warn, hot = node.temp_hot;
  return `<div class="card">
    <h2><span class="dot on"></span>GPU ${g.index} <span class="badge">${escH(g.name)}</span></h2>
    <div class="stack">
      <div><div class="sub">Temp</div>
        <div class="big">${g.temp!=null?g.temp.toFixed(0):'-'}<span style="font-size:16px;color:var(--dim)">°C</span></div>
        ${tempPill(g.temp,warn,hot)}</div>
      <div><div class="sub">Power</div>
        <div class="big">${g.power!=null?g.power.toFixed(0):'-'}<span style="font-size:16px;color:var(--dim)">W</span></div>
        <span class="pill ${g.power>pl*0.85?'red':g.power>pl*0.55?'yellow':'green'}">${g.power!=null?(g.power/pl*100).toFixed(0):'-'}% cap</span></div>
    </div>
    <div class="row"><span class="label">Utilization</span><span class="val">${g.util!=null?g.util.toFixed(0):'-'} %</span></div>
    <div class="bar"><span style="width:${g.util||0}%"></span></div>
    <div class="row"><span class="label">VRAM</span><span class="val">${g.mem_used_mb!=null?(g.mem_used_mb/1024).toFixed(1):'-'} / ${g.mem_total_mb!=null?(g.mem_total_mb/1024).toFixed(1):'-'} GiB</span></div>
    <div class="bar"><span style="width:${g.mem_pct||0}%"></span></div>
    <div class="row"><span class="label">Fan</span><span class="val">${g.fan!=null?g.fan.toFixed(0)+' %':'-'}</span></div>
    <div class="row"><span class="label">Graphics clock</span><span class="val">${g.gr_clock!=null?g.gr_clock.toFixed(0)+' MHz':'-'}</span></div>
    ${sparkline(sp)}
  </div>`;
}

function renderNodeSys(node){
  const reach = node.reachable;
  return `<div class="card ${reach?'':'stale'}">
    <h2><span class="dot ${reach?'on':'off'}"></span>${escH(node.name)} - system
      <span class="badge violet">HOST</span></h2>
    ${reach? `
    <div class="stack">
      <div><div class="sub">CPU Temp</div>
        <div class="big">${node.cpu_temp!=null?node.cpu_temp.toFixed(0):'-'}<span style="font-size:16px;color:var(--dim)">°C</span></div>
        ${tempPill(node.cpu_temp,node.temp_warn+5,node.temp_hot+6)}</div>
    </div>
    ${(node.cpu_temps||[]).slice(1).map((t,i)=>`<div class="row"><span class="label">cpu sensor #${i+2}</span><span class="val">${t.toFixed(1)} °C</span></div>`).join('')}
    <div class="row"><span class="label">System RAM</span><span class="val">${node.mem_used_mb!=null?(node.mem_used_mb/1024).toFixed(1):'-'} / ${node.mem_total_mb!=null?(node.mem_total_mb/1024).toFixed(1):'-'} GiB</span></div>
    <div class="bar"><span style="width:${node.mem_pct||0}%"></span></div>
    <div class="row"><span class="label">GPUs</span><span class="val">${(node.gpus||[]).length}</span></div>
    ${freshLine(node.ts)}
    ` : `<div class="err">node unreachable - ${escH(node.err||'no response')}</div>${freshLine(node.ts, node.err)}`}
  </div>`;
}

function fmtTok(n){
  n = +n || 0;
  if(n >= 1e9) return (n/1e9).toFixed(2)+'B';
  if(n >= 1e6) return (n/1e6).toFixed(2)+'M';
  if(n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return String(Math.round(n));
}
function renderTokens(tk){
  if(!tk || !tk.enabled || !(tk.models && tk.models.length)) return '';
  const cards = tk.models.map(m=>{
    const on = m.reachable;
    return `<div class="card modcard${on?'':' stale'}">
      <h2><span class="dot ${on?'on':'off'}"></span>${escH(m.label||m.key)}
        ${m.gpus?`<span class="badge violet">${escH(m.gpus)}</span>`:''}</h2>
      <div class="big" style="color:var(--accent)">${fmtTok(m.total_tokens)}</div>
      <div class="sub">tokens served · cumulative</div>
      <div class="stack" style="margin-top:12px">
        <div><div class="sub">prompt</div><div class="val" style="font-size:17px">${fmtTok(m.total_prompt)}</div></div>
        <div><div class="sub">generated</div><div class="val" style="font-size:17px">${fmtTok(m.total_gen)}</div></div>
        <div><div class="sub">today</div><div class="val" style="font-size:17px;color:var(--accent2)">${fmtTok(m.today_tokens)}</div></div>
      </div>
    </div>`;
  }).join('');
  return `<div class="section-h">🎫 Token Tracker
      <span class="badge">${fmtTok(tk.total)} total</span>
      <span class="badge violet">${fmtTok(tk.today)} today</span>
      <span class="ln"></span></div>
    <div class="grid">${cards}</div>`;
}
function fmtTps(v){
  if(v==null) return '-';
  return v>=100? v.toFixed(0) : v.toFixed(1);
}
function renderModel(m){
  const reach = m.reachable;
  const cls = reach? '':'stale';
  const engine = m.engine || 'inference';
  const busy = (m.running||0) > 0;
  const idleCls = (reach && !busy)? ' idle':'';
  const ttft = m.ttft_ms!=null ? (m.ttft_ms>=1000? (m.ttft_ms/1000).toFixed(2)+'<span class="pu">s</span>' : m.ttft_ms.toFixed(0)+'<span class="pu">ms</span>') : '-';
  return `<div class="card modcard ${cls}">
    <h2><span class="dot ${reach?'on':'off'}"></span>${escH(m.label)}
      <span class="badge">${escH(engine)}</span>
      ${m.port?`<span class="badge">:${m.port}</span>`:''}
      ${busy?'<span class="badge violet">BUSY</span>':'<span class="badge" style="background:rgba(138,150,170,0.12);color:var(--dim);border-color:rgba(138,150,170,0.25)">idle</span>'}</h2>
    ${reach? `
    <div class="row"><span class="label">Model</span><span class="val" style="font-family:'SF Mono',ui-monospace,Menlo,monospace;font-size:12px;color:var(--accent)">${escH(m.model||'-')}</span></div>
    <div class="perf">
      <div class="pc${idleCls}"><div class="pk">Decode</div><div class="pv">${fmtTps(m.decode_tps)}<span class="pu">tok/s</span></div></div>
      <div class="pc${idleCls}"><div class="pk">Prefill</div><div class="pv violet">${fmtTps(m.prefill_tps)}<span class="pu">tok/s</span></div></div>
      <div class="pc${idleCls}"><div class="pk">TTFT avg</div><div class="pv">${ttft}</div></div>
    </div>
    <div class="row"><span class="label">KV-cache</span><span class="val">${m.kv_pct!=null?m.kv_pct.toFixed(1)+' %':'-'}</span></div>
    <div class="bar"><span style="width:${m.kv_pct||0}%"></span></div>
    <div class="row"><span class="label">Requests</span><span class="val">${m.running!=null?m.running:'-'} running · ${m.waiting!=null?m.waiting:'-'} waiting</span></div>
    ${m.gpus?`<div class="row"><span class="label">On</span><span class="val">${escH(m.gpus)}</span></div>`:''}
    ${freshLine(m.ts)}
    ` : `<div class="err">${escH(m.label)} offline - ${escH(m.err||'no /metrics')}</div>${freshLine(m.ts, m.err)}`}
  </div>`;
}

function fmtBps(bps){
  if(bps==null) return '0';
  if(bps>=1e9) return (bps/1e9).toFixed(2)+' Gb/s';
  if(bps>=1e6) return (bps/1e6).toFixed(1)+' Mb/s';
  if(bps>=1e3) return (bps/1e3).toFixed(0)+' Kb/s';
  return bps.toFixed(0)+' b/s';
}

function renderSwitch(sw){
  if(!sw) return '';
  const reach = sw.reachable;
  const h = sw.health||{}, r = sw.resource||{};
  const warn = sw.temp_warn||55, hot = sw.temp_hot||70;
  const swTemp = h['switch-temperature']!=null? parseFloat(h['switch-temperature']):null;
  const cpuTemp = h['cpu-temperature']!=null? parseFloat(h['cpu-temperature']):null;
  const ports = sw.ports||[];
  const fanKeys = Object.keys(h).filter(k=>/^fan\d+-speed$/.test(k))
    .sort((a,b)=>parseInt(a.replace(/\D/g,''))-parseInt(b.replace(/\D/g,'')));
  const psuOk = (h['psu1-state']==='ok' && h['psu2-state']==='ok');
  return `<div class="section-h">◢ ${escH(sw.name)}
    <span class="badge ${reach?'':'off'}">${reach?escH(sw.badge||'FABRIC'):'OFFLINE'}</span><span class="ln"></span></div>
  <div class="grid solo"><div class="card swcard ${reach?'':'stale'}">
    <h2><span class="dot ${reach?'on':'off'}"></span>${escH(sw.name)}
      <span class="badge violet">${escH(sw.badge||'FABRIC')}</span></h2>
    ${reach? `
    <div class="stack">
      <div><div class="sub">Switch Temp</div>
        <div class="big">${swTemp!=null?swTemp.toFixed(0):'-'}<span style="font-size:16px;color:var(--dim)">°C</span></div>
        ${tempPill(swTemp,warn,hot)}</div>
      <div><div class="sub">CPU Temp</div>
        <div class="big">${cpuTemp!=null?cpuTemp.toFixed(0):'-'}<span style="font-size:16px;color:var(--dim)">°C</span></div>
        ${tempPill(cpuTemp,warn+5,hot+5)}</div>
    </div>
    ${fanKeys.length?`<div class="row"><span class="label">Fans <span style="color:var(--green)">${escH(h['fan-state']||'')}</span></span>
      <span class="val">${h['psu1-state']||h['psu2-state']?(psuOk?'2 PSU ok':'PSU?'):''}</span></div>
    <div class="fans">${fanKeys.map((f,i)=>`<div class="fan"><div class="fk">FAN${i+1}</div><div class="fv">${escH(h[f])}</div></div>`).join('')}</div>`:''}
    ${h['psu1-power']!=null?`<div class="row" style="margin-top:8px"><span class="label">PSU1</span><span class="val">${escH(h['psu1-power'])} W · ${escH(h['psu1-temperature']||'-')}°C</span></div>`:''}
    ${h['psu2-power']!=null?`<div class="row"><span class="label">PSU2</span><span class="val">${escH(h['psu2-power'])} W · ${escH(h['psu2-temperature']||'-')}°C</span></div>`:''}
    ${r['uptime']?`<div class="row"><span class="label">Uptime</span><span class="val">${escH(r['uptime'])}</span></div>`:''}
    ${r['version']?`<div class="row"><span class="label">RouterOS</span><span class="val">${escH(r['version'].split(' ')[0])} · cpu ${escH(r['cpu-load']||'-')}</span></div>`:''}
    ${ports.length?`<div class="sub" style="margin-top:10px">Fabric ports · live throughput</div>
    <div class="ports">${ports.map(p=>`<div class="port"><div class="pn">${escH(p.name)} <span class="pill ${p.running?'green':'muted'}" style="padding:1px 6px">${p.running?'link-ok':'down'}</span></div>
      <div class="pr">${fmtBps((p.rx_bps||0)+(p.tx_bps||0))}</div>
      <div class="pt">${p.rate?escH(p.rate)+' · ':''}↓${fmtBps(p.rx_bps)} ↑${fmtBps(p.tx_bps)}</div></div>`).join('')}</div>`:''}
    ${freshLine(sw.ts)}
    ` : `<div class="err">switch unreachable - ${escH(sw.err||'no response')}</div>${freshLine(sw.ts, sw.err)}`}
  </div></div>`;
}

function renderNode(node, hist){
  const gpus = node.gpus||[];
  const models = node.models||[];
  const reach = node.reachable;
  let html = `<div class="section-h">◢ ${escH(node.name)}
    <span class="badge ${reach?'':'off'}">${reach?(gpus.length+' GPU'):'OFFLINE'}</span><span class="ln"></span></div>`;
  if(gpus.length){
    html += `<div class="grid">${gpus.map(g=>renderGpu(node,g,hist)).join('')}</div>`;
  }
  html += `<div class="grid solo">${renderNodeSys(node)}</div>`;
  if(models.length){
    html += `<div class="section-h acc">▸ Model performance<span class="ln"></span></div>`;
    html += `<div class="grid">${models.map(m=>renderModel(m)).join('')}</div>`;
  }
  return html;
}

function render(s){
  document.title = s.title || 'LLM Fleet Monitor';
  document.getElementById('title').textContent = s.title || 'LLM Fleet Monitor';
  document.getElementById('subtitle').textContent = s.subtitle || 'read-only';
  document.getElementById('ts').textContent = fmtTs(s.ts);
  if(s.browser_refresh_ms){
    document.getElementById('refreshnote').textContent = 'browser refresh '+(s.browser_refresh_ms/1000)+'s';
  }
  const agg = s.agg||{};
  document.getElementById('pulse').className = 'pulse' + (agg.all_ok? '':' bad');

  document.getElementById('summary').innerHTML = `
    <div class="scard"><div class="k">Fleet GPUs</div><div class="v neon">${agg.gpu_count!=null?agg.gpu_count:'-'}<span class="u">online</span></div></div>
    <div class="scard"><div class="k">Total Power Draw</div><div class="v violet">${agg.total_power!=null?agg.total_power:'-'}<span class="u">W</span></div></div>
    <div class="scard"><div class="k">Hottest GPU</div><div class="v ${agg.hottest_temp>=84?'red':'green'}" style="font-size:22px">${escH(agg.hottest_unit||'-')}<span class="u">${agg.hottest_temp!=null?agg.hottest_temp.toFixed(0)+'°C':''}</span></div></div>
    <div class="scard"><div class="k">Fleet Status</div><div class="v ${agg.all_ok?'green':'red'}" style="font-size:22px">${agg.all_ok?'● ALL OK':'● DEGRADED'}</div></div>`;

  document.getElementById('token-tracker').innerHTML = renderTokens(s.tokens);

  const hist = s.history||{};
  document.getElementById('switch').innerHTML = renderSwitch(s.switch);
  document.getElementById('nodes').innerHTML =
    (s.nodes||[]).map(n=>renderNode(n,hist)).join('');

  // Fleet-wide model instances (not tied to a single node), grouped by label.
  const fm = s.fleet_models||[];
  let fmHtml = '';
  if(fm.length){
    const groups = {};
    fm.forEach(m=>{ const g=m.group||'Fleet models'; (groups[g]=groups[g]||[]).push(m); });
    Object.keys(groups).forEach(g=>{
      fmHtml += `<div class="section-h acc">▸ ${escH(g)}<span class="ln"></span></div>`;
      fmHtml += `<div class="grid">${groups[g].map(m=>renderModel(m)).join('')}</div>`;
    });
  }
  document.getElementById('fleet-models').innerHTML = fmHtml;
}

let interval = 2500;
async function tick(){
  try{
    const r = await fetch('/api/metrics',{cache:'no-store'});
    const s = await r.json();
    render(s);
    if(s.browser_refresh_ms && s.browser_refresh_ms !== interval){
      interval = s.browser_refresh_ms;
      clearInterval(timer); timer = setInterval(tick, interval);
    }
  }catch(e){
    document.getElementById('ts').textContent = 'FETCH ERR: '+e.message;
  }
}
tick();
let timer = setInterval(tick, interval);

// -- Video generation lanes (ComfyUI-style image/video servers) --------------
// Hidden entirely when no lanes are configured, so the panel costs nothing to
// a deployment that has none. VRAM is the live number to watch during a
// render; `peak while rendering` is the capacity-planning number.
function fmtGB(b){ return b==null ? '-' : (b/1073741824).toFixed(1); }
function renderComfy(lanes){
  return lanes.map(function(l){
    var dead = !l.reachable;
    var tot = l.vram_total, used = l.vram_used;
    var pct = (tot && used!=null) ? Math.min(100,(used/tot)*100) : 0;
    var col = dead ? 'var(--dim)' : pct>85 ? 'var(--red)' : pct>60 ? 'var(--yellow)' : 'var(--green)';
    var state = dead ? '\u25cf offline' : l.busy ? '\u25cf RENDERING' : '\u25cf idle';
    var stcol = dead ? 'var(--red)' : l.busy ? 'var(--accent)' : 'var(--green)';
    var queue = (l.running||0)+(l.pending||0);
    return '<div class="scard" style="text-align:left;padding:16px 18px">'
      + '<div class="k" style="display:flex;justify-content:space-between;align-items:center">'
      + '<span><span class="badge">LANE '+escH(l.lane)+'</span> '+escH(l.name||'')+'</span>'
      + '<span style="color:'+stcol+';font-size:11px">'+state+'</span></div>'
      + '<div class="v neon" style="font-size:28px">'+fmtGB(used)+'<span class="u">GB used</span></div>'
      + '<div style="height:6px;border-radius:3px;background:rgba(255,255,255,.08);margin:8px 0 6px">'
      + '<div style="height:100%;width:'+pct.toFixed(1)+'%;border-radius:3px;background:'+col+'"></div></div>'
      + '<div style="color:var(--dim);font-size:12px;display:flex;gap:16px;flex-wrap:wrap">'
      + '<span>'+fmtGB(l.vram_free)+' GB free of '+fmtGB(tot)+'</span><span>queue '+queue+'</span></div>'
      + '<div style="font-size:12px;margin-top:6px;display:flex;gap:16px;flex-wrap:wrap">'
      + '<span style="color:var(--accent2)">peak while rendering <b>'
      + (l.peak_busy!=null?fmtGB(l.peak_busy)+' GB':'-')+'</b></span>'
      + '<span style="color:var(--dim)">peak any '+fmtGB(l.peak_used)+' GB</span></div>'
      + '<div style="margin-top:10px"><a href="'+escH(l.url)+'" target="_blank" rel="noopener" '
      + 'style="color:var(--accent);font-size:13px;text-decoration:none">open UI &nbsp;'
      + escH((l.url||'').replace('http://',''))+' &rarr;</a></div>'
      + '<div style="color:var(--dim);font-size:11px;margin-top:4px">'+escH(l.host||'')
      + (l.version?' \u00b7 v'+escH(l.version):'')+'</div></div>';
  }).join('');
}
async function tickComfy(){
  try{
    const r = await fetch('/api/comfy',{cache:'no-store'});
    const lanes = (await r.json()).lanes||[];
    document.getElementById('mod-video').hidden = lanes.length===0;
    if(lanes.length) document.getElementById('comfy-grid').innerHTML = renderComfy(lanes);
  }catch(e){}
}
tickComfy();
setInterval(tickComfy, 5000);

// -- Module collapse + rearrange --------------------------------------------
// Per-browser layout in localStorage; the server stays stateless. A saved
// order lists data-mod keys: unknown keys are ignored and a module missing
// from the saved order keeps its place, so adding a panel later never
// strands it off-screen.
// ── Theme picker ───────────────────────────────────────────────────────────
// Sunset is the default and is plain :root, so an unset/corrupt value falls
// back to it rather than to an unstyled page.
const THEMES=['sunset','breeze','light','dark','matrix'];
const LS_THEME='acc.theme.v1';
function applyTheme(name){
  const t = THEMES.includes(name) ? name : 'sunset';
  // sunset is the base :root, so it carries no attribute at all
  if(t==='sunset') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme',t);
  document.querySelectorAll('.theme-opt').forEach(b=>
    b.setAttribute('aria-checked', String(b.dataset.set===t)));
  try{ localStorage.setItem(LS_THEME,t); }catch(e){}
}
(function initTheme(){
  let t; try{ t=localStorage.getItem(LS_THEME); }catch(e){}
  applyTheme(t||'sunset');
})();
const themeBtn=document.getElementById('theme-btn');
const themeMenu=document.getElementById('theme-menu');
function closeThemeMenu(){ themeMenu.classList.remove('open'); themeBtn.setAttribute('aria-expanded','false'); }
themeBtn.addEventListener('click', ev=>{
  ev.stopPropagation();
  const open=themeMenu.classList.toggle('open');
  themeBtn.setAttribute('aria-expanded',String(open));
});
themeMenu.addEventListener('click', ev=>{
  const opt=ev.target.closest('.theme-opt');
  if(!opt) return;
  ev.stopPropagation();
  applyTheme(opt.dataset.set);
  closeThemeMenu();
});
document.addEventListener('click', ev=>{
  if(!ev.target.closest('#theme-wrap')) closeThemeMenu();
});
document.addEventListener('keydown', ev=>{ if(ev.key==='Escape') closeThemeMenu(); });

const LS_ORDER='fleet.modOrder', LS_COLLAPSED='fleet.modCollapsed';
const modBox=document.getElementById('modules');
function mods(){ return [].slice.call(modBox.querySelectorAll(':scope > .mod')); }
function findMod(id){
  return modBox.querySelector(':scope > .mod[data-mod="'+CSS.escape(id)+'"]');
}
function saveOrder(){
  try{ localStorage.setItem(LS_ORDER, JSON.stringify(mods().map(m=>m.dataset.mod))); }catch(e){}
}
function saveCollapsed(){
  try{ localStorage.setItem(LS_COLLAPSED,
    JSON.stringify(mods().filter(m=>m.classList.contains('collapsed')).map(m=>m.dataset.mod))); }catch(e){}
}
function restore(key, fn){
  let ids; try{ ids=JSON.parse(localStorage.getItem(key)||'null'); }catch(e){}
  if(Array.isArray(ids)) ids.forEach(id=>{ const el=findMod(id); if(el) fn(el); });
}
function syncExpandBtn(){
  document.getElementById('expand-btn').innerHTML =
    mods().some(m=>m.classList.contains('collapsed')) ? '&#9776; EXPAND ALL' : '&#9776; COLLAPSE ALL';
}
modBox.addEventListener('click', ev=>{
  const bar = ev.target.closest('.modbar');
  if(!bar || document.body.classList.contains('rearranging')) return;
  bar.parentElement.classList.toggle('collapsed');
  saveCollapsed(); syncExpandBtn();
});
document.getElementById('expand-btn').addEventListener('click', ()=>{
  const any = mods().some(m=>m.classList.contains('collapsed'));
  mods().forEach(m=>m.classList.toggle('collapsed', !any));
  saveCollapsed(); syncExpandBtn();
});
let dragEl=null;
const rearrangeBtn=document.getElementById('rearrange-btn');
rearrangeBtn.addEventListener('click', ()=>{
  const on = document.body.classList.toggle('rearranging');
  rearrangeBtn.classList.toggle('active', on);
  rearrangeBtn.innerHTML = on ? '&#10003; DONE' : '&#8645; REARRANGE';
  mods().forEach(m=>{ m.draggable = on; });
});
modBox.addEventListener('dragstart', ev=>{
  const m=ev.target.closest('.mod'); if(!m) return;
  dragEl=m; m.classList.add('dragging');
  ev.dataTransfer.effectAllowed='move';
  ev.dataTransfer.setData('text/plain', m.dataset.mod);
});
modBox.addEventListener('dragend', ()=>{
  if(dragEl) dragEl.classList.remove('dragging');
  modBox.querySelectorAll('.drop-target').forEach(e=>e.classList.remove('drop-target'));
  dragEl=null; saveOrder();
});
modBox.addEventListener('dragover', ev=>{
  if(!dragEl) return;
  ev.preventDefault(); ev.dataTransfer.dropEffect='move';
  const over=ev.target.closest('.mod');
  if(!over || over===dragEl) return;
  modBox.querySelectorAll('.drop-target').forEach(e=>e.classList.remove('drop-target'));
  over.classList.add('drop-target');
  const r=over.getBoundingClientRect();
  modBox.insertBefore(dragEl, ev.clientY > r.top + r.height/2 ? over.nextSibling : over);
});
modBox.addEventListener('drop', ev=>ev.preventDefault());
restore(LS_ORDER, el=>modBox.appendChild(el));
restore(LS_COLLAPSED, el=>el.classList.add('collapsed'));
syncExpandBtn();
</script>
</body>
</html>"""


def main():
    global CFG, TOKEN_STORE
    CFG = load_config()
    if CFG["server"].get("token_tracking", True):
        store = CFG["server"].get("token_store") or "data/token_usage.json"
        TOKEN_STORE = os.path.expanduser(store)
        _load_tokens()
    start_pollers()
    bind, port = CFG["server"]["bind"], int(CFG["server"]["port"])
    httpd = ThreadingHTTPServer((bind, port), Handler)
    print(f"{CFG['server']['title']} on http://{bind}:{port}  "
          f"({len(CFG['nodes'])} nodes, {len(CFG['models'])} models)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
