#!/usr/bin/env python3
"""
Agency Command Center v2
========================
A read-only command center for a self-hosted AI fleet: GPU nodes (discrete cards or
unified-memory boxes like the DGX Spark), an optional RoCE fabric switch, the model
servers running on them (vLLM, SGLang, llama.cpp), ComfyUI render lanes, cumulative
token usage, and a built-in chat assistant that talks to ANY OpenAI-compatible
/v1/chat/completions endpoint (a local model, or an agent that exposes that API).

Everything site-specific lives in config.json (gitignored). See config.example.json
and README.md. Python 3.8+ standard library only; the web UI is a prebuilt Vite app
in web/dist.

Monitoring is READ-ONLY. Every remote command only queries state (nvidia-smi query,
/proc, RouterOS print/monitor). The few write-capable routes (GPU clock caps, key
lights) are refused while server.read_only is true, which is the default.
"""

import datetime
import json
import mimetypes
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "2.0.0"


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DEFAULTS = {
    "server": {
        "title": "Command Center",
        "subtitle": "Self-hosted AI fleet",
        "location": "",
        "timezone": "",                 # IANA name for the UI clock + chat, e.g. America/New_York
        "bind": ["127.0.0.1"],          # one address or a list
        "port": 8895,
        "allowed_hosts": [],            # extra Host header names allowed to POST (tailnet names)
        "read_only": True,              # refuse every write route (clock caps, key lights)
        "static_dir": "web/dist",
        "browser_refresh_ms": 2500,
    },
    "ssh": {
        "default_key": "",
        "connect_timeout": 8,
        "options": {"IdentitiesOnly": "yes", "BatchMode": "yes",
                    "StrictHostKeyChecking": "accept-new"},
        # Reuse one SSH connection per target instead of a full login every poll.
        "multiplex": True,
        "control_dir": "~/.ssh/command-center",
        "control_persist": "60s",
    },
    "defaults": {"poll_interval": 7.0, "model_poll_interval": 6.0,
                 "temp_warn": 70, "temp_hot": 84, "stale_after_s": 20},
    "nodes": [],
    "sections": [],
    "switch": None,
    "models": [],
    "comfy_lanes": [],
    "comfy_poll_seconds": 4.0,
    "tokens": {"enabled": True, "mode": "bank", "store": "data/token_usage.json",
               "poll_seconds": 120, "order": [], "models": []},
    "eco": {"enabled": False, "allow_writes": False, "levels": [], "info_url": ""},
    "stations": {"enabled": False, "items": []},
    "keylights": {"enabled": False, "url": ""},
    "chat": {
        "enabled": True,
        "name": "Jarvis",
        "base_url": "",
        "model": "",
        "api_key": "",
        "system_prompt": "",
        "temperature": 0.1,
        "max_tokens": 1200,
        "timeout": 180,
        "history_turns": 12,
        "grounding": True,
        "extra_body": {},
        "fallbacks": [],
        "suggestions": ["What is down right now?", "Which node is running hottest?",
                        "How fast is each model decoding?"],
    },
}


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_env_file(path):
    """Minimal KEY=VALUE .env loader. Never overrides a variable already set."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except FileNotFoundError:
        pass


def _abs(path):
    if not path:
        return path
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def load_config():
    _load_env_file(os.path.join(HERE, ".env"))
    path = os.environ.get("CC_CONFIG") or os.path.join(HERE, "config.json")
    raw, used = {}, None
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        used = path
    cfg = _deep_merge(DEFAULTS, raw)
    env = os.environ
    chat = cfg["chat"]
    if env.get("CC_CHAT_BASE_URL"):
        chat["base_url"] = env["CC_CHAT_BASE_URL"]
    if env.get("CC_CHAT_MODEL"):
        chat["model"] = env["CC_CHAT_MODEL"]
    if env.get("CC_CHAT_API_KEY"):
        chat["api_key"] = env["CC_CHAT_API_KEY"]
    if env.get("CC_CHAT_SYSTEM_PROMPT"):
        chat["system_prompt"] = env["CC_CHAT_SYSTEM_PROMPT"]
    if env.get("CC_CHAT_NAME"):
        chat["name"] = env["CC_CHAT_NAME"]
    if chat.get("api_key_file") and not chat.get("api_key"):
        try:
            with open(_abs(chat["api_key_file"])) as f:
                chat["api_key"] = f.read().strip()
        except Exception:
            pass
    if env.get("CC_PORT"):
        cfg["server"]["port"] = int(env["CC_PORT"])
    if env.get("CC_BIND"):
        cfg["server"]["bind"] = [b.strip() for b in env["CC_BIND"].split(",") if b.strip()]
    if env.get("CC_READ_ONLY"):
        cfg["server"]["read_only"] = env["CC_READ_ONLY"].lower() not in ("0", "false", "no")
    if isinstance(cfg["server"]["bind"], str):
        cfg["server"]["bind"] = [cfg["server"]["bind"]]
    return cfg, used


CFG, CFG_PATH = load_config()
READ_ONLY = bool(CFG["server"]["read_only"])
NODES = CFG["nodes"]
NODE_BY_KEY = {n["key"]: n for n in NODES}
MODELS = CFG["models"]
COMFY_LANES = CFG["comfy_lanes"]
SWITCH = CFG["switch"] if (CFG.get("switch") or {}).get("enabled", True) and CFG.get("switch") else None
HIST_LEN = 60


# ----------------------------------------------------------------------------
# Shared state
# ----------------------------------------------------------------------------
_lock = threading.Lock()
STATE = {
    "nodes": {n["key"]: {"key": n["key"], "name": n.get("name", n["key"]),
                         "profile": n.get("profile", "discrete"),
                         "reachable": False, "ts": 0, "err": "warming up"} for n in NODES},
    "switch": {"reachable": False, "ts": 0, "err": "warming up"},
    "models": {},
    "comfy": {},
}
_hist = {}


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


def _http_get(url, timeout=6, headers=None):
    """Stdlib GET. Returns (ok, text). Never raises."""
    try:
        h = {"User-Agent": "command-center-v2"}
        h.update(headers or {})
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, r.read().decode("utf-8", "replace")
    except Exception:
        return False, ""


def _http_status(url, timeout=3):
    """Any HTTP answer counts as up (a 404 still means the server is listening)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "command-center-v2"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def _http_post(url, data, timeout=8):
    try:
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "command-center-v2"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception:
            return e.code, ""
    except Exception as e:  # noqa
        return 0, str(e)[:140]


# ----------------------------------------------------------------------------
# SSH
# ----------------------------------------------------------------------------
def _ssh_flags(obj):
    s = CFG["ssh"]
    flags = []
    key = obj.get("ssh_key") or s.get("default_key")
    if key:
        flags += ["-i", os.path.expanduser(key)]
    opts = s.get("options") or {}
    for k, v in opts.items():
        flags += ["-o", "%s=%s" % (k, v)]
    flags += _multiplex_flags(obj, opts)
    flags += ["-o", "ConnectTimeout=%s" % int(obj.get("connect_timeout") or s.get("connect_timeout") or 8)]
    if obj.get("ssh_port"):
        flags += ["-p", str(obj["ssh_port"])]
    return flags


def _off(v):
    return v is False or v == 0 or str(v).strip().lower() in ("false", "no", "off")


def _multiplex_flags(obj, opts):
    """ControlMaster flags, unless turned off (globally or per node/switch), on
    Windows (no ControlMaster there), or the user already set ControlMaster/ControlPath."""
    s = CFG["ssh"]
    if os.name == "nt" or _off(s.get("multiplex", True)) or _off(obj.get("multiplex", True)):
        return []
    keys = {k.lower() for k in opts}
    if keys & {"controlmaster", "controlpath"}:
        return []
    d = os.path.expanduser(s.get("control_dir") or "~/.ssh/command-center")
    try:
        os.makedirs(d, mode=0o700, exist_ok=True)
        if os.stat(d).st_uid != os.getuid():
            return []
        os.chmod(d, 0o700)
    except OSError:
        return []
    # %C is a hash of host, port and user: short enough for the 104-byte socket path limit.
    flags = ["-o", "ControlMaster=auto", "-o", "ControlPath=%s" % os.path.join(d, "%C"),
             "-o", "ControlPersist=%s" % (s.get("control_persist") or "60s")]
    # A kept-open master to a node that rebooted would hang every poll until the poll
    # timeout; keepalives make it notice and exit within ~10s.
    if "serveraliveinterval" not in keys:
        flags += ["-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
    return flags


def _target(obj):
    return "%s@%s" % (obj["user"], obj["host"]) if obj.get("user") else obj["host"]


def remote_argv(node, script):
    """argv that runs `script` on a node. Three reach modes:
      direct             ssh user@host script
      jump.mode=proxy    ssh -J jumpuser@jump user@host script
      jump.mode=nested   ssh jumpuser@jump "ssh user@host 'script'"  (the jump host's own
                         key reaches the LAN node; use when only the jump host is trusted)
    """
    jump = node.get("jump")
    if not jump:
        return ["ssh"] + _ssh_flags(node) + [_target(node), script]
    jt = _target(jump)
    if jump.get("mode", "nested") in ("proxy", "proxyjump"):
        return ["ssh"] + _ssh_flags(node) + ["-J", jt, _target(node), script]
    inner = ("ssh -o BatchMode=yes -o ConnectTimeout=6 %s %s"
             % (shlex.quote(_target(node)), shlex.quote(script)))
    jflags = _ssh_flags({"ssh_key": jump.get("ssh_key") or node.get("ssh_key"),
                         "ssh_port": jump.get("ssh_port"),
                         "multiplex": jump.get("multiplex", node.get("multiplex", True))})
    return ["ssh"] + jflags + [jt, inner]


# ----------------------------------------------------------------------------
# Node pollers
#   profile "unified": one GPU sharing system RAM (DGX Spark / GB10, Jetson-class).
#     nvidia-smi has no memory fields there, so /proc/meminfo IS the GPU memory.
#   profile "discrete": a host with one or more PCIe GPUs (RTX, A-series ...).
# ----------------------------------------------------------------------------
_UNIFIED_REMOTE = (
    'nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,'
    'clocks.current.sm --format=csv,noheader,nounits; '
    'echo PIPEMEM; '
    'grep -E "^(MemTotal|MemAvailable):" /proc/meminfo; '
    'echo PIPEAPP; '
    'nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | paste -sd+ | bc'
)


def _discrete_remote(containers):
    s = ('nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,'
         'utilization.gpu,memory.used,memory.total,fan.speed,clocks.gr,clocks.mem '
         '--format=csv,noheader,nounits; '
         'echo PIPECPU; '
         'for h in /sys/class/hwmon/hwmon*; do n=$(cat $h/name 2>/dev/null); '
         'if [ "$n" = "k10temp" ] || [ "$n" = "coretemp" ] || [ "$n" = "zenpower" ]; then '
         'cat $h/temp*_input 2>/dev/null; break; fi; done; '
         'echo PIPEMEM; free -m | awk "/^Mem:/{print \\$2, \\$3}"; '
         'echo PIPEDOCKER; ')
    if containers:
        s += 'docker ps --format "{{.Names}}" 2>/dev/null'
    return s


def poll_unified(node):
    res = {
        "key": node["key"], "name": node.get("name", node["key"]), "profile": "unified",
        "node_id": node.get("node_id"), "rank": node.get("rank"), "reachable": False,
        "temp": None, "power": None, "util": None, "sm_clock": None,
        "mem_total_gib": None, "mem_used_gib": None, "mem_pct": None,
        "model_gib": None, "model": node.get("serving"), "pair": node.get("pair"),
        "ts": time.time(), "err": None,
    }
    rc, out, err = _run(remote_argv(node, _UNIFIED_REMOTE), timeout=16)
    if rc != 0 or not out.strip():
        res["err"] = (err or "no output").strip()[:140]
        return res
    try:
        gpu_part, _, rest = out.partition("PIPEMEM")
        mem_part, _, app_part = rest.partition("PIPEAPP")
        gline = gpu_part.strip().splitlines()
        if gline:
            f = [x.strip() for x in gline[0].split(",")]
            res["temp"] = _num(f[0]) if len(f) > 0 else None
            res["power"] = _num(f[1]) if len(f) > 1 else None
            res["util"] = _num(f[2]) if len(f) > 2 else None
            res["sm_clock"] = _num(f[3]) if len(f) > 3 else None
        tot_kb = avail_kb = None
        for line in mem_part.strip().splitlines():
            mm = re.match(r"(MemTotal|MemAvailable):\s+(\d+)", line.strip())
            if not mm:
                continue
            if mm.group(1) == "MemTotal":
                tot_kb = float(mm.group(2))
            else:
                avail_kb = float(mm.group(2))
        if tot_kb:
            used_kb = tot_kb - (avail_kb or 0)
            res["mem_total_gib"] = round(tot_kb / 1048576.0, 1)
            res["mem_used_gib"] = round(used_kb / 1048576.0, 1)
            res["mem_pct"] = round(used_kb / tot_kb * 100.0, 1)
        app_mib = _num(app_part.strip().splitlines()[0]) if app_part.strip() else None
        if app_mib is not None:
            res["model_gib"] = round(app_mib / 1024.0, 1)
        res["reachable"] = res["temp"] is not None
        if not res["reachable"]:
            res["err"] = "no GPU reading"
    except Exception as e:  # noqa
        res["err"] = ("parse: %s" % e)[:140]
    return res


def poll_discrete(node):
    res = {"key": node["key"], "name": node.get("name", node["key"]), "profile": "discrete",
           "host_label": node.get("badge") or "GPU HOST",
           "reachable": False, "ts": time.time(), "err": None,
           "gpus": [], "cpu_temp": None, "cpu_temps": [],
           "mem_total_mb": None, "mem_used_mb": None, "mem_pct": None, "models": []}
    labels = node.get("containers") or {}
    rc, out, err = _run(remote_argv(node, _discrete_remote(bool(node.get("list_containers", True)))),
                        timeout=14)
    if rc != 0 or not out.strip():
        res["err"] = (err or "no output").strip()[:140]
        return res
    try:
        gpu_part, _, rest = out.partition("PIPECPU")
        cpu_part, _, rest2 = rest.partition("PIPEMEM")
        mem_part, _, docker_part = rest2.partition("PIPEDOCKER")
        for line in gpu_part.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 11:
                continue
            idx = int(f[0])
            fan = _num(f[8]) if f[8].replace(".", "").isdigit() else None
            g = {"index": idx, "name": f[1].replace("NVIDIA GeForce ", ""),
                 "temp": _num(f[2]), "power": _num(f[3]), "power_limit": _num(f[4]),
                 "util": _num(f[5]), "mem_used_mb": _num(f[6]), "mem_total_mb": _num(f[7]),
                 "fan": fan, "gr_clock": _num(f[9]), "mem_clock": _num(f[10])}
            if g["mem_total_mb"]:
                g["mem_pct"] = round((g["mem_used_mb"] or 0) / g["mem_total_mb"] * 100.0, 1)
            res["gpus"].append(g)
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
        for name in [n.strip() for n in docker_part.strip().splitlines() if n.strip()]:
            meta = labels.get(name)
            if meta:
                res["models"].append({"name": name, "label": meta.get("label", name),
                                      "port": meta.get("port"), "gpus": meta.get("gpus"), "up": True})
            else:
                res["models"].append({"name": name, "label": name, "up": True})
        res["reachable"] = len(res["gpus"]) > 0
        if not res["reachable"]:
            res["err"] = "no GPUs reported"
    except Exception as e:  # noqa
        res["err"] = ("parse: %s" % e)[:140]
    return res


def _node_loop(node, offset):
    time.sleep(offset)
    interval = float(node.get("poll_interval") or CFG["defaults"]["poll_interval"])
    while True:
        try:
            r = poll_unified(node) if node.get("profile") == "unified" else poll_discrete(node)
        except Exception as e:  # noqa
            r = {"key": node["key"], "name": node.get("name"), "profile": node.get("profile"),
                 "reachable": False, "ts": time.time(), "err": str(e)[:140], "gpus": []}
        with _lock:
            STATE["nodes"][node["key"]] = r
            if r.get("reachable"):
                if r.get("profile") == "unified":
                    _push_hist("spark:%s:temp" % node["key"], r.get("temp"))
                    _push_hist("spark:%s:power" % node["key"], r.get("power"))
                else:
                    for g in r.get("gpus") or []:
                        _push_hist("gpu:%s:%s:temp" % (node["key"], g["index"]), g.get("temp"))
                        _push_hist("gpu:%s:%s:power" % (node["key"], g["index"]), g.get("power"))
        time.sleep(interval)


# ----------------------------------------------------------------------------
# Fabric switch (MikroTik RouterOS). One statement per call, no retry spam.
# Reach it with plain SSH (switch.host/user/ssh_key) or a custom command
# (switch.exec.argv with a "{cmd}" placeholder, e.g. an expect helper).
# ----------------------------------------------------------------------------
_iface_prev = {}
_port_rate = {}


def _switch_argv(statement):
    ex = (SWITCH or {}).get("exec")
    if ex and ex.get("argv"):
        return [a.replace("{cmd}", statement) for a in ex["argv"]], _abs(ex.get("cwd")) if ex.get("cwd") else None
    return ["ssh"] + _ssh_flags(SWITCH) + [_target(SWITCH), statement], None


def _sw(statement, timeout=30):
    argv, cwd = _switch_argv(statement)
    return _run(argv, timeout=timeout, cwd=cwd)


def _parse_health(text):
    h = {}
    for line in text.splitlines():
        mm = re.match(r"\s*\d+\s+([a-z0-9\-]+)\s+([0-9.]+|ok|fail|critical|warning)\b", line)
        if mm:
            h[mm.group(1)] = mm.group(2)
    return h


def _parse_resource(text):
    r = {}
    for line in text.splitlines():
        mm = re.match(r"\s*([a-z0-9\-]+):\s+(.+?)\s*$", line)
        if mm:
            r[mm.group(1)] = mm.group(2).strip()
    return r


def poll_switch():
    ports_cfg = SWITCH.get("ports") or []
    res = {"reachable": False, "ts": time.time(), "err": None,
           "health": {}, "resource": {}, "ports": [], "total_bps": 0}
    rc, out, err = _sw("/system health print")
    if rc != 0 or "NAME" not in out:
        res["err"] = (err or out or "switch unreachable").strip()[:140]
        return res
    res["health"] = _parse_health(out)
    rc, out, err = _sw("/system/resource/print")
    if rc == 0 and "version" in out:
        res["resource"] = _parse_resource(out)
    rc, out, err = _sw("/interface print stats where running")
    now = time.time()
    ports = {p: {"name": p, "running": False, "rx_bps": 0, "tx_bps": 0,
                 "rate": _port_rate.get(p)} for p in ports_cfg}
    if rc == 0 and "RX-BYTE" in out:
        for line in out.splitlines():
            for p in ports_cfg:
                if re.search(r"\b" + re.escape(p) + r"\b", line):
                    after = line.split(p, 1)[1].strip()
                    cols = re.split(r"\s{2,}", after)
                    vals = [int(c.replace(" ", "")) for c in cols if c.replace(" ", "").isdigit()]
                    if len(vals) >= 2:
                        rx, tx = vals[0], vals[1]
                        ports[p]["running"] = True
                        prev = _iface_prev.get(p)
                        if prev:
                            dt = now - prev[0]
                            if dt > 0:
                                ports[p]["rx_bps"] = max(0, (rx - prev[1]) * 8 / dt)
                                ports[p]["tx_bps"] = max(0, (tx - prev[2]) * 8 / dt)
                        _iface_prev[p] = (now, rx, tx)
                    break
        res["reachable"] = True
    for p in ports_cfg:
        if ports[p]["running"] and _port_rate.get(p) is None:
            rc2, out2, _ = _sw("/interface ethernet monitor %s once" % p)
            if rc2 == 0:
                rm = re.search(r"\brate:\s*([0-9A-Za-z]+)", out2)
                if rm:
                    _port_rate[p] = rm.group(1)
                    ports[p]["rate"] = rm.group(1)
            break
    res["ports"] = [ports[p] for p in ports_cfg]
    res["total_bps"] = sum(pp["rx_bps"] + pp["tx_bps"] for pp in res["ports"])
    return res


def _switch_loop():
    time.sleep(0.5)
    interval = float(SWITCH.get("poll_interval") or 8.0)
    while True:
        try:
            r = poll_switch()
        except Exception as e:  # noqa
            r = {"reachable": False, "ts": time.time(), "err": str(e)[:140],
                 "health": {}, "resource": {}, "ports": [], "total_bps": 0}
        with _lock:
            STATE["switch"] = r
        time.sleep(interval)


# ----------------------------------------------------------------------------
# Model servers: Prometheus /metrics + /v1/models (vLLM, SGLang, llama.cpp)
# ----------------------------------------------------------------------------
_model_prev = {}


def _prom_parse(text):
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


def _model_id(url, prefer=None, timeout=6, headers=None):
    ok, body = _http_get(url + "/v1/models", timeout=timeout, headers=headers)
    if not ok:
        return ""
    try:
        ids = [str(d.get("id", "") or "") for d in json.loads(body).get("data", [])]
        if prefer and prefer in ids:
            return prefer
        if ids:
            return ids[0]
    except Exception:  # noqa
        pass
    return ""


def _kv_pct(val):
    if val is None:
        return None
    return round(val * 100.0, 1) if val <= 1.0 else round(val, 1)


def _model_base(m):
    return {"key": m["key"], "label": m.get("label", m["key"]), "unit": m.get("unit", "fleet"),
            "port": m.get("port"), "gpus": m.get("gpus"), "node": m.get("node")}


def _auth_headers(m):
    k = m.get("api_key") or ""
    return {"Authorization": "Bearer " + k} if k else None


def poll_model(m):
    res = _model_base(m)
    res.update({"reachable": False, "engine": None, "model": None, "decode_tps": None,
                "prefill_tps": None, "ttft_ms": None, "kv_pct": None, "running": None,
                "waiting": None, "ts": time.time(), "err": None})
    url = m["endpoint"].rstrip("/")
    ok, body = _http_get(url + "/metrics", timeout=6, headers=_auth_headers(m))
    if not ok or not body.strip():
        res["err"] = "down / no /metrics"
        return res
    p = _prom_parse(body)
    now = time.time()
    prompt_tok = gen_tok = ttft_sum = ttft_cnt = None
    if any(k.startswith("vllm:") for k in p):
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
    elif any(k.startswith("llamacpp:") for k in p):
        res["engine"] = "llama.cpp"
        prompt_tok = p.get("llamacpp:prompt_tokens_total")
        gen_tok = p.get("llamacpp:tokens_predicted_total")
        res["kv_pct"] = _kv_pct(p.get("llamacpp:kv_cache_usage_ratio"))
        if "llamacpp:requests_processing" in p:
            res["running"] = int(p["llamacpp:requests_processing"])
        if "llamacpp:requests_deferred" in p:
            res["waiting"] = int(p["llamacpp:requests_deferred"])
        if p.get("llamacpp:predicted_tokens_seconds") is not None:
            res["decode_tps"] = round(p["llamacpp:predicted_tokens_seconds"], 1)
        if p.get("llamacpp:prompt_tokens_seconds") is not None:
            res["prefill_tps"] = round(p["llamacpp:prompt_tokens_seconds"], 1)
    elif any(k.startswith("sglang:") for k in p):
        res["engine"] = "SGLang"
        prompt_tok = p.get("sglang:prompt_tokens_total")
        gen_tok = p.get("sglang:generation_tokens_total")
        ttft_sum = p.get("sglang:time_to_first_token_seconds_sum")
        ttft_cnt = p.get("sglang:time_to_first_token_seconds_count")
        res["kv_pct"] = _kv_pct(p.get("sglang:token_usage"))
        if "sglang:num_running_reqs" in p:
            res["running"] = int(p["sglang:num_running_reqs"])
        if "sglang:num_queue_reqs" in p:
            res["waiting"] = int(p["sglang:num_queue_reqs"])
    else:
        res["err"] = "unknown /metrics format"
        return res
    res["reachable"] = True
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
    if res["decode_tps"] is None:
        res["decode_tps"] = 0.0
    if res["prefill_tps"] is None:
        res["prefill_tps"] = 0.0
    if res["ttft_ms"] is None and ttft_sum is not None and ttft_cnt and ttft_cnt > 0:
        res["ttft_ms"] = round(ttft_sum / ttft_cnt * 1000.0, 1)
    mid = _model_id(url, prefer=m.get("model"), timeout=6, headers=_auth_headers(m))
    res["model"] = mid or res["label"]
    return res


def _model_loop(m, offset):
    time.sleep(offset)
    interval = float(m.get("poll_interval") or CFG["defaults"]["model_poll_interval"])
    while True:
        try:
            r = poll_model(m)
        except Exception as e:  # noqa
            r = _model_base(m)
            r.update({"reachable": False, "ts": time.time(), "err": str(e)[:140]})
        with _lock:
            STATE["models"][m["key"]] = r
        time.sleep(interval)


# ----------------------------------------------------------------------------
# ComfyUI render lanes. Liveness + queue from ComfyUI; memory from the driver /
# kernel reading we already collect for that GPU (ComfyUI counts torch's cached
# blocks as free, which under-reports use).
# ----------------------------------------------------------------------------
_GIB = 1073741824


def poll_comfy(lane):
    out = {"key": lane["key"], "lane": lane.get("lane", ""), "name": lane.get("name", ""),
           "host": lane.get("host", ""), "url": lane["url"], "reachable": False,
           "ts": time.time(), "vram_total": None, "vram_free": None, "vram_used": None,
           "version": None, "running": 0, "pending": 0, "busy": False}
    ok, body = _http_get(lane["url"].rstrip("/") + "/system_stats", timeout=5)
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
            out["vram_total"], out["vram_free"], out["vram_used"] = tot, free, max(0, tot - free)
    ok2, qbody = _http_get(lane["url"].rstrip("/") + "/queue", timeout=5)
    if ok2:
        try:
            q = json.loads(qbody)
            out["running"] = len(q.get("queue_running") or [])
            out["pending"] = len(q.get("queue_pending") or [])
            out["busy"] = out["running"] > 0
        except Exception:
            pass
    return out


def _overlay_true_mem(r, lane):
    """Call with _lock held."""
    src = lane.get("src") or {}
    r["comfy_used"] = r.get("vram_used")
    r["mem_source"] = "comfyui"
    r["unified"] = False
    node = STATE["nodes"].get(src.get("node") or "") or {}
    if src.get("kind") == "gpu":
        for g in node.get("gpus") or []:
            if g.get("index") == src.get("index") and g.get("mem_used_mb") is not None:
                r["vram_used"] = g["mem_used_mb"] * 1048576
                r["vram_total"] = (g.get("mem_total_mb") or 0) * 1048576
                r["vram_free"] = max(0, r["vram_total"] - r["vram_used"])
                r["mem_source"] = "nvidia-smi"
                return
    elif src.get("kind") == "unified":
        if node.get("mem_used_gib") is not None:
            r["vram_used"] = node["mem_used_gib"] * _GIB
            r["vram_total"] = (node.get("mem_total_gib") or 0) * _GIB
            r["vram_free"] = max(0, r["vram_total"] - r["vram_used"])
            r["mem_source"] = "unified /proc/meminfo"
            r["unified"] = True


def _comfy_loop(lane, delay):
    time.sleep(delay)
    while True:
        try:
            r = poll_comfy(lane)
        except Exception as e:  # noqa
            r = {"key": lane["key"], "lane": lane.get("lane"), "name": lane.get("name"),
                 "host": lane.get("host"), "url": lane["url"], "reachable": False,
                 "ts": time.time(), "err": str(e)[:140], "running": 0, "pending": 0, "busy": False}
        with _lock:
            _overlay_true_mem(r, lane)
            prev = STATE["comfy"].get(lane["key"]) or {}
            for fld, gate in (("peak_used", False), ("peak_busy", True)):
                old = prev.get(fld)
                cur = r.get("vram_used")
                if gate and not r.get("busy"):
                    cur = None
                r[fld] = max(old or 0, cur) if cur is not None else old
            r["peak_since"] = prev.get("peak_since") or time.time()
            STATE["comfy"][lane["key"]] = r
        time.sleep(float(CFG.get("comfy_poll_seconds") or 4.0))


# ----------------------------------------------------------------------------
# Token tracker. mode "bank": poll each model's /metrics counters and bank the
# deltas across server restarts into tokens.store. mode "read": only read a store
# another process maintains (so two dashboards never double-bank).
# ----------------------------------------------------------------------------
TOK = CFG["tokens"]
TOK_STORE = _abs(TOK.get("store") or "data/token_usage.json")


def _token_models():
    if TOK.get("models"):
        return [(t["key"], t.get("name", t["key"]), t["metrics_url"]) for t in TOK["models"]]
    return [(m["key"], m.get("label", m["key"]), m["endpoint"].rstrip("/") + "/metrics")
            for m in MODELS if m.get("track_tokens", True)]


def _tok_scrape(url):
    ok, body = _http_get(url, timeout=8)
    if not ok:
        return None
    prompt = gen = None
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        for pre, which in (("vllm:prompt_tokens_total", "p"), ("sglang:prompt_tokens_total", "p"),
                           ("llamacpp:prompt_tokens_total", "p"),
                           ("vllm:generation_tokens_total", "g"), ("sglang:generation_tokens_total", "g"),
                           ("llamacpp:tokens_predicted_total", "g")):
            if line.startswith(pre):
                try:
                    v = float(line.rsplit(" ", 1)[1])
                except Exception:
                    continue
                if which == "p":
                    prompt = v
                else:
                    gen = v
    if prompt is None and gen is None:
        return None
    return (prompt or 0.0, gen or 0.0)


def _tok_bank(rec, cur, ft, fl, fd):
    last = rec.get(fl)
    if last is None:
        rec[ft] = cur
        rec[fl] = cur
        rec.setdefault(fd, 0.0)
        return
    delta = (cur - last) if cur >= last else cur
    rec[ft] = rec.get(ft, 0.0) + delta
    rec[fd] = rec.get(fd, 0.0) + delta
    rec[fl] = cur


def _tok_loop():
    try:
        with open(TOK_STORE) as f:
            state = json.load(f)
    except Exception:
        state = {}
    listed = {k for k, _, _ in _token_models()}
    for k in [k for k in list(state) if not k.startswith("_") and k not in listed]:
        rec = state.pop(k)
        rec["retired_on"] = datetime.date.today().isoformat()
        state.setdefault("_retired", {})[k] = rec
    while True:
        try:
            today = datetime.date.today().isoformat()
            for key, name, url in _token_models():
                rec = state.setdefault(key, {"name": name})
                rec["name"] = name
                if rec.get("today_date") != today:
                    rec["today_date"], rec["today_prompt"], rec["today_gen"] = today, 0.0, 0.0
                scr = _tok_scrape(url)
                rec["reachable"] = scr is not None
                if scr is not None:
                    _tok_bank(rec, scr[0], "total_prompt", "last_prompt", "today_prompt")
                    _tok_bank(rec, scr[1], "total_gen", "last_gen", "today_gen")
                    rec["total_tokens"] = rec.get("total_prompt", 0) + rec.get("total_gen", 0)
                rec["today_tokens"] = rec.get("today_prompt", 0) + rec.get("today_gen", 0)
            state["_updated"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            os.makedirs(os.path.dirname(TOK_STORE), exist_ok=True)
            tmp = TOK_STORE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, TOK_STORE)
        except Exception as e:  # noqa
            sys.stderr.write("token tracker: %r\n" % e)
        time.sleep(float(TOK.get("poll_seconds") or 120))


def read_tokens():
    try:
        with open(TOK_STORE) as f:
            return json.load(f)
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Snapshot (shape kept compatible with the v1 /api/metrics for API consumers)
# ----------------------------------------------------------------------------
def start_pollers():
    for i, n in enumerate(NODES):
        threading.Thread(target=_node_loop, args=(n, i * 1.2), daemon=True).start()
    if SWITCH:
        threading.Thread(target=_switch_loop, daemon=True).start()
    for i, ln in enumerate(COMFY_LANES):
        STATE["comfy"][ln["key"]] = {"key": ln["key"], "lane": ln.get("lane"), "name": ln.get("name"),
                                     "host": ln.get("host"), "url": ln["url"], "reachable": False,
                                     "ts": 0, "err": "warming up", "running": 0, "pending": 0, "busy": False}
        threading.Thread(target=_comfy_loop, args=(ln, 0.6 + i * 0.4), daemon=True).start()
    for i, m in enumerate(MODELS):
        r = _model_base(m)
        r.update({"reachable": False, "ts": 0, "err": "warming up"})
        STATE["models"][m["key"]] = r
        threading.Thread(target=_model_loop, args=(m, i * 0.7), daemon=True).start()
    if TOK.get("enabled", True) and TOK.get("mode", "bank") == "bank" and _token_models():
        threading.Thread(target=_tok_loop, daemon=True).start()


def snapshot():
    with _lock:
        nodes = [dict(STATE["nodes"][n["key"]]) for n in NODES]
        switch = dict(STATE["switch"]) if SWITCH else None
        hist = {k: list(v) for k, v in _hist.items()}
        models = [dict(STATE["models"][m["key"]]) for m in MODELS if m["key"] in STATE["models"]]
    sd = CFG["defaults"]
    unified = [x for x in nodes if x.get("profile") == "unified"]
    discrete = [x for x in nodes if x.get("profile") != "unified"]
    gpu_count = len([s for s in unified if s.get("reachable")]) + sum(len(b.get("gpus") or []) for b in discrete)
    total_power, hottest, all_ok = 0.0, {"unit": None, "temp": -1}, True
    down = []
    for s in unified:
        if s.get("reachable"):
            total_power += s.get("power") or 0
            if s.get("temp") is not None and s["temp"] > hottest["temp"]:
                hottest = {"unit": s["name"], "temp": s["temp"]}
        else:
            all_ok = False
            down.append(s["name"])
    for b in discrete:
        for g in b.get("gpus") or []:
            total_power += g.get("power") or 0
            if g.get("temp") is not None and g["temp"] > hottest["temp"]:
                hottest = {"unit": "%s GPU%s" % (b.get("name"), g["index"]), "temp": g["temp"]}
        if not b.get("reachable"):
            all_ok = False
            down.append(b.get("name"))
    if switch is not None and not switch.get("reachable"):
        all_ok = False
        down.append((SWITCH or {}).get("name", "switch"))
    legacy_box = discrete[0] if discrete else {"reachable": False, "gpus": []}
    if discrete:
        for g in legacy_box.get("gpus") or []:
            for f in ("temp", "power"):
                k = "gpu:%s:%s:%s" % (legacy_box["key"], g["index"], f)
                if k in hist:
                    hist["box:%s:%s" % (g["index"], f)] = hist[k]

    def by_unit(u):
        return [m for m in models if m.get("unit") == u]

    return {
        "ts": time.time(),
        "version": VERSION,
        "read_only": READ_ONLY,
        "nodes": nodes,
        "sparks": unified,             # v1 name, kept for API consumers
        "switch": switch or {"reachable": False, "ts": 0, "err": "not configured"},
        "box": legacy_box,             # v1 name: the first discrete host
        "history": hist,
        "glm_model": next((n.get("serving") for n in NODES if n.get("serving")), None),
        "models": models,
        "spark_models": by_unit("spark"),
        "box_models": by_unit("box"),
        "ds4_models": by_unit("ds4"),
        "glm53big_models": by_unit("glm53big"),
        "proxy": {"reachable": False, "ts": 0, "err": "not polled", "backends": {},
                  "sessions_pinned": None, "sessions": [], "lanes": []},
        "agg": {"gpu_count": gpu_count, "total_power": round(total_power),
                "hottest_unit": hottest["unit"],
                "hottest_temp": hottest["temp"] if hottest["temp"] >= 0 else None,
                "all_ok": all_ok, "down": down},
        "thresholds": {"temp_warn": sd["temp_warn"], "temp_hot": sd["temp_hot"],
                       "stale_after_s": sd["stale_after_s"]},
    }


def comfy_snapshot():
    with _lock:
        return [dict(STATE["comfy"].get(l["key"], {})) for l in COMFY_LANES]


_stations_cache = {"ts": 0, "data": []}


def stations_snapshot():
    st = CFG["stations"]
    if not st.get("enabled"):
        return []
    if time.time() - _stations_cache["ts"] < 10:
        return _stations_cache["data"]
    out = []
    for s in st.get("items") or []:
        code = _http_status(s.get("probe_url") or s["url"], timeout=3)
        out.append({"emoji": s.get("emoji", ""), "name": s["name"], "desc": s.get("desc", ""),
                    "port": s.get("port"), "url": s["url"], "up": code != 0})
    _stations_cache.update(ts=time.time(), data=out)
    return out


# ----------------------------------------------------------------------------
# Chat: an SSE proxy to any OpenAI-compatible /v1/chat/completions endpoint.
# ----------------------------------------------------------------------------
CHAT = CFG["chat"]


def chat_endpoints():
    eps = []
    if CHAT.get("base_url") and CHAT.get("model"):
        eps.append({"base_url": CHAT["base_url"], "model": CHAT["model"],
                    "api_key": CHAT.get("api_key") or ""})
    for fb in CHAT.get("fallbacks") or []:
        if fb.get("base_url") and fb.get("model"):
            eps.append({"base_url": fb["base_url"], "model": fb["model"],
                        "api_key": fb.get("api_key") or ""})
    return eps


def _ep_headers(ep, stream=False):
    h = {"Content-Type": "application/json", "User-Agent": "command-center-v2"}
    if stream:
        h["Accept"] = "text/event-stream"
    if ep.get("api_key"):
        h["Authorization"] = "Bearer " + ep["api_key"]
    return h


_chat_probe = {"ts": 0, "data": None}


def chat_status(force=False):
    if not CHAT.get("enabled", True):
        return {"enabled": False, "configured": False, "reachable": False, "name": CHAT.get("name")}
    eps = chat_endpoints()
    if not eps:
        return {"enabled": True, "configured": False, "reachable": False, "name": CHAT.get("name")}
    if not force and _chat_probe["data"] and time.time() - _chat_probe["ts"] < 20:
        return _chat_probe["data"]
    out = {"enabled": True, "configured": True, "reachable": False, "name": CHAT.get("name"),
           "model": eps[0]["model"], "active_model": None, "fallback": False,
           "endpoints": len(eps)}
    for i, ep in enumerate(eps):
        try:
            req = urllib.request.Request(ep["base_url"].rstrip("/") + "/models", headers=_ep_headers(ep))
            with urllib.request.urlopen(req, timeout=5) as r:
                r.read(65536)
            out.update(reachable=True, active_model=ep["model"], fallback=i > 0)
            break
        except urllib.error.HTTPError as e:
            # Some agent APIs have no /models route; any HTTP answer means it is listening.
            if e.code in (404, 405):
                out.update(reachable=True, active_model=ep["model"], fallback=i > 0)
                break
        except Exception:
            continue
    _chat_probe.update(ts=time.time(), data=out)
    return out


def _tz():
    name = CFG["server"].get("timezone") or ""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name) if name else None
    except Exception:
        return None


def _now_local():
    tz = _tz()
    return datetime.datetime.now(tz) if tz else datetime.datetime.now().astimezone()


def _age(ts):
    return int(time.time() - ts) if ts else None


def fleet_context():
    """A compact plain-text read of the live dashboard for the chat's system prompt."""
    s = snapshot()
    lines = []
    agg = s["agg"]
    lines.append("FLEET: %s. %s GPUs online, %s W total draw, hottest %s at %s C."
                 % ("ALL OK" if agg["all_ok"] else "DEGRADED (down: %s)" % ", ".join(agg["down"]),
                    agg["gpu_count"], agg["total_power"], agg["hottest_unit"] or "n/a",
                    "%.0f" % agg["hottest_temp"] if agg["hottest_temp"] is not None else "n/a"))
    for n in s["nodes"]:
        if not n.get("reachable"):
            lines.append("- NODE %s: UNREACHABLE (%s), last good poll %ss ago."
                         % (n["name"], n.get("err") or "no response", _age(n.get("ts"))))
            continue
        if n.get("profile") == "unified":
            lines.append("- NODE %s (unified-memory GPU%s): %s C, %s W, util %s%%, SM %s MHz, memory %s/%s GiB (%s%%), model resident %s GiB, serving %s%s."
                         % (n["name"], ", " + n["node_id"] if n.get("node_id") else "",
                            n.get("temp"), n.get("power"), n.get("util"), n.get("sm_clock"),
                            n.get("mem_used_gib"), n.get("mem_total_gib"), n.get("mem_pct"),
                            n.get("model_gib"), n.get("model") or "nothing",
                            " (%s)" % n["pair"] if n.get("pair") else ""))
        else:
            gp = "; ".join("GPU%s %s %s C %s/%s W util %s%% VRAM %.1f/%.1f GiB"
                           % (g["index"], g.get("name"), g.get("temp"), g.get("power"),
                              g.get("power_limit"), g.get("util"),
                              (g.get("mem_used_mb") or 0) / 1024, (g.get("mem_total_mb") or 0) / 1024)
                           for g in n.get("gpus") or [])
            cont = ", ".join(m.get("label") for m in n.get("models") or [])
            lines.append("- NODE %s (%s GPUs): CPU %s C, RAM %s%%. %s.%s"
                         % (n["name"], len(n.get("gpus") or []), n.get("cpu_temp"), n.get("mem_pct"),
                            gp, " Containers: %s." % cont if cont else ""))
    if SWITCH:
        sw = s["switch"]
        if sw.get("reachable"):
            h = sw.get("health") or {}
            lines.append("- SWITCH %s: up, switch %s C, cpu %s C, fans %s, PSU1 %s PSU2 %s, uptime %s, fabric %s. Ports: %s."
                         % (SWITCH.get("name", "switch"), h.get("switch-temperature"), h.get("cpu-temperature"),
                            h.get("fan-state"), h.get("psu1-state"), h.get("psu2-state"),
                            (sw.get("resource") or {}).get("uptime"), _fmt_bps(sw.get("total_bps")),
                            ", ".join("%s %s" % (p["name"], "link-ok" if p.get("running") else "DOWN")
                                      for p in sw.get("ports") or [])))
        else:
            lines.append("- SWITCH %s: UNREACHABLE (%s)." % (SWITCH.get("name", "switch"), sw.get("err")))
    for m in s["models"]:
        if m.get("reachable"):
            lines.append("- MODEL %s [%s, %s, port %s]: UP, decode %s tok/s, prefill %s tok/s, TTFT %s ms, KV %s%%, %s running, %s waiting."
                         % (m["label"], m.get("model"), m.get("engine"), m.get("port"), m.get("decode_tps"),
                            m.get("prefill_tps"), m.get("ttft_ms"), m.get("kv_pct"), m.get("running"),
                            m.get("waiting")))
        else:
            lines.append("- MODEL %s [port %s]: DOWN (%s)." % (m["label"], m.get("port"), m.get("err")))
    for l in comfy_snapshot():
        if not l:
            continue
        lines.append("- RENDER LANE %s %s (ComfyUI): %s, queue %s, node memory %s/%s GB."
                     % (l.get("lane"), l.get("name"),
                        "DOWN (ComfyUI is not answering on its port)" if not l.get("reachable")
                        else ("up, RENDERING" if l.get("busy") else "up, idle"),
                        (l.get("running") or 0) + (l.get("pending") or 0),
                        "%.1f" % (l["vram_used"] / _GIB) if l.get("vram_used") is not None else "?",
                        "%.1f" % (l["vram_total"] / _GIB) if l.get("vram_total") else "?"))
    if TOK.get("enabled", True):
        t = read_tokens()
        for k, v in t.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            lines.append("- TOKENS %s: %s total (%s prompt, %s generated), %s today, %s."
                         % (v.get("name", k), _fmt_tok(v.get("total_tokens")), _fmt_tok(v.get("total_prompt")),
                            _fmt_tok(v.get("total_gen")), _fmt_tok(v.get("today_tokens")),
                            "live" if v.get("reachable") else "offline"))
    stations = stations_snapshot()
    for st in stations:
        lines.append("- STATION %s: %s." % (st["name"], "up" if st["up"] else "DOWN"))
    # Precomputed totals + one down-list at the top, so the model never has to count.
    lanes = [l for l in comfy_snapshot() if l]
    down = ["node %s" % n["name"] for n in s["nodes"] if not n.get("reachable")]
    if SWITCH and not s["switch"].get("reachable"):
        down.append("switch %s" % SWITCH.get("name", "switch"))
    down += ["model %s" % m["label"] for m in s["models"] if not m.get("reachable")]
    down += ["render lane %s (%s)" % (l.get("lane"), l.get("name")) for l in lanes if not l.get("reachable")]
    down += ["station %s" % st["name"] for st in stations if not st["up"]]
    counts = ["%d nodes (%d answering)" % (len(s["nodes"]), sum(1 for n in s["nodes"] if n.get("reachable")))]
    if SWITCH:
        counts.append("1 switch (%s)" % ("up" if s["switch"].get("reachable") else "down"))
    counts.append("%d model servers (%d up)" % (len(s["models"]), sum(1 for m in s["models"] if m.get("reachable"))))
    if lanes:
        counts.append("%d render lanes (%d up, %d rendering)" % (len(lanes), sum(1 for l in lanes if l.get("reachable")), sum(1 for l in lanes if l.get("busy"))))
    if stations:
        counts.append("%d web stations (%d up)" % (len(stations), sum(1 for st in stations if st["up"])))
    head = ["TOTALS (use these numbers, do not count yourself): " + ", ".join(counts) + ".",
            "DOWN RIGHT NOW (%d): %s." % (len(down), "; ".join(down) if down else "nothing")]
    return "\n".join(head + lines)


def _fmt_bps(b):
    b = float(b or 0)
    for unit, div in (("Gbps", 1e9), ("Mbps", 1e6), ("Kbps", 1e3)):
        if b >= div:
            return "%.2f %s" % (b / div, unit)
    return "%.0f bps" % b


def _fmt_tok(n):
    n = float(n or 0)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return "%.2f%s" % (n / div, unit)
    return "%d" % n


def build_system_prompt():
    name = CHAT.get("name") or "Jarvis"
    title = CFG["server"].get("title") or "Command Center"
    now = _now_local()
    persona = CHAT.get("system_prompt") or (
        "You are %s, the assistant built into the %s, a read-only dashboard for a self-hosted AI fleet. "
        "Answer questions about the fleet from the live dashboard data below." % (name, title))
    rules = (
        "Rules: be direct and short, plain English. Never invent a number, model name or status: if the "
        "data below does not have it, say the dashboard does not report it. Anything marked DOWN or UNREACHABLE counts as down; do "
        "not explain it away or guess why. When something is down, say which one and the error text "
        "you were given. The dashboard is read-only: you cannot restart, "
        "stop or change anything, so say so if asked and suggest what a human could check. Do not use em "
        "dashes. Do not use markdown headings; short lists are fine.")
    ctx = ""
    if CHAT.get("grounding", True):
        try:
            ctx = "\n\nLIVE DASHBOARD DATA (read %s):\n%s" % (now.strftime("%I:%M:%S %p %Z"), fleet_context())
        except Exception as e:  # noqa
            ctx = "\n\n(The live data could not be read: %s)" % e
    return "%s\n\n%s\n\nIt is currently %s.%s" % (persona, rules, now.strftime("%A, %B %d, %Y %I:%M %p %Z"), ctx)


class ThinkStripper:
    """Removes <think>...</think> reasoning from a streamed reply. If a closing tag shows up
    without an opening one (templates that open the block inside the prompt), everything before
    it was reasoning: the caller gets reset=True and the text after the tag."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""
        self.inside = False
        self.started = False

    def feed(self, text):
        self.buf += text
        out, reset = "", False
        while self.buf:
            if self.inside:
                i = self.buf.find(self.CLOSE)
                if i < 0:
                    keep = len(self.CLOSE) - 1
                    self.buf = self.buf[-keep:] if len(self.buf) > keep else self.buf
                    return out, reset, True
                self.buf = self.buf[i + len(self.CLOSE):].lstrip()
                self.inside = False
                continue
            io, ic = self.buf.find(self.OPEN), self.buf.find(self.CLOSE)
            if ic >= 0 and (io < 0 or ic < io):
                reset = True
                out = ""
                self.buf = self.buf[ic + len(self.CLOSE):].lstrip()
                continue
            if io >= 0:
                out += self.buf[:io]
                self.buf = self.buf[io + len(self.OPEN):]
                self.inside = True
                continue
            # hold back a tail that could be the start of a tag
            hold = 0
            for tag in (self.OPEN, self.CLOSE):
                for k in range(len(tag) - 1, 0, -1):
                    if self.buf.endswith(tag[:k]):
                        hold = max(hold, k)
                        break
            out += self.buf[: len(self.buf) - hold]
            self.buf = self.buf[len(self.buf) - hold:]
            break
        return out, reset, False

    def flush(self):
        out = "" if self.inside else self.buf
        self.buf = ""
        return out


def open_chat_stream(messages):
    """Try each configured endpoint in order. A connection failure moves on to the next; an HTTP
    error is returned as-is (the model answered and said no)."""
    sysmsg = build_system_prompt()
    body_msgs = [{"role": "system", "content": sysmsg}]
    turns = [m for m in (messages or []) if m.get("role") in ("user", "assistant") and m.get("content")]
    for m in turns[-int(CHAT.get("history_turns") or 12):]:
        body_msgs.append({"role": m["role"], "content": str(m["content"])[:8000]})
    last_err = None
    for i, ep in enumerate(chat_endpoints()):
        payload = {"model": ep["model"], "messages": body_msgs, "stream": True,
                   "temperature": float(CHAT.get("temperature", 0.1)),
                   "max_tokens": int(CHAT.get("max_tokens") or 1200)}
        payload.update(CHAT.get("extra_body") or {})
        req = urllib.request.Request(ep["base_url"].rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers=_ep_headers(ep, stream=True), method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=float(CHAT.get("timeout") or 180))
            return resp, ep, i > 0, None
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                detail = ""
            return None, ep, i > 0, "HTTP %s from %s: %s" % (e.code, ep["model"], detail)
        except Exception as e:  # noqa
            last_err = "%s: %s" % (type(e).__name__, e)
            continue
    return None, None, False, last_err or "no chat endpoint configured"


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
STATIC = _abs(CFG["server"].get("static_dir") or "web/dist")


def public_config():
    """What the browser needs to lay the page out. No hosts, keys or endpoints."""
    sv = CFG["server"]
    sections = CFG.get("sections") or []
    if not sections:
        uni = [n["key"] for n in NODES if n.get("profile") == "unified"]
        dis = [n["key"] for n in NODES if n.get("profile") != "unified"]
        if uni:
            sections.append({"key": "unified", "title": "GPU cluster", "nodes": uni, "switch": True, "units": ["spark"]})
        for k in dis:
            sections.append({"key": k, "title": NODE_BY_KEY[k].get("name", k), "nodes": [k], "units": [k]})
        units = {m.get("unit", "fleet") for m in MODELS}
        used = {u for s in sections for u in s.get("units", [])}
        for u in sorted(units - used):
            sections.append({"key": "models-" + u, "title": "Models", "nodes": [], "units": [u]})
    return {
        "version": VERSION,
        "title": sv.get("title"), "subtitle": sv.get("subtitle"), "location": sv.get("location"),
        "timezone": sv.get("timezone") or None,
        "refresh_ms": int(sv.get("browser_refresh_ms") or 2500),
        "read_only": READ_ONLY,
        "sections": sections,
        "nodes": [{"key": n["key"], "name": n.get("name", n["key"]), "profile": n.get("profile", "discrete"),
                   "badge": n.get("badge"), "temp_warn": n.get("temp_warn"), "temp_hot": n.get("temp_hot")}
                  for n in NODES],
        "switch": {"name": SWITCH.get("name", "Fabric switch"), "badge": SWITCH.get("badge", "FABRIC"),
                   "temp_warn": SWITCH.get("temp_warn", 55), "temp_hot": SWITCH.get("temp_hot", 70)} if SWITCH else None,
        "models": [{"key": m["key"], "unit": m.get("unit", "fleet"), "node": m.get("node")} for m in MODELS],
        "comfy": {"enabled": bool(COMFY_LANES), "title": CFG.get("comfy_title") or "Render lanes"},
        "tokens": {"enabled": bool(TOK.get("enabled", True)), "order": TOK.get("order") or []},
        "eco": {"enabled": bool(CFG["eco"].get("enabled")),
                "writes": bool(CFG["eco"].get("allow_writes")) and not READ_ONLY,
                "levels": CFG["eco"].get("levels") or [],
                "nodes": [{"key": n["key"], "name": n.get("name", n["key"])} for n in NODES if n.get("profile") == "unified"],
                "info_url": CFG["eco"].get("info_url") or ""},
        "stations": {"enabled": bool(CFG["stations"].get("enabled"))},
        "keylights": {"enabled": bool(CFG["keylights"].get("enabled")),
                      "writes": bool(CFG["keylights"].get("allow_writes")) and not READ_ONLY},
        "chat": {"enabled": bool(CHAT.get("enabled", True)), "name": CHAT.get("name") or "Jarvis",
                 "suggestions": CHAT.get("suggestions") or []},
    }


def _allowed_hosts():
    hs = {"localhost", "127.0.0.1", "[::1]"}
    for b in CFG["server"]["bind"]:
        if b not in ("0.0.0.0", "::"):
            hs.add(b.lower())
    for h in CFG["server"].get("allowed_hosts") or []:
        hs.add(h.lower())
    return hs


class Handler(BaseHTTPRequestHandler):
    server_version = "CommandCenter/" + VERSION

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", cache="no-store"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def _same_origin(self, need_json=True):
        host = (self.headers.get("Host") or "").strip().lower()
        hostname = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        if "0.0.0.0" not in CFG["server"]["bind"] and hostname not in _allowed_hosts():
            return False, "host %r is not allowed" % host
        if (self.headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
            return False, "cross-site request"
        origin = self.headers.get("Origin")
        if origin is not None and urllib.parse.urlparse(origin.strip()).netloc.lower() != host:
            return False, "origin does not match host"
        if need_json and "application/json" not in (self.headers.get("Content-Type") or "").lower():
            return False, "content type must be application/json"
        return True, None

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n > 256 * 1024:
            return None
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or b"{}")
        except Exception:
            return None

    # -------------------------------------------------------------- GET
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/healthz":
            return self._send(200, "ok", "text/plain")
        if path == "/api/metrics":
            return self._json(snapshot())
        if path == "/api/config":
            return self._json(public_config())
        if path == "/api/comfy":
            return self._json({"lanes": comfy_snapshot()})
        if path == "/api/tokens":
            return self._json(read_tokens() if TOK.get("enabled", True) else {})
        if path == "/api/stations":
            return self._json({"stations": stations_snapshot()})
        if path == "/api/lights":
            kl = CFG["keylights"]
            if not kl.get("enabled") or not kl.get("url"):
                return self._json({"lights": [], "down": True, "disabled": True})
            ok, text = _http_get(kl["url"].rstrip("/") + "/api/lights", timeout=10)
            if not ok:
                return self._json({"lights": [], "down": True})
            return self._send(200, text or '{"lights":[]}')
        if path == "/api/eco-status":
            return self._eco_status()
        if path == "/api/eco-set":
            return self._json({"ok": False, "error": "use POST /api/eco-set"}, 405)
        if path == "/api/chat/status":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            return self._json(chat_status(force=q.get("force", ["0"])[0] == "1"))
        if path.startswith("/api/"):
            return self._json({"ok": False, "error": "not found"}, 404)
        return self._static(path)

    def _static(self, path):
        if not os.path.isdir(STATIC):
            return self._send(503, "<h1>Web UI not built</h1><p>Run <code>cd web && npm install && npm run build</code>.</p>",
                              "text/html; charset=utf-8")
        rel = os.path.normpath(urllib.parse.unquote(path)).lstrip("/")
        full = os.path.join(STATIC, rel)
        if not full.startswith(STATIC) or not rel or not os.path.isfile(full):
            full = os.path.join(STATIC, "index.html")
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        cache = "public, max-age=31536000, immutable" if "/assets/" in full else "no-store"
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype, cache)

    def _eco_status(self):
        eco = CFG["eco"]
        if not eco.get("enabled"):
            return self._json({"status": {}, "disabled": True})
        out = {}

        def one(n):
            try:
                rc, o, e = _run(remote_argv(n, "nvidia-smi --query-gpu=clocks.gr,temperature.gpu,power.draw --format=csv,noheader"), timeout=20)
                v = (o or e or "").strip()
                out[n["key"]] = v.split("\n")[0][:60] if v else "no reply"
            except Exception as ex:  # noqa
                out[n["key"]] = "err: %s" % str(ex)[:40]
        ths = [threading.Thread(target=one, args=(n,)) for n in NODES if n.get("profile") == "unified"]
        [t.start() for t in ths]
        [t.join(28) for t in ths]
        return self._json({"status": out})

    # -------------------------------------------------------------- POST
    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        ok, why = self._same_origin()
        if not ok:
            return self._json({"ok": False, "error": "same-origin only (%s)" % why}, 403)
        if path == "/api/chat":
            return self._chat()
        if path in ("/api/eco-set", "/api/lights-set", "/api/ds4-move"):
            return self._write_route(path)
        return self._json({"ok": False, "error": "not found"}, 404)

    def _write_route(self, path):
        """Every route that changes something on another machine. Refused in read-only mode."""
        body = self._body() or {}
        if READ_ONLY:
            return self._json({"ok": False, "read_only": True,
                               "error": "This dashboard is running read-only. Set server.read_only=false in config.json to enable actions."}, 403)
        if path == "/api/lights-set":
            kl = CFG["keylights"]
            if not (kl.get("enabled") and kl.get("allow_writes")):
                return self._json({"ok": False, "error": "key light control is disabled"}, 403)
            code, text = _http_post(kl["url"].rstrip("/") + "/api/set", json.dumps(body).encode(), timeout=10)
            if code == 0:
                return self._json({"ok": False, "error": "key light panel unreachable"}, 502)
            return self._send(code, text or "{}")
        if path == "/api/eco-set":
            eco = CFG["eco"]
            if not (eco.get("enabled") and eco.get("allow_writes")):
                return self._json({"ok": False, "error": "clock caps are disabled"}, 403)
            levels = {str(l["value"]): l.get("arg") for l in eco.get("levels") or [] if l.get("value") != "off"}
            level, node = str(body.get("level", "")), body.get("node", "fleet")
            if level != "off" and level not in levels:
                return self._json({"ok": False, "error": "bad level"}, 400)
            targets = [n for n in NODES if n.get("profile") == "unified" and (node == "fleet" or n["key"] == node)]
            if not targets:
                return self._json({"ok": False, "error": "bad node"}, 400)
            cmd = "sudo nvidia-smi -rgc" if level == "off" else "sudo nvidia-smi -lgc " + levels[level]
            out = {}

            def one(n):
                rc, o, e = _run(remote_argv(n, cmd), timeout=30)
                out[n["key"]] = ((o or e or "ok").strip())[:120]
            ths = [threading.Thread(target=one, args=(n,)) for n in targets]
            [t.start() for t in ths]
            [t.join(35) for t in ths]
            return self._json({"ok": True, "applied": level, "nodes": out})
        return self._json({"ok": False, "error": "this action is not available in v2"}, 410)

    # -------------------------------------------------------------- chat SSE
    def _sse_start(self, extra=None):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def _chat(self):
        body = self._body()
        if body is None:
            return self._json({"ok": False, "error": "bad JSON"}, 400)
        if not CHAT.get("enabled", True):
            return self._json({"ok": False, "error": "chat is disabled"}, 403)
        if not chat_endpoints():
            return self._json({"ok": False, "configured": False,
                               "error": "No model is configured. Set chat.base_url and chat.model in config.json (or CC_CHAT_BASE_URL / CC_CHAT_MODEL)."}, 503)
        resp, ep, fallback, err = open_chat_stream(body.get("messages") or [])
        self._sse_start()
        if resp is None:
            self._sse({"error": err, "delta": "I could not reach the model (%s)." % err})
            self.wfile.write(b"data: [DONE]\n\n")
            return
        strip = ThinkStripper()
        got, thinking = False, False
        try:
            self._sse({"meta": {"model": ep["model"], "fallback": fallback}})
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                d = choices[0].get("delta") or {}
                if (d.get("reasoning_content") or d.get("reasoning")) and not got and not thinking:
                    thinking = True
                    self._sse({"status": "thinking"})
                text = d.get("content")
                if not text:
                    continue
                out, reset, inside = strip.feed(text)
                if reset:
                    got = False
                    self._sse({"reset": True})
                if inside and not thinking and not got:
                    thinking = True
                    self._sse({"status": "thinking"})
                if out:
                    if not got:
                        out = out.lstrip()
                        if not out:
                            continue
                    got = True
                    self._sse({"delta": out})
            tail = strip.flush()
            if tail:
                got = True
                self._sse({"delta": tail})
            if not got:
                self._sse({"delta": "(The model finished without a text reply.)"})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:  # noqa
            try:
                self._sse({"delta": "\n[stream error: %s]" % e})
            except Exception:
                pass
        finally:
            try:
                resp.close()
            except Exception:
                pass


def _resolve_expect():
    """Resolve a bare 'expect' in switch.exec.argv to an installed binary."""
    ex = (SWITCH or {}).get("exec") or {}
    argv = ex.get("argv") or []
    if argv and argv[0] == "expect":
        for cand in ("/opt/homebrew/bin/expect", "/usr/local/bin/expect", "/usr/bin/expect"):
            if os.path.exists(cand):
                argv[0] = cand
                break


def main():
    _resolve_expect()
    start_pollers()
    port = int(CFG["server"]["port"])
    binds = CFG["server"]["bind"] or ["127.0.0.1"]
    servers = []
    for addr in binds:
        for attempt in range(30):
            try:
                servers.append(ThreadingHTTPServer((addr, port), Handler))
                break
            except OSError as e:
                if attempt == 29:
                    sys.stderr.write("could not bind %s:%s (%s)\n" % (addr, port, e))
                time.sleep(2)
    if not servers:
        sys.exit(1)
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print("Command Center v%s on %s (port %s) config=%s read_only=%s chat=%s"
          % (VERSION, ", ".join(binds), port, CFG_PATH or "(defaults)", READ_ONLY,
             "configured" if chat_endpoints() else "not configured"), flush=True)
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
