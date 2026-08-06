# Sparky Command Center

A lightweight, dependency-free command center for a **DGX Spark fleet** (and any GPU boxes you add alongside it, like 3090 rigs). Monitor the whole cluster over SSH on one glassmorphic page: live GPU / CPU / memory / temperature / power, an optional RoCE fabric-switch panel, and per-model decode, prefill, and TTFT.

Point it at your own hardware with a single `config.json`. No database, no build step, no third-party packages — just Python 3.8+ and SSH.

> Repo: **`sparky-command-center`**

## Features

- **Fleet at a glance** — one page with a summary strip (online GPUs, total power draw, hottest GPU, overall fleet status).
- **Per-GPU hardware metrics** — temperature, power draw and % of cap, utilization, VRAM used/total, fan speed, and graphics clock, with a live temperature sparkline per GPU.
- **Per-node system metrics** — CPU temperature (AMD `k10temp`, Intel `coretemp`, ARM `cpu_thermal`, and more), extra CPU sensors, and system RAM. Works for DGX Sparks and any other Linux GPU host.
- **Per-model inference metrics** — decode tok/s, prefill tok/s, TTFT, KV-cache usage, and running/waiting request counts, scraped from each model's Prometheus `/metrics`. Works with **vLLM** and **llama.cpp** servers.
- **Multiple instances, side by side** — run more than one model shape on a node, *or* run two instances across the fleet, and each gets its own perf card (separate decode / prefill / TTFT / KV).
- **Cumulative token tracker** — a **Token Tracker** panel (and a `/api/tokens` endpoint) showing total tokens served per model, split into prompt vs generated, plus a per-day bucket. It banks the same `/metrics` token counters the perf cards already read, so a model-server restart (which zeroes those counters) never wipes your running history. Works out of the box for every configured model; toggle with `server.token_tracking`.
- **Optional video / image generation lanes** — a **Video Generation** panel (and `/api/comfy`) for ComfyUI-style servers, one card per GPU or instance: live VRAM used with a usage bar, free/total, queue depth, an idle / RENDERING / offline state light, and a click-through link into that lane's own UI. Each card also carries a **peak-while-rendering high-water mark**, which is the number that actually matters for capacity planning: these models load their components one at a time, so idle understates the footprint and summing the parts overstates it. Add `comfy_lanes` to enable; the panel hides itself entirely when the list is empty.
- **Collapsible, re-orderable panels** — every panel has a header you can click to collapse, a **Collapse all / Expand all** toggle, and a **Rearrange** mode that lets you drag panels into whatever order you want. Layout is saved per browser in `localStorage`, so the server stays stateless and nothing is shared between viewers.
- **Optional RoCE fabric-switch panel** — MikroTik / RouterOS over SSH: switch and CPU temps, fans, PSUs, uptime, and live per-port fabric throughput with link speeds. Add a `switch` block to enable it, delete the block to hide it.
- **Config-driven** — every node, host, user, SSH key, jump host, model endpoint, port, and label lives in `config.json`. Add or remove nodes and models by editing one file.
- **Bastion / jump-host support** — reach nodes that are only accessible through another box via standard SSH ProxyJump.
- **Read-only and resilient** — only ever *queries* state (`nvidia-smi` queries, `/proc`, hwmon, RouterOS `print`/`monitor once`, HTTP GETs). Nothing restarts, reconfigures, or kills anything. Unreachable nodes/models/switch degrade to a "stale" state; a poller never crashes the app.
- **Dependency-free** — Python 3.8+ standard library only. No `pip install`.

## Screenshot

<!-- add a screenshot here -->

## Requirements

- **Python 3.8+** on the machine that runs the dashboard.
- **SSH access** from that machine to each node (key-based; the key should log in without a password prompt). The nodes are expected to be Linux.
- **`nvidia-smi`** on each GPU node (ships with the NVIDIA driver). CPU temps come from `hwmon` under `/sys/class/hwmon` and system RAM from `free`.
- **Optional:** a vLLM or llama.cpp server per model, exposing Prometheus `/metrics` and OpenAI-style `/v1/models`, for the model performance cards.
- **Optional:** a MikroTik / RouterOS fabric switch reachable over SSH, for the switch panel.

## Quick Start

```bash
# 1. Copy the example config and edit it for your hardware
cp config.example.json config.json
$EDITOR config.json

# 2. Run the dashboard (no dependencies to install)
python3 server.py

# 3. Open it
#    http://localhost:8890
```

To use a config file somewhere else, set the `CONFIG` environment variable:

```bash
CONFIG=/path/to/my-fleet.json python3 server.py
```

Verify SSH works first — the dashboard runs the same commands you can run by hand:

```bash
ssh youruser@spark1.example.local nvidia-smi
```

If that succeeds without prompting for a password, the dashboard will be able to poll the node.

## Configuration

The config is a single JSON file (default `./config.json`, override with the `CONFIG` env var). Fields marked *optional* fall back to the `defaults` block or a sensible built-in.

### `server`

| Field | Description | Example |
| --- | --- | --- |
| `title` | Header text and browser tab title. | `"Sparky Command Center"` |
| `subtitle` | Small line under the title. | `"my rig"` |
| `bind` | Interface to bind. `0.0.0.0` = all interfaces, `127.0.0.1` = localhost only. | `"0.0.0.0"` |
| `port` | HTTP port for the dashboard. | `8890` |
| `browser_refresh_ms` | How often the browser re-polls `/api/metrics`, in milliseconds. | `2500` |
| `token_tracking` | Track cumulative tokens served per model (the Token Tracker panel + `/api/tokens`). Reads the `/metrics` counters already polled and banks the deltas across inference-server restarts. Set `false` to disable. | `true` |
| `token_store` | Path to the persistent token-usage JSON (banked totals + per-day buckets). Relative paths resolve from the working directory. The default lives under the gitignored `data/` dir. | `"data/token_usage.json"` |
| `comfy_poll_seconds` | How often each video/image lane is polled, in seconds. Kept tight so a render's VRAM peak is not missed between samples. | `4` |

> **How the token tracker stays honest across restarts.** vLLM / llama.cpp expose *monotonic* `prompt_tokens_total` / `generation_tokens_total` counters that reset to `0` every time the server restarts. The tracker snapshots them each poll and banks the delta into `token_store`; when it sees the counter go *backwards* it treats that as a restart and adds the new value from `0`. On first sight of a model it seeds the total with the current running-session value so the numbers are real immediately, and the per-day bucket rolls over at local midnight.

### `ssh`

| Field | Description | Example |
| --- | --- | --- |
| `default_key` | SSH private key for any node/switch that doesn't set its own `ssh_key`. `~` is expanded. | `"~/.ssh/id_ed25519"` |
| `connect_timeout` | Seconds to wait for an SSH connection before giving up. | `8` |
| `options` | Extra `-o Key=Value` options applied to every `ssh` call. `BatchMode=yes` makes a node that would prompt for a password fail fast instead of hanging. | `{ "BatchMode": "yes" }` |

### `defaults`

Fallback values for any node/model that doesn't set its own.

| Field | Description | Example |
| --- | --- | --- |
| `poll_interval` | Seconds between SSH polls of a node. | `6.0` |
| `model_poll_interval` | Seconds between `/metrics` scrapes of a model. | `6.0` |
| `temp_warn` | GPU temperature (°C) that turns the pill yellow. | `70` |
| `temp_hot` | GPU temperature (°C) that turns the pill red. | `84` |

### `nodes[]`

One entry per machine to monitor — a DGX Spark, a 3090 box, any Linux GPU host.

| Field | Required | Description |
| --- | --- | --- |
| `name` | yes | Display name for the node. |
| `host` | yes | Hostname or IP reachable over SSH. |
| `user` | yes | SSH username. |
| `ssh_key` | no | Per-node key path (overrides `ssh.default_key`). |
| `port` | no | SSH port (default `22`). |
| `jump_host` | no | Bastion/jump host (SSH ProxyJump) for a node only reachable through another box. |
| `jump_user` | no | Username for the jump host. |
| `poll_interval` | no | Per-node SSH poll cadence in seconds. |
| `temp_warn` | no | GPU temp (°C) for the yellow pill. |
| `temp_hot` | no | GPU temp (°C) for the red pill. |
| `models` | no | List of inference endpoints running on this node (see below). |

### Model instances

Model cards can come from two places, and both accept the same fields:

- **`nodes[].models[]`** — inference servers running *on that node*. List more than one to watch multiple model shapes on the same box; each is rendered under that node's *Model performance* section.
- **`models[]`** (top level) — **fleet-wide** instances that aren't tied to a single node (e.g. run two instances across the cluster, each spanning several GPUs, and watch both). Each is rendered in its own section. Add an optional `group` to give a set of instances a shared header.

| Field | Required | Description |
| --- | --- | --- |
| `label` | yes | Display name for the model card. |
| `endpoint` | yes | Base URL of the inference server (vLLM or llama.cpp). `/metrics` and `/v1/models` are appended automatically. |
| `port` | no | Port label shown on the card. |
| `model` | no | Served alias to prefer when `/v1/models` lists several ids. |
| `gpus` | no | Human label for the GPUs the model uses, e.g. `"GPU 0-1"` or `"Sparks 1-2 (TP=2)"`. |
| `group` | no | (Top-level `models[]` only) Section header shared by a set of instances. |

### `comfy_lanes[]` (optional)

ComfyUI-style image/video servers. One entry per GPU or per instance. Omit the
list (or leave it empty) and the Video Generation panel does not render at all.

| Field | Meaning | Required |
|---|---|---|
| `url` | Base URL of the ComfyUI server, e.g. `http://10.0.0.10:8188`. Also the click-through link on the card. | yes |
| `key` | Stable id for the lane. Derived from `name` when omitted. | no |
| `lane` | Short label shown on the card badge, e.g. `A`. Defaults to the position. | no |
| `name` | Human label, e.g. `gpu-box · GPU 0`. | no |
| `host` | Free-text machine name shown in the card footer. | no |

```json
"comfy_lanes": [
  { "key": "lane-a", "lane": "A", "name": "gpu-box · GPU 0", "host": "gpu-box", "url": "http://10.0.0.10:8188" },
  { "key": "lane-b", "lane": "B", "name": "gpu-box · GPU 1", "host": "gpu-box", "url": "http://10.0.0.10:8189" }
]
```

Two instances on the *same* GPU box are normal and cheap: ComfyUI mmaps the
model weights, so a second instance costs a few GB rather than a second full
copy. Give each instance its own `--temp-directory`, `--database-url` and
`--user-directory` when you launch it, or they will fight over state.

Polling is read-only: `GET /system_stats` for liveness and VRAM, `GET /queue`
for running/pending counts. It is done server-side because ComfyUI sends no
CORS headers, so a browser could not read these endpoints cross-origin.

### `switch` (optional)

Add a `switch` block to show a RoCE fabric-switch panel; delete the block to hide it entirely. The switch is reached over plain SSH (RouterOS runs the query and prints the result), so it uses the same key / jump-host mechanism as nodes.

| Field | Required | Description |
| --- | --- | --- |
| `host` | yes | Hostname or IP of the switch, reachable over SSH. |
| `user` | no | RouterOS SSH username (default `admin`). |
| `name` | no | Display name for the panel. |
| `badge` | no | Small badge label (default `RoCE FABRIC`). |
| `ssh_key` | no | Per-switch key (overrides `ssh.default_key`). |
| `port` | no | SSH port (default `22`). |
| `jump_host` / `jump_user` | no | Bastion to reach the switch through. |
| `poll_interval` | no | Seconds between switch polls (default `8`). |
| `temp_warn` / `temp_hot` | no | Switch temperature thresholds in °C (default `55` / `70`). |
| `ports` | no | Fabric interfaces to show throughput for, e.g. `["qsfp28-1-1", ...]`. Omit or `[]` to auto-detect running interfaces. |

### Adding or removing things

- **Add a node:** append an object to `nodes`.
- **Remove a node:** delete its object from `nodes`.
- **Add a model on a node:** append to that node's `models` list.
- **Add a fleet-wide instance:** append to the top-level `models` list.
- **A node with no models:** set `"models": []` — you'll still get its GPU/CPU/RAM cards.
- **Hide the switch panel:** delete the `switch` block.

See `config.example.json` for a complete, ready-to-edit starting point (it shows a node running two model shapes, a second GPU box, a node reached through a bastion, two fleet-wide instances, and a switch block).

## How it works

- On startup the server loads `config.json` and spins up one background thread per node, one per model, and one for the switch (if configured), each on its own staggered timer.
- **Node pollers** open a single SSH connection per cycle and run read-only commands (`nvidia-smi --query-gpu=...`, a small `hwmon` temperature scan, and `free`), parse the output, and write the result into an in-memory cache.
- **Model pollers** fetch each endpoint's Prometheus `/metrics` and `/v1/models` over HTTP, and compute decode/prefill tok/s as a rate between consecutive polls (TTFT and KV-cache come straight from the metrics).
- **The switch poller** SSHes into RouterOS and runs `print` / `monitor once` queries for health, resources, and interface counters, computing per-port throughput as a delta between polls.
- The browser polls `/api/metrics` (a JSON snapshot of the cache) on the configured cadence and renders the dashboard client-side. Polling and the browser refresh are fully decoupled, so a slow or unreachable node never blocks the page.
- Endpoints: `/` (dashboard), `/api/metrics` (JSON snapshot), `/api/tokens` (banked cumulative token totals per model), `/healthz` (plain `ok`).

## Contributing

Issues and pull requests are welcome. This is intentionally a small, single-file, dependency-free project — please keep changes within the Python standard library and avoid adding a build step. Bug reports that include your (sanitized) config and the output of the SSH command the dashboard runs are the easiest to help with.

## License

[MIT](LICENSE)
