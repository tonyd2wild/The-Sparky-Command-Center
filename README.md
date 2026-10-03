# Command Center v2

A read-only command center for a self-hosted AI fleet, with a live Three.js constellation of your machines and a built-in chat assistant that runs on **your own model**.

It watches:

- **GPU nodes** over SSH: unified-memory boxes (DGX Spark / GB10 class) and hosts with discrete GPUs (RTX, A-series). Temperature, power, utilization, clocks, memory, fans, CPU temperature, running containers, and a temperature sparkline per GPU.
- **A fabric switch** (MikroTik RouterOS): temperatures, fans, PSUs, uptime, and live throughput per port.
- **Model servers** (vLLM, SGLang, llama.cpp) through their Prometheus `/metrics`: decode and prefill tok/s, TTFT, KV-cache use, running and waiting requests, and the live model id.
- **ComfyUI render lanes**: up/idle/rendering, queue depth, and memory, taken from the driver (not ComfyUI's own estimate), with peak-while-rendering marks.
- **Tokens served** per model, banked across server restarts, with a per-day count.
- Optional extras: GPU clock caps (status plus, if you allow it, apply), links to other web apps on the host with an up/down probe, and Elgato key lights.

The chat ("Jarvis" by default) sees the same live data the board shows, so you can ask *"what is down right now?"* or *"which node is running hottest?"* and get an answer grounded in the current numbers.

Stack: Python 3.9+ standard library for the server (no pip installs), Vite + React + TypeScript + React Three Fiber + GSAP + Lenis for the UI.

## Quick start

```bash
git clone https://github.com/tonyd2wild/The-Sparky-Command-Center.git && cd The-Sparky-Command-Center
cp config.example.json config.json      # then edit it for your machines
python3 server.py
# open http://localhost:8895
```

The built UI ships in `web/dist`, so running it needs nothing but Python 3.9+. Node 18+ is only needed if you want to change the UI: `cd web && npm install && npm run build`.

> **v2 (September 2026)** is a full rebuild: a Three.js constellation of your fleet, a new card layout, themes (Dark, Light, Sunset, Breeze, Matrix) and a built-in chat you can point at your own local model or agent. The original single-file v1 is still available at the [`v1` tag](https://github.com/tonyd2wild/The-Sparky-Command-Center/tree/v1).

The machines you monitor need:

- `nvidia-smi` installed.
- Key-based SSH from the box running this server. It uses `BatchMode=yes`, so a node that asks for a password fails fast instead of hanging.
- For model cards, an inference server with `/metrics` enabled (vLLM and SGLang expose it by default; for llama.cpp, start `llama-server` with `--metrics`).

Nothing needs to be installed on the nodes.

## Connect the chat to your own model

The chat speaks the OpenAI chat-completions protocol, so it works with anything that serves `/v1/chat/completions`. Set a base URL and a model name, either in `config.json`:

```json
"chat": {
  "base_url": "http://localhost:11434/v1",
  "model": "llama3.1:8b"
}
```

or in `.env` (copy `.env.example`), which overrides the config:

```bash
CC_CHAT_BASE_URL=http://localhost:11434/v1
CC_CHAT_MODEL=llama3.1:8b
```

| Server | `base_url` | Notes |
| --- | --- | --- |
| Ollama | `http://localhost:11434/v1` | model = the Ollama tag, e.g. `qwen2.5:14b` |
| vLLM | `http://localhost:8000/v1` | model = the `--served-model-name` |
| SGLang | `http://localhost:30000/v1` | |
| llama.cpp server | `http://localhost:8080/v1` | model can be any string |
| LM Studio | `http://localhost:1234/v1` | |
| Any hosted OpenAI-compatible API | its `/v1` URL | put the key in `CC_CHAT_API_KEY` |

Restart `server.py` after changing the chat settings. If no model is set, or the one you set cannot be reached, the chat panel shows a "connect a model" screen with these steps. The rest of the board keeps working.

### Or connect it to an agent

If you run an agent that exposes an OpenAI-compatible API (for example a Hermes profile with its API server turned on), point the chat at the agent instead of the raw model. You get the agent's memory and tools behind the same chat box:

```bash
CC_CHAT_BASE_URL=http://127.0.0.1:8643/v1
CC_CHAT_MODEL=<the model name the agent's API reports>
CC_CHAT_API_KEY=<the agent's API key>
```

No other change is needed. The dashboard still adds its live-data context as the system message, and the agent answers.

### How the chat behaves

- **Grounded.** Every turn gets a system message with a compact read of the live board, taken at the moment you ask: totals, a "down right now" list with error text, every node, the switch, each model server, each render lane, token counts and station status. Turn it off with `"grounding": false`.
- **Careful.** Temperature defaults to 0.1, and the prompt forbids inventing numbers and tells the model it cannot change anything.
- **Streams tokens.** Reasoning is stripped, whether it arrives as `reasoning_content` or inside `<think>` tags.
- **Fallbacks.** List extra endpoints in `chat.fallbacks`. A connection failure moves to the next endpoint. An HTTP error from a model that answered is shown as-is, not retried.
- **Local history.** The conversation lives in the viewer's browser (localStorage), with a Clear button.
- **Extra fields.** `chat.extra_body` is merged into every request. For example, `{"chat_template_kwargs": {"enable_thinking": false}}` turns thinking off on vLLM-served reasoning models. Leave it empty for servers that reject unknown fields.

## Configuration reference

`config.json` is gitignored and holds everything site-specific. `config.example.json` shows every block with made-up hosts. Keys starting with `_` are ignored.

### `server`

| Key | Default | Meaning |
| --- | --- | --- |
| `title`, `subtitle`, `location` | | Header text. The nav word is the title minus "Command Center". |
| `timezone` | system | IANA name used for the UI clock and the chat's sense of time. |
| `bind` | `["127.0.0.1"]` | One address or a list. Add a VPN/tailnet IP to reach it from your phone. Avoid `0.0.0.0` on untrusted networks. |
| `port` | `8895` | |
| `allowed_hosts` | `[]` | Extra Host names allowed to POST (for example a tailnet DNS name). |
| `read_only` | `true` | Refuses every route that changes another machine. |
| `browser_refresh_ms` | `2500` | How often the page re-polls `/api/metrics`. |

### `ssh`

`default_key`, `connect_timeout` and `options` (extra `-o` flags). A node can override the key with `ssh_key` and the port with `ssh_port`.

### `nodes[]`

| Key | Meaning |
| --- | --- |
| `key`, `name` | Id and display name. |
| `profile` | `unified` (GPU shares system RAM: DGX Spark, GB10, Jetson) or `discrete` (PCIe GPUs). |
| `user`, `host` | SSH target. |
| `jump` | Optional. `{"user", "host", "mode": "proxy"}` uses `ssh -J`. `"mode": "nested"` runs `ssh` from the jump host itself, for LAN nodes that only trust the jump host's key. |
| `node_id`, `rank`, `serving`, `pair` | Unified nodes: labels on the card (what the node is serving and its role). |
| `badge`, `list_containers`, `containers` | Discrete hosts: card badge, whether to list `docker ps` names, and friendly labels per container name. |
| `temp_warn`, `temp_hot`, `mem_warn_gib`, `mem_crit_gib`, `poll_interval` | Per-node overrides of `defaults`. |

Unified-memory nodes also get a memory headroom state from `MemAvailable`: `warn` below `mem_warn_gib` (default 8) and `crit` below `mem_crit_gib` (default 4), set in `defaults` or per node. GPU and OS share one pool on these boxes, so the card's memory bar follows headroom rather than percent used (a loaded GB10 sits above 90% normally). Low nodes are listed in `agg.low_mem` and flagged to the chat.

### `sections[]`

The board groups nodes into sections: `{"key", "eyebrow", "title", "subtitle", "nodes": [...], "switch": true, "units": [...], "models_title"}`. A section shows its nodes, the switch card if `switch` is true, and every model whose `unit` is in `units`. Leave `sections` out and the server builds a sensible default.

### `switch`

MikroTik RouterOS over plain SSH (`host`, `user`, `ssh_key`), or any command you like through `exec`, for example an `expect` helper: `{"argv": ["expect", "helper.exp", "{cmd}"], "cwd": "/path"}`. `ports` lists the interfaces to chart. Delete the block, or set `"enabled": false`, to hide it.

### `models[]`

`{"key", "label", "unit", "node", "endpoint", "port", "gpus", "model", "api_key"}`:

- `endpoint` is the server's base URL; `/metrics` and `/v1/models` are appended.
- `node` ties the model to a host in the 3D view.
- `model` picks the preferred alias when a server lists several.

### `comfy_lanes[]`

`{"key", "lane", "name", "host", "url", "src"}`. `src` points the memory reading at a GPU you already monitor:

- `{"kind": "gpu", "node": "<key>", "index": 0}` for a discrete GPU.
- `{"kind": "unified", "node": "<key>"}` for a unified node.

### `tokens`

- `mode: "bank"` (default): polls each model's counters every `poll_seconds` and banks them into `store`, surviving server restarts. Rows for models you remove are archived under `_retired`, not deleted.
- `mode: "read"`: only reads a store that another process maintains.

`order` sets the card order.

### `eco`, `stations`, `keylights`

Off by default.

- `eco`: GPU clock caps on unified nodes. Status is a read-only query. Apply needs `"allow_writes": true` and `server.read_only: false`, and uses `sudo nvidia-smi -lgc`.
- `stations`: a list of `{emoji, name, desc, url, probe_url}` links with an up/down probe.
- `keylights`: proxies an Elgato key-light panel's `/api/lights`. Setting the lights needs `allow_writes`.

### `chat`

`enabled`, `name`, `base_url`, `model`, `api_key` (or `api_key_file`), `system_prompt` (replaces the default persona line; the rules and live data are always added), `temperature`, `max_tokens`, `timeout`, `history_turns`, `grounding`, `extra_body`, `fallbacks[]`, `suggestions[]`.

Environment variables:

| Variable | Overrides |
| --- | --- |
| `CC_CONFIG` | path to the config file |
| `CC_PORT`, `CC_BIND` | `server.port`, `server.bind` |
| `CC_READ_ONLY` | `server.read_only` |
| `CC_CHAT_BASE_URL`, `CC_CHAT_MODEL`, `CC_CHAT_API_KEY` | the matching `chat` fields |
| `CC_CHAT_SYSTEM_PROMPT`, `CC_CHAT_NAME` | the matching `chat` fields |

## Safety

- Monitoring only runs queries: `nvidia-smi --query-*`, `/proc/meminfo`, `free`, `docker ps`, and RouterOS `print` / `monitor once`. Nothing restarts or reconfigures anything.
- With `read_only: true` (the default), `POST /api/eco-set` and `POST /api/lights-set` return 403, and their buttons stay visible but disabled.
- `POST` routes are same-origin only: a known Host, a matching Origin, and a JSON content type. This stops another site from driving your chat or your actions from a browser.
- The browser never receives hosts, keys or endpoints from the config. `/api/config` only sends layout and labels. Model and ComfyUI URLs do appear on cards and in `/api/metrics`, so keep the server on localhost or a private network.

## API

| Route | Returns |
| --- | --- |
| `GET /api/metrics` | Every node, the switch, the model servers, temperature history and fleet totals. It also carries `sparks` / `box` / `*_models` for v1 consumers. |
| `GET /api/comfy` | Render lanes. |
| `GET /api/tokens` | The token store. |
| `GET /api/stations` | Stations with up/down. |
| `GET /api/lights` | The key-light panel's lights. |
| `GET /api/eco-status` | Current GPU clock / temperature / power per unified node (read-only). |
| `GET /api/config` | UI layout and labels only. |
| `GET /api/chat/status` | Whether a model is configured and reachable. |
| `POST /api/chat` | `{"messages": [...]}`, answered as a server-sent-event stream of `{"delta"}` / `{"status"}` / `{"reset"}` / `{"meta"}`, then `[DONE]`. |
| `GET /healthz` | `ok` |

## Development

```bash
python3 server.py                     # API on :8895
cd web && npm run dev                 # UI on :5176, proxies /api to :8895
```

Useful query flags: `?nowebgl=1` shows the SVG fallback map, and `?motion=reduce` forces the reduced-motion still frame.

## License

MIT
