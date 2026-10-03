// Types + fetchers for server.py. Every call resolves to a Res, never throws.

export type Res<T> = { ok: true; data: T } | { ok: false; error: string }

export async function getJSON<T>(url: string, timeoutMs = 15000): Promise<Res<T>> {
  const ctl = new AbortController()
  const t = setTimeout(() => ctl.abort(), timeoutMs)
  try {
    const r = await fetch(url, { cache: 'no-store', signal: ctl.signal })
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}` }
    return { ok: true, data: (await r.json()) as T }
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) }
  } finally {
    clearTimeout(t)
  }
}

export async function postJSON<T>(url: string, body: unknown): Promise<{ status: number; data: T | null }> {
  try {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    let data: T | null = null
    try { data = (await r.json()) as T } catch { /* empty body */ }
    return { status: r.status, data }
  } catch {
    return { status: 0, data: null }
  }
}

/* ------------------------------------------------------------------ config */

export interface SectionCfg {
  key: string; title: string; eyebrow?: string; subtitle?: string
  nodes: string[]; switch?: boolean; units: string[]; models_title?: string
}
export interface EcoLevel { value: string; label: string; default?: boolean }
export interface UiConfig {
  version: string; title: string; subtitle?: string; location?: string; timezone: string | null
  refresh_ms: number; read_only: boolean
  sections: SectionCfg[]
  nodes: { key: string; name: string; profile: 'unified' | 'discrete'; badge?: string; temp_warn?: number; temp_hot?: number }[]
  switch: { name: string; badge: string; temp_warn: number; temp_hot: number } | null
  models: { key: string; unit: string; node?: string }[]
  comfy: { enabled: boolean; title: string }
  tokens: { enabled: boolean; order: string[] }
  eco: { enabled: boolean; writes: boolean; levels: EcoLevel[]; nodes: { key: string; name: string }[]; info_url: string }
  stations: { enabled: boolean }
  keylights: { enabled: boolean; writes: boolean }
  chat: { enabled: boolean; name: string; suggestions: string[] }
}

/* ------------------------------------------------------------------ metrics */

export interface UnifiedNode {
  key: string; name: string; profile: 'unified'; node_id?: string; rank?: number; reachable: boolean
  temp: number | null; power: number | null; util: number | null; sm_clock: number | null
  mem_total_gib: number | null; mem_used_gib: number | null; mem_pct: number | null; model_gib: number | null
  mem_avail_gib?: number | null; mem_state?: 'ok' | 'warn' | 'crit' | null
  model?: string | null; pair?: string | null; ts: number; err: string | null
}
export interface Gpu {
  index: number; name: string; temp: number | null; power: number | null; power_limit: number | null; util: number | null
  mem_used_mb: number | null; mem_total_mb: number | null; mem_pct?: number; fan: number | null; gr_clock: number | null; mem_clock: number | null
}
export interface DiscreteNode {
  key: string; name: string; profile: 'discrete'; host_label?: string; reachable: boolean; ts: number; err: string | null
  gpus: Gpu[]; cpu_temp: number | null; cpu_temps: number[]; mem_total_mb: number | null; mem_used_mb: number | null; mem_pct: number | null
  models: { name: string; label: string; port?: number; gpus?: string; up: boolean }[]
}
export type FleetNode = UnifiedNode | DiscreteNode
export interface SwitchPort { name: string; running: boolean; rx_bps: number; tx_bps: number; rate: string | null }
export interface SwitchState {
  reachable: boolean; ts: number; err: string | null
  health?: Record<string, string>; resource?: Record<string, string>; ports?: SwitchPort[]; total_bps?: number
}
export interface ModelState {
  key: string; label: string; unit: string; port?: number; gpus?: string; node?: string
  reachable: boolean; engine?: string | null; model?: string | null
  decode_tps?: number | null; prefill_tps?: number | null; ttft_ms?: number | null; kv_pct?: number | null
  running?: number | null; waiting?: number | null; ts: number; err: string | null
}
export interface Metrics {
  ts: number; version: string; read_only: boolean
  nodes: FleetNode[]; switch: SwitchState; history: Record<string, number[]>; models: ModelState[]
  agg: { gpu_count: number; total_power: number; hottest_unit: string | null; hottest_temp: number | null; all_ok: boolean; down: string[]; low_mem?: string[] }
  thresholds: { temp_warn: number; temp_hot: number; stale_after_s: number; mem_warn_gib?: number; mem_crit_gib?: number }
}
export interface ComfyLane {
  key: string; lane: string; name: string; host: string; url: string; reachable: boolean; ts: number
  vram_total: number | null; vram_free: number | null; vram_used: number | null; version: string | null
  running: number; pending: number; busy: boolean; peak_used?: number | null; peak_busy?: number | null
  mem_source?: string; unified?: boolean; comfy_used?: number | null
}
export interface TokenRow {
  name?: string; reachable?: boolean; total_tokens?: number; total_prompt?: number; total_gen?: number; today_tokens?: number
}
export type Tokens = Record<string, TokenRow | unknown>
export interface Station { emoji: string; name: string; desc: string; port?: number; url: string; up: boolean }
export interface Light { name?: string; ip?: string; on?: number; brightness?: number; temperature?: number; [k: string]: unknown }
export interface ChatStatus {
  enabled: boolean; configured: boolean; reachable: boolean; name: string
  model?: string; active_model?: string | null; fallback?: boolean; endpoints?: number
}

export const getConfig = () => getJSON<UiConfig>('/api/config')
export const getMetrics = () => getJSON<Metrics>('/api/metrics')
export const getComfy = () => getJSON<{ lanes: ComfyLane[] }>('/api/comfy')
export const getTokens = () => getJSON<Tokens>('/api/tokens')
export const getStations = () => getJSON<{ stations: Station[] }>('/api/stations')
export const getLights = () => getJSON<{ lights: Light[]; down?: boolean; disabled?: boolean }>('/api/lights')
export const getEcoStatus = () => getJSON<{ status: Record<string, string>; disabled?: boolean }>('/api/eco-status', 40000)
export const getChatStatus = (force = false) => getJSON<ChatStatus>(`/api/chat/status${force ? '?force=1' : ''}`)
