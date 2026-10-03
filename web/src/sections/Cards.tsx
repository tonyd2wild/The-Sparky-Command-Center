import type { DiscreteNode, Gpu, ModelState, SwitchState, UnifiedNode } from '../lib/api'
import { fmtBps, fmtTps, fx, isNum, tempTone, tempWord } from '../lib/format'
import { Bar, Big, Chip, Fresh, KV, Led, Panel, Pill, Sparkline } from '../components/Panel'

const MEM_TONE = { ok: 'good', warn: 'warn', crit: 'bad' } as const

// One card per unit, the same fields v1 showed, in the JARVIS instrument style.

const TempExtra = ({ t, warn, hot }: { t: number | null; warn: number; hot: number }) => {
  const tone = tempTone(t, warn, hot)
  return <Pill tone={tone}>{tempWord(tone)}</Pill>
}

export function UnifiedCard({ s, hist, warn, hot, stale }: { s: UnifiedNode; hist?: number[]; warn: number; hot: number; stale: number }) {
  const reach = s.reachable
  return (
    <Panel id={`card-node-${s.key}`} dim={!reach} tone={!reach ? 'bad' : undefined}
      title={<><Led on={reach} /> <span>{s.name}</span></>}
      chips={<>{s.pair && <Chip tone="violet" title={s.pair}>{s.pair}</Chip>}{s.node_id && <Chip>{s.node_id}</Chip>}</>}>
      {reach ? (
        <>
          <div className="bigrow">
            <Big label="Temp" value={fx(s.temp)} unit="°C" extra={<TempExtra t={s.temp} warn={warn} hot={hot} />} />
            <Big label="Power" value={fx(s.power)} unit="W" />
            <Big label="SM clock" value={fx(s.sm_clock)} unit="MHz" small />
          </div>
          <KV k="GPU util">{fx(s.util)} %</KV>
          <Bar pct={s.util} />
          <KV k="Unified mem">{fx(s.mem_used_gib, 1)} / {fx(s.mem_total_gib, 1)} GiB{isNum(s.mem_avail_gib) ? ` · ${s.mem_avail_gib.toFixed(1)} free` : ''}</KV>
          <Bar pct={s.mem_pct} tone={s.mem_state ? MEM_TONE[s.mem_state] : undefined} />
          <KV k="Model resident">{isNum(s.model_gib) ? `${s.model_gib.toFixed(1)} GiB` : '-'}</KV>
          <KV k="Serving" accent>{s.model || '-'}</KV>
          <Sparkline values={hist} />
        </>
      ) : <p className="card__err">Unreachable: {s.err || 'no response'}</p>}
      <Fresh ts={s.ts} err={reach ? null : s.err} staleAfter={stale} />
    </Panel>
  )
}

export function GpuCard({ g, hist, warn, hot }: { g: Gpu; hist?: number[]; warn: number; hot: number }) {
  const pl = g.power_limit || 350
  const capPct = isNum(g.power) ? (g.power / pl) * 100 : null
  const capTone = !isNum(g.power) ? 'muted' : g.power > pl * 0.85 ? 'bad' : g.power > pl * 0.55 ? 'warn' : 'good'
  return (
    <Panel title={<><Led on /> <span>GPU {g.index}</span></>} chips={<Chip>{g.name}</Chip>}>
      <div className="bigrow bigrow--2">
        <Big label="Temp" value={fx(g.temp)} unit="°C" extra={<TempExtra t={g.temp} warn={warn} hot={hot} />} />
        <Big label="Power" value={fx(g.power)} unit="W" extra={<Pill tone={capTone}>{isNum(capPct) ? `${capPct.toFixed(0)}% cap` : '-'}</Pill>} />
      </div>
      <KV k="Utilization">{fx(g.util)} %</KV>
      <Bar pct={g.util} />
      <KV k="VRAM">{isNum(g.mem_used_mb) ? (g.mem_used_mb / 1024).toFixed(1) : '-'} / {isNum(g.mem_total_mb) ? (g.mem_total_mb / 1024).toFixed(1) : '-'} GiB</KV>
      <Bar pct={g.mem_pct} />
      <KV k="Fan">{isNum(g.fan) ? `${g.fan.toFixed(0)} %` : '-'}</KV>
      <KV k="Graphics clock">{isNum(g.gr_clock) ? `${g.gr_clock.toFixed(0)} MHz` : '-'}</KV>
      <Sparkline values={hist} />
    </Panel>
  )
}

export function HostCard({ box, stale }: { box: DiscreteNode; stale: number }) {
  const reach = box.reachable
  return (
    <Panel id={`card-node-${box.key}`} className="card--wide" dim={!reach} tone={!reach ? 'bad' : undefined}
      title={<><Led on={reach} /> <span>{box.name}</span></>} chips={<Chip tone="violet">{box.host_label || 'GPU HOST'}</Chip>}>
      {reach ? (
        <div className="hostgrid">
          <div>
            <Big label="CPU Tctl" value={fx(box.cpu_temp)} unit="°C" extra={<TempExtra t={box.cpu_temp} warn={75} hot={90} />} />
            {(box.cpu_temps || []).slice(1).map((t, i) => <KV key={i} k={`k10temp #${i + 2}`}>{t.toFixed(1)} °C</KV>)}
            <KV k="System RAM">{isNum(box.mem_used_mb) ? (box.mem_used_mb / 1024).toFixed(1) : '-'} / {isNum(box.mem_total_mb) ? (box.mem_total_mb / 1024).toFixed(1) : '-'} GiB</KV>
            <Bar pct={box.mem_pct} />
          </div>
          <div>
            <p className="sublabel">Running containers</p>
            {(box.models || []).length ? box.models.map(m => (
              <KV key={m.name} k={m.label}><Pill tone="good">UP</Pill> {m.port ? `:${m.port}` : ''} {m.gpus ? `· ${m.gpus}` : ''}</KV>
            )) : <KV k="-">none</KV>}
          </div>
        </div>
      ) : <p className="card__err">Unreachable: {box.err || 'no response'}</p>}
      <Fresh ts={box.ts} err={reach ? null : box.err} staleAfter={stale} />
    </Panel>
  )
}

export function SwitchCard({ sw, name, badge, warn, hot, stale }: { sw: SwitchState; name: string; badge: string; warn: number; hot: number; stale: number }) {
  const reach = sw.reachable
  const h = sw.health || {}
  const r = sw.resource || {}
  const swT = h['switch-temperature'] != null ? parseFloat(h['switch-temperature']) : null
  const cpuT = h['cpu-temperature'] != null ? parseFloat(h['cpu-temperature']) : null
  const fans = ['fan1-speed', 'fan2-speed', 'fan3-speed', 'fan4-speed'].filter(f => h[f] != null)
  return (
    <Panel id="card-switch" className="card--wide" dim={!reach} tone={!reach ? 'bad' : 'violet'}
      title={<><Led on={reach} /> <span>{name}</span></>} chips={<><Chip tone="violet">{badge}</Chip>{reach && <Chip tone="core">{fmtBps(sw.total_bps)}</Chip>}</>}>
      {reach ? (
        <div className="switchgrid">
          <div>
            <div className="bigrow bigrow--2">
              <Big label="Switch temp" value={fx(swT)} unit="°C" extra={<TempExtra t={swT} warn={warn} hot={hot} />} />
              <Big label="CPU temp" value={fx(cpuT)} unit="°C" extra={<TempExtra t={cpuT} warn={60} hot={75} />} />
            </div>
            <KV k={<>Fans <span className={h['fan-state'] === 'ok' ? 'good' : 'amber'}>{h['fan-state'] || '-'}</span></>}>
              {h['psu1-state'] === 'ok' && h['psu2-state'] === 'ok' ? '2 PSU ok' : 'PSU?'}
            </KV>
            {fans.length > 0 && <div className="fans">{fans.map((f, i) => <div key={f} className="fan"><span>FAN{i + 1}</span><b className="tnum">{h[f]}</b></div>)}</div>}
            <KV k="PSU1">{h['psu1-power'] || '-'} W · {h['psu1-temperature'] || '-'} °C</KV>
            <KV k="PSU2">{h['psu2-power'] || '-'} W · {h['psu2-temperature'] || '-'} °C</KV>
            <KV k="Uptime">{r['uptime'] || '-'}</KV>
            <KV k="RouterOS">{r['version'] ? r['version'].split(' ')[0] : '-'} · cpu {r['cpu-load'] || '-'}</KV>
          </div>
          <div>
            <p className="sublabel">Fabric ports · live throughput</p>
            <div className="ports">
              {(sw.ports || []).map(p => (
                <div key={p.name} className={`port ${p.running ? '' : 'is-down'}`}>
                  <div className="port__n mono"><span>{p.name}</span><Pill tone={p.running ? 'good' : 'muted'}>{p.running ? 'link-ok' : 'down'}</Pill></div>
                  <div className="port__v tnum">{fmtBps((p.rx_bps || 0) + (p.tx_bps || 0))}</div>
                  <div className="port__m mono">{p.rate || '...'} · ↓{fmtBps(p.rx_bps)} ↑{fmtBps(p.tx_bps)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : <p className="card__err">Switch unreachable: {sw.err || 'no response'}</p>}
      <Fresh ts={sw.ts} err={reach ? null : sw.err} staleAfter={stale} />
    </Panel>
  )
}

function LiveRow({ m }: { m: ModelState }) {
  const n = m.running || 0
  const d = m.decode_tps
  if (!n || !isNum(d)) return null
  if (d <= 0) {
    const pf = m.prefill_tps || 0
    return <div className="liverow"><span className="liverow__dot" />{n} live · {pf > 0 ? `prefilling ${fmtTps(pf)} tok/s` : 'starting up'}<em>no decode yet</em></div>
  }
  return n > 1
    ? <div className="liverow"><span className="liverow__dot" />{n} live · ~{fmtTps(d / n)} tok/s each<em>{fmtTps(d)} combined</em></div>
    : <div className="liverow"><span className="liverow__dot" />1 live · {fmtTps(d)} tok/s<em>decode</em></div>
}

export function ModelCard({ m, stale }: { m: ModelState; stale: number }) {
  const reach = m.reachable
  const busy = (m.running || 0) > 0
  const ttft = isNum(m.ttft_ms) ? (m.ttft_ms >= 1000 ? <>{(m.ttft_ms / 1000).toFixed(2)}<em>s</em></> : <>{m.ttft_ms.toFixed(0)}<em>ms</em></>) : '-'
  return (
    <Panel id={`card-model-${m.key}`} dim={!reach} tone={!reach ? 'bad' : busy ? 'violet' : undefined} className="card--model"
      title={<><Led on={reach} /> <span>{m.label}</span></>}
      chips={<>{<Chip tone="violet">{m.engine || 'inference'}</Chip>}{m.port && <Chip>:{m.port}</Chip>}{reach && (busy ? <Chip tone="violet">BUSY</Chip> : <Chip tone="muted">idle</Chip>)}</>}>
      {reach ? (
        <>
          <KV k="Model" accent><span className="mono">{m.model || '-'}</span></KV>
          <div className={`perf ${busy ? '' : 'is-idle'}`}>
            <div><span>Decode</span><b className="tnum">{fmtTps(m.decode_tps)}<em>tok/s</em></b></div>
            <div><span>Prefill</span><b className="tnum violet">{fmtTps(m.prefill_tps)}<em>tok/s</em></b></div>
            <div><span>TTFT avg</span><b className="tnum">{ttft}</b></div>
          </div>
          <KV k="KV-cache">{isNum(m.kv_pct) ? `${m.kv_pct.toFixed(1)} %` : '-'}</KV>
          <Bar pct={m.kv_pct} />
          <KV k="Requests">{m.running ?? '-'} running · {m.waiting ?? '-'} waiting</KV>
          <LiveRow m={m} />
          {m.gpus && <KV k="On">{m.gpus}</KV>}
        </>
      ) : <p className="card__err">{m.label} offline: {m.err || 'no /metrics'}</p>}
      <Fresh ts={m.ts} err={reach ? null : m.err} staleAfter={stale} />
    </Panel>
  )
}
