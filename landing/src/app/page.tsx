'use client'

import { useState, useCallback, CSSProperties } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'

// ---------------------------------------------------------------------------
// Design tokens — single source of truth for every colour
// ---------------------------------------------------------------------------
const C = {
  bg:      '#0d1117',
  surface: '#161b22',
  border:  '#21262d',
  border2: '#30363d',
  green:   '#58d68d',
  red:     '#e74c3c',
  yellow:  '#f39c12',
  text:    '#e6edf3',
  muted:   '#8b949e',
  dim:     '#c9d1d9',
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface VisionDetection {
  predicted_class: string
  confidence: number
  confidence_label: string
}
interface SimulationSummary {
  grid_size: number[]
  days_simulated: number
  baseline_crop_loss: number
  best_crop_loss: number
  crop_loss_saving: number
}
interface RankedIntervention {
  rank: number
  intervention_name: string
  strategy_key: string
  crop_loss_percent: number
  final_score: number
}
interface Recommendation {
  best_strategy: string
  action_summary: string
  projected_loss: number
  saving_vs_baseline: number
  urgency: string
}
interface Scenario {
  intervention_name: string
  daily_counts: { day: number; susceptible: number; infected: number; removed: number }[]
  crop_loss_percent: number
}
interface DSSReport {
  disease_class?: string
  disease_name?: string
  is_healthy?: boolean
  urgency_level: string
  expert_advice?: string
  vision_detection: VisionDetection
  simulation_summary: SimulationSummary | null
  ranked_interventions: RankedIntervention[]
  recommendation: Recommendation
  scenarios?: Scenario[]
  rag_treatment_protocol?: { retrieved_context: string }
}

// ---------------------------------------------------------------------------
// All styles as inline CSSProperties — zero Tailwind dependency
// ---------------------------------------------------------------------------
const S: Record<string, CSSProperties> = {
  page:       { minHeight: '100vh', backgroundColor: C.bg, color: C.text, fontFamily: "'DM Sans','Inter',system-ui,sans-serif", margin: 0 },
  nav:        { position: 'sticky', top: 0, zIndex: 50, backgroundColor: C.surface, borderBottom: `1px solid ${C.border}`, padding: '14px 24px', display: 'flex', alignItems: 'center', gap: 12 },
  navLogo:    { fontSize: 22 },
  navTitle:   { fontSize: 15, fontWeight: 700, color: C.green, letterSpacing: '-0.3px' },
  navSub:     { marginLeft: 'auto', fontSize: 12, color: C.muted },
  main:       { maxWidth: 960, margin: '0 auto', padding: '32px 24px 80px' },
  h1:         { fontSize: 26, fontWeight: 700, color: C.text, marginBottom: 6, marginTop: 0 },
  subtitle:   { fontSize: 14, color: C.muted, marginBottom: 28, marginTop: 0 },
  grid2:      { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' },
  card:       { backgroundColor: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden' },
  cardBody:   { padding: 16 },
  imgPrev:    { width: '100%', maxHeight: 220, objectFit: 'cover', display: 'block' },
  filename:   { fontSize: 12, color: C.muted, marginBottom: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: 0 },
  error:      { marginTop: 16, padding: 16, backgroundColor: 'rgba(231,76,60,0.1)', border: '1px solid rgba(231,76,60,0.3)', borderRadius: 12, color: C.red, fontSize: 13 },
  resultHdr:  { display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 20, marginTop: 36 },
  h2:         { fontSize: 20, fontWeight: 700, color: C.text, margin: 0 },
  confCard:   { backgroundColor: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, marginBottom: 12 },
  confRow:    { display: 'flex', justifyContent: 'space-between', fontSize: 12, color: C.muted, marginBottom: 6 },
  confBar:    { height: 6, backgroundColor: C.border, borderRadius: 999, overflow: 'hidden' },
  recBanner:  { backgroundColor: 'rgba(88,214,141,0.05)', border: '1px solid rgba(88,214,141,0.2)', borderRadius: 12, padding: 20, marginBottom: 20 },
  recLabel:   { fontSize: 11, fontWeight: 700, color: C.green, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 8, marginTop: 0 },
  recText:    { fontSize: 14, color: C.text, lineHeight: 1.7, marginBottom: 16, marginTop: 0 },
  statsRow:   { display: 'flex', gap: 32, flexWrap: 'wrap' },
  statLabel:  { fontSize: 11, color: C.muted, marginBottom: 2 },
  tabRow:     { display: 'flex', borderBottom: `1px solid ${C.border}`, marginBottom: 20 },
  tableCard:  { backgroundColor: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, marginTop: 12 },
  tableTitle: { fontSize: 13, fontWeight: 600, color: C.text, marginBottom: 14, marginTop: 0 },
  table:      { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th:         { textAlign: 'left', paddingBottom: 10, color: C.muted, fontWeight: 500, borderBottom: `1px solid ${C.border}` },
  td:         { padding: '10px 0', borderBottom: `1px solid rgba(33,38,45,0.5)` },
  statGrid3:  { display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12, marginBottom: 12 },
  statCard:   { backgroundColor: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 16 },
  chartCard:  { backgroundColor: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, marginTop: 12 },
  adviceCard: { backgroundColor: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 24, fontSize: 14, lineHeight: 1.7, color: C.dim },
  chatCard:   { backgroundColor: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, display: 'flex', flexDirection: 'column', height: 380 },
  chatMsgs:   { flex: 1, overflowY: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 10 },
  chatEmpty:  { color: C.muted, fontSize: 13, textAlign: 'center', marginTop: 60 },
  chatFoot:   { borderTop: `1px solid ${C.border}`, padding: 12, display: 'flex', gap: 8 },
  chatInput:  { flex: 1, backgroundColor: C.bg, border: `1px solid ${C.border2}`, borderRadius: 8, padding: '8px 12px', fontSize: 13, color: C.text, outline: 'none' },
  chatBtn:    { padding: '8px 16px', backgroundColor: C.green, color: C.bg, border: 'none', borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: 'pointer' },
  modGrid:    { display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 },
  modCard:    { backgroundColor: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 16 },
  modIcon:    { fontSize: 22, marginBottom: 8 },
  modName:    { fontSize: 13, fontWeight: 600, color: C.text, marginBottom: 4 },
  modDesc:    { fontSize: 12, color: C.muted, lineHeight: 1.5, margin: 0 },
  secTitle:   { fontSize: 15, fontWeight: 600, color: C.text, marginBottom: 16, marginTop: 40 },
}

// Dynamic styles (functions)
const dropStyle = (active: boolean): CSSProperties => ({
  border: `2px dashed ${active ? C.green : C.border2}`,
  borderRadius: 12, padding: '48px 24px', textAlign: 'center',
  cursor: 'pointer', backgroundColor: active ? 'rgba(88,214,141,0.04)' : 'transparent', transition: 'all 0.2s',
})
const btnStyle = (disabled: boolean): CSSProperties => ({
  width: '100%', padding: '10px 0', borderRadius: 8, fontWeight: 600, fontSize: 14,
  backgroundColor: disabled ? '#1f3329' : C.green, color: disabled ? C.muted : C.bg,
  border: 'none', cursor: disabled ? 'not-allowed' : 'pointer', transition: 'all 0.2s',
})
const badgeStyle = (urgency: string): CSSProperties => {
  const m: Record<string, [string, string, string]> = {
    HIGH:   ['rgba(231,76,60,0.1)',  'rgba(231,76,60,0.3)',  C.red],
    MEDIUM: ['rgba(243,156,18,0.1)', 'rgba(243,156,18,0.3)', C.yellow],
    LOW:    ['rgba(88,214,141,0.1)', 'rgba(88,214,141,0.3)', C.green],
    NONE:   ['rgba(88,214,141,0.1)', 'rgba(88,214,141,0.3)', C.green],
  }
  const [bg, border, color] = m[urgency] ?? m.LOW
  return { padding: '4px 12px', borderRadius: 999, fontSize: 11, fontWeight: 700, border: `1px solid ${border}`, backgroundColor: bg, color, letterSpacing: '0.5px' }
}
const tabStyle = (active: boolean): CSSProperties => ({
  padding: '10px 18px', fontSize: 13, fontWeight: 500, background: 'none', border: 'none',
  borderBottom: `2px solid ${active ? C.green : 'transparent'}`, color: active ? C.green : C.muted,
  cursor: 'pointer', transition: 'all 0.15s', marginBottom: -1,
})
const confFill = (pct: number): CSSProperties => ({
  height: '100%', width: `${pct}%`, borderRadius: 999, transition: 'width 0.7s ease',
  backgroundColor: pct >= 80 ? C.green : pct >= 50 ? C.yellow : C.red,
})
const chatMsgStyle = (isUser: boolean): CSSProperties => ({
  alignSelf: isUser ? 'flex-end' : 'flex-start', maxWidth: '75%',
  padding: '8px 14px', borderRadius: 12, fontSize: 13, lineHeight: 1.5,
  backgroundColor: isUser ? 'rgba(88,214,141,0.15)' : C.border2, color: C.text,
})

const SCENE_COLOURS = ['#e74c3c', '#f39c12', '#58d68d']

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------
function DropZone({ onFile }: { onFile: (f: File) => void }) {
  const onDrop = useCallback((files: File[]) => { if (files[0]) onFile(files[0]) }, [onFile])
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'image/*': ['.jpg', '.jpeg', '.png'] }, maxFiles: 1,
  })
  return (
    <div {...getRootProps()} style={dropStyle(isDragActive)}>
      <input {...getInputProps()} />
      <div style={{ fontSize: 48, marginBottom: 12 }}>🌿</div>
      <p style={{ color: C.text, fontWeight: 500, marginBottom: 4, marginTop: 0 }}>
        {isDragActive ? 'Drop it here!' : 'Drop a leaf image here'}
      </p>
      <p style={{ color: C.muted, fontSize: 13, margin: 0 }}>or click to browse — JPG, JPEG, PNG</p>
    </div>
  )
}

function SpreadChart({ scenarios }: { scenarios: Scenario[] }) {
  if (!scenarios?.length) return null
  const days = scenarios[0]?.daily_counts?.length ?? 0
  const data = Array.from({ length: days }, (_, i) => {
    const pt: Record<string, number | string> = { day: i }
    scenarios.forEach(sc => { pt[sc.intervention_name] = sc.daily_counts[i]?.infected ?? 0 })
    return pt
  })
  return (
    <div style={S.chartCard}>
      <p style={{ ...S.tableTitle, marginBottom: 16 }}>🦠 Infection Spread — Scenario Comparison</p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
          <XAxis dataKey="day" stroke={C.muted} tick={{ fontSize: 11, fill: C.muted }} />
          <YAxis stroke={C.muted} tick={{ fontSize: 11, fill: C.muted }} />
          <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} labelStyle={{ color: C.muted }} />
          <Legend wrapperStyle={{ fontSize: 12, color: C.muted }} />
          {scenarios.map((sc, i) => (
            <Line key={sc.intervention_name} type="monotone" dataKey={sc.intervention_name}
              stroke={SCENE_COLOURS[i] ?? C.muted} strokeWidth={2} dot={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------
export default function Home() {
  const [imageFile, setImageFile]   = useState<File | null>(null)
  const [imagePreview, setPreview]  = useState<string | null>(null)
  const [loading, setLoading]       = useState(false)
  const [report, setReport]         = useState<DSSReport | null>(null)
  const [error, setError]           = useState<string | null>(null)
  const [tab, setTab]               = useState<'detection'|'simulation'|'treatment'|'chatbot'>('detection')
  const [chatInput, setChatInput]   = useState('')
  const [chatLog, setChatLog]       = useState<{ role: string; text: string }[]>([])
  const [chatBusy, setChatBusy]     = useState(false)

  const handleFile = (f: File) => {
    setImageFile(f); setPreview(URL.createObjectURL(f)); setReport(null); setError(null)
  }
  const analyse = async () => {
    if (!imageFile) return
    setLoading(true); setError(null); setReport(null)
    const form = new FormData(); form.append('file', imageFile)
    try {
      const res = await axios.post<DSSReport>('/api/full_analysis/', form)
      setReport(res.data); setTab('detection')
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Cannot connect to backend. Make sure FastAPI is running on port 8000.')
    } finally { setLoading(false) }
  }
  const sendChat = async () => {
    if (!chatInput.trim()) return
    const q = chatInput.trim(); setChatLog(h => [...h, { role: 'user', text: q }]); setChatInput(''); setChatBusy(true)
    try {
      const res = await axios.post('/api/chat/', { question: q })
      setChatLog(h => [...h, { role: 'ai', text: res.data.answer }])
    } catch { setChatLog(h => [...h, { role: 'ai', text: 'Chatbot unavailable right now.' }]) }
    finally { setChatBusy(false) }
  }

  const name    = (report?.disease_class ?? report?.disease_name ?? '').replace(/___/g, ' — ').replace(/_/g, ' ')
  const advice  = report?.expert_advice ?? report?.rag_treatment_protocol?.retrieved_context ?? ''
  const urgency = report?.urgency_level ?? ''

  return (
    <div style={S.page}>
      {/* ── Nav ── */}
      <nav style={S.nav}>
        <span style={S.navLogo}>🌿</span>
        <span style={S.navTitle}>AgriVision AI</span>
        <span style={S.navSub}>Crop Disease Detection &amp; Decision Support</span>
      </nav>

      <main style={S.main}>
        <h1 style={S.h1}>Crop Disease Detection</h1>
        <p style={S.subtitle}>Upload a leaf image to get AI-powered diagnosis, spread simulation, and treatment recommendations.</p>

        {/* ── Upload row ── */}
        <div style={S.grid2}>
          <DropZone onFile={handleFile} />
          {imagePreview && (
            <div style={S.card}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={imagePreview} alt="Uploaded leaf" style={S.imgPrev as CSSProperties} />
              <div style={S.cardBody}>
                <p style={S.filename}>{imageFile?.name}</p>
                <button style={btnStyle(loading)} onClick={analyse} disabled={loading}>
                  {loading ? '⏳  Analysing…' : '🚀  Analyse Crop'}
                </button>
              </div>
            </div>
          )}
        </div>
        {error && <div style={S.error}>{error}</div>}

        {/* ── Results ── */}
        {report && (
          <div>
            <div style={S.resultHdr}>
              <h2 style={S.h2}>{report.is_healthy ? '✅ Healthy Plant' : `🦠 ${name}`}</h2>
              {urgency && <span style={badgeStyle(urgency)}>{urgency === 'NONE' ? 'NO ACTION NEEDED' : `${urgency} URGENCY`}</span>}
            </div>

            {/* Confidence */}
            <div style={S.confCard}>
              <div style={S.confRow}>
                <span>Model confidence</span>
                <span style={{ color: report.vision_detection.confidence >= 80 ? C.green : C.yellow }}>
                  {report.vision_detection.confidence.toFixed(1)}%
                </span>
              </div>
              <div style={S.confBar}><div style={confFill(report.vision_detection.confidence)} /></div>
              <p style={{ fontSize: 12, color: C.muted, marginTop: 6, marginBottom: 0 }}>{report.vision_detection.confidence_label}</p>
            </div>

            {/* Recommendation */}
            {report.recommendation && (
              <div style={S.recBanner}>
                <p style={S.recLabel}>Recommendation</p>
                <p style={S.recText}>{report.recommendation.action_summary}</p>
                {report.simulation_summary && (
                  <div style={S.statsRow}>
                    {[
                      { label: 'Best strategy',      value: report.recommendation.best_strategy,               color: C.green },
                      { label: 'Projected crop loss', value: `${report.recommendation.projected_loss?.toFixed(1)}%`, color: C.red },
                      { label: 'Saving vs no action', value: `${report.recommendation.saving_vs_baseline?.toFixed(1)}%`, color: C.green },
                    ].map(st => (
                      <div key={st.label}>
                        <div style={S.statLabel}>{st.label}</div>
                        <div style={{ fontSize: 14, fontWeight: 700, color: st.color }}>{st.value}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Tabs */}
            <div style={S.tabRow}>
              {(['detection','simulation','treatment','chatbot'] as const).map(t => (
                <button key={t} style={tabStyle(tab === t)} onClick={() => setTab(t)}>
                  {{ detection:'🔬 Detection', simulation:'📈 Simulation', treatment:'📋 Treatment', chatbot:'💬 Chatbot' }[t]}
                </button>
              ))}
            </div>

            {/* Detection */}
            {tab === 'detection' && (
              <div>
                <div style={S.grid2}>
                  <div style={S.statCard}>
                    <div style={S.statLabel}>Detected class</div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: C.text, marginTop: 4 }}>{name}</div>
                  </div>
                  <div style={S.statCard}>
                    <div style={S.statLabel}>Urgency level</div>
                    <div style={{ fontSize: 13, fontWeight: 600, marginTop: 4, color: urgency==='HIGH'?C.red:urgency==='MEDIUM'?C.yellow:C.green }}>
                      {urgency || 'NONE'}
                    </div>
                  </div>
                </div>
                {report.ranked_interventions?.length > 0 && (
                  <div style={S.tableCard}>
                    <p style={S.tableTitle}>📊 Intervention Rankings</p>
                    <table style={S.table}>
                      <thead>
                        <tr>
                          <th style={S.th}>Rank</th>
                          <th style={S.th}>Strategy</th>
                          <th style={{ ...S.th, textAlign: 'right' }}>Crop Loss</th>
                          <th style={{ ...S.th, textAlign: 'right' }}>Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.ranked_interventions.map(r => (
                          <tr key={r.strategy_key}>
                            <td style={S.td}>{r.rank===1?'🥇':r.rank===2?'🥈':'🥉'}</td>
                            <td style={{ ...S.td, color: C.text }}>{r.intervention_name}</td>
                            <td style={{ ...S.td, textAlign:'right', fontFamily:'monospace', color: r.rank===1?C.green:C.red }}>
                              {r.crop_loss_percent.toFixed(1)}%
                            </td>
                            <td style={{ ...S.td, textAlign:'right', fontFamily:'monospace', color: C.muted }}>
                              {r.final_score.toFixed(4)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {/* Simulation */}
            {tab === 'simulation' && (
              <div>
                {report.simulation_summary ? (
                  <>
                    <div style={S.statGrid3}>
                      {[
                        { label: 'Grid size',               value: `${report.simulation_summary.grid_size[0]}×${report.simulation_summary.grid_size[1]}` },
                        { label: 'Days simulated',          value: String(report.simulation_summary.days_simulated) },
                        { label: 'Baseline loss (no action)', value: `${report.simulation_summary.baseline_crop_loss?.toFixed(1)}%` },
                      ].map(st => (
                        <div key={st.label} style={S.statCard}>
                          <div style={S.statLabel}>{st.label}</div>
                          <div style={{ fontSize: 14, fontWeight: 600, color: C.text, marginTop: 4 }}>{st.value}</div>
                        </div>
                      ))}
                    </div>
                    {report.scenarios && <SpreadChart scenarios={report.scenarios} />}
                  </>
                ) : (
                  <div style={{ ...S.statCard, padding: 40, textAlign: 'center', color: C.muted, fontSize: 14 }}>
                    No simulation — healthy plants don't require spread modelling.
                  </div>
                )}
              </div>
            )}

            {/* Treatment */}
            {tab === 'treatment' && (
              <div style={S.adviceCard}><ReactMarkdown>{advice}</ReactMarkdown></div>
            )}

            {/* Chatbot */}
            {tab === 'chatbot' && (
              <div style={S.chatCard}>
                <div style={S.chatMsgs}>
                  {chatLog.length === 0 && <p style={S.chatEmpty}>Ask any question about crop diseases and treatments.</p>}
                  {chatLog.map((m, i) => <div key={i} style={chatMsgStyle(m.role==='user')}>{m.text}</div>)}
                  {chatBusy && <div style={chatMsgStyle(false)}>Thinking…</div>}
                </div>
                <div style={S.chatFoot}>
                  <input style={S.chatInput} value={chatInput}
                    onChange={e => setChatInput(e.target.value)}
                    onKeyDown={e => e.key==='Enter' && sendChat()}
                    placeholder="Ask about this disease or treatment…" />
                  <button style={S.chatBtn} onClick={sendChat} disabled={chatBusy}>Send</button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── System overview ── */}
        {!report && !loading && (
          <div>
            <p style={S.secTitle}>System Components</p>
            <div style={S.modGrid}>
              {[
                { icon:'🤖', name:'Chatbot Q&A',      desc:'Groq LLM grounded in RAG knowledge base' },
                { icon:'📝', name:'Prompt Engine',    desc:'Centralised prompt templates for all LLM calls' },
                { icon:'🗄️', name:'Data Pipeline',    desc:'PlantVillage + PlantDoc hybrid, 21 classes' },
                { icon:'🔍', name:'RAG Knowledge',    desc:'ChromaDB + HuggingFace embeddings' },
                { icon:'👁️', name:'Vision Model',     desc:'ResNet-50 fine-tuned, softmax confidence' },
                { icon:'🎨', name:'GAN Synthesis',    desc:'DCGAN augments underrepresented classes' },
                { icon:'📈', name:'Simulation',       desc:'SIR spread model, 3 intervention scenarios' },
                { icon:'🖥️', name:'Dashboard',        desc:'This web interface — end-to-end workflow' },
                { icon:'🧠', name:'Decision Support', desc:'Scores and ranks interventions by crop loss' },
              ].map(c => (
                <div key={c.name} style={S.modCard}>
                  <div style={S.modIcon}>{c.icon}</div>
                  <div style={S.modName}>{c.name}</div>
                  <p style={S.modDesc}>{c.desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}