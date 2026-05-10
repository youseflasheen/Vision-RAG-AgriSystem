'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'

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
// Colour helpers
// ---------------------------------------------------------------------------
const URGENCY_COLOURS: Record<string, string> = {
  HIGH:   'text-red-400 bg-red-400/10 border-red-400/30',
  MEDIUM: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30',
  LOW:    'text-green-400 bg-green-400/10 border-green-400/30',
  NONE:   'text-green-400 bg-green-400/10 border-green-400/30',
}

const CHART_COLOURS = {
  susceptible: '#58d68d',
  infected:    '#e74c3c',
  removed:     '#8b949e',
}

const SCENARIO_LINE_COLOURS = ['#e74c3c', '#f39c12', '#58d68d']

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function NavBar() {
  return (
    <nav className="sticky top-0 z-50 bg-[#161b22] border-b border-[#21262d] px-6 py-3 flex items-center gap-4">
      <span className="text-2xl">🌿</span>
      <span className="font-bold text-[#58d68d] text-base tracking-tight">AgriVision AI</span>
      <span className="ml-auto text-xs text-[#8b949e]">Crop Disease Detection & Decision Support</span>
    </nav>
  )
}

function UploadZone({ onFile }: { onFile: (f: File) => void }) {
  const onDrop = useCallback((files: File[]) => { if (files[0]) onFile(files[0]) }, [onFile])
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'image/*': ['.jpg', '.jpeg', '.png'] }, maxFiles: 1,
  })

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all
        ${isDragActive
          ? 'border-[#58d68d] bg-[#58d68d]/5'
          : 'border-[#30363d] hover:border-[#58d68d]/50 hover:bg-[#161b22]'}`}
    >
      <input {...getInputProps()} />
      <div className="text-5xl mb-4">🌿</div>
      <p className="text-[#e6edf3] font-medium mb-1">
        {isDragActive ? 'Drop the leaf image here' : 'Drop a leaf image here'}
      </p>
      <p className="text-[#8b949e] text-sm">or click to browse — JPG, JPEG, PNG</p>
    </div>
  )
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const colour = pct >= 80 ? '#58d68d' : pct >= 50 ? '#f39c12' : '#e74c3c'
  return (
    <div>
      <div className="flex justify-between text-xs text-[#8b949e] mb-1">
        <span>Model confidence</span><span style={{ color: colour }}>{pct}%</span>
      </div>
      <div className="h-2 bg-[#21262d] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: colour }}
        />
      </div>
    </div>
  )
}

function SpreadChart({ scenarios }: { scenarios: Scenario[] }) {
  if (!scenarios?.length) return null

  // Build chart data: one point per day, infected count per scenario
  const maxDays = scenarios[0]?.daily_counts?.length ?? 0
  const chartData = Array.from({ length: maxDays }, (_, i) => {
    const point: Record<string, number | string> = { day: i }
    scenarios.forEach(s => {
      point[s.intervention_name] = s.daily_counts[i]?.infected ?? 0
    })
    return point
  })

  return (
    <div className="bg-[#161b22] border border-[#21262d] rounded-xl p-6">
      <h3 className="font-semibold text-[#e6edf3] mb-4 text-sm">
        🦠 Infection Spread — Scenario Comparison
      </h3>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
          <XAxis dataKey="day" stroke="#8b949e" tick={{ fontSize: 11 }} label={{ value: 'Day', position: 'insideBottomRight', offset: -5, fill: '#8b949e', fontSize: 11 }} />
          <YAxis stroke="#8b949e" tick={{ fontSize: 11 }} label={{ value: 'Infected cells', angle: -90, position: 'insideLeft', fill: '#8b949e', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#8b949e' }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: '#8b949e' }} />
          {scenarios.map((s, i) => (
            <Line
              key={s.intervention_name}
              type="monotone"
              dataKey={s.intervention_name}
              stroke={SCENARIO_LINE_COLOURS[i] ?? '#8b949e'}
              strokeWidth={2}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function InterventionTable({ interventions }: { interventions: RankedIntervention[] }) {
  if (!interventions?.length) return null
  return (
    <div className="bg-[#161b22] border border-[#21262d] rounded-xl p-6">
      <h3 className="font-semibold text-[#e6edf3] mb-4 text-sm">📊 Intervention Rankings</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[#8b949e] border-b border-[#21262d]">
            <th className="text-left pb-2 font-medium">Rank</th>
            <th className="text-left pb-2 font-medium">Strategy</th>
            <th className="text-right pb-2 font-medium">Crop Loss</th>
            <th className="text-right pb-2 font-medium">Score</th>
          </tr>
        </thead>
        <tbody>
          {interventions.map(row => (
            <tr key={row.strategy_key} className="border-b border-[#21262d]/50">
              <td className="py-2 text-[#8b949e]">
                {row.rank === 1 ? '🥇' : row.rank === 2 ? '🥈' : '🥉'}
              </td>
              <td className="py-2 text-[#e6edf3]">{row.intervention_name}</td>
              <td className="py-2 text-right font-mono">
                <span className={row.rank === 1 ? 'text-[#58d68d]' : 'text-[#e74c3c]'}>
                  {row.crop_loss_percent.toFixed(1)}%
                </span>
              </td>
              <td className="py-2 text-right font-mono text-[#8b949e]">
                {row.final_score.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------
export default function Home() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState<DSSReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'diagnosis' | 'simulation' | 'advice' | 'chatbot'>('diagnosis')
  const [chatInput, setChatInput] = useState('')
  const [chatHistory, setChatHistory] = useState<{ role: string; text: string }[]>([])
  const [chatLoading, setChatLoading] = useState(false)

  const handleFile = (file: File) => {
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setReport(null)
    setError(null)
  }

  const handleAnalyse = async () => {
    if (!imageFile) return
    setLoading(true)
    setError(null)
    setReport(null)

    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const res = await axios.post<DSSReport>('/api/full_analysis/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setReport(res.data)
      setActiveTab('diagnosis')
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? 'Failed to connect to backend. Is the FastAPI server running?')
    } finally {
      setLoading(false)
    }
  }

  const handleChat = async () => {
    if (!chatInput.trim()) return
    const question = chatInput.trim()
    setChatHistory(h => [...h, { role: 'user', text: question }])
    setChatInput('')
    setChatLoading(true)
    try {
      const res = await axios.post('/api/chat/', { question })
      setChatHistory(h => [...h, { role: 'ai', text: res.data.answer }])
    } catch {
      setChatHistory(h => [...h, { role: 'ai', text: 'Sorry, the chatbot is unavailable right now.' }])
    } finally {
      setChatLoading(false)
    }
  }

  const diseaseName = report?.disease_class ?? report?.disease_name ?? ''
  const adviceText = report?.expert_advice ?? report?.rag_treatment_protocol?.retrieved_context ?? ''
  const urgency = report?.urgency_level ?? ''
  const isHealthy = report?.is_healthy

  return (
    <div className="min-h-screen bg-[#0d1117]">
      <NavBar />

      <main className="max-w-5xl mx-auto px-4 py-8">

        {/* ── Upload section ── */}
        <section className="mb-8">
          <h1 className="text-2xl font-bold text-[#e6edf3] mb-1">Crop Disease Detection</h1>
          <p className="text-[#8b949e] text-sm mb-6">
            Upload a leaf image to get AI-powered diagnosis, spread simulation, and treatment recommendations.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
            <UploadZone onFile={handleFile} />

            {imagePreview && (
              <div className="bg-[#161b22] border border-[#21262d] rounded-xl overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={imagePreview} alt="Uploaded leaf" className="w-full object-cover max-h-56" />
                <div className="p-4">
                  <p className="text-xs text-[#8b949e] truncate mb-3">{imageFile?.name}</p>
                  <button
                    onClick={handleAnalyse}
                    disabled={loading}
                    className="w-full py-2.5 rounded-lg font-semibold text-sm
                      bg-[#58d68d] text-[#0d1117] hover:bg-[#58d68d]/90
                      disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    {loading ? '⏳ Analysing...' : '🚀 Analyse Crop'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
              {error}
            </div>
          )}
        </section>

        {/* ── Results ── */}
        {report && (
          <section>
            {/* Header */}
            <div className="flex flex-wrap items-center gap-3 mb-6">
              <h2 className="text-xl font-bold text-[#e6edf3]">
                {isHealthy ? '✅ Healthy Plant' : `🦠 ${diseaseName.replace(/___/g, ' — ').replace(/_/g, ' ')}`}
              </h2>
              {urgency && urgency !== 'NONE' && (
                <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${URGENCY_COLOURS[urgency]}`}>
                  {urgency} URGENCY
                </span>
              )}
              {isHealthy && (
                <span className="px-3 py-1 rounded-full text-xs font-semibold border text-green-400 bg-green-400/10 border-green-400/30">
                  NO ACTION NEEDED
                </span>
              )}
            </div>

            {/* Confidence */}
            <div className="bg-[#161b22] border border-[#21262d] rounded-xl p-5 mb-4">
              <ConfidenceBar value={report.vision_detection.confidence / 100} />
              <p className="text-xs text-[#8b949e] mt-2">
                {report.vision_detection.confidence_label} — {report.vision_detection.confidence.toFixed(2)}%
              </p>
            </div>

            {/* Recommendation banner */}
            {report.recommendation && (
              <div className="bg-[#58d68d]/5 border border-[#58d68d]/20 rounded-xl p-5 mb-6">
                <p className="text-xs font-semibold text-[#58d68d] uppercase tracking-wider mb-2">
                  Recommendation
                </p>
                <p className="text-[#e6edf3] text-sm leading-relaxed">
                  {report.recommendation.action_summary}
                </p>
                {report.simulation_summary && (
                  <div className="flex gap-6 mt-4">
                    <div>
                      <p className="text-xs text-[#8b949e]">Best strategy</p>
                      <p className="text-sm font-semibold text-[#58d68d]">{report.recommendation.best_strategy}</p>
                    </div>
                    <div>
                      <p className="text-xs text-[#8b949e]">Projected crop loss</p>
                      <p className="text-sm font-semibold text-[#e74c3c]">
                        {report.recommendation.projected_loss?.toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-[#8b949e]">Saving vs no action</p>
                      <p className="text-sm font-semibold text-[#58d68d]">
                        {report.recommendation.saving_vs_baseline?.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Tabs */}
            <div className="flex gap-1 mb-5 border-b border-[#21262d]">
              {(['diagnosis', 'simulation', 'advice', 'chatbot'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 text-sm font-medium capitalize transition-all border-b-2
                    ${activeTab === tab
                      ? 'border-[#58d68d] text-[#58d68d]'
                      : 'border-transparent text-[#8b949e] hover:text-[#e6edf3]'}`}
                >
                  {tab === 'diagnosis' ? '🔬 Detection' :
                   tab === 'simulation' ? '📈 Simulation' :
                   tab === 'advice' ? '📋 Treatment' : '💬 Chatbot'}
                </button>
              ))}
            </div>

            {/* Tab: Detection */}
            {activeTab === 'diagnosis' && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-[#161b22] border border-[#21262d] rounded-xl p-5">
                    <p className="text-xs text-[#8b949e] mb-1">Detected class</p>
                    <p className="text-sm font-semibold text-[#e6edf3]">
                      {diseaseName.replace(/___/g, ' — ').replace(/_/g, ' ')}
                    </p>
                  </div>
                  <div className="bg-[#161b22] border border-[#21262d] rounded-xl p-5">
                    <p className="text-xs text-[#8b949e] mb-1">Urgency level</p>
                    <p className={`text-sm font-semibold ${
                      urgency === 'HIGH' ? 'text-red-400' :
                      urgency === 'MEDIUM' ? 'text-yellow-400' : 'text-green-400'
                    }`}>{urgency || 'NONE'}</p>
                  </div>
                </div>
                {report.ranked_interventions?.length > 0 && (
                  <InterventionTable interventions={report.ranked_interventions} />
                )}
              </div>
            )}

            {/* Tab: Simulation */}
            {activeTab === 'simulation' && (
              <div className="space-y-4">
                {report.simulation_summary ? (
                  <>
                    <div className="grid grid-cols-3 gap-4">
                      {[
                        { label: 'Grid size', value: `${report.simulation_summary.grid_size[0]}×${report.simulation_summary.grid_size[1]}` },
                        { label: 'Days simulated', value: report.simulation_summary.days_simulated },
                        { label: 'Baseline loss (no action)', value: `${report.simulation_summary.baseline_crop_loss?.toFixed(1)}%` },
                      ].map(stat => (
                        <div key={stat.label} className="bg-[#161b22] border border-[#21262d] rounded-xl p-4">
                          <p className="text-xs text-[#8b949e] mb-1">{stat.label}</p>
                          <p className="text-sm font-semibold text-[#e6edf3]">{stat.value}</p>
                        </div>
                      ))}
                    </div>
                    {report.scenarios && <SpreadChart scenarios={report.scenarios} />}
                  </>
                ) : (
                  <div className="bg-[#161b22] border border-[#21262d] rounded-xl p-8 text-center text-[#8b949e] text-sm">
                    No simulation data — healthy plants are not simulated.
                  </div>
                )}
              </div>
            )}

            {/* Tab: Treatment advice */}
            {activeTab === 'advice' && (
              <div className="bg-[#161b22] border border-[#21262d] rounded-xl p-6 prose max-w-none">
                <ReactMarkdown>{adviceText}</ReactMarkdown>
              </div>
            )}

            {/* Tab: Chatbot */}
            {activeTab === 'chatbot' && (
              <div className="bg-[#161b22] border border-[#21262d] rounded-xl flex flex-col h-96">
                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                  {chatHistory.length === 0 && (
                    <p className="text-[#8b949e] text-sm text-center mt-8">
                      Ask any question about crop diseases and treatments.
                    </p>
                  )}
                  {chatHistory.map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-xs px-4 py-2 rounded-xl text-sm
                        ${msg.role === 'user'
                          ? 'bg-[#58d68d]/20 text-[#e6edf3]'
                          : 'bg-[#21262d] text-[#c9d1d9]'}`}>
                        {msg.text}
                      </div>
                    </div>
                  ))}
                  {chatLoading && (
                    <div className="flex justify-start">
                      <div className="bg-[#21262d] px-4 py-2 rounded-xl text-sm text-[#8b949e]">
                        Thinking...
                      </div>
                    </div>
                  )}
                </div>
                <div className="border-t border-[#21262d] p-3 flex gap-2">
                  <input
                    value={chatInput}
                    onChange={e => setChatInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleChat()}
                    placeholder="Ask about this disease or treatment..."
                    className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-lg px-3 py-2 text-sm
                      text-[#e6edf3] placeholder-[#8b949e] focus:outline-none focus:border-[#58d68d]"
                  />
                  <button
                    onClick={handleChat}
                    disabled={chatLoading}
                    className="px-4 py-2 bg-[#58d68d] text-[#0d1117] rounded-lg text-sm font-semibold
                      hover:bg-[#58d68d]/90 disabled:opacity-50 transition-all"
                  >
                    Send
                  </button>
                </div>
              </div>
            )}
          </section>
        )}

        {/* ── System overview (shown when no result) ── */}
        {!report && !loading && (
          <section className="mt-12">
            <h2 className="text-base font-semibold text-[#e6edf3] mb-4">System Components</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {[
                { icon: '🤖', name: 'Chatbot Q&A',     desc: 'Groq LLM grounded in RAG knowledge base' },
                { icon: '📝', name: 'Prompt Engine',   desc: 'Centralised prompt templates for all LLM calls' },
                { icon: '🗄️', name: 'Data Pipeline',   desc: 'PlantVillage + PlantDoc hybrid, 21 classes' },
                { icon: '🔍', name: 'RAG Knowledge',   desc: 'ChromaDB + HuggingFace embeddings' },
                { icon: '👁️', name: 'Vision Model',    desc: 'ResNet-50 fine-tuned, softmax confidence' },
                { icon: '🎨', name: 'GAN Synthesis',   desc: 'DCGAN augments underrepresented classes' },
                { icon: '📈', name: 'Simulation',      desc: 'SIR spread model, 3 intervention scenarios' },
                { icon: '🖥️', name: 'Dashboard',       desc: 'This web interface — end-to-end workflow' },
                { icon: '🧠', name: 'Decision Support', desc: 'Scores and ranks interventions by crop loss' },
              ].map(c => (
                <div key={c.name} className="bg-[#161b22] border border-[#21262d] rounded-xl p-4
                  hover:border-[#30363d] transition-all">
                  <div className="text-xl mb-2">{c.icon}</div>
                  <div className="text-sm font-semibold text-[#e6edf3] mb-1">{c.name}</div>
                  <div className="text-xs text-[#8b949e] leading-relaxed">{c.desc}</div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}