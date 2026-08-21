let cachedPort: number | null = null

async function resolvePort(): Promise<number> {
  if (cachedPort != null) return cachedPort
  if (window.electronAPI) {
    cachedPort = await window.electronAPI.backend.port()
    return cachedPort
  }
  cachedPort = 8000
  return cachedPort
}

async function getBaseUrl(): Promise<string> {
  if (window.electronAPI) {
    const port = await resolvePort()
    return `http://127.0.0.1:${port}/api`
  }
  return '/api'
}

// Eagerly resolve the base URL; exported helper awaits resolution.
const basePromise = getBaseUrl()

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const BASE_URL = await basePromise
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    throw new ApiError(res.status, text)
  }

  return res.json()
}

export const api = {
  // System
  health: () => request<{ status: string }>('/system/health'),
  gpuInfo: () => request<{ gpus: import('./types').GpuStats[] }>('/system/gpu'),
  configs: () => request<{ accelerate: string[]; deepspeed: string[] }>('/system/configs'),

  // Training
  startTraining: (config: Record<string, unknown>) =>
    request<{ run_id: string }>('/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  stopTraining: (runId: string) =>
    request<{ success: boolean }>(`/training/stop/${runId}`, { method: 'POST' }),

  listRuns: () => request<{ runs: import('./types').TrainingRun[] }>('/training/runs'),

  getRun: (runId: string) => request<import('./types').TrainingRun>(`/training/runs/${runId}`),

  deleteRun: (runId: string) =>
    request<{ success: boolean }>(`/training/runs/${runId}`, { method: 'DELETE' }),

  continueTraining: (runId: string, config?: Record<string, unknown>) =>
    request<{ run_id: string }>(`/training/runs/${runId}/continue`, {
      method: 'POST',
      body: JSON.stringify(config ?? {}),
    }),

  // Models
  listModels: () => request<{ models: import('./types').ModelInfo[] }>('/models/local'),
  listTrainedModels: () => request<{ models: import('./types').ModelInfo[] }>('/models/trained'),
  downloadModel: (data: import('./types').HFDownloadRequest) =>
    request<{ success: boolean; path: string }>('/models/download', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  deleteLocalModel: (name: string) =>
    request<{ success: boolean }>(`/models/local/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  estimateMemory: (config: Record<string, unknown>) =>
    request<{ vram_gb: number; params: number }>('/models/estimate-memory', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  // Datasets
  listDatasets: () => request<{ datasets: import('./types').DatasetInfo[] }>('/datasets/local'),
  downloadDataset: (data: import('./types').HFDatasetDownloadRequest) =>
    request<{ success: boolean }>('/datasets/download', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  previewDataset: (name: string) =>
    request<{ rows: Record<string, unknown>[] }>(`/datasets/preview/${encodeURIComponent(name)}`),
  deleteDataset: (name: string) =>
    request<{ success: boolean }>(`/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  // Chat
  loadChatModel: (modelPath: string, backend: string) =>
    request<{ success: boolean }>('/chat/load', {
      method: 'POST',
      body: JSON.stringify({ model_path: modelPath, backend }),
    }),
  unloadChatModel: () => request<{ success: boolean }>('/chat/unload', { method: 'POST' }),
  chatBackends: () => request<{ backends: Array<'transformers' | 'vllm' | 'llama.cpp'> }>('/chat/backends'),

  // Agent
  startAgentServer: (config?: Record<string, unknown>) =>
    request<{ success: boolean }>('/agent/start-server', {
      method: 'POST',
      body: JSON.stringify(config ?? {}),
    }),
  stopAgentServer: () =>
    request<{ success: boolean }>('/agent/stop-server', { method: 'POST' }),
  agentServerStatus: () =>
    request<{ running: boolean; model?: string }>('/agent/server-status'),
  listAgentSessions: () =>
    request<{ sessions: import('./types').AgentSession[] }>('/agent/sessions'),
  createAgentSession: () =>
    request<import('./types').AgentSession>('/agent/sessions', { method: 'POST' }),
  deleteAgentSession: (id: string) =>
    request<{ success: boolean }>(`/agent/sessions/${id}`, { method: 'DELETE' }),
  agentTools: () =>
    request<{ groups: import('./types').AgentToolGroup[]; tools: import('./types').AgentTool[] }>('/agent/tools'),
  agentCapabilities: () =>
    request<{ capabilities: Record<string, unknown> }>('/agent/capabilities'),
  agentRuns: (status: string = 'all') =>
    request<{ status_filter: string; count: number; runs: import('./types').AgentRun[] }>(`/agent/runs?status=${encodeURIComponent(status)}`),
  stopAgentRun: (runId: string) =>
    request<{ run_id: string; stopped: boolean; reason?: string }>(`/agent/runs/${encodeURIComponent(runId)}/stop`, { method: 'POST' }),

  // VLM
  startVLMTraining: (config: Record<string, unknown>) =>
    request<{ run_id: string }>('/vlm/start', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  stopVLMTraining: (runId: string) =>
    request<{ success: boolean }>(`/vlm/stop/${runId}`, { method: 'POST' }),
  listVLMRuns: () => request<{ runs: import('./types').TrainingRun[] }>('/vlm/runs'),
  loadVLM: (modelPath: string) =>
    request<{ success: boolean }>('/vlm/load', {
      method: 'POST',
      body: JSON.stringify({ model_path: modelPath }),
    }),
  unloadVLM: () => request<{ success: boolean }>('/vlm/unload', { method: 'POST' }),
  generateVLM: (data: { prompt: string; image_base64: string; max_tokens?: number; temperature?: number }) =>
    request<{ text: string }>('/vlm/generate', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Study
  studyIndex: (lang: string) =>
    request<{ documents: Array<{ name: string; path: string; source: 'local' | 'github' }> }>(`/study/index?lang=${lang}`),
  studyDocument: (p: string) =>
    request<{ content: string }>(`/study/document?path=${encodeURIComponent(p)}`),

  // Blueprints
  listBlueprints: () =>
    request<{ blueprints: Array<{ name: string; path: string; hidden_size: number; vocab_size: number; num_blocks: number; modified_at: string }> }>('/blueprints/list'),
  getBlueprint: (name: string) =>
    request<{ blueprint: Record<string, unknown> }>(`/blueprints/get/${encodeURIComponent(name)}`),
  saveBlueprint: (name: string, blueprint: Record<string, unknown>) =>
    request<{ success: boolean; path: string }>('/blueprints/save', {
      method: 'POST',
      body: JSON.stringify({ name, blueprint }),
    }),
  deleteBlueprint: (name: string) =>
    request<{ success: boolean }>(`/blueprints/delete/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  listBlocks: () =>
    request<{ blocks: Array<{ type: string; description: string }> }>('/blueprints/blocks'),

  // Notebooks
  listNotebooks: () =>
    request<{ notebooks: Array<{ name: string; path: string; size_bytes: number; modified_at: string }> }>('/notebooks/list'),
  notebookTemplates: () =>
    request<{ templates: Array<{ name: string; description: string; file: string }> }>('/notebooks/templates'),
  jupyterStatus: (url: string) =>
    request<{ running: boolean; url: string }>(`/notebooks/jupyter-status?url=${encodeURIComponent(url)}`),
  startJupyter: () =>
    request<{ running: boolean; url: string; token: string; root_dir: string }>('/notebooks/jupyter-start', { method: 'POST' }),
  stopJupyter: () =>
    request<{ success: boolean }>('/notebooks/jupyter-stop', { method: 'POST' }),

  // Training extras
  grpoSamples: (runId: string) =>
    request<{ samples: Array<Record<string, unknown>> }>(`/training/runs/${runId}/grpo-samples`),
  runLogs: (runId: string, lines = 500) =>
    request<{ run_id: string; stdout: string; stderr: string }>(`/training/runs/${runId}/logs?lines=${lines}`),
  estimateMemoryDetailed: (config: Record<string, unknown>, batchSize: number, distributedMode: string, numGpus: number) =>
    request<Record<string, unknown>>('/models/estimate-memory-detailed', {
      method: 'POST',
      body: JSON.stringify({ config, batch_size: batchSize, distributed_mode: distributedMode, num_gpus: numGpus }),
    }),
  getChatTemplate: (modelPath: string) =>
    request<{ has_template: boolean; template: string | null }>(`/models/chat-template?model_path=${encodeURIComponent(modelPath)}`),
}

export function createMetricsWebSocket(
  runId: string,
  onMessage: (data: import('./types').MetricsSnapshot) => void,
  onError?: (err: Event) => void,
): Promise<WebSocket> {
  return resolvePort().then((port) => {
    const wsUrl = window.electronAPI
      ? `ws://127.0.0.1:${port}/ws/metrics/${runId}`
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/metrics/${runId}`
    const ws = new WebSocket(wsUrl)

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        onMessage(data)
      } catch { /* ignore malformed */ }
    }

    ws.onerror = (err) => onError?.(err)
    return ws
  })
}

export function createChatStream(
  request: import('./types').ChatRequest,
  onChunk: (text: string) => void,
  onDone: () => void,
  onError?: (err: Error) => void,
): AbortController {
  const controller = new AbortController()

  basePromise.then((BASE_URL) => fetch(`${BASE_URL}/chat/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal: controller.signal,
  }))
    .then(async (res) => {
      if (!res.ok) throw new Error(await res.text())
      const reader = res.body?.getReader()
      if (!reader) throw new Error('No reader')

      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const text = decoder.decode(value, { stream: true })
        for (const line of text.split('\n')) {
          if (line.startsWith('data: ')) {
            const payload = line.slice(6)
            if (payload === '[DONE]') {
              onDone()
              return
            }
            try {
              const parsed = JSON.parse(payload)
              if (parsed.text) onChunk(parsed.text)
            } catch { /* skip */ }
          }
        }
      }
      onDone()
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onError?.(err)
    })

  return controller
}

export function createAgentChatStream(
  sessionId: string,
  message: string,
  onChunk: (text: string) => void,
  onToolCall: (tool: string, args: Record<string, unknown>) => void,
  onDone: () => void,
  onError?: (err: Error) => void,
): AbortController {
  const controller = new AbortController()

  basePromise.then((BASE_URL) => fetch(`${BASE_URL}/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal: controller.signal,
  }))
    .then(async (res) => {
      if (!res.ok) throw new Error(await res.text())
      const reader = res.body?.getReader()
      if (!reader) throw new Error('No reader')

      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const text = decoder.decode(value, { stream: true })
        for (const line of text.split('\n')) {
          if (line.startsWith('data: ')) {
            const payload = line.slice(6)
            if (payload === '[DONE]') {
              onDone()
              return
            }
            let parsed: { type?: string; content?: string; tool?: string; args?: Record<string, unknown> } | null = null
            try {
              parsed = JSON.parse(payload)
            } catch { /* skip */ }
            if (parsed?.type === 'text') onChunk(parsed.content ?? '')
            if (parsed?.type === 'tool_call') onToolCall(parsed.tool ?? '', parsed.args ?? {})
            if (parsed?.type === 'error') throw new Error(parsed.content ?? 'Agent error')
          }
        }
      }
      onDone()
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onError?.(err)
      onDone()
    })

  return controller
}
