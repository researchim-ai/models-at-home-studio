import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import React from 'react'
import { render, waitFor, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from '@/App'
import '@/i18n'

// ---------------------------------------------------------------------------
// Fake backend. Responses intentionally mimic the *worst realistic* shapes the
// real API can return — including the legacy Streamlit session shape (which
// uses `session_id`, has no `name`, and can carry non-string message content).
// These are precisely the shapes that previously crashed the Agent page to a
// black screen, so the harness now guards against that class of regression.
// ---------------------------------------------------------------------------

const legacySession = {
  session_id: '20260706_073432',
  updated_at: '2026-07-06T07:37:32.873274',
  model_path: '/app/models/Qwen3.5-9B.gguf',
  messages: [
    { role: 'user', content: 'Привет' },
    { role: 'assistant', content: 'Привет!', trace: [{ step: 1, prompt: '...' }] },
    // Deliberately malformed: non-string content must not crash rendering.
    { role: 'assistant', content: { tool: 'list_datasets' } },
    { role: 'assistant', content: null },
  ],
}

const modernSession = {
  id: 'de52c5166a4e',
  name: 'Session de52c5',
  messages: [],
  created_at: '2026-07-07T11:43:50.353035',
}

const agentTools = {
  groups: [
    {
      category: 'discovery',
      title: 'Разведка',
      tools: [
        { name: 'list_datasets', category: 'discovery', description: 'Datasets', arguments: {} },
      ],
    },
  ],
  tools: [{ name: 'list_datasets', category: 'discovery', description: 'Datasets', arguments: {} }],
}

const agentCapabilities = {
  capabilities: {
    text_training: {
      stages: ['pretrain', 'sft'],
      tool: 'start_text_training',
      supports: ['LoRA', 'multi-GPU'],
      key_config_params: ['stage', 'learning_rate'],
    },
  },
}

const agentRuns = {
  status_filter: 'all',
  count: 1,
  runs: [
    {
      run_id: 'run_abc',
      run_dir: '/runs/run_abc',
      pid: 1234,
      is_running: true,
      metrics: { status: 'running', step: 10, total_steps: 100, loss: 0.42 },
      config_summary: { stage: 'sft', training_backend: 'models-at-home' },
    },
  ],
}

function payloadFor(url: string): unknown {
  if (url.includes('/agent/sessions')) return { sessions: [legacySession, modernSession] }
  if (url.includes('/agent/tools')) return agentTools
  if (url.includes('/agent/capabilities')) return agentCapabilities
  if (url.includes('/agent/runs')) return agentRuns
  if (url.includes('/agent/server-status')) return { running: false }
  // Generous superset for everything else. Pages use `data?.x ?? []`, so this
  // satisfies the remaining endpoints without a per-route map.
  return {
    status: 'ok',
    running: false,
    count: 0,
    runs: [],
    sessions: [],
    groups: [],
    tools: [],
    gpus: [],
    backends: [],
    accelerate: [],
    deepspeed: [],
    notebooks: [],
    files: [],
    models: [],
    datasets: [],
    capabilities: {},
    metrics: {},
  }
}

function makeResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
    headers: new Headers(),
  } as unknown as Response
}

function installFetchMock() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      return makeResponse(payloadFor(url))
    }),
  )
}

// React logs render errors to console.error (before any error boundary). We
// record them so a crashing page fails the test with the real stack trace.
function installErrorCapture(sink: Error[]) {
  return vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    const first = args[0]
    if (first instanceof Error) sink.push(first)
    else if (
      typeof first === 'string' &&
      /Error|Uncaught|not a function|undefined is not|Cannot read/.test(first)
    ) {
      sink.push(new Error(args.map(String).join(' ')))
    }
  })
}

function renderAt(path: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  const utils = render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return { queryClient, ...utils }
}

const routes = [
  { path: '/', name: 'TrainingStudio' },
  { path: '/vlm', name: 'VLMStudio' },
  { path: '/builder', name: 'ModelBuilder' },
  { path: '/study', name: 'StudyCenter' },
  { path: '/notebooks', name: 'Notebooks' },
  { path: '/agent', name: 'AgentStudio' },
  { path: '/settings', name: 'Settings' },
]

describe('page smoke tests', () => {
  const capturedErrors: Error[] = []
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    capturedErrors.length = 0
    installFetchMock()
    consoleErrorSpy = installErrorCapture(capturedErrors)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    consoleErrorSpy.mockRestore()
  })

  for (const route of routes) {
    it(`renders ${route.name} (${route.path}) without crashing`, async () => {
      let renderError: unknown = null
      let queryClient: QueryClient | null = null
      try {
        const r = renderAt(route.path)
        queryClient = r.queryClient
        await waitFor(() => expect(screen.getByText('Models at Home')).toBeInTheDocument(), {
          timeout: 5000,
        })
        await new Promise((res) => setTimeout(res, 150))
      } catch (err) {
        renderError = err
      }

      expect(renderError, `${route.name} threw during render`).toBeNull()
      expect(
        capturedErrors,
        `${route.name} produced React errors:\n${capturedErrors
          .map((e) => e.stack ?? e.message)
          .join('\n---\n')}`,
      ).toHaveLength(0)

      queryClient?.clear()
    })
  }

  it('AgentStudio survives all tabs and an active legacy session', async () => {
    let queryClient: QueryClient | null = null
    let renderError: unknown = null
    try {
      const r = renderAt('/agent')
      queryClient = r.queryClient
      await waitFor(() => expect(screen.getByText('Models at Home')).toBeInTheDocument(), {
        timeout: 5000,
      })

      // Wait for the tab bar to mount, then visit every tab (by index, so this
      // is language-agnostic) so Runs/Tools/Capabilities panels render.
      await waitFor(() => expect(screen.getAllByRole('tab').length).toBeGreaterThan(1))
      const tabCount = screen.getAllByRole('tab').length
      for (let i = 0; i < tabCount; i++) {
        fireEvent.click(screen.getAllByRole('tab')[i])
        await new Promise((res) => setTimeout(res, 60))
      }
      // Return to the first tab (chat).
      fireEvent.click(screen.getAllByRole('tab')[0])

      // Select the legacy session tab (this loads its messages, exercising the
      // non-string content guard in the chat panel).
      await new Promise((res) => setTimeout(res, 100))
      const sessionButtons = screen.queryAllByRole('button')
      const legacyTab = sessionButtons.find((b) => /20260706|session/i.test(b.textContent ?? ''))
      if (legacyTab) fireEvent.click(legacyTab)
      await new Promise((res) => setTimeout(res, 150))
    } catch (err) {
      renderError = err
    }

    expect(renderError, 'AgentStudio deep test threw').toBeNull()
    expect(
      capturedErrors,
      `AgentStudio produced React errors:\n${capturedErrors
        .map((e) => e.stack ?? e.message)
        .join('\n---\n')}`,
    ).toHaveLength(0)

    queryClient?.clear()
  })
})
