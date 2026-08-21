import { app, BrowserWindow, ipcMain, shell, dialog, Menu, nativeImage } from 'electron'
import path from 'path'
import fs from 'fs'
import net from 'net'
import { spawn, spawnSync, type ChildProcess } from 'child_process'
import { DockerManager } from './docker'
import { loadConfig, saveConfig, type AppConfig, type BackendMode } from './config'

if (process.platform === 'linux') {
  app.disableHardwareAcceleration()
  app.commandLine.appendSwitch('disable-gpu')
  app.commandLine.appendSwitch('disable-gpu-compositing')
  app.commandLine.appendSwitch('disable-accelerated-2d-canvas')
}

let mainWindow: BrowserWindow | null = null
let dockerManager: DockerManager | null = null
let backendProcess: ChildProcess | null = null
let backendPort = 8000
let backendStartError: string | null = null
let currentMode: 'native' | 'docker' | 'offline' = 'offline'
let booting = false

// Boot phases — streamed to renderer for splash screen feedback
type BootPhase =
  | { phase: 'starting' }
  | { phase: 'checking-docker' }
  | { phase: 'docker-unavailable'; detail?: string }
  | { phase: 'docker-no-image' }
  | { phase: 'building-image'; line?: string }
  | { phase: 'starting-container' }
  | { phase: 'waiting-container-health'; attempt: number }
  | { phase: 'starting-python' }
  | { phase: 'python-starting'; line?: string }
  | { phase: 'ready'; mode: 'native' | 'docker'; port: number }
  | { phase: 'failed'; error: string }

let currentBootPhase: BootPhase = { phase: 'starting' }

function emitBootPhase(p: BootPhase) {
  currentBootPhase = p
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:boot-phase', p)
  }
}

const isDev = !!process.env.VITE_DEV_SERVER_URL
const VITE_DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

function getResourcePath(...segments: string[]): string {
  if (isDev) {
    return path.join(__dirname, '..', ...segments)
  }
  return path.join(process.resourcesPath, ...segments)
}

function getIconPath(): string | undefined {
  const candidates = isDev
    ? [
        path.join(__dirname, '..', 'build', 'icons', '512x512.png'),
        path.join(__dirname, '..', 'build', 'icon.png'),
      ]
    : [
        path.join(process.resourcesPath, 'icons', '512x512.png'),
        path.join(process.resourcesPath, 'icon.png'),
      ]
  for (const p of candidates) {
    if (fs.existsSync(p)) return p
  }
  return undefined
}

// ---------------------------------------------------------------------------
// Port helpers
// ---------------------------------------------------------------------------

function findAvailablePort(start = 8000): Promise<number> {
  return new Promise((resolve) => {
    const srv = net.createServer()
    srv.unref()
    srv.on('error', () => resolve(findAvailablePort(start + 1)))
    srv.listen(start, '127.0.0.1', () => {
      const addr = srv.address()
      const port = typeof addr === 'object' && addr ? addr.port : start
      srv.close(() => resolve(port))
    })
  })
}

// ---------------------------------------------------------------------------
// Python resolver
// ---------------------------------------------------------------------------

function resolvePythonCommand(): string | null {
  const candidates = process.platform === 'win32'
    ? ['python', 'py', 'python3']
    : ['python3', 'python']
  for (const cmd of candidates) {
    try {
      const result = spawnSync(cmd, ['--version'], { stdio: 'pipe' })
      if (result.status === 0) return cmd
    } catch { /* continue */ }
  }
  return null
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

function createWindow() {
  const iconPath = getIconPath()

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    title: 'Models at Home Studio',
    icon: iconPath ? nativeImage.createFromPath(iconPath) : undefined,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    backgroundColor: '#09090b',
    show: false,
  })

  mainWindow.once('ready-to-show', () => mainWindow?.show())

  if (VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL)
    mainWindow.webContents.openDevTools({ mode: 'detach' })
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  mainWindow.webContents.on('did-finish-load', () => {
    emitBootPhase(currentBootPhase)
  })

  mainWindow.on('closed', () => { mainWindow = null })
}

// ---------------------------------------------------------------------------
// Application menu
// ---------------------------------------------------------------------------

function buildAppMenu() {
  const isMac = process.platform === 'darwin'

  const template: Electron.MenuItemConstructorOptions[] = [
    ...(isMac
      ? [{
          label: app.name,
          submenu: [
            { role: 'about' as const },
            { type: 'separator' as const },
            { role: 'services' as const },
            { type: 'separator' as const },
            { role: 'hide' as const },
            { role: 'hideOthers' as const },
            { role: 'unhide' as const },
            { type: 'separator' as const },
            { role: 'quit' as const },
          ],
        }]
      : []),
    {
      label: 'File',
      submenu: [
        isMac ? { role: 'close' as const } : { role: 'quit' as const },
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Backend',
      submenu: [
        {
          label: 'Restart Backend',
          click: async () => {
            await stopBackend()
            await startBackend().catch((err) => console.error('Restart failed:', err))
          },
        },
        {
          label: 'Show Backend Logs',
          click: () => {
            const logPath = path.join(app.getPath('logs'), 'backend.log')
            shell.showItemInFolder(logPath)
          },
        },
      ],
    },
    {
      role: 'help',
      submenu: [
        {
          label: 'Documentation',
          click: () => shell.openExternal('https://github.com/models-at-home'),
        },
        {
          label: 'About',
          click: () => {
            dialog.showMessageBox(mainWindow!, {
              type: 'info',
              title: 'Models at Home Studio',
              message: `Models at Home Studio v${app.getVersion()}`,
              detail: 'Desktop studio for training ML models at home.',
              icon: getIconPath() ? nativeImage.createFromPath(getIconPath()!) : undefined,
            })
          },
        },
      ],
    },
  ]

  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

// ---------------------------------------------------------------------------
// Backend lifecycle — dispatcher based on config.backendMode
// ---------------------------------------------------------------------------

async function startBackend(): Promise<void> {
  if (booting || backendProcess || currentMode === 'docker') return
  booting = true
  backendStartError = null
  emitBootPhase({ phase: 'starting' })

  const config = loadConfig()
  const mode = config.backendMode

  try {
    if (mode === 'docker') {
      const ok = await startBackendDocker(config)
      if (!ok) throw new Error(backendStartError ?? 'Docker start failed')
      return
    }

    if (mode === 'auto') {
      emitBootPhase({ phase: 'checking-docker' })
      const dockerOk = dockerManager ? await dockerManager.isAvailable() : false
      if (dockerOk && dockerManager?.hasDockerfile()) {
        const ok = await startBackendDocker(config)
        if (ok) return
        console.warn('[backend] Docker attempt failed, falling back to native')
      } else {
        emitBootPhase({
          phase: 'docker-unavailable',
          detail: dockerOk ? 'No Dockerfile' : 'Docker daemon not reachable',
        })
      }
    }

    await startBackendNative()
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    backendStartError = msg
    emitBootPhase({ phase: 'failed', error: msg })
    throw err
  } finally {
    booting = false
  }
}

async function startBackendDocker(config: AppConfig): Promise<boolean> {
  if (!dockerManager) return false

  const ok = await dockerManager.isAvailable()
  if (!ok) {
    emitBootPhase({
      phase: 'docker-unavailable',
      detail: 'Docker daemon not reachable',
    })
    return false
  }

  if (!dockerManager.hasDockerfile()) {
    emitBootPhase({
      phase: 'docker-unavailable',
      detail: 'Dockerfile not found',
    })
    return false
  }

  const existing = await dockerManager.getStatus()

  if (!(await dockerManager.imageExists())) {
    emitBootPhase({ phase: 'docker-no-image' })
    if (!config.dockerAutoBuild) {
      backendStartError = 'Docker image missing and auto-build is disabled'
      return false
    }
    emitBootPhase({ phase: 'building-image' })
    const build = await dockerManager.buildImage((line) =>
      emitBootPhase({ phase: 'building-image', line }),
    )
    if (!build.success) {
      backendStartError = build.error ?? 'Image build failed'
      return false
    }
  }

  const hostPort = existing.hostPort ?? (await findAvailablePort(8000))
  const jupyterPort = await findAvailablePort(8888)
  emitBootPhase({ phase: 'starting-container' })
  const started = await dockerManager.startContainer({
    hostPort,
    jupyterPort,
    gpu: config.dockerGpu,
  })
  if (!started.success) {
    backendStartError = started.error ?? 'Container start failed'
    return false
  }

  backendPort = started.hostPort ?? hostPort
  const healthy = await dockerManager.waitForHealth(backendPort, {
    timeoutMs: 120_000,
    onTick: (attempt) =>
      emitBootPhase({ phase: 'waiting-container-health', attempt }),
  })
  if (!healthy) {
    backendStartError = 'Container did not become healthy in 2 minutes'
    return false
  }

  currentMode = 'docker'
  emitBootPhase({ phase: 'ready', mode: 'docker', port: backendPort })
  return true
}

function startBackendNative(): Promise<void> {
  return new Promise(async (resolve, reject) => {
    emitBootPhase({ phase: 'starting-python' })

    const pythonCmd = resolvePythonCommand()
    if (!pythonCmd) {
      const err = new Error('Python 3 not found in PATH')
      backendStartError = err.message
      reject(err)
      return
    }

    backendPort = await findAvailablePort(8000)
    const projectRoot = getResourcePath()

    console.log(`[backend] Starting native on port ${backendPort} with ${pythonCmd}`)

    backendProcess = spawn(
      pythonCmd,
      [
        '-m', 'uvicorn', 'backend.api:app',
        '--host', '127.0.0.1',
        '--port', String(backendPort),
        ...(isDev ? ['--reload'] : []),
      ],
      {
        cwd: projectRoot,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          PYTHONPATH: projectRoot,
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    )

    let started = false
    const logPath = path.join(app.getPath('logs'), 'backend.log')
    fs.mkdirSync(path.dirname(logPath), { recursive: true })
    const logStream = fs.createWriteStream(logPath, { flags: 'a' })
    logStream.write(`\n--- Session start ${new Date().toISOString()} ---\n`)

    const onData = (buf: Buffer) => {
      const line = buf.toString()
      logStream.write(line)
      process.stdout.write(`[backend] ${line}`)
      if (!started) {
        const trimmed = line.split('\n').find((l) => l.trim()) ?? ''
        if (trimmed) emitBootPhase({ phase: 'python-starting', line: trimmed })
      }
      if (!started && /Uvicorn running|Application startup complete/.test(line)) {
        started = true
        backendStartError = null
        currentMode = 'native'
        emitBootPhase({ phase: 'ready', mode: 'native', port: backendPort })
        resolve()
      }
    }

    backendProcess.stdout?.on('data', onData)
    backendProcess.stderr?.on('data', onData)

    backendProcess.on('error', (err) => {
      console.error('[backend] spawn error:', err.message)
      backendStartError = err.message
      backendProcess = null
      if (!started) reject(err)
    })

    backendProcess.on('exit', (code, signal) => {
      console.log(`[backend] Exited with code=${code} signal=${signal}`)
      logStream.write(`--- Exit code=${code} signal=${signal} ---\n`)
      logStream.end()
      backendProcess = null
      if (!started) {
        backendStartError = `Backend exited early (code ${code})`
        reject(new Error(backendStartError))
      }
    })

    const startupTimeout = setTimeout(() => {
      if (!started) {
        const err = new Error('Backend did not become ready in 60 seconds')
        backendStartError = err.message
        emitBootPhase({ phase: 'failed', error: err.message })
        try { backendProcess?.kill('SIGTERM') } catch { /* ignore */ }
        reject(err)
      }
    }, 60_000)

    backendProcess.once('exit', () => clearTimeout(startupTimeout))
  })
}

async function stopBackend() {
  if (currentMode === 'docker' && dockerManager) {
    try {
      await dockerManager.stopContainer()
    } catch (err) {
      console.error('[backend] docker stop error:', err)
    }
    currentMode = 'offline'
    return
  }

  if (!backendProcess) return
  console.log('[backend] Stopping native...')
  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(backendProcess.pid), '/f', '/t'])
    } else {
      backendProcess.kill('SIGTERM')
      setTimeout(() => backendProcess?.kill('SIGKILL'), 3000)
    }
  } catch (err) {
    console.error('[backend] Error stopping:', err)
  }
  backendProcess = null
  currentMode = 'offline'
}

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

const gotSingleInstanceLock = app.requestSingleInstanceLock()
if (!gotSingleInstanceLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
    }
  })
}

app.whenReady().then(async () => {
  buildAppMenu()
  dockerManager = new DockerManager(getResourcePath('models-at-home'))
  registerIpcHandlers()

  createWindow()

  startBackend().catch((err) => {
    console.error('[backend] Could not auto-start:', err)
  })

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', async (e) => {
  if (currentMode === 'docker' && dockerManager) {
    e.preventDefault()
    try { await dockerManager.stopContainer() } catch { /* ignore */ }
    currentMode = 'offline'
    app.quit()
    return
  }
  await stopBackend()
})

// ---------------------------------------------------------------------------
// IPC handlers
// ---------------------------------------------------------------------------

function registerIpcHandlers() {
  ipcMain.handle('backend:status', async () => {
    try {
      const res = await fetch(`http://127.0.0.1:${backendPort}/api/system/health`)
      return {
        running: res.ok,
        port: backendPort,
        error: backendStartError,
        mode: currentMode,
        phase: currentBootPhase,
      }
    } catch {
      return {
        running: false,
        port: backendPort,
        error: backendStartError,
        mode: currentMode,
        phase: currentBootPhase,
      }
    }
  })

  ipcMain.handle('backend:restart', async () => {
    await stopBackend()
    try {
      await startBackend()
      return { success: true, port: backendPort, mode: currentMode }
    } catch (err) {
      return { success: false, error: String(err), port: backendPort, mode: currentMode }
    }
  })

  ipcMain.handle('backend:port', () => backendPort)

  ipcMain.handle('backend:boot-phase', () => currentBootPhase)

  ipcMain.handle('backend:open-logs', () => {
    const logPath = path.join(app.getPath('logs'), 'backend.log')
    shell.showItemInFolder(logPath)
  })

  // Config
  ipcMain.handle('config:get', () => loadConfig())
  ipcMain.handle('config:set', (_e, patch: Partial<AppConfig>) => saveConfig(patch))
  ipcMain.handle('config:set-backend-mode', async (_e, mode: BackendMode) => {
    const next = saveConfig({ backendMode: mode })
    await stopBackend()
    startBackend().catch((err) => console.error('[backend] Restart after mode change failed:', err))
    return next
  })

  // Docker
  ipcMain.handle('docker:status', async () => {
    return dockerManager?.getStatus() ?? { running: false }
  })
  ipcMain.handle('docker:available', async () =>
    dockerManager ? dockerManager.isAvailable() : false,
  )
  ipcMain.handle('docker:image-exists', async () =>
    dockerManager ? dockerManager.imageExists() : false,
  )
  ipcMain.handle('docker:start', async () => {
    if (!dockerManager) return { success: false, error: 'Docker not initialized' }
    const port = await findAvailablePort(8000)
    return dockerManager.startContainer({ hostPort: port, gpu: loadConfig().dockerGpu })
  })
  ipcMain.handle('docker:stop', async () => dockerManager?.stopContainer())
  ipcMain.handle('docker:build', async () => {
    if (!dockerManager) return { success: false, error: 'Docker not initialized' }
    return dockerManager.buildImage((line) =>
      mainWindow?.webContents.send('docker:build-progress', line),
    )
  })
  ipcMain.handle('docker:logs', async (_event, tail?: number) =>
    dockerManager?.getContainerLogs(tail),
  )

  ipcMain.handle('app:version', () => app.getVersion())
  ipcMain.handle('app:platform', () => process.platform)
  ipcMain.handle('app:paths', () => ({
    userData: app.getPath('userData'),
    logs: app.getPath('logs'),
    temp: app.getPath('temp'),
  }))

  ipcMain.handle('shell:openExternal', (_event, url: string) => shell.openExternal(url))
  ipcMain.handle('shell:showInFolder', (_event, p: string) => shell.showItemInFolder(p))

  ipcMain.handle('dialog:pickDirectory', async () => {
    if (!mainWindow) return { canceled: true, filePaths: [] }
    return dialog.showOpenDialog(mainWindow, {
      properties: ['openDirectory'],
    })
  })

  ipcMain.handle('dialog:pickFile', async (_event, filters?: Electron.FileFilter[]) => {
    if (!mainWindow) return { canceled: true, filePaths: [] }
    return dialog.showOpenDialog(mainWindow, {
      properties: ['openFile'],
      filters,
    })
  })
}
