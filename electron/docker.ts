import Docker from 'dockerode'
import fs from 'fs'
import path from 'path'

const BASE_IMAGE_NAME = 'models-at-home-base:latest'
const IMAGE_NAME = 'models-at-home-studio:latest'
const CONTAINER_NAME = 'mah-studio-backend'
const CONTAINER_PORT = 8000

export type BuildProgress = (line: string) => void

export interface DockerStatus {
  running: boolean
  containerId?: string
  hostPort?: number
  error?: string
}

export class DockerManager {
  private docker: Docker
  private containerId: string | null = null
  private hostPort: number | null = null
  private projectRoot: string
  private studioRoot: string

  constructor(projectRoot?: string) {
    this.docker = new Docker()
    this.projectRoot =
      projectRoot ?? path.resolve(__dirname, '..', '..', 'models-at-home')
    this.studioRoot = path.dirname(this.projectRoot)
  }

  // -----------------------------------------------------------------------
  // Availability
  // -----------------------------------------------------------------------

  async isAvailable(): Promise<boolean> {
    try {
      await this.docker.ping()
      return true
    } catch {
      return false
    }
  }

  hasDockerfile(): boolean {
    return (
      fs.existsSync(path.join(this.projectRoot, 'Dockerfile')) &&
      fs.existsSync(path.join(this.studioRoot, 'backend', 'Dockerfile'))
    )
  }

  async imageExists(): Promise<boolean> {
    try {
      const images = await this.docker.listImages({
        filters: { reference: [IMAGE_NAME] },
      })
      return images.length > 0
    } catch {
      return false
    }
  }

  // -----------------------------------------------------------------------
  // Status
  // -----------------------------------------------------------------------

  async getStatus(): Promise<DockerStatus> {
    try {
      const containers = await this.docker.listContainers({
        all: true,
        filters: { name: [CONTAINER_NAME] },
      })

      if (containers.length > 0) {
        const c = containers[0]
        this.containerId = c.Id
        const mapped = c.Ports?.find((p) => p.PrivatePort === CONTAINER_PORT)
        const port = mapped?.PublicPort ?? this.hostPort ?? undefined
        if (port) this.hostPort = port
        return {
          running: c.State === 'running',
          containerId: c.Id,
          hostPort: port,
        }
      }
      return { running: false }
    } catch (error) {
      return { running: false, error: String(error) }
    }
  }

  // -----------------------------------------------------------------------
  // Build
  // -----------------------------------------------------------------------

  async buildImage(
    onProgress?: BuildProgress,
  ): Promise<{ success: boolean; error?: string }> {
    if (!this.hasDockerfile()) {
      return {
        success: false,
        error:
          `Dockerfile not found. Expected ${this.projectRoot}/Dockerfile ` +
          `and ${this.studioRoot}/backend/Dockerfile`,
      }
    }

    try {
      await this.buildTaggedImage(
        { context: this.projectRoot, src: ['.'] },
        {
          t: BASE_IMAGE_NAME,
          dockerfile: 'Dockerfile',
          buildargs: { LLAMA_CPP_BACKEND: 'vulkan' },
        },
        onProgress,
        'base',
      )

      await this.buildTaggedImage(
        { context: this.studioRoot, src: ['backend'] },
        {
          t: IMAGE_NAME,
          dockerfile: 'backend/Dockerfile',
        },
        onProgress,
        'backend',
      )

      return { success: true }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  }

  private async buildTaggedImage(
    context: { context: string; src: string[] },
    opts: Record<string, unknown>,
    onProgress?: BuildProgress,
    label = 'image',
  ): Promise<void> {
    onProgress?.(`[${label}] Building ${String(opts.t ?? 'image')}`)
    const stream = await this.docker.buildImage(
      context,
      opts,
    )

    await new Promise<void>((resolve, reject) => {
      this.docker.modem.followProgress(
        stream,
        (err: Error | null) => (err ? reject(err) : resolve()),
        (event: Record<string, unknown>) => {
          if (!onProgress) return
          const stream = event.stream as string | undefined
          const status = event.status as string | undefined
          const progress = event.progress as string | undefined
          const errorDetail = event.errorDetail as { message?: string } | undefined
          const text =
            errorDetail?.message ||
            (stream ? stream.trimEnd() : '') ||
            (status ? `${status}${progress ? ' ' + progress : ''}` : '')
          if (text) onProgress(`[${label}] ${text}`)
        },
      )
    })
  }

  // -----------------------------------------------------------------------
  // Run
  // -----------------------------------------------------------------------

  async ensureContainerRemoved(): Promise<void> {
    try {
      const containers = await this.docker.listContainers({
        all: true,
        filters: { name: [CONTAINER_NAME] },
      })
      for (const c of containers) {
        const container = this.docker.getContainer(c.Id)
        try {
          if (c.State === 'running') await container.stop({ t: 5 })
        } catch { /* ignore */ }
        try {
          await container.remove({ force: true })
        } catch { /* ignore */ }
      }
    } catch { /* ignore */ }
  }

  async startContainer(opts: {
    hostPort: number
    jupyterPort?: number
    gpu?: boolean
  }): Promise<{ success: boolean; hostPort?: number; error?: string }> {
    try {
      await this.ensureContainerRemoved()
      for (const dir of ['datasets', 'out', '.runs', 'blueprints', 'models', 'configs', 'notebooks']) {
        fs.mkdirSync(path.join(this.projectRoot, dir), { recursive: true })
      }

      const deviceRequests = opts.gpu
        ? [{ Count: -1, Capabilities: [['gpu']] }]
        : undefined

      const container = await this.docker.createContainer({
        Image: IMAGE_NAME,
        name: CONTAINER_NAME,
        Cmd: [
          'python',
          '-m',
          'uvicorn',
          'backend.api:app',
          '--host',
          '0.0.0.0',
          '--port',
          String(CONTAINER_PORT),
        ],
        ExposedPorts: { [`${CONTAINER_PORT}/tcp`]: {}, '8888/tcp': {} },
        HostConfig: {
          PortBindings: {
            [`${CONTAINER_PORT}/tcp`]: [{ HostPort: String(opts.hostPort) }],
            '8888/tcp': [{ HostPort: String(opts.jupyterPort ?? 8888) }],
          },
          Binds: [
            // Studio FastAPI package is NOT baked into the image (it is built
            // from the models-at-home context); mount it like docker-compose does.
            `${this.studioRoot}/backend:/app/backend`,
            `${this.projectRoot}/datasets:/app/datasets`,
            `${this.projectRoot}/out:/app/out`,
            `${this.projectRoot}/.runs:/app/.runs`,
            `${this.projectRoot}/blueprints:/app/blueprints`,
            `${this.projectRoot}/models:/app/models`,
            `${this.projectRoot}/configs:/app/configs`,
            `${this.projectRoot}/notebooks:/app/notebooks`,
          ],
          ShmSize: 8 * 1024 * 1024 * 1024,
          DeviceRequests: deviceRequests,
          RestartPolicy: { Name: 'unless-stopped' },
        },
        Env: [
          'PYTHONUNBUFFERED=1',
          'MODELS_AT_HOME_ROOT=/app',
          'MKL_THREADING_LAYER=GNU',
          'HOME=/tmp',
          'HF_HOME=/tmp/.cache/huggingface',
          'NVIDIA_VISIBLE_DEVICES=all',
          'NVIDIA_DRIVER_CAPABILITIES=all',
          `JUPYTER_PUBLIC_URL=http://127.0.0.1:${opts.jupyterPort ?? 8888}`,
        ],
      })

      await container.start()
      this.containerId = container.id
      this.hostPort = opts.hostPort
      return { success: true, hostPort: opts.hostPort }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  }

  async stopContainer(): Promise<{ success: boolean; error?: string }> {
    try {
      const status = await this.getStatus()
      if (!status.running || !status.containerId) return { success: true }

      const container = this.docker.getContainer(status.containerId)
      await container.stop({ t: 10 })
      return { success: true }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  }

  async waitForHealth(
    hostPort: number,
    opts: { timeoutMs?: number; onTick?: (attempt: number) => void } = {},
  ): Promise<boolean> {
    const timeoutMs = opts.timeoutMs ?? 120_000
    const start = Date.now()
    let attempt = 0
    while (Date.now() - start < timeoutMs) {
      attempt += 1
      opts.onTick?.(attempt)
      try {
        const res = await fetch(
          `http://127.0.0.1:${hostPort}/api/system/health`,
        )
        if (res.ok) return true
      } catch { /* not ready */ }
      await new Promise((r) => setTimeout(r, 1000))
    }
    return false
  }

  async getContainerLogs(tail = 200): Promise<string> {
    try {
      const status = await this.getStatus()
      if (!status.containerId) return ''

      const container = this.docker.getContainer(status.containerId)
      const logs = await container.logs({
        stdout: true,
        stderr: true,
        tail,
        timestamps: true,
      })
      return logs.toString()
    } catch {
      return ''
    }
  }
}
