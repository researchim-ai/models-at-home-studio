import { useState, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import {
  Play,
  Square,
  ImageIcon,
  Download,
  Send,
  Trash2,
  CheckCircle2,
  Loader2,
} from 'lucide-react'
import { useVLMRuns, useModels, useDatasets, useGpuInfo } from '@/api/hooks'
import { api } from '@/api/client'
import type { TrainingRun } from '@/api/types'
import { toast } from 'sonner'

function VLMConfigSidebar({
  stage, setStage,
  model, setModel,
  datasetPath, setDatasetPath,
  batchSize, setBatchSize,
  lr, setLr,
  maxSteps, setMaxSteps,
  useLora, setUseLora,
}: {
  stage: string; setStage: (v: string) => void
  model: string; setModel: (v: string) => void
  datasetPath: string; setDatasetPath: (v: string) => void
  batchSize: number; setBatchSize: (v: number) => void
  lr: number; setLr: (v: number) => void
  maxSteps: number; setMaxSteps: (v: number) => void
  useLora: boolean; setUseLora: (v: boolean) => void
}) {
  const { t } = useTranslation()
  const { data: gpuData } = useGpuInfo()
  const { data: modelsData } = useModels()
  const { data: datasetsData } = useDatasets()

  return (
    <ScrollArea className="w-80 shrink-0 border-r border-border p-4 space-y-6 h-full">
      <div className="space-y-6">
        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            VLM Stage
          </Label>
          <Select
            options={[
              { value: 'vlm_pretrain', label: t('vlm.stage.pretrain') },
              { value: 'vlm_sft', label: t('vlm.stage.sft') },
              { value: 'vlm_grpo', label: t('vlm.stage.grpo') },
            ]}
            value={stage}
            onChange={(e) => setStage(e.target.value)}
          />
        </div>

        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('sidebar.base_model')}
          </Label>
          <Select
            options={[
              { value: '', label: '-- HF repo or local --' },
              ...(modelsData?.models ?? []).map((m) => ({ value: m.path, label: m.name })),
            ]}
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />
          <Input
            placeholder="e.g. Qwen/Qwen2-VL-2B-Instruct"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />
        </div>

        <div className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Dataset
          </Label>
          <Select
            options={[
              { value: '', label: '-- select dataset or type path --' },
              ...(datasetsData?.datasets ?? []).map((d) => ({ value: d.path, label: d.name })),
            ]}
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
          />
          <Input
            placeholder="datasets/my_vlm_dataset.jsonl"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
          />
          {!datasetPath && (
            <p className="text-[11px] text-amber-500">
              Для VLM-тренировки нужен dataset_path
            </p>
          )}
        </div>

        <div className="space-y-3">
          <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t('sidebar.hyperparams')}
          </Label>
          <Slider
            label={t('training.batch_size')}
            value={batchSize}
            onValueChange={setBatchSize}
            min={1} max={16} step={1}
          />
          <div className="space-y-1">
            <Label className="text-xs">{t('training.learning_rate')}</Label>
            <Input
              type="number"
              step="0.000001"
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
            />
          </div>
          <Slider
            label={t('training.max_steps')}
            value={maxSteps}
            onValueChange={setMaxSteps}
            min={50} max={10000} step={50}
          />
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={useLora}
              onChange={(e) => setUseLora(e.target.checked)}
              className="rounded border-border"
            />
            <span>Use LoRA</span>
          </label>
        </div>

        {gpuData?.gpus?.map((gpu) => (
          <div key={gpu.id} className="rounded-lg bg-muted/50 p-2.5 space-y-1">
            <div className="flex justify-between text-xs">
              <span>GPU {gpu.id}</span>
              <span>{gpu.memory_used_gb.toFixed(1)} / {gpu.memory_total_gb.toFixed(1)} GB</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
              <div className="h-full bg-primary" style={{ width: `${gpu.memory_percent}%` }} />
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  )
}

function VLMLaunchTab({ config }: { config: Record<string, unknown> }) {
  const { t } = useTranslation()
  const [running, setRunning] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)

  const handleStart = async () => {
    if (!config.model_path || !config.dataset_path) {
      toast.error('Выбери модель и dataset_path')
      return
    }
    try {
      setRunning(true)
      const { run_id } = await api.startVLMTraining(config)
      setRunId(run_id)
      toast.success(`Training started: ${run_id}`)
    } catch (e) {
      toast.error(`Failed to start: ${e}`)
      setRunning(false)
    }
  }

  const handleStop = async () => {
    if (!runId) return
    try {
      await api.stopVLMTraining(runId)
      toast.success('Training stopped')
      setRunning(false)
      setRunId(null)
    } catch (e) {
      toast.error(`Failed to stop: ${e}`)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>{t('vlm.title')}</CardTitle>
          <CardDescription>{t('vlm.subtitle')}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            {!running ? (
              <Button onClick={handleStart}>
                <Play className="h-4 w-4" />
                {t('button.start_training')}
              </Button>
            ) : (
              <Button variant="destructive" onClick={handleStop}>
                <Square className="h-4 w-4" />
                {t('button.stop_training')}
              </Button>
            )}
            {runId && (
              <Badge variant="secondary" className="font-mono">{runId}</Badge>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Config Preview</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="text-xs font-mono bg-muted/30 rounded-lg p-3 overflow-auto max-h-64">
            {JSON.stringify(config, null, 2)}
          </pre>
        </CardContent>
      </Card>
    </div>
  )
}

function VLMInferenceTab() {
  const { data: modelsData } = useModels()
  const [modelPath, setModelPath] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [prompt, setPrompt] = useState('Describe this image.')
  const [imageData, setImageData] = useState<string | null>(null)
  const [response, setResponse] = useState('')
  const [generating, setGenerating] = useState(false)
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(256)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleLoad = async () => {
    if (!modelPath) { toast.error('Select a model'); return }
    setLoading(true)
    try {
      await api.loadVLM(modelPath)
      setLoaded(true)
      toast.success(`Model loaded: ${modelPath}`)
    } catch (e) {
      toast.error(`Failed to load: ${e}`)
    } finally {
      setLoading(false)
    }
  }

  const handleUnload = async () => {
    try {
      await api.unloadVLM()
      setLoaded(false)
      toast.success('Model unloaded')
    } catch { /* ignore */ }
  }

  const handleImageUpload = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const result = e.target?.result as string
      setImageData(result)
    }
    reader.readAsDataURL(file)
  }

  const handleGenerate = async () => {
    if (!imageData || !prompt) { toast.error('Image and prompt required'); return }
    setGenerating(true)
    setResponse('')
    try {
      const { text } = await api.generateVLM({
        prompt,
        image_base64: imageData,
        max_tokens: maxTokens,
        temperature,
      })
      setResponse(text)
    } catch (e) {
      toast.error(`Generation failed: ${e}`)
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            Model Loading
            {loaded && <CheckCircle2 className="h-4 w-4 text-green-500" />}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <div className="flex-1 space-y-1">
              <Select
                options={[
                  { value: '', label: '-- local model --' },
                  ...(modelsData?.models ?? []).map((m) => ({ value: m.path, label: m.name })),
                ]}
                value={modelPath}
                onChange={(e) => setModelPath(e.target.value)}
              />
              <Input
                placeholder="or HF repo_id (e.g. Qwen/Qwen2-VL-2B-Instruct)"
                value={modelPath}
                onChange={(e) => setModelPath(e.target.value)}
              />
            </div>
            {!loaded ? (
              <Button onClick={handleLoad} disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                Load
              </Button>
            ) : (
              <Button variant="outline" onClick={handleUnload}>Unload</Button>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Image</CardTitle>
          </CardHeader>
          <CardContent
            onDragOver={(e) => { e.preventDefault() }}
            onDrop={(e) => {
              e.preventDefault()
              const file = e.dataTransfer.files[0]
              if (file && file.type.startsWith('image/')) handleImageUpload(file)
            }}
          >
            {imageData ? (
              <div className="space-y-2">
                <img src={imageData} alt="Input" className="w-full rounded-lg max-h-96 object-contain bg-muted/30" />
                <Button variant="outline" size="sm" onClick={() => setImageData(null)}>
                  <Trash2 className="h-3.5 w-3.5" />
                  Clear
                </Button>
              </div>
            ) : (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="flex flex-col items-center justify-center h-64 border-2 border-dashed border-border rounded-lg cursor-pointer hover:bg-accent/20 transition-colors"
              >
                <ImageIcon className="h-12 w-12 text-muted-foreground/40 mb-2" />
                <p className="text-sm text-muted-foreground">Click or drag to upload image</p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) handleImageUpload(file)
              }}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Prompt & Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1">
              <Label className="text-xs">Prompt</Label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={4}
                className="w-full rounded-md border border-border bg-transparent px-3 py-2 text-sm resize-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            </div>
            <Slider
              label="Temperature"
              value={temperature}
              onValueChange={setTemperature}
              min={0} max={2} step={0.1}
            />
            <Slider
              label="Max tokens"
              value={maxTokens}
              onValueChange={setMaxTokens}
              min={16} max={2048} step={16}
            />
            <Button
              onClick={handleGenerate}
              disabled={!loaded || !imageData || generating}
              className="w-full"
            >
              {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              Generate
            </Button>
          </CardContent>
        </Card>
      </div>

      {(response || generating) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Response</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-lg bg-muted/30 p-4 text-sm whitespace-pre-wrap">
              {generating && !response ? (
                <span className="text-muted-foreground italic">Generating...</span>
              ) : (
                response
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function VLMHistoryTab() {
  const { t } = useTranslation()
  const { data, isLoading } = useVLMRuns()
  const runs = data?.runs ?? []

  const statusColor = (s: string): 'success' | 'default' | 'destructive' | 'secondary' => {
    switch (s) {
      case 'running': return 'success'
      case 'completed': return 'default'
      case 'error': return 'destructive'
      default: return 'secondary'
    }
  }

  return (
    <div className="space-y-3">
      {isLoading && <div className="text-muted-foreground text-sm">{t('common.loading')}</div>}
      {runs.length === 0 && !isLoading && (
        <div className="text-muted-foreground text-sm py-8 text-center">No VLM runs yet</div>
      )}
      {runs.map((run: TrainingRun) => (
        <Card key={run.run_id}>
          <CardContent className="p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Badge variant={statusColor(run.status)}>{run.status}</Badge>
              <span className="font-mono text-sm">{run.experiment_name || run.run_id.slice(0, 12)}</span>
              <span className="text-xs text-muted-foreground">{run.stage}</span>
            </div>
            <span className="text-xs text-muted-foreground">
              {run.current_step}/{run.total_steps}
            </span>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

export function VLMStudio() {
  const { t } = useTranslation()
  const [activeTab, setActiveTab] = useState('launch')

  const [stage, setStage] = useState('vlm_sft')
  const [model, setModel] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [batchSize, setBatchSize] = useState(2)
  const [lr, setLr] = useState(2e-5)
  const [maxSteps, setMaxSteps] = useState(500)
  const [useLora, setUseLora] = useState(true)

  const config = {
    stage,
    model_path: model,
    dataset_path: datasetPath,
    batch_size: batchSize,
    learning_rate: lr,
    max_steps: maxSteps,
    use_lora: useLora,
    experiment_name: `vlm_${stage}_${Date.now().toString(36).slice(-4)}`,
  }

  return (
    <div className="flex h-full">
      <VLMConfigSidebar
        stage={stage} setStage={setStage}
        model={model} setModel={setModel}
          datasetPath={datasetPath} setDatasetPath={setDatasetPath}
        batchSize={batchSize} setBatchSize={setBatchSize}
        lr={lr} setLr={setLr}
        maxSteps={maxSteps} setMaxSteps={setMaxSteps}
        useLora={useLora} setUseLora={setUseLora}
      />
      <div className="flex-1 p-6 overflow-auto">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="launch">{t('tabs.launch')}</TabsTrigger>
            <TabsTrigger value="inference">Inference</TabsTrigger>
            <TabsTrigger value="history">{t('tabs.history')}</TabsTrigger>
          </TabsList>

          <TabsContent value="launch"><VLMLaunchTab config={config} /></TabsContent>
          <TabsContent value="inference"><VLMInferenceTab /></TabsContent>
          <TabsContent value="history"><VLMHistoryTab /></TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
