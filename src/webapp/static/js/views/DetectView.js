// Detect View — Premium B&W Minimal
const { createApp, ref, computed, onMounted } = Vue

export default {
  template: `
    <div class="detect-root">

      <!-- Header Bar -->
      <div class="detect-header">
        <div class="header-left">
          <div class="page-eyebrow">检测</div>
          <h1 class="page-title">缺陷识别</h1>
        </div>
        <div class="header-right">
          <div class="mode-toggle">
            <button
              v-for="m in modes"
              :key="m.id"
              @click="settings.mode = m.id"
              :class="['mode-btn', settings.mode === m.id ? 'mode-btn--active' : '']"
            >
              <span class="mode-icon">{{ m.icon }}</span>
              {{ m.label }}
            </button>
          </div>
        </div>
      </div>

      <div class="detect-layout">
        <!-- LEFT: Config Panel -->
        <div class="panel panel--config">
          <!-- Step indicator -->
          <div class="step-row">
            <div class="step-num">1</div>
            <span class="step-label">配置与上传</span>
          </div>

          <!-- Model selector -->
          <div class="config-section">
            <label class="config-label">模型</label>
            <select v-model="settings.modelId" class="config-select">
              <option :value="null">默认模型</option>
              <option v-for="m in modelList" :key="m.id" :value="m.id">{{ m.name }}</option>
            </select>
            <div v-if="selectedModelPath" class="model-path">{{ selectedModelPath }}</div>
          </div>

          <!-- Upload zone -->
          <div class="config-section">
            <label class="config-label">{{ settings.mode === 'single' ? '图片' : '文件夹' }}</label>
            <div
              v-if="settings.mode === 'single'"
              @dragover.prevent="isDragOver = true"
              @dragleave="isDragOver = false"
              @drop.prevent="onDrop"
              @click="triggerFileInput"
              :class="['drop-zone', isDragOver ? 'drop-zone--over' : '', preview ? 'drop-zone--has-file' : '']"
            >
              <input ref="fileInput" type="file" accept="image/*" class="hidden" @change="onFileChange" @click.stop />
              <div v-if="preview" class="preview-wrap">
                <img :src="preview" class="preview-img" />
                <div class="preview-meta">
                  <span class="preview-name">{{ fileName }}</span>
                  <span class="preview-replace">点击更换</span>
                </div>
              </div>
              <div v-else class="drop-prompt">
                <svg class="drop-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span class="drop-text">拖入图片 或 点击选择</span>
              </div>
            </div>

            <div
              v-else
              @click="triggerBatchInput"
              :class="['drop-zone', 'drop-zone--batch', batchFiles.length ? 'drop-zone--has-file' : '']"
            >
              <input ref="batchInput" type="file" accept="image/*" webkitdirectory directory multiple class="hidden" @change="onBatchChange" @click.stop />
              <div v-if="batchFiles.length" class="batch-info">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" stroke-linecap="round" stroke-linejoin="round"/></svg>
                <span>{{ batchFiles.length }} 个文件已选择</span>
              </div>
              <div v-else class="drop-prompt">
                <svg class="drop-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" stroke-linecap="round" stroke-linejoin="round"/></svg>
                <span class="drop-text">点击选择文件夹</span>
              </div>
            </div>
          </div>

          <!-- Presets -->
          <div class="config-section">
            <label class="config-label">阈值预设</label>
            <div class="preset-row">
              <button
                v-for="p in presets"
                :key="p.id"
                @click="applyPreset(p.id)"
                :class="['preset-btn', settings.preset === p.id ? 'preset-btn--active' : '']"
              >
                {{ p.label }}
              </button>
            </div>
          </div>

          <!-- Sliders -->
          <div class="config-section">
            <div class="slider-row">
              <div class="slider-header">
                <span class="slider-label">置信度</span>
                <span class="slider-val">{{ settings.conf.toFixed(2) }}</span>
              </div>
              <input type="range" v-model.number="settings.conf" min="0.01" max="0.99" step="0.01" class="slider" />
            </div>
            <div class="slider-row">
              <div class="slider-header">
                <span class="slider-label">IoU</span>
                <span class="slider-val">{{ settings.iou.toFixed(2) }}</span>
              </div>
              <input type="range" v-model.number="settings.iou" min="0.1" max="0.95" step="0.01" class="slider" />
            </div>
          </div>

          <!-- Run button -->
          <button
            @click="runDetect"
            :disabled="isRunning || !canRun"
            :class="['run-btn', isRunning || !canRun ? 'run-btn--disabled' : '']"
          >
            <svg v-if="isRunning" class="spin-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
            {{ isRunning ? (settings.mode === 'single' ? '检测中...' : '处理中...') : '开始检测' }}
          </button>
        </div>

        <!-- RIGHT: Result Panel -->
        <div class="panel panel--result">
          <div class="step-row">
            <div class="step-num step-num--green">2</div>
            <span class="step-label">检测结果</span>
          </div>

          <!-- Empty state -->
          <div v-if="!result && !isRunning" class="result-empty">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <p class="empty-text">上传图片开始检测</p>
          </div>

          <!-- Loading skeleton -->
          <div v-else-if="isRunning && settings.mode === 'single'" class="skeleton-wrap">
            <div class="skeleton-img"></div>
            <div class="skeleton-line w-3/4"></div>
          </div>

          <!-- Result -->
          <div v-else-if="result" class="result-content">

            <!-- Single Result -->
            <div v-if="settings.mode === 'single'" class="single-result">
              <div class="annotated-wrap">
                <img :src="'data:image/png;base64,' + result.annotated_image_base64" class="annotated-img" />
                <div :class="['badge', result.num_detections > 0 ? 'badge--active' : 'badge--empty']">
                  {{ result.num_detections > 0 ? result.num_detections + ' 个缺陷' : '无缺陷' }}
                </div>
              </div>

              <div class="result-meta-grid">
                <div class="meta-card">
                  <div class="meta-card-label">图片尺寸</div>
                  <div class="meta-card-val">{{ result.image_size?.width }} × {{ result.image_size?.height }}</div>
                </div>
                <div class="meta-card">
                  <div class="meta-card-label">检测数量</div>
                  <div :class="['meta-card-val', result.num_detections > 0 ? 'text-active' : '']">{{ result.num_detections }}</div>
                </div>
              </div>

              <div v-if="result.detections?.length > 0" class="detections-list">
                <div v-for="(d, i) in result.detections" :key="i" class="detection-item">
                  <div class="detection-dot" :style="{ background: getColor(d.class_id) }"></div>
                  <div class="detection-info">
                    <div class="detection-name">{{ d.class_name }}</div>
                    <div class="detection-bbox">[{{ d.bbox_xyxy.map(v => v.toFixed(0)).join(', ') }}]</div>
                  </div>
                  <div class="detection-conf">{{ (d.confidence * 100).toFixed(1) }}%</div>
                </div>
              </div>
            </div>

            <!-- Batch Result -->
            <div v-else class="batch-result">
              <div class="batch-stats">
                <div class="batch-stat">
                  <div class="batch-stat-val">{{ result.total_files }}</div>
                  <div class="batch-stat-label">总文件</div>
                </div>
                <div class="batch-stat">
                  <div class="batch-stat-val text-green">{{ result.success_count }}</div>
                  <div class="batch-stat-label">成功</div>
                </div>
                <div class="batch-stat">
                  <div :class="['batch-stat-val', result.failure_count > 0 ? 'text-red' : '']">{{ result.failure_count }}</div>
                  <div class="batch-stat-label">失败</div>
                </div>
              </div>
              <div class="batch-outdir">输出目录: <code>{{ result.output_dir }}</code></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const modes = [
      { id: 'single', label: '单图检测', icon: '◻' },
      { id: 'batch', label: '批量检测', icon: '▤' },
    ]
    const presets = [
      { id: 'balanced', label: '平衡', conf: 0.25, iou: 0.6 },
      { id: 'high_recall', label: '高召回', conf: 0.1, iou: 0.7 },
      { id: 'high_precision', label: '高精度', conf: 0.5, iou: 0.5 },
    ]
    const presetValues = { balanced: { conf: 0.25, iou: 0.6 }, high_recall: { conf: 0.1, iou: 0.7 }, high_precision: { conf: 0.5, iou: 0.5 } }

    const settings = ref({ mode: 'single', modelId: null, conf: 0.25, iou: 0.6, preset: 'balanced' })
    const isRunning = ref(false)
    const isDragOver = ref(false)
    const result = ref(null)
    const modelList = ref([])
    const preview = ref('')
    const fileName = ref('')
    const currentFile = ref(null)
    const batchFiles = ref([])
    const fileInput = ref(null)
    const batchInput = ref(null)

    const classColors = ['#ef4444','#f97316','#eab308','#22c55e','#14b8a6','#3b82f6','#8b5cf6','#ec4899','#f43f5e','#6366f1']
    const getColor = (id) => classColors[id % classColors.length]

    const canRun = computed(() => settings.value.mode === 'single' ? !!currentFile.value : batchFiles.value.length > 0)

    const selectedModelPath = computed(() => {
      const m = modelList.value.find(m => m.id === settings.value.modelId)
      return m?.path || ''
    })

    const applyPreset = (id) => {
      settings.value.preset = id
      const v = presetValues[id]
      if (v) { settings.value.conf = v.conf; settings.value.iou = v.iou }
    }

    const triggerFileInput = () => fileInput.value?.click()
    const triggerBatchInput = () => batchInput.value?.click()

    const onFileChange = (e) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    }
    const onDrop = (e) => {
      isDragOver.value = false
      const file = e.dataTransfer.files?.[0]
      if (file?.type.startsWith('image/')) handleFile(file)
    }
    const handleFile = (file) => {
      fileName.value = file.name
      currentFile.value = file
      const reader = new FileReader()
      reader.onload = e => { preview.value = e.target.result }
      reader.readAsDataURL(file)
    }

    const onBatchChange = (e) => {
      batchFiles.value = Array.from(e.target.files || [])
    }

    const showToast = (message, type = 'success') => {
      // simple alert for now
      if (type === 'error') alert(message)
    }

    const loadModels = async () => {
      try {
        const api = (await import('../api/client.js')).default
        const res = await api.get('/models')
        modelList.value = res.data.models
      } catch {}
    }

    const runDetect = async () => {
      if (!canRun.value) { showToast(settings.value.mode === 'single' ? '请先选择图片' : '请先选择文件', 'error'); return }
      isRunning.value = true
      const query = new URLSearchParams({
        conf: settings.value.conf.toString(), iou: settings.value.iou.toString(),
        imgsz: '640', max_det: '300', model_id: settings.value.modelId ?? ''
      })
      try {
        const api = (await import('../api/client.js')).default
        if (settings.value.mode === 'single') {
          const formData = new FormData()
          formData.append('file', currentFile.value)
          const res = await api.post(`/detections/predict?${query.toString()}`, formData)
          result.value = res.data
          showToast('检测完成')
        } else {
          const formData = new FormData()
          batchFiles.value.forEach(f => formData.append('files', f, f.name))
          const res = await api.post(`/detections/batch?${query.toString()}`, formData)
          result.value = res.data
          showToast(`完成：${res.data.success_count}/${res.data.total_files} 个文件`)
        }
      } catch (e) {
        showToast(e.response?.data?.detail || '检测失败', 'error')
      } finally {
        isRunning.value = false
      }
    }

    onMounted(() => { loadModels() })

    return { modes, presets, settings, isRunning, isDragOver, result, modelList, preview, fileName, currentFile, batchFiles, canRun, fileInput, batchInput, getColor, applyPreset, triggerFileInput, triggerBatchInput, onFileChange, onDrop, onBatchChange, runDetect, selectedModelPath }
  }
}
