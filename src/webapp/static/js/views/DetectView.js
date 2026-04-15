// Detect View - main detection interface
const { createApp, ref, computed, onMounted } = Vue

export default {
  template: `
    <div class="space-y-6">
      <!-- Mode Tabs -->
      <div class="flex gap-1 bg-surface-800 p-1 rounded-xl w-fit">
        <button v-for="m in modes" :key="m.id" @click="settings.mode = m.id"
          :class="['flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all', settings.mode === m.id ? 'bg-accent-600 text-white shadow' : 'text-surface-400 hover:text-white']">
          <span>{{ m.icon }}</span>{{ m.label }}
        </button>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- LEFT: Config -->
        <div class="bg-surface-800 rounded-2xl border border-surface-700 p-5 space-y-5">
          <h2 class="text-lg font-semibold text-white flex items-center gap-2">
            <span class="w-6 h-6 rounded bg-accent-600 flex items-center justify-center text-xs">1</span>
            上传与设置
          </h2>

          <!-- Model -->
          <div>
            <label class="block text-sm font-medium text-surface-300 mb-2">模型选择</label>
            <select v-model="settings.modelId" class="w-full bg-surface-700 border border-surface-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500">
              <option :value="null">默认模型</option>
              <option v-for="m in modelList" :key="m.id" :value="m.id">{{ m.name }}</option>
            </select>
          </div>

          <!-- Upload -->
          <div>
            <label class="block text-sm font-medium text-surface-300 mb-2">
              {{ settings.mode === 'single' ? '上传图片' : '上传文件夹' }}
            </label>
            <div v-if="settings.mode === 'single'"
              @dragover.prevent="isDragOver = true" @dragleave="isDragOver = false" @drop.prevent="onDrop"
              @click="triggerFileInput"
              :class="['drop-zone border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all', isDragOver ? 'border-accent-500 bg-accent-500/10' : 'border-surface-600 hover:border-surface-500']">
              <input ref="fileInput" type="file" accept="image/*" class="hidden" @change="onFileChange" />
              <div v-if="preview" class="space-y-2">
                <img :src="preview" class="max-h-36 mx-auto rounded-lg object-contain" />
                <p class="text-sm text-surface-300">{{ fileName }}</p>
              </div>
              <div v-else class="space-y-2">
                <svg class="w-10 h-10 mx-auto text-surface-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
                <p class="text-sm text-surface-400">拖拽图片或点击选择</p>
              </div>
            </div>
            <div v-else @click="triggerBatchInput"
              :class="['drop-zone border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all', isDragOver ? 'border-accent-500 bg-accent-500/10' : 'border-surface-600 hover:border-surface-500']">
              <input ref="batchInput" type="file" accept="image/*" webkitdirectory directory multiple class="hidden" @change="onBatchChange" />
              <div v-if="batchFiles.length" class="text-emerald-400 font-medium">{{ batchFiles.length }} 个文件已选择</div>
              <div v-else class="space-y-2">
                <svg class="w-10 h-10 mx-auto text-surface-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>
                <p class="text-sm text-surface-400">拖拽文件夹或点击选择</p>
              </div>
            </div>
          </div>

          <!-- Presets -->
          <div>
            <label class="block text-sm font-medium text-surface-300 mb-2">阈值预设</label>
            <div class="flex gap-2">
              <button v-for="p in presets" :key="p.id" @click="applyPreset(p.id)"
                :class="['flex-1 py-1.5 rounded-lg text-sm font-medium transition-all', settings.preset === p.id ? 'bg-accent-600 text-white' : 'bg-surface-700 text-surface-300 hover:bg-surface-600']">
                {{ p.label }}
              </button>
            </div>
          </div>

          <!-- Sliders -->
          <div class="space-y-3">
            <div>
              <div class="flex justify-between mb-1">
                <label class="text-sm text-surface-300">置信度 conf</label>
                <span class="text-sm text-accent-400 font-mono">{{ settings.conf.toFixed(2) }}</span>
              </div>
              <input type="range" v-model.number="settings.conf" min="0.01" max="0.99" step="0.01" class="w-full" />
            </div>
            <div>
              <div class="flex justify-between mb-1">
                <label class="text-sm text-surface-300">IoU iou</label>
                <span class="text-sm text-accent-400 font-mono">{{ settings.iou.toFixed(2) }}</span>
              </div>
              <input type="range" v-model.number="settings.iou" min="0.1" max="0.95" step="0.01" class="w-full" />
            </div>
          </div>

          <!-- Run -->
          <button @click="runDetect" :disabled="isRunning || !canRun"
            class="w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all"
            :class="isRunning || !canRun ? 'bg-surface-700 text-surface-400 cursor-not-allowed' : 'bg-accent-600 hover:bg-accent-500 text-white'">
            <svg v-if="isRunning" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>
            {{ isRunning ? (settings.mode === 'single' ? '检测中...' : '处理中...') : '开始检测' }}
          </button>
        </div>

        <!-- RIGHT: Result -->
        <div class="bg-surface-800 rounded-2xl border border-surface-700 p-5 space-y-5">
          <h2 class="text-lg font-semibold text-white flex items-center gap-2">
            <span class="w-6 h-6 rounded bg-emerald-600 flex items-center justify-center text-xs">2</span>
            检测结果
          </h2>

          <div v-if="!result && !isRunning" class="flex flex-col items-center justify-center py-16 text-surface-500">
            <svg class="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
            <p>上传图片开始检测</p>
          </div>

          <div v-else-if="isRunning && settings.mode === 'single'" class="space-y-4 animate-pulse">
            <div class="h-48 bg-surface-700 rounded-xl"></div>
            <div class="h-4 bg-surface-700 rounded w-3/4"></div>
          </div>

          <div v-else-if="result" class="space-y-4">
            <!-- Single Result -->
            <div v-if="settings.mode === 'single'" class="space-y-4">
              <div class="relative rounded-xl overflow-hidden bg-surface-900">
                <img :src="'data:image/png;base64,' + result.annotated_image_base64" class="w-full max-h-[360px] object-contain" />
                <div class="absolute top-3 right-3 px-3 py-1.5 rounded-full text-xs font-bold shadow-lg"
                  :class="result.num_detections > 0 ? 'bg-emerald-500 text-white' : 'bg-surface-700 text-surface-300'">
                  {{ result.num_detections > 0 ? result.num_detections + ' 个缺陷' : '无缺陷' }}
                </div>
              </div>
              <div class="grid grid-cols-2 gap-3">
                <div class="bg-surface-900 rounded-xl p-3">
                  <div class="text-xs text-surface-400">图片尺寸</div>
                  <div class="text-sm font-semibold text-white">{{ result.image_size?.width }} × {{ result.image_size?.height }}</div>
                </div>
                <div class="bg-surface-900 rounded-xl p-3">
                  <div class="text-xs text-surface-400">检测数量</div>
                  <div class="text-sm font-semibold" :class="result.num_detections > 0 ? 'text-accent-400' : 'text-surface-400'">{{ result.num_detections }}</div>
                </div>
              </div>
              <div v-if="result.detections?.length > 0" class="space-y-2 max-h-64 overflow-y-auto">
                <div v-for="(d, i) in result.detections" :key="i" class="flex items-center gap-3 bg-surface-900 rounded-lg p-3">
                  <div class="w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold flex-shrink-0"
                    :style="{ backgroundColor: getColor(d.class_id) + '30', color: getColor(d.class_id) }">{{ d.class_id }}</div>
                  <div class="flex-1 min-w-0">
                    <div class="text-sm font-medium text-white truncate">{{ d.class_name }}</div>
                    <div class="text-xs text-surface-400 font-mono">[{{ d.bbox_xyxy.map(v => v.toFixed(0)).join(', ') }}]</div>
                  </div>
                  <div class="text-sm font-bold text-accent-400">{{ (d.confidence * 100).toFixed(1) }}%</div>
                </div>
              </div>
            </div>

            <!-- Batch Result -->
            <div v-else class="space-y-4">
              <div class="grid grid-cols-3 gap-3">
                <div class="bg-surface-900 rounded-xl p-3 text-center">
                  <div class="text-xl font-bold text-accent-400">{{ result.total_files }}</div>
                  <div class="text-xs text-surface-400">总文件</div>
                </div>
                <div class="bg-surface-900 rounded-xl p-3 text-center">
                  <div class="text-xl font-bold text-emerald-400">{{ result.success_count }}</div>
                  <div class="text-xs text-surface-400">成功</div>
                </div>
                <div class="bg-surface-900 rounded-xl p-3 text-center">
                  <div class="text-xl font-bold" :class="result.failure_count > 0 ? 'text-red-400' : 'text-surface-400'">{{ result.failure_count }}</div>
                  <div class="text-xs text-surface-400">失败</div>
                </div>
              </div>
              <div class="text-xs text-surface-400">输出目录: <code class="text-surface-300">{{ result.output_dir }}</code></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const modes = [
      { id: 'single', label: '单图检测', icon: '🖼' },
      { id: 'batch', label: '批量检测', icon: '📁' },
    ]
    const presets = [
      { id: 'balanced', label: '平衡', conf: 0.25, iou: 0.6 },
      { id: 'high_recall', label: '高召回', conf: 0.1, iou: 0.7 },
      { id: 'high_precision', label: '高精度', conf: 0.5, iou: 0.5 },
    ]
    const presetValues = { balanced: { conf: 0.25, iou: 0.6 }, high_recall: { conf: 0.1, iou: 0.7 }, high_precision: { conf: 0.5, iou: 0.5 } }

    const settings = reactive({ mode: 'single', modelId: null, conf: 0.25, iou: 0.6, preset: 'balanced' })
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
    const toast = reactive({ show: false, message: '', type: 'success' })

    const classColors = ['#ef4444','#f97316','#eab308','#22c55e','#14b8a6','#3b82f6','#8b5cf6','#ec4899','#f43f5e','#6366f1']
    const getColor = (id) => classColors[id % classColors.length]

    const canRun = computed(() => settings.mode === 'single' ? !!currentFile.value : batchFiles.value.length > 0)

    const applyPreset = (id) => {
      settings.preset = id
      const v = presetValues[id]
      if (v) { settings.conf = v.conf; settings.iou = v.iou }
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
      toast.message = message; toast.type = type; toast.show = true
      setTimeout(() => { toast.show = false }, 4000)
    }

    const loadModels = async () => {
      try {
        const api = (await import('../api/client.js')).default
        const res = await api.get('/models')
        modelList.value = res.data.models
      } catch {}
    }

    const runDetect = async () => {
      if (!canRun.value) { showToast(settings.mode === 'single' ? '请先选择图片' : '请先选择文件', 'error'); return }
      isRunning.value = true
      const query = new URLSearchParams({
        conf: settings.conf.toString(), iou: settings.iou.toString(),
        imgsz: '640', max_det: '300', model_id: settings.modelId || ''
      })
      try {
        const api = (await import('../api/client.js')).default
        if (settings.mode === 'single') {
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

    return { modes, presets, settings, isRunning, isDragOver, result, modelList, preview, fileName, currentFile, batchFiles, canRun, fileInput, batchInput, toast, getColor, applyPreset, triggerFileInput, triggerBatchInput, onFileChange, onDrop, onBatchChange, runDetect }
  }
}
