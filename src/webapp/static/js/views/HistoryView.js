// History View
const { createApp, ref, computed, onMounted } = Vue

export default {
  template: `
    <div class="space-y-5">
      <div class="flex items-center justify-between">
        <h2 class="text-xl font-semibold text-white">检测历史</h2>
        <div class="flex items-center gap-3 text-sm text-surface-400">
          共 <strong class="text-white">{{ total }}</strong> 条记录
        </div>
      </div>

      <!-- Filters -->
      <div class="bg-surface-800 rounded-xl border border-surface-700 p-4">
        <div class="flex flex-wrap gap-3">
          <input v-model="filters.search" type="text" placeholder="搜索文件名..." @input="debounceSearch"
            class="flex-1 min-w-[200px] bg-surface-700 border border-surface-600 rounded-lg px-3 py-2 text-sm text-white placeholder-surface-500 focus:outline-none focus:border-accent-500" />
          <select v-model="filters.model" @change="loadHistory"
            class="bg-surface-700 border border-surface-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500">
            <option value="">全部模型</option>
            <option v-for="m in modelOptions" :key="m" :value="m">{{ m }}</option>
          </select>
          <button @click="clearFilters" class="px-4 py-2 bg-surface-700 hover:bg-surface-600 rounded-lg text-sm text-surface-300 transition-colors">
            清空筛选
          </button>
        </div>
      </div>

      <!-- Records -->
      <div v-if="loading" class="text-center py-16 text-surface-400">加载中...</div>
      <div v-else-if="records.length === 0" class="text-center py-16 text-surface-500">
        <svg class="w-16 h-16 mx-auto mb-4 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        <p>暂无记录</p>
      </div>
      <div v-else class="space-y-2">
        <div v-for="r in records" :key="r.id"
          class="bg-surface-800 rounded-xl border border-surface-700 p-4 flex items-center gap-4 hover:border-surface-500 transition-all cursor-pointer"
          @click="showDetail(r)">
          <div class="w-16 h-16 rounded-lg bg-surface-900 flex-shrink-0 overflow-hidden">
            <img v-if="r.num_detections > 0 && r.annotated_image_base64" :src="'data:image/png;base64,' + r.annotated_image_base64" class="w-full h-full object-cover" />
            <div v-else class="w-full h-full flex items-center justify-center text-surface-600">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
            </div>
          </div>
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2 mb-1">
              <span class="text-sm font-medium text-white truncate">{{ r.filename }}</span>
              <span class="text-[10px] px-1.5 py-0.5 rounded font-semibold" :class="r.mode === 'single' ? 'bg-accent-500/15 text-accent-400' : 'bg-purple-500/15 text-purple-400'">单图</span>
            </div>
            <div class="text-xs text-surface-400">{{ r.model_name }} · {{ formatDate(r.created_at) }}</div>
          </div>
          <div class="text-right flex-shrink-0">
            <div class="text-sm font-semibold" :class="r.num_detections > 0 ? 'text-accent-400' : 'text-surface-400'">{{ r.num_detections }} 个</div>
            <div class="text-xs text-surface-500">{{ r.image_size?.width }}×{{ r.image_size?.height }}</div>
          </div>
        </div>
      </div>

      <!-- Pagination -->
      <div v-if="totalPages > 1" class="flex justify-center gap-2">
        <button @click="page--; loadHistory()" :disabled="page === 1"
          class="px-4 py-2 bg-surface-800 border border-surface-700 rounded-lg text-sm text-surface-300 hover:bg-surface-700 disabled:opacity-50">
          上一页
        </button>
        <span class="px-4 py-2 text-sm text-surface-400">第 {{ page }} / {{ totalPages }} 页</span>
        <button @click="page++; loadHistory()" :disabled="page === totalPages"
          class="px-4 py-2 bg-surface-800 border border-surface-700 rounded-lg text-sm text-surface-300 hover:bg-surface-700 disabled:opacity-50">
          下一页
        </button>
      </div>

      <!-- Detail Modal -->
      <div v-if="showModal" class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4" @click.self="showModal = false">
        <div class="bg-surface-800 rounded-2xl border border-surface-700 w-full max-w-4xl max-h-[85vh] overflow-hidden">
          <div class="flex items-center justify-between p-5 border-b border-surface-700">
            <div>
              <h3 class="text-lg font-semibold text-white">{{ detailItem?.filename }}</h3>
              <p class="text-xs text-surface-400 mt-0.5">{{ detailItem?.model_name }} · {{ formatDate(detailItem?.created_at) }}</p>
            </div>
            <button @click="showModal = false" class="text-surface-400 hover:text-white p-1 rounded-lg hover:bg-surface-700">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
            </button>
          </div>
          <div class="p-5 overflow-y-auto max-h-[calc(85vh-72px)]" v-if="detailItem">
            <div class="space-y-4">
              <div class="grid grid-cols-3 gap-3">
                <div class="bg-surface-900 rounded-lg p-3 text-center">
                  <div class="text-xs text-surface-400">图片尺寸</div>
                  <div class="text-sm font-semibold text-white">{{ detailItem.image_size?.width }} × {{ detailItem.image_size?.height }}</div>
                </div>
                <div class="bg-surface-900 rounded-lg p-3 text-center">
                  <div class="text-xs text-surface-400">检测数量</div>
                  <div class="text-sm font-semibold text-accent-400">{{ detailItem.num_detections }}</div>
                </div>
                <div class="bg-surface-900 rounded-lg p-3 text-center">
                  <div class="text-xs text-surface-400">置信度阈值</div>
                  <div class="text-sm font-semibold text-white">{{ (detailItem.conf * 100).toFixed(0) }}%</div>
                </div>
              </div>
              <div v-if="detailItem.detections?.length > 0" class="space-y-2">
                <h4 class="text-sm font-medium text-surface-300">检测详情</h4>
                <div v-for="(d, i) in detailItem.detections" :key="i" class="flex items-center gap-3 bg-surface-900 rounded-lg p-3">
                  <div class="w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold"
                    :style="{ backgroundColor: getColor(d.class_id) + '30', color: getColor(d.class_id) }">{{ d.class_id }}</div>
                  <div class="flex-1">
                    <div class="text-sm font-medium text-white">{{ d.class_name }}</div>
                    <div class="text-xs text-surface-400 font-mono">[{{ d.bbox_xyxy.map(v => v.toFixed(0)).join(', ') }}]</div>
                  </div>
                  <div class="text-sm font-bold text-accent-400">{{ (d.confidence * 100).toFixed(1) }}%</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const records = ref([])
    const loading = ref(false)
    const total = ref(0)
    const page = ref(1)
    const limit = 20
    const filters = reactive({ search: '', model: '' })
    const modelOptions = ref([])
    const showModal = ref(false)
    const detailItem = ref(null)

    const totalPages = computed(() => Math.ceil(total.value / limit))

    const classColors = ['#ef4444','#f97316','#eab308','#22c55e','#14b8a6','#3b82f6','#8b5cf6','#ec4899','#f43f5e','#6366f1']
    const getColor = (id) => classColors[id % classColors.length]

    const formatDate = (iso) => {
      if (!iso) return ''
      return new Date(iso).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    }

    let searchTimer = null
    const debounceSearch = () => {
      clearTimeout(searchTimer)
      searchTimer = setTimeout(() => { page.value = 1; loadHistory() }, 400)
    }

    const clearFilters = () => {
      filters.search = ''; filters.model = ''
      page.value = 1; loadHistory()
    }

    const loadHistory = async () => {
      loading.value = true
      try {
        const api = (await import('../api/client.js')).default
        const params = { page: page.value, limit, search: filters.search || undefined, model_name: filters.model || undefined }
        const res = await api.get('/detections', { params })
        records.value = res.data.records
        total.value = res.data.total
      } catch (e) { console.error(e) }
      finally { loading.value = false }
    }

    const loadModels = async () => {
      try {
        const api = (await import('../api/client.js')).default
        const res = await api.get('/models')
        modelOptions.value = res.data.models.map(m => m.name)
      } catch {}
    }

    const showDetail = async (r) => {
      try {
        const api = (await import('../api/client.js')).default
        const res = await api.get('/detections/' + r.id)
        detailItem.value = res.data
        showModal.value = true
      } catch {}
    }

    onMounted(() => { loadHistory(); loadModels() })

    return { records, loading, total, page, limit, filters, modelOptions, showModal, detailItem, totalPages, getColor, formatDate, debounceSearch, clearFilters, loadHistory, showDetail }
  }
}
