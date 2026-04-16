// History View — Premium B&W Minimal
const { createApp, ref, computed, onMounted } = Vue

export default {
  template: `
    <div class="history-root">
      <div class="history-header">
        <div>
          <div class="page-eyebrow">记录</div>
          <h1 class="page-title">检测历史</h1>
        </div>
        <div class="header-total">共 <strong>{{ total }}</strong> 条记录</div>
      </div>

      <!-- Filter Bar -->
      <div class="filter-bar">
        <input v-model="filters.search" type="text" placeholder="搜索文件名..." @input="debounceSearch" class="filter-input" />
        <select v-model="filters.model" @change="loadHistory" class="filter-select">
          <option value="">全部模型</option>
          <option v-for="m in modelOptions" :key="m" :value="m">{{ m }}</option>
        </select>
        <button @click="clearFilters" class="clear-btn">清空</button>
      </div>

      <!-- Records -->
      <div v-if="loading" class="list-state">加载中...</div>
      <div v-else-if="records.length === 0" class="empty-state">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <p>暂无记录</p>
      </div>
      <div v-else class="records-list">
        <div v-for="r in records" :key="r.id" class="record-item">
          <div class="record-thumb">
            <img v-if="r.num_detections > 0 && r.annotated_image_base64" :src="'data:image/png;base64,' + r.annotated_image_base64" class="thumb-img" />
            <div v-else class="thumb-placeholder">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
          </div>
          <div class="record-info">
            <div class="record-name">{{ r.filename }}</div>
            <div class="record-meta">{{ r.model_name }} · {{ formatDate(r.created_at) }}</div>
          </div>
          <div class="record-right">
            <div :class="['record-count', r.num_detections > 0 ? 'count--active' : '']">{{ r.num_detections }} 个</div>
            <div class="record-size">{{ r.image_size?.width }}×{{ r.image_size?.height }}</div>
          </div>
        </div>
      </div>

      <!-- Pagination -->
      <div v-if="totalPages > 1" class="pagination">
        <button @click="page--; loadHistory()" :disabled="page === 1" class="page-btn">上一页</button>
        <span class="page-info">{{ page }} / {{ totalPages }}</span>
        <button @click="page++; loadHistory()" :disabled="page === totalPages" class="page-btn">下一页</button>
      </div>

      <!-- Detail Modal -->
      <div v-if="showModal" class="modal-overlay" @click.self="showModal = false">
        <div class="modal-box modal-box--wide">
          <div class="modal-header">
            <div>
              <div class="modal-title">{{ detailItem?.filename }}</div>
              <div class="modal-sub">{{ detailItem?.model_name }} · {{ formatDate(detailItem?.created_at) }}</div>
            </div>
            <button @click="showModal = false" class="modal-close">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
          <div class="modal-body" v-if="detailItem">
            <div v-if="detailItem.annotated_image_base64" class="detail-image-wrap">
              <img :src="'data:image/png;base64,' + detailItem.annotated_image_base64" class="detail-annotated-img" />
            </div>
            <div class="detail-stats">
              <div class="detail-stat">
                <div class="detail-stat-val">{{ detailItem.image_size?.width }} × {{ detailItem.image_size?.height }}</div>
                <div class="detail-stat-label">图片尺寸</div>
              </div>
              <div class="detail-stat">
                <div class="detail-stat-val text-active">{{ detailItem.num_detections }}</div>
                <div class="detail-stat-label">检测数量</div>
              </div>
              <div class="detail-stat">
                <div class="detail-stat-val">{{ (detailItem.conf * 100).toFixed(0) }}%</div>
                <div class="detail-stat-label">置信度</div>
              </div>
            </div>
            <div v-if="detailItem.detections?.length > 0" class="detections-list">
              <div v-for="(d, i) in detailItem.detections" :key="i" class="detection-item">
                <div class="detection-dot" :style="{ background: getColor(d.class_id) }"></div>
                <div class="detection-info">
                  <div class="detection-name">{{ d.class_name }}</div>
                  <div class="detection-bbox">[{{ d.bbox_xyxy.map(v => v.toFixed(0)).join(', ') }}]</div>
                </div>
                <div class="detection-conf">{{ (d.confidence * 100).toFixed(1) }}%</div>
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
    const filters = ref({ search: '', model: '' })
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
      filters.value.search = ''; filters.value.model = ''
      page.value = 1; loadHistory()
    }

    const loadHistory = async () => {
      loading.value = true
      try {
        const api = (await import('../api/client.js')).default
        const params = { page: page.value, limit, search: filters.value.search || undefined, model_name: filters.value.model || undefined }
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

    return { records, loading, total, page, filters, modelOptions, showModal, detailItem, totalPages, getColor, formatDate, debounceSearch, clearFilters, loadHistory, loadModels, showDetail }
  }
}
