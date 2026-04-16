// Models View — Premium B&W Minimal
const { createApp, ref, onMounted } = Vue

export default {
  template: `
    <div class="models-root">
      <div class="models-header">
        <div>
          <div class="page-eyebrow">管理</div>
          <h1 class="page-title">模型</h1>
        </div>
        <button @click="showUploadModal = true" class="add-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          上传模型
        </button>
      </div>

      <div v-if="loading" class="list-state">加载中...</div>
      <div v-else-if="models.length === 0" class="empty-state">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><path d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <p>暂无自定义模型</p>
      </div>
      <div v-else class="models-grid">
        <div v-for="m in models" :key="m.id" class="model-card">
          <div class="model-card-left">
            <div class="model-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <div class="model-info">
              <div class="model-name">{{ m.name }}</div>
              <div class="model-path">{{ m.path }}</div>
              <div class="model-date" v-if="m.uploaded_at">上传于 {{ formatDate(m.uploaded_at) }}</div>
            </div>
          </div>
          <button v-if="m.uploaded_by === currentUserId" @click="deleteModel(m)" class="delete-btn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2"/></svg>
          </button>
        </div>
      </div>

      <!-- Upload Modal -->
      <div v-if="showUploadModal" class="modal-overlay" @click.self="showUploadModal = false">
        <div class="modal-box">
          <div class="modal-header">
            <h3 class="modal-title">上传模型</h3>
            <button @click="showUploadModal = false" class="modal-close">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
          <div class="modal-body">
            <label for="modelFileInput" class="upload-zone" style="cursor:pointer;display:block">
              <input ref="uploadInput" id="modelFileInput" type="file" accept=".pt" class="hidden" @change="onFileSelected" />
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke-linecap="round" stroke-linejoin="round"/></svg>
              <p class="upload-text">{{ uploadFile ? uploadFile.name : '点击选择 .pt 模型文件' }}</p>
            </label>
            <div v-if="uploadError" class="error-msg">{{ uploadError }}</div>
            <button @click="uploadModel" :disabled="!uploadFile || uploading" :class="['submit-btn', !uploadFile || uploading ? 'submit-btn--disabled' : '']">
              <span v-if="uploading" class="spinner"></span>
              <span v-else>开始上传</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const models = ref([])
    const loading = ref(false)
    const showUploadModal = ref(false)
    const uploadFile = ref(null)
    const uploadError = ref('')
    const uploading = ref(false)
    const uploadInput = ref(null)
    const isAdmin = ref(false)
    const currentUserId = ref(null)
    const user = JSON.parse(localStorage.getItem('user') || 'null')
    isAdmin.value = user?.role === 'admin'
    currentUserId.value = user?.id || null

    const formatDate = (iso) => {
      if (!iso) return ''
      return new Date(iso).toLocaleString('zh-CN')
    }

    const loadModels = async () => {
      loading.value = true
      try {
        const api = (await import('../api/client.js')).default
        const res = await api.get('/models')
        models.value = res.data.models
      } catch (e) { console.error(e) }
      finally { loading.value = false }
    }

    const onFileSelected = (e) => {
      uploadFile.value = e.target.files?.[0] || null
      uploadError.value = ''
    }

    const uploadModel = async () => {
      if (!uploadFile.value) return
      uploading.value = true
      uploadError.value = ''
      try {
        const api = (await import('../api/client.js')).default
        const formData = new FormData()
        formData.append('file', uploadFile.value)
        await api.post('/models/upload', formData)
        showUploadModal.value = false
        uploadFile.value = null
        await loadModels()
      } catch (e) {
        uploadError.value = e.response?.data?.detail || '上传失败'
      } finally {
        uploading.value = false
      }
    }

    const deleteModel = async (m) => {
      if (!confirm('确定删除该模型？')) return
      try {
        const api = (await import('../api/client.js')).default
        await api.delete('/models/' + m.id)
        await loadModels()
      } catch (e) { console.error(e) }
    }

    onMounted(() => { loadModels() })

    return { models, loading, showUploadModal, uploadFile, uploadError, uploading, uploadInput, isAdmin, currentUserId, formatDate, loadModels, onFileSelected, uploadModel, deleteModel }
  }
}
