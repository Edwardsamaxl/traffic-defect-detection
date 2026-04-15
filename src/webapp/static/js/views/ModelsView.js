// Models View
const { createApp, ref, onMounted } = Vue

export default {
  template: `
    <div class="space-y-5">
      <div class="flex items-center justify-between">
        <h2 class="text-xl font-semibold text-white">模型管理</h2>
        <button @click="showUploadModal = true" class="px-4 py-2 bg-accent-600 hover:bg-accent-500 rounded-lg text-sm font-medium text-white transition-colors">
          上传模型
        </button>
      </div>

      <div v-if="loading" class="text-center py-16 text-surface-400">加载中...</div>
      <div v-else-if="models.length === 0" class="text-center py-16 text-surface-500">
        <svg class="w-16 h-16 mx-auto mb-4 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
        <p>暂无自定义模型</p>
      </div>
      <div v-else class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div v-for="m in models" :key="m.id" class="bg-surface-800 rounded-xl border border-surface-700 p-4">
          <div class="flex items-start justify-between mb-3">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-lg bg-accent-500/15 flex items-center justify-center">
                <svg class="w-5 h-5 text-accent-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/></svg>
              </div>
              <div>
                <div class="text-sm font-medium text-white">{{ m.name }}</div>
                <div class="text-xs text-surface-400">{{ m.path }}</div>
              </div>
            </div>
            <button v-if="m.uploaded_by && isAdmin" @click="deleteModel(m)"
              class="p-1.5 text-surface-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
            </button>
          </div>
          <div class="text-xs text-surface-400" v-if="m.uploaded_at">上传于 {{ formatDate(m.uploaded_at) }}</div>
        </div>
      </div>

      <!-- Upload Modal -->
      <div v-if="showUploadModal" class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4" @click.self="showUploadModal = false">
        <div class="bg-surface-800 rounded-2xl border border-surface-700 w-full max-w-md">
          <div class="flex items-center justify-between p-5 border-b border-surface-700">
            <h3 class="text-lg font-semibold text-white">上传模型</h3>
            <button @click="showUploadModal = false" class="text-surface-400 hover:text-white">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
            </button>
          </div>
          <div class="p-5 space-y-4">
            <div class="border-2 border-dashed border-surface-600 rounded-xl p-8 text-center cursor-pointer hover:border-accent-500 transition-colors"
              @click="$refs.uploadInput.click()">
              <input ref="uploadInput" type="file" accept=".pt" class="hidden" @change="onFileSelected" />
              <svg class="w-10 h-10 mx-auto text-surface-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
              <p class="text-sm text-surface-400">{{ uploadFile ? uploadFile.name : '点击选择 .pt 模型文件' }}</p>
            </div>
            <div v-if="uploadError" class="bg-red-500/15 border border-red-500/30 rounded-lg px-3 py-2 text-red-400 text-sm">{{ uploadError }}</div>
            <button @click="uploadModel" :disabled="!uploadFile || uploading"
              class="w-full py-2.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2"
              :class="!uploadFile || uploading ? 'bg-surface-700 text-surface-400 cursor-not-allowed' : 'bg-accent-600 hover:bg-accent-500 text-white'">
              <svg v-if="uploading" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>
              {{ uploading ? '上传中...' : '开始上传' }}
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
    const user = JSON.parse(localStorage.getItem('user') || 'null')
    isAdmin.value = user?.role === 'admin'

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

    return { models, loading, showUploadModal, uploadFile, uploadError, uploading, uploadInput, isAdmin, formatDate, loadModels, onFileSelected, uploadModel, deleteModel }
  }
}
