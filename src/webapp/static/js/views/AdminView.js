// Admin View — Premium B&W Minimal
const { createApp, ref, onMounted } = Vue

export default {
  template: `
    <div class="admin-root">
      <div class="admin-header">
        <div>
          <div class="page-eyebrow">管理</div>
          <h1 class="page-title">用户管理</h1>
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">用户列表 <span class="title-count">{{ users.length }}</span></div>

        <div v-if="loading" class="list-state">加载中...</div>
        <div v-else class="user-table">
          <div class="table-head">
            <div class="col-user">用户名</div>
            <div class="col-role">角色</div>
            <div class="col-date">注册时间</div>
            <div class="col-action">操作</div>
          </div>
          <div v-for="u in users" :key="u.id" class="table-row">
            <div class="col-user">
              <div class="user-avatar">{{ u.username[0].toUpperCase() }}</div>
              <span class="user-name">{{ u.username }}</span>
            </div>
            <div class="col-role">
              <select v-model="u.role" @change="updateRole(u)" class="role-select">
                <option value="user">user</option>
                <option value="admin">admin</option>
              </select>
            </div>
            <div class="col-date text-muted">{{ formatDate(u.created_at) }}</div>
            <div class="col-action">
              <button
                @click="deleteUser(u)"
                :disabled="u.id === currentUserId"
                :class="['action-btn', u.id === currentUserId ? 'action-btn--disabled' : 'action-btn--danger']"
              >删除</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const users = ref([])
    const loading = ref(false)
    const currentUserId = ref(null)
    const user = JSON.parse(localStorage.getItem('user') || 'null')
    currentUserId.value = user?.id

    const formatDate = (iso) => {
      if (!iso) return ''
      return new Date(iso).toLocaleString('zh-CN')
    }

    const loadUsers = async () => {
      loading.value = true
      try {
        const api = (await import('../api/client.js')).default
        const res = await api.get('/users')
        users.value = res.data.users
      } catch (e) { console.error(e) }
      finally { loading.value = false }
    }

    const updateRole = async (u) => {
      try {
        const api = (await import('../api/client.js')).default
        await api.put('/users/' + u.id + '/role', { role: u.role })
      } catch (e) { console.error(e) }
    }

    const deleteUser = async (u) => {
      if (!confirm('确定删除用户 ' + u.username + '？')) return
      try {
        const api = (await import('../api/client.js')).default
        await api.delete('/users/' + u.id)
        await loadUsers()
      } catch (e) { console.error(e) }
    }

    onMounted(() => { loadUsers() })

    return { users, loading, currentUserId, formatDate, loadUsers, updateRole, deleteUser }
  }
}
