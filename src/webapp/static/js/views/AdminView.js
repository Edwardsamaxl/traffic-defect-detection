// Admin View - user management
const { createApp, ref, onMounted } = Vue

export default {
  template: `
    <div class="space-y-5">
      <h2 class="text-xl font-semibold text-white">用户管理</h2>

      <div class="bg-surface-800 rounded-xl border border-surface-700 p-5">
        <div class="flex items-center justify-between mb-4">
          <div class="text-sm text-surface-400">共 <strong class="text-white">{{ users.length }}</strong> 位用户</div>
        </div>

        <div v-if="loading" class="text-center py-8 text-surface-400">加载中...</div>
        <div v-else class="overflow-x-auto">
          <table class="w-full">
            <thead>
              <tr class="text-left text-xs text-surface-400 uppercase tracking-wider border-b border-surface-700">
                <th class="pb-3 pr-4 font-medium">用户名</th>
                <th class="pb-3 pr-4 font-medium">角色</th>
                <th class="pb-3 pr-4 font-medium">注册时间</th>
                <th class="pb-3 font-medium">操作</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-surface-700">
              <tr v-for="u in users" :key="u.id" class="text-sm">
                <td class="py-3 pr-4 text-white font-medium">{{ u.username }}</td>
                <td class="py-3 pr-4">
                  <select v-model="u.role" @change="updateRole(u)"
                    class="bg-surface-700 border border-surface-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-accent-500">
                    <option value="user">user</option>
                    <option value="admin">admin</option>
                  </select>
                </td>
                <td class="py-3 pr-4 text-surface-400 text-xs">{{ formatDate(u.created_at) }}</td>
                <td class="py-3">
                  <button @click="deleteUser(u)" class="text-surface-500 hover:text-red-400 text-xs transition-colors"
                    :disabled="u.id === currentUserId">
                    删除
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
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
