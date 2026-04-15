// Login View
const { createApp, ref } = Vue

export default {
  template: `
    <div class="min-h-screen bg-surface-900 flex items-center justify-center p-4">
      <div class="w-full max-w-sm">
        <div class="text-center mb-8">
          <div class="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent-500 to-accent-700 flex items-center justify-center mx-auto mb-4 shadow-lg">
            <svg class="w-9 h-9 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
            </svg>
          </div>
          <h1 class="text-2xl font-bold text-white">交通缺陷检测</h1>
          <p class="text-surface-400 text-sm mt-1">登录到您的账号</p>
        </div>

        <div class="bg-surface-800 rounded-2xl border border-surface-700 p-6 space-y-4">
          <div>
            <label class="block text-sm font-medium text-surface-300 mb-1.5">用户名</label>
            <input v-model="username" type="text" placeholder="请输入用户名" @keyup.enter="handleLogin"
              class="w-full bg-surface-700 border border-surface-600 rounded-lg px-3 py-2.5 text-sm text-white placeholder-surface-500 focus:outline-none focus:border-accent-500" />
          </div>
          <div>
            <label class="block text-sm font-medium text-surface-300 mb-1.5">密码</label>
            <input v-model="password" type="password" placeholder="请输入密码" @keyup.enter="handleLogin"
              class="w-full bg-surface-700 border border-surface-600 rounded-lg px-3 py-2.5 text-sm text-white placeholder-surface-500 focus:outline-none focus:border-accent-500" />
          </div>

          <div v-if="error" class="bg-red-500/15 border border-red-500/30 rounded-lg px-3 py-2 text-red-400 text-sm">
            {{ error }}
          </div>

          <button @click="handleLogin" :disabled="loading"
            class="w-full py-2.5 rounded-xl font-semibold text-sm transition-all flex items-center justify-center gap-2"
            :class="loading ? 'bg-surface-700 text-surface-400 cursor-not-allowed' : 'bg-accent-600 hover:bg-accent-500 text-white'">
            <svg v-if="loading" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>
            {{ loading ? '登录中...' : '登录' }}
          </button>

          <div class="text-center text-sm text-surface-400">
            还没有账号？<router-link to="/register" class="text-accent-400 hover:text-accent-300 font-medium">立即注册</router-link>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const username = ref('')
    const password = ref('')
    const loading = ref(false)
    const error = ref('')

    const handleLogin = async () => {
      if (!username.value || !password.value) {
        error.value = '请填写用户名和密码'
        return
      }
      loading.value = true
      error.value = ''
      try {
        const { auth } = await import('../stores/auth.js')
        await auth.actions.login({ username: username.value, password: password.value })
        window.location.hash = '#/dashboard'
        window.location.reload()
      } catch (e) {
        error.value = e.response?.data?.detail || '登录失败'
      } finally {
        loading.value = false
      }
    }

    return { username, password, loading, error, handleLogin }
  }
}
