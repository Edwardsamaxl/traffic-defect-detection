// Login View — Premium B&W Minimal
const { createApp, ref } = Vue

export default {
  template: `
    <div class="login-root">
      <!-- Atmospheric background -->
      <div class="login-bg">
        <div class="bg-orb bg-orb-1"></div>
        <div class="bg-orb bg-orb-2"></div>
        <div class="noise-overlay"></div>
      </div>

      <div class="login-container">
        <!-- Left: Branding -->
        <div class="login-brand">
          <div class="brand-mark">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
              <rect x="1" y="1" width="38" height="38" rx="8" stroke="currentColor" stroke-width="1.5"/>
              <path d="M12 20h16M20 12v16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
          </div>
          <h1 class="brand-title">Traffic<br/>Defect<br/>Detection</h1>
          <p class="brand-sub">智能交通缺陷检测系统</p>
        </div>

        <!-- Right: Form -->
        <div class="login-form-wrap">
          <div class="form-header">
            <h2 class="form-title">登录</h2>
            <p class="form-desc">输入您的凭证以继续</p>
          </div>

          <div class="form-body">
            <div class="field">
              <label class="field-label">用户名</label>
              <input
                v-model="username"
                type="text"
                placeholder="username"
                @keyup.enter="handleLogin"
                class="field-input"
                autocomplete="username"
              />
            </div>
            <div class="field">
              <label class="field-label">密码</label>
              <input
                v-model="password"
                type="password"
                placeholder="password"
                @keyup.enter="handleLogin"
                class="field-input"
                autocomplete="current-password"
              />
            </div>

            <div v-if="error" class="error-msg">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              {{ error }}
            </div>

            <button @click="handleLogin" :disabled="loading" class="submit-btn">
              <span v-if="loading" class="spinner"></span>
              <span v-else>登录</span>
            </button>
          </div>

          <div class="form-footer">
            <span class="footer-text">没有账号？</span>
            <router-link to="/register" class="footer-link">立即注册</router-link>
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
