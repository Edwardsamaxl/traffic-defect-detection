// Register View — Premium B&W Minimal
const { createApp, ref } = Vue

export default {
  template: `
    <div class="login-root">
      <div class="login-bg">
        <div class="bg-orb bg-orb-1"></div>
        <div class="bg-orb bg-orb-2"></div>
        <div class="noise-overlay"></div>
      </div>

      <div class="login-container">
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

        <div class="login-form-wrap">
          <div class="form-header">
            <h2 class="form-title">注册</h2>
            <p class="form-desc">创建您的账户</p>
          </div>

          <div class="form-body">
            <div class="field">
              <label class="field-label">用户名</label>
              <input
                v-model="username"
                type="text"
                placeholder="username"
                @keyup.enter="handleRegister"
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
                @keyup.enter="handleRegister"
                class="field-input"
                autocomplete="new-password"
              />
            </div>
            <div class="field">
              <label class="field-label">确认密码</label>
              <input
                v-model="confirmPassword"
                type="password"
                placeholder="confirm password"
                @keyup.enter="handleRegister"
                class="field-input"
                autocomplete="new-password"
              />
            </div>

            <div v-if="error" class="error-msg">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              {{ error }}
            </div>

            <button @click="handleRegister" :disabled="loading" class="submit-btn">
              <span v-if="loading" class="spinner"></span>
              <span v-else>注册</span>
            </button>
          </div>

          <div class="form-footer">
            <span class="footer-text">已有账号？</span>
            <router-link to="/login" class="footer-link">立即登录</router-link>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const username = ref('')
    const password = ref('')
    const confirmPassword = ref('')
    const loading = ref(false)
    const error = ref('')

    const handleRegister = async () => {
      if (!username.value || !password.value) {
        error.value = '请填写所有字段'
        return
      }
      if (password.value !== confirmPassword.value) {
        error.value = '两次密码不一致'
        return
      }
      loading.value = true
      error.value = ''
      try {
        const { auth } = await import('../stores/auth.js')
        await auth.actions.register({ username: username.value, password: password.value })
        window.location.hash = '#/dashboard'
        window.location.reload()
      } catch (e) {
        error.value = e.response?.data?.detail || '注册失败'
      } finally {
        loading.value = false
      }
    }

    return { username, password, confirmPassword, loading, error, handleRegister }
  }
}
