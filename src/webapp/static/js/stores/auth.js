// Auth store - browser compatible version
const TOKEN_KEY = 'token'
const USER_KEY = 'user'

// Setup axios interceptor for 401
axios.interceptors.response.use(
  res => res,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem(USER_KEY)
      window.location.href = '/static/index.html#/login'
    }
    return Promise.reject(err)
  }
)

// Attach token to requests
axios.interceptors.request.use(config => {
  const token = localStorage.getItem(TOKEN_KEY)
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Simple reactive state
const state = Vue.reactive({
  user: JSON.parse(localStorage.getItem(USER_KEY) || 'null'),
  token: localStorage.getItem(TOKEN_KEY) || null,
  loading: false,
})

const auth = {
  state,

  actions: {
    async login({ username, password }) {
      const res = await axios.post('/api/auth/login', { username, password })
      const { user, token } = res.data
      localStorage.setItem(TOKEN_KEY, token)
      localStorage.setItem(USER_KEY, JSON.stringify(user))
      state.user = user
      state.token = token
      return user
    },

    async register({ username, password }) {
      const res = await axios.post('/api/auth/register', { username, password })
      const { user, token } = res.data
      localStorage.setItem(TOKEN_KEY, token)
      localStorage.setItem(USER_KEY, JSON.stringify(user))
      state.user = user
      state.token = token
      return user
    },

    logout() {
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem(USER_KEY)
      state.user = null
      state.token = null
    },

    async fetchMe() {
      if (!state.token) return null
      try {
        const res = await axios.get('/api/auth/me')
        const user = res.data.user
        localStorage.setItem(USER_KEY, JSON.stringify(user))
        state.user = user
        return user
      } catch {
        localStorage.removeItem(TOKEN_KEY)
        localStorage.removeItem(USER_KEY)
        state.user = null
        state.token = null
        return null
      }
    },
  },
}

export { auth }
