// Axios API client with auth interceptors
const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

// Attach token
api.interceptors.request.use(config => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Handle 401 → redirect to login
api.interceptors.response.use(
  res => res,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/static/index.html#/login'
    }
    return Promise.reject(err)
  }
)

export default api
