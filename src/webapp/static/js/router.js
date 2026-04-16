// Vue Router setup
const routes = [
  { path: '/login', name: 'login', component: () => import('./views/LoginView.js') },
  { path: '/register', name: 'register', component: () => import('./views/RegisterView.js') },
  { path: '/', redirect: '/detect' },
  { path: '/detect', name: 'detect', component: () => import('./views/DetectView.js'), meta: { requiresAuth: true } },
  { path: '/history', name: 'history', component: () => import('./views/HistoryView.js'), meta: { requiresAuth: true } },
  { path: '/dashboard', name: 'dashboard', component: () => import('./views/DashboardView.js'), meta: { requiresAuth: true } },
  { path: '/models', name: 'models', component: () => import('./views/ModelsView.js'), meta: { requiresAuth: true } },
]

const router = VueRouter.createRouter({
  history: VueRouter.createWebHashHistory(),
  routes,
})

// Navigation guards
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  if (to.meta.requiresAuth && !token) {
    next({ name: 'login' })
  } else if ((to.name === 'login' || to.name === 'register') && token) {
    next({ name: 'dashboard' })
  } else {
    next()
  }
})

export default router
