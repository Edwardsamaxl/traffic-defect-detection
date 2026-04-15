// Dashboard View with charts
const { createApp, ref, onMounted } = Vue

export default {
  template: `
    <div class="space-y-6">
      <div class="flex items-center justify-between">
        <h2 class="text-xl font-semibold text-white">数据统计</h2>
        <span class="text-sm text-surface-400">最近30天数据</span>
      </div>

      <!-- Stats Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-surface-800 rounded-xl border border-surface-700 p-5">
          <div class="text-sm text-surface-400 mb-1">总检测次数</div>
          <div class="text-3xl font-bold text-accent-400">{{ stats.total_detections }}</div>
        </div>
        <div class="bg-surface-800 rounded-xl border border-surface-700 p-5">
          <div class="text-sm text-surface-400 mb-1">今日检测</div>
          <div class="text-3xl font-bold text-emerald-400">{{ stats.detections_today }}</div>
        </div>
        <div class="bg-surface-800 rounded-xl border border-surface-700 p-5">
          <div class="text-sm text-surface-400 mb-1">平均每日</div>
          <div class="text-3xl font-bold text-white">{{ avgPerDay }}</div>
        </div>
      </div>

      <!-- Charts Row -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="bg-surface-800 rounded-xl border border-surface-700 p-5">
          <h3 class="text-sm font-medium text-surface-300 mb-4">每日检测趋势</h3>
          <canvas id="dailyChart" height="200"></canvas>
        </div>
        <div class="bg-surface-800 rounded-xl border border-surface-700 p-5">
          <h3 class="text-sm font-medium text-surface-300 mb-4">缺陷类型分布</h3>
          <canvas id="classChart" height="200"></canvas>
        </div>
      </div>

      <!-- Recent Detections -->
      <div class="bg-surface-800 rounded-xl border border-surface-700 p-5">
        <h3 class="text-sm font-medium text-surface-300 mb-4">最近检测记录</h3>
        <div v-if="recentLoading" class="text-center py-8 text-surface-400">加载中...</div>
        <div v-else-if="recentRecords.length === 0" class="text-center py-8 text-surface-400">暂无数据</div>
        <div v-else class="space-y-2">
          <div v-for="r in recentRecords" :key="r.id" class="flex items-center gap-3 p-3 bg-surface-900 rounded-lg">
            <div class="flex-1 min-w-0">
              <div class="text-sm font-medium text-white truncate">{{ r.filename }}</div>
              <div class="text-xs text-surface-400">{{ r.model_name }} · {{ formatDate(r.created_at) }}</div>
            </div>
            <span class="px-2 py-1 rounded text-xs font-semibold" :class="r.num_detections > 0 ? 'bg-accent-500/15 text-accent-400' : 'bg-surface-700 text-surface-400'">
              {{ r.num_detections }} 个
            </span>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const stats = ref({ total_detections: 0, detections_today: 0, by_day: [], by_class: [] })
    const recentRecords = ref([])
    const recentLoading = ref(false)
    const avgPerDay = computed(() => {
      const days = stats.value.by_day?.length || 1
      return Math.round(stats.value.total_detections / days)
    })

    const formatDate = (iso) => {
      if (!iso) return ''
      return new Date(iso).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    }

    const loadStats = async () => {
      try {
        const api = (await import('../api/client.js')).default
        const res = await api.get('/dashboard/stats')
        stats.value = res.data

        // Draw charts after data loads
        setTimeout(() => {
          drawDailyChart(stats.value.by_day || [])
          drawClassChart(stats.value.by_class || [])
        }, 100)
      } catch (e) {
        console.error('Failed to load stats', e)
      }
    }

    const loadRecent = async () => {
      recentLoading.value = true
      try {
        const api = (await import('../api/client.js')).default
        const res = await api.get('/detections?limit=5')
        recentRecords.value = res.data.records
      } catch (e) {
        console.error('Failed to load recent', e)
      } finally {
        recentLoading.value = false
      }
    }

    const drawDailyChart = (data) => {
      const canvas = document.getElementById('dailyChart')
      if (!canvas || !data.length) return
      const ctx = canvas.getContext('2d')
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.map(d => d.date.slice(5)),
          datasets: [{
            label: '检测数',
            data: data.map(d => d.count),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59,130,246,0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: 3,
            pointBackgroundColor: '#3b82f6',
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#64748b' }, grid: { color: '#334155' } },
            y: { ticks: { color: '#64748b' }, grid: { color: '#334155' }, beginAtZero: true }
          }
        }
      })
    }

    const drawClassChart = (data) => {
      const canvas = document.getElementById('classChart')
      if (!canvas || !data.length) return
      const ctx = canvas.getContext('2d')
      new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: data.map(d => d.class_name),
          datasets: [{
            data: data.map(d => d.count),
            backgroundColor: ['#ef4444','#f97316','#eab308','#22c55e','#14b8a6','#3b82f6','#8b5cf6','#ec4899','#f43f5e','#6366f1'],
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: { position: 'right', labels: { color: '#94a3b8', boxWidth: 12, padding: 8 } }
          }
        }
      })
    }

    onMounted(() => {
      loadStats()
      loadRecent()
    })

    return { stats, recentRecords, recentLoading, avgPerDay, formatDate }
  }
}
