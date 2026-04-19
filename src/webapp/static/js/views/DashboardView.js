// Dashboard View — Premium B&W Minimal
const { createApp, ref, computed, onMounted } = Vue

export default {
  template: `
    <div class="dash-root">
      <div class="dash-header">
        <div>
          <div class="page-eyebrow">概览</div>
          <h1 class="page-title">数据统计</h1>
        </div>
        <span class="header-note">最近7天</span>
      </div>

      <!-- Stats Row -->
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-label">总检测次数</div>
          <div class="stat-val">{{ stats.total_detections }}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">今日检测</div>
          <div class="stat-val stat-val--accent">{{ stats.detections_today }}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">日均检测</div>
          <div class="stat-val">{{ avgPerDay }}</div>
        </div>
      </div>

      <!-- Charts -->
      <div class="charts-row">
        <div class="panel chart-panel">
          <div class="panel-title">每日检测趋势</div>
          <canvas id="dailyChart" height="180"></canvas>
        </div>
        <div class="panel chart-panel">
          <div class="panel-title">缺陷类型分布</div>
          <canvas id="classChart" height="180"></canvas>
        </div>
      </div>

      <!-- Recent -->
      <div class="panel recent-panel">
        <div class="panel-title">最近检测记录</div>
        <div v-if="recentLoading" class="list-state">加载中...</div>
        <div v-else-if="recentRecords.length === 0" class="list-state">暂无数据</div>
        <div v-else class="recent-list">
          <div v-for="r in recentRecords" :key="r.id" class="recent-item">
            <div class="recent-info">
              <div class="recent-name">{{ r.filename }}</div>
              <div class="recent-meta">{{ r.model_name }} · {{ formatDate(r.created_at) }}</div>
            </div>
            <div :class="['recent-badge', r.num_detections > 0 ? 'badge--active' : '']">
              {{ r.num_detections }} 个
            </div>
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
      return Math.round(stats.value.total_detections / 7)
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
        setTimeout(() => {
          drawDailyChart(stats.value.by_day || [])
          drawClassChart(stats.value.by_class || [])
        }, 100)
      } catch (e) { console.error('Failed to load stats', e) }
    }

    const loadRecent = async () => {
      recentLoading.value = true
      try {
        const api = (await import('../api/client.js')).default
        const res = await api.get('/detections?limit=5')
        recentRecords.value = res.data.records
      } catch (e) { console.error('Failed to load recent', e) }
      finally { recentLoading.value = false }
    }

    const drawDailyChart = (data) => {
      const canvas = document.getElementById('dailyChart')
      if (!canvas || !data.length) return
      const ctx = canvas.getContext('2d')
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.map(d => {
            const datePart = d.date.split(' ')[0]
            const [year, month, day] = datePart.split('-')
            return `${parseInt(month)}/${parseInt(day)}`
          }),
          datasets: [{
            label: '检测数',
            data: data.map(d => d.count),
            borderColor: '#111111',
            borderWidth: 2,
            backgroundColor: 'transparent',
            fill: false,
            tension: 0.35,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: '#ffffff',
            pointBorderColor: '#111111',
            pointBorderWidth: 2,
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#111111',
              titleColor: '#f5f5f3',
              bodyColor: '#f5f5f3',
              padding: 10,
              cornerRadius: 6,
              callbacks: {
                title: (items) => data[items[0].dataIndex].date,
                label: (item) => `${item.raw} 次检测`,
              }
            }
          },
          scales: {
            x: {
              ticks: { color: '#6b7280', font: { size: 11 } },
              grid: { display: false },
              border: { display: false }
            },
            y: {
              ticks: { color: '#6b7280', font: { size: 11 } },
              grid: { color: '#e5e5e3', lineWidth: 1 },
              border: { display: false },
              beginAtZero: true,
              grace: '10%'
            }
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
            borderWidth: 0,
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: { position: 'right', labels: { color: '#94a3b8', boxWidth: 10, padding: 10 } }
          }
        }
      })
    }

    onMounted(() => { loadStats(); loadRecent() })

    return { stats, recentRecords, recentLoading, avgPerDay, formatDate }
  }
}
