// ── Live power chart setup ───────────────────────────────────────────────────
const powerLabels = [];
const powerData   = [];

const powerCtx = document.getElementById('powerChart').getContext('2d');
const powerChart = new Chart(powerCtx, {
  type: 'line',
  data: {
    labels: powerLabels,
    datasets: [{
      label: 'kW',
      data: powerData,
      borderColor: '#38bdf8',
      backgroundColor: 'rgba(56,189,248,0.1)',
      borderWidth: 2,
      tension: 0.4,
      fill: true,
      pointRadius: 3
    }]
  },
  options: {
    animation: false,
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
    },
    plugins: { legend: { labels: { color: '#94a3b8' } } }
  }
});

// ── Bill forecast bar chart ──────────────────────────────────────────────────
const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const billCtx = document.getElementById('billChart').getContext('2d');
const billChart = new Chart(billCtx, {
  type: 'bar',
  data: {
    labels: months,
    datasets: [{
      label: 'Forecasted kWh',
      data: Array.from({length: 12}, () => Math.round(180 + Math.random() * 80)),
      backgroundColor: 'rgba(52,211,153,0.6)',
      borderColor: '#34d399',
      borderWidth: 1
    }]
  },
  options: {
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
    },
    plugins: { legend: { labels: { color: '#94a3b8' } } }
  }
});

// ── Live data polling every 3 seconds ───────────────────────────────────────
function fetchLive() {
  fetch('/api/live')
    .then(r => r.json())
    .then(d => {
      document.getElementById('voltage').textContent = d.voltage + ' V';
      document.getElementById('current').textContent = d.current + ' A';
      document.getElementById('power').textContent   = d.power_kw + ' kW';
      document.getElementById('bill').textContent    = 'LKR ' + d.est_bill_lkr.toLocaleString();

      // Update live chart (keep last 20 points)
      if (powerLabels.length >= 20) { powerLabels.shift(); powerData.shift(); }
      powerLabels.push(d.timestamp);
      powerData.push(d.power_kw);
      powerChart.update();
    });
}

// ── Clock ────────────────────────────────────────────────────────────────────
function updateClock() {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
}

setInterval(fetchLive,    3000);
setInterval(updateClock,  1000);
fetchLive();
updateClock();

// ── Manual predictor ─────────────────────────────────────────────────────────
function runPredict() {
  const payload = {
    voltage:    parseFloat(document.getElementById('f_voltage').value),
    intensity:  parseFloat(document.getElementById('f_intensity').value),
    sub1:       parseFloat(document.getElementById('f_sub1').value),
    sub2:       parseFloat(document.getElementById('f_sub2').value),
    sub3:       parseFloat(document.getElementById('f_sub3').value),
    month:      parseInt(document.getElementById('f_month').value),
    day_of_week: new Date().getDay()
  };

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(r => r.json())
  .then(d => {
    document.getElementById('res_power').textContent = d.predicted_power_kw;
    document.getElementById('res_kwh').textContent   = d.predicted_kwh_month;
    document.getElementById('res_bill').textContent  = d.estimated_bill_lkr.toLocaleString();
    document.getElementById('result').style.display  = 'block';
  });
}