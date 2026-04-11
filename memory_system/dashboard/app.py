"""Lightweight analytics dashboard for MemorySystem."""

from typing import Any

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MemorySystem Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #0f172a; color: #e2e8f0; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { font-size: 24px; margin-bottom: 20px; color: #38bdf8; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
        .card h3 { font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        .card .value { font-size: 32px; font-weight: 700; color: #f1f5f9; }
        .card .value.green { color: #4ade80; }
        .card .value.blue { color: #38bdf8; }
        .chart-container { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; margin-bottom: 24px; }
        .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }
        th { color: #94a3b8; font-weight: 600; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
        .badge.keyword { background: #166534; color: #4ade80; }
        .badge.embedding { background: #1e3a5f; color: #38bdf8; }
        .badge.llm { background: #713f12; color: #fbbf24; }
        .badge.fallback { background: #44403c; color: #a8a29e; }
        @media (max-width: 768px) { .chart-row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>MemorySystem Dashboard</h1>

        <div class="grid" id="stats-grid"></div>

        <div class="chart-row">
            <div class="chart-container">
                <h3 style="color:#94a3b8;margin-bottom:12px">Intent Distribution</h3>
                <canvas id="intentChart" height="200"></canvas>
            </div>
            <div class="chart-container">
                <h3 style="color:#94a3b8;margin-bottom:12px">Prediction Method</h3>
                <canvas id="methodChart" height="200"></canvas>
            </div>
        </div>

        <div class="chart-container">
            <h3 style="color:#94a3b8;margin-bottom:12px">Avg Latency by Stage (ms)</h3>
            <canvas id="latencyChart" height="120"></canvas>
        </div>

        <div class="chart-container">
            <h3 style="color:#94a3b8;margin-bottom:12px">Recent Requests</h3>
            <table>
                <thead><tr><th>#</th><th>Intent</th><th>Method</th><th>Confidence</th><th>Smart Tokens</th><th>Reduction</th><th>Total Latency</th></tr></thead>
                <tbody id="recent-table"></tbody>
            </table>
        </div>
    </div>

    <script>
        const COLORS = ['#38bdf8','#4ade80','#fbbf24','#f87171','#a78bfa','#fb923c','#2dd4bf','#e879f9'];
        let intentChart, methodChart, latencyChart;

        async function fetchData() {
            const resp = await fetch('/dashboard/api/analytics');
            return resp.json();
        }

        function renderStats(data) {
            document.getElementById('stats-grid').innerHTML = `
                <div class="card"><h3>Total Requests</h3><div class="value blue">${data.total_requests}</div></div>
                <div class="card"><h3>Avg Reduction</h3><div class="value green">${data.avg_reduction_percent}%</div></div>
                <div class="card"><h3>Avg Smart Tokens</h3><div class="value">${Math.round(data.avg_smart_tokens)}</div></div>
                <div class="card"><h3>Cache Hit Rate</h3><div class="value blue">${(data.cache_hit_rate * 100).toFixed(1)}%</div></div>
            `;
        }

        function renderCharts(data) {
            const intentLabels = Object.keys(data.intent_distribution);
            const intentValues = Object.values(data.intent_distribution);
            const methodLabels = Object.keys(data.prediction_method_distribution);
            const methodValues = Object.values(data.prediction_method_distribution);
            const latencyLabels = Object.keys(data.avg_latency_ms);
            const latencyValues = Object.values(data.avg_latency_ms);

            if (intentChart) intentChart.destroy();
            if (methodChart) methodChart.destroy();
            if (latencyChart) latencyChart.destroy();

            intentChart = new Chart(document.getElementById('intentChart'), {
                type: 'doughnut',
                data: { labels: intentLabels, datasets: [{ data: intentValues, backgroundColor: COLORS }] },
                options: { plugins: { legend: { labels: { color: '#94a3b8' } } } }
            });

            methodChart = new Chart(document.getElementById('methodChart'), {
                type: 'doughnut',
                data: { labels: methodLabels, datasets: [{ data: methodValues, backgroundColor: COLORS.slice(2) }] },
                options: { plugins: { legend: { labels: { color: '#94a3b8' } } } }
            });

            latencyChart = new Chart(document.getElementById('latencyChart'), {
                type: 'bar',
                data: { labels: latencyLabels, datasets: [{ label: 'ms', data: latencyValues, backgroundColor: '#38bdf8' }] },
                options: { indexAxis: 'y', scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' } } }, plugins: { legend: { display: false } } }
            });
        }

        function renderTable(data) {
            const rows = (data.recent_requests || []).slice(-20).reverse().map((r, i) => `
                <tr>
                    <td>${i + 1}</td>
                    <td>${r.intent}</td>
                    <td><span class="badge ${r.method}">${r.method}</span></td>
                    <td>${(r.confidence * 100).toFixed(0)}%</td>
                    <td>${r.token_estimate}</td>
                    <td style="color:#4ade80">${r.reduction_percent.toFixed(1)}%</td>
                    <td>${(r.latency_ms.total_ms || 0).toFixed(0)}ms</td>
                </tr>`).join('');
            document.getElementById('recent-table').innerHTML = rows || '<tr><td colspan="7" style="text-align:center;color:#64748b">No requests yet</td></tr>';
        }

        async function refresh() {
            const data = await fetchData();
            renderStats(data);
            renderCharts(data);
            renderTable(data);
        }

        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>
"""


def create_dashboard(ctx: Any):
    """
    Create a FastAPI dashboard app connected to a MemorySystem instance.

    Usage:
        from memory_system import MemorySystem
        from memory_system.dashboard import create_dashboard

        ctx = MemorySystem.from_yaml("bot.yaml")
        app = create_dashboard(ctx)

        # Run with: uvicorn module:app
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError:
        raise ImportError(
            "Dashboard requires FastAPI. "
            "Install with: pip install memory_system[dashboard]"
        )

    app = FastAPI(title="MemorySystem Dashboard")

    @app.get("/dashboard/", response_class=HTMLResponse)
    async def dashboard_page():
        return DASHBOARD_HTML

    @app.get("/dashboard/api/analytics")
    async def dashboard_analytics():
        if ctx.analytics:
            return JSONResponse(content=ctx.analytics.export())
        return JSONResponse(content={})

    @app.get("/dashboard/api/config")
    async def dashboard_config():
        return JSONResponse(content=ctx.config.model_dump())

    return app
