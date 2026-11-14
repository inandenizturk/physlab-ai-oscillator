import asyncio
import socket
import threading
import time
import webbrowser
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import numpy as np
from scipy.integrate import solve_ivp

app = FastAPI(title="PhysLab AI")

HTML_UI = """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>PhysLab AI | Nyquist & Debounce Kontrollü Simülasyon</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-slate-950 text-slate-100 p-6 font-sans">
    <div class="max-w-6xl mx-auto space-y-6">
        <div class="flex justify-between items-center border-b border-slate-800 pb-4">
            <h1 class="text-2xl font-bold text-cyan-400">PhysLab AI <span class="text-sm text-slate-400 font-normal">| Dinamik Süreli (7τ) & Anti-Aliasing Osilatör</span></h1>
            <span id="status" class="text-xs px-3 py-1 rounded-full bg-slate-800 text-slate-300">Yükleniyor...</span>
        </div>
        <div id="error-box" class="hidden bg-rose-950/80 border border-rose-500 text-rose-200 text-xs p-3 rounded-xl"></div>
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div class="lg:col-span-4 bg-slate-900 p-5 rounded-xl border border-slate-800 space-y-4">
                <div>
                    <label class="text-xs text-slate-400 flex justify-between"><span>Kütle m (kg)</span><span id="vm">1.0</span></label>
                    <input type="range" id="m" min="0.2" max="5" step="0.1" value="1.0" oninput="onSliderInput()" class="w-full accent-cyan-400">
                </div>
                <div>
                    <label class="text-xs text-slate-400 flex justify-between"><span>Yay Sabiti k (N/m)</span><span id="vk">4.0</span></label>
                    <input type="range" id="k" min="0.5" max="15" step="0.5" value="4.0" oninput="onSliderInput()" class="w-full accent-cyan-400">
                </div>
                <div>
                    <label class="text-xs text-slate-400 flex justify-between"><span>Sönümleme b (kg/s)</span><span id="vg">0.25</span></label>
                    <input type="range" id="g" min="0.05" max="2.5" step="0.05" value="0.25" oninput="onSliderInput()" class="w-full accent-cyan-400">
                </div>
                <div>
                    <label class="text-xs text-slate-400 flex justify-between"><span>Sürücü Frekans ω_d (rad/s)</span><span id="vw">1.8</span></label>
                    <input type="range" id="w" min="0.2" max="5" step="0.1" value="1.8" oninput="onSliderInput()" class="w-full accent-cyan-400">
                </div>
                <div id="stats" class="pt-3 border-t border-slate-800 text-xs space-y-1.5 text-cyan-300"></div>
            </div>
            <div class="lg:col-span-8 space-y-4">
                <div class="bg-slate-900 p-4 rounded-xl border border-slate-800">
                    <canvas id="steadyChart" height="120"></canvas>
                </div>
                <div class="bg-slate-900 p-4 rounded-xl border border-slate-800">
                    <canvas id="envelopeChart" height="100"></canvas>
                </div>
            </div>
        </div>
    </div>
    <script>
        let steadyChart = null;
        let envelopeChart = null;
        let debounceTimer = null;
        let activeController = null;

        function onSliderInput() {
            document.getElementById('vm').innerText = document.getElementById('m').value;
            document.getElementById('vk').innerText = document.getElementById('k').value;
            document.getElementById('vg').innerText = document.getElementById('g').value;
            document.getElementById('vw').innerText = document.getElementById('w').value;
            document.getElementById('status').innerText = 'Ayarlanıyor...';

            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(runSimulation, 200);
        }

        async function runSimulation() {
            if (activeController) activeController.abort();
            activeController = new AbortController();

            const errBox = document.getElementById('error-box');
            errBox.classList.add('hidden');

            const m = document.getElementById('m').value;
            const k = document.getElementById('k').value;
            const g = document.getElementById('g').value;
            const w = document.getElementById('w').value;
            document.getElementById('status').innerText = 'Hesaplanıyor...';

            try {
                if (typeof Chart === 'undefined') {
                    throw new Error('Chart.js yüklenemedi.');
                }
                const res = await fetch(`/sim?m=${m}&k=${k}&g=${g}&w=${w}`, { signal: activeController.signal });
                if (!res.ok) throw new Error(`API Hatası: ${res.status}`);
                const d = await res.json();
                document.getElementById('status').innerText = 'Doğrulandı ✓';

                document.getElementById('stats').innerHTML = `
                    <div>Doğal Frekans ω₀: <b>${d.w0} rad/s</b></div>
                    <div>Zaman Sabiti τ (2m/b): <b>${d.tau} s</b></div>
                    <div>Toplam Süre (7τ + 10T_d): <b>${d.t_end} s</b></div>
                    <div class="pt-1 border-t border-slate-800">Analitik Genlik A: <b>${d.analytic_amp} m</b></div>
                    <div>Sayısal Kararlı Genlik: <b>${d.numeric_amp} m</b></div>
                    <div>Bağıl Genlik Hatası: <b class="text-emerald-400">%${d.amp_err_pct}</b></div>
                    <div class="pt-1 border-t border-slate-800">Analitik Ort. Enerji ⟨E⟩: <b>${d.analytic_e} J</b></div>
                    <div>Sayısal Kararlı ⟨E⟩: <b>${d.numeric_e} J</b></div>
                `;

                if (steadyChart) steadyChart.destroy();
                steadyChart = new Chart(document.getElementById('steadyChart'), {
                    type: 'line',
                    data: {
                        labels: d.steady_t,
                        datasets: [{ label: 'Kararlı Hal Yakın Çekim x(t) [Son 5 Periyot]', data: d.steady_x, borderColor: '#22d3ee', borderWidth: 2, pointRadius: 0 }]
                    },
                    options: { animation: false }
                });

                if (envelopeChart) envelopeChart.destroy();
                envelopeChart = new Chart(document.getElementById('envelopeChart'), {
                    type: 'line',
                    data: {
                        labels: d.env_t,
                        datasets: [
                            { label: 'Genlik Zarfı max|x(t)| [Tüm Süre]', data: d.env_amp, borderColor: '#f43f5e', borderWidth: 2, pointRadius: 0 },
                            { label: 'Pencere Ort. Enerji ⟨E⟩ [J]', data: d.env_e, borderColor: '#34d399', borderWidth: 1.5, pointRadius: 0 }
                        ]
                    },
                    options: { animation: false }
                });
            } catch (err) {
                if (err.name !== 'AbortError') {
                    document.getElementById('status').innerText = 'Hata!';
                    errBox.innerText = err.message;
                    errBox.classList.remove('hidden');
                }
            }
        }

        window.addEventListener('load', runSimulation);
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_UI

@app.get("/sim")
def sim(m: float = 1.0, k: float = 4.0, g: float = 0.25, w: float = 1.8, f0: float = 1.0):
    tau = (2.0 * m) / g
    t_d = (2.0 * np.pi) / w
    t_transient = 7.0 * tau
    t_end = t_transient + 10.0 * t_d

    n_steps = max(1200, int((t_end / t_d) * 80))
    t = np.linspace(0.0, t_end, n_steps)

    sol = solve_ivp(
        lambda t, y: [y[1], (f0 * np.cos(w * t) - g * y[1] - k * y[0]) / m],
        [0.0, t_end],
        [0.0, 0.0],
        t_eval=t,
        rtol=1e-8,
        atol=1e-8,
    )
    x, v = sol.y
    e = 0.5 * k * x**2 + 0.5 * m * v**2

    steady_mask = t >= t_transient
    numeric_amp = float(np.max(np.abs(x[steady_mask])))
    numeric_e = float(np.mean(e[steady_mask]))

    analytic_amp = f0 / np.sqrt((k - m * w**2)**2 + (g * w)**2)
    analytic_e = 0.25 * (analytic_amp**2) * (k + m * w**2)
    amp_err_pct = abs(numeric_amp - analytic_amp) / analytic_amp * 100.0

    zoom_mask = t >= (t_end - 5.0 * t_d)
    steady_t = np.round(t[zoom_mask], 2).tolist()
    steady_x = np.round(x[zoom_mask], 4).tolist()

    chunks_t = np.array_split(t, 150)
    chunks_x = np.array_split(x, 150)
    chunks_e = np.array_split(e, 150)
    env_t = [round(float(c[len(c) // 2]), 1) for c in chunks_t if len(c) > 0]
    env_amp = [round(float(np.max(np.abs(c))), 4) for c in chunks_x if len(c) > 0]
    env_e = [round(float(np.mean(c)), 4) for c in chunks_e if len(c) > 0]

    return {
        "steady_t": steady_t,
        "steady_x": steady_x,
        "env_t": env_t,
        "env_amp": env_amp,
        "env_e": env_e,
        "w0": round(float(np.sqrt(k / m)), 3),
        "tau": round(float(tau), 2),
        "t_end": round(float(t_end), 2),
        "analytic_amp": round(float(analytic_amp), 4),
        "numeric_amp": round(numeric_amp, 4),
        "amp_err_pct": round(float(amp_err_pct), 3),
        "analytic_e": round(float(analytic_e), 4),
        "numeric_e": round(numeric_e, 4),
    }

def find_free_port(default_port: int = 8000) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", default_port)) != 0:
            return default_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

if __name__ == "__main__":
    port = find_free_port(8000)
    threading.Thread(
        target=lambda: (time.sleep(1.2), webbrowser.open(f"http://127.0.0.1:{port}")),
        daemon=True,
    ).start()
    try:
        asyncio.get_running_loop()
        threading.Thread(
            target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="error"),
            daemon=True,
        ).start()
    except RuntimeError:
        uvicorn.run(app, host="127.0.0.1", port=port)
