/**
 * WeatherAI — API Docs Page
 * Live API testers
 */

document.addEventListener('DOMContentLoaded', () => {
    // Set base URL display
    const baseUrlEl = document.getElementById('base-url');
    if (baseUrlEl) baseUrlEl.textContent = window.location.origin;

    // Predict tester
    const predictBtn = document.getElementById('test-predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', async () => {
            const lat = document.getElementById('test-lat').value;
            const lon = document.getElementById('test-lon').value;
            const month = document.getElementById('test-month').value;
            const resultEl = document.getElementById('test-predict-result');

            resultEl.innerHTML = '<code>Loading...</code>';

            try {
                const res = await fetch(`${API_BASE}/api/predict?lat=${lat}&lon=${lon}&month=${month}`);
                const data = await res.json();
                resultEl.innerHTML = `<code>${JSON.stringify(data, null, 2)}</code>`;
            } catch (err) {
                resultEl.innerHTML = `<code style="color:#ef4444;">Error: ${err.message}</code>`;
            }
        });
    }

    // Model info tester
    const modelBtn = document.getElementById('test-model-btn');
    if (modelBtn) {
        modelBtn.addEventListener('click', async () => {
            const resultEl = document.getElementById('test-model-result');
            resultEl.innerHTML = '<code>Loading...</code>';

            try {
                const res = await fetch(`${API_BASE}/api/model-info`);
                const data = await res.json();
                resultEl.innerHTML = `<code>${JSON.stringify(data, null, 2)}</code>`;
            } catch (err) {
                resultEl.innerHTML = `<code style="color:#ef4444;">Error: ${err.message}</code>`;
            }
        });
    }
});
