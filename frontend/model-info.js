/**
 * WeatherAI — Model Info Page
 * Loads and displays model metrics
 */

document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
});

async function loadModelInfo() {
    try {
        const res = await fetch(`${API_BASE}/api/model-info`);
        const data = await res.json();
        const metrics = data.metrics;

        // Temperature
        setEl('temp-r2', metrics.temperature.r2_score);
        setEl('temp-mae', `${metrics.temperature.mae}°C`);
        setEl('temp-rmse', `${metrics.temperature.rmse}°C`);

        // Humidity
        setEl('hum-r2', metrics.humidity.r2_score);
        setEl('hum-mae', `${metrics.humidity.mae}%`);
        setEl('hum-rmse', `${metrics.humidity.rmse}%`);

        // Wind
        setEl('wind-r2', metrics.wind_speed.r2_score);
        setEl('wind-mae', `${metrics.wind_speed.mae} km/h`);
        setEl('wind-rmse', `${metrics.wind_speed.rmse} km/h`);

        // Rain
        setEl('rain-acc', `${(metrics.rain.accuracy * 100).toFixed(1)}%`);
        setEl('rain-prec', `${(metrics.rain.precision * 100).toFixed(1)}%`);
        setEl('rain-f1', `${(metrics.rain.f1_score * 100).toFixed(1)}%`);

        // Training info
        if (data.training_info) {
            setEl('train-samples', data.training_info.train_samples.toLocaleString());
            setEl('test-samples', data.training_info.test_samples.toLocaleString());
        }
    } catch (err) {
        console.error('Failed to load model info:', err);
    }
}

function setEl(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}
