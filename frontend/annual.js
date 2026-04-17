/**
 * WeatherAI — Annual View Page
 * Mini-map, 12-month charts, monthly data table
 */

let map, marker;
let annualChart = null, annualChart2 = null;

document.addEventListener('DOMContentLoaded', () => {
    initAnnualPage();
});

function initAnnualPage() {
    map = createMap('map', { center: [20, 0], zoom: 2, minZoom: 2, maxZoom: 10 });

    map.on('click', (e) => {
        selectAnnualLocation(e.latlng.lat, e.latlng.lng);
    });

    initSearchFeature((lat, lon, name) => {
        selectAnnualLocation(lat, lon, name);
    });

    window.switchMapTiles = function(theme) {
        if (!map) return;
        if (theme === 'light') {
            if (map.hasLayer(map._darkTiles)) map.removeLayer(map._darkTiles);
            map._lightTiles.addTo(map);
        } else {
            if (map.hasLayer(map._lightTiles)) map.removeLayer(map._lightTiles);
            map._darkTiles.addTo(map);
        }
    };

    // Auto-load saved location from Forecast page
    const saved = loadSavedLocation();
    if (saved) {
        selectAnnualLocation(saved.lat, saved.lon, saved.name);
    }
}

function selectAnnualLocation(lat, lon, placeName) {
    if (marker) { marker.setLatLng([lat, lon]); }
    else { marker = L.marker([lat, lon], { icon: createMarkerIcon() }).addTo(map); }

    const placeEl = document.getElementById('place-display');
    const coordsEl = document.getElementById('coords-display');

    if (placeName) {
        placeEl.textContent = placeName;
        saveLocation(lat, lon, placeName);
    } else {
        reverseGeocode(lat, lon, (name) => {
            placeEl.textContent = name;
            saveLocation(lat, lon, name);
        });
    }
    coordsEl.textContent = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;

    map.flyTo([lat, lon], Math.max(map.getZoom(), 4), { duration: 1 });
    fetchAnnualForecast(lat, lon);
}

async function fetchAnnualForecast(lat, lon) {
    try {
        const res = await fetch(`${API_BASE}/api/forecast?lat=${lat}&lon=${lon}`);
        const data = await res.json();
        if (data.error) { console.error(data.error); return; }
        displayAnnualCharts(data.forecasts);
        displayMonthlyTable(data.forecasts);
    } catch (err) {
        console.error('Forecast error:', err);
    }
}

function displayAnnualCharts(forecasts) {
    document.getElementById('annual-container').classList.remove('hidden');
    document.getElementById('annual-placeholder').style.display = 'none';

    const labels = forecasts.map(f => f.month_name);
    const temps = forecasts.map(f => f.temperature);
    const humidity = forecasts.map(f => f.humidity);
    const wind = forecasts.map(f => f.wind_speed);
    const rain = forecasts.map(f => f.rain_probability);

    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
    const textColor = isDark ? '#94a3b8' : '#475569';

    if (annualChart) annualChart.destroy();
    const ctx1 = document.getElementById('annual-chart').getContext('2d');
    annualChart = new Chart(ctx1, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Temperature (°C)', data: temps,
                    borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)',
                    fill: true, tension: 0.4,
                    pointBackgroundColor: '#ef4444', pointBorderColor: '#fff', pointBorderWidth: 2, pointRadius: 5, pointHoverRadius: 7
                },
                {
                    label: 'Humidity (%)', data: humidity,
                    borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)',
                    fill: true, tension: 0.4,
                    pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff', pointBorderWidth: 2, pointRadius: 5, pointHoverRadius: 7
                }
            ]
        },
        options: {
            responsive: true, interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { labels: { color: textColor, font: { family: 'Inter' }, usePointStyle: true } } },
            scales: { x: { grid: { color: gridColor }, ticks: { color: textColor } }, y: { grid: { color: gridColor }, ticks: { color: textColor } } }
        }
    });

    if (annualChart2) annualChart2.destroy();
    const ctx2 = document.getElementById('annual-chart-2').getContext('2d');
    annualChart2 = new Chart(ctx2, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Rain Probability (%)', data: rain,
                    backgroundColor: 'rgba(6,182,212,0.4)', borderColor: '#06b6d4', borderWidth: 2, borderRadius: 6, order: 1
                },
                {
                    label: 'Wind Speed (km/h)', data: wind, type: 'line',
                    borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)',
                    fill: true, tension: 0.4,
                    pointBackgroundColor: '#10b981', pointBorderColor: '#fff', pointBorderWidth: 2, pointRadius: 5, pointHoverRadius: 7, order: 0
                }
            ]
        },
        options: {
            responsive: true, interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { labels: { color: textColor, font: { family: 'Inter' }, usePointStyle: true } } },
            scales: { x: { grid: { color: gridColor }, ticks: { color: textColor } }, y: { grid: { color: gridColor }, ticks: { color: textColor } } }
        }
    });
}

function displayMonthlyTable(forecasts) {
    const wrap = document.getElementById('monthly-table-wrap');
    const tbody = document.getElementById('monthly-tbody');
    wrap.classList.remove('hidden');

    tbody.innerHTML = forecasts.map(f => `
        <tr>
            <td>${f.month_name}</td>
            <td>${f.temperature}°C</td>
            <td>${f.humidity}%</td>
            <td>${f.wind_speed} km/h</td>
            <td>${f.rain_probability}%</td>
            <td>${f.rainfall_mm || 0} mm</td>
        </tr>
    `).join('');
}
