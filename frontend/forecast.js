/**
 * WeatherAI — Forecast Page
 * Map interaction, predictions, result display
 */

let map, marker;
let selectedLat = null, selectedLon = null;
let selectedMonth = new Date().getMonth() + 1;

document.addEventListener('DOMContentLoaded', () => {
    initForecast();
});

function initForecast() {
    // Initialize map
    map = createMap('map');
    document.getElementById('map-loading').classList.remove('active');

    map.on('click', (e) => {
        setLocation(e.latlng.lat, e.latlng.lng);
    });

    // Month selector
    const buttons = document.querySelectorAll('#month-selector .month-btn');
    buttons.forEach(btn => {
        const month = parseInt(btn.dataset.month);
        if (month === selectedMonth) btn.classList.add('active');
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedMonth = month;
            if (selectedLat !== null) fetchPrediction();
        });
    });

    // Search
    initSearchFeature((lat, lon, name) => {
        setLocation(lat, lon, name);
    });

    // Override global switchMapTiles
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

    // Auto-load saved location from another page
    const saved = loadSavedLocation();
    if (saved) {
        setLocation(saved.lat, saved.lon, saved.name);
    }
}

function setLocation(lat, lon, placeName) {
    selectedLat = lat;
    selectedLon = lon;

    if (marker) {
        marker.setLatLng([lat, lon]);
    } else {
        marker = L.marker([lat, lon], { icon: createMarkerIcon() }).addTo(map);
    }

    document.getElementById('lat-display').textContent = lat.toFixed(4);
    document.getElementById('lon-display').textContent = lon.toFixed(4);

    if (placeName) {
        document.getElementById('place-display').textContent = placeName;
        saveLocation(lat, lon, placeName);
    } else {
        reverseGeocode(lat, lon, (name) => {
            document.getElementById('place-display').textContent = name;
            saveLocation(lat, lon, name);
        });
    }

    fetchPrediction();
    map.flyTo([lat, lon], Math.max(map.getZoom(), 5), { duration: 1.2 });
}

async function fetchPrediction() {
    if (selectedLat === null) return;

    const overlay = document.getElementById('map-loading');
    overlay.classList.add('active');

    try {
        const res = await fetch(
            `${API_BASE}/api/predict?lat=${selectedLat}&lon=${selectedLon}&month=${selectedMonth}`
        );
        const data = await res.json();
        if (data.error) { console.error(data.error); return; }
        displayResults(data);
    } catch (err) {
        console.error('Fetch error:', err);
    } finally {
        overlay.classList.remove('active');
    }
}

function displayResults(data) {
    const container = document.getElementById('results-container');
    container.classList.remove('hidden');

    const { predictions, condition, additional, month_name } = data;
    const placeName = document.getElementById('place-display').textContent;

    // Condition banner
    document.getElementById('condition-icon').textContent = condition.icon;
    document.getElementById('condition-name').textContent = condition.text;
    document.getElementById('condition-location').textContent = `${placeName} · ${month_name}`;
    document.getElementById('temp-big').textContent = predictions.temperature.value;

    // Cards
    animateValue('temp-value', predictions.temperature.value);
    document.getElementById('feels-like').textContent = `${predictions.temperature.feels_like}°C`;
    animateBar('temp-bar', mapRange(predictions.temperature.value, -30, 50, 0, 100));

    animateValue('rain-value', predictions.rain.probability);
    document.getElementById('rain-prediction').textContent = predictions.rain.prediction;
    animateBar('rain-bar', predictions.rain.probability);

    animateValue('humidity-value', predictions.humidity.value);
    document.getElementById('dew-point').textContent = `${additional.dew_point}°C`;
    animateBar('humidity-bar', predictions.humidity.value);

    animateValue('wind-value', predictions.wind_speed.value);
    document.getElementById('visibility').textContent = `${additional.visibility} km`;
    animateBar('wind-bar', mapRange(predictions.wind_speed.value, 0, 80, 0, 100));

    // Additional
    document.getElementById('uv-index').textContent = additional.uv_index;
    document.getElementById('visibility-metric').textContent = `${additional.visibility} km`;
    document.getElementById('dew-point-metric').textContent = `${additional.dew_point}°C`;
    document.getElementById('pressure').textContent = `${additional.pressure} hPa`;

    // Re-trigger card animations
    document.querySelectorAll('.prediction-card').forEach((card, i) => {
        card.style.animation = 'none';
        card.offsetHeight;
        card.style.animation = `fadeUp 0.5s ease-out ${0.1 + i * 0.1}s both`;
    });

    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function animateValue(elementId, targetValue) {
    const el = document.getElementById(elementId);
    const start = parseFloat(el.textContent) || 0;
    const duration = 800;
    const startTime = performance.now();
    function update(t) {
        const p = Math.min((t - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = (start + (targetValue - start) * eased).toFixed(1);
        if (p < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function animateBar(elementId, pct) {
    const bar = document.getElementById(elementId);
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = `${Math.max(0, Math.min(100, pct))}%`; }, 100);
}

function mapRange(v, iMin, iMax, oMin, oMax) {
    return ((v - iMin) * (oMax - oMin)) / (iMax - iMin) + oMin;
}
