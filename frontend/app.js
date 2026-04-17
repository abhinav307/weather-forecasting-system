/**
 * WeatherAI — Frontend Application
 * Interactive map, API integration, and weather visualization
 */

// ── Configuration ─────────────────────────────────────────────────
const API_BASE = window.location.origin;
const NOMINATIM_URL = 'https://nominatim.openstreetmap.org';

// ── State ─────────────────────────────────────────────────────────
let map;
let marker;
let selectedLat = null;
let selectedLon = null;
let selectedMonth = new Date().getMonth() + 1;
let annualChart = null;
let annualChart2 = null;
let searchTimeout = null;

// ── Initialize ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initMonthSelector();
    initSearch();
    initThemeToggle();
    initNavScroll();
    loadModelInfo();
});

// ── Map Initialization ────────────────────────────────────────────
function initMap() {
    map = L.map('map', {
        center: [20, 0],
        zoom: 2,
        minZoom: 2,
        maxZoom: 12,
        zoomControl: true
    });

    // Dark tile layer
    const darkTiles = L.tileLayer(
        'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }
    );

    const lightTiles = L.tileLayer(
        'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }
    );

    // Use dark tiles by default
    const currentTheme = document.documentElement.getAttribute('data-theme');
    if (currentTheme === 'light') {
        lightTiles.addTo(map);
    } else {
        darkTiles.addTo(map);
    }

    // Store tile layers for theme switching
    map._darkTiles = darkTiles;
    map._lightTiles = lightTiles;

    // Click handler
    map.on('click', (e) => {
        const { lat, lng } = e.latlng;
        setLocation(lat, lng);
    });

    // Remove loading overlay
    document.getElementById('map-loading').classList.remove('active');
}

// ── Location Selection ────────────────────────────────────────────
function setLocation(lat, lon, placeName) {
    selectedLat = lat;
    selectedLon = lon;

    // Update/create marker
    const markerIcon = L.divIcon({
        className: 'custom-marker',
        html: `<div style="
            width: 24px; height: 24px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            border-radius: 50%;
            border: 3px solid white;
            box-shadow: 0 2px 10px rgba(59,130,246,0.5);
            animation: pulse 2s infinite;
        "></div>
        <style>
            @keyframes pulse {
                0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.5); }
                50% { box-shadow: 0 0 0 12px rgba(59,130,246,0); }
            }
        </style>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12]
    });

    if (marker) {
        marker.setLatLng([lat, lon]);
    } else {
        marker = L.marker([lat, lon], { icon: markerIcon }).addTo(map);
    }

    // Update display
    document.getElementById('lat-display').textContent = lat.toFixed(4);
    document.getElementById('lon-display').textContent = lon.toFixed(4);

    // Reverse geocode if no name provided
    if (placeName) {
        document.getElementById('place-display').textContent = placeName;
    } else {
        reverseGeocode(lat, lon);
    }

    // Fetch predictions
    fetchPrediction();
    fetchAnnualForecast();

    // Smooth pan
    map.flyTo([lat, lon], Math.max(map.getZoom(), 5), { duration: 1.2 });
}

async function reverseGeocode(lat, lon) {
    try {
        const res = await fetch(
            `${NOMINATIM_URL}/reverse?format=json&lat=${lat}&lon=${lon}&zoom=8&accept-language=en`
        );
        const data = await res.json();
        const name = data.address?.city || data.address?.town || data.address?.state ||
            data.address?.country || `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
        document.getElementById('place-display').textContent = name;
    } catch {
        document.getElementById('place-display').textContent =
            `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
    }
}

// ── Predictions ───────────────────────────────────────────────────
async function fetchPrediction() {
    if (selectedLat === null || selectedLon === null) return;

    const overlay = document.getElementById('map-loading');
    overlay.classList.add('active');

    try {
        const res = await fetch(
            `${API_BASE}/api/predict?lat=${selectedLat}&lon=${selectedLon}&month=${selectedMonth}`
        );
        const data = await res.json();

        if (data.error) {
            console.error('API Error:', data.error);
            return;
        }

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
    document.getElementById('condition-location').textContent =
        `${placeName} · ${month_name}`;
    document.getElementById('temp-big').textContent = predictions.temperature.value;

    // Temperature card
    animateValue('temp-value', predictions.temperature.value);
    document.getElementById('feels-like').textContent =
        `${predictions.temperature.feels_like}°C`;
    animateBar('temp-bar', mapRange(predictions.temperature.value, -30, 50, 0, 100));

    // Rain card
    animateValue('rain-value', predictions.rain.probability);
    const rainfallMm = predictions.rain.rainfall_mm || 0;
    document.getElementById('rain-prediction').textContent = `${rainfallMm} mm`;
    animateBar('rain-bar', predictions.rain.probability);

    // Humidity card
    animateValue('humidity-value', predictions.humidity.value);
    document.getElementById('dew-point').textContent = `${additional.dew_point}°C`;
    animateBar('humidity-bar', predictions.humidity.value);

    // Wind card
    animateValue('wind-value', predictions.wind_speed.value);
    document.getElementById('visibility').textContent = `${additional.visibility} km`;
    animateBar('wind-bar', mapRange(predictions.wind_speed.value, 0, 80, 0, 100));

    // Additional metrics
    document.getElementById('uv-index').textContent = additional.uv_index;
    document.getElementById('visibility-metric').textContent = `${additional.visibility} km`;
    document.getElementById('dew-point-metric').textContent = `${additional.dew_point}°C`;
    document.getElementById('pressure').textContent = `${additional.pressure} hPa`;

    // Re-trigger card animations
    document.querySelectorAll('.prediction-card').forEach((card, i) => {
        card.style.animation = 'none';
        card.offsetHeight; // trigger reflow
        card.style.animation = `fadeUp 0.5s ease-out ${0.1 + i * 0.1}s both`;
    });

    // Scroll to results
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function animateValue(elementId, targetValue) {
    const el = document.getElementById(elementId);
    const start = parseFloat(el.textContent) || 0;
    const duration = 800;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        const current = start + (targetValue - start) * eased;
        el.textContent = current.toFixed(1);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function animateBar(elementId, percentage) {
    const bar = document.getElementById(elementId);
    bar.style.width = '0%';
    setTimeout(() => {
        bar.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
    }, 100);
}

function mapRange(value, inMin, inMax, outMin, outMax) {
    return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
}

// ── Annual Forecast ───────────────────────────────────────────────
async function fetchAnnualForecast() {
    if (selectedLat === null || selectedLon === null) return;

    try {
        const res = await fetch(
            `${API_BASE}/api/forecast?lat=${selectedLat}&lon=${selectedLon}`
        );
        const data = await res.json();

        if (data.error) {
            console.error('API Error:', data.error);
            return;
        }

        displayAnnualCharts(data.forecasts);
    } catch (err) {
        console.error('Forecast fetch error:', err);
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

    // Chart 1: Temperature & Humidity
    if (annualChart) annualChart.destroy();
    const ctx1 = document.getElementById('annual-chart').getContext('2d');
    annualChart = new Chart(ctx1, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Temperature (°C)',
                    data: temps,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#ef4444',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7
                },
                {
                    label: 'Humidity (%)',
                    data: humidity,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#3b82f6',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { labels: { color: textColor, font: { family: 'Inter' }, usePointStyle: true } }
            },
            scales: {
                x: { grid: { color: gridColor }, ticks: { color: textColor } },
                y: { grid: { color: gridColor }, ticks: { color: textColor } }
            }
        }
    });

    // Chart 2: Wind & Rain
    if (annualChart2) annualChart2.destroy();
    const ctx2 = document.getElementById('annual-chart-2').getContext('2d');
    annualChart2 = new Chart(ctx2, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Rain Probability (%)',
                    data: rain,
                    backgroundColor: 'rgba(6, 182, 212, 0.4)',
                    borderColor: '#06b6d4',
                    borderWidth: 2,
                    borderRadius: 6,
                    order: 1
                },
                {
                    label: 'Wind Speed (km/h)',
                    data: wind,
                    type: 'line',
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#10b981',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    order: 0
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { labels: { color: textColor, font: { family: 'Inter' }, usePointStyle: true } }
            },
            scales: {
                x: { grid: { color: gridColor }, ticks: { color: textColor } },
                y: { grid: { color: gridColor }, ticks: { color: textColor } }
            }
        }
    });
}

// ── Month Selector ────────────────────────────────────────────────
function initMonthSelector() {
    const selector = document.getElementById('month-selector');
    const buttons = selector.querySelectorAll('.month-btn');

    // Set current month as active
    buttons.forEach(btn => {
        const month = parseInt(btn.dataset.month);
        if (month === selectedMonth) {
            btn.classList.add('active');
        }

        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedMonth = month;
            fetchPrediction();
        });
    });
}

// ── Search ────────────────────────────────────────────────────────
function initSearch() {
    const input = document.getElementById('location-search');
    const searchBtn = document.getElementById('search-btn');
    const results = document.getElementById('search-results');

    input.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        const query = input.value.trim();

        if (query.length < 2) {
            results.classList.add('hidden');
            return;
        }

        searchTimeout = setTimeout(() => searchLocation(query), 400);
    });

    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const query = input.value.trim();
            if (query.length >= 2) searchLocation(query);
        }
    });

    searchBtn.addEventListener('click', () => {
        const query = input.value.trim();
        if (query.length >= 2) searchLocation(query);
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) {
            results.classList.add('hidden');
        }
    });
}

async function searchLocation(query) {
    const results = document.getElementById('search-results');

    try {
        const res = await fetch(
            `${NOMINATIM_URL}/search?format=json&q=${encodeURIComponent(query)}&limit=5&accept-language=en`
        );
        const data = await res.json();

        if (data.length === 0) {
            results.innerHTML = '<div class="search-result-item">No results found</div>';
            results.classList.remove('hidden');
            return;
        }

        results.innerHTML = data.map(item => `
            <div class="search-result-item" 
                 data-lat="${item.lat}" 
                 data-lon="${item.lon}"
                 data-name="${item.display_name}">
                📍 ${item.display_name}
            </div>
        `).join('');

        results.querySelectorAll('.search-result-item').forEach(item => {
            item.addEventListener('click', () => {
                const lat = parseFloat(item.dataset.lat);
                const lon = parseFloat(item.dataset.lon);
                const name = item.dataset.name.split(',')[0];

                document.getElementById('location-search').value = name;
                results.classList.add('hidden');
                setLocation(lat, lon, name);
            });
        });

        results.classList.remove('hidden');
    } catch (err) {
        console.error('Search error:', err);
    }
}

// ── Model Info ────────────────────────────────────────────────────
async function loadModelInfo() {
    try {
        const res = await fetch(`${API_BASE}/api/model-info`);
        const data = await res.json();
        const metrics = data.metrics;

        // Temperature
        document.getElementById('temp-r2').textContent = metrics.temperature.r2_score;
        document.getElementById('temp-mae').textContent = `${metrics.temperature.mae}°C`;
        document.getElementById('temp-rmse').textContent = `${metrics.temperature.rmse}°C`;

        // Humidity
        document.getElementById('hum-r2').textContent = metrics.humidity.r2_score;
        document.getElementById('hum-mae').textContent = `${metrics.humidity.mae}%`;
        document.getElementById('hum-rmse').textContent = `${metrics.humidity.rmse}%`;

        // Wind
        document.getElementById('wind-r2').textContent = metrics.wind_speed.r2_score;
        document.getElementById('wind-mae').textContent = `${metrics.wind_speed.mae} km/h`;
        document.getElementById('wind-rmse').textContent = `${metrics.wind_speed.rmse} km/h`;

        // Rain
        document.getElementById('rain-acc').textContent = `${(metrics.rain.accuracy * 100).toFixed(1)}%`;
        document.getElementById('rain-prec').textContent = `${(metrics.rain.precision * 100).toFixed(1)}%`;
        document.getElementById('rain-f1').textContent = `${(metrics.rain.f1_score * 100).toFixed(1)}%`;

        // Training info
        if (data.training_info) {
            document.getElementById('train-samples').textContent =
                data.training_info.train_samples.toLocaleString();
            document.getElementById('test-samples').textContent =
                data.training_info.test_samples.toLocaleString();
        }
    } catch (err) {
        console.error('Failed to load model info:', err);
    }
}

// ── Theme Toggle ──────────────────────────────────────────────────
function initThemeToggle() {
    const toggle = document.getElementById('theme-toggle');
    const icon = toggle.querySelector('.theme-icon');

    // Check stored preference
    const stored = localStorage.getItem('weatherai-theme');
    if (stored === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        icon.textContent = '☀️';
    }

    toggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        if (current === 'light') {
            document.documentElement.removeAttribute('data-theme');
            icon.textContent = '🌙';
            localStorage.setItem('weatherai-theme', 'dark');
            switchMapTiles('dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            icon.textContent = '☀️';
            localStorage.setItem('weatherai-theme', 'light');
            switchMapTiles('light');
        }
    });
}

function switchMapTiles(theme) {
    if (!map) return;
    if (theme === 'light') {
        map.removeLayer(map._darkTiles);
        map._lightTiles.addTo(map);
    } else {
        map.removeLayer(map._lightTiles);
        map._darkTiles.addTo(map);
    }
}

// ── Nav Scroll Highlight ──────────────────────────────────────────
function initNavScroll() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    navLinks.forEach(link => link.classList.remove('active'));
                    const id = entry.target.getAttribute('id');
                    const activeLink = document.querySelector(`.nav-link[href="#${id}"]`);
                    if (activeLink) activeLink.classList.add('active');
                }
            });
        },
        { threshold: 0.3 }
    );

    sections.forEach(s => observer.observe(s));
}
