/**
 * WeatherAI — Shared utilities across all pages
 * Theme toggle, common helpers
 */

const API_BASE = window.location.origin;
const NOMINATIM_URL = 'https://nominatim.openstreetmap.org';

// ── Theme Toggle ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
});

function initThemeToggle() {
    const toggle = document.getElementById('theme-toggle');
    if (!toggle) return;
    const icon = toggle.querySelector('.theme-icon');

    const stored = localStorage.getItem('weatherai-theme');
    if (stored === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        if (icon) icon.textContent = '☀️';
    }

    toggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        if (current === 'light') {
            document.documentElement.removeAttribute('data-theme');
            if (icon) icon.textContent = '🌙';
            localStorage.setItem('weatherai-theme', 'dark');
            if (typeof switchMapTiles === 'function') switchMapTiles('dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            if (icon) icon.textContent = '☀️';
            localStorage.setItem('weatherai-theme', 'light');
            if (typeof switchMapTiles === 'function') switchMapTiles('light');
        }
    });
}

// ── Search Helper (used by forecast + annual pages) ───────────────
function initSearchFeature(onSelect) {
    const input = document.getElementById('location-search');
    const searchBtn = document.getElementById('search-btn');
    const results = document.getElementById('search-results');
    if (!input || !searchBtn || !results) return;

    let searchTimeout = null;

    input.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        const query = input.value.trim();
        if (query.length < 2) { results.classList.add('hidden'); return; }
        searchTimeout = setTimeout(() => searchLocation(query, results, onSelect), 400);
    });

    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const query = input.value.trim();
            if (query.length >= 2) searchLocation(query, results, onSelect);
        }
    });

    searchBtn.addEventListener('click', () => {
        const query = input.value.trim();
        if (query.length >= 2) searchLocation(query, results, onSelect);
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) results.classList.add('hidden');
    });
}

async function searchLocation(query, resultsEl, onSelect) {
    try {
        const res = await fetch(
            `${NOMINATIM_URL}/search?format=json&q=${encodeURIComponent(query)}&limit=5&accept-language=en`
        );
        const data = await res.json();

        if (data.length === 0) {
            resultsEl.innerHTML = '<div class="search-result-item">No results found</div>';
            resultsEl.classList.remove('hidden');
            return;
        }

        resultsEl.innerHTML = data.map(item => `
            <div class="search-result-item" 
                 data-lat="${item.lat}" data-lon="${item.lon}"
                 data-name="${item.display_name}">
                📍 ${item.display_name}
            </div>
        `).join('');

        resultsEl.querySelectorAll('.search-result-item').forEach(item => {
            item.addEventListener('click', () => {
                const lat = parseFloat(item.dataset.lat);
                const lon = parseFloat(item.dataset.lon);
                const name = item.dataset.name.split(',')[0];
                document.getElementById('location-search').value = name;
                resultsEl.classList.add('hidden');
                onSelect(lat, lon, name);
            });
        });

        resultsEl.classList.remove('hidden');
    } catch (err) {
        console.error('Search error:', err);
    }
}

async function reverseGeocode(lat, lon, callback) {
    try {
        const res = await fetch(
            `${NOMINATIM_URL}/reverse?format=json&lat=${lat}&lon=${lon}&zoom=8&accept-language=en`
        );
        const data = await res.json();
        const name = data.address?.city || data.address?.town || data.address?.state ||
            data.address?.country || `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
        callback(name);
    } catch {
        callback(`${lat.toFixed(2)}, ${lon.toFixed(2)}`);
    }
}

// ── Map initialization helper ─────────────────────────────────────
function createMap(elementId, options = {}) {
    const defaults = { center: [20, 0], zoom: 2, minZoom: 2, maxZoom: 12, zoomControl: true };
    const config = { ...defaults, ...options };
    const map = L.map(elementId, config);

    const darkTiles = L.tileLayer(
        'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        { attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>', subdomains: 'abcd', maxZoom: 19 }
    );
    const lightTiles = L.tileLayer(
        'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        { attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>', subdomains: 'abcd', maxZoom: 19 }
    );

    const currentTheme = document.documentElement.getAttribute('data-theme');
    (currentTheme === 'light' ? lightTiles : darkTiles).addTo(map);

    map._darkTiles = darkTiles;
    map._lightTiles = lightTiles;

    return map;
}

function switchMapTiles(theme) {
    // This will be overridden by page-specific code if needed
}

function createMarkerIcon() {
    return L.divIcon({
        className: 'custom-marker',
        html: `<div style="
            width: 24px; height: 24px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            border-radius: 50%;
            border: 3px solid white;
            box-shadow: 0 2px 10px rgba(59,130,246,0.5);
        "></div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12]
    });
}

// ── Location Persistence ──────────────────────────────────────────
// Saves & loads selected location across pages via localStorage

function saveLocation(lat, lon, name) {
    localStorage.setItem('weatherai-location', JSON.stringify({ lat, lon, name, ts: Date.now() }));
}

function loadSavedLocation() {
    try {
        const stored = localStorage.getItem('weatherai-location');
        if (!stored) return null;
        const loc = JSON.parse(stored);
        // Only use if saved in the last 30 minutes
        if (Date.now() - loc.ts > 30 * 60 * 1000) return null;
        return loc;
    } catch {
        return null;
    }
}

