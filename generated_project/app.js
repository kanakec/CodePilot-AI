const form = document.getElementById("weatherForm");
const cityInput = document.getElementById("cityInput");
const searchButton = document.getElementById("searchButton");

const statusElement = document.getElementById("status");
const weatherCard = document.getElementById("weatherCard");

const locationName = document.getElementById("locationName");
const temperature = document.getElementById("temperature");
const humidity = document.getElementById("humidity");
const windSpeed = document.getElementById("windSpeed");
const condition = document.getElementById("condition");
const conditionIcon = document.getElementById("conditionIcon");

function setStatus(message) {
    statusElement.textContent = message;
}

function weatherDescription(code) {
    const descriptions = {
        0: ["Clear sky", "☀"],
        1: ["Mainly clear", "🌤"],
        2: ["Partly cloudy", "⛅"],
        3: ["Overcast", "☁"],
        45: ["Fog", "🌫"],
        48: ["Depositing rime fog", "🌫"],
        51: ["Light drizzle", "🌦"],
        53: ["Moderate drizzle", "🌦"],
        55: ["Dense drizzle", "🌧"],
        61: ["Light rain", "🌦"],
        63: ["Moderate rain", "🌧"],
        65: ["Heavy rain", "🌧"],
        71: ["Light snow", "🌨"],
        73: ["Moderate snow", "🌨"],
        75: ["Heavy snow", "❄"],
        80: ["Rain showers", "🌦"],
        81: ["Rain showers", "🌧"],
        82: ["Heavy rain showers", "⛈"],
        95: ["Thunderstorm", "⛈"],
        96: ["Thunderstorm with hail", "⛈"],
        99: ["Thunderstorm with hail", "⛈"]
    };

    return descriptions[code] || ["Unknown conditions", "🌡"];
}

async function geocodeCity(city) {
    const url =
        "https://geocoding-api.open-meteo.com/v1/search" +
        `?name=${encodeURIComponent(city)}` +
        "&count=1&language=en&format=json";

    const response = await fetch(url);

    if (!response.ok) {
        throw new Error("Could not find the city.");
    }

    const data = await response.json();

    if (!data.results || !data.results.length) {
        throw new Error("City not found. Try another city.");
    }

    return data.results[0];
}

async function fetchWeather(latitude, longitude) {
    const url =
        "https://api.open-meteo.com/v1/forecast" +
        `?latitude=${latitude}` +
        `&longitude=${longitude}` +
        "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m" +
        "&daily=weather_code,temperature_2m_max&timezone=auto";

    const response = await fetch(url);

    if (!response.ok) {
        throw new Error("Weather service is unavailable.");
    }

    return response.json();
}

async function searchWeather(city) {
    setStatus("Searching for weather...");
    weatherCard.classList.add("hidden");
    searchButton.disabled = true;

    try {
        const location = await geocodeCity(city);

        const weather = await fetchWeather(
            location.latitude,
            location.longitude
        );

        const current = weather.current;

        const [description, icon] =
            weatherDescription(current.weather_code);

        locationName.textContent =
            `${location.name}, ${location.country}`;

        temperature.textContent =
            Math.round(current.temperature_2m);

        humidity.textContent =
            `${current.relative_humidity_2m}%`;

        windSpeed.textContent =
            `${Math.round(current.wind_speed_10m)} km/h`;

        condition.textContent = description;
        conditionIcon.textContent = icon;

        weatherCard.classList.remove("hidden");

        updateFiveDayForecast(weather);
        setStatus("Weather updated successfully.");
    } catch (error) {
        setStatus(
            error.message || "Something went wrong."
        );
    } finally {
        searchButton.disabled = false;
    }
}

form.addEventListener("submit", event => {
    event.preventDefault();

    const city = cityInput.value.trim();

    if (!city) {
        setStatus("Enter a city name.");
        cityInput.focus();
        return;
    }

    searchWeather(city);
});


// ========================================================
// 5-DAY FORECAST
// ========================================================

function renderForecast(forecastData) {
    const forecastElement = document.getElementById("forecast");

    if (!forecastElement) {
        return;
    }

    forecastElement.innerHTML = "";

    forecastData.slice(0, 5).forEach((day) => {
        const card = document.createElement("article");
        card.className = "forecast-card";

        const condition = day.condition || "Unknown";
        const date = day.date || "Forecast";
        const temperature = Number.isFinite(day.temperature)
            ? day.temperature
            : "--";

        card.innerHTML = `
            <h3>${date}</h3>
            <div class="forecast-icon" role="img" aria-label="${condition}">
                ${getForecastIcon(condition)}
            </div>
            <div class="forecast-temperature">${temperature}°</div>
            <div class="forecast-condition">${condition}</div>
        `;

        forecastElement.appendChild(card);
    });
}


function getForecastIcon(condition) {
    const value = String(condition || "").toLowerCase();

    if (value.includes("storm") || value.includes("thunder")) {
        return "⛈️";
    }
    if (value.includes("rain") || value.includes("drizzle")) {
        return "🌧️";
    }
    if (value.includes("snow") || value.includes("sleet")) {
        return "❄️";
    }
    if (value.includes("partly") || value.includes("mostly cloudy")) {
        return "⛅";
    }
    if (value.includes("cloud") || value.includes("overcast")) {
        return "☁️";
    }
    if (value.includes("fog") || value.includes("mist")) {
        return "🌫️";
    }
    return "☀️";
}

function buildFiveDayForecast(weather) {
    const dates = weather.daily?.time || [];
    const codes = weather.daily?.weather_code || [];
    const maxTemps = weather.daily?.temperature_2m_max || [];

    return dates.slice(0, 5).map((date, index) => {
        const code = codes[index];
        const descriptionResult =
            typeof weatherDescription === "function"
                ? weatherDescription(code)
                : ["Weather"];
        const condition = Array.isArray(descriptionResult)
            ? descriptionResult[0]
            : descriptionResult;

        return {
            date,
            temperature: Number.isFinite(maxTemps[index])
                ? Math.round(maxTemps[index])
                : null,
            condition,
        };
    });
}

function updateFiveDayForecast(weather) {
    if (!weather || !weather.daily) {
        return;
    }

    const forecastDays = buildFiveDayForecast(weather);
    renderForecast(forecastDays);
}
