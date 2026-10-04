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
        "&timezone=auto";

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
