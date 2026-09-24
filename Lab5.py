import json

import requests
import streamlit as st
from openai import OpenAI

MODEL = "gpt-4.1-mini"
DEFAULT_LOCATION = "Syracuse, NY"


# ---------- Part A: the weather function ----------
def get_current_weather(location):
    """Return current conditions plus today's forecast for a location.

    location can be a city, a zip code, an airport code ('SYR'),
    or a landmark ('Eiffel+Tower'). Units are hard coded to Fahrenheit.
    """
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        # unknown locations come back as plain text, not JSON
        raise Exception(f"Could not find a location named {location}")

    current = data["current_condition"][0]
    today = data["weather"][0]
    astronomy = today["astronomy"][0]

    # What wttr.in actually matched, which may differ from what was typed.
    area = data["nearest_area"][0]
    matched = ", ".join(
        part for part in (
            area["areaName"][0]["value"],
            area["region"][0]["value"],
            area["country"][0]["value"],
        ) if part
    )

    # Rest of today, in three hour steps. The bot needs this because advice
    # for a 45 degree morning differs from advice for a 70 degree afternoon.
    hourly = []
    for block in today["hourly"]:
        # 'time' is "0", "300", "600" ... so 900 means 9:00.
        hour = int(block["time"]) // 100
        hourly.append({
            "hour": f"{hour:02d}:00",
            "temperature": float(block["tempF"]),
            "feels_like": float(block["FeelsLikeF"]),
            "description": block["weatherDesc"][0]["value"],
            "chance_of_rain": int(block["chanceofrain"]),
            "chance_of_snow": int(block["chanceofsnow"]),
            "wind_mph": float(block["windspeedMiles"]),
        })

    return {
        "location": matched or location,
        # Conditions now
        "temperature": float(current["temp_F"]),
        "feels_like": float(current["FeelsLikeF"]),
        "description": current["weatherDesc"][0]["value"],
        "humidity": int(current["humidity"]),
        "wind_mph": float(current["windspeedMiles"]),
        "wind_direction": current["winddir16Point"],
        "cloud_cover": int(current["cloudcover"]),
        "uv_index": int(current["uvIndex"]),
        "precip_inches": float(current["precipInches"]),
        "visibility_miles": float(current["visibilityMiles"]),
        # Today overall
        "high_today": float(today["maxtempF"]),
        "low_today": float(today["mintempF"]),
        "total_snow_cm": float(today["totalSnow_cm"]),
        "sunrise": astronomy["sunrise"],
        "sunset": astronomy["sunset"],
        # How the day develops
        "hourly_forecast": hourly,
    }


# ---------- Part B: expose it to the LLM as a tool ----------
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get the current weather and today's hourly forecast for a "
            "location. Call this whenever answering requires knowing the "
            "weather."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "City name, zip code, airport code, or landmark. "
                        "For example 'Syracuse, NY', 'Paris', or 'SYR'."
                    ),
                },
            },
            # location is not required, so the model may omit it and we
            # fall back to the default.
            "required": [],
        },
    },
}]

SYSTEM_PROMPT = (
    "You are a 'what to wear' assistant. When the user names a place, look up "
    "the weather for it and then tell them two things:\n\n"
    "1. What to wear today. Be specific about layers, outerwear, footwear, and "
    "accessories such as an umbrella, sunglasses, or gloves. Account for how "
    "the temperature changes between morning, afternoon, and evening, and "
    "mention the 'feels like' temperature when it differs noticeably from the "
    "actual one.\n"
    "2. Outdoor activities that suit the weather, with a suggested time of day "
    "and a short reason tied to the forecast.\n\n"
    "Keep it practical and friendly, and lead with the current conditions so "
    "the reader knows what you based the advice on."
)


def advise(user_request):
    """Two-call pattern: let the model request weather, then answer with it."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]

    # First call: the model decides whether it needs the weather.
    first = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    reply = first.choices[0].message

    if not reply.tool_calls:
        # No weather needed, so the first answer stands.
        return reply.content, None

    messages.append(reply)
    weather = None

    for call in reply.tool_calls:
        arguments = json.loads(call.function.arguments or "{}")
        location = arguments.get("location") or DEFAULT_LOCATION
        try:
            weather = get_current_weather(location)
            result = json.dumps(weather)
        except Exception as error:
            result = json.dumps({"error": str(error)})

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result,
        })

    # Second call: the weather is now in the conversation, so ask for advice.
    messages.append({
        "role": "user",
        "content": (
            "Using that weather data, tell me what to wear today and suggest "
            "outdoor activities that suit these conditions."
        ),
    })

    second = client.chat.completions.create(model=MODEL, messages=messages)
    return second.choices[0].message.content, weather


# ---------- UI ----------
st.title("Lab 5: What to Wear Bot")
st.write(
    "Enter a city and get advice on what to wear today, plus outdoor "
    "activities that suit the weather. Weather data comes from wttr.in."
)

with st.form("what_to_wear"):
    location = st.text_input("City", placeholder=DEFAULT_LOCATION)
    submitted = st.form_submit_button("Get advice")

if submitted:
    request = location.strip() or DEFAULT_LOCATION
    with st.spinner(f"Checking the weather in {request}..."):
        advice, weather = advise(f"What should I wear today in {request}?")

    if weather:
        st.subheader(weather["location"])
        left, middle, right = st.columns(3)
        left.metric("Now", f"{weather['temperature']:.0f}°F",
                    f"feels like {weather['feels_like']:.0f}°F")
        middle.metric("Today", f"{weather['high_today']:.0f}°F",
                      f"low {weather['low_today']:.0f}°F")
        right.metric("Conditions", weather["description"])
        st.caption(
            f"Humidity {weather['humidity']}% | Wind {weather['wind_mph']:.0f} mph "
            f"{weather['wind_direction']} | UV {weather['uv_index']} | "
            f"Sunrise {weather['sunrise']} | Sunset {weather['sunset']}"
        )

    st.markdown(advice)

    if weather:
        with st.expander("Raw weather data returned to the LLM"):
            st.json(weather)