import asyncio
import os
import time

import pandas as pd
import requests
import streamlit as st
from agents import Agent, ModelSettings, Runner, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

MODEL = "gpt-5-mini"
DEFAULT_LOCATION = "Syracuse, NY"


# ---------- Part B: my weather function from Lab 5, unchanged ----------
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

    # Rest of today, in three hour steps.
    hourly = []
    for block in today["hourly"]:
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
        "high_today": float(today["maxtempF"]),
        "low_today": float(today["mintempF"]),
        "total_snow_cm": float(today["totalSnow_cm"]),
        "sunrise": astronomy["sunrise"],
        "sunset": astronomy["sunset"],
        "hourly_forecast": hourly,
    }


@function_tool
def get_weather(location: str = DEFAULT_LOCATION) -> dict:
    """Get the current weather and today's forecast for one location.

    Returns the matched location name, current temperature and feels-like
    temperature in Fahrenheit, conditions, humidity, wind, UV index, today's
    high and low, sunrise and sunset, and a forecast for the rest of the day in
    three-hour steps with chances of rain and snow.

    Accepts a city ("Paris" or "Syracuse, NY"), a zip code, an airport code
    ("SYR"), or a landmark ("Eiffel Tower"). Defaults to Syracuse, NY.
    """
    return get_current_weather(location)


# ---------- Part C: three specialists and a manager ----------
weather_agent = Agent(
    name="Weather agent",
    instructions=(
        "You report the weather. Call get_weather for the location you are given. "
        "Then summarize the facts in a few short lines: the location the tool actually "
        "matched (say so if it differs from what was asked), current temperature and "
        "feels-like, conditions, today's high and low, how the temperature and chance of "
        "rain or snow change through the day, wind, UV index, and sunset. "
        "Report facts only. Give no clothing or activity advice. "
        "If the tool can't find the location, say that plainly."
    ),
    tools=[get_weather],
    model=MODEL,
)

stylist_agent = Agent(
    name="Stylist agent",
    instructions=(
        "You are a stylist. You receive a weather report for today. Recommend what to "
        "wear: layers, outerwear, footwear, and accessories such as an umbrella, "
        "sunglasses, or gloves. Account for how the weather changes from morning to "
        "evening, and mention the feels-like temperature when it differs from the actual "
        "one. Use only the weather you are given. Keep it to a short list."
    ),
    model=MODEL,
)

activity_agent = Agent(
    name="Activity agent",
    instructions=(
        "You plan outdoor activities. You receive a weather report for today. Suggest two "
        "or three outdoor activities that suit the conditions, each with a time of day and "
        "a one-line reason tied to the forecast. If it's a bad day to be outside, say so "
        "and name the best window or an indoor alternative. Use only the weather you are given."
    ),
    model=MODEL,
)

manager_agent = Agent(
    name="Manager",
    instructions=(
        "You coordinate a team that tells the user what to wear and do today.\n"
        "1. Call weather_report first, with the user's location.\n"
        "2. Then send that weather report to stylist_advice and to activity_ideas.\n"
        "3. Combine their answers into one reply: a one-line weather summary, then "
        "'What to wear', then 'What to do'.\n"
        "Don't add advice the specialists didn't give. If weather_report says the location "
        "couldn't be found, tell the user and don't call the other two tools."
    ),
    tools=[
        weather_agent.as_tool(
            tool_name="weather_report",
            tool_description=(
                "Gets today's weather for a location and returns a factual summary: "
                "temperatures, conditions, rain and snow chances through the day, wind, "
                "UV, and sunset. Input: the location name."
            ),
        ),
        stylist_agent.as_tool(
            tool_name="stylist_advice",
            tool_description=(
                "Recommends what to wear today. Input: the weather report from "
                "weather_report."
            ),
        ),
        activity_agent.as_tool(
            tool_name="activity_ideas",
            tool_description=(
                "Suggests outdoor activities for today, with times of day. Input: the "
                "weather report from weather_report."
            ),
        ),
    ],
    model=MODEL,
)


# ---------- Part E: one agent doing the whole job ----------
single_agent = Agent(
    name="Single agent",
    instructions=(
        "You are a 'what to wear' assistant. When the user names a place, call get_weather "
        "for it, then tell them two things. "
        "1. What to wear today: layers, outerwear, footwear, and accessories such as an "
        "umbrella, sunglasses, or gloves. Account for how the temperature changes between "
        "morning, afternoon, and evening, and mention the feels-like temperature when it "
        "differs from the actual one. "
        "2. Two or three outdoor activities that suit the weather, each with a time of day "
        "and a short reason tied to the forecast. "
        "Lead with a one-line summary of the current conditions. If the location can't be "
        "found, say so."
    ),
    tools=[get_weather],
    model=MODEL,
)


# ---------- Bonus: triage with handoffs ----------
packing_agent = Agent(
    name="Packing agent",
    handoff_description="Builds a packing list for a trip to another city.",
    instructions=prompt_with_handoff_instructions(
        "You have taken over from the triage agent and answer the user yourself. "
        "Always call get_weather for the destination first, then write a packing list."),
    tools=[get_weather],
    model_settings=ModelSettings(tool_choice="required"),  # call a tool first
    model=MODEL,
)

outfit_team = manager_agent.clone(name="Outfit team",
                                  handoff_description="What to wear and do today.")

triage_agent = Agent(
    name="Triage agent",
    instructions="Hand off trips to the Packing agent and "
                 "today's outfits to the Outfit team.",
    handoffs=[packing_agent, outfit_team],
    model=MODEL,
)


# ---------- Part C, step 2: one event loop for the whole app ----------
@st.cache_resource
def get_event_loop():
    return asyncio.new_event_loop()


def run(agent, prompt):
    start = time.perf_counter()
    result = get_event_loop().run_until_complete(Runner.run(agent, prompt))
    return result, time.perf_counter() - start


# ---------- Page ----------
st.title("Lab: The \"What to Wear\" Crew")
st.write(
    "The same bot as Lab 5, split into a team: a weather agent, a stylist, and an "
    "activity planner, coordinated by a manager. Switch to the single agent to compare."
)

MODES = {
    "Team (manager + 3 agents)": manager_agent,
    "Single agent": single_agent,
    "Triage with handoffs (bonus)": triage_agent,
}
mode = st.radio("Run with", list(MODES), horizontal=True)

if "lab_ma_runs" not in st.session_state:
    st.session_state.lab_ma_runs = []

with st.form("crew"):
    if mode.startswith("Triage"):
        text = st.text_input(
            "Question", value="I'm flying to Lima, Peru tomorrow for 3 days. What should I pack?")
    else:
        text = st.text_input("City", placeholder=DEFAULT_LOCATION)
    submitted = st.form_submit_button("Get advice")

if submitted:
    if mode.startswith("Triage"):
        prompt = text.strip() or "I'm flying to Lima, Peru tomorrow for 3 days. What should I pack?"
        label = prompt
    else:
        location = text.strip() or DEFAULT_LOCATION
        prompt = f"What should I wear and do today in {location}?"
        label = location

    try:
        with st.spinner("The team is working. This usually takes 20 to 40 seconds..."):
            result, seconds = run(MODES[mode], prompt)
    except Exception as error:
        st.error(f"The run failed: {error}")
        st.stop()

    with st.chat_message("assistant"):
        st.write(result.final_output)

    # ---------- Part D: behind the scenes ----------
    usage = result.context_wrapper.usage
    with st.expander("Behind the scenes"):
        st.write(f"Model requests: {usage.requests} · Tokens: {usage.total_tokens:,} · "
                 f"Time: {seconds:.1f}s · Answered by: {result.last_agent.name}")
        for item in result.new_items:
            if item.type == "tool_call_item":
                st.write(f"{item.agent.name} called {item.raw_item.name}: "
                         f"{item.raw_item.arguments}")
            elif item.type == "tool_call_output_item":
                st.write(f"result: {str(item.output)[:300]}")
            elif item.type == "handoff_output_item":
                st.write(f"handoff: {item.source_agent.name} -> {item.target_agent.name}")

    # Part E: keep a log of every run, so the comparison table fills itself in.
    st.session_state.lab_ma_runs.append({
        "Mode": mode.split(" (")[0],
        "Input": label,
        "Model requests": usage.requests,
        "Total tokens": usage.total_tokens,
        "Seconds": round(seconds, 1),
        "Answered by": result.last_agent.name,
    })

if st.session_state.lab_ma_runs:
    st.subheader("Runs so far")
    st.dataframe(pd.DataFrame(st.session_state.lab_ma_runs), hide_index=True)
    if st.button("Clear runs"):
        st.session_state.lab_ma_runs = []
        st.rerun()


# ---------- Part D, step 2: what I saw behind the scenes ----------
# CHECK THESE AGAINST YOUR OWN "Behind the scenes" OUTPUT AND EDIT IF YOURS DIFFERS.
#
# a. Which tools did the manager call, and in what order?
#    weather_report first, with the city as its input. Then stylist_advice and
#    activity_ideas, which both take the weather report, so they come after it.
#    (The manager may request those two in the same turn.)
#
# b. What did the manager send to the stylist: the full forecast, or a summary?
#    Who wrote that text?
#    A summary. The full forecast from get_weather only ever reaches the weather
#    agent, which turns it into a short report. That report comes back to the
#    manager as the result of weather_report, and the manager's model then writes
#    the input it sends to stylist_advice. So the stylist reads text written by the
#    manager, based on text written by the weather agent, two steps away from the
#    raw data.
#
# c. "Model requests" is higher than the number of tool calls. Where did the other
#    model calls happen?
#    Every agent turn is a model request, and the trace only shows tool calls. The
#    manager makes a request to decide which tools to call and another to write the
#    final combined answer. Each specialist inside a tool also runs its own model
#    request: the weather agent makes two (one to call get_weather, one to write its
#    summary), and the stylist and activity agents make one each. Those inner calls,
#    including the weather agent's own get_weather call, don't appear in the
#    manager's list of items.
