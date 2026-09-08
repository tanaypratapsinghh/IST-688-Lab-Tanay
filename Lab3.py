import streamlit as st
from openai import OpenAI

MODEL = "gpt-4.1-mini"

MAX_USER_TURNS = 2

SYSTEM_PROMPT = (
    "You are a friendly helper talking to a 10-year-old. "
    "Explain everything in simple words and short sentences. Avoid jargon, "
    "and if you have to use a hard word, explain what it means.\n\n"
    "After you answer a question, always end your message with exactly this "
    "question: Do you want more info?\n\n"
    "If the user says yes (or anything that means yes), give more information "
    "about the same topic, then end with: Do you want more info?\n\n"
    "If the user says no (or anything that means no), do not give more "
    "information. Say you are happy to help, then ask what else they would "
    "like to know about. Do not ask 'Do you want more info?' in that case."
)


def buffered(messages, max_user_turns=MAX_USER_TURNS):
    """Return only the last N user messages and everything after them."""
    user_positions = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_positions) <= max_user_turns:
        return messages
    return messages[user_positions[-max_user_turns]:]


st.title("Lab 3: Streaming Chatbot with Memory")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Session state is shared across pages, so use a key specific to this page.
if "lab3_messages" not in st.session_state:
    st.session_state.lab3_messages = [
        {"role": "assistant", "content": "Hi! What would you like to know about?"}
    ]

# Show the whole conversation, even though only part of it is sent to the LLM.
for message in st.session_state.lab3_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything"):
    st.session_state.lab3_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    request_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    request_messages += buffered(st.session_state.lab3_messages)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=request_messages,
            stream=True,
        )
        answer = st.write_stream(stream)

    st.session_state.lab3_messages.append({"role": "assistant", "content": answer})