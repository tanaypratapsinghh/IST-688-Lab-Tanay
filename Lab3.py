import streamlit as st
import tiktoken
from openai import OpenAI

MODEL = "gpt-4.1-mini"

# Part B item 3: keep only the last two user messages and the replies to them.
MAX_USER_TURNS = 2

# Part B item 4: cap the total tokens sent to the LLM per request.
MAX_TOKENS = 1500

# Which buffer to use: "turns" for item 3, "tokens" for item 4.
BUFFER_MODE = "tokens"

# Part C: the system prompt drives the answer / "more info?" loop.
# It is added at request time, so the buffer can never remove it.
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


@st.cache_resource
def get_encoding():
    """Load the tokenizer once and reuse it across reruns."""
    return tiktoken.get_encoding("o200k_base")


def count_tokens(messages):
    """Total tokens for a list of messages, including per-message overhead."""
    encoding = get_encoding()
    return sum(len(encoding.encode(m["content"])) + 4 for m in messages)


def buffer_by_turns(messages, max_user_turns=MAX_USER_TURNS):
    """Keep only the last N user messages and everything after them."""
    user_positions = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_positions) <= max_user_turns:
        return messages
    return messages[user_positions[-max_user_turns]:]


def buffer_by_tokens(messages, max_tokens=MAX_TOKENS):
    """Drop the oldest messages until the request fits under max_tokens.

    The system prompt is counted but never dropped, and the most recent
    message is always kept even if it alone exceeds the budget.
    """
    system_cost = count_tokens([{"role": "system", "content": SYSTEM_PROMPT}])
    kept = list(messages)
    while len(kept) > 1 and system_cost + count_tokens(kept) > max_tokens:
        kept.pop(0)
    return kept


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

    if BUFFER_MODE == "tokens":
        history = buffer_by_tokens(st.session_state.lab3_messages)
    else:
        history = buffer_by_turns(st.session_state.lab3_messages)

    request_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=request_messages,
            stream=True,
        )
        answer = st.write_stream(stream)
        st.caption(
            f"Sent {len(request_messages)} messages, "
            f"{count_tokens(request_messages)} tokens "
            f"(limit {MAX_TOKENS})"
        )

    st.session_state.lab3_messages.append({"role": "assistant", "content": answer})