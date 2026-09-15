import glob
import os
import sys

# ChromaDB needs a newer sqlite3 than Streamlit Cloud ships with.
# Swap in pysqlite3 before chromadb is imported. Harmless if not installed.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

PDF_DIR = "pdfs"
COLLECTION_NAME = "Lab4Collection"
CHROMA_PATH = "./ChromaDB_for_lab4"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# Conversation memory: last 6 messages (3 user-assistant exchanges).
BUFFER_SIZE = 6

# How many syllabi to pull from the vector DB for each question.
TOP_K = 3

SYSTEM_TEMPLATE = (
    "You are a course advisor for the Syracuse University iSchool. Answer "
    "questions about the courses using the syllabus excerpts provided below.\n\n"
    "Be explicit about where your answer comes from. When you use the course "
    "materials, start your answer by naming the syllabus or syllabi you used, "
    "for example: 'Based on the IST 418 syllabus...'. If the excerpts do not "
    "contain the answer, say clearly that the course materials do not cover it "
    "before you say anything else.\n\n"
    "--- COURSE MATERIALS ---\n{context}"
)


def get_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def read_pdf(path):
    """Read a PDF file and return its text."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def embed(text):
    """Return the embedding vector for a piece of text."""
    response = get_client().embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def build_vector_db():
    """Create the Lab4Collection ChromaDB collection from the PDFs in PDF_DIR.

    The filename is used as the document id, so a file is embedded only once
    even if this runs again.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    already_stored = set(collection.get()["ids"])
    paths = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))

    for path in paths:
        filename = os.path.basename(path)
        if filename in already_stored:
            continue

        text = read_pdf(path)
        if not text.strip():
            continue

        collection.add(
            ids=[filename],
            documents=[text],
            embeddings=[embed(text)],
            metadatas=[{
                "filename": filename,
                "course": filename.split(" Syllabus")[0],
            }],
        )

    return collection


def search(collection, query, k=TOP_K):
    """Return [(filename, document_text), ...] for the k closest documents."""
    results = collection.query(query_embeddings=[embed(query)], n_results=k)
    return list(zip(results["ids"][0], results["documents"][0]))


st.title("Lab 4: Course Information Chatbot (RAG)")

# ---------- Build the vector DB once per session ----------
if "Lab4_VectorDB" not in st.session_state:
    if not glob.glob(os.path.join(PDF_DIR, "*.pdf")):
        st.error(f"No PDFs found in the '{PDF_DIR}' folder.")
        st.stop()
    with st.spinner("Building the vector database (first run only)..."):
        st.session_state.Lab4_VectorDB = build_vector_db()

collection = st.session_state.Lab4_VectorDB
st.caption(f"{COLLECTION_NAME}: {collection.count()} syllabi indexed")

# ---------- Part A: vector DB test ----------
# Turn this on to verify retrieval. Part B works with it switched off.
if st.sidebar.checkbox("Show vector DB test (Part A)"):
    test_query = st.sidebar.text_input("Test search", value="Generative AI")
    if test_query:
        st.subheader("Part A: top 3 matches")
        for rank, (filename, _) in enumerate(search(collection, test_query), start=1):
            st.write(f"{rank}. {filename}")

# ---------- Part B: the chatbot ----------
if "lab4_messages" not in st.session_state:
    st.session_state.lab4_messages = [
        {"role": "assistant", "content": "Ask me anything about the iSchool courses."}
    ]

for message in st.session_state.lab4_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the courses"):
    st.session_state.lab4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve the most relevant syllabi and put them in the system prompt.
    matches = search(collection, prompt)
    context = "\n\n".join(
        f"--- {filename} ---\n{text}" for filename, text in matches
    )
    system_prompt = SYSTEM_TEMPLATE.format(context=context)

    history = st.session_state.lab4_messages[-BUFFER_SIZE:]
    request_messages = [{"role": "system", "content": system_prompt}] + history

    with st.chat_message("assistant"):
        stream = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=request_messages,
            stream=True,
        )
        answer = st.write_stream(stream)
        st.caption("Retrieved: " + ", ".join(filename for filename, _ in matches))

    st.session_state.lab4_messages.append({"role": "assistant", "content": answer})