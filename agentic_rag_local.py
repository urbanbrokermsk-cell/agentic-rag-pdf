"""
Agentic RAG - облачная версия (aitunnel)
Эмбеддинги: text-embedding-3-small | Генерация: Gemini | Без Ollama
"""

import os
import hashlib
import tempfile
from pathlib import Path

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.agent import AgentKnowledge
from agno.document.base import Document
from agno.document.reader.pdf_reader import PDFReader
from agno.document.chunking.fixed import FixedSizeChunking

load_dotenv()

# ========================= НАСТРОЙКИ =========================
DB_DIR = "data/lancedb"
TABLE_NAME = "documents"
LLM_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
API_KEY = os.getenv("AITUNNEL_API_KEY")
BASE_URL = "https://api.aitunnel.ru/v1"
NUM_DOCUMENTS = 10
SEARCH_TYPE = SearchType.vector

CHUNKER = FixedSizeChunking(chunk_size=800, overlap=150)


# ================== КЭШИРОВАННЫЕ РЕСУРСЫ ==================
@st.cache_resource
def get_embedder():
    return OpenAIEmbedder(
        id=EMBED_MODEL,
        dimensions=EMBED_DIM,
        api_key=API_KEY,
        base_url=BASE_URL,
    )


@st.cache_resource
def get_knowledge_base() -> AgentKnowledge:
    vector_db = LanceDb(
        table_name=TABLE_NAME,
        uri=DB_DIR,
        use_tantivy=False,
        search_type=SEARCH_TYPE,
        embedder=get_embedder(),
    )
    return AgentKnowledge(vector_db=vector_db, num_documents=NUM_DOCUMENTS)


@st.cache_resource
def get_agent() -> Agent:
    return Agent(
        model=OpenAIChat(id=LLM_MODEL, api_key=API_KEY, base_url=BASE_URL),
        knowledge=get_knowledge_base(),
        search_knowledge=True,
        instructions=[
            "Отвечай ТОЛЬКО на основе найденных документов.",
            "Если информации нет в документах - честно скажи об этом.",
            "Указывай, из какого документа взята информация.",
            "Отвечай на русском языке.",
        ],
        markdown=True,
    )


def check_embedder() -> int:
    try:
        v = get_embedder().get_embedding("проверка работы эмбеддера")
        return len(v) if v else 0
    except Exception as e:
        st.session_state["embed_error"] = str(e)
        return 0


# ================== ОБРАБОТКА ДОКУМЕНТОВ ==================
def html_to_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def chunk_text(name: str, text: str) -> list:
    doc = Document(name=name, content=text)
    chunks = CHUNKER.chunk(doc)
    return [c for c in chunks if c.content and len(c.content.strip()) > 30]


def read_pdf_bytes(name: str, data: bytes) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(data)
        tmp_path = f.name
    reader = PDFReader(chunk=False)
    raw_docs = reader.read(tmp_path)
    chunks = []
    for d in raw_docs:
        d.name = name
        chunks.extend(chunk_text(name, d.content))
    Path(tmp_path).unlink(missing_ok=True)
    return chunks


def process_file(filename: str, data: bytes) -> list:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return read_pdf_bytes(filename, data)
    if suffix in (".html", ".htm"):
        return chunk_text(filename, html_to_text(data.decode("utf-8", errors="ignore")))
    if suffix in (".txt", ".md"):
        return chunk_text(filename, data.decode("utf-8", errors="ignore"))
    return []


def ingest(docs: list, source_name: str) -> bool:
    if not docs:
        st.sidebar.warning(f"Не удалось извлечь текст: {source_name}")
        return False
    kb = get_knowledge_base()
    try:
        kb.load_documents(documents=docs, skip_existing=True)
        st.session_state.sources.append(f"{source_name} ({len(docs)} чанков)")
        st.sidebar.success(f"{source_name}: {len(docs)} чанков")
        return True
    except Exception as e:
        st.sidebar.error(f"Ошибка {source_name}: {e}")
        return False


def file_fingerprint(name: str, data: bytes) -> str:
    return hashlib.md5(name.encode() + str(len(data)).encode()).hexdigest()


# ================== ИНТЕРФЕЙС ==================
st.set_page_config(page_title="Agentic RAG", page_icon="🔥")

if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = set()
if "sources" not in st.session_state:
    st.session_state.sources = []

st.title("🔥 Agentic RAG - чат с документами")

dim = check_embedder()
if dim == 0:
    st.error(
        "Эмбеддер не работает. Проверьте ключ AITUNNEL_API_KEY в файле .env "
        "и баланс на aitunnel.\n\n"
        f"Детали: {st.session_state.get('embed_error', 'нет данных')}"
    )
    st.stop()
else:
    st.caption(f"Эмбеддер работает (облако), размерность вектора: {dim}")

st.markdown(
    "Облачная RAG-система: **OpenAI Embeddings** + **LanceDB** + **Gemini**.\n\n"
    "Добавьте PDF / HTML / TXT в боковую панель и задавайте вопросы."
)

# ---------- Боковая панель ----------
st.sidebar.header("Добавление источников знаний")

url = st.sidebar.text_input("Добавить URL-адрес", placeholder="https://example.com/sample.pdf")
if st.sidebar.button("Добавить URL"):
    if url and url not in st.session_state.loaded_files:
        try:
            with st.spinner("Скачиваю и индексирую..."):
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if url.lower().endswith(".pdf") or "pdf" in ctype:
                    docs = read_pdf_bytes(url.split("/")[-1], resp.content)
                else:
                    docs = chunk_text(url, html_to_text(resp.text))
                if ingest(docs, url):
                    st.session_state.loaded_files.add(url)
        except Exception as e:
            st.sidebar.error(f"Ошибка: {e}")
    elif url:
        st.sidebar.info("Этот URL уже добавлен.")

uploaded = st.sidebar.file_uploader(
    "Загрузить файлы (PDF / HTML / TXT)",
    type=["pdf", "html", "htm", "txt", "md"],
    accept_multiple_files=True,
)
if uploaded:
    for f in uploaded:
        data = f.getvalue()
        fid = file_fingerprint(f.name, data)
        if fid in st.session_state.loaded_files:
            continue
        with st.spinner(f"Индексирую {f.name}..."):
            try:
                docs = process_file(f.name, data)
                if ingest(docs, f.name):
                    st.session_state.loaded_files.add(fid)
            except Exception as e:
                st.sidebar.error(f"Ошибка {f.name}: {e}")

st.sidebar.header("Текущие источники знаний")
if st.session_state.sources:
    for s in st.session_state.sources:
        st.sidebar.markdown(f"- {s}")
else:
    st.sidebar.caption("Источники пока не добавлены.")

if st.sidebar.button("Очистить базу знаний"):
    try:
        get_knowledge_base().vector_db.drop()
    except Exception:
        pass
    st.session_state.loaded_files = set()
    st.session_state.sources = []
    st.cache_resource.clear()
    st.rerun()

# ---------- Вопрос и ответ ----------
question = st.text_input("Введите свой вопрос:", placeholder="О чём этот документ?")

if st.button("Получить ответ") and question:
    kb = get_knowledge_base()

    with st.spinner("Ищу в базе знаний..."):
        try:
            results = kb.search(query=question)
        except Exception:
            results = []

    if results:
        with st.expander(f"Найдено фрагментов: {len(results)} (источники)"):
            for i, doc in enumerate(results, 1):
                st.markdown(f"**{i}. {doc.name}**")
                st.caption(doc.content[:400] + "...")
    else:
        st.warning("В базе знаний ничего не найдено. Сначала загрузите документы слева.")

    with st.spinner("Генерирую ответ..."):
        agent = get_agent()
        if results:
            context = "\n\n".join(f"[Источник: {d.name}]\n{d.content}" for d in results)
            prompt = (
                "Ответь на вопрос, используя ТОЛЬКО текст из документов ниже. "
                "Если ответа нет в тексте - честно скажи об этом. "
                "Отвечай на русском языке. В конце укажи, из какого файла взята информация.\n\n"
                "ТЕКСТ ДОКУМЕНТОВ:\n" + context + "\n\nВОПРОС: " + question
            )
            response = agent.run(prompt)
        else:
            response = agent.run(question)

    st.header("Ответ")
    st.markdown(response.content)

with st.expander("Как это работает"):
    st.markdown(
        "1. Файлы разбиваются на чанки по 800 символов (перекрытие 150).\n"
        "2. OpenAI Embeddings создаёт векторы.\n"
        "3. LanceDB ищет похожие фрагменты.\n"
        "4. Gemini отвечает только по найденным фрагментам."
    )