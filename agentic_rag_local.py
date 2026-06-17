"""
Поисковая система - облако (aitunnel) + веб-поиск (Tavily)
Режимы: Документы / Веб | Тёмная тема, карточки источников
"""

import os
import json
import hashlib
import tempfile
from pathlib import Path

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader as PyPdfReader

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.agent import AgentKnowledge
from agno.document.base import Document
from agno.document.chunking.fixed import FixedSizeChunking

load_dotenv()

APP_NAME = "🔥 Agentic RAG - чат с документами"   # <-- сюда впиши своё название

DB_DIR = "data/lancedb"
TABLE_NAME = "documents"
LLM_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
API_KEY = os.getenv("AITUNNEL_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
BASE_URL = "https://api.aitunnel.ru/v1"
NUM_DOCUMENTS = 10
SEARCH_TYPE = SearchType.vector
CHUNKER = FixedSizeChunking(chunk_size=800, overlap=150)

CARD_CSS = """
<style>
.src-card {
    background: #1a1d24;
    border: 1px solid #2a2f3a;
    border-left: 3px solid #00d97e;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.src-card a { color: #00d97e; text-decoration: none; font-weight: 600; }
.src-card a:hover { text-decoration: underline; }
.src-title { font-weight: 600; color: #e6e6e6; margin-bottom: 4px; }
.src-snippet { color: #9aa0aa; font-size: 0.85rem; line-height: 1.4; }
code { color: #00d97e !important; background: #11151c !important;
       padding: 2px 6px; border-radius: 4px; }
.stMarkdown pre code { color: #00d97e !important; }
</style>
"""


@st.cache_resource
def get_embedder():
    return OpenAIEmbedder(id=EMBED_MODEL, dimensions=EMBED_DIM,
                          api_key=API_KEY, base_url=BASE_URL)


@st.cache_resource
def get_knowledge_base() -> AgentKnowledge:
    vdb = LanceDb(table_name=TABLE_NAME, uri=DB_DIR, use_tantivy=False,
                  search_type=SEARCH_TYPE, embedder=get_embedder())
    return AgentKnowledge(vector_db=vdb, num_documents=NUM_DOCUMENTS)


@st.cache_resource
def get_agent() -> Agent:
    return Agent(
        model=OpenAIChat(id=LLM_MODEL, api_key=API_KEY, base_url=BASE_URL),
        instructions=["Отвечай ТОЛЬКО по переданному контексту.",
                      "Если ответа нет - честно скажи.", "Отвечай на русском."],
        markdown=True,
    )


def web_search(query: str, max_results: int = 6) -> list:
    if not TAVILY_KEY:
        return []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_KEY}"},
            json={"query": query, "search_depth": "advanced",
                  "max_results": max_results},
            timeout=40,
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        st.error(f"Ошибка веб-поиска: {e}")
        return []


def check_embedder() -> int:
    try:
        v = get_embedder().get_embedding("проверка")
        return len(v) if v else 0
    except Exception as e:
        st.session_state["embed_error"] = str(e)
        return 0


def page_of(doc) -> str:
    md = getattr(doc, "meta_data", None) or {}
    p = md.get("page")
    return f", стр. {p}" if p else ""


def _parse_payload(p):
    if isinstance(p, str):
        try:
            return json.loads(p)
        except Exception:
            return {}
    return p or {}


def _open_table():
    return get_knowledge_base().vector_db.table


def list_documents() -> list:
    try:
        rows = _open_table().search().limit(100000).to_list()
        names = set()
        for r in rows:
            n = _parse_payload(r.get("payload")).get("name")
            if n:
                names.add(n)
        return sorted(names)
    except Exception:
        return []


def delete_document(name: str) -> bool:
    try:
        tbl = _open_table()
        rows = tbl.search().limit(100000).to_list()
        keep = [{"vector": r["vector"], "id": r["id"], "payload": r["payload"]}
                for r in rows if _parse_payload(r.get("payload")).get("name") != name]
        db = get_knowledge_base().vector_db
        try:
            db.connection.drop_table(TABLE_NAME)
        except Exception:
            pass
        if keep:
            db.connection.create_table(TABLE_NAME, data=keep)
        st.cache_resource.clear()
        return True
    except Exception as e:
        st.sidebar.error(f"Ошибка удаления: {e}")
        return False


def html_to_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def chunk_text(name: str, text: str, page=None) -> list:
    doc = Document(name=name, content=text)
    good = []
    for c in CHUNKER.chunk(doc):
        if c.content and len(c.content.strip()) > 30:
            c.name = name
            c.meta_data = {"source": name, "page": page}
            good.append(c)
    return good


def read_pdf_bytes(name: str, data: bytes) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(data)
        tmp = f.name
    chunks = []
    try:
        for i, page in enumerate(PyPdfReader(tmp).pages, 1):
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.extend(chunk_text(name, txt, page=i))
    finally:
        Path(tmp).unlink(missing_ok=True)
    return chunks


def process_file(filename: str, data: bytes) -> list:
    s = Path(filename).suffix.lower()
    if s == ".pdf":
        return read_pdf_bytes(filename, data)
    if s in (".html", ".htm"):
        return chunk_text(filename, html_to_text(data.decode("utf-8", errors="ignore")))
    if s in (".txt", ".md"):
        return chunk_text(filename, data.decode("utf-8", errors="ignore"))
    return []


def ingest(docs: list, name: str) -> bool:
    if not docs:
        st.sidebar.warning(f"Не извлечён текст: {name}")
        return False
    try:
        get_knowledge_base().load_documents(documents=docs, skip_existing=True)
        st.session_state.sources.append(f"{name} ({len(docs)} чанков)")
        st.sidebar.success(f"{name}: {len(docs)} чанков")
        return True
    except Exception as e:
        st.sidebar.error(f"Ошибка {name}: {e}")
        return False


def file_fingerprint(name: str, data: bytes) -> str:
    return hashlib.sha256(name.encode() + data).hexdigest()


# ================== ИНТЕРФЕЙС ==================
st.set_page_config(page_title=APP_NAME, page_icon="🔍", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)

for k, v in [("loaded_files", set()), ("sources", []), ("history", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

st.title(APP_NAME)

dim = check_embedder()
if dim == 0:
    st.error("Эмбеддер не работает. Проверьте ключ в .env.\n\n"
             f"Детали: {st.session_state.get('embed_error', 'нет данных')}")
    st.stop()
st.caption(f"Готов к работе | Веб-поиск: {'включён' if TAVILY_KEY else 'нет ключа'}")

st.sidebar.header("Источники знаний")
url = st.sidebar.text_input("Добавить URL", placeholder="https://...")
if st.sidebar.button("Добавить URL") and url and url not in st.session_state.loaded_files:
    try:
        with st.spinner("Скачиваю..."):
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            if url.lower().endswith(".pdf") or "pdf" in r.headers.get("content-type", ""):
                docs = read_pdf_bytes(url.split("/")[-1], r.content)
            else:
                docs = chunk_text(url, html_to_text(r.text))
            if ingest(docs, url):
                st.session_state.loaded_files.add(url)
    except Exception as e:
        st.sidebar.error(f"Ошибка: {e}")

uploaded = st.sidebar.file_uploader("Загрузить файлы",
    type=["pdf", "html", "htm", "txt", "md"], accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        data = f.getvalue()
        fid = file_fingerprint(f.name, data)
        if fid in st.session_state.loaded_files:
            continue
        with st.spinner(f"Индексирую {f.name}..."):
            if ingest(process_file(f.name, data), f.name):
                st.session_state.loaded_files.add(fid)

st.sidebar.divider()
st.sidebar.header("Документы в базе")
docs_in_db = list_documents()
if docs_in_db:
    sel = st.sidebar.selectbox("Документ для удаления", docs_in_db)
    if st.sidebar.button("🗑 Удалить выбранный"):
        if delete_document(sel):
            st.session_state.sources = [s for s in st.session_state.sources
                                        if not s.startswith(sel)]
            st.sidebar.success(f"Удалён: {sel}")
            st.rerun()
else:
    st.sidebar.caption("База пуста.")

if st.sidebar.button("Очистить базу полностью"):
    try:
        get_knowledge_base().vector_db.connection.drop_table(TABLE_NAME)
    except Exception:
        pass
    st.session_state.loaded_files = set()
    st.session_state.sources = []
    st.session_state.history = []
    st.cache_resource.clear()
    st.rerun()

mode = st.radio("Где искать:", ["📄 Документы", "🌐 Веб"], horizontal=True)
question = st.text_input("Ваш вопрос:", placeholder="Задайте вопрос")

if st.button("Получить ответ") and question:
    agent = get_agent()
    response = None

    if mode == "🌐 Веб":
        with st.spinner("Ищу в интернете..."):
            results = web_search(question)
        if results:
            st.subheader("Источники")
            for i, r in enumerate(results, 1):
                title = r.get("title", "без названия")
                link = r.get("url", "")
                snip = (r.get("content") or "")[:250]
                st.markdown(
                    f'<div class="src-card"><div class="src-title">{i}. '
                    f'<a href="{link}" target="_blank">{title}</a></div>'
                    f'<div class="src-snippet">{snip}...</div></div>',
                    unsafe_allow_html=True)
            ctx = "\n\n".join(f"[Источник: {r.get('title')} ({r.get('url')})]\n{r.get('content','')}"
                              for r in results)
            prompt = ("Ответь на вопрос, используя текст из веб-источников ниже. "
                      "Отвечай на русском. В конце приведи список источников со "
                      "ссылками.\n\nИСТОЧНИКИ:\n" + ctx + "\n\nВОПРОС: " + question)
            with st.spinner("Генерирую ответ..."):
                response = agent.run(prompt)
        else:
            st.warning("Веб-поиск ничего не вернул.")
    else:
        kb = get_knowledge_base()
        with st.spinner("Ищу в документах..."):
            try:
                results = kb.search(query=question)
            except Exception:
                results = []
        if results:
            st.subheader("Источники")
            for i, doc in enumerate(results, 1):
                snip = doc.content[:250]
                st.markdown(
                    f'<div class="src-card"><div class="src-title">{i}. '
                    f'{doc.name}{page_of(doc)}</div>'
                    f'<div class="src-snippet">{snip}...</div></div>',
                    unsafe_allow_html=True)
            ctx = "\n\n".join(f"[Источник: {d.name}{page_of(d)}]\n{d.content}" for d in results)
            prompt = ("Ответь, используя ТОЛЬКО текст из документов ниже. "
                      "Если ответа нет - честно скажи. Отвечай на русском. "
                      "В конце укажи файл и номера страниц.\n\n"
                      "ТЕКСТ:\n" + ctx + "\n\nВОПРОС: " + question)
        else:
            st.warning("Ничего не найдено. Загрузите документы слева.")
            prompt = question
        with st.spinner("Генерирую ответ..."):
            response = agent.run(prompt)

    if response:
        st.header("Ответ")
        st.markdown(response.content)
        st.session_state.history.insert(0, (f"[{mode}] {question}", response.content))

if st.session_state.history:
    st.divider()
    st.subheader("История вопросов")
    for q, a in st.session_state.history:
        with st.expander(q):
            st.markdown(a)