"""
Agentic RAG - Браузер AI: память + веб + оркестрация + реранкер
+ постоянные карточки источников (как Perplexity)
"""

import os
import json
import hashlib
import tempfile
from pathlib import Path
from urllib.parse import urlparse

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

APP_NAME = "🔥 Agentic RAG - чат с документами. Браузер AI."
MEMORY_PAIRS = 5
RERANK_KEEP = 5

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
.src-card { background:#1a1d24; border:1px solid #2a2f3a;
    border-left:3px solid #00d97e; border-radius:8px;
    padding:10px 12px; margin-bottom:8px; }
.src-card a { color:#00d97e; text-decoration:none; font-weight:600;
    font-size:0.95rem; }
.src-card a:hover { text-decoration:underline; }
.src-title { font-weight:600; color:#00d97e; margin-bottom:2px;
    font-size:0.95rem; }
.src-meta { color:#6b7280; font-size:0.75rem; margin-bottom:5px; }
.src-snippet { color:#9aa0aa; font-size:0.83rem; line-height:1.4;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
    overflow:hidden; }
.src-head { color:#9aa0aa; font-size:0.8rem; margin:6px 0 4px; }
code { color:#00d97e !important; background:#11151c !important;
    padding:2px 6px; border-radius:4px; }
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
        instructions=["Отвечай на русском.", "Будь точным и опирайся на контекст."],
        markdown=True,
    )


def web_search(query: str, max_results: int = 8) -> list:
    if not TAVILY_KEY:
        return []
    try:
        r = requests.post("https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_KEY}"},
            json={"query": query, "search_depth": "advanced",
                  "max_results": max_results}, timeout=40)
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


def page_num(doc):
    md = getattr(doc, "meta_data", None) or {}
    return md.get("page")


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


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


def has_documents() -> bool:
    return len(list_documents()) > 0


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
        st.sidebar.success(f"{name}: {len(docs)} чанков")
        return True
    except Exception as e:
        st.sidebar.error(f"Ошибка {name}: {e}")
        return False


def file_fingerprint(name: str, data: bytes) -> str:
    return hashlib.sha256(name.encode() + data).hexdigest()


def memory_context() -> str:
    msgs = st.session_state.messages
    pairs, i, count = [], len(msgs) - 1, 0
    while i >= 1 and count < MEMORY_PAIRS:
        if msgs[i]["role"] == "assistant" and msgs[i - 1]["role"] == "user":
            pairs.insert(0, f"Вопрос: {msgs[i-1]['content']}\nОтвет: {msgs[i]['content']}")
            count += 1
            i -= 2
        else:
            i -= 1
    return "\n\n".join(pairs)


def rerank(question: str, items: list, get_text) -> list:
    if len(items) <= RERANK_KEEP:
        return items
    try:
        agent = get_agent()
        listing = "\n".join(f"[{i}] {get_text(it)[:300]}" for i, it in enumerate(items))
        p = (f"Вопрос: {question}\n\nНиже фрагменты с номерами. Выбери до "
             f"{RERANK_KEEP} САМЫХ релевантных. Ответь ТОЛЬКО номерами через "
             f"запятую по убыванию релевантности.\n\n{listing}")
        r = agent.run(p)
        nums = []
        for tok in r.content.replace(".", ",").split(","):
            tok = tok.strip()
            if tok.isdigit():
                idx = int(tok)
                if 0 <= idx < len(items) and idx not in nums:
                    nums.append(idx)
        return [items[i] for i in nums[:RERANK_KEEP]] if nums else items[:RERANK_KEEP]
    except Exception:
        return items[:RERANK_KEEP]


def classify_query(question: str) -> str:
    if not has_documents():
        return "веб"
    try:
        agent = get_agent()
        p = ("Определи, где искать ответ. Ответь ОДНИМ словом: "
             "'документы' (вопрос про загруженные файлы), "
             "'веб' (свежая/общая информация из интернета), "
             f"'оба'. ВОПРОС: {question}")
        ans = agent.run(p).content.lower()
        if "оба" in ans:
            return "оба"
        if "веб" in ans or "интернет" in ans:
            return "веб"
        return "документы"
    except Exception:
        return "документы"


def search_docs(question: str) -> list:
    try:
        return get_knowledge_base().search(query=question)
    except Exception:
        return []


def build_sources(doc_results, web_results) -> list:
    """Единый список источников для сохранения с сообщением."""
    src = []
    for d in doc_results:
        src.append({"type": "doc", "title": d.name, "page": page_num(d),
                    "url": "", "snippet": (d.content or "")[:200]})
    for r in web_results:
        src.append({"type": "web", "title": r.get("title", "без названия"),
                    "page": None, "url": r.get("url", ""),
                    "snippet": (r.get("content") or "")[:200]})
    return src


def render_sources(sources: list):
    if not sources:
        return
    html = '<div class="src-head">Источники:</div>'
    for i, s in enumerate(sources, 1):
        if s["type"] == "web":
            meta = domain_of(s["url"])
            title = (f'<a href="{s["url"]}" target="_blank">{i}. {s["title"]}</a>')
        else:
            meta = s["title"] + (f' · стр. {s["page"]}' if s["page"] else "")
            title = f'<div class="src-title">{i}. {s["title"]}</div>'
        html += (f'<div class="src-card">{title}'
                 f'<div class="src-meta">{meta}</div>'
                 f'<div class="src-snippet">{s["snippet"]}...</div></div>')
    st.markdown(html, unsafe_allow_html=True)


def suggest_questions(question: str, answer: str) -> list:
    try:
        agent = get_agent()
        p = (f"На основе вопроса и ответа предложи 3 коротких уточняющих вопроса. "
             f"Только вопросы, каждый с новой строки, без нумерации.\n\n"
             f"ВОПРОС: {question}\n\nОТВЕТ: {answer[:1500]}")
        r = agent.run(p)
        lines = [l.strip(" -•0123456789.").strip()
                 for l in r.content.split("\n") if l.strip()]
        return [l for l in lines if len(l) > 5][:3]
    except Exception:
        return []


def answer_question(question: str):
    agent = get_agent()
    mem = memory_context()
    mode = st.session_state.get("mode", "🧠 Авто")

    with st.status("Обрабатываю запрос...", expanded=True) as status:
        if mode == "🧠 Авто":
            st.write("🧠 Выбираю источник...")
            route = classify_query(question)
            st.write(f"➡️ Решение: {route}")
        elif mode == "📄 Документы":
            route = "документы"
        else:
            route = "веб"

        doc_results, web_results = [], []

        if route in ("документы", "оба"):
            st.write("🔎 Ищу в документах...")
            doc_results = search_docs(question)
            if doc_results:
                st.write("⚖️ Переранжирую...")
                doc_results = rerank(question, doc_results, lambda d: d.content)
            st.write(f"📄 Фрагментов: {len(doc_results)}")

        if route in ("веб", "оба"):
            st.write("🔎 Ищу в интернете...")
            web_results = web_search(question)
            if web_results:
                st.write("⚖️ Переранжирую...")
                web_results = rerank(question, web_results,
                                     lambda r: r.get("content", ""))
            st.write(f"🌐 Источников: {len(web_results)}")

        parts = []
        for d in doc_results:
            parts.append(f"[Документ: {d.name}{page_of(d)}]\n{d.content}")
        for r in web_results:
            parts.append(f"[Веб: {r.get('title')}]\n{r.get('content','')}")
        ctx = "\n\n".join(parts)

        if ctx:
            prompt = (f"Контекст беседы:\n{mem}\n\n" if mem else "") + \
                ("Ответь на новый вопрос, используя источники ниже и учитывая беседу. "
                 "На русском. НЕ перечисляй источники и ссылки в конце ответа - они "
                 f"показываются отдельно.\n\nИСТОЧНИКИ:\n{ctx}\n\nВОПРОС: {question}")
        else:
            st.write("ℹ️ Источников нет, отвечаю напрямую.")
            prompt = (f"Контекст:\n{mem}\n\n" if mem else "") + question

        st.write("✍️ Формирую ответ...")
        response = agent.run(prompt)
        status.update(label="Готово", state="complete", expanded=False)

    sources = build_sources(doc_results, web_results)
    st.markdown(response.content)
    render_sources(sources)
    st.session_state.messages.append({"role": "assistant",
                                       "content": response.content,
                                       "sources": sources})
    sugg = suggest_questions(question, response.content)
    if sugg:
        st.session_state.suggestions = sugg


# ================== ИНТЕРФЕЙС ==================
st.set_page_config(page_title="Браузер AI", page_icon="🔥", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = set()
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "pending" not in st.session_state:
    st.session_state.pending = None

st.title(APP_NAME)

dim = check_embedder()
if dim == 0:
    st.error("Эмбеддер не работает. Проверьте ключ в .env.\n\n"
             f"Детали: {st.session_state.get('embed_error', 'нет данных')}")
    st.stop()
st.caption(f"Готов | Веб: {'включён' if TAVILY_KEY else 'нет ключа'} | "
           f"Память: {MEMORY_PAIRS} пар | Реранкер: вкл")

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
    st.cache_resource.clear()
    st.rerun()

st.sidebar.divider()
st.session_state.mode = st.sidebar.radio("Где искать:",
    ["🧠 Авто", "📄 Документы", "🌐 Веб"])
if st.sidebar.button("🧹 Очистить диалог"):
    st.session_state.messages = []
    st.session_state.suggestions = []
    st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            render_sources(m["sources"])

if st.session_state.suggestions and st.session_state.messages:
    st.caption("Связанные вопросы:")
    cols = st.columns(len(st.session_state.suggestions))
    for col, sq in zip(cols, st.session_state.suggestions):
        if col.button(sq, key=f"sg_{sq}"):
            st.session_state.pending = sq
            st.session_state.suggestions = []
            st.rerun()

if st.session_state.pending:
    q = st.session_state.pending
    st.session_state.pending = None
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        answer_question(q)
    st.rerun()

if question := st.chat_input("Задайте вопрос..."):
    st.session_state.suggestions = []
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer_question(question)
    st.rerun()