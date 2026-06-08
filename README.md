# 🔥 Agentic RAG с PDF (OpenAI-совместимый API)

Агентное RAG-приложение: загружаешь PDF, задаёшь вопросы — ИИ отвечает строго по содержимому документов. Адаптация локального шаблона под облачный OpenAI-совместимый API (AITunnel / OpenAI / Gemini). Готово к запуску на VPS.

## Возможности

- 📄 База знаний из PDF — добавляй документы по URL
- 🔍 Семантический поиск по смыслу (LanceDB)
- 🤖 Принудительный retrieval — ответ всегда на основе найденного текста
- 🌐 Ответы на русском языке
- ⚡ Потоковая генерация ответов
- ☁️ Облачные эмбеддинги — не требует мощного железа

## Стек

| Компонент | Технология |
|---|---|
| Эмбеддинги | text-embedding-3-small (OpenAI-совместимый API) |
| Генерация | Gemini 3.1 Flash Lite / любая LLM |
| Векторная БД | LanceDB |
| Агенты | Agno |
| Интерфейс | Streamlit |

## Быстрый старт

```bash
git clone https://github.com/Kolcoin/agentic-rag-pdf.git
cd agentic-rag-pdf

pip install -r requirements.txt

cp .env.example .env
# впиши свой ключ в .env: AITUNNEL_API_KEY=sk-...

streamlit run agentic_rag_embeddinggemma.py

Открой http://localhost:8501


Как пользоваться


Вставь URL PDF в боковой панели → «Add URL»

Дождись индексации (Found N documents в логе)

Задай вопрос — получишь ответ по документу


Развёртывание на VPS

streamlit run agentic_rag_embeddinggemma.py --server.port 8501 --server.address 0.0.0.0

Рекомендуется поставить за Nginx с HTTPS и добавить авторизацию.


Лицензия

MIT. Основано на шаблоне awesome-llm-apps.




