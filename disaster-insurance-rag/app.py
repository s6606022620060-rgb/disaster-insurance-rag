"""
app.py — ศูนย์ข้อมูลเยียวยาน้ำท่วม & ประกันภัยพิบัติ 2569 (มีแชตบอต RAG "น้องพร้อมรับ")
รันในเครื่อง:  streamlit run app.py

โครงหน้าเว็บ
- แถบบน: ปรับขนาดตัวอักษร 3 ระดับ + โหมดตัดสี (High Contrast) สำหรับผู้สูงอายุ/ผู้มีปัญหาการมองเห็น
- Hero: ช่องถามคำถามใหญ่ (ส่งเข้าแชต RAG)
- เมนูหลัก: 💬 ถามน้องพร้อมรับ (RAG) | ❓ คำถามยอดฮิต | 🧮 เช็กสิทธิ์เบื้องต้น | 📑 เอกสาร & ขั้นตอน | 📞 สายด่วน
- แถบด้านข้าง (ซ่อน): เมนูผู้ดูแลระบบ — ตั้งค่าการค้นหา / ทดสอบระบบ / ดูเอกสาร
"""

import html
import os

import pandas as pd
import streamlit as st

from rag_core import (
    CONDENSE_PROMPT,
    EMBEDDING_MODEL_NAME,
    NOT_FOUND_MESSAGE,
    VectorStore,
    build_messages,
    chunk_documents,
    cited_ranks,
    is_refusal,
    load_documents,
    make_sentence_transformer_embedder,
)
from ui_content import CATEGORIES, CHECKLIST, FAQS, HOTLINES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_CSV = os.path.join(BASE_DIR, "test_questions.csv")
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

NAV_CHAT, NAV_FAQ, NAV_CALC, NAV_DOCS, NAV_CALL = (
    "💬 ถามน้องพร้อมรับ", "❓ คำถามยอดฮิต", "🧮 เช็กสิทธิ์เบื้องต้น", "📑 เอกสาร & ขั้นตอน", "📞 สายด่วน",
)
NAV_ITEMS = [NAV_CHAT, NAV_FAQ, NAV_CALC, NAV_DOCS, NAV_CALL]

EXAMPLE_QUESTIONS = [
    "น้ำท่วมบ้าน ได้เงินเท่าไหร่",
    "ต้องลงทะเบียนก่อนไหม",
    "บ้านท่วมตั้งแต่เดือนกันยายน เคลมได้ไหม",
    "แจ้งเคลมทางไหน ใช้เอกสารอะไร",
    "ทีวี ตู้เย็นเสียหาย ได้รับความคุ้มครองไหม",
    "มีคนโทรมาขอ OTP บอกจะโอนเงินเยียวยา",
]

st.set_page_config(
    page_title="ศูนย์ข้อมูลเยียวยาน้ำท่วม & ประกันภัยพิบัติ 2569",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# State เริ่มต้น
# ---------------------------------------------------------------------------
DEFAULTS = {
    "top_k": 4, "threshold": 0.0, "use_condense": True, "show_debug": False,
    "admin_page": "ปิด", "nav": NAV_CHAT, "font_size": "ก", "contrast": False,
    "faq_cat": "all", "saved": [], "messages": [],
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# สไตล์ (ฟอนต์ Prompt + Sarabun, โทนน้ำเงิน, ขนาดตัวอักษร, High Contrast)
# ---------------------------------------------------------------------------
FONT_PX = {"ก": 16, "ก+": 18, "ก++": 20}.get(st.session_state.font_size or "ก", 16)

BASE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@400;500;600;700&family=Sarabun:wght@400;500;600;700&display=swap');
html {{ font-size: {FONT_PX}px; }}
html, body, .stApp, .stMarkdown, .stChatMessage, button, input, textarea, select, label, p, li {{
    font-family: 'Sarabun', sans-serif !important;
}}
h1, h2, h3, h4, .heading {{ font-family: 'Prompt', sans-serif !important; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; pointer-events: none; }}
header[data-testid="stHeader"] * {{ pointer-events: auto; }}
.stApp {{ background: #F8FAFC; }}
.block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1180px; }}

/* แถบบน */
.st-key-topbar {{
    background: #0C4A6E; border-radius: 14px; padding: 6px 14px; color: #fff;
}}
.st-key-topbar p, .st-key-topbar label, .st-key-topbar span {{ color: #E0F2FE !important; }}
.st-key-topbar [data-testid="stHorizontalBlock"] {{ align-items: center; }}
.st-key-topbar [data-testid="stButtonGroup"] button {{
    background: transparent !important; border: 1px solid #38BDF8 !important; min-height: 1.9rem;
}}
.st-key-topbar [data-testid="stButtonGroup"] button p {{ color: #E0F2FE !important; font-weight: 700; }}
.st-key-topbar [data-testid="stButtonGroup"] button[kind$="Active"],
.st-key-topbar [data-testid="stButtonGroup"] button[data-testid$="Active"],
.st-key-topbar [data-testid="stButtonGroup"] button[aria-checked="true"] {{ background: #FBBF24 !important; border-color:#FBBF24 !important; }}
.st-key-topbar [data-testid="stButtonGroup"] button[kind$="Active"] p,
.st-key-topbar [data-testid="stButtonGroup"] button[data-testid$="Active"] p,
.st-key-topbar [data-testid="stButtonGroup"] button[aria-checked="true"] p {{ color: #0F172A !important; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; background:#34D399;
        margin-right:6px; box-shadow:0 0 0 3px rgba(52,211,153,.25); }}

/* Header */
.appbar {{ display:flex; align-items:center; justify-content:space-between; gap:12px;
          flex-wrap:wrap; padding: 14px 4px 10px 4px; }}
.brand {{ display:flex; align-items:center; gap:12px; }}
.logo {{ width:52px; height:52px; border-radius:14px; display:flex; align-items:center;
        justify-content:center; font-size:28px; background:linear-gradient(135deg,#0284C7,#06B6D4);
        box-shadow:0 6px 16px rgba(2,132,199,.25); }}
.brand h1 {{ font-size:1.45rem; margin:0; padding:0; color:#0F172A; line-height:1.25; }}
.brand h1 span {{ color:#0369A1; }}
.brand p {{ margin:0; color:#64748B; font-size:.88rem; }}
.chips a {{ display:inline-block; text-decoration:none; padding:6px 12px; border-radius:999px;
           font-weight:600; font-size:.88rem; margin-left:6px; }}
.chip-red {{ background:#FFE4E6; color:#BE123C !important; }}
.chip-blue {{ background:#E0F2FE; color:#075985 !important; }}

/* Hero */
.st-key-hero {{
    background: radial-gradient(circle at 85% 15%, rgba(34,211,238,.25), transparent 40%),
                linear-gradient(160deg, #0C4A6E 0%, #075985 45%, #0F172A 100%);
    border-radius: 22px; padding: 30px 28px 22px 28px; color: #fff;
}}
.st-key-hero h2 {{ color:#fff; font-size:2rem; margin:6px 0 6px 0; padding:0; text-align:center; }}
.st-key-hero p {{ color:#CBD5E1; text-align:center; }}
.pill {{ display:inline-block; padding:4px 12px; border-radius:999px; font-size:.82rem;
        background:rgba(59,130,246,.18); border:1px solid rgba(147,197,253,.35); color:#BFDBFE; }}
.st-key-hero [data-testid="stForm"] {{ border:none; padding:0; max-width:720px; margin:0 auto; }}
.st-key-hero input {{ font-size:1.08rem !important; padding:14px 16px !important; border-radius:14px !important; }}
.st-key-hero [data-testid="stFormSubmitButton"] button {{
    background:#F59E0B; color:#0F172A; border:none; border-radius:14px; font-weight:700;
    height:3.1rem; width:100%;
}}
.st-key-hero [data-testid="stFormSubmitButton"] button p {{ color:#0F172A !important; font-weight:700; font-size:1.02rem; }}

/* เมนูหลัก */
.st-key-navwrap [data-testid="stButtonGroup"] {{ justify-content:center; flex-wrap:wrap; }}
.st-key-navwrap button {{ border-radius:12px !important; font-weight:600; }}

/* การ์ด */
.card {{ background:#fff; border:1px solid #E2E8F0; border-radius:18px; padding:16px 18px;
        box-shadow:0 1px 2px rgba(15,23,42,.04); margin-bottom:12px; }}
.card h3 {{ font-size:1.08rem; margin:0 0 6px 0; padding:0; color:#0F172A; }}
.card.dark {{ background:linear-gradient(160deg,#0F172A,#0C4A6E); border-color:#1E293B; color:#E2E8F0; }}
.card.dark h3 {{ color:#fff; }}
.fact {{ display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px dashed #E2E8F0; }}
.fact:last-child {{ border-bottom:none; }}
.fact b {{ color:#0369A1; }}
.hot {{ display:flex; justify-content:space-between; align-items:center; gap:10px;
       background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12);
       border-radius:14px; padding:9px 12px; margin-top:8px; }}
.hot .n {{ font-weight:600; color:#fff; font-size:.92rem; }}
.hot .d {{ font-size:.78rem; color:#94A3B8; }}
.hot a {{ background:#E11D48; color:#fff !important; text-decoration:none; font-weight:700;
         padding:6px 12px; border-radius:10px; white-space:nowrap; }}
.badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem;
         background:#EFF6FF; color:#075985; border:1px solid #BFDBFE; margin-bottom:4px; }}
.hl li {{ margin-bottom:4px; }}
.result {{ background:linear-gradient(160deg,#0F172A,#0C4A6E); color:#fff; border-radius:18px;
          padding:18px; text-align:center; }}
.result .amt {{ font-family:'Prompt',sans-serif; font-size:2.2rem; font-weight:700; color:#FBBF24; }}
.result .lbl {{ color:#94A3B8; font-size:.85rem; }}
.src-title {{ font-weight:600; margin-bottom:2px; }}
.src-meta {{ font-size:.82rem; color:#64748B; margin-bottom:6px; }}
.src-body {{ font-size:.9rem; background:#F8FAFC; border-left:3px solid #0284C7;
            padding:8px 10px; border-radius:6px; white-space:pre-wrap; }}
.foot {{ background:#0F172A; color:#94A3B8; border-radius:18px; padding:18px 20px; margin-top:22px;
        font-size:.85rem; }}
.foot b {{ color:#E2E8F0; font-family:'Prompt',sans-serif; }}
[data-testid="stExpander"] {{ background:#fff; border-radius:14px !important; }}
div[data-testid="stChatMessage"] {{ background:#fff; border:1px solid #E2E8F0; border-radius:16px; }}
</style>
"""

CONTRAST_CSS = """
<style>
.stApp, .block-container { background:#000 !important; color:#fff !important; }
.stApp p, .stApp li, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.brand h1, .brand p, .card h3 { color:#fff !important; }
.card, [data-testid="stExpander"], div[data-testid="stChatMessage"], .src-body, .result, .card.dark,
.st-key-hero, .st-key-topbar, .foot {
    background:#121212 !important; border:2px solid #FFF !important; color:#fff !important;
}
.brand h1 span, .fact b, .badge, a { color:#FFE600 !important; }
.badge { background:#000 !important; border-color:#FFE600 !important; }
button { border:2px solid #fff !important; }
.hot a { background:#FFE600 !important; color:#000 !important; }
.stApp button { background:#000 !important; }
.stApp button p, .stApp button span { color:#fff !important; }
.stApp button[aria-checked="true"], .st-key-hero [data-testid="stFormSubmitButton"] button {
    background:#FFE600 !important; }
.stApp button[aria-checked="true"] p, .st-key-hero [data-testid="stFormSubmitButton"] button p {
    color:#000 !important; }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)
if st.session_state.contrast:
    st.markdown(CONTRAST_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# โหลดโมเดลและ index เพียงครั้งเดียว (cache) — สำคัญมากบน Streamlit Community Cloud
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังเตรียมผู้ช่วย... (ครั้งแรกอาจใช้เวลา 1-2 นาที)")
def get_embedder():
    return make_sentence_transformer_embedder(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="กำลังเตรียมเอกสารความรู้...")
def get_vector_store():
    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs)
    store = VectorStore(chunks, get_embedder())
    return docs, chunks, store


@st.cache_resource
def _make_groq_client(api_key: str):
    from groq import Groq

    return Groq(api_key=api_key)


def get_groq_client():
    """อ่าน key จาก st.secrets ทุกครั้ง (ไม่ cache ค่า None) เพื่อให้ใส่ key ภายหลังแล้วใช้ได้ทันที"""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None
    return _make_groq_client(api_key) if api_key else None


def get_model_name() -> str:
    try:
        return st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)
    except Exception:
        return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def llm_complete(messages, temperature=0.2, stream=False):
    client = get_groq_client()
    model = get_model_name()
    try:
        return client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, stream=stream
        )
    except Exception as e:
        if "model" in str(e).lower() and model != FALLBACK_MODEL:
            return client.chat.completions.create(
                model=FALLBACK_MODEL, messages=messages, temperature=temperature, stream=stream
            )
        raise


def condense_question(question: str, history: list[dict]) -> str:
    """เขียนคำถามต่อเนื่อง (เช่น 'แล้วถ้าเป็นพายุล่ะ') ให้เป็นคำถามสมบูรณ์ก่อนนำไปค้นหา"""
    if not history:
        return question
    history_text = "\n".join(
        f"{'ผู้ใช้' if h['role'] == 'user' else 'ผู้ช่วย'}: {h['content'][:400]}" for h in history[-4:]
    )
    try:
        resp = llm_complete(
            [{"role": "user", "content": CONDENSE_PROMPT.format(history=history_text, question=question)}],
            temperature=0.0,
        )
        rewritten = (resp.choices[0].message.content or "").strip().splitlines()[0]
        return rewritten or question
    except Exception:
        return question


def stream_answer(messages):
    resp = llm_complete(messages, stream=True)
    for chunk in resp:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def esc(text) -> str:
    return html.escape(str(text))


def render_sources(results: list[dict], answer: str, search_query: str):
    used = cited_ranks(answer)
    refused = is_refusal(answer)
    debug = st.session_state.show_debug
    if refused:
        shown, label = results, "🔍 เอกสารที่ใกล้เคียงที่สุด (ไม่มีคำตอบของคำถามนี้)"
    else:
        shown = [r for r in results if r["rank"] in used] or results
        label = f"📚 ดูแหล่งอ้างอิง ({len(shown)})"
    if debug:
        shown = results
    with st.expander(label, expanded=False):
        if debug:
            st.caption(f"🔎 คำค้นที่ใช้: {search_query} · Top-k = {st.session_state.top_k}")
        if not shown:
            st.caption("ไม่พบเอกสารที่เกี่ยวข้อง")
        for r in shown:
            body = r["text"].split("\n", 2)[-1]
            body = body[:450] + ("..." if len(body) > 450 else "")
            meta = f"ข้อมูล ณ {r['as_of']}"
            if debug:
                status = "✅ ใช้ตอบ" if r["rank"] in used else "▫️ ค้นพบ"
                meta = f"{status} · score {r['score']:.3f} · {r['file_name']} · {meta}"
            st.markdown(
                f"<div class='src-title'>[{r['rank']}] {esc(r['doc_title'])}</div>"
                f"<div class='src-meta'>หัวข้อ: {esc(r['section'])} · {esc(meta)}</div>"
                f"<div class='src-body'>{esc(body)}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"ที่มา: {r['source']}")


def go_ask(question: str):
    """ส่งคำถามเข้าแชต แล้วสลับไปเมนูแชต (ใช้กับปุ่ม 'ถามต่อ' และช่องค้นหาใน hero)"""
    st.session_state.pending_question = question
    st.session_state.nav = NAV_CHAT


# ---------------------------------------------------------------------------
# โหลดข้อมูล
# ---------------------------------------------------------------------------
docs, chunks, store = get_vector_store()


# ---------------------------------------------------------------------------
# Sidebar: เมนูผู้ดูแลระบบ
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ ผู้ดูแลระบบ")
    st.radio("หน้า", ["ปิด", "🧪 ทดสอบระบบ", "📁 เอกสารในระบบ"], key="admin_page")
    st.markdown("**ตั้งค่าการค้นหา**")
    st.slider("Top-k (จำนวน chunk ที่ดึงมา)", 1, 8, key="top_k")
    st.slider(
        "Similarity Threshold (0 = ปิด)", 0.0, 0.95, step=0.01, key="threshold",
        help="chunk ที่คะแนนต่ำกว่านี้จะไม่ถูกส่งให้ LLM ถ้าไม่เหลือเลย ระบบตอบว่าไม่พบข้อมูลทันที",
    )
    st.toggle("เขียนคำถามต่อเนื่องใหม่ก่อนค้นหา", key="use_condense")
    st.toggle("แสดงรายละเอียดการค้นหา (score, ไฟล์)", key="show_debug")
    if st.button("🗑️ ล้างประวัติแชต", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption(
        f"{len(docs)} ไฟล์ · {len(chunks)} chunks  \n"
        f"Embedding: `{EMBEDDING_MODEL_NAME}`  \nLLM: `{get_model_name()}` (Groq)"
    )


# ---------------------------------------------------------------------------
# 1) แถบบน: สถานะข้อมูล + เครื่องมือช่วยการมองเห็น
# ---------------------------------------------------------------------------
with st.container(key="topbar"):
    c1, c2, c3 = st.columns([4.6, 2.2, 1.9])
    c1.markdown(
        "<span class='dot'></span>อัปเดตข้อมูล 7 ต.ค. 2569 · ประกันภัยพิบัติเริ่มคุ้มครอง 1 ต.ค. 69",
        unsafe_allow_html=True,
    )
    with c2:
        st.segmented_control(
            "ขนาดตัวอักษร", ["ก", "ก+", "ก++"], key="font_size",
            label_visibility="collapsed", help="ปรับขนาดตัวอักษร: ปกติ / ใหญ่ / ใหญ่พิเศษ",
        )
    with c3:
        st.toggle("🌓 ตัดสี", key="contrast")


# ---------------------------------------------------------------------------
# 2) Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="appbar">
      <div class="brand">
        <div class="logo">🌊</div>
        <div>
          <h1>เยียวยาน้ำท่วม &amp; ประกันภัยพิบัติ <span>2569</span></h1>
          <p>ถาม-ตอบด้วย AI จากข้อมูลทางการ · เช็กสิทธิ์ · เตรียมเอกสาร</p>
        </div>
      </div>
      <div class="chips">
        <a class="chip-red" href="tel:1784">📞 ปภ. 1784</a>
        <a class="chip-blue" href="tel:020125555">🛡️ ThaiNATCAT 02-012-5555</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 3) Hero + ช่องถามคำถาม
# ---------------------------------------------------------------------------
with st.container(key="hero"):
    st.markdown(
        "<div style='text-align:center'><span class='pill'>🛡️ อ้างอิงข่าวและประกาศจาก ปภ. · คปภ. · มติ ครม. · กทม.</span></div>"
        "<h2>สงสัยเรื่องเงินช่วยเหลือ หรือประกันภัยพิบัติ?</h2>"
        "<p>พิมพ์คำถามได้เลย น้องพร้อมรับจะค้นจากเอกสารและตอบพร้อมแหล่งอ้างอิง</p>",
        unsafe_allow_html=True,
    )
    with st.form("hero_form", clear_on_submit=True, border=False):
        f1, f2 = st.columns([5, 1.3])
        hero_q = f1.text_input(
            "คำถาม", placeholder="เช่น บ้านโดนพายุพัดหลังคาเสียหาย ได้เงินเท่าไหร่",
            label_visibility="collapsed",
        )
        if f2.form_submit_button("ถามเลย ➜") and hero_q.strip():
            go_ask(hero_q.strip())


# ---------------------------------------------------------------------------
# 4) เมนูหลัก
# ---------------------------------------------------------------------------
st.write("")
with st.container(key="navwrap"):
    st.segmented_control("เมนู", NAV_ITEMS, key="nav", label_visibility="collapsed")
nav = st.session_state.nav or NAV_CHAT


# ---------------------------------------------------------------------------
# หน้า: แชต RAG
# ---------------------------------------------------------------------------
def ask(question: str):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🌊"):
        with st.spinner("กำลังค้นข้อมูลให้..."):
            search_query = (
                condense_question(question, history) if st.session_state.use_condense else question
            )
            results = store.search(
                search_query, top_k=st.session_state.top_k, threshold=st.session_state.threshold
            )
        if not results:
            answer = NOT_FOUND_MESSAGE
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(stream_answer(build_messages(question, results, history)))
            except Exception as e:
                answer = "ขออภัย ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง"
                st.error(answer)
                if st.session_state.show_debug:
                    st.exception(e)
        render_sources(results, answer, search_query)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "results": results, "search_query": search_query}
    )


def page_chat():
    if get_groq_client() is None:
        st.error("ระบบแชตยังไม่พร้อมใช้งาน: ผู้ดูแลยังไม่ได้ตั้งค่า GROQ_API_KEY ใน Secrets")
        return

    head_l, head_r = st.columns([4, 1.4])
    head_l.markdown("### 💬 ถามน้องพร้อมรับ")
    if st.session_state.messages and head_r.button("✨ เริ่มแชตใหม่", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🌊"):
            st.markdown(
                "สวัสดีครับ ผม **น้องพร้อมรับ** 🌊 ถามเรื่องประกันภัยพิบัติแห่งชาติ "
                "เงินเยียวยาน้ำท่วม หรือการเตรียมตัวรับมือได้เลย ทุกคำตอบมีแหล่งอ้างอิงให้ตรวจสอบ"
            )
        st.caption("ลองถามเรื่องเหล่านี้")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"ex_{i}", width="stretch"):
                st.session_state.pending_question = q
                st.rerun()

    # กล่องประวัติแชตอยู่ "เหนือ" ช่องพิมพ์ คำตอบใหม่จะต่อท้ายในกล่องนี้
    history_box = st.container()
    with history_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "🌊"):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "results" in msg:
                    render_sources(msg["results"], msg["content"], msg.get("search_query", ""))

    question = st.chat_input("พิมพ์คำถามของคุณ...")
    if not question and st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")
    if question:
        with history_box:
            ask(question)


# ---------------------------------------------------------------------------
# หน้า: คำถามยอดฮิต (FAQ)
# ---------------------------------------------------------------------------
def toggle_saved(fid: str):
    saved = st.session_state.saved
    st.session_state.saved = [s for s in saved if s != fid] if fid in saved else saved + [fid]


def page_faq():
    st.markdown("### ❓ คำถามยอดฮิต")
    st.segmented_control(
        "หมวดหมู่", list(CATEGORIES), key="faq_cat",
        format_func=lambda c: CATEGORIES[c] + (f" ({len(st.session_state.saved)})" if c == "saved" else ""),
        label_visibility="collapsed",
    )
    query = st.text_input("ค้นหาในคำถามยอดฮิต", placeholder="🔍 พิมพ์คำค้น เช่น 9,000 / OTP / กรุงเทพ")
    cat = st.session_state.faq_cat or "all"

    def match(f):
        if cat == "saved":
            ok_cat = f["id"] in st.session_state.saved
        else:
            ok_cat = cat == "all" or f["category"] == cat
        text = " ".join([f["question"], f["answer"], *f["highlights"]]).lower()
        return ok_cat and (not query or query.lower() in text)

    items = [f for f in FAQS if match(f)]
    st.caption(f"พบ {len(items)} รายการ · กดที่คำถามเพื่อดูคำตอบ")
    if not items:
        st.info("ไม่พบคำถามที่ตรงกัน ลองถามน้องพร้อมรับในเมนูแชตได้เลย")
        if query:
            st.button(f"💬 ถามน้องพร้อมรับว่า “{query}”", on_click=go_ask, args=(query,))
        return

    for f in items:
        star = "⭐ " if f["id"] in st.session_state.saved else ""
        with st.expander(f"{star}{f['question']}"):
            st.markdown(f"<span class='badge'>✔ {esc(f['badge'])}</span>", unsafe_allow_html=True)
            st.markdown(f"**สรุป:** {f['answer']}")
            st.markdown(
                "<ul class='hl'>" + "".join(f"<li>{esc(h)}</li>" for h in f["highlights"]) + "</ul>",
                unsafe_allow_html=True,
            )
            st.caption(f"📄 อ้างอิงเอกสาร: {f['ref']}")
            b1, b2 = st.columns(2)
            b1.button(
                "⭐ เลิกบันทึก" if f["id"] in st.session_state.saved else "☆ บันทึกไว้อ่าน",
                key=f"save_{f['id']}", on_click=toggle_saved, args=(f["id"],), width="stretch",
            )
            b2.button(
                "💬 ถามต่อกับน้องพร้อมรับ", key=f"ask_{f['id']}", width="stretch",
                on_click=go_ask, args=(f["question"],),
            )


# ---------------------------------------------------------------------------
# หน้า: เช็กสิทธิ์เบื้องต้น (ใช้เกณฑ์จากเอกสารใน data/)
# ---------------------------------------------------------------------------
def page_calc():
    st.markdown("### 🧮 เช็กสิทธิ์และประมาณเงินช่วยเหลือเบื้องต้น")
    st.caption("คำนวณจากเกณฑ์ที่ประกาศ ณ 7 ต.ค. 2569 · ยอดจริงขึ้นกับการสำรวจและอนุมัติของหน่วยงาน")
    left, right = st.columns([1.25, 1])

    with left:
        when = st.radio(
            "1. ความเสียหายเกิดขึ้นเมื่อไหร่?",
            ["ก่อน 1 ต.ค. 2569", "ตั้งแต่ 1 ต.ค. 2569 เป็นต้นไป"], horizontal=True,
        )
        declared = st.checkbox("บ้านอยู่ในเขตที่ทางราชการประกาศเป็นพื้นที่ประสบภัย", value=True)
        lines, notes, total = [], [], 0

        if when.startswith("ก่อน"):
            situation = st.selectbox(
                "2. สภาพบ้านช่วงน้ำท่วม (15 พ.ค.–30 ก.ย. 69)",
                [
                    "น้ำท่วมเข้าบ้านไม่เกิน 7 วัน และทรัพย์สินเสียหาย",
                    "น้ำท่วมเข้าบ้านติดต่อกันเกิน 7 วัน",
                    "บ้านถูกน้ำล้อม (น้ำไม่เข้าบ้าน)",
                    "น้ำท่วมไม่เกิน 7 วัน และไม่มีทรัพย์สินเสียหาย",
                ],
            )
            repair = st.number_input("3. ประมาณค่าวัสดุซ่อมแซมบ้าน (บาท)", 0, 1_000_000, 0, step=1000)
            tools = st.number_input("4. ค่าเสียหายเครื่องมือประกอบอาชีพ (บาท)", 0, 1_000_000, 0, step=500)

            if declared:
                if situation.startswith("บ้านถูกน้ำล้อม"):
                    lines.append(("เงินเยียวยาตามมติ ครม. (บ้านถูกน้ำล้อม)", 5_000, "07_ddpm_relief_3_parts.md"))
                elif situation.startswith("น้ำท่วมไม่เกิน 7 วัน และไม่มี"):
                    notes.append("กรณีท่วมไม่เกิน 7 วันและไม่มีทรัพย์สินเสียหาย ไม่เข้าเกณฑ์เงิน 9,000 บาท")
                else:
                    lines.append(("เงินเยียวยาตามมติ ครม. 29 ก.ย. 69", 9_000, "08_relief_9000_baht.md"))
                if repair:
                    lines.append(("ค่าซ่อมแซมบ้าน (เงินทดรองราชการ สูงสุด 88,600)", min(repair, 88_600), "07_ddpm_relief_3_parts.md"))
                if tools:
                    lines.append(("เครื่องมือประกอบอาชีพ (สูงสุด 13,500)", min(tools, 13_500), "07_ddpm_relief_3_parts.md"))
                notes.append("ลงทะเบียน 9,000 บาท ผ่านแอป 'ทางรัฐ' หรือ อบต./เทศบาล · เงินซ่อมบ้านต้องรอ อปท. สำรวจ")
                notes.append("ความเสียหายก่อน 1 ต.ค. เคลมประกันภัยพิบัติไม่ได้")
        else:
            hazard = st.selectbox("2. ประเภทภัย", ["น้ำท่วม (อุทกภัย)", "ลมพายุ (วาตภัย)", "แผ่นดินไหว"])
            damage = st.number_input(
                "3. ประมาณค่าเสียหายของโครงสร้างบ้าน (บาท)", 0, 5_000_000, 0, step=1000,
                help="นับเฉพาะตัวบ้าน/สิ่งปลูกสร้าง/บิวต์อิน — ไม่รวมเครื่องใช้ไฟฟ้าและทรัพย์สินส่วนตัว",
            )
            pre_declared = st.checkbox("พื้นที่นี้ถูกประกาศเขตภัยไว้ตั้งแต่ก่อน 1 ต.ค. สำหรับภัยครั้งเดียวกัน")
            initial = 10_000 if hazard.startswith("น้ำท่วม") else 5_000

            if declared and not pre_declared:
                lines.append((f"ประกันภัยพิบัติ เงินเบื้องต้น ({hazard.split(' ')[0]})", initial, "02_coverage_amounts.md"))
                extra = max(0, min(damage, 100_000) - initial)
                if extra:
                    lines.append(("ประกันภัยพิบัติ จ่ายเพิ่มตามความเสียหายจริง", extra, "02_coverage_amounts.md"))
                notes.append("ไม่ต้องลงทะเบียนล่วงหน้า แจ้งเคลมที่ thainatcat.org / LINE @ThaiNATCAT / 02-012-5555")
                notes.append("รวมสูงสุด 100,000 บาทต่อหลังต่อครั้ง · ไม่คุ้มครองเครื่องใช้ไฟฟ้าและทรัพย์สินส่วนตัว")
            elif pre_declared:
                notes.append("พื้นที่ที่ประกาศเขตภัยไว้ก่อน 1 ต.ค. ไม่อยู่ในความคุ้มครองของประกันสำหรับภัยครั้งนั้น ให้ใช้สิทธิ์เยียวยาตามระเบียบเดิมแทน")

        if not declared:
            notes.insert(0, "ต้องอยู่ในเขตที่ประกาศเป็นพื้นที่ประสบภัยจึงจะมีสิทธิ์ ควรสอบถาม อปท. หรือ ปภ. 1784")
        total = sum(v for _, v, _ in lines)

    with right:
        rows = "".join(
            f"<div class='fact' style='border-color:#334155'><span>{esc(n)}<br>"
            f"<small style='color:#94A3B8'>📄 {esc(r)}</small></span><b style='color:#FBBF24'>{v:,.0f}</b></div>"
            for n, v, r in lines
        ) or "<div style='color:#94A3B8'>ยังไม่เข้าเกณฑ์ที่คำนวณได้</div>"
        st.markdown(
            f"<div class='result'><div class='lbl'>ประมาณการเงินช่วยเหลือรวม</div>"
            f"<div class='amt'>{total:,.0f} บาท</div>"
            f"<div style='text-align:left;margin-top:10px'>{rows}</div></div>",
            unsafe_allow_html=True,
        )
        for n in notes:
            st.info(n, icon="ℹ️")
        st.caption("ไม่รวมกรณีเสียชีวิต/บาดเจ็บ (ดูในคำถามยอดฮิต) · เป็นการประมาณเพื่อการศึกษา ไม่ใช่ผลอนุมัติจริง")


# ---------------------------------------------------------------------------
# หน้า: เอกสาร & ขั้นตอน (Checklist)
# ---------------------------------------------------------------------------
def page_docs_checklist():
    st.markdown("### 📑 เตรียมเอกสาร & ขั้นตอนการยื่น")
    kind = st.radio(
        "ต้องการยื่นเรื่องอะไร", list(CHECKLIST), horizontal=True,
        format_func=lambda k: CHECKLIST[k]["title"],
    )
    data = CHECKLIST[kind]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**✅ เช็กลิสต์เอกสาร**")
        done = sum(st.checkbox(item, key=f"chk_{kind}_{i}") for i, item in enumerate(data["items"]))
        pct = done / len(data["items"])
        st.progress(pct, text=f"ความพร้อม {pct:.0%} ({done}/{len(data['items'])})")
        if pct == 1:
            st.success("เอกสารพร้อมแล้ว ยื่นเรื่องได้เลย 🎉")
    with c2:
        st.markdown("**🪜 ขั้นตอน**")
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(data["steps"], 1)))
        st.caption(f"📄 อ้างอิงเอกสาร: {data['ref']}")
        st.warning("ช่องทางทางการไม่เคยขอ OTP หรือให้โอนเงินก่อน", icon="⚠️")


# ---------------------------------------------------------------------------
# หน้า: สายด่วน
# ---------------------------------------------------------------------------
def page_hotlines():
    st.markdown("### 📞 สายด่วน & หน่วยงานติดต่อ")
    st.caption("บนมือถือ กดที่เบอร์เพื่อโทรออกได้ทันที")
    cols = st.columns(2)
    for i, (name, desc, num, tel) in enumerate(HOTLINES):
        cols[i % 2].markdown(
            f"<div class='card dark' style='margin-bottom:10px'><div class='hot' style='margin:0'>"
            f"<div><div class='n'>{esc(name)}</div><div class='d'>{esc(desc)}</div></div>"
            f"<a href='tel:{tel}'>📞 {esc(num)}</a></div></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# แถบขวา: สรุปสิทธิ์ด่วน + สายด่วน
# ---------------------------------------------------------------------------
def side_panel():
    st.markdown(
        """
        <div class="card">
          <h3>⚡ สรุปสิทธิ์ด่วน</h3>
          <div class="fact"><span>เริ่มคุ้มครอง</span><b>1 ต.ค. 69</b></div>
          <div class="fact"><span>น้ำท่วม เงินเบื้องต้น</span><b>10,000 บาท</b></div>
          <div class="fact"><span>พายุ / แผ่นดินไหว</span><b>5,000 บาท</b></div>
          <div class="fact"><span>สูงสุดต่อหลังต่อครั้ง</span><b>100,000 บาท</b></div>
          <div class="fact"><span>เยียวยา (ท่วมก่อน 1 ต.ค.)</span><b>9,000 บาท</b></div>
          <div class="fact"><span>ลงทะเบียนประกัน</span><b>ไม่ต้อง</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    hot = "".join(
        f"<div class='hot'><div><div class='n'>{esc(n)}</div><div class='d'>{esc(d)}</div></div>"
        f"<a href='tel:{t}'>{esc(num)}</a></div>"
        for n, d, num, t in HOTLINES[:4]
    )
    st.markdown(f"<div class='card dark'><h3>🎧 สายด่วนสำคัญ</h3>{hot}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# หน้า: ผู้ดูแลระบบ — ทดสอบ / เอกสาร
# ---------------------------------------------------------------------------
def page_eval():
    st.markdown("### 🧪 ทดสอบระบบด้วย test_questions.csv")
    top_k = st.session_state.top_k
    df = pd.read_csv(TEST_CSV)
    with st.expander(f"ดูชุดคำถามทดสอบ ({len(df)} ข้อ)"):
        st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("**1) Retrieval** — Hit@k: ใน Top-k มีไฟล์ที่ควรเจออย่างน้อย 1 รายการหรือไม่")
    rows = []
    for _, r in df.iterrows():
        res = store.search(r["question"], top_k=top_k)
        files = [x["file_name"] for x in res]
        expected = "" if pd.isna(r.get("expected_source")) else str(r["expected_source"])
        answerable = str(r["answerable"]).strip().lower() in ("yes", "true", "1")
        hit = (expected in files) if answerable else None
        rows.append({
            "id": r["id"], "question": r["question"], "answerable": answerable,
            "expected_source": expected or "-", "top1_file": files[0] if files else "-",
            "top1_score": round(res[0]["score"], 3) if res else 0.0,
            f"hit@{top_k}": "✅" if hit else ("❌" if hit is False else "—"),
        })
    ret_df = pd.DataFrame(rows)
    ans_rows = ret_df[ret_df["answerable"]]
    st.metric(f"Hit Rate@{top_k}", f"{(ans_rows[f'hit@{top_k}'] == '✅').mean():.0%}" if len(ans_rows) else "-")
    st.dataframe(ret_df, width="stretch", hide_index=True)

    st.markdown("**2) End-to-end** — คำถามที่ไม่มีคำตอบต้องถูกปฏิเสธ และคำถามที่มีคำตอบต้องไม่ถูกปฏิเสธ")
    if get_groq_client() is None:
        st.warning("ต้องตั้งค่า GROQ_API_KEY ก่อน")
        return
    if st.button("▶️ รันคำถามทดสอบทั้งหมด", key="run_eval"):
        out = []
        bar = st.progress(0.0)
        for i, r in df.iterrows():
            res = store.search(r["question"], top_k=top_k, threshold=st.session_state.threshold)
            if res:
                ans = llm_complete(build_messages(r["question"], res)).choices[0].message.content or ""
            else:
                ans = NOT_FOUND_MESSAGE
            answerable = str(r["answerable"]).strip().lower() in ("yes", "true", "1")
            out.append({
                "id": r["id"], "question": r["question"], "expected_answer": r["expected_answer"],
                "system_answer": ans, "behavior_ok": "✅" if is_refusal(ans) != answerable else "❌",
                "cited": ", ".join(f"[{n}]" for n in sorted(cited_ranks(ans))) or "-",
            })
            bar.progress((i + 1) / len(df))
        out_df = pd.DataFrame(out)
        st.metric("ตอบ/ปฏิเสธ ได้ถูกประเภท", f"{(out_df['behavior_ok'] == '✅').mean():.0%}")
        st.dataframe(out_df, width="stretch", hide_index=True)


def page_docs_admin():
    st.markdown("### 📁 เอกสารความรู้ในระบบ")
    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนไฟล์", len(docs))
    c2.metric("จำนวน chunks", len(chunks))
    c3.metric("ตัวอักษรรวม", f"{sum(len(d.text) for d in docs):,}")
    for d in docs:
        n = sum(1 for c in chunks if c.file_name == d.file_name)
        with st.expander(f"📄 {d.title}  ·  {n} chunks"):
            st.caption(f"ไฟล์: `{d.file_name}` · ข้อมูล ณ {d.as_of}")
            st.caption(f"แหล่งที่มา: {d.source}")
            for c in chunks:
                if c.file_name == d.file_name:
                    st.markdown(f"**chunk #{c.chunk_id} — {c.section}** ({len(c.text)} ตัวอักษร)")
                    st.text(c.text.split("\n", 2)[-1])


# ---------------------------------------------------------------------------
# วางเลย์เอาต์หลัก
# ---------------------------------------------------------------------------
st.write("")
if st.session_state.admin_page == "🧪 ทดสอบระบบ":
    page_eval()
elif st.session_state.admin_page == "📁 เอกสารในระบบ":
    page_docs_admin()
else:
    main_col, side_col = st.columns([2.3, 1], gap="large")
    with main_col:
        {
            NAV_CHAT: page_chat, NAV_FAQ: page_faq, NAV_CALC: page_calc,
            NAV_DOCS: page_docs_checklist, NAV_CALL: page_hotlines,
        }[nav]()
    with side_col:
        side_panel()

st.markdown(
    """
    <div class="foot">
      <b>ศูนย์ข้อมูลเยียวยาน้ำท่วม &amp; ประกันภัยพิบัติ 2569</b><br>
      รวบรวมจากข่าวและประกาศของ กรมป้องกันและบรรเทาสาธารณภัย (ปภ.), สำนักงาน คปภ., กรมประชาสัมพันธ์,
      มติคณะรัฐมนตรี และกรุงเทพมหานคร ข้อมูล ณ 7 ต.ค. 2569<br>
      ⚠️ เว็บไซต์นี้เป็นโครงงานเพื่อการศึกษา ไม่ใช่ช่องทางทางการ โปรดตรวจสอบสิทธิ์จริงกับ ThaiNATCAT 02-012-5555 หรือ ปภ. 1784
    </div>
    """,
    unsafe_allow_html=True,
)
