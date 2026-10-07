"""
app.py — น้องพร้อมรับ: แชตบอต RAG ตอบคำถามประกันภัยพิบัติแห่งชาติ & เงินเยียวยาน้ำท่วม 2569
รันในเครื่อง:  streamlit run app.py

โครงหน้าเว็บ
- หน้าหลัก (ผู้ใช้ทั่วไป): แชต + คำถามแนะนำ + แหล่งอ้างอิงใต้คำตอบ
- แถบด้านข้าง: ปุ่มเริ่มแชตใหม่, เบอร์สำคัญ, และเมนู "ผู้ดูแลระบบ" (ตั้งค่าการค้นหา / ทดสอบระบบ / ดูเอกสาร)
"""

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_CSV = os.path.join(BASE_DIR, "test_questions.csv")
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

EXAMPLE_QUESTIONS = [
    ("💧", "น้ำท่วมบ้าน ได้เงินเท่าไหร่"),
    ("📝", "ต้องลงทะเบียนก่อนไหม"),
    ("📅", "บ้านท่วมตั้งแต่เดือนกันยายน เคลมได้ไหม"),
    ("📲", "แจ้งเคลมทางไหน ใช้เอกสารอะไร"),
    ("📺", "ทีวี ตู้เย็นเสียหาย ได้รับความคุ้มครองไหม"),
    ("⚠️", "มีคนโทรมาขอ OTP บอกจะโอนเงินเยียวยา"),
]

HOTLINES = [
    ("ประกันภัยพิบัติ (ThaiNATCAT)", "02-012-5555"),
    ("ปภ. แจ้งเหตุสาธารณภัย", "1784"),
    ("เจ็บป่วยฉุกเฉิน", "1669"),
    ("กทม.", "1555"),
    ("แจ้งถูกหลอกออนไลน์", "1441"),
]

st.set_page_config(
    page_title="น้องพร้อมรับ | ถามเรื่องประกันภัยพิบัติ & เงินเยียวยาน้ำท่วม",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# สไตล์หน้าเว็บ
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, .stChatMessage, button, input, textarea {
        font-family: 'Sarabun', sans-serif !important;
    }
    #MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 3.5rem; padding-bottom: 6rem; max-width: 760px;}
    .hero {
        background: linear-gradient(135deg, #0E7490 0%, #0891B2 60%, #22D3EE 100%);
        color: #fff; border-radius: 18px; padding: 22px 24px; margin-bottom: 14px;
    }
    .hero h1 {color:#fff; font-size: 1.65rem; margin: 0 0 4px 0; padding:0;}
    .hero p {margin: 0; opacity: .95; font-size: 1.02rem;}
    .facts {display:flex; gap:10px; margin: 6px 0 18px 0; flex-wrap: wrap;}
    .fact {
        flex: 1 1 150px; background:#F0F9FB; border:1px solid #CDEAF0; border-radius:14px;
        padding:10px 14px;
    }
    .fact .v {font-size:1.15rem; font-weight:700; color:#0E7490; line-height:1.3;}
    .fact .k {font-size:.85rem; color:#4B5563;}
    .src-title {font-weight:600; margin-bottom:2px;}
    .src-meta {font-size:.82rem; color:#6B7280; margin-bottom:6px;}
    .src-body {font-size:.9rem; background:#F9FAFB; border-left:3px solid #0891B2;
               padding:8px 10px; border-radius:6px; white-space:pre-wrap;}
    .note {font-size:.8rem; color:#6B7280; text-align:center; margin-top: 8px;}
    div[data-testid="stButton"] > button {border-radius: 999px;}
    </style>
    """,
    unsafe_allow_html=True,
)


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
    except Exception:  # ยังไม่มีไฟล์ secrets หรือยังไม่ได้ใส่ key
        api_key = None
    return _make_groq_client(api_key) if api_key else None


def get_model_name() -> str:
    try:
        return st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)
    except Exception:
        return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# ค่าตั้งค่า (เก็บใน session_state เพื่อให้เมนูผู้ดูแลระบบปรับได้)
# ---------------------------------------------------------------------------
DEFAULTS = {"top_k": 4, "threshold": 0.0, "use_condense": True, "show_debug": False, "page": "chat"}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("messages", [])


# ---------------------------------------------------------------------------
# เรียก LLM
# ---------------------------------------------------------------------------
def llm_complete(messages, temperature=0.2, stream=False):
    client = get_groq_client()
    model = get_model_name()
    try:
        return client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, stream=stream
        )
    except Exception as e:
        # ถ้าชื่อโมเดลถูกปลดระวาง ให้ลองโมเดลสำรอง
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
    prompt = CONDENSE_PROMPT.format(history=history_text, question=question)
    try:
        resp = llm_complete([{"role": "user", "content": prompt}], temperature=0.0)
        rewritten = (resp.choices[0].message.content or "").strip().splitlines()[0]
        return rewritten or question
    except Exception:
        return question


def stream_answer(messages):
    """generator สำหรับ st.write_stream"""
    resp = llm_complete(messages, stream=True)
    for chunk in resp:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# แสดงแหล่งอ้างอิง (แบบเรียบง่ายสำหรับผู้ใช้ / แบบละเอียดเมื่อเปิดโหมดผู้ดูแล)
# ---------------------------------------------------------------------------
def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_sources(results: list[dict], answer: str, search_query: str):
    used = cited_ranks(answer)
    refused = is_refusal(answer)
    debug = st.session_state.show_debug

    if refused:
        shown = results
        label = "🔍 เอกสารที่ใกล้เคียงที่สุด (ไม่มีคำตอบของคำถามนี้)"
    else:
        shown = [r for r in results if r["rank"] in used] or results
        label = f"📚 ดูแหล่งอ้างอิง ({len(shown)})"

    if debug:
        shown = results  # โหมดผู้ดูแล: แสดงทุก chunk ที่ค้นได้

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
                f"<div class='src-title'>[{r['rank']}] {_esc(r['doc_title'])}</div>"
                f"<div class='src-meta'>หัวข้อ: {_esc(r['section'])} · {_esc(meta)}</div>"
                f"<div class='src-body'>{_esc(body)}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"ที่มา: {r['source']}")


# ---------------------------------------------------------------------------
# โหลดข้อมูล
# ---------------------------------------------------------------------------
docs, chunks, store = get_vector_store()


# ---------------------------------------------------------------------------
# Sidebar: สำหรับผู้ใช้ด้านบน / เมนูผู้ดูแลระบบซ่อนไว้ด้านล่าง
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌊 น้องพร้อมรับ")
    if st.button("✨ เริ่มแชตใหม่", width="stretch"):
        st.session_state.messages = []
        st.session_state.page = "chat"
        st.rerun()

    st.markdown("**📞 เบอร์สำคัญ**")
    for name, num in HOTLINES:
        st.markdown(f"{name}  \n**{num}**")

    st.divider()
    with st.expander("🛠️ ผู้ดูแลระบบ"):
        page_labels = {"chat": "💬 แชต", "eval": "🧪 ทดสอบระบบ", "docs": "📁 เอกสารในระบบ"}
        st.radio(
            "หน้า", list(page_labels), key="page",
            format_func=lambda p: page_labels[p], label_visibility="collapsed",
        )
        st.markdown("**ตั้งค่าการค้นหา**")
        st.slider("Top-k (จำนวน chunk ที่ดึงมา)", 1, 8, key="top_k")
        st.slider(
            "Similarity Threshold (0 = ปิด)", 0.0, 0.95, step=0.01, key="threshold",
            help="chunk ที่คะแนนต่ำกว่านี้จะไม่ถูกส่งให้ LLM ถ้าไม่เหลือเลย ระบบตอบว่าไม่พบข้อมูลทันที",
        )
        st.toggle("เขียนคำถามต่อเนื่องใหม่ก่อนค้นหา", key="use_condense")
        st.toggle("แสดงรายละเอียดการค้นหา (score, ไฟล์)", key="show_debug")
        st.caption(
            f"{len(docs)} ไฟล์ · {len(chunks)} chunks  \n"
            f"Embedding: `{EMBEDDING_MODEL_NAME}`  \nLLM: `{get_model_name()}` (Groq)"
        )


# ---------------------------------------------------------------------------
# หน้า: แชต (หน้าหลัก)
# ---------------------------------------------------------------------------
def ask(question: str):
    """ประมวลผลคำถาม 1 ข้อ: ค้นหา -> สร้าง prompt -> สตรีมคำตอบ -> แสดงแหล่งอ้างอิง"""
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
        st.error("ระบบยังไม่พร้อมใช้งาน: ผู้ดูแลยังไม่ได้ตั้งค่า GROQ_API_KEY ใน Secrets")
        st.stop()

    first_visit = not st.session_state.messages

    if first_visit:
        st.markdown(
            """
            <div class="hero">
              <h1>🌊 น้องพร้อมรับ</h1>
              <p>ถามเรื่องประกันภัยพิบัติแห่งชาติ และเงินเยียวยาน้ำท่วม ปี 2569 ได้เลย
              ทุกคำตอบอ้างอิงจากข่าวและประกาศทางการ</p>
            </div>
            <div class="facts">
              <div class="fact"><div class="v">1 ต.ค. 69</div><div class="k">เริ่มคุ้มครอง</div></div>
              <div class="fact"><div class="v">100,000 บาท</div><div class="k">สูงสุดต่อหลังต่อครั้ง</div></div>
              <div class="fact"><div class="v">ไม่ต้องลงทะเบียน</div><div class="k">ได้สิทธิ์อัตโนมัติ</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**ลองถามเรื่องเหล่านี้**")
        cols = st.columns(2)
        for i, (icon, q) in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(f"{icon} {q}", key=f"ex_{i}", width="stretch"):
                st.session_state.pending_question = q
                st.rerun()
    else:
        st.markdown("#### 🌊 น้องพร้อมรับ")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "🌊"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "results" in msg:
                render_sources(msg["results"], msg["content"], msg.get("search_query", ""))

    question = st.chat_input("พิมพ์คำถาม เช่น หลังคาบ้านโดนพายุพัด ได้เงินเท่าไหร่")
    if not question and st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")
    if question:
        ask(question)

    st.markdown(
        "<div class='note'>ข้อมูล ณ 7 ต.ค. 2569 ใช้เพื่อการศึกษา · ตรวจสอบสิทธิ์จริงได้ที่ "
        "ThaiNATCAT 02-012-5555 หรือ ปภ. 1784</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# หน้า: ทดสอบระบบ (ผู้ดูแล)
# ---------------------------------------------------------------------------
def page_eval():
    st.subheader("🧪 ทดสอบระบบด้วย test_questions.csv")
    top_k = st.session_state.top_k
    df = pd.read_csv(TEST_CSV)
    with st.expander(f"ดูชุดคำถามทดสอบ ({len(df)} ข้อ)"):
        st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("**1) Retrieval** — Hit@k: ใน Top-k มีไฟล์ที่ควรเจออย่างน้อย 1 รายการหรือไม่")
    rows = []
    for _, r in df.iterrows():
        res = store.search(r["question"], top_k=top_k)
        files = [x["file_name"] for x in res]
        expected = str(r.get("expected_source", "") or "")
        if expected == "nan":
            expected = ""
        answerable = str(r["answerable"]).strip().lower() in ("yes", "true", "1")
        hit = (expected in files) if answerable else None
        rows.append({
            "id": r["id"], "question": r["question"], "answerable": answerable,
            "expected_source": expected or "-", "top1_file": files[0] if files else "-",
            "top1_score": round(res[0]["score"], 3) if res else 0.0,
            f"hit@{top_k}": "✅" if hit else ("❌" if hit is False else "—"),
        })
    ret_df = pd.DataFrame(rows)
    answerable_rows = ret_df[ret_df["answerable"]]
    hit_rate = (answerable_rows[f"hit@{top_k}"] == "✅").mean() if len(answerable_rows) else 0
    st.metric(f"Hit Rate@{top_k}", f"{hit_rate:.0%}")
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
                resp = llm_complete(build_messages(r["question"], res))
                ans = resp.choices[0].message.content or ""
            else:
                ans = NOT_FOUND_MESSAGE
            answerable = str(r["answerable"]).strip().lower() in ("yes", "true", "1")
            out.append({
                "id": r["id"], "question": r["question"], "expected_answer": r["expected_answer"],
                "system_answer": ans,
                "behavior_ok": "✅" if is_refusal(ans) != answerable else "❌",
                "cited": ", ".join(f"[{n}]" for n in sorted(cited_ranks(ans))) or "-",
            })
            bar.progress((i + 1) / len(df))
        out_df = pd.DataFrame(out)
        st.metric("ตอบ/ปฏิเสธ ได้ถูกประเภท", f"{(out_df['behavior_ok'] == '✅').mean():.0%}")
        st.dataframe(out_df, width="stretch", hide_index=True)
        st.caption("เทียบ system_answer กับ expected_answer ด้วยตนเองเพื่อตรวจความถูกต้องของเนื้อหา")


# ---------------------------------------------------------------------------
# หน้า: เอกสารในระบบ (ผู้ดูแล)
# ---------------------------------------------------------------------------
def page_docs():
    st.subheader("📁 เอกสารความรู้ในระบบ")
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


if st.session_state.page == "eval":
    page_eval()
elif st.session_state.page == "docs":
    page_docs()
else:
    page_chat()
