"""
app.py — น้องพร้อมรับ: แชตบอต RAG ตอบคำถามประกันภัยพิบัติแห่งชาติ & เงินเยียวยาน้ำท่วม 2569
รันในเครื่อง:  streamlit run app.py
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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEST_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_questions.csv")
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

EXAMPLE_QUESTIONS = [
    "น้ำท่วมบ้าน ประกันภัยพิบัติจ่ายเท่าไหร่",
    "ต้องลงทะเบียนก่อนไหมถึงจะได้รับความคุ้มครอง",
    "บ้านท่วมตั้งแต่เดือนกันยายน เคลมประกันได้ไหม",
    "แจ้งเคลมได้ทางไหนบ้าง ต้องใช้เอกสารอะไร",
    "ทีวีกับตู้เย็นเสียหายจากน้ำท่วม ได้รับความคุ้มครองไหม",
    "มีคนโทรมาขอ OTP บอกว่าจะโอนเงินเยียวยา ทำยังไงดี",
]

st.set_page_config(page_title="น้องพร้อมรับ | ประกันภัยพิบัติ & เยียวยาน้ำท่วม", page_icon="🌊", layout="wide")


# ---------------------------------------------------------------------------
# โหลดโมเดลและ index เพียงครั้งเดียว (cache) — สำคัญมากบน Streamlit Community Cloud
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล Embedding (ครั้งแรกอาจใช้เวลา 1-2 นาที)...")
def get_embedder():
    return make_sentence_transformer_embedder(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="กำลังเตรียมเอกสารและสร้าง FAISS index...")
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
# แสดงแหล่งอ้างอิง
# ---------------------------------------------------------------------------
def render_sources(results: list[dict], answer: str, search_query: str):
    used = cited_ranks(answer)
    refused = is_refusal(answer)
    if refused:
        label = f"📚 เอกสารที่ระบบค้นพบ {len(results)} รายการ (ไม่มีข้อมูลที่ตอบคำถามนี้ได้)"
    else:
        label = f"📚 แหล่งอ้างอิงที่ใช้ตอบ ({len(used) or len(results)} จาก {len(results)} รายการที่ค้นพบ)"
    with st.expander(label, expanded=False):
        st.caption(f"🔎 คำค้นที่ใช้ค้นหา: {search_query}")
        if not results:
            st.info("ไม่พบเอกสารที่ผ่านเกณฑ์ความคล้าย (Similarity Threshold)")
        for r in results:
            mark = "✅ ใช้ตอบ" if r["rank"] in used else "▫️ ค้นพบ"
            st.markdown(
                f"**[{r['rank']}] {r['doc_title']}** — {r['section']}  \n"
                f"{mark} · score = `{r['score']:.3f}` · ไฟล์ `{r['file_name']}` · ข้อมูล ณ {r['as_of']}"
            )
            st.caption(f"ที่มา: {r['source']}")
            body = r["text"].split("\n", 2)[-1]
            st.text(body[:600] + ("..." if len(body) > 600 else ""))
            st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
docs, chunks, store = get_vector_store()

with st.sidebar:
    st.title("🌊 น้องพร้อมรับ")
    st.caption("ผู้ช่วยตอบคำถามประกันภัยพิบัติแห่งชาติ & เงินเยียวยาน้ำท่วม ปี 2569")
    page = st.radio("เมนู", ["💬 แชตถามตอบ", "📊 ทดสอบระบบ", "📁 เอกสารในระบบ"], label_visibility="collapsed")

    st.subheader("⚙️ ตั้งค่าการค้นหา")
    top_k = st.slider("Top-k (จำนวน chunk ที่ดึงมา)", 1, 8, 4)
    threshold = st.slider(
        "Similarity Threshold (0 = ปิด)", 0.0, 0.95, 0.0, 0.01,
        help="chunk ที่คะแนนต่ำกว่านี้จะไม่ถูกส่งให้ LLM ถ้าไม่เหลือเลย ระบบจะตอบว่าไม่พบข้อมูลทันที",
    )
    use_condense = st.toggle("เขียนคำถามต่อเนื่องใหม่ก่อนค้นหา", value=True,
                             help="ช่วยให้คำถามอย่าง 'แล้วถ้าเป็นพายุล่ะ' ค้นเอกสารได้ถูก")

    if st.button("🗑️ ล้างประวัติแชต", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        f"เอกสาร {len(docs)} ไฟล์ · {len(chunks)} chunks  \n"
        f"Embedding: `{EMBEDDING_MODEL_NAME}`  \nLLM: `{get_model_name()}` (Groq)"
    )
    st.caption("⚠️ ข้อมูลรวบรวม ณ 7 ต.ค. 2569 จากข่าวและประกาศทางการ ใช้เพื่อการศึกษา "
               "โปรดตรวจสอบสิทธิ์จริงกับ ThaiNATCAT 02-012-5555 หรือ ปภ. 1784")


# ---------------------------------------------------------------------------
# หน้า 1: แชต
# ---------------------------------------------------------------------------
def page_chat():
    st.header("💬 ถามเรื่องประกันภัยพิบัติ & เงินเยียวยาน้ำท่วม")

    if get_groq_client() is None:
        st.error("ยังไม่ได้ตั้งค่า GROQ_API_KEY ใน Secrets — ไปที่ App settings → Secrets แล้วเพิ่ม "
                 '`GROQ_API_KEY = "gsk_..."`')
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.info("ลองกดคำถามตัวอย่าง หรือพิมพ์คำถามของคุณที่ช่องด้านล่าง")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"ex_{i}", width="stretch"):
                st.session_state.pending_question = q
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "🌊"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "results" in msg:
                render_sources(msg["results"], msg["content"], msg.get("search_query", ""))

    question = st.chat_input("พิมพ์คำถาม เช่น บ้านโดนพายุพัดหลังคาเสียหาย ได้เงินเท่าไหร่")
    if not question and st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🌊"):
        with st.spinner("กำลังค้นหาเอกสาร..."):
            search_query = condense_question(question, history) if use_condense else question
            results = store.search(search_query, top_k=top_k, threshold=threshold)

        if not results:
            answer = NOT_FOUND_MESSAGE
            st.markdown(answer)
        else:
            messages = build_messages(question, results, history)
            try:
                answer = st.write_stream(stream_answer(messages))
            except Exception as e:
                answer = f"เกิดข้อผิดพลาดในการเรียก LLM: {e}"
                st.error(answer)
        render_sources(results, answer, search_query)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "results": results, "search_query": search_query}
    )


# ---------------------------------------------------------------------------
# หน้า 2: ทดสอบระบบด้วย test_questions.csv
# ---------------------------------------------------------------------------
def page_eval():
    st.header("📊 ทดสอบระบบด้วยชุดคำถาม test_questions.csv")
    df = pd.read_csv(TEST_CSV)
    st.dataframe(df, width="stretch", hide_index=True)

    st.subheader("1) ประเมิน Retrieval (ไม่ใช้ LLM)")
    st.caption("Hit@k = ใน Top-k มีไฟล์ที่ควรเจอ (expected_source) อย่างน้อย 1 รายการหรือไม่ — คิดเฉพาะคำถามที่มีคำตอบ")
    rows = []
    for _, r in df.iterrows():
        res = store.search(r["question"], top_k=top_k)
        files = [x["file_name"] for x in res]
        expected = str(r.get("expected_source", "") or "")
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
    st.metric(f"Hit Rate@{top_k}", f"{hit_rate:.0%}", help="ค่าเฉลี่ยของ 0/1 ทุกคำถามที่มีคำตอบ")
    st.dataframe(ret_df, width="stretch", hide_index=True)

    st.subheader("2) ประเมินคำตอบจาก LLM (End-to-end)")
    st.caption("จะเรียก Groq API 1 ครั้งต่อคำถาม · ตรวจว่าคำถามที่ไม่มีคำตอบถูกปฏิเสธ และคำถามที่มีคำตอบไม่ถูกปฏิเสธ")
    if get_groq_client() is None:
        st.warning("ต้องตั้งค่า GROQ_API_KEY ก่อน")
        return
    if st.button("▶️ รันคำถามทดสอบทั้งหมด", key="run_eval"):
        out = []
        bar = st.progress(0.0)
        for i, r in df.iterrows():
            res = store.search(r["question"], top_k=top_k, threshold=threshold)
            if res:
                resp = llm_complete(build_messages(r["question"], res))
                ans = resp.choices[0].message.content or ""
            else:
                ans = NOT_FOUND_MESSAGE
            answerable = str(r["answerable"]).strip().lower() in ("yes", "true", "1")
            refused = is_refusal(ans)
            out.append({
                "id": r["id"], "question": r["question"], "expected_answer": r["expected_answer"],
                "system_answer": ans,
                "behavior_ok": "✅" if refused != answerable else "❌",
                "cited": ", ".join(f"[{n}]" for n in sorted(cited_ranks(ans))) or "-",
            })
            bar.progress((i + 1) / len(df))
        out_df = pd.DataFrame(out)
        ok_rate = (out_df["behavior_ok"] == "✅").mean()
        st.metric("ตอบ/ปฏิเสธ ได้ถูกประเภท", f"{ok_rate:.0%}")
        st.dataframe(out_df, width="stretch", hide_index=True)
        st.caption("คอลัมน์ expected_answer ใช้เทียบความถูกต้องของเนื้อหาด้วยตนเอง (manual check)")


# ---------------------------------------------------------------------------
# หน้า 3: เอกสารในระบบ
# ---------------------------------------------------------------------------
def page_docs():
    st.header("📁 เอกสารความรู้ในระบบ")
    total_chars = sum(len(d.text) for d in docs)
    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนไฟล์", len(docs))
    c2.metric("จำนวน chunks", len(chunks))
    c3.metric("ตัวอักษรรวม", f"{total_chars:,}")

    for d in docs:
        n = sum(1 for c in chunks if c.file_name == d.file_name)
        with st.expander(f"📄 {d.title}  ·  {n} chunks"):
            st.caption(f"ไฟล์: `{d.file_name}` · ข้อมูล ณ {d.as_of}")
            st.caption(f"แหล่งที่มา: {d.source}")
            for c in chunks:
                if c.file_name == d.file_name:
                    st.markdown(f"**chunk #{c.chunk_id} — {c.section}** ({len(c.text)} ตัวอักษร)")
                    st.text(c.text.split("\n", 2)[-1])


if page.startswith("💬"):
    page_chat()
elif page.startswith("📊"):
    page_eval()
else:
    page_docs()
