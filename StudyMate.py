
import os
import io
import time
from typing import List, Dict, Tuple

import streamlit as st
import fitz  
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import Model
    _WATSONX_AVAILABLE = True
except Exception:
    _WATSONX_AVAILABLE = False


def load_env():
    """Load .env if present and return watsonx config."""
    load_dotenv(override=False)
    return {
        "api_key": os.getenv("WATSONX_API_KEY", ""),
        "url": os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        "project_id": os.getenv("WATSONX_PROJECT_ID", ""),
        "model_id": os.getenv("WATSONX_MODEL_ID", "mistralai/mixtral-8x7b-instruct-v0.1"),
    }


def extract_text_from_pdf(file_bytes: bytes) -> List[Dict]:
    """Extract text by page using PyMuPDF.

    Returns a list of dicts [{"page": int, "text": str}] for pages with non-empty text.
    """
    results = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            text = text.strip()
            if text:
                results.append({"page": i + 1, "text": text})
    return results


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200) -> List[str]:
    """Simple character-based chunking with overlap.
    The sizes are tuned for MiniLM embeddings and typical Q&A context windows.
    """
    chunks = []
    if not text:
        return chunks
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_corpus(pages: List[Dict], doc_name: str) -> List[Dict]:
    """Create a flat list of chunk records with metadata."""
    corpus = []
    for p in pages:
        for ch in chunk_text(p["text"]):
            corpus.append({
                "doc": doc_name,
                "page": p["page"],
                "chunk": ch,
            })
    return corpus


def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def search_index(index: faiss.Index, query_emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(query_emb, k)
    return D, I


def format_context(chunks: List[Dict]) -> str:
    """Create a concise context string for the LLM with light separators and metadata markers."""
    blocks = []
    for i, ch in enumerate(chunks, 1):
        header = f"[Source {i}] Document: {ch['doc']} | Page: {ch['page']}"
        blocks.append(f"{header}\n{ch['chunk']}")
    return "\n\n---\n\n".join(blocks)


def build_llm_prompt(question: str, context: str) -> str:
    return (
        "You are StudyMate, an academic assistant. Answer the student's question\n"
        "USING ONLY the information in the provided sources. If the answer is not in the\n"
        "sources, say you don't know and suggest where it might be in the docs.\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{context}\n\n"
        "Provide a concise, well-structured answer with bullet points when helpful.\n"
        "Cite your sources inline as [Source #] where appropriate."
    )


def generate_answer_watsonx(prompt: str, cfg: Dict) -> str:
    """Call IBM watsonx.ai foundation model (Mixtral-8x7B-Instruct by default).

    Requires env: WATSONX_API_KEY, WATSONX_URL, WATSONX_PROJECT_ID, WATSONX_MODEL_ID
    """
    if not _WATSONX_AVAILABLE:
        return (
            "[watsonx.ai SDK not installed] Please `pip install ibm-watsonx-ai` "
            "or enable it in your environment."
        )

    if not (cfg.get("api_key") and cfg.get("url") and cfg.get("project_id")):
        return (
            "[watsonx.ai not configured] Set WATSONX_API_KEY, WATSONX_URL, "
            "and WATSONX_PROJECT_ID (and optionally WATSONX_MODEL_ID)."
        )

    creds = Credentials(api_key=cfg["api_key"], url=cfg["url"])  
    model = Model(
        model_id=cfg.get("model_id", "mistralai/mixtral-8x7b-instruct-v0.1"),
        credentials=creds,
        project_id=cfg["project_id"],
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 512,
            "temperature": 0.2,
            "stop_sequences": [],
        },
    )
    try:
        resp = model.generate(prompt=prompt)
        if isinstance(resp, dict) and resp.get("results"):
            return resp["results"][0].get("generated_text", "")
        return str(resp)
    except Exception as e:
        return f"[watsonx.ai error] {e}"



st.set_page_config(page_title="StudyMate – PDF Q&A", page_icon="📘", layout="wide")

st.title("📘 StudyMate: AI-Powered Academic PDF Q&A")

cfg = load_env()

with st.sidebar:
    st.header("Configuration")
    st.caption("Watsonx.ai (LLM) configuration – from environment variables")
    st.text_input("Model ID", value=cfg.get("model_id", ""), disabled=True)
    api_ok = bool(cfg.get("api_key"))
    proj_ok = bool(cfg.get("project_id"))
    st.markdown(
        f"**API Key:** {'✅ set' if api_ok else '❌ missing'}\n\n"
        f"**Project ID:** {'✅ set' if proj_ok else '❌ missing'}\n\n"
        f"**Region URL:** `{cfg.get('url', '')}`"
    )

    st.divider()
    st.subheader("Embedding Model")
    emb_model_name = st.text_input(
        "SentenceTransformer (embeddings)",
        value="sentence-transformers/all-MiniLM-L6-v2",
        help="Swap to a multilingual or domain-specific model if needed.",
    )
    top_k = st.slider("Top-K passages", 3, 10, 5)

st.write(
    "Upload one or more PDFs, index them, and ask questions in natural language. "
    "Answers are grounded in your documents with inline source citations."
)

uploaded_files = st.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)


ss = st.session_state
if "corpus" not in ss:
    ss.corpus: List[Dict] = []
if "embeddings" not in ss:
    ss.embeddings = None
if "index" not in ss:
    ss.index = None
if "embedder_name" not in ss:
    ss.embedder_name = None
if "embedder" not in ss:
    ss.embedder = None

col_idx, col_q = st.columns([1, 2])

with col_idx:
    if st.button("📥 Ingest & Index PDFs", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            ss.corpus = []
            total_pages = 0
            with st.spinner("Extracting text and chunking..."):
                for file in uploaded_files:
                    name = file.name
                    data = file.read()
                    pages = extract_text_from_pdf(data)
                    total_pages += len(pages)
                    corpus = build_corpus(pages, name)
                    ss.corpus.extend(corpus)

            if not ss.corpus:
                st.error("No extractable text found in the uploaded PDFs.")
            else:
    
                if ss.embedder_name != emb_model_name or ss.embedder is None:
                    with st.spinner(f"Loading embedder: {emb_model_name}"):
                        ss.embedder = load_embedder(emb_model_name)
                        ss.embedder_name = emb_model_name

                texts = [c["chunk"] for c in ss.corpus]
                with st.spinner("Embedding chunks and building FAISS index..."):
                    embs = embed_texts(ss.embedder, texts)
                    ss.embeddings = embs
                    ss.index = build_faiss_index(embs)

                st.success(
                    f"Indexed {len(ss.corpus):,} chunks from {len(uploaded_files)} PDFs "
                    f"({total_pages} pages)."
                )

with col_q:
    question = st.text_area("❓ Ask a question about your PDFs", height=120,
                            placeholder="e.g., Summarize the main theorem on convergence and list its assumptions.")
    ask = st.button("🧠 Generate Answer", type="secondary")

if ask:
    if not ss.index or ss.embeddings is None or not ss.corpus:
        st.error("Please ingest and index PDFs first.")
    elif not question.strip():
        st.warning("Type a question to proceed.")
    else:
 
        with st.spinner("Searching relevant passages..."):
            q_emb = embed_texts(ss.embedder, [question])  
            D, I = search_index(ss.index, q_emb, k=top_k)
            idxs = I[0]
            scores = D[0]
            retrieved = []
            for rank, (i, sc) in enumerate(zip(idxs, scores), start=1):
                if i < 0:
                    continue
                rec = ss.corpus[i]
                rec = {**rec, "score": float(sc), "rank": rank}
                retrieved.append(rec)

        context = format_context(retrieved)
        prompt = build_llm_prompt(question, context)

        with st.spinner("Calling LLM (watsonx.ai)..."):
            answer = generate_answer_watsonx(prompt, cfg)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for rec in retrieved:
            with st.expander(f"[Source {rec['rank']}] {rec['doc']} – Page {rec['page']} (score={rec['score']:.3f})"):
                st.write(rec["chunk"])

st.divider()
with st.expander("ℹ️ Tips & Troubleshooting"):
    st.markdown(
        "- If the answer seems incomplete, increase **Top-K passages** or re-ask with more specifics.\n"
        "- For multilingual or domain-specific PDFs, try a different **SentenceTransformer** model.\n"
        "- Ensure watsonx.ai credentials are set. If you cannot use watsonx now, you can temporarily replace\n"
        "  `generate_answer_watsonx` with a local/alternative LLM call.\n"
        "- Very large PDFs: consider indexing fewer pages first or upgrading hardware for faster embedding."
    )
