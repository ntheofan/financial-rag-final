"""
Financial RAG — FinanceBench
Streamlit application για αλληλεπιδραστική εξερεύνηση του RAG pipeline.
"""

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (πρέπει να είναι το ΠΡΩΤΟ st call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial RAG — FinanceBench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
RETRIEVAL_DIR = PROCESSED_DIR / "retrieval_results"
QA_DIR = PROCESSED_DIR / "qa_results"
EVAL_DIR = PROCESSED_DIR / "evaluation"
FIGURES_DIR = BASE_DIR / "notebooks" / "figures"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

# dimension → suggested model (for auto-suggest in sidebar)
DIM_TO_MODEL: dict[int, str] = {
    384:  "BAAI/bge-small-en-v1.5",
    768:  "BAAI/bge-base-en-v1.5",
    1024: "BAAI/bge-m3",
    4096: "intfloat/e5-mistral-7b-instruct",
}

SYSTEM_PROMPT = """You are a financial question answering assistant.

Answer the question using ONLY the provided context.

If the answer is not supported by the context, say:
"Insufficient evidence in the retrieved context."

Rules:
- Be precise and concise.
- Use exact financial wording, values, periods, or business terms found in the context.
- Do not invent facts.
- If the question asks whether a metric is not useful or not relevant, explain that only if the context supports it.
"""

FINANCE_ALIAS_MAP = {
    "ppe": ["property plant equipment", "property plant and equipment", "pp&e", "balance sheet"],
    "net ppe": ["property plant and equipment net", "net property plant equipment", "balance sheet"],
    "capex": [
        "capital expenditure", "capital expenditures",
        "purchases of property plant and equipment",
        "cash flow statement",
    ],
    "cogs": ["cost of goods sold", "cost of sales", "income statement"],
    "ebitda": ["earnings before interest taxes depreciation and amortization"],
    "opex": ["operating expenses", "selling general and administrative", "sg&a"],
    "gross margin": ["gross profit margin", "gross profit", "income statement"],
    "operating cash flow": ["net cash provided by operating activities", "cash flow statement"],
    "free cash flow": ["free cash flow", "net cash provided by operating activities", "capital expenditures"],
    "working capital": ["current assets", "current liabilities", "balance sheet"],
    "payout ratio": ["dividends", "net income", "dividend payout ratio"],
}

PIPELINES = ["dense", "hybrid", "hybrid_reranked"]
PIPELINE_LABELS = {
    "dense": "Dense (Baseline)",
    "hybrid": "Hybrid (Dense + BM25)",
    "hybrid_reranked": "Hybrid + Reranking",
}
PIPELINE_COLORS = {
    "dense": "#4C72B0",
    "hybrid": "#55A868",
    "hybrid_reranked": "#C44E52",
}

# ---------------------------------------------------------------------------
# Query expansion helpers
# ---------------------------------------------------------------------------

def expand_finance_query(query: str) -> str:
    q = re.sub(r"\s+", " ", str(query).lower().strip())
    expansions = []
    for alias in sorted(FINANCE_ALIAS_MAP, key=len, reverse=True):
        if alias in q:
            expansions.extend(FINANCE_ALIAS_MAP[alias])
    if "balance sheet" in q:
        expansions.extend(["balance sheet", "assets liabilities equity"])
    if "cash flow" in q:
        expansions.extend(["cash flow statement", "operating activities investing activities financing activities"])
    if "income statement" in q:
        expansions.extend(["income statement", "net sales operating income net income"])
    if "year end" in q or "year-end" in q:
        expansions.extend(["at december 31", "balance sheet"])
    seen: set = set()
    unique = []
    for item in expansions:
        item = item.strip().lower()
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return q + (" " + " ".join(unique) if unique else "")


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Φόρτωση embedding model…")
def load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None


@st.cache_data(show_spinner="Φόρτωση chunks…")
def load_chunks() -> pd.DataFrame | None:
    for path in [CHUNKS_DIR / "financebench_chunks.parquet", CHUNKS_DIR / "financebench_chunks.csv"]:
        if path.exists():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return None


@st.cache_data(show_spinner="Φόρτωση embeddings…")
def load_embeddings() -> np.ndarray | None:
    npy = EMBEDDINGS_DIR / "chunk_embeddings.npy"
    return np.load(npy) if npy.exists() else None


@st.cache_resource(show_spinner="Φόρτωση / δημιουργία FAISS index…")
def get_faiss_index(_embeddings: np.ndarray):
    """Load pre-saved index if available, otherwise build from embeddings."""
    try:
        import faiss  # type: ignore

        saved_index = EMBEDDINGS_DIR / "financebench_faiss.index"
        if saved_index.exists():
            index = faiss.read_index(str(saved_index))
            return index

        vectors = np.ascontiguousarray(_embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_working_dataset() -> pd.DataFrame | None:
    for path in [
        INTERIM_DIR / "financebench_open_source_working.parquet",
        INTERIM_DIR / "financebench_open_source_working.csv",
    ]:
        if path.exists():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_qa_results(run_name: str) -> pd.DataFrame | None:
    candidates = [
        QA_DIR / f"rag_qa_results_{run_name}.parquet",
        QA_DIR / f"rag_qa_results_{run_name}.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_retrieval_results(run_name: str) -> pd.DataFrame | None:
    candidates = [
        RETRIEVAL_DIR / f"retrieval_results_{run_name}.parquet",
        RETRIEVAL_DIR / f"retrieval_results_{run_name}.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_csv_eval(filename: str) -> pd.DataFrame | None:
    path = EVAL_DIR / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    return df


# ---------------------------------------------------------------------------
# Gemini generation
# ---------------------------------------------------------------------------

def call_gemini(question: str, context_text: str, api_key: str, model: str) -> str:
    try:
        os.environ["GEMINI_API_KEY"] = api_key
        from google import genai  # type: ignore

        client = genai.Client()
        user_prompt = f"Context:\n{context_text}\n\nQuestion:\n{question}"
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config={"system_instruction": SYSTEM_PROMPT, "temperature": 0.0},
        )
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        return "Insufficient evidence in the retrieved context."
    except Exception as exc:
        return f"❌ Generation error: {exc}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Financial RAG")
    st.caption("FinanceBench · Thesis Project")
    st.divider()

    st.subheader("⚙️ Ρυθμίσεις")

    gemini_api_key: str = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIzaSy…",
        help="Απαιτείται για live παραγωγή απαντήσεων (Tab: Live Query)",
    )

    retrieval_method: str = st.selectbox(
        "Μέθοδος Retrieval",
        options=PIPELINES,
        format_func=lambda x: PIPELINE_LABELS[x],
        index=2,
    )

    top_k: int = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5)

    generation_model: str = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
    )

    # Auto-detect embedding dimension from .npy file and suggest model
    _npy = EMBEDDINGS_DIR / "chunk_embeddings.npy"
    _detected_dim: int | None = None
    if _npy.exists():
        try:
            _arr = np.load(_npy, mmap_mode="r")
            _detected_dim = int(_arr.shape[1])
        except Exception:
            pass
    _suggested_model = DIM_TO_MODEL.get(_detected_dim or 1024, DEFAULT_EMBEDDING_MODEL)

    embedding_model_name: str = st.text_input(
        "Embedding Model",
        value=_suggested_model,
        help=(
            f"Μοντέλο που χρησιμοποιήθηκε για τα chunk embeddings. "
            + (f"Auto-detected dim: **{_detected_dim}**." if _detected_dim else "")
        ),
    )
    if _detected_dim:
        st.caption(f"Detected embedding dim: `{_detected_dim}`")

    st.divider()
    st.caption("📁 Διαθεσιμότητα δεδομένων")

    def _status(cond: bool) -> str:
        return "✅" if cond else "❌"

    chunks_ok = (CHUNKS_DIR / "financebench_chunks.parquet").exists() or (CHUNKS_DIR / "financebench_chunks.csv").exists()
    emb_ok = (EMBEDDINGS_DIR / "chunk_embeddings.npy").exists()
    qa_ok = any((QA_DIR / f"rag_qa_results_{p}.csv").exists() or (QA_DIR / f"rag_qa_results_{p}.parquet").exists() for p in PIPELINES)
    eval_ok = (EVAL_DIR / "retrieval_evaluation_comparison.csv").exists()
    figs_ok = FIGURES_DIR.exists() and any(FIGURES_DIR.glob("*.png"))

    st.markdown(f"{_status(chunks_ok)} Chunks & Embeddings")
    st.markdown(f"{_status(emb_ok)} FAISS-ready embeddings (.npy)")
    st.markdown(f"{_status(qa_ok)} QA Results")
    st.markdown(f"{_status(eval_ok)} Evaluation CSVs")
    st.markdown(f"{_status(figs_ok)} Evaluation Figures")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_home, tab_query, tab_results, tab_eval = st.tabs(
    ["🏠 Αρχική", "🔍 Live RAG Query", "📊 Αποτελέσματα", "📈 Αξιολόγηση"]
)


# ===========================================================================
# TAB 1 — HOME
# ===========================================================================

with tab_home:
    st.title("Financial RAG Pipeline")
    st.markdown(
        "Σύστημα **Retrieval-Augmented Generation** πάνω στο "
        "[FinanceBench](https://github.com/patronus-ai/financebench) dataset — "
        "150 ερωτήσεις από 84 οικονομικές εκθέσεις (10-K, 10-Q, 8-K) "
        "32 εισηγμένων εταιριών."
    )

    st.divider()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ερωτήσεις", "150")
    c2.metric("Έγγραφα", "84")
    c3.metric("Εταιρίες", "32")
    c4.metric("Text Chunks", "37,037")
    c5.metric("Embedding Dims", "1,024")
    c6.metric("Embedding Model", "BAAI/bge-m3")

    st.divider()

    col_pipeline, col_results = st.columns([1, 1], gap="large")

    with col_pipeline:
        st.subheader("🔄 Pipeline")
        st.markdown(
            """
| Βήμα | Εργαλείο |
|------|---------|
| PDF Parsing | Docling |
| Chunking | RecursiveCharacterTextSplitter (800 chars, 150 overlap) |
| Embeddings | BAAI/bge-m3 (1024-dim) |
| Vector Store | FAISS IndexFlatIP |
| Dense Retrieval | Cosine similarity |
| Hybrid Retrieval | Dense + BM25 → RRF fusion |
| Reranking | Cross-encoder |
| Generation | Gemini 2.5 Flash |
"""
        )

    with col_results:
        st.subheader("📊 Retrieval Performance")
        perf_data = {
            "Pipeline": ["Dense (Baseline)", "Hybrid", "Hybrid + Reranking ⭐"],
            "Hit@1": ["47.3%", "56.0%", "67.3%"],
            "Hit@5": ["74.0%", "78.7%", "84.0%"],
            "MRR": ["58.6%", "64.8%", "73.8%"],
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

        st.subheader("🤖 Generation Quality (RAGChecker F1)")
        gen_data = {
            "Pipeline": ["Dense", "Hybrid", "Hybrid + Reranking ⭐"],
            "Faithfulness": ["37.7%", "40.9%", "46.3%"],
            "Hallucination ↓": ["38.5%", "36.7%", "31.2%"],
            "F1": ["19.0%", "18.9%", "25.7%"],
        }
        st.dataframe(pd.DataFrame(gen_data), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("📁 Εταιρίες στο Dataset")
    companies_info = {
        "Sector": [
            "Industrials", "Healthcare", "Financials", "Energy",
            "Consumer Staples", "Technology", "Communication Services", "Other",
        ],
        "Παραδείγματα": [
            "3M, Boeing, Lockheed Martin",
            "Johnson & Johnson, Pfizer",
            "American Express, Goldman Sachs",
            "ExxonMobil, Chevron",
            "Coca-Cola, Walmart, P&G",
            "Adobe, Activision Blizzard",
            "MGM Resorts",
            "American Water Works, General Mills",
        ],
    }
    st.dataframe(pd.DataFrame(companies_info), use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 2 — LIVE RAG QUERY
# ===========================================================================

with tab_query:
    st.header("🔍 Live RAG Query")
    st.markdown(
        "Δοκιμάστε το pipeline με δική σας ερώτηση. "
        "Απαιτούνται τα αρχεία δεδομένων (chunks + embeddings) και ένα Gemini API key."
    )

    data_ready = chunks_ok and emb_ok

    if not data_ready:
        st.warning(
            "⚠️ Τα αρχεία δεδομένων δεν βρέθηκαν. "
            "Τρέξτε πρώτα τα notebooks **01–06** για να δημιουργηθούν τα chunks και τα embeddings."
        )
        with st.expander("ℹ️ Πώς να στήσετε τα δεδομένα"):
            st.markdown(
                """
1. `01_setup_and_config.ipynb` — ρύθμιση paths
2. `02_load_financebench_sample.ipynb` — φόρτωση dataset
3. `03_parse_pdfs_with_docling.ipynb` — parsing PDF→Markdown
4. `04_clean_markdown_and_inspect.ipynb` — καθαρισμός
5. `05_chunking.ipynb` — chunking
6. `06_embeddings_and_vectorstore.ipynb` — embeddings + FAISS

Τα αρχεία αποθηκεύονται στο `data/`.
                """
            )

    st.divider()

    col_q, col_ex = st.columns([2, 1])

    with col_ex:
        st.markdown("**Παραδείγματα ερωτήσεων:**")
        example_questions = [
            "What is the FY2018 capital expenditure for 3M?",
            "What is the year-end FY2018 net PP&E for 3M?",
            "Does 3M maintain a stable trend of dividend distribution?",
            "What drove operating margin change as of FY2022 for 3M?",
            "What is the FY2019 fixed asset turnover ratio for Activision Blizzard?",
        ]
        for eq in example_questions:
            if st.button(eq, key=f"ex_{eq[:20]}", use_container_width=True, disabled=not data_ready):
                st.session_state["query_text"] = eq

    with col_q:
        query_text = st.text_area(
            "Ερώτηση",
            value=st.session_state.get("query_text", ""),
            placeholder="What is the FY2018 capital expenditure for 3M?",
            height=120,
        )

        run_disabled = not data_ready or not gemini_api_key or not query_text.strip()
        run_btn = st.button("🚀 Εκτέλεση Query", type="primary", disabled=run_disabled)

        if not gemini_api_key:
            st.caption("⚠️ Εισάγετε Gemini API Key στο sidebar για παραγωγή απάντησης.")
        if not data_ready:
            st.caption("⚠️ Απαιτούνται chunks + embeddings.")

    if run_btn and query_text.strip() and data_ready and gemini_api_key:
        with st.status("Εκτελείται RAG pipeline…", expanded=True) as status:
            st.write(f"Φόρτωση embedding model `{embedding_model_name}`…")
            embed_model = load_embedding_model(embedding_model_name)
            if embed_model is None:
                st.error("Αδυναμία φόρτωσης `sentence-transformers`. Ελέγξτε τις εξαρτήσεις.")
                st.stop()

            st.write("Φόρτωση chunks & embeddings…")
            chunks_df = load_chunks()
            embeddings_matrix = load_embeddings()
            if chunks_df is None or embeddings_matrix is None:
                st.error("Αδυναμία φόρτωσης δεδομένων.")
                st.stop()

            st.write("Φόρτωση / δημιουργία FAISS index…")
            faiss_index = get_faiss_index(embeddings_matrix)
            if faiss_index is None:
                st.error("Αδυναμία δημιουργίας FAISS index. Εγκαταστήστε `faiss-cpu`.")
                st.stop()

            st.write("Κωδικοποίηση ερώτησης…")
            expanded_q = expand_finance_query(query_text)
            qvec = embed_model.encode([expanded_q], normalize_embeddings=True)
            qvec = np.ascontiguousarray(qvec, dtype=np.float32).reshape(1, -1)

            # Dimension sanity check — helpful error instead of cryptic AssertionError
            index_dim = faiss_index.d
            query_dim = qvec.shape[1]
            if query_dim != index_dim:
                st.error(
                    f"Διάσταση mismatch: FAISS index έχει {index_dim} dims, "
                    f"αλλά το query vector έχει {query_dim} dims. "
                    f"Embeddings shape: {embeddings_matrix.shape}."
                )
                st.stop()

            st.write("Αναζήτηση στον vector store…")
            scores_arr, idx_arr = faiss_index.search(qvec, int(top_k))
            scores_arr, idx_arr = scores_arr[0], idx_arr[0]

            retrieved = []
            for rank, (score, idx) in enumerate(zip(scores_arr, idx_arr), start=1):
                row = chunks_df.iloc[int(idx)]
                retrieved.append(
                    {
                        "rank": rank,
                        "score": float(score),
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "chunk_text": str(row["chunk_text"]),
                    }
                )

            context_parts = [
                f"[Rank {c['rank']} | Doc {c['doc_id']} | Chunk {c['chunk_id']}]\n{c['chunk_text']}"
                for c in retrieved
            ]
            context_text = "\n\n" + ("\n\n" + "—" * 60 + "\n\n").join(context_parts)

            st.write("Παραγωγή απάντησης με Gemini…")
            answer = call_gemini(query_text, context_text, gemini_api_key, generation_model)

            status.update(label="✅ Query ολοκληρώθηκε!", state="complete")

        st.subheader("💬 Απάντηση")
        st.info(answer)

        with st.expander("🔬 Expanded Query"):
            st.code(expanded_q, language="text")

        st.subheader(f"📄 Top-{top_k} Retrieved Chunks")
        for chunk in retrieved:
            with st.expander(
                f"Rank {chunk['rank']} — **{chunk['doc_id']}** (score: {chunk['score']:.4f})"
            ):
                st.markdown(f"**Chunk ID:** `{chunk['chunk_id']}`")
                st.text_area(
                    "Text",
                    value=chunk["chunk_text"],
                    height=180,
                    disabled=True,
                    key=f"ct_{chunk['rank']}",
                )


# ===========================================================================
# TAB 3 — RESULTS EXPLORER
# ===========================================================================

with tab_results:
    st.header("📊 Εξερεύνηση QA Αποτελεσμάτων")
    st.caption(f"Pipeline: **{PIPELINE_LABELS[retrieval_method]}**")

    qa_df = load_qa_results(retrieval_method)

    if qa_df is None:
        st.warning(
            f"⚠️ Δεν βρέθηκαν QA results για `{retrieval_method}`. "
            "Τρέξτε το notebook **10_rag_qa_baseline.ipynb**."
        )
    else:
        # Summary metrics
        total = len(qa_df)
        doc_match_rate = qa_df["doc_match"].mean() * 100 if "doc_match" in qa_df.columns else None
        avg_len = qa_df["generated_answer_len"].mean() if "generated_answer_len" in qa_df.columns else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ερωτήσεις", total)
        if doc_match_rate is not None:
            m2.metric("Doc Match Rate", f"{doc_match_rate:.1f}%")
        if avg_len is not None:
            m3.metric("Μέσο μήκος απάντησης", f"{avg_len:.0f} chars")
        insuff = (
            qa_df["generated_answer"]
            .str.contains("Insufficient evidence", na=False)
            .sum()
        ) if "generated_answer" in qa_df.columns else 0
        m4.metric("Ανεπαρκής context", insuff)

        st.divider()

        # Filters
        st.subheader("🔎 Φίλτρα")
        f1, f2, f3, f4 = st.columns(4)

        companies = ["(Όλες)"]
        if "expected_company" in qa_df.columns:
            companies += sorted(qa_df["expected_company"].dropna().unique().tolist())
        sel_company = f1.selectbox("Εταιρία", companies, key="r_company")

        doc_types = ["(Όλοι)"]
        if "expected_doc_name" in qa_df.columns:
            # extract type from doc name pattern COMPANY_YEAR_TYPE
            types = qa_df["expected_doc_name"].dropna().str.extract(r"_(\d{4}Q?\d*_)?(\w+)$")[1].dropna().unique()
            doc_types += sorted(types.tolist())
        sel_type = f2.selectbox("Τύπος εγγράφου", doc_types, key="r_type")

        search_q = f3.text_input("Αναζήτηση στην ερώτηση", placeholder="capital expenditure…", key="r_search")

        only_match = f4.checkbox("Μόνο Doc Match ✅", key="r_match")
        only_insuff = f4.checkbox("Ανεπαρκής context ⚠️", key="r_insuff")

        # Apply filters
        fdf = qa_df.copy()
        if sel_company != "(Όλες)" and "expected_company" in fdf.columns:
            fdf = fdf[fdf["expected_company"] == sel_company]
        if sel_type != "(Όλοι)" and "expected_doc_name" in fdf.columns:
            fdf = fdf[fdf["expected_doc_name"].str.contains(sel_type, na=False)]
        if search_q:
            fdf = fdf[fdf["question"].str.contains(search_q, case=False, na=False)]
        if only_match and "doc_match" in fdf.columns:
            fdf = fdf[fdf["doc_match"] == True]
        if only_insuff and "generated_answer" in fdf.columns:
            fdf = fdf[fdf["generated_answer"].str.contains("Insufficient evidence", na=False)]

        st.markdown(f"Εμφανίζονται **{len(fdf)}** / {total} αποτελέσματα")

        for _, row in fdf.head(25).iterrows():
            question = str(row.get("question", ""))
            expected = str(row.get("expected_answer", ""))
            generated = str(row.get("generated_answer", ""))
            doc_match = row.get("doc_match")
            top_doc = str(row.get("top_doc_id", ""))
            expected_doc = str(row.get("expected_doc_name", ""))

            icon = "✅" if doc_match else "❌"
            label = f"{icon} {question[:110]}{'…' if len(question) > 110 else ''}"

            with st.expander(label):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Expected Answer:**")
                    st.success(expected[:400])
                    st.caption(f"📄 Expected Doc: `{expected_doc}`")
                with cb:
                    st.markdown("**Generated Answer:**")
                    preview = generated[:500] + ("…" if len(generated) > 500 else "")
                    if "Insufficient evidence" in generated:
                        st.warning(preview)
                    else:
                        st.info(preview)
                    st.caption(f"📄 Retrieved Doc: `{top_doc}`")

                if "context_text" in row and pd.notna(row.get("context_text")):
                    with st.expander("📄 Context chunks"):
                        st.text(str(row["context_text"])[:3000])


# ===========================================================================
# TAB 4 — EVALUATION
# ===========================================================================

with tab_eval:
    st.header("📈 Αξιολόγηση Pipeline")

    # ---- Numeric summaries ------------------------------------------------
    df_ret_eval = load_csv_eval("retrieval_evaluation_comparison.csv")
    df_qa_eval = load_csv_eval("qa_evaluation_comparison.csv")
    df_ragas = load_csv_eval("ragas_results.csv")

    # Build combined RAGChecker wide table
    rc_frames = []
    for run in PIPELINES:
        df_rc = load_csv_eval(f"ragchecker_summary_{run}.csv")
        if df_rc is not None:
            df_rc["run_name"] = run
            rc_frames.append(df_rc)
    df_rc_wide = None
    if rc_frames:
        df_rc_all = pd.concat(rc_frames, ignore_index=True)
        try:
            df_rc_wide = df_rc_all.pivot_table(
                index="run_name", columns="metric_name", values="metric_value"
            ).reset_index()
        except Exception:
            pass

    has_numeric = any(x is not None for x in [df_ret_eval, df_qa_eval, df_rc_wide])

    if has_numeric:
        st.subheader("📐 Αριθμητικά Αποτελέσματα")

        tab_ret, tab_qa_m, tab_rc, tab_ragas = st.tabs(
            ["Retrieval", "QA Metrics", "RAGChecker", "RAGAS"]
        )

        with tab_ret:
            if df_ret_eval is not None:
                display = df_ret_eval.copy()
                if "run_name" in display.columns:
                    display["run_name"] = display["run_name"].map(PIPELINE_LABELS).fillna(display["run_name"])
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.info("Δεν βρέθηκε `retrieval_evaluation_comparison.csv`.")

        with tab_qa_m:
            if df_qa_eval is not None:
                display = df_qa_eval.copy()
                if "run_name" in display.columns:
                    display["run_name"] = display["run_name"].map(PIPELINE_LABELS).fillna(display["run_name"])
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.info("Δεν βρέθηκε `qa_evaluation_comparison.csv`.")

        with tab_rc:
            if df_rc_wide is not None:
                display = df_rc_wide.copy()
                if "run_name" in display.columns:
                    display["run_name"] = display["run_name"].map(PIPELINE_LABELS).fillna(display["run_name"])
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.info("Δεν βρέθηκαν RAGChecker summary CSVs.")

        with tab_ragas:
            if df_ragas is not None:
                display = df_ragas.copy()
                if "run_name" in display.columns:
                    display["run_name"] = display["run_name"].map(PIPELINE_LABELS).fillna(display["run_name"])
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.info("Δεν βρέθηκε `ragas_results.csv`.")

        st.divider()

    # ---- Figures ----------------------------------------------------------
    st.subheader("📊 Evaluation Figures")

    if not figs_ok:
        st.warning(
            "⚠️ Δεν βρέθηκαν evaluation figures. "
            "Τρέξτε τα notebooks **12–15** για να παραχθούν."
        )
    else:
        figure_catalog = [
            ("fig1_retrieval_performance.png", "Figure 1: Retrieval Performance — Hit@k & MRR"),
            ("fig2_ragchecker_metrics.png", "Figure 2: RAGChecker — Retriever & Generator Metrics"),
            ("fig3_ragchecker_overall.png", "Figure 3: RAGChecker — Overall F1 / Precision / Recall"),
            ("fig4_qa_deterministic.png", "Figure 4: QA Deterministic Metrics"),
            ("fig5_radar_chart.png", "Figure 5: Radar Chart — Multi-Dimensional Σύγκριση"),
            ("fig6_per_query_distributions.png", "Figure 6: Per-Query Score Distributions"),
            ("fig7_summary_table.png", "Table 1: Summary Table — Όλα τα Metrics"),
            ("fig8_delta_improvement.png", "Figure 8: Βελτίωση vs Dense Baseline (Δ pp)"),
            ("fig9_generator_deep_dive.png", "Figure 9: Generator Deep-Dive"),
        ]

        # display in 2-column grid
        it = iter(figure_catalog)
        for left_item in it:
            right_item = next(it, None)
            col_l, col_r = st.columns(2)
            for col, item in [(col_l, left_item), (col_r, right_item)]:
                if item is None:
                    break
                fname, title = item
                fpath = FIGURES_DIR / fname
                if fpath.exists():
                    with col:
                        st.markdown(f"**{title}**")
                        st.image(str(fpath), use_container_width=True)
            st.divider()
