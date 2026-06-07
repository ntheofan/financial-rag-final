"""
Financial RAG — FinanceBench
Streamlit application για αλληλεπιδραστική εξερεύνηση του RAG pipeline.
"""

import re
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ρύθμιση σελίδας (πρέπει να είναι το ΠΡΩΤΟ st call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Διαδρομές
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
EVAL_DIR = PROCESSED_DIR / "evaluation"
FIGURES_DIR = BASE_DIR / "notebooks" / "figures"

# ---------------------------------------------------------------------------
# Σταθερές
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

# διάσταση embedding → προτεινόμενο μοντέλο στο sidebar
DIM_TO_MODEL: dict[int, str] = {
    384:  "BAAI/bge-small-en-v1.5",
    768:  "BAAI/bge-base-en-v1.5",
    1024: "BAAI/bge-m3"
}

SYSTEM_PROMPT = """You are a financial analyst.

Answer the question using ONLY the provided context.

If the answer is not supported by the context, say:
"Insufficient evidence in the retrieved context."

Rules:
- Be precise and concise.
- Use exact financial wording, values, periods, or business terms found in the context.
- Do not invent facts.
- If the question asks whether a metric is not useful or not relevant, explain that only if the context supports it.
"""

PIPELINES = ["dense", "hybrid", "hybrid_reranked"]
PIPELINE_LABELS = {
    "dense": "Dense (Baseline)",
    "hybrid": "Hybrid (Dense + BM25)",
    "hybrid_reranked": "Hybrid + Reranking",
}

# ---------------------------------------------------------------------------
# Βοηθητικές συναρτήσεις parsing
# ---------------------------------------------------------------------------


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9&\.]+")

def tokenize_for_bm25(text: str):
    text = str(text).lower()
    return TOKEN_PATTERN.findall(text)

# ---------------------------------------------------------------------------
# Cached loaders δεδομένων
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

@st.cache_resource(show_spinner="Φόρτωση / δημιουργία BM25 index…")
def get_bm25_data(chunks_df: pd.DataFrame):
    bm25_docs_tokens = [tokenize_for_bm25(text) for text in chunks_df["chunk_text"].tolist()]
    doc_freq = Counter()
    doc_lens = []

    for tokens in bm25_docs_tokens:
        doc_lens.append(len(tokens))
        for tok in set(tokens):
            doc_freq[tok] += 1

    N_DOCS = len(bm25_docs_tokens)
    AVG_DOC_LEN = sum(doc_lens) / max(N_DOCS, 1)

    return bm25_docs_tokens, doc_freq, doc_lens, N_DOCS, AVG_DOC_LEN

def bm25_score_query_app(query_text: str, bm25_data, top_k: int):
    bm25_docs_tokens, doc_freq, doc_lens, N_DOCS, AVG_DOC_LEN = bm25_data
    BM25_K1 = 1.5
    BM25_B = 0.75

    query_tokens = tokenize_for_bm25(query_text)
    if not query_tokens:
        return []

    scores = np.zeros(N_DOCS, dtype=np.float32)
    query_token_counts = Counter(query_tokens)

    for token, qtf in query_token_counts.items():
        df = doc_freq.get(token, 0)
        if df == 0: continue

        idf = math.log(1 + (N_DOCS - df + 0.5) / (df + 0.5))

        for doc_idx, doc_tokens in enumerate(bm25_docs_tokens):
            tf = doc_tokens.count(token)
            if tf == 0: continue

            doc_len = doc_lens[doc_idx]
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / AVG_DOC_LEN))
            score = idf * ((tf * (BM25_K1 + 1)) / denom)
            scores[doc_idx] += score

    if np.all(scores == 0):
        return []

    top_indices = np.argsort(-scores)[:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

@st.cache_resource(show_spinner="Φόρτωση Cross-Encoder (Reranker)…")
def load_reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("BAAI/bge-reranker-v2-m3")
    except Exception:
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
# Παραγωγή απάντησης με GPT
# ---------------------------------------------------------------------------

def call_gpt(question: str, context_text: str, api_key: str, model: str) -> str:
    try:
        from openai import OpenAI

        # Περνάμε το API key μόνο στον τοπικό client, χωρίς αποθήκευση.
        client = OpenAI(api_key=api_key)

        user_prompt = f"Context:\n{context_text}\n\nQuestion:\n{question}"

        # Κλήση του OpenAI Chat Completions API.
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        # Ασφαλής ανάκτηση του κειμένου απάντησης.
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()

        return "Insufficient evidence in the retrieved context."
    except Exception as exc:
        return f"Generation error: {exc}"


# ---------------------------------------------------------------------------
# Πλευρική μπάρα
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Financial RAG")
    st.caption("FinanceBench · Thesis Project")
    st.divider()

    st.subheader("⚙️ Ρυθμίσεις")

    gpt_api_key: str = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-proj…",
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
        "Μοντέλο Παραγωγής (OpenAI)",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    )

    # Αυτόματος εντοπισμός διάστασης embedding από το .npy και πρόταση μοντέλου.
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
        return "OK" if cond else "Missing"

    chunks_ok = (CHUNKS_DIR / "financebench_chunks.parquet").exists() or (CHUNKS_DIR / "financebench_chunks.csv").exists()
    emb_ok = (EMBEDDINGS_DIR / "chunk_embeddings.npy").exists()
    eval_ok = (EVAL_DIR / "retrieval_evaluation_comparison.csv").exists()
    figs_ok = FIGURES_DIR.exists() and any(FIGURES_DIR.glob("*.png"))

    st.markdown(f"{_status(chunks_ok)} Chunks & Embeddings")
    st.markdown(f"{_status(emb_ok)} FAISS-ready embeddings (.npy)")
    st.markdown(f"{_status(eval_ok)} Evaluation CSVs")
    st.markdown(f"{_status(figs_ok)} Evaluation Figures")


# ---------------------------------------------------------------------------
# Κύρια tabs
# ---------------------------------------------------------------------------

tab_home, tab_query, tab_eval = st.tabs(
    ["🏠 Αρχική", "🔍 Live RAG Query","📈 Αξιολόγηση"]
)


# ===========================================================================
# TAB 1 — ΑΡΧΙΚΗ
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
| Chunking | Table-aware RecursiveCharacterTextSplitter (1500 chars, 200 overlap) |
| Embeddings | BAAI/bge-m3 (1024-dim) |
| Vector Store | FAISS IndexFlatIP |
| Dense Retrieval | Cosine similarity |
| Hybrid Retrieval | Dense + BM25 → RRF fusion |
| Reranking | Cross-encoder (BAAI/bge-reranker-v2-m3) |
| Generation | gpt-4o-mini |
"""
        )

    with col_results:
        st.subheader("📊 Retrieval Performance")
        perf_data = {
            "Pipeline": ["Dense (Baseline)", "Hybrid", "Hybrid + Reranking ⭐"],
            "Hit@1": ["47.3%", "48.6%", "63.3%"],
            "Hit@5": ["82.0%", "81.3%", "86.0%"],
            "MRR": ["61.5%", "61.1%", "73.0%"],
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

        st.subheader("🤖 Generation Quality (RAGChecker F1)")
        gen_data = {
            "Pipeline": ["Dense", "Hybrid", "Hybrid + Reranking ⭐"],
            "Faithfulness": ["45.0%", "45.3%", "50.8%"],
            "Hallucination ↓": ["38.3%", "34.1%", "28.8%"],
            "F1": ["18.3%", "22.9%", "27.0%"],
        }
        st.dataframe(pd.DataFrame(gen_data), use_container_width=True, hide_index=True)

    st.divider()


# ===========================================================================
# TAB 2 — LIVE RAG QUERY
# ===========================================================================

with tab_query:
    st.header("🔍 Live RAG Query")
    st.markdown(
        "Δοκιμάστε το pipeline με δική σας ερώτηση. "
        "Απαιτούνται τα αρχεία δεδομένων (chunks + embeddings) και ένα OpenAI API Key."
    )

    data_ready = chunks_ok and emb_ok

    if not data_ready:
        st.warning(
            "Τα αρχεία δεδομένων δεν βρέθηκαν. "
            "Τρέξτε πρώτα τα notebooks **01–06** για να δημιουργηθούν τα chunks και τα embeddings."
        )
        with st.expander("ℹ️ Πώς να στήσετε τα δεδομένα"):
            st.markdown(
                """
1. `01_setup_and_config.ipynb` — ρύθμιση paths
2. `02_prepare_financebench_dataset.ipynb` — προετοιμασία dataset
3. `03_parse_pdfs_with_docling.ipynb` — parsing PDF→Markdown
4. `04_clean_and_inspect_documents.ipynb` — καθαρισμός κειμένων
5. `05_create_chunks.ipynb` — δημιουργία chunks
6. `06_create_embeddings_and_vectorstore.ipynb` — embeddings + FAISS

Τα αρχεία αποθηκεύονται στο `data/`.
                """
            )

    if data_ready:
        with st.expander("📚 Δείτε τα διαθέσιμα έγγραφα της βάσης"):
            chunks_df = load_chunks()
            if chunks_df is not None and "doc_id" in chunks_df.columns:
                unique_docs = sorted(chunks_df["doc_id"].dropna().unique())
                st.markdown(f"**Σύνολο:** {len(unique_docs)} έγγραφα")

                # Προβολή σε μορφή πίνακα με σταθερό ύψος (scrollable)
                st.dataframe(
                    pd.DataFrame({"Όνομα Εγγράφου (doc_id)": unique_docs}),
                    use_container_width=True,
                    hide_index=True,
                    height=250
                )
            else:
                st.info("Δεν ήταν δυνατή η ανάκτηση των εγγράφων από τα chunks.")

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

        run_disabled = not data_ready or not gpt_api_key or not query_text.strip()
        run_btn = st.button("🚀 Εκτέλεση Query", type="primary", disabled=run_disabled)

        if not gpt_api_key:
            st.caption("Εισάγετε OpenAI API Key στο sidebar για παραγωγή απάντησης.")
        if not data_ready:
            st.caption("Απαιτούνται chunks + embeddings.")

    if run_btn and query_text.strip() and data_ready and gpt_api_key:
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

            st.write("Φόρτωση / δημιουργία Indices…")
            faiss_index = get_faiss_index(embeddings_matrix)
            if faiss_index is None:
                st.error("Αδυναμία δημιουργίας FAISS index. Εγκαταστήστε `faiss-cpu`.")
                st.stop()

            st.write("Κωδικοποίηση ερώτησης…")
            qvec = embed_model.encode(query_text, normalize_embeddings=True)
            qvec = np.ascontiguousarray(qvec, dtype=np.float32).reshape(1, -1)

            index_dim = faiss_index.d
            query_dim = qvec.shape[1]
            if query_dim != index_dim:
                st.error(
                    f"Διάσταση mismatch: FAISS index έχει {index_dim} dims, "
                    f"αλλά το query vector έχει {query_dim} dims."
                )
                st.stop()

            st.write(f"Αναζήτηση & Retrieval: **{PIPELINE_LABELS[retrieval_method]}**")

            DENSE_CANDIDATES_K = 20
            BM25_CANDIDATES_K = 10
            RRF_K = 30
            DENSE_WEIGHT = 1.0
            BM25_WEIGHT = 0.2

            final_retrieved_indices = []

            if retrieval_method == "dense":
                scores_arr, idx_arr = faiss_index.search(qvec, int(top_k))
                final_retrieved_indices = list(zip(idx_arr[0], scores_arr[0]))

            else:
                # Πυκνή αναζήτηση
                dense_scores, dense_idx = faiss_index.search(qvec, DENSE_CANDIDATES_K)
                dense_results = [(int(idx), float(score)) for idx, score in zip(dense_idx[0], dense_scores[0])]

                # Αναζήτηση BM25
                bm25_data = get_bm25_data(chunks_df)
                bm25_results = bm25_score_query_app(query_text, bm25_data, BM25_CANDIDATES_K)

                # Συνένωση αποτελεσμάτων με RRF
                fused = {}
                for rank, (global_idx, score) in enumerate(dense_results, start=1):
                    if global_idx not in fused: fused[global_idx] = {"rrf_score": 0.0}
                    fused[global_idx]["rrf_score"] += DENSE_WEIGHT / (RRF_K + rank)

                for rank, (global_idx, score) in enumerate(bm25_results, start=1):
                    if global_idx not in fused: fused[global_idx] = {"rrf_score": 0.0}
                    fused[global_idx]["rrf_score"] += BM25_WEIGHT / (RRF_K + rank)

                fused_list = sorted([(idx, data["rrf_score"]) for idx, data in fused.items()], key=lambda x: x[1], reverse=True)

                if retrieval_method == "hybrid":
                    final_retrieved_indices = fused_list[:int(top_k)]

                elif retrieval_method == "hybrid_reranked":
                    st.write("Επαναξιολόγηση με Cross-Encoder (Reranking)…")
                    reranker = load_reranker()
                    if reranker is None:
                        st.error("Αδυναμία φόρτωσης CrossEncoder. Ελέγξτε τις εξαρτήσεις σας.")
                        st.stop()

                    candidates = fused_list[:20]
                    pairs = []
                    for idx, _ in candidates:
                        chunk_text = str(chunks_df.iloc[int(idx)]["chunk_text"])
                        pairs.append([query_text, chunk_text])

                    rerank_scores = reranker.predict(pairs)
                    reranked_results = [(candidates[i][0], float(rerank_scores[i])) for i in range(len(candidates))]
                    reranked_results.sort(key=lambda x: x[1], reverse=True)

                    final_retrieved_indices = reranked_results[:int(top_k)]

            retrieved = []
            for rank, (idx, score) in enumerate(final_retrieved_indices, start=1):
                row = chunks_df.iloc[int(idx)]
                retrieved.append(
                    {
                        "rank": rank,
                        "score": score,
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

            st.write("Παραγωγή απάντησης με OpenAI…")
            answer = call_gpt(query_text, context_text, gpt_api_key, generation_model)

            status.update(label="Query ολοκληρώθηκε.", state="complete")

        st.subheader("💬 Απάντηση")
        st.info(answer)

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
# TAB 3 — ΑΞΙΟΛΟΓΗΣΗ
# ===========================================================================

with tab_eval:
    st.header("📈 Αξιολόγηση Pipeline")

    # ---- Numeric summaries ------------------------------------------------
    df_ret_eval = load_csv_eval("retrieval_evaluation_comparison.csv")
    df_qa_eval = load_csv_eval("qa_evaluation_comparison.csv")
    df_ragas = load_csv_eval("ragas_results.csv")

    # Δημιουργία ενιαίου wide table για τα RAGChecker summaries.
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
            "Δεν βρέθηκαν evaluation figures. "
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

        # Προβολή των figures σε πλέγμα δύο στηλών.
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
