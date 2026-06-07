# Financial RAG Final

Κώδικας και παραγόμενα notebooks για τη διπλωματική εργασία:
**Σχεδίαση και Εμπειρική Αξιολόγηση Συστημάτων Retrieval-Augmented Generation για την Ανάλυση Ετήσιων Οικονομικών Εκθέσεων**.

Το project υλοποιεί και αξιολογεί ένα end-to-end RAG pipeline πάνω στο δημόσιο υποσύνολο του FinanceBench, με document-unknown retrieval σε 150 ερωτήσεις και 84 οικονομικά έγγραφα.

## Περιεχόμενα

- `app.py`: Streamlit εφαρμογή για live ερωτήσεις, επιλογή retrieval strategy και προβολή evaluation αποτελεσμάτων.
- `notebooks/`: τελική notebook ροή από setup μέχρι thesis figures.
- `notebooks/figures/`: τελικά διαγράμματα που χρησιμοποιούνται στην ανάλυση.
- `data/raw/`: source PDFs και FinanceBench metadata.
- `data/interim/`: parsed/cleaned markdown και ενδιάμεσοι πίνακες.
- `data/processed/`: chunks, embeddings, retrieval, QA και evaluation outputs.

Τα μεγάλα artifacts στο `data/` αγνοούνται από git μέσω `.gitignore`. Για πλήρη αναπαραγωγή, τα notebooks πρέπει να τρέξουν με τη σειρά ή να προστεθούν τοπικά τα αντίστοιχα artifacts.

## Pipeline

1. PDF parsing με Docling
2. Markdown cleaning και επιθεώρηση
3. Table-aware chunking με `chunk_size=1500`, `chunk_overlap=200`
4. Embeddings με `BAAI/bge-m3` και FAISS `IndexFlatIP`
5. Dense retrieval baseline
6. Hybrid retrieval με BM25 και Reciprocal Rank Fusion
7. Cross-encoder reranking με `BAAI/bge-reranker-v2-m3`
8. RAG answer generation με OpenAI API
9. Evaluation με retrieval metrics, deterministic QA metrics, RAGAS και RAGChecker
10. Παραγωγή τελικών figures για τη διπλωματική

## Notebook Ροή

1. `01_setup_and_config.ipynb` — αρχική ρύθμιση διαδρομών και φακέλων.
2. `02_prepare_financebench_dataset.ipynb` — προετοιμασία του FinanceBench dataset.
3. `03_parse_pdfs_with_docling.ipynb` — μετατροπή PDF εκθέσεων σε markdown.
4. `04_clean_and_inspect_documents.ipynb` — καθαρισμός και έλεγχος των κειμένων.
5. `05_create_chunks.ipynb` — δημιουργία chunks και μεταδεδομένων.
6. `06_create_embeddings_and_vectorstore.ipynb` — embeddings και FAISS vector store.
7. `07_dense_retrieval.ipynb` — dense retrieval baseline.
8. `08_hybrid_retrieval.ipynb` — υβριδικό retrieval με BM25 και dense retrieval.
9. `09_reranking.ipynb` — reranking των ανακτημένων αποσπασμάτων.
10. `10_answer_generation.ipynb` — παραγωγή απαντήσεων με βάση τα ανακτημένα contexts.
11. `11_evaluate_retrieval_and_qa.ipynb` — αξιολόγηση retrieval και απαντήσεων.
12. `12_error_analysis.ipynb` — ανάλυση σφαλμάτων ανά pipeline.
13. `13_ragas_evaluation.ipynb` — αξιολόγηση με RAGAS.
14. `14_ragchecker_evaluation.ipynb` — αξιολόγηση με RAGChecker.
15. `15_thesis_figures_and_tables.ipynb` — παραγωγή σχημάτων και πινάκων για τη διπλωματική.

## Εγκατάσταση

Για πλήρη εκτέλεση των notebooks:

```bash
pip install -r requirements.txt
```

Για την Streamlit εφαρμογή μόνο:

```bash
pip install -r streamlit_requirements.txt
```

## API Keys

Μην αποθηκεύετε API keys μέσα σε notebooks ή source files.

Δημιουργήστε τοπικά ένα `.env` από το `.env.example` ή ορίστε τη μεταβλητή περιβάλλοντος:

```bash
setx OPENAI_API_KEY ...
```

Σε PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "...", "User")
```

Το `.env` αγνοείται από git.

## Εκτέλεση Streamlit

```bash
streamlit run app.py
```

Η εφαρμογή αναμένει τα παραγόμενα artifacts κάτω από `data/processed/`.

## Κύρια Αποτελέσματα

Στα τελικά evaluation artifacts, η παραλλαγή **Hybrid + Reranking** εμφανίζει την καλύτερη συνολική απόδοση:

- Hit@1: 63.3%
- Hit@5: 86.0%
- MRR: 73.0%
- RAGChecker F1: 27.0%
- RAGChecker Faithfulness: 50.8%
- RAGChecker Hallucination: 28.8%

## Σημειώσεις Αναπαραγωγής

- Τα notebooks είναι καθαρισμένα από execution outputs ώστε το GitHub diff να παραμένει ευανάγνωστο.
- Τα μεγάλα δεδομένα και embeddings δεν ανεβαίνουν στο repository.
- Αν χρειάζεται πλήρης αναπαραγωγή από την αρχή, ξεκινήστε από το `01_setup_and_config.ipynb` και συνεχίστε σειριακά.
- Τα notebooks που χρησιμοποιούν OpenAI/RAGAS/RAGChecker απαιτούν `OPENAI_API_KEY` στο περιβάλλον.
