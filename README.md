# Financial RAG Final

Κώδικας και παραγόμενα notebooks για τη διπλωματική εργασία:
**Σχεδίαση και Εμπειρική Αξιολόγηση Συστημάτων Retrieval-Augmented Generation για την Ανάλυση Ετήσιων Οικονομικών Εκθέσεων**.

Το project υλοποιεί και αξιολογεί ένα end-to-end RAG pipeline πάνω στο δημόσιο υποσύνολο του FinanceBench, με document-unknown retrieval σε 150 ερωτήσεις και 84 οικονομικά έγγραφα.

## Περιεχόμενα

- `app.py`: Streamlit εφαρμογή για live ερωτήσεις, επιλογή retrieval strategy και προβολή evaluation αποτελεσμάτων.
- `notebooks/`: τελική notebook ροή από setup μέχρι thesis figures.
- `scripts/run_kaggle_pipeline.py`: canonical orchestration των 15 notebooks.
- `scripts/pipeline_profile.py`: κοινή επιλογή και επαλήθευση full/pilot dataset profile.
- `configs/kaggle_pipeline.json`: δηλωτική σειρά σταδίων και απαιτούμενων outputs.
- `notebooks/figures/`: τελικά διαγράμματα που χρησιμοποιούνται στην ανάλυση.
- `data/raw/`: source PDFs και FinanceBench metadata.
- `data/interim/`: parsed/cleaned markdown και ενδιάμεσοι πίνακες.
- `data/processed/`: chunks, embeddings, retrieval, QA και evaluation outputs.

Τα μεγάλα artifacts στο `data/` αγνοούνται από git μέσω `.gitignore`. Για πλήρη αναπαραγωγή, τα notebooks πρέπει να τρέξουν με τη σειρά ή να προστεθούν τοπικά τα αντίστοιχα artifacts.

## Pipeline

1. PDF parsing με Docling
2. Markdown cleaning και επιθεώρηση
3. Markdown-aware structural chunking με `chunk_size=1500`, `chunk_overlap=200`
4. Embeddings με `BAAI/bge-m3` και FAISS `IndexFlatIP`
5. Dense retrieval baseline
6. Hybrid retrieval με BM25 και Reciprocal Rank Fusion (20 υποψήφιοι ανά query)
7. Metadata-aware cross-encoder reranking με `BAAI/bge-reranker-v2-m3`,
   score fusion με το Hybrid RRF και τελικό context-aligned top-5
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

Αντιγράψτε το `.env.example` σε `.env` και συμπληρώστε το κλειδί σας, ή ορίστε τη μεταβλητή περιβάλλοντος:

```bash
setx OPENAI_API_KEY ...
```

Σε PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "...", "User")
```

Σε περιβάλλον Kaggle, το κλειδί ορίζεται μέσω Kaggle Secrets (Add-ons → Secrets) και διαβάζεται από το περιβάλλον εκτέλεσης.

## Περιβάλλον Εκτέλεσης

Το canonical full run μπορεί να εκτελεστεί εξ ολοκλήρου στο Kaggle με GPU και
Internet ενεργοποιημένα. Η επίσημη διαδρομή αναπαραγωγής χρησιμοποιεί τον runner,
ώστε κάθε notebook να εκτελείται σε καθορισμένη σειρά και να ελέγχονται τα
αναμενόμενα outputs, τα API-dependent αποτελέσματα και το provenance του run.

Αναλυτικές, έτοιμες για αντιγραφή οδηγίες: [Canonical full run στο Kaggle](docs/KAGGLE_FULL_RUN.md).

Πριν από κάθε publication run εκτελείται πραγματικό end-to-end pilot. Το
profile εφαρμόζεται πριν από το PDF parsing, ώστε parsing, chunks, embeddings,
retrieval, generation και evaluation να χρησιμοποιούν ακριβώς το ίδιο δείγμα:

```bash
python scripts/run_kaggle_pipeline.py --stage all --profile pilot --run-id pilot-v1 --require-kaggle-dataset
```

Το προεπιλεγμένο pilot περιλαμβάνει 15 deterministic, stratified ερωτήσεις
(seed 42) και μόνο τα αντίστοιχα οικονομικά έγγραφα. Δεν είναι κατάλληλο για
αναφορά τελικών μετρικών σε δημοσίευση.

```bash
python scripts/run_kaggle_pipeline.py --stage all --profile full --run-id publication-v1 --require-kaggle-dataset
```


## Εκτέλεση Streamlit

```bash
streamlit run app.py
```

Η εφαρμογή αναμένει τα παραγόμενα artifacts κάτω από `data/processed/`.

## Κύρια Αποτελέσματα — ιστορικό thesis run

Τα παρακάτω νούμερα προέρχονται από το υπάρχον thesis run. Δεν αποτελούν ακόμη
publication-run αποτελέσματα, επειδή το repository πλέον επιβάλλει metadata-aware
reranking 20 Hybrid candidates σε τελικό top-5, ίδιο με το generation context.
Πριν χρησιμοποιηθούν σε paper,
πρέπει να αναπαραχθούν από νέο canonical Kaggle run και να αντικατασταθούν μαζί
με τα αντίστοιχα figures.

Τα historical artifacts δεν χρησιμοποιούνται για επιλογή νικητή. Το νέο
evaluation αποθηκεύει per-query retrieval/QA αποτελέσματα και χρησιμοποιεί
finance-aware αριθμητική αξιολόγηση για ποσά, μονάδες, ποσοστά και λογιστικά
αρνητικά. Η τελική επιλογή pipeline θα γίνει μόνο μετά από νέο paired full run.

Μετρικές ανάκτησης σε επίπεδο αποσπάσματος (passage-level Evidence Hit@K, κατώφλι set-based token-overlap F1 ≥ 0.3 — όπως αναφέρονται στη διπλωματική):

- Evidence Hit@1: 26.0%
- Evidence Hit@5: 44.7%
- MRR: 32.8%

Μετρικές γένεσης (RAGChecker, claim-level):

- RAGChecker F1: 27.0%
- RAGChecker Faithfulness: 50.8%
- RAGChecker Hallucination: 28.8%

> Σημείωση: Η κάλυψη σε **επίπεδο εγγράφου** (document-level top-k match) είναι υψηλότερη
> (περ. 63% στο top-1 και 86%–88% στο top-5/top-10), επειδή μετρά απλώς αν ανακτήθηκε
> το σωστό έγγραφο, ανεξάρτητα από το αν εντοπίστηκε το σωστό απόσπασμα. Τα νούμερα
> που αναφέρονται στη διπλωματική είναι τα αυστηρότερα passage-level παραπάνω.

## Σημειώσεις Αναπαραγωγής

- Για πλήρη αναπαραγωγή από την αρχή, χρησιμοποιήστε τον Kaggle runner από καθαρό clone.
- Τα notebooks που χρησιμοποιούν OpenAI/RAGAS/RAGChecker απαιτούν `OPENAI_API_KEY` στο περιβάλλον.
- Canonical runs αποτυγχάνουν αν η παραγωγή γίνει σε dry-run mode ή αν RAGAS/RAGChecker γράψουν placeholder metrics.
