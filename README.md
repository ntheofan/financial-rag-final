# Financial RAG Final

Clean final repository for the FinanceBench RAG thesis pipeline.

## Structure

- `app.py`: Streamlit app for live querying and results exploration
- `notebooks/`: final notebook pipeline from setup to evaluation
- `data/raw/`: source PDFs and FinanceBench metadata
- `data/interim/`: parsed markdown and intermediate tables
- `data/processed/`: chunks, embeddings, retrieval, QA, and evaluation outputs

## Final Notebook Pipeline

1. `01_setup_and_config.ipynb`
2. `02_load_financebench_sample.ipynb`
3. `03_parse_pdfs_with_docling.ipynb`
4. `04_clean_markdown_and_inspect.ipynb`
5. `05_chunking.ipynb`
6. `06_embeddings_and_vectorstore.ipynb`
7. `07_retrieval_baseline.ipynb`
8. `08_hybrid_retrieval.ipynb`
9. `09_reranking.ipynb`
10. `10_rag_qa_baseline.ipynb`
11. `11_error_analysis.ipynb`
12. `12_evaluation.ipynb`
13. `13_ragas_openai_kaggle_fast.ipynb`
14. `14_ragchecker_evaluation_notebook.ipynb`
15. `15_thesis_evaluation_analysis.ipynb`

## Install

For the full notebook workflow:

```bash
pip install -r requirements.txt
```

For the Streamlit app only:

```bash
pip install -r streamlit_requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app expects the generated pipeline artifacts under `data/processed/`.
