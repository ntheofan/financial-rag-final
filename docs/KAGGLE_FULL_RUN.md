# Canonical full run στο Kaggle

Το πλήρες pipeline εκτελείται στο Kaggle από καθαρό clone του repository. Ο
runner εκτελεί σειριακά τα 15 notebooks, σταματά στο πρώτο σφάλμα και δημιουργεί
run manifest, logs, executed notebooks, SHA-256 checksums και ZIP με τα artifacts.

## 1. Ρυθμίσεις Kaggle

Δημιουργήστε νέο Kaggle Notebook και ορίστε:

- Accelerator: GPU (T4 ή ισοδύναμη)
- Internet: On
- Secret με όνομα `OPENAI_API_KEY`, ενεργοποιημένο για το notebook
- Input dataset: `theofanisnikolaou/financebench-sample-dataset`

Χρησιμοποιήστε ένα καθαρό Kaggle session για κάθε canonical full run.

Το pipeline χρειάζεται ολόκληρο το dataset directory και όχι μόνο ένα DataFrame.
Επομένως δεν χρειάζεται να καλέσετε το `KaggleDatasetAdapter.PANDAS` snippet με
κενό `file_path`. Ο runner βρίσκει πρώτα το mounted Input και, αν δεν υπάρχει,
χρησιμοποιεί `kagglehub.dataset_download()` για την πλήρη τελευταία έκδοση. Στο
run manifest καταγράφονται handle, resolved path, αριθμός αρχείων, μέγεθος και
SHA-256 fingerprint του dataset.

## 2. Clone και εγκατάσταση

Στο πρώτο cell:

```python
!git clone https://github.com/ntheofan/financial-rag-final.git
%cd /kaggle/working/financial-rag-final
!python -m pip install -r requirements.txt
```

Αν έχει ήδη δημιουργηθεί ο φάκελος σε προηγούμενη απόπειρα, κάντε restart το
Kaggle session αντί να επαναχρησιμοποιήσετε πιθανά ενδιάμεσα artifacts.

## 3. Φόρτωση του Secret

Στο επόμενο cell:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["OPENAI_API_KEY"] = UserSecretsClient().get_secret("OPENAI_API_KEY")
print("OPENAI_API_KEY loaded:", bool(os.environ.get("OPENAI_API_KEY")))
```

Το κλειδί παραμένει μεταβλητή περιβάλλοντος και δεν γράφεται σε notebook,
manifest ή artifact.

## 4. Προέλεγχος χωρίς εκτέλεση

```python
!python scripts/run_kaggle_pipeline.py \
    --stage all \
    --run-id publication-dry-run \
    --require-kaggle-dataset \
    --dry-run
```

Ο προέλεγχος επαληθεύει το pipeline configuration, τη σειρά και την ύπαρξη των
notebooks. Δεν εκτελεί τα πειράματα.

## 5. Πλήρης εκτέλεση

```python
!python scripts/run_kaggle_pipeline.py \
    --stage all \
    --run-id publication-v1 \
    --require-kaggle-dataset
```

Μην χρησιμοποιείτε `--allow-existing-artifacts` για publication run. Ο runner
απαιτεί καθαρά `data/interim/` και `data/processed/`, ώστε να μην αναμειχθούν
artifacts από διαφορετικές εκτελέσεις.

## 6. Παραδοτέα του run

Με `--run-id publication-v1` δημιουργούνται:

```text
/kaggle/working/financial-rag-final/runs/publication-v1/
├── artifacts/
├── executed_notebooks/
├── logs/
├── checksums.sha256
├── environment.txt
└── run_manifest.json

/kaggle/working/financial-rag-final/runs/publication-v1.zip
```

Κατεβάστε το ZIP από το file browser του Kaggle πριν κλείσετε το session. Ένα
run θεωρείται έγκυρο μόνο όταν το `run_manifest.json` έχει
`"status": "completed"`. Τα notebooks παραγωγής και LLM evaluation αποτυγχάνουν
αν λείπει το API key ή αν δημιουργηθούν dry-run/placeholder αποτελέσματα.

## Προαιρετική εκτέλεση ανά στάδιο

Ο runner υποστηρίζει `bootstrap`, `retrieval`, `generation` και `evaluation`:

```python
!python scripts/run_kaggle_pipeline.py --stage retrieval --run-id retrieval-v1
```

Η τμηματική εκτέλεση προϋποθέτει ότι τα artifacts των προηγούμενων σταδίων
βρίσκονται ήδη στις αναμενόμενες διαδρομές του ίδιου checkout. Για το τελικό
πειραματικό αποτέλεσμα προτιμάται το ενιαίο `--stage all`.
