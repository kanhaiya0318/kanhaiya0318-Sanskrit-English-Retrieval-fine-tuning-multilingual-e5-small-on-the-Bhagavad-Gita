# Sanskrit–English Retrieval: fine-tuning multilingual-e5-small on the Bhagavad Gita

Take-home assignment, Option 2 (multilingual embedding model for Sanskrit + English retrieval).

## Summary

I fine-tuned `intfloat/multilingual-e5-small` with contrastive learning so that English questions, Devanagari verses and romanised Sanskrit (IAST and plain ASCII) all retrieve the right Gita verse.

On held-out chapters 17–18 (1,144 queries over 106 verses), overall MRR rises from **0.257 to 0.451** (ΔMRR +0.193, 95% CI [0.164, 0.223]). Every query type improves and every confidence interval excludes zero (`ascii_half` only barely). Romanised Sanskrit → English is still the weakest area in absolute terms (test MRR 0.15–0.28).

Full write-up: [REPORT.md](REPORT.md).

## Repository layout

```
<repo-name>/
├── README.md
├── REPORT.md
├── notebooks/
│   ├── 01_data_prep.ipynb       # Stages 1-2: tokenizer study, cleaning, splits, training pairs, eval sets
│   ├── 02_train_eval.ipynb      # Stages 3-4: baseline, fine-tuning, ablations, test eval, failure analysis
│   └── 03_demo.ipynb            # Stage 5: retrieval demo (Gradio)
├── data/
│   ├── gita_clean.csv
│   ├── gita_{train,val,test}.csv
│   ├── train_pairs.jsonl        # 5,669 training pairs with 5 hard negatives each
│   └── {val,test}_eval.jsonl    # labeled eval queries
├── results/
│   ├── baseline_*_metrics.csv, ft_v1_*_metrics.csv
│   ├── abl_*_val_metrics.csv    # ablations
│   ├── final_test_comparison.csv
│   └── sample_outputs.md
└── models/                      # optional: adapter/checkpoint or a download link
```

## Setup

Developed on Google Colab with a T4 GPU.

```bash
pip install sentence-transformers datasets accelerate pandas numpy indic-transliteration gradio google-genai
```

`google-genai` is only needed to regenerate the synthetic questions. The generated questions are already included in `data/`.

Notebooks read and write a Google Drive folder (`MyDrive/sanskrit_retrieval/`). Change `ROOT` / `D` at the top of each notebook if you use another location.

## Reproduce (run in order)

1. `01_data_prep.ipynb` – builds the cleaned dataset, the chapter-level splits, the training pairs with hard negatives and the eval sets. To regenerate the LLM questions, add `GEMINI_API_KEY` to Colab secrets.
2. `02_train_eval.ipynb` – baseline on val, fine-tuning, ablations, a single test evaluation and the failure analysis.
3. `03_demo.ipynb` – search demo that compares the base and fine-tuned models.

## Data

- **Source:** `OEvortex/Bhagavad_Gita` on Hugging Face, 700 verses with Devanagari and an English translation. Check its license before redistributing the data.
- **Cleaning:** NFC normalisation, removal of embedded verse numbers and speaker tags. IAST is generated from Devanagari with `indic-transliteration`, and ASCII is IAST with diacritics stripped.
- **Split by chapter, not by verse:** train = chapters 1–13 (523 verses), val = 14–16 (71), test = 17–18 (106). A random split would leak neighbouring verses.
- **Eval corpus** = the split's own verses only (71 for val, 106 for test), so absolute scores are high and only the before/after comparison is meaningful.

### Query types

| Type | Query → target |
|---|---|
| `deva_full`, `iast_full`, `ascii_full` | full verse in that script → English translation |
| `deva_half`, `ascii_half` | first half-line → English translation |
| `ascii_short` | first 4 words in ASCII → English translation |
| `ascii_to_deva` | ASCII verse → Devanagari verse |
| `english_to_deva` | English translation → Devanagari verse |
| `q_natural`, `q_keyword`, `q_casual` | LLM-written question → English translation |

Questions were generated with the Gemini API (`<model used>`) from each verse's translation, with instructions not to reuse its wording. Queries sharing a 3-word phrase with the translation were removed (11 of 2,100).

## Method

- **Base model:** `intfloat/multilingual-e5-small` (118M parameters). The tokenizer handles Devanagari without unknown tokens, unlike `bge-small-en-v1.5`, which produced 9 `[UNK]` tokens on six test verses and split Devanagari into single characters.
- **Loss:** `MultipleNegativesRankingLoss` on (query, positive, hard negative) triplets, plus in-batch negatives.
- **Hard negatives:** for each verse, the most similar other verses under the base model, skipping adjacent verses and near-duplicates. Five per pair, one sampled per epoch step.
- **Batching:** `NO_DUPLICATES` sampler, so the same verse never appears twice in a batch.
- **Settings:** 3 epochs, batch 32, lr 2e-5, warmup 10%, fp16, max sequence length 256, seed 42. Training takes a few minutes on a T4.
- **Prefixes:** `query: ` and `passage: ` are required by e5 and used throughout.

## Results

### Test set (chapters 17–18), base vs fine-tuned

| Query type | n | R@1 base | R@1 ft | MRR base | MRR ft | ΔMRR [95% CI] |
|---|---|---|---|---|---|---|
| ascii_full | 106 | 0.028 | 0.170 | 0.071 | 0.282 | +0.210 [0.145, 0.282] |
| ascii_half | 98 | 0.051 | 0.082 | 0.095 | 0.161 | +0.066 [0.003, 0.126] |
| ascii_short | 106 | 0.038 | 0.075 | 0.079 | 0.151 | +0.072 [0.023, 0.125] |
| ascii_to_deva | 106 | 0.321 | 0.783 | 0.419 | 0.841 | +0.422 [0.346, 0.498] |
| deva_full | 106 | 0.160 | 0.481 | 0.267 | 0.598 | +0.330 [0.241, 0.416] |
| deva_half | 98 | 0.163 | 0.306 | 0.245 | 0.428 | +0.184 [0.115, 0.261] |
| english_to_deva | 106 | 0.236 | 0.491 | 0.370 | 0.605 | +0.235 [0.152, 0.316] |
| iast_full | 106 | 0.019 | 0.132 | 0.057 | 0.244 | +0.186 [0.127, 0.252] |
| q_casual | 103 | 0.301 | 0.359 | 0.411 | 0.511 | +0.100 [0.039, 0.163] |
| q_keyword | 106 | 0.274 | 0.462 | 0.422 | 0.591 | +0.169 [0.108, 0.236] |
| q_natural | 103 | 0.262 | 0.379 | 0.390 | 0.526 | +0.137 [0.066, 0.205] |
| **ALL** | 1144 | 0.169 | 0.340 | 0.257 | 0.451 | +0.193 [0.164, 0.223] |

Confidence intervals come from a bootstrap that resamples gold verses, since queries about the same verse are correlated. Recall@5/10 and nDCG@10 are in `results/`.

### Ablations (validation MRR, single seed)

| Type | base | full model | no hard negatives | no romanised training |
|---|---|---|---|---|
| ascii_full | 0.077 | 0.247 | 0.241 | 0.091 |
| ascii_to_deva | 0.376 | 0.862 | 0.869 | 0.164 |
| deva_full | 0.233 | 0.586 | 0.565 | 0.580 |
| q_natural | 0.303 | 0.546 | 0.567 | 0.590 |
| ALL* | 0.249 | 0.460 | 0.458 | 0.349 |

\* The last column's ALL is lower only because it includes romanised query types it was not trained on, so compare per-type rows.

- Hard negatives made no measurable difference at this data size.
- Romanised queries only improve when romanised text is in the training data. Without it, `ascii_to_deva` falls below the untuned baseline.

## Failure analysis (summary)

- Romanised → English remains weak: 51–69% of these queries still have the gold verse outside the top 10.
- Errors concentrate on a few "hub" verses (e.g. 18.70, 18.20). Fine-tuning reduced this concentration compared with the base model but did not remove it.
- Many wrong answers are reasonable (same theme or near-paraphrase). Only about 1 in 10 wrong top-1 results is an adjacent verse.
- On a small out-of-domain Ayurveda set (21 sentences), the fine-tuned model matched the base on Devanagari and English queries, showing no sign of over-specialisation. Its small gain on romanised queries was not verified statistically.

## Limitations

- The LLM-written questions were **not manually verified**. Generic questions can fit several verses, so scores on `q_*` types may be understated.
- Eval corpora are small (71 and 106 verses), and results come from a single training seed. The val set contains one duplicated English translation, so one query has two valid gold verses.
- 148 of 700 rows in the source data contain `...` artefacts, and some translations are loose paraphrases with commentary. I did not check how the artefacts are distributed across the splits.
- Test chapters 17–18 may differ in style from the rest of the Gita.
- The demo searches all 700 verses, including training verses, so demo results look better than the held-out numbers above.

## Using the model

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("<path or hub id of the fine-tuned model>")
q = model.encode(["query: karmanyevadhikaraste ma phalesu kadacana"], normalize_embeddings=True)
d = model.encode(["passage: " + text for text in passages], normalize_embeddings=True)
scores = q @ d.T
```

## Credits

Dataset: `OEvortex/Bhagavad_Gita`. Base model: `intfloat/multilingual-e5-small`. Built with Sentence-Transformers.
