# Report: Sanskrit–English retrieval with a fine-tuned embedding model

Option 2 of the take-home. Code, data and results are described in [README.md](README.md).

## 1. Problem understanding

Given a query as an English question, a Devanagari verse or romanised Sanskrit (IAST or plain ASCII), retrieve the matching Bhagavad Gita verse, either its Sanskrit text or its English translation. The interesting difficulty is that these inputs look completely different to a model. A user is likely to type `karmanyevadhikaraste` rather than `karmaṇy evādhikāras te`, and an English-centred model has no reason to link either to the meaning.

I treated this as three related problems and evaluated each separately:

- **Cross-lingual:** Devanagari or English query → the other language.
- **Cross-script:** ASCII/IAST ↔ Devanagari for the same text.
- **Partial and natural queries:** half-lines, 4-word fragments and free-form questions.

## 2. Dataset preparation

**Source.** `OEvortex/Bhagavad_Gita` (700 verses: Devanagari, Hindi, English). The English text is often a loose paraphrase with commentary, not a literal translation.

**Cleaning.** Around 600 rows had verse numbers embedded in the Sanskrit (`॥ २.४७ ॥`) and some had narrator prefixes ("… उवाच"). Left in, the model could match on the number instead of the meaning, so both were stripped (634 rows changed; 3 narrator prefixes remained). Text was NFC-normalised. IAST is generated from Devanagari with `indic-transliteration`, and ASCII is IAST with diacritics removed. Verse ids were reloaded as strings after I found that `1.10` was being read as the float `1.1` and colliding with verse `1.1` (caught by a uniqueness check).

**Split by chapter.** Train = chapters 1–13 (523 verses), val = 14–16 (71), test = 17–18 (106). Neighbouring verses share themes and wording, so a random split would leak. There were no id overlaps and no duplicate Sanskrit text across splits.

**Training pairs (5,669).** Each verse yields queries of 11 types:

- Rule-based: full verse in Devanagari / IAST / ASCII → English; first half-line (Devanagari, ASCII) → English; first 4 words (ASCII) → English; ASCII verse → Devanagari; English → Devanagari.
- LLM-written (Gemini API, `<model used>`): a natural question, a keyword query and a casual query per verse. The prompt forbade reusing the translation's wording. A 3-word-overlap filter removed 11 of 2,100 questions, mostly common phrases.

**Hard negatives.** My first attempt, mining negatives that score close to the query with a score margin, failed: base e5 scores are compressed into a narrow range, so 1,352 of 5,669 pairs got fewer than 5 negatives and the ones found were unrelated (a question about the cosmic form got negatives about archers' names). The final method takes, for each verse, its most similar other verses under the base model (pool of 15), skipping adjacent verses and near-duplicates (similarity ≥ 0.95), and samples 5 per pair. This is independent of the query's script, which matters because base e5 cannot read romanised text.

**Eval sets.** Val has 772 queries over 71 verses and test has 1,144 over 106. Each split is searched only against its own verses.

## 3. Base model selection

I compared the tokenizers on six well-known verses.

| Model | Devanagari tokens/char | IAST | ASCII | English | `[UNK]` (Devanagari) |
|---|---|---|---|---|---|
| multilingual-e5-small | 0.44 | 0.53 | 0.43 | 0.33 | 0 |
| bge-small-en-v1.5 | 0.68 | 0.44 | 0.44 | 0.28 | 9 |

`bge-small-en` splits Devanagari into single characters and emits unknown tokens. e5 keeps whole units (`▁कर्म`, `▁फल`). IAST is the most fragmented script even in e5, because diacritics split off as separate pieces (`▁karma`, `ṇ`, `y`, …). Plain ASCII is about 20% more compact (0.43 vs 0.53 tokens/char), so it is the cheaper input form. bge's tokenizer strips accents, so it gives the same result for both. The IAST/ASCII fragmentation is the transliteration mismatch, and it predicted the weakness seen later. `gte-multilingual-base` shares e5's tokenizer, but it failed to run in my setup (a cuBLAS error on the T4 GPU and an index-out-of-bounds error on CPU, neither diagnosed), so it was not scored.

A 6-pair zero-shot smoke test (chance MRR ≈ 0.41) gave e5 an MRR of 0.81 for Devanagari → English and 1.00 for English → Devanagari, against 0.39 and 0.42 for bge, with IAST → English near chance for both (0.41, 0.40). On the larger val set the untuned e5 was also near chance for ASCII and IAST (R@1 0.014–0.028, chance 0.014).

I chose **multilingual-e5-small** (118M parameters): it is the only tested model that handles Devanagari, it already aligns English and Devanagari reasonably, and it leaves a clear weakness (romanised text) to fix.

I did not adapt the tokenizer. It has no unknown tokens on Devanagari, and adding tokens would need new embeddings trained from very little data (523 verses), which seemed a poor trade. Instead I trained on all three scripts so the model learns the mapping through the existing subwords.

## 4. Fine-tuning approach

- **Loss:** `MultipleNegativesRankingLoss` on (query, positive, hard negative), plus in-batch negatives.
- **Batching:** a no-duplicates sampler. Each verse appears in about 10 queries, so an ordinary batch could contain the same verse twice and penalise a correct match.
- **Settings:** 3 epochs (about 530 steps), batch 32, lr 2e-5, 10% warmup, fp16, max sequence length 256, seed 42.
- **Prefixes and normalisation:** `query: ` / `passage: ` (required by e5), L2-normalised embeddings, cosine similarity.
- **Chunking:** one verse is one document. Verses are short, so no sub-verse chunking was needed.

## 5. Hardware constraints and optimisations

Everything ran on a single Colab T4. The T4 has no bf16 support, so training used fp16. The model is small enough for a few minutes of training with batch 32, and inference works on CPU (I used CPU for hard-negative mining). I hit two GPU faults: the `gte` cuBLAS error above and a `device-side assert` that persisted across a runtime restart until I moved to a new notebook. I did not find the root cause of either. Keeping everything on Drive made the restarts cheap.

## 6. Evaluation methodology

- **Metrics:** Recall@1/5/10, MRR and nDCG@10, reported per query type, since the aggregate is dominated by the 8 rule-based types.
- **Protocol:** val was used for every decision (model choice, ablations). The test set was scored once, after all decisions were made.
- **Uncertainty:** a bootstrap that resamples gold verses (2,000 draws) gives confidence intervals on the MRR gain. Queries about the same verse are correlated, so resampling queries directly would overstate confidence.
- **Corpora are small** (71 and 106 verses), so absolute scores are high. Only the before/after comparison is meaningful. The val set also contains one duplicated English translation, so one query has two valid gold verses.

## 7. Results

**Test set, MRR (base → fine-tuned).** Recall and nDCG are in `results/`.

| Query type | Base | Fine-tuned | ΔMRR [95% CI] |
|---|---|---|---|
| ascii_full | 0.071 | 0.282 | +0.210 [0.145, 0.282] |
| ascii_half | 0.095 | 0.161 | +0.066 [0.003, 0.126] |
| ascii_short | 0.079 | 0.151 | +0.072 [0.023, 0.125] |
| iast_full | 0.057 | 0.244 | +0.186 [0.127, 0.252] |
| ascii_to_deva | 0.419 | 0.841 | +0.422 [0.346, 0.498] |
| deva_full | 0.267 | 0.598 | +0.330 [0.241, 0.416] |
| deva_half | 0.245 | 0.428 | +0.184 [0.115, 0.261] |
| english_to_deva | 0.370 | 0.605 | +0.235 [0.152, 0.316] |
| q_natural | 0.390 | 0.526 | +0.137 [0.066, 0.205] |
| q_keyword | 0.422 | 0.591 | +0.169 [0.108, 0.236] |
| q_casual | 0.411 | 0.511 | +0.100 [0.039, 0.163] |
| **ALL** | 0.257 | 0.451 | +0.193 [0.164, 0.223] |

Every interval excludes zero (`ascii_half` only barely), so no type regressed. Overall Recall@1 rose from 0.169 to 0.340.

**Ablations (validation MRR, single seed).**

| Type | Base | Full | No hard negatives | No romanised training |
|---|---|---|---|---|
| ascii_full | 0.077 | 0.247 | 0.241 | 0.091 |
| ascii_to_deva | 0.376 | 0.862 | 0.869 | 0.164 |
| deva_full | 0.233 | 0.586 | 0.565 | 0.580 |
| q_natural | 0.303 | 0.546 | 0.567 | 0.590 |

- **Hard negatives made no measurable difference** here. Possible reasons, untested: a small corpus, in-batch negatives already supplying enough signal, and some mined negatives being false negatives.
- **Romanised training is what fixes romanised queries.** Without it, `ascii_to_deva` falls to 0.164, below the untuned baseline of 0.376. Training the Devanagari side toward English meaning pulled it away from its own Latin spelling until romanised data put them back together.
- The `q_*` rows are slightly higher without romanised data. With about 70 queries per type and one seed this is suggestive at most.

**Out-of-domain check.** On 21 Ayurveda sentences with English glosses I wrote myself, the fine-tuned model matched the base on Devanagari (MRR 0.921 vs 0.929) and English questions (0.854 vs 0.854). Romanised MRR was 0.457 vs 0.402, a difference of a few queries. The set is too small and too easy to say more than "no sign of over-specialisation".

**Demo check on verse 2.47** (a training verse; ranks out of 700): ASCII half-line 131 → 4, ASCII full verse 368 → 3, Devanagari half-line 314 → 8. The Devanagari full verse went 446 → 7. Even on a training verse the model does not reach rank 1, which fits three short epochs at a low learning rate.

## 8. Failure cases and error analysis

- **Romanised → English is still the weak area.** On test, the gold verse is outside the top 10 for 52% of `ascii_full`, 51% of `iast_full`, 67% of `ascii_half` and 69% of `ascii_short` queries (base: 85–89%). In contrast `ascii_to_deva` misses only 10%. The model has learned that the two scripts of one text match, but mapping Sanskrit sound to English meaning is much harder from 523 verses. This is my reading and I did not test it directly.
- **Hub verses.** Some verses attract many wrong answers. The fine-tuned model's worst are 18.70 (31 wrong top-1s), 18.20 (24), 18.65 (23), against about 7 expected if errors were spread evenly. For the base model the worst verse (18.29) took 180. Fine-tuning reduced the concentration but did not remove it. Three of the top confusions (18.4→18.40, 18.6→18.60, 18.7→18.70) looked like a numbering bug, but they turned out to be hub effects between unrelated texts.
- **Reasonable mistakes dominate the failures I inspected.** Examples: 18.36 → 18.4 (both open "Now … I will disclose …"), 18.26 → 17.17 (both about harmony), 18.52 → 17.7 (both about moderate eating). Only about 9.5% of wrong top-1 results are an adjacent verse, against 1.8% by chance.
- **Label noise.** For the query "How can a person attain the ultimate state of spiritual detachment?" the gold verse is 18.5, but the model's answer 18.49 arguably fits better. The LLM questions were not manually verified, so scores on `q_*` types may be understated.
- **Individual regressions exist.** Verse 18.52 went from base rank 1 to rank 3 and 18.5 from 21 to 69. The gains are on average and not for every query.
- **Difficulty is concentrated in chapter 18** (78 of 106 test verses), which restates renunciation, duty and the three gunas from several angles. Verse length had only a weak effect (Spearman −0.17).
- **Source data quality.** 148 of 700 rows contain `...`, some texts start with `…`, and 18.18 loses its first letter. I did not check how these are distributed across splits.

There is no generation step, so hallucination is not applicable. The analogue here is a confident wrong retrieval, and similarity scores do not signal it: base e5 scores sit in a narrow band (about 0.83–0.92), and fine-tuning spread them out (about 0.33–0.74), so scores are not comparable across the two models.

## 9. Challenges encountered

- The GPU faults described in section 5.
- The verse-id float bug, and my first hard-negative method producing unrelated negatives (section 2).
- Base e5's compressed scores, which broke the score-margin filter.
- A retired Gemini model name (404) when generating questions, fixed by listing the models my key could call.
- An early smoke test on 6 verses was too easy to separate the models; the 71- and 106-verse sets were far more informative.

## 10. What I would improve

- **Verify the labels:** manually review a sample of the LLM questions and report metrics with and without ambiguous ones.
- **Fix hubness:** mine hub verses as hard negatives, or apply a query-time correction such as CSLS. Both are untested ideas.
- **Romanised robustness:** augment training with spelling variants and noise (missing sandhi, inconsistent diacritics), since real users type inconsistently.
- **More data:** more Sanskrit and English pairs (other Upanishads; several English translations per verse) and a larger validation corpus.
- **Better evidence:** several training seeds, and a corpus of thousands of passages instead of 71 or 106, closer to real search.
- **Larger base model** (for example `gte-multilingual-base` once the GPU error is understood), and a comparison against full fine-tuning and LoRA.
- **Deployment:** the model is about 490 MB, small enough to serve on CPU. A real system would need chunking for longer passages, a vector index, and a fallback for romanised queries, which remain the weakest input.
