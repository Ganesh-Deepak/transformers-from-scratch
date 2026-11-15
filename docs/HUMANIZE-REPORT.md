# Humanizing the Transformers prose: before and after

Rules: github.com/blader/humanizer (SKILL.md). Each chapter went through draft, check and final;
`tests/humanize/seg.py` refuses any rewrite that changes a number, inline code, link, maths span or
shape badge, so every fact is preserved by construction. Scores from `tests/humanize/scan.py` over the
prose only (code, tables, figures excluded). Tells = AI-writing patterns per 1,000 words. Reading ease =
Flesch (higher is easier; 60-70 is plain English).

| chapter | tells/1k before | after | bold/1k before | after | words/sentence before | after | reading ease before | after |
|---|---|---|---|---|---|---|---|---|
| ch00-numbers.html | 15.2 | 0.2 | 36.5 | 29.4 | 11.5 | 11.9 | 73.0 | 75.4 |
| ch01-tensors.html | 13.1 | 0.3 | 16.7 | 15.1 | 14.1 | 14.8 | 68.4 | 69.5 |
| ch02-matmul.html | 10.5 | 0.0 | 19.8 | 18.3 | 13.8 | 14.4 | 68.0 | 70.3 |
| ch03-tokenization.html | 13.5 | 0.3 | 11.9 | 11.1 | 17.1 | 17.0 | 53.0 | 57.3 |
| ch04-embeddings.html | 8.2 | 0.0 | 11.6 | 10.4 | 16.9 | 17.2 | 56.9 | 60.2 |
| ch05-attention.html | 11.5 | 0.2 | 14.2 | 12.5 | 16.1 | 16.3 | 62.3 | 67.1 |
| ch06-mha.html | 14.2 | 0.0 | 15.1 | 12.5 | 14.8 | 15.1 | 61.7 | 65.7 |
