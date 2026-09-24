# Corpus sources

These three files are the fixed benchmark corpus. They are committed to the repo
(not gitignored) so the eval is reproducible from a clone — no download or
license-gated fetch step required before `mmu index build`.

**`doc_id` is the slugified filename stem.** `data/questions.jsonl` references
documents by that slug (`gdpr`, `national-law`, `company-policy`). Renaming any
of these three files changes its `doc_id` and breaks the seed questions.

## gdpr.txt

Regulation (EU) 2016/679 (General Data Protection Regulation), full consolidated
text, sourced from EUR-Lex:
https://eur-lex.europa.eu/eli/reg/2016/679/oj

Reuse is permitted under Commission Decision 2011/833/EU of 12 December 2011 on
the reuse of Commission documents, subject to acknowledgement of the source.
EUR-Lex text is authoritative only in its published Official Journal form; this
copy is provided for benchmark use and is not the authoritative legal text.

## national-law.txt

Federal Data Protection Act (Bundesdatenschutzgesetz, BDSG), English
translation provided by the Language Service of the Federal Ministry of the
Interior, via gesetze-im-internet.de:
https://www.gesetze-im-internet.de/englisch_bdsg/

The translation is provided for information purposes only; only the German
original is legally binding. Includes amendments through Article 10 of the Act
of 23 June 2021.

## company-policy.txt

**Synthetic. Written for this project.** A fictional internal incident-response
policy for a fictional company ("Northwind Analytics GmbH"). It is not a real
policy, is not derived from any real company's document, and carries no legal
weight.

It exists because the cross-document questions need a third document whose
content is controlled: it deliberately contains `Data Protection Officer` and
`four hours`, and deliberately does **not** contain `72 hours` or `supervisory
authority`, so that questions `xdr-002` and `xdr-003` cannot be answered from
it alone. Licensed with the rest of the repo (MIT — see `LICENSE`).
