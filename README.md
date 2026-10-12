# Semantic concept detection on claim descriptions: handover

Prototyping spike results for detecting stored concepts (for example "fluid leak") in free-text damage and accident descriptions, with paraphrase tolerance and negation handling.

Read in this order:

1. `HANDOVER.md`: conclusions, measurements, decisions, open items, first actions.
2. `PoC-semantic-concept-detection.md`: design doc, 13 sections.
3. `LABELING_README.md`: labeling rules and concept definitions.
4. `claims.json`: 200 public-domain NHTSA vehicle complaint narratives, business phone numbers redacted.
5. `labeling_sheet.csv`: 200 rows x 6 concept columns, 0/1 labels, some cells blank.

No customer data, company names or internal identifiers. Python harness, model checkpoints and session logs are not included.
