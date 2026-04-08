# Call Transcript Dataset Sources

Use these public sources as stand-ins for call records.

## Recommended public datasets

1. Hugging Face Datasets (dialogue/support corpora)
   - DailyDialog
   - MultiWOZ
   - Taskmaster
   - Schema Guided Dialogue (SGD)

2. Rasa and open customer-support corpora
   - intent-rich support dialogues useful for FAQ and policy-like responses

3. Kaggle dialogue/helpdesk datasets
   - search terms: "customer support chat", "helpdesk conversations", "call center transcript"

## Practical selection rule

Pick 2-4 datasets with these properties:

1. Turn-based conversation format
2. Clear speaker fields (user/agent)
3. Permissive license for research and dissertation use
4. University-service-like intents (accounts, deadlines, fees, registration)

## Normalization target format

Each row in JSONL should match:

```json
{"doc_id":"...","conv_id":"...","turn_id":1,"speaker":"agent","text":"..."}
```

## Important note

Always document dataset name, source link, version, and license in `docs/dataset-cards.md`.
