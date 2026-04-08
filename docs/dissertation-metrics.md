# Dissertation Metrics Draft

## Primary metrics

- Specificity (0-2): generic to concrete evidence-grounded answers.
- Hallucination proxy: unsupported claims flagged in manual review.
- Refusal quality: does the assistant correctly abstain when evidence is absent?
- Latency: time per answer across hardware profiles.

## Experiment schedule

- Baseline: no RAG
- RAG only
- RAG + chatlog ingestion

Evaluate each stage on the same fixed question set.
