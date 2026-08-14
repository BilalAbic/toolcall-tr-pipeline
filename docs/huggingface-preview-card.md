---
pretty_name: Turkish Tool-Calling Quality-Gated Preview
language:
- tr
license: cc-by-4.0
task_categories:
- text-generation
tags:
- tool-calling
- function-calling
- turkish
- quality-gated
- silver-candidate
- pre-release
- sampled-human-review
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Turkish Tool-Calling Quality-Gated Preview

> **Preview, not Gold:** this public research preview is quality-gated but is not
> human-verified at dataset level. The pipeline's formal `publish_allowed=false`
> state remains unchanged.

## Review statement

A maintainer performed a limited manual spot-check of **six** diverse records,
covering tool calls, multiple calls, no-tool behavior, and clarification. This
is a qualitative sample review only; it is not a row-by-row human annotation,
acceptance, or safety certification for all 733 records.

## Composition

- Split: `train`
- Rows: 733 quality-gated `silver_candidate` records
- Upstream candidate cohort: 1,000 episodes
- Sources: 258 Salesforce xLAM records and 475 NVIDIA When2Call records
- Conversation shape: single-turn `user -> assistant` only
- Format: JSONL with `id`, `messages`, `tools`, source provenance, quality tier,
  and consensus status
- Train SHA-256: `57951945ab70e8aa4d7556b6e6d4e692a429b80e8c95a51ed3b5321983888956`

## What quality-gated means

1. Revision-pinned source snapshot, ingest, canonicalization, and conflict audit.
2. Host-owned preservation of tool names, arguments, schemas, and call structure.
3. DeepSeek Flash translation with a single safe DeepSeek Pro fallback.
4. OpenAI mini quality evaluation; non-passes and a deterministic pass sample
   escalate to the strong judge.
5. Strict JSONL and JSON Schema validation, content hashes, and immutable manifests.

Every tool-call argument in this preview passed validation against its declared
tool JSON Schema. `quality-gated` does not mean `human-verified` or `Gold`.

## Source provenance and attribution

- [Salesforce xLAM function-calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k),
  revision `26d14ebfe18b1f7b524bd39b404b50af5dc97866`, CC-BY-4.0.
- [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call),
  revision `0582f7749df63a96fdc3070932e83e72396ace53`, CC-BY-4.0.

Every row retains `source_dataset_namespace` and `source_snapshot_ids`.
Technical fields are preserved rather than rewritten by the translation model.

## Limitations and responsible use

- This is a public preview for research and inspection, not a final training or
  benchmark release.
- A small-sample manual review cannot establish quality, safety, or provenance
  for every row. Do not describe this preview as human-verified.
- Source-derived text may contain contact-like or credential-related phrases.
  No PII-free claim is made; review and policy checks remain the user's duty.
- The preview has no multi-turn conversations. Later batches need separately
  reviewed multi-turn coverage.

## Reproducibility

The pipeline, validation contracts, quality gates, and release process are in
[BilalAbic/toolcall-tr-pipeline](https://github.com/BilalAbic/toolcall-tr-pipeline).
The source package ID is
`hfpackage_46306948b55f91188bc307aaa57180196586b9273e87bfc2f6752dc0a8b16c20`.
