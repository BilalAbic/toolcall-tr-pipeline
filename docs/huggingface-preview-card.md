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

> **Preview, not Gold:** This public research preview is quality-gated, but it
> is not human-verified at dataset level. The pipeline's formal
> `publish_allowed=false` state remains unchanged.

## Review statement

A maintainer performed a limited manual spot-check of **six** diverse records,
covering tool calls, multiple calls, no-tool behavior, and clarification. This
is a qualitative sample review only; it is not a row-by-row human annotation,
acceptance, or safety certification for all 733 records.

## Data at a glance

| Property | Value |
| --- | --- |
| Split | `train` |
| Language | Turkish natural-language fields |
| Records | 733 quality-gated `silver_candidate` episodes |
| Candidate cohort | 1,000 source episodes |
| Sources | 258 Salesforce xLAM + 475 NVIDIA When2Call records |
| Conversation shape | Single-turn `user -> assistant` |
| Format | JSONL with OpenAI-style messages and function tools |

### Behavior mix

| Final assistant behavior | Records |
| --- | ---: |
| Tool call | 347 |
| No-tool response or clarification | 386 |
| **Total** | **733** |

Tool-call records include 216 single-call, 106 two-call, 23 three-call, one
four-call, and one nine-call assistant turns. No-tool examples include both
capability refusals and clarification questions when required information is
missing.

### Row structure

Each JSONL line contains an episode `id`, OpenAI-style `messages`, the available
function `tools`, and provenance fields. An assistant message contains either
Turkish natural-language `content` or structured `tool_calls`; tool names,
argument keys, schemas, enum values, and argument values remain technical fields
and are preserved exactly by the host-side pipeline.

```json
{"id":"ep_<content-addressed-id>","messages":[...],"tools":[...],"source_dataset_namespace":"...","quality_tier":"silver_candidate"}
```

## What quality-gated means

1. Revision-pinned source snapshot, ingest, canonicalization, and conflict audit.
2. Host-owned preservation of tool names, arguments, schemas, and call structure.
3. DeepSeek Flash translation with a single safe DeepSeek Pro fallback.
4. OpenAI mini quality evaluation; non-passes and a deterministic pass sample
   escalate to the strong judge.
5. Strict JSONL and JSON Schema validation, content hashes, and immutable manifests.

Every tool-call argument in this preview passed validation against its declared
tool JSON Schema. `quality-gated` does not mean `human-verified` or `Gold`.

## Intended use

Use this preview to inspect Turkish tool-calling supervision, prototype parsers
for the `messages` and `tools` fields, and evaluate models on tool selection,
no-tool behavior, and clarification decisions. It is not a certified benchmark,
a final training release, or evidence that a model is safe for real-world tool
execution.

## Source provenance and attribution

- [Salesforce xLAM function-calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k),
  revision `26d14ebfe18b1f7b524bd39b404b50af5dc97866`, CC-BY-4.0.
- [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call),
  revision `0582f7749df63a96fdc3070932e83e72396ace53`, CC-BY-4.0.

Every row retains `source_dataset_namespace` and `source_snapshot_ids`.
Technical fields are preserved rather than rewritten by the translation model.

## Citation and attribution

This derivative preview keeps the CC-BY-4.0 attribution requirements of the
upstream dataset cards. If you use records originating from either source,
please cite the corresponding upstream work.

### Cite Salesforce xLAM / APIGen

```bibtex
@article{liu2024apigen,
  title = {APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets},
  author = {Liu, Zuxin and Hoang, Thai and Zhang, Jianguo and Zhu, Ming and Lan, Tian and Kokane, Shirley and Tan, Juntao and Yao, Weiran and Liu, Zhiwei and Feng, Yihao and others},
  journal = {arXiv preprint arXiv:2406.18518},
  year = {2024}
}
```

### Cite NVIDIA When2Call

```bibtex
@inproceedings{ross-etal-2025-when2call,
  title = {When2Call: When (not) to Call Tools},
  author = {Ross, Hayley and Mahabaleshwarkar, Ameya Sunil and Suhara, Yoshi},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  year = {2025},
  address = {Albuquerque, New Mexico},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2025.naacl-long.174/},
  pages = {3391--3409}
}
```

## Optional standalone conversation export

`data/train.jsonl` is the canonical, provenance-preserving preview format.
Consumers who need only the compact `conversation` JSON array can use the
[standalone exporter](https://github.com/BilalAbic/toolcall-tr-pipeline/tree/main/standalone_tools).
It reads a local JSONL copy and writes a separate file without changing the
source data or invoking a model/API. The simplified export deliberately omits
IDs and quality/provenance metadata, so it should not replace the canonical
JSONL for auditing or release work.

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
GitHub contains code, tests, documentation, and this Card source; it does not
contain the JSONL preview data. The data files are hosted only in this Hugging
Face dataset repository. Machine-readable integrity metadata remains in
`manifest.json` for reproducibility, not as part of this reader-facing summary.