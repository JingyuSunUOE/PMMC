# MemGallery Local Assets

This directory contains a locally retained 20-scenario MemGallery package for
PMMC experiments. The payload is not distributed with PMMC. Consult the
upstream Mem-Gallery project for licensing and citation requirements before
redistributing any data, images, or source prompts.

## Expected Layout

```text
memgallery/
  data/dialogs/<scenario>.json
  images/<scenario>/...
  prompts/source/
```

Each scenario JSON contains `character_profile`, `multi_session_dialogues`, and
`human-annotated QAs`. A dialogue round has user and assistant text plus
parallel `image_id`, `input_image`, and `image_caption` arrays. A QA contains a
category in `point`, a question, an answer, source `session_id` values, source
round IDs in `clue`, and an optional `question_image`.

The adapter creates one synthetic profile session before D1. Every source round
becomes separate user and assistant Memory Units; history images and benchmark
captions are attached to the user unit. `question_image` remains a runtime query
input. Answers, clue expansion, and gold sessions are isolated in
`query_gold.jsonl`.

## PMMC Mapping

| Source category | Canonical type | Default evaluation |
| --- | --- | --- |
| `FR` | `textual_fact` | included |
| `VS` | `visual_search` | included |
| `TTL` | `test_time_learning` | included |
| `TR` | `temporal` | included |
| `VR` | `visual_reasoning` | included |
| `MR` | `multi_entity_reasoning` | included |
| `KR` | `knowledge_update` | included |
| `AR` | `answer_refusal` | excluded |
| `CD` | `conflict_detection` | excluded |

## Expected Counts

```text
scenario JSON files: 20
queries: 1,711 (1,446 included, 265 excluded)
canonical sessions: 260 (including 20 synthetic profiles)
memory units: 7,944
image assets: 1,490
```

Validate the local package with:

```bash
pmmc validate-data --dataset memgallery --level fast
pmmc validate-data --dataset memgallery --level full
```
