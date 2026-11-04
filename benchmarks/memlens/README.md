# MEMLENS 32K Local Assets

This directory contains the locally retained 32K-context portion of MEMLENS for
PMMC experiments. The payload is not distributed with PMMC.

Original sources:

- Paper: <https://arxiv.org/abs/2605.14906>
- Code: <https://github.com/xrenaf/MEMLENS>
- Dataset: <https://huggingface.co/datasets/xiyuRenBill/MEMLENS>

Consult the upstream project for licensing and citation requirements before
redistributing any payload.

## Expected Layout

```text
memlens/
  data/dataset_32k.json
  images/
    haystack_images/
    needle_images/
  prompts/judge/
```

`dataset_32k.json` is a JSON array. Each item includes `question_id`,
`question_type`, `question_subtype`, `question`, `answer`, `question_date`,
parallel `haystack_dates`, `haystack_session_ids`, `haystack_sessions`, and
evaluation-only `answer_session_ids`. Image paths in turns are relative to
`images/`; adapters prefer local files while retaining the source URL and BLIP
caption as metadata.

PMMC preserves source session and turn order. Session IDs are namespaced by
question because source session IDs repeat across records. Five source turns
with no text or image are retained as explicit `[EMPTY SOURCE TURN]` records so
canonical indices remain aligned with the source.

## PMMC Mapping

| Source category | Canonical type | Default evaluation |
| --- | --- | --- |
| `information_extraction` | `textual_fact` | included |
| `multi_session_reasoning` | `multi_hop` | included |
| `temporal_reasoning` | `temporal` | included |
| `knowledge_update` | `knowledge_update` | included |
| `answer_refusal` | `answer_refusal` | excluded |

Gold answers and `answer_session_ids` are written only to `query_gold.jsonl`,
never to runtime query records.

## Expected Counts

```text
queries: 789 (699 included, 90 excluded)
sessions: 10,616
memory units: 97,446
image files: 4,695
unique referenced image paths: 4,090
```

Validate the local package with:

```bash
pmmc validate-data --dataset memlens --level fast
pmmc validate-data --dataset memlens --level full
```
