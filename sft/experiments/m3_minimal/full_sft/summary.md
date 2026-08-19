# Full-parameter SFT control

- base: `/home/jovyan/Qwen/Qwen3-1.7B`
- data: `/home/jovyan/toolcredit/sft/data/sft_traces.jsonl` (6056 examples)
- epochs / effective batch / LR: 2.0 / 32 / 1e-05
- loss: 0.262 -> 0.1906
- runtime: 669.1s
- checkpoint: `/home/jovyan/toolcredit/sft/experiments/m3_minimal/full_sft/checkpoint`

## Held-out comparison

Same 200 held-out questions and generation protocol as the accepted LoRA checkpoint
(temperature 0.6, 4 samples/question, bare TIR):

| | LoRA SFT | Full SFT | Full − LoRA |
|---|---:|---:|---:|
| TIR pass@1 | 0.5100 | 0.5188 | +0.0088 |
| CoT pass@1 | 0.5725 | 0.5900 | +0.0175 |
| Tool error rate | 0.2013 | 0.2654 | +0.0641 |
| Tool abandon rate | 0.1050 | 0.1150 | +0.0100 |

Decision: full SFT is not worth switching to. Its small pass@1 changes do not offset
the worse tool reliability. Keep `sft/checkpoints/qwen3-1.7b-sft`.

Raw predictions and scored metrics: `data/probe_heldout_full_sft/`.
