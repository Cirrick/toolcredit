# M7 final analysis and documentation handoff

The original `summary.md` is the immutable generation/scoring handoff covered by canonical
`hashes.sha256`; it is intentionally not rewritten. This additive summary records the completed downstream analysis and
documentation synthesis. The canonical generation, scored trajectories, evaluator, protocol, checkpoints, and frozen hash
ledger remain unchanged.

## Integrity and analysis gates

- Raw and strict-scored trajectories: **15,200/15,200** each; greedy 3,040 and sampled 12,160.
- 32 exact shards, 3,800 four-role pairing groups, 15,200 unique trajectory UIDs.
- Missing, unexpected, duplicate, conflicting, and infrastructure-failure keys: **0**.
- Full exact-key digest: `c43ce71901f61c24ca0510b1af6fd70a149d4b4692793cff0326c57285efa8b8`.
- Full taxonomy: 10,085 correct and 5,115 wrong; every wrong row has one frozen coarse proposal and every correct row has
  null primary; unclassified count 0.
- Blind audit: 192 stable-hash checkpoint-blinded cases; manual Codex reviewer, not independent human audit; agreement
  151/192 and uncertain 48/192. E5 agreement was only 6/24 greedy and 3/24 sampled, so E5 coarse taxonomy migration is
  not scientifically interpreted.
- Paired rows: 11,400 = three comparisons × (760 greedy + 3,040 sampled); all 30 source/setting/full transition matrices
  conserve their denominators.
- Bootstrap: 10,000 replicates, seed 42, two-sided percentile 95% CI, source-stratified question-cluster resampling;
  sampled indices remain clustered with their question. Accuracy reports have 10,000 effective replicates; conditional
  statistics retain genuine zero-denominator replicates without pseudocounts.
- The 17 frozen downstream evidence artifacts are covered by `analysis_hashes.sha256`.

## Quantitative conclusion

Greedy FULL760 pass@1 is raw 426/760 = .5605, SFT 460/760 = .6053, E3 557/760 = .7329, and E5
549/760 = .7224. Preregistered paired accuracy deltas are:

| Transition | Greedy delta [95% CI] | Sampled matched-index delta [95% CI] |
|---|---:|---:|
| Raw→SFT | +.0447 [.0105, .0789] | +.0474 [.0276, .0671] |
| SFT→E3 | +.1276 [.0974, .1592] | +.1286 [.1089, .1484] |
| E3→E5 | −.0105 [−.0368, .0158] | −.0043 [−.0178, .0092] |

E3 is the strongest endpoint. E5 has no final accuracy benefit, but it has a severe behavioral treatment effect: E3→E5
mean calls are .888→3.755 greedy and .958→3.935 sampled; four-call rates are 1.3%→92.9% and 1.2%→97.2%.
The increase appears in every source. Per-call success also rises to approximately 98%, so the dominant failure is repeated,
successful over-calling rather than tool syntax collapse.

M7 validates M6's over-calling/no-endpoint-gain finding across the full source panel. It cannot validate M6's
faster-early-learning finding because M7 is endpoint-only. E5 is described as credit-signal/proxy gaming, not automatically
classic reward hacking; E4 provides the cleaner direct reward-shaping exploitation cases.

## Documentation outputs

- `reports/02_main_results.md`: M6 preserved; M7 four-role/source/setting performance, behavior, paired matrices, and CIs.
- `reports/03_badcase_taxonomy.md`: full taxonomy workflow, blind audit, transition matrices, and stable-hash paired examples.
- `reports/04_reward_hacking.md`: three closed-loop E4 cases and the E4/E5 terminology boundary.
- `reports/interview_outline.md`: 60-second and five-minute evidence-linked narratives.
- `README.md`, `reports/technical_report.md`, `plans/M7.md`, `LOG.md`, and `HANDOFF.md`: project synthesis and review state.

No new experiment, generation, scoring, training, E7, extra seed, beta sweep, Tier B, or post-hoc tuning was run. The user
approved the final documentation and acceptance artifacts on 2026-08-25; M7 acceptance criteria 1–13 are complete, and the
completion commit/tag `m7` were authorized. Git object identity is recorded by the repository rather than self-referenced here.
