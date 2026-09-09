"""Infrastructure check: run the frozen E3 recipe for 5 steps with FSDP CPU offload off.

Background (HANDOFF "下一步" item 5, qa_log Q17): every 200-step formal run was killed by
the container's 128 GiB cgroup limit; the dominant resident cost is veRL's FSDP offload
pinned pool (~45-50 GB inside the actor worker).  This launcher reuses the frozen E3
launcher unchanged and differs from the E3 resolved config in exactly six paths:

* ``actor.fsdp_config.param_offload``, ``actor.fsdp_config.optimizer_offload``,
  ``ref.fsdp_config.param_offload`` -> ``false`` (the treatment);
* ``trainer.total_training_steps`` / ``save_freq`` / ``test_freq`` -> 5 (five formal-shape
  steps, one real checkpoint write so the checkpoint peak is covered).

Everything else (data, seeds, batch shape, reward, tool loop) is the formal E3 recipe, so
steps 1-5 are directly comparable with ``rl/runs/e3_grpo_baseline_20260819_224555``.
Nothing here touches E3 configs, the E3 launcher, or veRL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from rl.launch.e3_grpo_baseline import (
    DEFAULT_CONFIG as E3_CONFIG,
    RUN_ROOT,
    compose_config,
    launch,
    resource_preflight,
    validate_m4_config,
)
from rl.launch.e5v2_conserved_credit import diff_paths, resolved

CHECK_STEPS = 5
OFFLOAD_PATHS = {
    "actor_rollout_ref.actor.fsdp_config.param_offload",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
    "actor_rollout_ref.ref.fsdp_config.param_offload",
}
SCHEDULE_PATHS = {
    "trainer.total_training_steps",
    "trainer.save_freq",
    "trainer.test_freq",
}
ALLOWED_DIFF_PATHS = OFFLOAD_PATHS | SCHEDULE_PATHS


def compose_check_config(run_name: str) -> DictConfig:
    config = compose_config(E3_CONFIG, run_name)
    config.actor_rollout_ref.actor.fsdp_config.param_offload = False
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = False
    config.actor_rollout_ref.ref.fsdp_config.param_offload = False
    config.trainer.total_training_steps = CHECK_STEPS
    config.trainer.save_freq = CHECK_STEPS
    config.trainer.test_freq = CHECK_STEPS
    return config


def config_diff_gate(run_name: str) -> list[str]:
    """Fail closed unless the check differs from E3 in exactly the six allowed paths."""
    e3 = resolved(compose_config(E3_CONFIG, run_name))
    check = resolved(compose_check_config(run_name))
    observed = diff_paths(e3, check)
    if observed != ALLOWED_DIFF_PATHS:
        raise ValueError(
            "offload-check config diff gate failed: "
            f"unexpected={sorted(observed - ALLOWED_DIFF_PATHS)} "
            f"missing={sorted(ALLOWED_DIFF_PATHS - observed)}"
        )
    return sorted(observed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = RUN_ROOT / args.run_name
    diff = config_diff_gate(args.run_name)
    config = compose_check_config(args.run_name)
    validate_m4_config(config, run_dir)
    print(json.dumps({"config_diff_vs_e3": diff}, indent=2))
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    preflight = resource_preflight(config, smoke=False)
    preflight["config_diff_vs_e3"] = diff
    preflight["purpose"] = "fsdp-offload-off infrastructure check (HANDOFF item 5)"
    launch(config, run_dir, preflight)


if __name__ == "__main__":
    main()
