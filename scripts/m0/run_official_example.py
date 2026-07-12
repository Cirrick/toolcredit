"""M0: faithful port of verl v0.8.0 official agent-loop tutorial.

Source: third_party/verl/examples/tutorial/agent_loop_get_started/agent_loop_tutorial.ipynb
(ReAct agent + code sandbox + MATH + GRPO, 5 training steps).

Deviations from the notebook (single-GPU GH200, no wandb yet), each flagged inline:
  [DEV-1] trainer.n_gpus_per_node: 8 -> 1
  [DEV-2] trainer.logger: drop wandb (console + tensorboard only)
  [DEV-3] rollout_name pinned to "sglang" (notebook leaves it as "???")
  [DEV-4] attn_implementation=sdpa — no flash-attn aarch64 wheel; verl defaults to
          flash_attention_2 (verl/workers/config/model.py:185) which crashes without it
Everything else (model, data, batch sizes, 5 steps) is unchanged.
"""

import argparse
import asyncio
import json
import os
import socket
import sys
import tempfile

import fastapi
import ray
import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse

import verl

ROLLOUT_NAME = "sglang"  # [DEV-3]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- notebook cell 18: sandbox ray actor (verbatim) ---
@ray.remote(num_cpus=1)
class Sandbox:
    """Sandbox to execute python code."""

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = self._get_free_port()
        asyncio.create_task(self._start_fastapi_server())

    async def code_execution(self, request: Request):
        request_json = await request.json()
        code = request_json["code"]

        _, temp_file = tempfile.mkstemp(suffix=".py", prefix="temp_code", dir=None, text=True)
        with open(temp_file, "w") as f:
            f.write(code)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            response = {
                "status": "Success" if process.returncode == 0 else "Failed",
                "run_result": {
                    "status": "Finished",
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "return_code": process.returncode,
                },
            }
            return JSONResponse(content=response)
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def _get_free_port(self):
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    async def _start_fastapi_server(self):
        app = fastapi.FastAPI()
        app.router.add_api_route("/run_code", self.code_execution, methods=["POST"])
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> str:
        """Get FastAPI server address."""
        return f"{self.address}:{self.port}"


def download_assets() -> tuple[str, str, str]:
    # --- notebook cell 4: model + dataset download (verbatim paths) ---
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="verl-team/lighteval-MATH-preprocessed",
        repo_type="dataset",
        local_dir=os.path.expanduser("~/verl-team/lighteval-MATH-preprocessed"),
    )
    snapshot_download(
        repo_id="Qwen/Qwen3-1.7B",
        repo_type="model",
        local_dir=os.path.expanduser("~/Qwen/Qwen3-1.7B"),
    )

    model_path = os.path.expanduser("~/Qwen/Qwen3-1.7B")
    train_file = os.path.expanduser("~/verl-team/lighteval-MATH-preprocessed/train.parquet")
    test_file_full = os.path.expanduser("~/verl-team/lighteval-MATH-preprocessed/test.parquet")

    test = pq.read_table(test_file_full)
    test_file = os.path.expanduser("~/verl-team/lighteval-MATH-preprocessed/test_100.parquet")
    pq.write_table(test[:100], test_file)
    return model_path, train_file, test_file


def make_smoke_data(train_file: str, test_file: str) -> tuple[str, str]:
    """Slice 20 train / 8 val rows for the <10min smoke test (PLAN M0)."""
    import pyarrow.parquet as pq

    smoke_dir = os.path.join(SCRIPT_DIR, "smoke_data")
    os.makedirs(smoke_dir, exist_ok=True)
    train_smoke = os.path.join(smoke_dir, "train_20.parquet")
    test_smoke = os.path.join(smoke_dir, "test_8.parquet")
    pq.write_table(pq.read_table(train_file)[:20], train_smoke)
    pq.write_table(pq.read_table(test_file)[:8], test_smoke)
    return train_smoke, test_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="20 samples, 1 gradient step, no val")
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)  # so class_name "sandbox.SandboxTool" and tool_config.json resolve
    ray.init(runtime_env={"env_vars": {"PYTHONPATH": SCRIPT_DIR}, "working_dir": None})

    model_path, train_file, test_file = download_assets()
    if args.smoke:
        train_file, test_file = make_smoke_data(train_file, test_file)

    # --- notebook cell 33: sandbox + tool config ---
    sandbox = Sandbox.remote()
    sandbox_address = ray.get(sandbox.get_server_address.remote())
    tool_config = {
        "tools": [
            {
                "class_name": "sandbox.SandboxTool",
                "config": {"type": "native", "sandbox_fusion_url": f"http://{sandbox_address}/run_code"},
            },
        ],
    }
    tool_config_path = os.path.join(SCRIPT_DIR, "tool_config.json")
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)

    # --- notebook cell 34: hydra config ---
    from hydra import compose, initialize_config_dir

    smoke_overrides = (
        [
            "data.train_batch_size=16",
            "actor_rollout_ref.rollout.n=4",
            "trainer.val_before_train=False",
            "trainer.total_training_steps=1",
            "trainer.experiment_name=smoke_test",
        ]
        if args.smoke
        else []
    )

    verl_config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer/config")
    with initialize_config_dir(config_dir=verl_config_dir, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adv_estimator=grpo",
                "data.train_files=" + train_file,
                "data.val_files=" + test_file,
                "data.return_raw_chat=True",
                "data.train_batch_size=32",
                "data.max_prompt_length=1024",
                "data.max_response_length=1024",
                "+data.apply_chat_template_kwargs.enable_thinking=False",
                "actor_rollout_ref.model.path=" + model_path,
                "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",  # [DEV-4]
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                "actor_rollout_ref.rollout.name=" + ROLLOUT_NAME,
                "actor_rollout_ref.rollout.mode=async",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                "actor_rollout_ref.rollout.n=8",
                "actor_rollout_ref.rollout.multi_turn.tool_config_path=" + tool_config_path,
                "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
                "trainer.val_before_train=True",
                "trainer.log_val_generations=10",
                "trainer.n_gpus_per_node=1",  # [DEV-1]
                "trainer.test_freq=-1",
                "trainer.total_training_steps=5",
                "trainer.logger=['console','tensorboard']",  # [DEV-2]
                "trainer.project_name=toolcredit-m0",
                "trainer.experiment_name=" + os.path.basename(model_path),
            ]
            + smoke_overrides,
        )

    # --- notebook cell 35 ---
    from verl.trainer.main_ppo import main as ppo_main

    ppo_main(config)


if __name__ == "__main__":
    main()
