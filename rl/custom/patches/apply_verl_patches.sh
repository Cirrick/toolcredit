#!/usr/bin/env bash
# Idempotently apply toolcredit's minimal verl patches (see ../CHANGES.md).
set -euo pipefail

PY=~/.conda/envs/toolcredit/bin/python
F=$($PY -c "import verl.workers.rollout.sglang_rollout.sglang_rollout as m; print(m.__file__)")

if grep -q "toolcredit patch" "$F"; then
    echo "already applied: $F"
    exit 0
fi

sed -i 's/        except AssertionError:/        except Exception:  # toolcredit patch: sglang 0.5.8 pairs with old-name sgl_kernel/' "$F"
grep -n "toolcredit patch" "$F" && echo "applied: $F"
