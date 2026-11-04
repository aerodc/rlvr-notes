# S0 — See the mechanism

One evening, no GPU, no dependencies.

```bash
make toy          # a learnable task: the policy climbs, then saturates
make toy-flat     # every reward is 1.0: nothing happens, and nothing can
make toy-noisy    # the verifier lies 20% of the time: the signal never dies
make toy-hard     # 0.1% correct: the signal never arrives
```

## What to look for

Run them in that order and read the `|advantage|` track, not the pass rate. The pass rate is what the verifier says; the advantage is what the model actually learns from. Those are the same thing only when the verifier is honest and the task is calibrated.

| Task | Steps with any signal | What it teaches |
|---|---|---|
| `learnable` | ~24% | Signal exists while the task is unsolved; it disappears when solved |
| `flat` | 0% | A saturated environment and a broken one are indistinguishable from inside |
| `hard` | ~2% | "Too hard" fails identically to "too easy", at the same cost |
| `noisy` | ~90% | A flaky verifier produces permanent advantage that is pure noise |

The `noisy` case is the one to sit with. The model *does* learn the task — true accuracy goes up — but the verifier keeps reporting something else, and the advantage never decays the way it does in `learnable`. That permanent advantage is the flaky verifier being optimised.

## The three lines that matter

In `grpo_step()`:

```python
mean_r = sum(rewards) / group_size
std_r  = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / group_size)
advantages = [(r - mean_r) / (std_r + eps) for r in rewards]
```

If `std_r == 0`, every advantage is 0, and the update is exactly zero. Not small — zero.

## Then, optionally, on a GPU

`train_grpo.py` runs the same loop against a real model with TRL. It is deliberately thin: the point is to see the same four failure modes on real loss curves, not to learn TRL.

```bash
uv sync --extra train
python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 100
```

Watch the logged `reward` and `reward_std` per group. When `reward_std` collapses, training has stopped, whatever the loss says.
