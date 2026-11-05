# rlvr-notes

Notes and small runnable experiments on how **RLVR environments** work — from the
side that *builds* them, not the side that trains on them.

> **Companion repo.** The hands-on half lives in TypeScript at
> [`aerodc/rl-env-lab`](https://github.com/aerodc/rl-env-lab): a mail-app clone, a
> gym-style `reset / step / grade` environment, graders with oracles, and a
> parallel rollout harness. **That** is the platform; **this** is the theory and
> the small Python illustrations explaining why the platform is shaped that way.
>
> If you read one file here, read [`s3_difficulty`](s3_difficulty): it derives the
> 20–80% band from the binomial, which turns calibration from folklore into
> arithmetic.

Most RL material is written for people who want to train models. This is written for people who want to build the **environments models are trained in**: task specs, sandboxes, deterministic verifiers, difficulty calibration, and the cost and latency profile of running thousands of rollouts.

You do not need a GPU to start. Stage 0 runs in two seconds on a laptop and teaches the one thing everything else depends on.

---

## The whole thing in two questions

> **1. Where does the reward for a trajectory come from?**
> **2. Within a group of K trajectories, is the spread in reward large enough to learn from?**

Everything else — PPO vs GRPO, KL coefficients, clip ratios — is the trainer's problem. These two are the environment's problem.

The second question is the one people underestimate. In GRPO the advantage of a sample is computed *relative to its group*:

```
A_i = (r_i - mean(r)) / (std(r) + eps)
```

If every rollout in the group gets the same reward — all correct, or all wrong — then `std(r) = 0` and `A_i ≈ 0`. The sample contributes **nothing** to the gradient. A 100%-pass-rate environment is not "easy", it is *worthless*. This is why difficulty calibration is not a quality tweak; it is the product.

---

## Stages

Each stage produces a **number**, not just code. The numbers go in [`measures/RESULTS.md`](measures/RESULTS.md).

| Stage | What you build | Runs on | What you can say afterwards |
|---|---|---|---|
| **S0** [`s0_baseline`](s0_baseline) | A 120-line GRPO in pure Python, no deps | CPU, ~2s | Why a flat reward distribution produces zero learning |
| **S1** [`s1_verifier`](s1_verifier) | A deterministic verifier with per-check evidence | CPU, ~1s | Why `evidence` is not optional in a verifier |
| **S2** [`s2_env_service`](s2_env_service) | An environment served over HTTP, with reset/step/score | CPU | Why the environment should be a service, not a library |
| **S3** [`s3_difficulty`](s3_difficulty) | A calibration sweep over pass rate | CPU, ~5s | Where the learnable band is, and why 20–80% |
| **S4** [`s4_failure`](s4_failure) | A verifier with a deliberate hole, and a policy that finds it | CPU | The three ways code environments get reward-hacked |

Optional, GPU: [`s0_baseline/train_grpo.py`](s0_baseline/train_grpo.py) runs the same loop against a real small model with TRL. Do it after S3, not before — you will understand the curves much better once you have seen the toy version.

---

## Quickstart

```bash
git clone git@github.com:aerodc/rlvr-notes.git && cd rlvr-notes
make toy        # S0: watch a policy learn, then watch it fail to learn
make test       # S1: verifier tests, including a determinism check
make env        # S2: start the env service on :8787
make sweep      # S3: the calibration sweep
```

No dependencies for any of the above. Python 3.11+ and Node 20+.

---

## Reading order

1. [`docs/00-mental-model.md`](docs/00-mental-model.md) — 5 minutes, the frame
2. [`docs/01-glossary.md`](docs/01-glossary.md) — 25 terms, one line each
3. [`docs/03-grpo-in-60-lines.md`](docs/03-grpo-in-60-lines.md) — the math, alongside the code
4. [`docs/12-questions.md`](docs/12-questions.md) — the self-test. If you can answer these, you can hold a technical conversation with a lab.

---

## What this is not

- Not a benchmark. Nothing here is competitive with real environments.
- Not a training framework. `train_grpo.py` is a thin wrapper over TRL.
- Not a claim that toy results transfer. The point is calibration of intuition, not of models.

## License

MIT
