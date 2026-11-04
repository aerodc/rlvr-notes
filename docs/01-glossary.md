# Glossary

One line each, in the order you will hit them.

## The RL side

| Term | Meaning |
|---|---|
| **RLVR** | Reinforcement learning with *verifiable* rewards. The reward comes from a program, not a human or a learned judge. If a judge decides, it is not RLVR. |
| **Policy** | The model being trained, viewed as a distribution over next tokens. |
| **Rollout** | One sampled trajectory: prompt in, completion (and any tool calls) out. |
| **Trajectory** | Same as rollout, usually implying the multi-step form: a sequence of observations and actions. |
| **Group** | The K rollouts sampled from the *same* prompt, used to compute relative advantage. |
| **Advantage** | `(r_i - mean(r)) / std(r)`. How much better than its siblings this rollout was. |
| **GRPO** | Group Relative Policy Optimization. Drops the value network; uses the group mean as the baseline. The default for RLVR. |
| **PPO** | The older algorithm GRPO simplified. Learn that clipping limits step size; the rest you can skip. |
| **KL penalty** | A pull toward the reference model, so the policy does not drift into nonsense while chasing reward. |
| **Entropy** | How spread out the policy's distribution is. Falling to ~0 means it stopped exploring, usually dead. |
| **Temperature** | Sampling sharpness. RLVR needs `T > 0` — greedy decoding gives K identical rollouts and zero variance. |
| **pass@k** | Probability that at least one of k samples is correct. For training, what matters is pass rate at the rollout temperature. |
| **Reward hacking** | The policy finds a way to satisfy the checker without doing the task. |
| **Length bias** | Longer (or shorter) answers get rewarded incidentally, and the model learns the length, not the skill. |
| **Mode collapse** | The policy converges to one output pattern and stops producing diversity. |

## The environment side

| Term | Meaning |
|---|---|
| **Environment** | The thing that gives the agent a world to act in and a score at the end. |
| **Reset** | Return the world to a known initial state. Must be cheap and exact; it runs thousands of times per step. |
| **Step** | Apply one action, return the new observation. |
| **Observation** | What the agent sees after a step (or at reset). |
| **Action** | What the agent does. For UI agents this is usually a structured click/type command, not raw pixels. |
| **Sandbox** | The isolation boundary: container, VM, or process. Prevents one rollout from affecting another. |
| **Task spec** | The machine-readable definition of a task: initial state, allowed actions, success conditions. |
| **Deterministic verifier** | Same trajectory in, same score out, always. Non-negotiable. |
| **Grader** | The thing that returns a score. A verifier is one kind of grader; an LLM judge is another. |
| **Evidence** | Per-check output explaining *why* the score is what it is. The difference between a debuggable verifier and a black box. |
| **Difficulty calibration** | Choosing task difficulty so pass rate lands in a learnable band (roughly 20–80%). |
| **Contamination** | The model has already seen the task or its answer during pretraining. Makes the pass rate meaningless. |
| **Fidelity** (for cloned software) | How closely a simulated app matches the real one's behaviour, not just its pixels. |
| **Transfer gap** | The difference between "solved it in the clone" and "solved it in the real product". The central risk of clone-based environments. |
