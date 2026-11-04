# The twelve questions

The test is not "can you define these terms". It is "can you answer in one sentence, out loud, to someone who will interrupt you".

Answer each one in your own words before you start the next stage. If an answer takes more than a sentence, it is not yet understood.

---

**1. Why does a 100%-pass-rate environment teach a model nothing?**

<details><summary>Check</summary>

All rewards in the group are 1, so `std(r) = 0`, so every advantage is 0, so the gradient is exactly zero. It is not a weak signal, it is no signal — and it costs the same to run as a useful one.
</details>

---

**2. How is the advantage computed in GRPO, and what happens if all 8 rollouts are correct?**

<details><summary>Check</summary>

`A_i = (r_i - mean) / (std + eps)`. If all 8 are correct, `mean = 1`, `std = 0`, and every `A_i ≈ 0`. The sample is discarded.
</details>

---

**3. Why use a group baseline instead of a critic network?**

<details><summary>Check</summary>

A critic costs another network of the same size and has to be trained; the group mean is a free, unbiased-enough baseline that needs only the reward you already have. That is what makes GRPO practical when the reward is a single test outcome.
</details>

---

**4. What is the KL penalty protecting against?**

<details><summary>Check</summary>

Drift. Without it the policy can wander far from the reference model into a region that scores well on the verifier but is no longer coherent — the classic reward-hacking attractor.
</details>

---

**5. What are the three most common reward hacks in a code/UI environment?**

<details><summary>Check</summary>

(a) **Answer leakage** — the expected value is visible in the prompt, the files, or the test names. (b) **Checker shortcut** — satisfying the assertion without the state change (returning early, monkey-patching, editing the test). (c) **Unenforced state** — the verifier reads a field the environment never actually enforces, so the agent writes it directly.
</details>

---

**6. Why does a nondeterministic verifier damage the gradient, not just the score?**

<details><summary>Check</summary>

Identical trajectories get different rewards, so the spread inside a group is partly noise. The advantage then rewards luck, and the policy is trained toward whatever the noise favours. It is worse than a broken verifier that fails consistently, because that one at least produces a stable (wrong) target.
</details>

---

**7. What does a large gap between pass@1 and pass@8 tell you?**

<details><summary>Check</summary>

The model can reach the answer but does not reliably. That is the regime where RL helps most — there is reachable signal and lots of variance. A high pass@8 with near-zero pass@1 is a good place to train; both high is a saturated environment.
</details>

---

**8. During training the average response length keeps growing. Good or bad?**

<details><summary>Check</summary>

Neither by itself. It is a known side effect of RLVR and often correlates with better reasoning, but if length grows while accuracy is flat, the model has learned that length earns reward and you have a length-bias problem.
</details>

---

**9. Entropy drops to almost zero. What has happened?**

<details><summary>Check</summary>

The policy has collapsed to a near-deterministic output. Exploration is gone, so every group will produce near-identical rollouts and `std(r)` falls toward zero. Training has effectively stopped; it usually means the task was too easy or the learning rate too high.
</details>

---

**10. How do you define and measure "difficulty" for an environment?**

<details><summary>Check</summary>

Pass rate at the rollout temperature, measured over enough rollouts to bound the noise. Since it is a binomial, the standard error at n rollouts is `sqrt(p(1-p)/n)`. Difficulty is measurable *before* training, which is the point: you can reject an environment without spending GPU hours.
</details>

---

**11. Why sample K rollouts per prompt rather than one?**

<details><summary>Check</summary>

One sample gives a reward but no baseline — you cannot tell whether it was good or lucky. K samples give you the mean, which is the baseline, and the spread, which is the signal. K also multiplies the environment cost, which is why reset latency is a budget line.
</details>

---

**12. A lab says "we trained on your environment and the eval did not move". Three likely causes.**

<details><summary>Check</summary>

(a) **No signal** — pass rate saturated at one end, so the gradient was near zero. (b) **Signal from the wrong thing** — reward hacking, so the model learned the clone's quirks rather than the task. (c) **Signal that does not transfer** — the clone's behaviour diverges from the real product, so the skill learned is clone-specific. Diagnose in that order: it is cheapest to check the pass-rate distribution first.
</details>
