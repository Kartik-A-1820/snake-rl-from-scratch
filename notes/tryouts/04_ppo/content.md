---
title: "04 · Policy gradients and PPO on Snake (batched environments on the GPU)"
subtitle: "From REINFORCE to PPO: the maths, the code, and why an on-policy method suits an environment that runs thousands of games at once"
---

# Summary

Tryouts 01–03 learned a *value function* $Q(s,a)$ and acted greedily on it. This tryout learns the **policy** $\pi_\theta(a\mid s)$ directly, together with a value function that is used only to reduce variance (an *actor-critic*). The algorithm is PPO (Schulman et al. 2017) with the usual implementation details: a shared CNN trunk, generalized advantage estimation, the clipped surrogate objective, value-loss clipping, an entropy bonus, advantage normalisation and orthogonal initialisation. An optional arm adds **invalid-action masking** (Huang & Ontañón 2020).

PPO is on-policy: it learns from the data its *current* policy just produced and then discards it. That needs a lot of data and little reuse, which is the opposite of DQN's replay buffer, and it is exactly what a batched GPU environment is good at.

```widget
summary
```

# 1. The policy gradient theorem

Let the return of a trajectory $\tau = (s_0, a_0, r_0, s_1, \dots)$ be $G(\tau) = \sum_t \gamma^t r_t$ and the objective $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)]$. The probability of a trajectory factorises into environment terms, which do not depend on $\theta$, and policy terms:

$$p_\theta(\tau) = p(s_0)\prod_t \pi_\theta(a_t\mid s_t)\, p(s_{t+1}\mid s_t, a_t).$$

Use the log-derivative trick $\nabla_\theta p = p\,\nabla_\theta \log p$:

$$\nabla_\theta J = \int \nabla_\theta p_\theta(\tau)\, G(\tau)\, d\tau = \mathbb{E}_\tau\bigl[G(\tau)\,\nabla_\theta \log p_\theta(\tau)\bigr] = \mathbb{E}_\tau\Bigl[G(\tau)\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\Bigr].$$

The environment's dynamics have vanished: the gradient can be estimated from sampled games, without knowing the transition model. Because an action at time $t$ cannot affect rewards before $t$, each term can use only the *reward-to-go* $G_t = \sum_{k\ge t}\gamma^{k-t} r_k$:

$$\nabla_\theta J = \mathbb{E}\Bigl[\sum_t \nabla_\theta \log\pi_\theta(a_t\mid s_t)\; G_t\Bigr]. \qquad\text{(policy gradient theorem)}$$

**REINFORCE** is the Monte-Carlo estimate of this: play episodes, then step $\theta$ along $\sum_t G_t\nabla\log\pi(a_t|s_t)$. It is unbiased and slow, because $G_t$ is a sum of many random rewards: in Snake, one lucky meal early changes the return of every later step, and the gradient is dominated by noise.

# 2. Baselines and the advantage

Subtract anything that does not depend on the action. For any function $b(s)$,

$$\mathbb{E}_{a\sim\pi}\bigl[b(s)\,\nabla_\theta\log\pi_\theta(a\mid s)\bigr] = b(s)\sum_a \nabla_\theta \pi_\theta(a\mid s) = b(s)\,\nabla_\theta \underbrace{\textstyle\sum_a \pi_\theta(a\mid s)}_{=1} = 0,$$

so replacing $G_t$ with $G_t - b(s_t)$ leaves the expected gradient unchanged while it can shrink the variance a lot. The natural baseline is the value function $V^\pi(s)$, giving the **advantage** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$: how much better this action is than the policy's average action in this state. Actions with a positive advantage get more probable, negative ones less.

# 3. Actor-critic and generalized advantage estimation

We do not know $V^\pi$, so a second head of the network, the **critic** $V_\phi(s)$, learns it by regression. Using the critic to bootstrap gives a family of advantage estimators with the TD residual $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$:

$$\hat A_t^{(1)} = \delta_t \;\;(\text{low variance, biased by critic errors}), \qquad \hat A_t^{(\infty)} = \sum_{l\ge 0}\gamma^l \delta_{t+l} = G_t - V(s_t)\;\;(\text{unbiased, high variance}).$$

**GAE** (Schulman et al. 2016) takes an exponentially weighted mixture of the $k$-step estimators, controlled by $\lambda\in[0,1]$:

$$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\,\delta_{t+l}.$$

*Derivation.* The $k$-step advantage is $\hat A_t^{(k)} = \sum_{l=0}^{k-1}\gamma^l\delta_{t+l}$ (the sum telescopes the value terms: $\hat A^{(k)}_t = -V(s_t) + r_t + \gamma r_{t+1} + \dots + \gamma^{k}V(s_{t+k})$). Weight $\hat A^{(k)}$ by $(1-\lambda)\lambda^{k-1}$:

$$(1-\lambda)\sum_{k\ge1}\lambda^{k-1}\sum_{l=0}^{k-1}\gamma^l\delta_{t+l} = (1-\lambda)\sum_{l\ge0}\gamma^l\delta_{t+l}\sum_{k\ge l+1}\lambda^{k-1} = \sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l}.$$

$\lambda = 0$ gives the one-step TD residual, $\lambda = 1$ gives the Monte-Carlo advantage. $\lambda\approx 0.95$ is the usual compromise. In code it is a backwards recursion over the rollout, $\hat A_t = \delta_t + \gamma\lambda(1-d_t)\hat A_{t+1}$, where $d_t$ marks the end of an episode.

```widget
gae_demo
```

**Timeouts are not terminals.** Snake ends an episode by starvation after $100\cdot$length moves without food. That is an artefact of the time limit, not a property of the state, so the value of the state after a timeout is *not* zero. The code folds $\gamma V(s')$ into the last reward for a truncated step and then cuts the GAE trace, which is the same truncation-versus-termination rule used for DQN in tryout 02. Only deaths and wins bootstrap from zero.

# 4. From TRPO to PPO: why clip?

A policy-gradient step is only trustworthy near the policy that collected the data. Reusing a batch for several gradient steps pushes $\pi_\theta$ away from $\pi_{\theta_{\text{old}}}$, and the estimated gradient stops being valid. The surrogate objective with an importance ratio $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$,

$$L^{\text{PG}}(\theta) = \mathbb{E}_t\bigl[r_t(\theta)\,\hat A_t\bigr],$$

is correct to first order at $\theta = \theta_{\text{old}}$ but unbounded away from it. TRPO (Schulman et al. 2015) maximises it subject to a trust-region constraint $\mathbb{E}[\mathrm{KL}(\pi_{\text{old}}\|\pi_\theta)] \le \delta$, which needs a second-order (conjugate-gradient) step. PPO gets a similar effect with a first-order objective by *clipping the ratio*:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Bigl[\min\bigl(r_t\hat A_t,\ \operatorname{clip}(r_t,\,1-\epsilon,\,1+\epsilon)\,\hat A_t\bigr)\Bigr].$$

Reading the $\min$: for $\hat A_t>0$ the objective rises with $r_t$ only until $r_t = 1+\epsilon$, after which the gradient is zero, so there is no incentive to make a good action much more likely than it was. For $\hat A_t<0$ it stops at $r_t = 1-\epsilon$. The pessimistic $\min$ keeps the *unclipped* term when it is worse, so a policy that has moved the wrong way is still pushed back.

```widget
clip_demo
```

The clip is a heuristic, not a guarantee: a single step can still overshoot $1\pm\epsilon$, and what keeps the update safe in practice is the small number of epochs and the small learning rate. The code logs the approximate KL and the fraction of clipped samples to watch this.

# 5. The full loss and the implementation details

$$L(\theta,\phi) = -L^{\text{CLIP}} + c_v\,L^{V} - c_e\,\mathcal H[\pi_\theta].$$

| Part | What it does | Setting |
|---|---|---|
| Clipped surrogate | trust-region-like step | $\epsilon = 0.2$ |
| Value loss $L^V$ | critic regression to $\hat R_t = \hat A_t + V_{\text{old}}(s_t)$, with the new value clipped to $V_{\text{old}}\pm\epsilon$ and the *larger* of the two squared errors used | $c_v = 0.5$ |
| Entropy bonus $\mathcal H$ | keeps the policy stochastic, so it keeps exploring | $c_e = 0.01$ |
| Advantage normalisation | per-minibatch zero mean, unit variance, so the policy-gradient scale does not depend on the reward scale | on |
| Orthogonal init | gain $\sqrt2$ in the trunk, $0.01$ on the policy output (the first policy is almost uniform), $1$ on the value output | on |
| Learning rate | Adam, $2.5\cdot10^{-4}$, decayed linearly to 0 | |
| Gradient clip | global norm 0.5 | |
| Rollouts | 256 envs × 64 steps = 16 384 transitions per update, 4 epochs over 8 minibatches | |

# 6. Invalid-action masking

Snake has a rule the agent can read from the board: some moves kill at once. With masking, the logits of those moves are set to $-\infty$ before the softmax, so they have probability exactly 0:

$$\pi_\theta(a\mid s) = \operatorname{softmax}\bigl(z(s) + m(s)\bigr)_a,\qquad m_a = 0 \text{ if safe},\ -\infty \text{ otherwise}.$$

Two consequences matter. The **gradient** of a masked action's logit is zero, because its probability does not depend on it: the network is never pushed to "unlearn" a move that is not in the distribution. This is the point of Huang & Ontañón's analysis, which shows that masking is a valid policy gradient on the restricted action set, whereas penalising invalid actions with a negative reward leaves the network wasting probability on them. And the **entropy** is computed over the valid moves only. The mask must be identical in the rollout (when sampling) and in the update (when recomputing $\log\pi$): the code stores it with each transition. If every move kills, the mask is lifted.

# 7. On-policy versus off-policy, and why PPO fits this machine

|  | DQN (off-policy) | PPO (on-policy) |
|---|---|---|
| Learns from | a replay buffer of old transitions from any past policy | the last rollout, from the current policy only |
| Data reuse | each transition is replayed many times | 4 epochs, then discarded |
| Sample efficiency | higher | lower |
| Needs | a Q-function; a target network; replay | many parallel environments for a decorrelated, big batch |
| Cost per env step on this machine | one gradient step per ~8 env steps, 3 forward passes each | one forward pass; the 4 epochs of updates are amortised over 16 k steps |

The batched torch environment steps hundreds of thousands of games per second on the GPU, so *data is nearly free* and the update is the cost. That reverses the usual situation, in which the simulator is the bottleneck and sample efficiency matters most. On this machine PPO's lower sample efficiency is paid for by its higher throughput: one job runs about 7 000 env steps/s against about 900 for the DQN.

# 8. Throughput lesson: how many jobs should share one small GPU?

The obvious bottleneck for several training jobs on a 4 GB GPU looks like VRAM. It is not. Each job here holds only a few hundred MB, but a GTX 1650 Ti has few multiprocessors and the CPU cores that launch its kernels are shared, so compute saturates long before memory does, and past that point extra jobs slow each other down. The sensible way to schedule them is to *measure*: run $k = 1, 2, 3, \dots$ identical jobs, record the aggregate steps/s, GPU utilisation, memory and CPU load, and stop adding jobs when the aggregate stops rising.

```widget
parallel
```

# 9. Results

```widget
ppo_table
```

```widget
ppo_curves
```

**Analysis.** Two arms, three training seeds each, 10 M environment steps per run, the same 500 held-out games as every other tryout, best checkpoint chosen on 200 separate validation games.

| Arm | Seeds (test mean) | Mean ± sd | Best game | Deaths wall / self / timeout (mean of 500) |
|---|---|---|---|---|
| PPO | 21.4, 18.7, 7.9 | 16.0 ± 7.1 | 49 | 8 / 204 / 289 |
| PPO + invalid-action masking | 26.5, 26.3, 25.7 | **26.2 ± 0.4** | 51 | 20 / 404 / 76 |

1. **Masking is the difference between an unreliable agent and a reliable one.** The masked arm lands at 26.2 in all three seeds (range 25.7–26.5); the unmasked arm ranges from 7.9 to 21.4. The unmasked runs mostly fail by *circling*: 289 of 500 games end by starvation, against 76 for the masked arm. A policy that is trying to avoid walls and its own body while its probabilities are still near uniform spends many steps in loops that cost nothing, and the entropy bonus keeps it from committing out of them. The mask removes the unsafe options outright, so the policy has fewer ways to be indecisive.
2. **Masked PPO is the best learned agent on the standard 12×12 board in this repository**: 26.2 ± 0.4 (three seeds, 10 M steps) against 24.5 for the CNN DQN (one seed, 5 M steps, best checkpoint) and the tabular agent's 20. Caveats that matter: PPO had twice the environment steps; the two used different recipes tuned by hand to a different degree; and the DQN figure is a best-over-checkpoints value from a single seed, which flatters it. The gap to DQN is small and is not a clean win for the algorithm.
3. **Throughput, not sample efficiency, is where PPO wins on this machine.** A PPO run of 10 M steps took about an hour on the shared GPU (about 2.6 k environment steps/s with other jobs running, about 7 k/s alone); the 5 M-step DQN run took 98 minutes (about 0.9 k/s). DQN needs fewer samples for a given score; PPO collects them far faster here because the batched environment makes data nearly free (§7).
4. **The remaining deaths are slow traps.** For the masked arm, 404 of 500 games end in self-collision. The mask removes instant deaths and wall hits (20 of 500), but a long snake still coils itself into a dead end. The next tryout (05) asks what it takes to avoid that.
5. **What is not claimed.** One hyperparameter set, no tuning of $\lambda$, learning rate or the entropy coefficient, a fixed horizon of 64 steps, and a policy evaluated greedily (always the most likely move). A greedy evaluation of an entropy-regularised policy can loop where a sampled one would not, which probably lowers the unmasked arm's numbers.

# 10. Failure modes to watch for

* **Entropy collapse:** the policy becomes deterministic too early and stops exploring (entropy → 0, return flat).
* **Value-loss domination:** a large critic loss with a shared trunk can swamp the policy gradient; watch the value loss and clip fraction.
* **Too many epochs or too high a learning rate:** approximate KL jumps and the clip fraction climbs; performance can collapse.
* **Mask mismatch between rollout and update** silently breaks the importance ratio.
* **Treating a timeout as a terminal**, which teaches the critic that waiting is "free of future".

# Self-check

```quiz
Q: Why is subtracting a state-dependent baseline $b(s)$ from the return safe for the policy gradient?
- [ ] Because $b$ is small
- [x] $\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi(a|s)] = \nabla_\theta\sum_a\pi(a|s) = \nabla_\theta 1 = 0$, so any $b(s)$ adds zero in expectation
- [ ] Because the baseline is learned
Why: The baseline only changes the variance of the estimate, never its mean, as long as it does not depend on the action.
```

```quiz
Q: In PPO's clipped objective with $\hat A_t > 0$ and $r_t > 1 + \epsilon$, what is the gradient with respect to the policy?
- [ ] positive, so the action keeps becoming more likely
- [x] zero, the clipped term is the minimum and it does not depend on $\theta$
- [ ] negative
Why: $\min(r\hat A, (1+\epsilon)\hat A) = (1+\epsilon)\hat A$ when $r>1+\epsilon$, a constant. The step has already used up its allowance for this sample.
```

# References

- Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. *ICML*.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8.
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS*.
- Huang, S., & Ontañón, S. (2020). A closer look at invalid action masking in policy gradient algorithms. *arXiv:2006.14171*.
- Engstrom, L., et al. (2020). Implementation matters in deep policy gradients: a case study on PPO and TRPO. *ICLR*.
- Andrychowicz, M., et al. (2021). What matters in on-policy reinforcement learning? A large-scale empirical study. *ICLR*.
