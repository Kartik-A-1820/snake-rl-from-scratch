---
title: "03 · Upgrading the DQN: Double, dueling, prioritized replay, action masking"
subtitle: "Four ideas, the maths behind each, and an ablation that measures what each one buys on Snake"
---

# Summary

Tryout 02's CNN agent plateaued at a test mean of about 21 (best checkpoint 24.5) on 12×12. About 88% of its deaths were *self-collisions*: the snake coils itself into a dead end. This tryout adds, one at a time and then all together, four well-known upgrades to the same agent and measures each on the same 500 held-out seeds, with several training seeds per arm:

| Idea | One line | Paper |
|---|---|---|
| Double DQN | choose the next action with the online net, score it with the target net | van Hasselt et al. 2016 |
| Dueling head | $Q = V + (A - \bar A)$: learn "how good is this state" separately from "which action is better" | Wang et al. 2016 |
| Prioritized replay | replay surprising transitions more often, correct the bias with importance weights | Schaul et al. 2016 |
| Action masking | never choose (or bootstrap through) a move that kills the snake on the next tick | Huang & Ontañón 2020 |

The baseline already has Double DQN and 3-step returns on (§8b of tryout 02 explains why), so "+Double" is the baseline itself.

```widget
summary
```

# 1. What limits the tryout-02 agent?

Look at how it dies. In the best checkpoint 443 of 500 games end by self-collision, 39 by hitting a wall, 18 by starvation. Two quite different failure types hide under "self":

1. **Instant mistakes**: the agent turns into its own body when a safe move existed. A 1-step lookahead would have prevented it.
2. **Traps**: every move is safe right now, but the snake has already coiled itself into a region it cannot leave. Only planning many steps ahead prevents this.

Action masking can only fix type 1. The other three upgrades improve how well the value function is learned, which is what type 2 needs. Measuring the two types apart is part of the analysis in §8.

# 2. Double DQN: the max operator is biased upward

Q-learning's target is $y = r + \gamma \max_{a'} Q(s', a')$. Suppose the estimates are $Q(s',a') = q^*(a') + \epsilon_{a'}$ with zero-mean noise $\epsilon$. Then

$$\mathbb{E}\Bigl[\max_{a'}\bigl(q^*(a') + \epsilon_{a'}\bigr)\Bigr] \;\ge\; \max_{a'} \mathbb{E}\bigl[q^*(a') + \epsilon_{a'}\bigr] = \max_{a'} q^*(a')$$

by Jensen's inequality ($\max$ is convex). The *same* noisy estimate is used to pick the action and to score it, so a lucky overestimate is selected precisely because it is lucky. The bias is positive, and bootstrapping feeds it back into the next target.

Double DQN separates selection from evaluation:

$$y = r + \gamma\, Q_{\theta^-}\!\bigl(s',\ \arg\max_{a'} Q_\theta(s', a')\bigr).$$

If the two networks' errors are independent, the selected action's score has zero-mean error again. The cost is one extra forward pass. With only 3 actions the bias is small (the expected maximum of $m$ i.i.d. noise terms grows slowly with $m$), which is why this tryout does not expect a large gain from it on Snake.

# 3. Dueling: $Q = V + A$ is not identifiable

Write $Q(s,a) = V(s) + A(s,a)$ with the advantage $A(s,a) = Q(s,a) - V(s)$. A network that outputs a scalar $V$ and a vector $A$ and simply adds them has a degeneracy: $(V + c,\ A - c)$ gives the same $Q$ for any constant $c(s)$, so $V$ and $A$ are not determined by $Q$. The fix (Wang et al.) forces a convention:

$$Q(s,a) = V(s) + \Bigl(A(s,a) - \tfrac{1}{|\mathcal A|}\textstyle\sum_{a'} A(s,a')\Bigr).$$

The centred advantages sum to zero, so $V(s) = \tfrac{1}{|\mathcal A|}\sum_a Q(s,a)$ is now the *mean* of the action values and the split is unique. (The paper's first form subtracts $\max_a A$ instead, which makes $V = \max_a Q$; the mean version is smoother in practice.)

Why it can help: in most Snake positions the three moves have almost the same value (nothing is near the head), and what matters is the *state*: is the snake in an open area or squeezed? A plain Q-head has to learn this separately for each of the three outputs, and each update only touches the output of the action taken. In the dueling head every update moves $V$, shared by all three actions. The price is a bigger head: here 582 k parameters against 320 k, because the value and advantage streams each get their own 256-unit dense layer.

# 4. Prioritized experience replay

Uniform replay wastes updates on transitions the network already predicts well. Prioritized replay samples transition $i$ with probability

$$P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}, \qquad p_i = |\delta_i| + \varepsilon,$$

where $\delta_i$ is its latest TD error (here the 3-step TD error) and $\alpha \in [0,1]$ interpolates between uniform ($\alpha = 0$) and greedy-by-error ($\alpha = 1$). New transitions get the current maximum priority so each is replayed at least once.

**The bias and its correction.** Sampling non-uniformly changes the distribution the gradient is averaged over, so the update no longer follows the gradient of the intended loss. Importance sampling restores it. With weights

$$w_i = \Bigl(\tfrac{1}{N \cdot P(i)}\Bigr)^{\beta}, \qquad \text{normalised by } \max_j w_j \text{ for stability},$$

the expected update is $\sum_i P(i)\, w_i\, g_i \propto \sum_i P(i)\,\frac{1}{(N P(i))^{\beta}}\, g_i$, which for $\beta = 1$ equals $\frac{1}{N}\sum_i g_i$, the uniform-replay gradient. Because the correction matters most near convergence, $\beta$ is annealed from $\beta_0 = 0.4$ to $1$ over training; with $\beta < 1$ it is a partial correction that trades some bias for lower variance.

**On a GPU.** The original uses a sum-tree for $O(\log N)$ sampling. Here the 200 k priorities live in a tensor and `torch.multinomial` samples a whole batch at once. A sampling step costs about 1.2 ms against 0.3 ms for uniform, which is 2 ms on a ~10 ms update (measured in `python/train_dqn.py`, §7).

# 5. Action masking

Let $\mathcal A_{\text{safe}}(s) \subseteq \mathcal A$ be the moves that do not hit a wall or body segment on the next tick (the engine already computes this: it is the "danger" part of the 11 features). Masking restricts the policy to $\mathcal A_{\text{safe}}(s)$ wherever it is non-empty:

* **Acting:** greedy picks $\arg\max_{a \in \mathcal A_{\text{safe}}} Q$; exploration draws uniformly from $\mathcal A_{\text{safe}}$.
* **Targets:** the bootstrap uses $\max_{a' \in \mathcal A_{\text{safe}}(s')} Q(s',a')$ (and the same restriction for Double DQN's argmax).
* **Evaluation and the game:** the exported model carries `mask: true`, and the game applies the same rule.

**Does it change the optimal policy?** No, and the argument is short. An instant-death move yields reward $-1$ and ends the episode, so its value is exactly $-1$. Any move that survives has value at least $-\gamma$ (the worst case is dying on the *next* step, which is discounted), and $-\gamma > -1$. So whenever a safe move exists, the optimal policy of the *original* MDP already prefers it, and masking only removes moves the optimal policy never takes. What it changes is learning, not the optimum: the network no longer has to discover "do not turn into the body", exploration is not wasted on suicide, and Q-estimates are not dragged down by transitions the agent would never choose. (When every move kills, the mask is lifted, since the game is lost anyway.)

**Masking against learned avoidance.** A learned policy has to spend capacity and samples on a rule the engine already knows, and it can still slip up. A mask gives a guarantee for free but needs access to the simulator's rule at decision time, so the deployed agent is the network *plus* the mask. That is why the game applies the mask for models that declare it.

Two honest limits. A mask is a 1-step lookahead: it cannot prevent traps (§1, type 2). And in a task where the safety rule is not known in advance, masking is not available at all.

# 6. How each piece maps to the code

| Piece | Where | Flag |
|---|---|---|
| Double DQN target | `td_loss` in `python/train_dqn.py` | on by default, `--no-double` to turn it off |
| Dueling head | `QNet(dueling=True)` in `python/snake_rl/nets.py`; the JS op `dueling` in the game | `--dueling` |
| Prioritized replay | `Replay(per=True)`: priorities, multinomial sampling, IS weights, `update_priorities` | `--per --per-alpha 0.6 --per-beta0 0.4` |
| Action masking | `TorchSnakeVecEnv.danger()`, `safe_moves`, `NStep` carries the mask of the bootstrap state | `--mask` |

# 7. Setup

All arms: CNN on the 4×14×14 board image, 12×12, 3-step returns, batch 128 with 4 gradient steps per vector step, target sync every 500 updates, Adam 2.5e-4, ε decays over the first 40% of training, 1 M env steps, seeds 0–2, greedy evaluation on 500 held-out seeds (validation on 200 other seeds picks the checkpoint). The baseline arm is tryout 02's repaired DQN. The final arm combines everything and is then trained for 5 M steps.

# 8. Results

```widget
ablation_table
```

```widget
ablation_curves
```

```widget
long_curves
```

**Analysis.** Three training seeds per arm, 1 M steps each, everything else identical (CNN on the board image, 3-step returns, Double DQN, 32 envs, batch 128, 4 updates per step).

| Arm | Seeds (test mean) | Mean ± sd | Deaths wall / self / timeout |
|---|---|---|---|
| baseline | 16.8, 18.2, 14.3 | **16.5 ± 2.0** | 96 / 363 / 42 |
| + dueling | 11.1, 13.0, 11.3 | 11.8 ± 1.1 | 35 / 382 / 83 |
| + dueling + PER | 7.4, 7.8, 7.1 | 7.5 ± 0.4 | 44 / 302 / 154 |
| + masking only | 15.2, 14.0, 14.2 | 14.5 ± 0.7 | 9 / 336 / 155 |
| + dueling + PER + masking | 13.5, 11.2, 11.5 | 12.1 ± 1.2 | 12 / 306 / 181 |

1. **No upgrade beat the baseline, and the ordering is the same in every seed.** The ranges of the baseline (14.3–18.2) and of each other arm do not overlap except for masking alone (14.0–15.2), which sits at the bottom of the baseline's range. The honest reading is: *at these hyperparameters and this training budget, none of the four additions helps*, and dueling and PER hurt clearly.
2. **Dueling lost about a third of the score** (11.8 against 16.5). The dueling network has 582 k parameters against 320 k (two dense streams instead of one), and in a run limited to 1 M steps the extra capacity has to be trained before it pays off. The argument for dueling (a shared state value learned from every update) matters when many actions are close in value; with three actions, most of which are close in value in most positions, the saving is small and the cost is a slower head.
3. **Prioritized replay lost another third** (7.5 on top of dueling). The most likely contributors are the untuned defaults used here ($lpha = 0.6$, $eta_0 = 0.4$ annealed to 1, new transitions at maximum priority, sampling with replacement) and the 3-step TD errors, which are noisy priorities in a game where most transitions have reward 0. It is also a lot more machinery in a training loop that was only just repaired (tryout 02, §8b). This tryout did not tune any of it, so it cannot say PER *cannot* help here, only that it did not at the defaults.
4. **Masking changed *how* the agent fails, not whether it does.** Wall deaths fall from 96 to 9 of 500 (the mask removes them), but timeouts rise from 42 to 155. With no instant deaths to be punished for, a greedy policy that is unsure circles until the starvation limit, which costs nothing. Self-collisions barely move (363 → 336), which is the point of §1: they are mostly slow traps, and a one-step mask cannot see those.
5. **At 5 M steps the picture is the same.** The combined agent (DQN+) reaches a best-checkpoint test mean of **19.4** (one seed); tryout 02's plain CNN, trained with the same recipe for the same 5 M steps, reached 24.5. One seed each, so only the direction is meaningful.
6. **Why this is still a useful result.** Each upgrade is a published improvement on its own benchmark, and none of them transferred automatically to this problem. Reporting that, with the seeds, is more informative than a table where everything helps. The next experiments to run, not done here, are: tuning $lpha$ and the learning rate for PER, a wider (not deeper) plain head to separate "dueling" from "more parameters", and a longer training budget.

# 9. Interactive: priorities, sampling probabilities, importance weights

```widget
per_demo
```

# 10. Failure modes to watch for

* **PER on a tiny buffer or with $\alpha \to 1$** concentrates on a few high-error transitions (often noisy ones) and starves the rest.
* **A mask that disagrees with the engine** silently trains the wrong policy. The mask must come from the same rule as `step`; `tests/test_env.py` checks the vec env against the scalar engine.
* **Masking changes the policy the game must run.** An exported masked network played *without* the mask can walk into walls, which is why `mask` is part of the model file.
* **Comparing arms with one seed.** Seed-to-seed spread in these runs is several points; only differences that survive across seeds count.

# Self-check

```quiz
Q: In prioritized replay with $\beta = 1$, why does the expected update equal the uniform-replay gradient?
- [ ] Because the priorities are normalised
- [x] Each sample is sampled with probability $P(i)$ and weighted by $1/(N P(i))$, so $\sum_i P(i)\cdot\frac{1}{N P(i)}\,g_i = \frac{1}{N}\sum_i g_i$
- [ ] Because the TD errors average to zero
Why: Importance weights are exactly the ratio of the target distribution (uniform, $1/N$) to the sampling distribution $P(i)$.
```

```quiz
Q: Why does action masking not change the optimal policy of the Snake MDP, as long as a safe move exists?
- [ ] Because dying is rare
- [x] An instant-death move has value $-1$, any surviving move has value at least $-\gamma > -1$, so the optimum never picks the former
- [ ] Because the mask is learned
Why: The mask removes only actions that are strictly worse than some alternative, so the argmax is unchanged. It still changes how fast and how well the network learns.
```

# References

- Huang, S., & Ontañón, S. (2020). A closer look at invalid action masking in policy gradient algorithms. *arXiv:2006.14171*.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *ICLR*.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.
- Hessel, M., et al. (2018). Rainbow: combining improvements in deep reinforcement learning. *AAAI*.
