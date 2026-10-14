---
title: "01 · Tabular Q-learning on Snake"
subtitle: "MDPs, Bellman equations, temporal-difference learning, and the price of a bad state"
---

# Summary

We trained a one-step Q-learning agent (Watkins, 1989; Watkins & Dayan, 1992) to play 12×12 Snake. Its state is 11 binary features: danger ahead/right/left, heading, and food direction. The Q-table has 2048 rows, but only 256 can ever occur.

Over the first 100 epochs (1 epoch = 100 episodes), the greedy score on 500 held-out seeds rose from 0.04 to 20.0. After that it plateaued at 19–22 across three training seeds. That is the same level as a ten-line hand-written heuristic (20.4). The agent never filled the board on 12×12. On 6×6 it won once in 13,500 evaluation games.

The algorithm worked. The *state* is the bottleneck: about 90% of deaths are self-collisions, and nothing in the 11 features can describe free space beyond the next cell. This note derives the theory behind every line of the training loop, shows where each assumption holds or breaks for Snake, and uses the failure to motivate tryout 02.

```widget
curves
```

# 1. Snake as a Markov decision process

A finite MDP is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$ is the set of states.
- $\mathcal{A}$ is the set of actions.
- $P(s' \mid s, a)$ gives the transition probabilities.
- $R(s, a, s')$ is the reward.
- $\gamma \in [0, 1)$ is the discount factor (Puterman, 1994; Sutton & Barto, 2018, ch. 3).

The **Markov property** requires that the next state and reward depend only on the current state and action:

$$
\Pr(S_{t+1}=s', R_{t+1}=r \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots) = \Pr(S_{t+1}=s', R_{t+1}=r \mid S_t, A_t).
$$

**The true Snake state** is the ordered body (a path of cells), the heading, the food cell, and the step counter used for timeouts. With the body as a self-avoiding path on a 12×12 grid, the number of states is astronomically large, far beyond any table. Food placement is the only source of randomness: after eating, the next food cell is drawn uniformly from the free cells by a seeded mulberry32 generator. That makes $P$ stochastic, but every episode is exactly reproducible from its seed.

**Actions** are relative: $\mathcal{A} = \{\text{straight}, \text{right}, \text{left}\}$. Relative actions remove the illegal 180° reversal by construction. They also make the policy symmetric under rotation, so "turn toward the food" means the same thing whichever way the snake faces.

**Rewards:** $+1$ for eating, $-1$ for dying, and $0$ otherwise. Rewards are deliberately sparse and sign-coded. Any shaping (§6) is an experiment, not a default.

**Episodes end in two different ways**, and the difference matters for learning:

- **Termination**: hitting a wall or the body, or filling the board. The episode ends because the MDP really ended. The value of the next state is $0$.
- **Truncation**: going $100 \times \text{length}$ steps without eating. The *environment* didn't end; we stopped watching. Treating truncation as termination teaches the agent that circling is fatal, when it is merely boring. That mis-specifies the value function (Pardo et al., 2018).

The engine reports them separately (`terminated`, `truncated`), as in Gymnasium. The update bootstraps through truncation.

```quiz
Q: A looping agent is cut off by the 100×length timeout. What target should the last transition use?
- [ ] r + 0, because the episode is over
- [x] r + γ·max_a Q(s', a), because the state was not terminal
- [ ] −1, to punish looping
Why: Truncation is a property of our training loop, not of the MDP. Using r + 0 would encode "this state has value 0", which is false and biases Q downward near loops (Pardo et al., 2018). If looping is undesirable, that belongs in the reward or the time limit, stated explicitly.
```

# 2. Returns, value functions and Bellman equations

The **return** from time $t$ is the discounted sum of future rewards:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma G_{t+1}.
$$

The recursion on the right is the whole trick of dynamic programming.

For a policy $\pi(a \mid s)$, the **state-value** and **action-value** functions are

$$
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s], \qquad q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a].
$$

**Bellman expectation equation.** Substitute $G_t = R_{t+1} + \gamma G_{t+1}$ and use the Markov property, which lets $\mathbb{E}[G_{t+1} \mid S_{t+1}=s']$ be written as $v_\pi(s')$:

$$
\begin{aligned}
q_\pi(s,a) &= \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid S_t=s, A_t=a] \\
&= \sum_{s'} P(s' \mid s,a)\Big[R(s,a,s') + \gamma \sum_{a'} \pi(a' \mid s')\, q_\pi(s',a')\Big].
\end{aligned}
$$

**Bellman optimality equation.** An optimal policy acts greedily with respect to its own values, so the inner average over $a'$ becomes a maximum:

$$
q_*(s,a) = \sum_{s'} P(s' \mid s,a)\Big[R(s,a,s') + \gamma \max_{a'} q_*(s',a')\Big].
$$

**Why a solution exists and is unique.** Define the Bellman optimality operator

$$
(\mathcal{T}q)(s,a) = \sum_{s'} P(s' \mid s,a)\big[R + \gamma \max_{a'} q(s',a')\big].
$$

For any two Q-functions $q_1$ and $q_2$:

$$
|(\mathcal{T}q_1)(s,a) - (\mathcal{T}q_2)(s,a)| \le \gamma \sum_{s'} P(s' \mid s,a)\, \big|\max_{a'} q_1(s',a') - \max_{a'} q_2(s',a')\big| \le \gamma\, \|q_1 - q_2\|_\infty .
$$

This uses $|\max f - \max g| \le \max|f-g|$. So $\mathcal{T}$ is a $\gamma$-contraction in the sup norm. By Banach's fixed-point theorem it has a unique fixed point $q_*$, and iterating $q_{k+1} = \mathcal{T}q_k$ (value iteration) converges to it geometrically:

$$
\|q_k - q_*\|_\infty \le \gamma^k \|q_0 - q_*\|_\infty .
$$

Q-learning is a *sampled*, *asynchronous* version of this iteration.

```widget
gamma_horizon
```

**Reading γ for Snake.** The **effective horizon** is about $1/(1-\gamma)$ steps: 20 steps at $\gamma=0.95$ and 100 at $0.99$.

- On a 12×12 board the nearest food is typically 5–15 steps away, so $\gamma=0.95$ "sees" the next apple.
- Avoiding a self-trap can require planning dozens of steps ahead, around the whole body. A short horizon is one reason a myopic agent coils into dead ends.
- A long horizon does not help when the state can't represent the trap (§5), which the γ sweep below confirms.

# 3. From dynamic programming to temporal-difference learning

We don't know $P$. **Monte Carlo** methods estimate $q_\pi(s,a)$ by averaging complete returns. That is unbiased, but the variance is high and you must wait until the episode ends.

**Temporal-difference (TD)** methods (Sutton, 1988) replace the unknown expectation with one sampled transition and *bootstrap* from the current estimate.

The **Q-learning** update for a transition $(s, a, r, s')$ is

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \,\underbrace{\big[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a) \,\big]}_{\delta \;=\; \text{TD error}} ,
$$

with $\max_{a'} Q(s', a') := 0$ when $s'$ is terminal.

- The bracketed quantity is a sample of $(\mathcal{T}Q)(s,a) - Q(s,a)$.
- Its expectation over $s'$ is exactly the Bellman residual.
- The update is a stochastic-approximation step toward the fixed point of $\mathcal{T}$.

**Q-learning is off-policy.** The target uses $\max_{a'}$, the greedy action, whatever action the behaviour policy takes next. So we can explore with $\varepsilon$-greedy and still learn about the greedy policy. SARSA uses the action actually taken next and learns the value of the exploring policy instead.

**Convergence** (Watkins & Dayan, 1992; Jaakkola, Jordan & Singh, 1994; Tsitsiklis, 1994). Tabular Q-learning converges to $q_*$ with probability 1 if:

1. the process is a finite MDP;
2. every $(s,a)$ pair is updated infinitely often;
3. the step sizes satisfy the Robbins–Monro conditions $\sum_t \alpha_t(s,a) = \infty$ and $\sum_t \alpha_t^2(s,a) < \infty$;
4. rewards are bounded.

Our run breaks two of these on purpose:

- **Constant α = 0.1.** This violates $\sum \alpha^2 < \infty$. The table never settles; it tracks recent experience. That is useful if the environment changes, but it shows up as checkpoint-to-checkpoint jitter (§8).
- **The process we learn on is not an MDP** in the feature space (§5). The guarantee simply doesn't apply.

```widget
q_update
```

```quiz
Q: In the update, why is the target r + γ·max Q(s',·) and not r + γ·Q(s', a_next), where a_next is the ε-greedy action actually taken?
- [ ] It is cheaper to compute
- [x] max makes the target the greedy (optimal) policy's value, so learning is off-policy
- [ ] a_next is not known yet
Why: Using the action actually taken gives SARSA, which learns q for the ε-greedy behaviour policy itself. Q-learning's max learns q* while behaving exploratorily. The cost of that is maximisation bias with noisy estimates, which Double Q-learning fixes (van Hasselt, 2010).
```

# 4. Exploration

**ε-greedy** takes a uniformly random action with probability $\varepsilon$ and otherwise the greedy one. With $\varepsilon$ decaying to 0 and all pairs visited infinitely often (GLIE: greedy in the limit with infinite exploration), the behaviour policy itself becomes optimal (Singh et al., 2000).

We decay linearly from $1.0$ to $0.01$ over the first 50% of episodes, then hold at $0.01$.

```widget
epsilon_schedule
```

Exploration costs the most in Snake when the snake is long. A single random turn next to the body is fatal. That is why the *training* score at $\varepsilon = 0.01$ (18.2) stays below the greedy *evaluation* score (20.1), and why the training curve is a poor measure of policy quality. **Always evaluate greedily, on held-out seeds.**

# 5. The state we actually learn on, and why it limits everything

The agent never sees the true state. It sees an 11-bit observation $o = \phi(s)$:

| Bits | Meaning |
|---|---|
| 0–2 | danger if moving straight / right / left (wall or body, tail excluded because it moves) |
| 3–6 | heading one-hot (R, D, L, U) |
| 7–10 | food is left / right / up / down of the head |

**Only 256 rows of the 2048 can occur.** Heading is one-hot, so 4 of the 16 patterns are possible. Food direction has 8 valid combinations: at most one of left/right and one of up/down is set, and the food is never under the head. Danger has 8. So $8 \times 4 \times 8 = 256$. The agent visited exactly 256 rows by epoch 125, which is a nice sanity check that exploration covered the reachable table.

**State aliasing.** Many true states map to the same observation. "Body ahead-left, food up-right" looks identical whether the free region on the right is 100 cells or 2 cells. The second case is a death trap, but the observation can't say so. Formally the agent faces a **partially observable MDP** (POMDP; Kaelbling, Littman & Cassandra, 1998).

Q-learning on observations treats $o$ as if it were Markov, and that has known consequences (Singh, Jaakkola & Jordan, 1994):

- the best *memoryless deterministic* policy can be arbitrarily worse than the optimal one;
- Q-learning need not converge to it;
- values can chatter as the mix of underlying states behind each $o$ shifts with the policy.

```widget
aliasing
```

That is the root cause of the plateau. With this observation, a near-optimal memoryless policy *is* "go toward the food unless the next cell is deadly". The learned table agrees with exactly that heuristic on **87.6%** of the states the heuristic visits, and scores the same (20.1 vs 20.4).

The same property has an upside. The features are size-invariant, so the 12×12 table plays other board sizes unchanged:

| Board | 10×10 | 16×16 | 20×20 |
|---|---:|---:|---:|
| Q-table | 18.4 | 25.4 | 29.7 |
| Heuristic | 17.9 | 24.5 | 28.0 |

(300 seeds each.)

```quiz
Q: The agent has visited all 256 reachable rows, ε is 0.01, and it has trained for another 10,000 episodes. Will more training eventually stop self-collisions?
- [ ] Yes, Q-learning converges to the optimal policy given enough data
- [x] No: the information needed (free space beyond one cell) is not in the observation
- [ ] Only if α is decayed
Why: Convergence theorems are about the MDP whose state you observe. Two situations that need opposite actions share one table row, so no table entry can be right for both. Decaying α would only freeze the compromise. The fix is a richer state: reachable-area features or the raw grid fed to a CNN (tryout 02).
```

# 6. Reward design and shaping

The reward defines the task. Anything you add changes the task unless it has a special form.

**Ng, Harada & Russell (1999)** proved that potential-based shaping, of the form

$$
F(s, s') = \gamma\,\Phi(s') - \Phi(s),
$$

leaves the optimal policy unchanged for *any* potential $\Phi$. The reason is that the shaping terms telescope inside the return:

$$
\sum_{k} \gamma^k F(S_{t+k}, S_{t+k+1}) = -\Phi(S_t) + \lim_{K\to\infty}\gamma^K \Phi(S_{t+K}) = -\Phi(S_t).
$$

So every action's value shifts by the same state-dependent constant.

An ad-hoc "+0.1 for moving closer to food, −0.1 for moving away" is *almost* this with $\Phi = -\text{distance}$. But without the $\gamma$ on $\Phi(s')$ and the $\Phi(\text{terminal}) = 0$ convention, it rewards back-and-forth oscillation, a classic case of specification gaming (Krakovna et al., 2020).

The Python environment offers `shaping="potential"` with $\Phi(s) = -\|\text{head}-\text{food}\|_1 / 2n$. Tryout 01 used **no shaping**: the ±1 reward is dense enough here, and adding shaping would confound the representation finding.

# 7. How the theory maps to the code

`tools/train_tabular.js` (Node) runs the same engine as the browser game, line for line:

| Theory | Code |
|---|---|
| $o = \phi(s)$, 11 bits → table row | `S.basicStateIndex(g)` packs `basicFeatures` into an integer 0–2047 |
| $\varepsilon$-greedy behaviour | `rng.nextU32()/2**32 < eps ? rng.int(3) : argmax(q[s])` |
| terminal ⇒ no bootstrap | `r.terminated ? 0 : gamma * Math.max(...q[s2])` |
| truncation ⇒ bootstrap | timeouts return `terminated:false, truncated:true`, so the `max` term stays |
| TD update with step size α | `q[s][a] += alpha * (target - q[s][a])` |
| greedy evaluation, held-out | `S.evaluate(greedy, {seeds: 100000…100499})` at every checkpoint |
| reproducibility | engine RNG (mulberry32) seeded per run; the same seed gives the identical run (verified: 2,173,663 steps twice) |

**Experimental protocol.**

- One epoch is 100 episodes. We checkpoint every 25 epochs, so there are 8 checkpoints in 200 epochs (20,000 episodes, about 2.2 M environment steps, about 2 s of CPU).
- Hyperparameters: α = 0.1, γ = 0.95, ε from 1.0 to 0.01 over the first 100 epochs.
- Every checkpoint evaluates the greedy policy on the same 500 held-out seeds, so checkpoints are directly comparable.
- Three training seeds on 12×12 and three on 6×6.
- Baselines on the same 500 seeds: random play scores 0.17; the greedy heuristic scores 20.4.

# 8. Results

```widget
checkpoint_table
```

**12×12, seed 1:**

- Eval mean goes 0.04 → 6.67 → 12.51 → 18.09 → 20.03 by epoch 100, the point where ε reaches its floor.
- After that the checkpoints read 19.16, 19.75, 21.68 and 20.08.
- Wins: 0 in 20,000 training episodes and 4,500 evaluation games.
- The best single game filled 36% of the board (score 49).

Watch the trained table play any of the held-out evaluation seeds. The heatmap colours every free cell by $\max_a Q$, as if the head were standing there with the current heading. Notice that the colours depend only on the 3-cell neighbourhood and the food direction, never on how much room is left.

```widget
replay
```

**Is the late jitter meaningful?** The spread of eval scores is about 6.3. That is estimated from the percentiles: p10 = 12 and p90 = 28, and for a roughly normal distribution $\sigma \approx (p_{90}-p_{10})/2.56$. With 500 games, the standard error of the mean is about $6.3/\sqrt{500} \approx 0.28$.

Swings of ±1.6 are therefore larger than sampling noise. They are *real* changes in the greedy policy, caused by the constant step size continually rewriting the table (§3), not measurement noise. Across three seeds the final means are 20.1, 21.3 and 20.1. Report the band, not the best checkpoint (Henderson et al., 2018; Agarwal et al., 2021).

**6×6 (win = score 33).** The plateau is at about 13. Across all seeds the run produced a handful of wins during training and **1 evaluation win in 13,500 games**. Those wins happen when food happens to land conveniently, not from planning.

```widget
sweep_heatmap
```

**γ and α sweep** (10k episodes, 3 seeds × 300 evaluation games per cell):

- Every setting with γ ≤ 0.95 and α ≤ 0.1 lands at 19–23.
- γ = 0.99 is *worse* and much less stable (seed spread 10–24 at α = 0.3). A longer horizon can't help when the trap isn't observable. It only propagates noisy, aliased values further.
- Large α amplifies the chattering predicted for aliased states.

# 9. Failure analysis

1. **Self-trapping (~90% of deaths).** Root cause: there is no information about free space beyond one cell. *Fix:* add reachable-area features per action (flood fill), or give a CNN the whole grid.
2. **Early loops (epoch 25: 21% of games timed out).** Unvisited rows hold ties at 0. `argmax` breaks ties toward "straight", and some feature patterns lead to cycles. This disappears once the table fills. *Lesson:* tie-breaking and the time limit are part of the algorithm.
3. **Wall deaths still 8–12%.** These come mostly from forced situations where all three moves are blocked, plus aliasing: a row learned mostly from safe cases occasionally fires in an unsafe one.
4. **No convergence, by construction.** Constant α plus a non-Markov observation gives a policy that keeps moving. If a stable artefact is needed, decay α, or keep the best checkpoint *by evaluation on separate validation seeds*, never by test-set score.
5. **Not a GPU problem.** The table has 2048×3 floats (24 KB dense, 8 KB as sparse JSON). Training takes 2 seconds on one CPU core. PyTorch would add overhead and nothing else, which is why this tryout uses none.

# 10. What this sets up

- The algorithm was never the bottleneck; the observation was. Tryout 02 keeps Q-learning but:
  1. feeds the **whole grid** (body recency, head, food, walls) to a small CNN, so the Q-function can see traps;
  2. replaces the table with a neural network, which needs **experience replay** and a **target network** to stay stable (Mnih et al., 2015).
- Expect sample efficiency to *drop*. The table needs about 2 M steps; a CNN DQN typically needs tens of millions. That is where vectorised environments on the GPU earn their keep.

# Self-check

```quiz
Q: Why is the effective number of table rows 256, not 2048?
- [ ] Exploration did not reach the rest
- [x] One-hot heading (4 of 16) × valid food directions (8) × danger patterns (8)
- [ ] The engine forbids some states
Why: The feature encoding has structural constraints. Recognising unreachable rows is a quick sanity check on exploration coverage and on bugs in feature code.
```

```quiz
Q: Training score at ε = 0.01 is 18.2 while greedy evaluation gives 20.1. Which statement is right?
- [ ] The evaluation is overfitting the eval seeds
- [x] Even 1% random actions are costly when the snake is long, so training scores understate the policy
- [ ] The two numbers come from different policies, so the gap is noise
Why: In Snake one random turn beside the body is fatal, and long episodes contain hundreds of such moments. Evaluate greedily, on held-out seeds, with enough episodes to put a standard error on the mean.
```

# References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS*.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep reinforcement learning that matters. *AAAI*.
- Jaakkola, T., Jordan, M. I., & Singh, S. P. (1994). On the convergence of stochastic iterative dynamic programming algorithms. *Neural Computation*, 6(6).
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. *Artificial Intelligence*, 101.
- Krakovna, V., et al. (2020). Specification gaming: the flip side of AI ingenuity. *DeepMind blog*.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. *ICML*.
- Pardo, F., Tavakoli, A., Levdik, V., & Kormushev, P. (2018). Time limits in reinforcement learning. *ICML*.
- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
- Singh, S. P., Jaakkola, T., & Jordan, M. I. (1994). Learning without state-estimation in partially observable Markovian decision processes. *ICML*.
- Singh, S., Jaakkola, T., Littman, M. L., & Szepesvári, C. (2000). Convergence results for single-step on-policy reinforcement-learning algorithms. *Machine Learning*, 38.
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, 3.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapters 3–6.
- Tsitsiklis, J. N. (1994). Asynchronous stochastic approximation and Q-learning. *Machine Learning*, 16.
- van Hasselt, H. (2010). Double Q-learning. *NeurIPS*.
- Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, University of Cambridge.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.
