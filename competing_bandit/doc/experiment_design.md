### Layer 1：忠实复现论文中的三张图

这是最基本的 sanity check，目的是验证你对算法和 setting 的理解是否正确。

#### 1.1 通用基础设施

需要先实现的 building blocks：

**Stable matching 求解器**：实现 agent-proposing Gale-Shapley 算法。这是所有 centralized 算法的核心 subroutine。注意要支持"输入 agents 的 ranking + arms 的 ranking → 输出 matching"。建议自己写而不是调包，因为后面 UCB 算法需要在每个 timestep 调用它，且要能 handle ties（虽然论文里默认 strict preferences）。

**Reward 模型**：1-sub-Gaussian，论文实验里实际用的是方差为 1 的 Gaussian。注意 reward 是 *per-agent* 的——同一个 arm 对不同 agent 给出不同分布的 reward。

**两个 Platform**：

- ETC Platform（Table 1 左）：前 hKhK hK 轮 cyclic assignment，第 hK+1hK+1 hK+1 轮调用 GS，之后 commit。
- GS Platform（Table 1 右）：每轮都接收 rankings 然后跑 GS。

**UCB ranking**：根据公式 (6) 计算 ui,j(t)=μ^i,j(t)+3log⁡t/(2Ti,j(t−1))u_{i,j}(t) = \hat{\mu}_{i,j}(t) + \sqrt{3\log t / (2T_{i,j}(t-1))} ui,j(t)=μ^i,j(t)+3logt/(2Ti,j(t−1))，然后按降序排列。⚠️ 注意论文公式 (6) 的下标看起来是 Ti(t−1)T_i(t-1) Ti(t−1) 但应该是 Ti,j(t−1)T_{i,j}(t-1) Ti,j(t−1)——这是个小 typo，提交 ranking 时未拉过的 arm 给 ∞\infty ∞ UCB。

#### 1.2 三个具体实验

**Figure 1a (Example 2)**：N=K=2N=K=2 N=K=2。扫描 Δ∈[0,1]\Delta \in [0, 1] Δ∈[0,1]（建议用 21 个点），固定 horizon n=400n=400 n=400，每个 Δ\Delta Δ 跑 100 trials，画 expected regret vs Δ\Delta Δ。关键观察：p2p_2 p2 的 regret 应该在 Δ\Delta Δ 小的时候很大（因为 p1p_1 p1 学不出来 a1≻a2a_1 \succ a_2 a1≻a2，导致 p2p_2 p2 拿不到自己的 optimal arm a2a_2 a2）。

**Figure 1b (Example 6)**：N=K=3N=K=3 N=K=3，preference 结构按论文给定。Agent p1,p2p_1, p_2 p1,p2 的 consecutive arm gap 是 1，p3p_3 p3 在 a1a_1 a1 和 a3a_3 a3 之间的 gap 只有 0.05。跑 centralized UCB 到 horizon 200，100 trials。**这里要观察的是 linear regret**——验证 centralized UCB 不能保证 sublinear *agent-optimal* regret（只能保证 agent-pessimal）。

**Figure 1c (Example 7)**：N=K=20N=K=20 N=K=20，global preferences（所有 agents 同 ranking，所有 arms 同 ranking），consecutive gap = 0.1，horizon 8000，50 trials。展示 5 个有代表性的 agents（rank 1, 5, 10, 15, 20）的 regret 曲线。Top-ranked agent 应该 sublinear，bottom-ranked 应该接近 0 甚至负的。

#### 1.3 复现的诚实性检查

跑完之后要回答：你的曲线和论文 Figure 1 在 *qualitative shape* 上一致吗？常见的坑：

- ETC 的 cyclic assignment 索引（论文写的是 at+i−1mod  K+1a_{t+i-1 \mod K + 1} at+i−1modK+1，注意 1-indexed vs 0-indexed）。
- UCB confidence width 的常数（论文用 3log⁡t/2T\sqrt{3\log t / 2T} 3logt/2T，是 Lattimore-Szepesvári 的 UCB1 standard form）。
- "successful pull" 的统计：Ti,j(t)T_{i,j}(t) Ti,j(t) 只在成功匹配时增加，*失败的 attempt 不计入*。这点对 decentralized ETC 的 Stage 1 尤其关键。

# Layer 1 完整实验设计方案

## Replication of Liu, Mania, Jordan (2020) — "Competing Bandits in Matching Markets"

本方案的设计原则：(i) 让一个 LLM coding agent 能够基于此文档**无歧义地**生成代码；(ii) 每个组件都有明确的单元测试；(iii) 实验结果可与论文 Figure 1 直接对比。

------

## 0. 项目总体架构

### 0.1 目录结构

```
matching_bandits_replication/
├── README.md
├── requirements.txt
├── config/
│   ├── figure_1a.yaml
│   ├── figure_1b.yaml
│   └── figure_1c.yaml
├── src/
│   ├── __init__.py
│   ├── market.py              # Market instance (preferences, reward distributions)
│   ├── gale_shapley.py        # Core GS algorithm
│   ├── platforms.py           # ETC Platform, GS Platform
│   ├── agents.py              # Empirical mean tracker, UCB ranker
│   ├── simulator.py           # Main simulation loop
│   ├── regret.py              # Regret computation utilities
│   └── utils.py               # RNG management, logging
├── tests/
│   ├── test_gale_shapley.py
│   ├── test_market.py
│   ├── test_platforms.py
│   ├── test_agents.py
│   └── test_regret.py
├── experiments/
│   ├── run_figure_1a.py
│   ├── run_figure_1b.py
│   └── run_figure_1c.py
├── analysis/
│   ├── plot_figure_1a.py
│   ├── plot_figure_1b.py
│   └── plot_figure_1c.py
├── data/                       # output: trajectories, regret arrays
└── figures/                    # output: PNG/PDF plots
```

### 0.2 依赖

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pyyaml>=6.0
pytest>=7.4
tqdm>=4.66
pandas>=2.0
```

### 0.3 全局约定

- **Indexing**：所有 agents 和 arms **0-indexed**（Python convention）。论文用 1-indexed，所有从论文公式翻译过来的地方需注释说明。
- **Preferences as rankings**：Agent $i$ 对 arms 的 preference 表示为 numpy array `pref[i]`，其中 `pref[i][k] = j` 意味着 agent $i$ 第 $k$-preferred 的 arm 是 $j$。Arms 对 agents 的 preference 同理。
- **RNG**：全程使用 `numpy.random.Generator`（即 `np.random.default_rng(seed)`），**禁止使用全局 `np.random` 状态**。
- **Reward**：所有实验 reward 为 Gaussian with variance 1（满足 1-sub-Gaussian）。
- **Logging level**：每个 trial 仅记录必要数据（pulled arm, reward, success flag），避免 OOM。

------

## 1. 核心数据结构

### 1.1 `Market` 类（src/market.py）

**用途**：封装一个 market instance（preferences + reward distributions），以及计算 ground-truth stable matchings。

**Class signature**：

```python
class Market:
    def __init__(
        self,
        N: int,                     # number of agents
        K: int,                     # number of arms (K >= N)
        mean_rewards: np.ndarray,   # shape (N, K), mean_rewards[i, j] = mu_i(j)
        arm_preferences: np.ndarray,# shape (K, N), arm_preferences[j] = ranking over agents (0=most preferred)
        reward_std: float = 1.0,    # std of Gaussian reward noise
    ):
        ...
    
    @property
    def agent_preferences(self) -> np.ndarray:
        """Derived from mean_rewards: shape (N, K), agent i's true preference ranking over arms."""
    
    def sample_reward(self, agent: int, arm: int, rng: np.random.Generator) -> float:
        """Sample N(mean_rewards[agent, arm], reward_std^2)."""
    
    def optimal_stable_matching(self) -> np.ndarray:
        """Returns m_optimal: shape (N,), m_optimal[i] = arm matched to agent i in agent-optimal stable matching.
        Computed via agent-proposing GS on TRUE preferences."""
    
    def pessimal_stable_matching(self) -> np.ndarray:
        """Returns m_pessimal: shape (N,), via arm-proposing GS on TRUE preferences."""
```

**实现细节**：

- `agent_preferences` 通过对 `mean_rewards[i]` 做 `np.argsort(-mean_rewards[i])` 得到（降序，最高 mean reward 排第一）。
- 必须 assert `mean_rewards` 的每一行没有 ties（否则 stable matching 可能不唯一且 GS 行为依赖 tie-breaking）。
- `arm_preferences[j]` 是一个 length-$N$ 的 permutation，`arm_preferences[j][r]` 是 arm $j$ 的第 $r$ 偏好的 agent index。

### 1.2 `Market.from_example_2` 等工厂方法

为了精确复现，需要 hard-code 论文 Example 2, 6, 7 的 market：

```python
@classmethod
def from_example_2(cls, delta: float, reward_std: float = 1.0) -> "Market":
    """
    Example 2: N=K=2.
    p1: a1 > a2 (with reward gap delta), so mu_1 = [delta, 0]
    p2: a2 > a1 (with gap 1), so mu_2 = [0, 1]
    Both arms prefer p1 over p2.
    """
    N, K = 2, 2
    mean_rewards = np.array([[delta, 0.0], [0.0, 1.0]])
    arm_preferences = np.array([[0, 1], [0, 1]])  # both arms: p1 > p2
    return cls(N, K, mean_rewards, arm_preferences, reward_std)

@classmethod
def from_example_6(cls, p3_gap: float = 0.05, reward_std: float = 1.0) -> "Market":
    """
    Example 6: N=K=3.
    p1: a1 > a2 > a3, gaps of 1: mu_1 = [2, 1, 0]
    p2: a2 > a1 > a3, gaps of 1: mu_2 = [1, 2, 0]
    p3: a3 > a1 > a2, with gap between a1 and a3 = p3_gap: mu_3 = [p3_gap, 0, p3_gap + small]
        Wait — we need a3 > a1 > a2 AND mu_3(a3) - mu_3(a1) = p3_gap.
        Set mu_3 = [1.0, 0.0, 1.0 + p3_gap].  Then a3 > a1 > a2 is satisfied.
    Arms:
        a1: p2 > p3 > p1
        a2: p1 > p2 > p3
        a3: p3 > p1 > p2
    """
    N, K = 3, 3
    mean_rewards = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [1.0, 0.0, 1.0 + p3_gap],
    ])
    arm_preferences = np.array([
        [1, 2, 0],   # a1: p2, p3, p1
        [0, 1, 2],   # a2: p1, p2, p3
        [2, 0, 1],   # a3: p3, p1, p2
    ])
    return cls(N, K, mean_rewards, arm_preferences, reward_std)

@classmethod
def from_example_7(cls, N: int = 20, gap: float = 0.1, reward_std: float = 1.0) -> "Market":
    """
    Example 7: Global preferences.
    All agents share the same arm preference: a_0 > a_1 > ... > a_{N-1}.
    All arms share the same agent preference: p_0 > p_1 > ... > p_{N-1}.
    Mean rewards: mu_i(j) = (N - 1 - j) * gap, identical across all agents.
    """
    K = N
    mean_rewards = np.tile(
        np.array([(N - 1 - j) * gap for j in range(K)]),
        (N, 1)
    )
    arm_preferences = np.tile(np.arange(N), (K, 1))
    return cls(N, K, mean_rewards, arm_preferences, reward_std)
```

⚠️ **关键陷阱**：Example 6 里 $p_3$ 的 mean rewards 必须严格满足 $\mu_3(a_3) > \mu_3(a_1) > \mu_3(a_2)$，且 $\mu_3(a_3) - \mu_3(a_1) = p_3_gap$。论文用了 0.05 的 gap，意味着 $p_3$ 学起来很慢。

------

## 2. Gale-Shapley 算法（src/gale_shapley.py）

**接口**：

```python
def gale_shapley(
    proposer_prefs: np.ndarray,    # shape (N_proposers, K_receivers)
    receiver_prefs: np.ndarray,    # shape (K_receivers, N_proposers)
) -> np.ndarray:
    """
    Returns matching: shape (N_proposers,), matching[i] = index of receiver matched to proposer i,
    or -1 if unmatched (only possible if N_proposers > K_receivers).
    
    Implements agent-proposing deferred acceptance:
    - Each round, every unmatched proposer proposes to their most-preferred receiver they haven't been rejected by.
    - Each receiver tentatively holds the most-preferred proposal seen so far, rejecting others.
    - Terminates when no proposer is unmatched or all rejections exhausted.
    """
```

**实现细节**：

```python
def gale_shapley(proposer_prefs, receiver_prefs):
    N = proposer_prefs.shape[0]
    K = receiver_prefs.shape[0]
    assert proposer_prefs.shape == (N, K)
    assert receiver_prefs.shape == (K, N)
    
    # Build receiver's rank lookup: receiver_rank[j, i] = rank of proposer i in receiver j's preferences (lower = better)
    receiver_rank = np.full((K, N), N, dtype=int)  # default: worst rank
    for j in range(K):
        for r, i in enumerate(receiver_prefs[j]):
            receiver_rank[j, i] = r
    
    # next_proposal[i] = index into proposer_prefs[i] of next receiver to propose to
    next_proposal = np.zeros(N, dtype=int)
    # current_match[j] = proposer index currently held by receiver j, or -1
    current_match = np.full(K, -1, dtype=int)
    # proposer_match[i] = receiver currently holding proposer i, or -1
    proposer_match = np.full(N, -1, dtype=int)
    
    free_proposers = list(range(N))
    
    while free_proposers:
        i = free_proposers.pop(0)
        if next_proposal[i] >= K:
            continue  # exhausted, remains unmatched
        j = proposer_prefs[i, next_proposal[i]]
        next_proposal[i] += 1
        
        if current_match[j] == -1:
            current_match[j] = i
            proposer_match[i] = j
        else:
            i_held = current_match[j]
            if receiver_rank[j, i] < receiver_rank[j, i_held]:
                # j prefers i over i_held
                current_match[j] = i
                proposer_match[i] = j
                proposer_match[i_held] = -1
                free_proposers.append(i_held)
            else:
                free_proposers.append(i)
    
    return proposer_match
```

**单元测试**（tests/test_gale_shapley.py）：

```python
def test_gs_classic_3x3():
    """Standard textbook example with known unique stable matching."""
    # Men's prefs (rows) and women's prefs (rows)
    men_prefs = np.array([[0, 1, 2], [1, 0, 2], [0, 1, 2]])
    women_prefs = np.array([[2, 0, 1], [0, 2, 1], [0, 1, 2]])
    matching = gale_shapley(men_prefs, women_prefs)
    # Verify stability manually
    assert is_stable(matching, men_prefs, women_prefs)

def test_gs_unique_pairs():
    """Each agent prefers their 'partner' arm and vice versa → identity matching."""
    N = 5
    prefs_a = np.array([np.roll(np.arange(N), -i) for i in range(N)])  # i prefers i first
    prefs_b = prefs_a.copy()
    matching = gale_shapley(prefs_a, prefs_b)
    np.testing.assert_array_equal(matching, np.arange(N))

def test_gs_global_preferences():
    """Example 7 setup: identity matching."""
    N = 10
    prefs = np.tile(np.arange(N), (N, 1))
    matching = gale_shapley(prefs, prefs)
    np.testing.assert_array_equal(matching, np.arange(N))

def test_gs_optimality_for_proposers():
    """Verify proposer-optimal property: no proposer prefers another stable matching's partner."""
    # Use a known case with multiple stable matchings
    ...

def is_stable(matching, proposer_prefs, receiver_prefs):
    """Checker: returns True iff no blocking pair exists."""
    N, K = proposer_prefs.shape
    inverse_match = {-1: -1}
    for i, j in enumerate(matching):
        inverse_match[j] = i
    for i in range(N):
        for j in range(K):
            if matching[i] == j:
                continue
            i_prefers_j = (
                np.where(proposer_prefs[i] == j)[0][0]
                < np.where(proposer_prefs[i] == matching[i])[0][0]
            ) if matching[i] != -1 else True
            current_holder = inverse_match.get(j, -1)
            j_prefers_i = (
                current_holder == -1
                or np.where(receiver_prefs[j] == i)[0][0]
                < np.where(receiver_prefs[j] == current_holder)[0][0]
            )
            if i_prefers_j and j_prefers_i:
                return False
    return True
```

------

## 3. Empirical mean tracker 与 UCB ranker（src/agents.py）

### 3.1 `EmpiricalMeanTracker`

```python
class EmpiricalMeanTracker:
    """Per-agent state: tracks pull counts and empirical means for each arm."""
    
    def __init__(self, K: int):
        self.K = K
        self.counts = np.zeros(K, dtype=int)        # T_{i,j}(t)
        self.sum_rewards = np.zeros(K, dtype=float)
    
    def update(self, arm: int, reward: float) -> None:
        """Called only when pull is SUCCESSFUL."""
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
    
    def empirical_mean(self) -> np.ndarray:
        """Returns mu_hat: shape (K,). Returns 0 for arms with zero pulls (will be overridden by UCB)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_hat = np.where(self.counts > 0, self.sum_rewards / np.maximum(self.counts, 1), 0.0)
        return mu_hat
```

### 3.2 Ranking utilities

```python
def rank_by_empirical_mean(tracker: EmpiricalMeanTracker, rng: np.random.Generator) -> np.ndarray:
    """Returns ranking: shape (K,), ranking[r] = arm at rank r (0 = most preferred).
    For ETC: just sort descending by empirical mean. Break ties RANDOMLY (important for Stage 1 cyclic exploration consistency)."""
    mu_hat = tracker.empirical_mean()
    # Add tiny random noise for tie-breaking
    noise = rng.uniform(0, 1e-10, size=tracker.K)
    return np.argsort(-(mu_hat + noise))


def rank_by_ucb(tracker: EmpiricalMeanTracker, t: int, rng: np.random.Generator) -> np.ndarray:
    """
    UCB ranking per Equation (6) in the paper:
        u_{i,j}(t) = inf if T_{i,j}(t) == 0
                   = mu_hat_{i,j}(t) + sqrt(3 log(t) / (2 * T_{i,j}(t-1)))   otherwise
    
    NOTE on the t indexing: the paper writes T_i(t-1) but means T_{i,j}(t) (the count
    at the time UCB is computed). We use current count.
    NOTE on log(t): when t == 1, log(t) = 0 → confidence width = 0; this only matters
    if any arm has been pulled zero times (handled by the inf rule). For t >= 2 it's fine.
    """
    K = tracker.K
    ucb = np.full(K, np.inf)
    pulled = tracker.counts > 0
    if pulled.any() and t >= 1:
        log_t = np.log(max(t, 1))
        ucb[pulled] = (
            tracker.sum_rewards[pulled] / tracker.counts[pulled]
            + np.sqrt(3.0 * log_t / (2.0 * tracker.counts[pulled]))
        )
    # Tie-break randomly among +inf entries (otherwise argsort gives lexicographic order)
    noise = rng.uniform(0, 1e-10, size=K)
    return np.argsort(-(ucb + noise))
```

⚠️ **关键注释**：UCB 公式里 `t` 应该是 *当前 round number*（global time），不是 per-agent pull count。论文用 $\log t$ 的 $t$ 指 global time。

------

## 4. Platforms（src/platforms.py）

### 4.1 ETC Platform

```python
class ETCPlatform:
    """Explore-then-Commit Platform (Table 1 left).
    
    For t = 1, ..., hK: cyclic assignment, each agent samples each arm exactly h times.
    At t = hK + 1: collect rankings from agents, run GS, commit forever.
    """
    
    def __init__(self, N: int, K: int, h: int, arm_preferences: np.ndarray):
        assert K >= N
        self.N = N
        self.K = K
        self.h = h
        self.arm_preferences = arm_preferences
        self.committed_matching = None  # shape (N,) once Stage 2 starts
    
    def assign(self, t: int, rankings: np.ndarray) -> np.ndarray:
        """
        Args:
            t: 1-indexed round number.
            rankings: shape (N, K). Only used at t == h*K + 1.
        Returns:
            assignment: shape (N,), assignment[i] = arm assigned to agent i.
        """
        if t <= self.h * self.K:
            # Cyclic: at round t, agent i is assigned arm (t + i - 1) mod K
            # (Translating from paper's 1-indexed: agent index i in [1, N], round t in [1, hK].
            # Paper formula: m_t(i) = a_{(t + i - 1) mod K + 1}.
            # In 0-indexed Python with t_py = t-1, i_py = i-1:
            #     arm = ((t_py + 1) + (i_py + 1) - 1 - 1) mod K = (t_py + i_py) mod K
            t_py = t - 1
            assignment = np.array([(t_py + i_py) % self.K for i_py in range(self.N)])
            return assignment
        elif t == self.h * self.K + 1:
            # Run GS using submitted rankings (proposer = agents)
            self.committed_matching = gale_shapley(rankings, self.arm_preferences)
            return self.committed_matching.copy()
        else:
            return self.committed_matching.copy()
```

**重要的 cyclic assignment 验证**：必须保证经过 $hK$ 轮后，每个 agent 恰好与每个 arm 配对 $h$ 次。可以单元测试这一点。

### 4.2 GS Platform

```python
class GSPlatform:
    """Gale-Shapley Platform (Table 1 right).
    
    Every round, runs GS on agents' current submitted rankings.
    """
    
    def __init__(self, N: int, K: int, arm_preferences: np.ndarray):
        self.N = N
        self.K = K
        self.arm_preferences = arm_preferences
    
    def assign(self, t: int, rankings: np.ndarray) -> np.ndarray:
        """Returns matching from GS on rankings."""
        return gale_shapley(rankings, self.arm_preferences)
```

注意：GSPlatform 永远不会有 collision —— GS 输出的 matching 是 injective。所以"successful pull"对每个被 match 上的 agent 都成立。

------

## 5. Simulator（src/simulator.py）

### 5.1 单次 trial 的主循环

```python
def run_trial(
    market: Market,
    platform,                    # ETCPlatform or GSPlatform
    ranking_strategy: str,       # "empirical_mean" or "ucb"
    horizon: int,
    rng: np.random.Generator,
) -> dict:
    """
    Returns a dict with:
        "matchings":  shape (horizon, N), arm assigned to each agent each round
        "rewards":    shape (horizon, N), realized reward (0 if no successful pull, but in centralized
                      both platforms always produce conflict-free assignments → all pulls successful)
        "successful": shape (horizon, N), bool flag
    """
    N, K = market.N, market.K
    trackers = [EmpiricalMeanTracker(K) for _ in range(N)]
    
    matchings = np.zeros((horizon, N), dtype=int)
    rewards = np.zeros((horizon, N), dtype=float)
    successful = np.zeros((horizon, N), dtype=bool)
    
    for t_py in range(horizon):
        t = t_py + 1  # 1-indexed for platform/UCB
        
        # 1. Each agent computes a ranking.
        rankings = np.zeros((N, K), dtype=int)
        for i in range(N):
            if ranking_strategy == "empirical_mean":
                rankings[i] = rank_by_empirical_mean(trackers[i], rng)
            elif ranking_strategy == "ucb":
                rankings[i] = rank_by_ucb(trackers[i], t, rng)
            else:
                raise ValueError(ranking_strategy)
        
        # 2. Platform assigns.
        assignment = platform.assign(t, rankings)
        matchings[t_py] = assignment
        
        # 3. For ETC platform during exploration, conflicts can happen in principle but
        #    cyclic assignment is collision-free. For GS platform, output is also collision-free.
        #    So all assignments are successful pulls.
        for i in range(N):
            arm = assignment[i]
            r = market.sample_reward(i, arm, rng)
            rewards[t_py, i] = r
            successful[t_py, i] = True
            trackers[i].update(arm, r)
    
    return {
        "matchings": matchings,
        "rewards": rewards,
        "successful": successful,
    }
```

⚠️ **设计决策**：在 Layer 1 的 centralized setting 下，所有 assignment 都是 conflict-free 的（cyclic 保证、GS 保证）。所以"反映 reward 的 mean"等于直接 sample 并更新。如果未来扩展到 decentralized ETC，则需要在更新前检查 collision —— 那时 `successful` 才会有 False。

### 5.2 Multi-trial driver

```python
def run_experiment(
    market_factory,              # callable: () -> Market (allows fresh randomness for stochastic markets)
    platform_factory,            # callable: (market) -> Platform
    ranking_strategy: str,
    horizon: int,
    n_trials: int,
    seed: int,
    n_jobs: int = 1,
) -> list[dict]:
    """Runs n_trials independent trials with seeds derived from `seed`."""
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_trials)
    
    results = []
    for trial_idx in range(n_trials):
        rng = np.random.default_rng(child_seeds[trial_idx])
        market = market_factory()  # for deterministic markets, returns same instance
        platform = platform_factory(market)
        result = run_trial(market, platform, ranking_strategy, horizon, rng)
        results.append(result)
    return results
```

对于 Layer 1，三个实验的 market 都是 deterministic（无 randomness），所以 `market_factory` 每次返回同一个 `Market` 实例。如果用 `joblib.Parallel` 加速，注意 `Market` 必须 picklable。

------

## 6. Regret 计算（src/regret.py）

```python
def agent_optimal_regret(
    market: Market,
    matchings: np.ndarray,       # shape (horizon, N)
    rewards: np.ndarray,         # shape (horizon, N) — used for empirical regret
    use_expected: bool = True,
) -> np.ndarray:
    """
    Per Equation (1): R_i(n) = n * mu_i(m_optimal(i)) - sum_t E[X_{i, m_t(i)}(t)]
    
    If use_expected=True: subtract mean rewards (for variance reduction in plots).
    If use_expected=False: subtract realized rewards.
    
    Returns: shape (horizon, N), CUMULATIVE regret per agent over time.
    """
    N = market.N
    horizon = matchings.shape[0]
    m_opt = market.optimal_stable_matching()
    
    # Per-round regret = mu_i(m_opt(i)) - mu_i(m_t(i))
    optimal_means = market.mean_rewards[np.arange(N), m_opt]  # shape (N,)
    if use_expected:
        played_means = market.mean_rewards[
            np.arange(N)[None, :].repeat(horizon, axis=0),
            matchings
        ]  # shape (horizon, N)
        instantaneous = optimal_means[None, :] - played_means
    else:
        instantaneous = optimal_means[None, :] - rewards
    
    return np.cumsum(instantaneous, axis=0)


def agent_pessimal_regret(market, matchings, rewards, use_expected=True):
    """Per Equation (2): same as above but using pessimal stable matching as baseline."""
    # Identical structure, but using market.pessimal_stable_matching()
    ...
```

**一个细节**：论文 Figure 1 用的是 *expected regret*（有 `E` 符号），所以我们用 mean rewards 而非 realized rewards 计算（这等价于无穷次 trial 平均，但对于 finite trial 平均能显著降低画图方差）。实际代码两种都实现，作为 sanity check 它们的 trial-average 应该收敛到同一条曲线。

------

## 7. 三个实验的具体配置

### 7.1 Figure 1a：Example 2，UCB on GS Platform

**Config (config/figure_1a.yaml)**：

```yaml
experiment: figure_1a
algorithm: centralized_ucb
N: 2
K: 2
horizon: 400
n_trials: 100
seed: 20240101
delta_grid:
  start: 0.0
  stop: 1.0
  num: 21         # 0.00, 0.05, 0.10, ..., 1.00
plot:
  yscale: log
  ylim: [1, 200]
  agents_to_show: [0, 1]
```

**Script (experiments/run_figure_1a.py)**：

```python
def main(cfg):
    deltas = np.linspace(cfg["delta_grid"]["start"],
                          cfg["delta_grid"]["stop"],
                          cfg["delta_grid"]["num"])
    
    # Output: shape (len(deltas), N) = mean final regret per agent
    mean_regrets = np.zeros((len(deltas), 2))
    std_regrets = np.zeros((len(deltas), 2))
    
    for d_idx, delta in enumerate(deltas):
        market = Market.from_example_2(delta=delta)
        platform_factory = lambda m: GSPlatform(m.N, m.K, m.arm_preferences)
        results = run_experiment(
            market_factory=lambda: market,
            platform_factory=platform_factory,
            ranking_strategy="ucb",
            horizon=cfg["horizon"],
            n_trials=cfg["n_trials"],
            seed=cfg["seed"] + d_idx,
        )
        # Compute final cumulative agent-optimal regret per trial
        final_regrets = np.zeros((cfg["n_trials"], 2))
        for t_idx, res in enumerate(results):
            cum = agent_optimal_regret(market, res["matchings"], res["rewards"], use_expected=True)
            final_regrets[t_idx] = cum[-1]  # final value
        mean_regrets[d_idx] = final_regrets.mean(axis=0)
        std_regrets[d_idx] = final_regrets.std(axis=0) / np.sqrt(cfg["n_trials"])  # SEM
    
    np.savez("data/figure_1a.npz",
             deltas=deltas, mean_regrets=mean_regrets, std_regrets=std_regrets)
```

**预期结果**：

- $p_2$ 的 expected regret 在小 $\Delta$ 时显著高于 $p_1$（解释：$p_1$ 学不出 $a_1 \succ a_2$ 时，$p_2$ 拿不到 optimal arm $a_2$，被迫匹配到 $a_1$，损失 $\Delta_{2,1} = 1$ 每轮）。
- $p_1$ 的 regret 应该接近常数（与 $\Delta$ 弱相关），但 $p_2$ 的 regret 应随 $\Delta$ 增大而下降。

### 7.2 Figure 1b：Example 6，UCB on GS Platform

**Config (config/figure_1b.yaml)**：

```yaml
experiment: figure_1b
algorithm: centralized_ucb
N: 3
K: 3
horizon: 200
n_trials: 100
seed: 20240102
p3_gap: 0.05
plot:
  yscale: linear
  agents_to_show: [0, 1, 2]
```

**Script**：跑一次（不扫参），输出 shape `(horizon, N)` 的 mean cumulative agent-optimal regret 曲线及 SEM band。

**预期结果**：所有三个 agent 的 agent-optimal regret 都呈 **linear** 增长。这对应论文 Example 6 说明 centralized UCB 不能保证 sublinear agent-optimal regret —— 当 $p_3$ 一旦在前几轮中误判 $a_1 \succ a_3$ 而被分配到 $a_3$（恰好是 $p_3$ 的 optimal arm！wait 这里需要再读论文…）。

⚠️ **再读 Example 6**：

- True optimal stable matching: $(p_1, a_1), (p_2, a_2), (p_3, a_3)$.
- 当 $p_3$ 误把 $a_1 \succ a_3$ 排序，GS 给出 $(p_1, a_2), (p_2, a_1), (p_3, a_3)$.
- 结果：$p_3$ 仍然得到 $a_3$（其 optimal），所以 $p_3$ 的 regret = 0；但 $p_1, p_2$ 的 optimal regret 是 linear（他们没拿到自己的 optimal）。
- 而 $p_3$ 因为一直 match 到 $a_3$，永远不会更新 $a_1$ 的 UCB → 错误永久存在。

所以预期：**$p_1, p_2$ linear regret，$p_3$ regret ≈ 0**。这与论文 Figure 1b 的描述一致："the optimal regret for $p_1$ and $p_2$ is seen to be linear in $n$"。

### 7.3 Figure 1c：Example 7，UCB on GS Platform

**Config (config/figure_1c.yaml)**：

```yaml
experiment: figure_1c
algorithm: centralized_ucb
N: 20
K: 20
horizon: 8000
n_trials: 50
seed: 20240103
gap: 0.1
plot:
  yscale: linear
  agents_to_show: [0, 4, 9, 14, 19]   # 1st, 5th, 10th, 15th, 20th in 1-indexed paper notation
```

**Script**：单次实验，记录 5 个特定 agent 的 cumulative agent-optimal regret 曲线（with SEM band）。

**预期结果**：

- Agent 0（top-ranked）：sublinear (logarithmic) regret，正值。
- Agent 19（bottom-ranked）：regret 接近 0 或为负（因为他在 stable matching 里只能拿到最差的 arm，几乎不可能"更糟"）。
- 中间 ranked agents：介于两者之间。

⚠️ 论文 Figure 1c 显示某些 agent 的 expected regret 是**负的**——这是因为 agent-optimal regret 对 sub-optimal agent 来说，played reward 经常高于"optimal stable match reward"（因为 platform 偶尔会 mismatch 给他更好的 arm）。

------

## 8. 测试矩阵

### 8.1 单元测试列表

| 测试模块               | 测试内容                                                     | 期望行为                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `test_gale_shapley.py` | (a) 3×3 textbook case                                        | 输出唯一 stable matching                                     |
|                        | (b) Identity preferences (Example 7)                         | identity matching                                            |
|                        | (c) Stability check on random instances                      | 100 random 5×5 → 100 stable                                  |
|                        | (d) Proposer optimality                                      | 与 arm-proposing GS 对比，proposer 至少不差                  |
| `test_market.py`       | (a) `from_example_2` 的 optimal matching                     | $(p_1, a_1), (p_2, a_2)$                                     |
|                        | (b) `from_example_6` 的 optimal matching                     | $(p_1, a_1), (p_2, a_2), (p_3, a_3)$                         |
|                        | (c) `from_example_7` 的 optimal matching                     | identity                                                     |
|                        | (d) `agent_preferences` 与 `mean_rewards` 一致               | argsort 验证                                                 |
|                        | (e) Reward sample 的均值 (10000 samples)                     | $|\bar{X} - \mu| < 0.05$                                     |
| `test_platforms.py`    | (a) ETC cyclic phase: 经过 $hK$ 轮后每对 (i,j) 配对恰好 $h$ 次 | counter check                                                |
|                        | (b) ETC 在 $t = hK+1$ 时使用 GS 输出                         | 对比独立 GS 调用                                             |
|                        | (c) ETC 在 commit 阶段不变                                   | $t > hK+1$ 时同 $t = hK+1$                                   |
|                        | (d) GSPlatform 输出 stable                                   | `is_stable` check                                            |
| `test_agents.py`       | (a) `EmpiricalMeanTracker` 增量更新正确                      | 与离线计算对比                                               |
|                        | (b) `rank_by_ucb` 在 t=1 时所有 arm 都是 inf → 随机排列      | 多次调用，验证非 deterministic                               |
|                        | (c) `rank_by_ucb` 在大量 pull 后收敛到真实 ranking           | 用已知 mean 跑 10000 次更新                                  |
| `test_regret.py`       | (a) Optimal matching 下 expected regret = 0                  | 强制 matching = m_opt                                        |
|                        | (b) 累积单调非负（agent-optimal w.r.t. matching = m_opt 是 0） | 一般 instance 上 cum regret 非递减（对 agent-optimal）—注意不一定对所有 agent |

### 8.2 集成测试

**Smoke test**：用极小参数（horizon=10, n_trials=2）跑完三个实验脚本，验证无报错且输出文件结构正确。

**Sanity test**：

- Figure 1b 跑 horizon=50, n_trials=20，检查 $p_3$ 的 final regret < 5（应该接近 0），$p_1, p_2$ 的 final regret > 10（linear-ish）。
- Figure 1c 跑 horizon=2000, n_trials=10, agent 0 的 final regret 应该明显小于 horizon × max gap（即 < 200），agent 19 的 final regret 应该接近 0 或为负。

------

## 9. 画图规范（analysis/）

### 9.1 通用样式

```python
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "figure.figsize": (4.5, 3.0),
    "figure.dpi": 120,
})
```

### 9.2 Figure 1a

```python
data = np.load("data/figure_1a.npz")
fig, ax = plt.subplots()
for i, label in enumerate(["Agent 1", "Agent 2"]):
    ax.errorbar(data["deltas"], data["mean_regrets"][:, i],
                yerr=data["std_regrets"][:, i], label=label, marker="o", capsize=2)
ax.set_yscale("log")
ax.set_xlabel("Reward Gap of Agent 1 ($\\Delta$)")
ax.set_ylabel("Expected Regret")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/figure_1a.pdf")
```

### 9.3 Figure 1b & 1c

```python
# Figure 1b: line plot of cumulative regret vs time, with SEM shading
data = np.load("data/figure_1b.npz")  # contains "cum_regret_mean" (T, N) and "cum_regret_sem"
fig, ax = plt.subplots()
T = data["cum_regret_mean"].shape[0]
times = np.arange(1, T + 1)
for i in range(3):
    mean = data["cum_regret_mean"][:, i]
    sem = data["cum_regret_sem"][:, i]
    ax.plot(times, mean, label=f"Agent {i+1}")
    ax.fill_between(times, mean - sem, mean + sem, alpha=0.2)
ax.set_xlabel("Time")
ax.set_ylabel("Expected Regret")
ax.legend()
plt.tight_layout()
plt.savefig("figures/figure_1b.pdf")
```

Figure 1c 几乎一样，只是 agents_to_show 是 `[0, 4, 9, 14, 19]`。

------

## 10. 实验运行清单（按顺序执行）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 单元测试（必须全过）
pytest tests/ -v

# 3. Smoke test（参数缩小到 horizon=10, n_trials=2）
python experiments/run_figure_1a.py --config config/figure_1a.yaml --smoke
python experiments/run_figure_1b.py --config config/figure_1b.yaml --smoke
python experiments/run_figure_1c.py --config config/figure_1c.yaml --smoke

# 4. 完整运行
python experiments/run_figure_1a.py --config config/figure_1a.yaml   # ~2 min
python experiments/run_figure_1b.py --config config/figure_1b.yaml   # ~10 sec
python experiments/run_figure_1c.py --config config/figure_1c.yaml   # ~5 min

# 5. 画图
python analysis/plot_figure_1a.py
python analysis/plot_figure_1b.py
python analysis/plot_figure_1c.py
```

------

## 11. 容易出错的地方（给 Opus 4.7 的明确警示）

把这一节作为 prompt 的一部分喂给 coding agent，能显著降低 bug 率：

1. **Indexing 一致性**：所有内部 array 0-indexed；UCB 公式中的 `t` 必须是 1-indexed 的 round number。
2. **Cyclic assignment 公式**：见 4.1 节注释。验证方式：`hK` 轮后所有 `(agent, arm)` pair 的 count 都恰好等于 `h`。
3. **UCB confidence width 中的计数**：用 `T_{i,j}(t)`（成功 pull 次数），不是 attempt 次数。Centralized 下两者相等，但记得这点对未来扩展重要。
4. **Tie-breaking**：empirical mean 完全相等的 arm（特别在 t=0 或 t=1 时）必须随机打破，否则 GS 永远偏向小 index 的 arm，引入 systematic bias。
5. **Expected regret vs realized regret**：画图用 expected（基于 `market.mean_rewards`），单元测试两种方式应在大 trial 平均下一致。
6. **`market.mean_rewards` 的索引**：`mean_rewards[i, j]` 是 agent $i$ 在 arm $j$ 上的 mean，**不是** ranking。Agent 的 preference ranking 是 *derived*，不要混淆。
7. **GS 输入的 prefs 矩阵**：`proposer_prefs[i]` 是 agent $i$ 提交的 ranking（即"按偏好排序的 arm 序列"），不是 mean rewards。当 ranking 来自 `rank_by_ucb` 时，输出格式正确——但要 double-check。
8. **Negative regret 不是 bug**：Example 6 和 Example 7 中部分 agent 的 expected agent-optimal regret 为负是预期行为，不要在代码里 clip 到非负。

------

## 12. 交付物清单

完成 Layer 1 后应有：

- ✅ `src/` 完整代码，所有单元测试通过

- ✅ 三个 config YAML

- ✅ 三个 experiment script + 三个 plotting script

- ✅ `data/figure_1a.npz`, `data/figure_1b.npz`, `data/figure_1c.npz`（含 `mean_rewards` 矩阵 metadata）

- ✅ `figures/figure_1a.pdf`, `figure_1b.pdf`, `figure_1c.pdf`

- ✅ 一份 

  ```
  REPLICATION_NOTES.md
  ```

  ，记录：

  - 实际运行时间
  - 与论文 Figure 1 的 qualitative 对比（哪些一致、哪些有差异、可能的原因）
  - 所有 random seed
  - Numpy / Python 版本

------

需要我再把这套方案的某一部分（比如 GS 算法的伪代码、UCB 公式的 derivation 注释、或者 plotting 的 publication-quality styling）展开到更细颗粒度吗？或者直接帮你写出第一个 prompt 模板（用来让 Opus 4.7 一次性生成 `src/gale_shapley.py` + 对应测试）？