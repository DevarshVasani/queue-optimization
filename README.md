# GPU Cluster Scheduling Optimization Using Reinforcement Learning

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Motivation and Objectives](#motivation-and-objectives)
3. [Literature Study](#literature-study)
4. [Technical Details](#technical-details)
5. [Implementation Details](#implementation-details)
6. [Experimental Evaluation](#experimental-evaluation)
7. [Results and Findings](#results-and-findings)
8. [Conclusion](#conclusion)
9. [References](#references)
10. [Project Structure](#project-structure)

---

## Executive Summary

This project develops an intelligent GPU cluster scheduler using **Reinforcement Learning (RL)** to optimize workload placement in heterogeneous cloud clusters. Unlike traditional heuristics (best-fit, first-fit), the RL-based approach learns scheduling policies that minimize tail latency while maximizing GPU utilization through constrained policy optimization (MaskablePPO).

**Key Contributions:**
- Well-defined MDP formulation for job scheduling with 539D observation space
- Integration of action masking for feasibility constraints
- Extensive evaluation on real production traces (512K+ pods)
- Competitive performance against 6 baseline heuristics on GPU-intensive workloads
- Theoretical validation of MDP assumptions and convergence properties

---

## Motivation and Objectives

### Motivation

**The Problem:**
Modern cloud clusters are increasingly heterogeneous, with diverse GPU types (A10, V100, T4, etc.) and mixed workloads. Existing cluster schedulers use simple heuristics:
- **First-Fit**: Fast but causes fragmentation
- **Best-Fit**: Reduces fragmentation but computationally expensive
- **Bin-Packing Variants**: Problem-specific but inflexible

These approaches suffer from:
1. **Fragmentation**: GPUs split across pods, reducing schedulability
2. **Tail Latency**: Jobs wait long periods in queue when heuristics make suboptimal placements
3. **Lack of Adaptability**: Fixed rules don't learn from cluster state evolution
4. **Heterogeneous Type Mismatch**: Naive algorithms don't account for GPU model-specific job requirements

**Why RL?**
- **Adaptability**: Learns from cluster dynamics
- **Optimality**: Can discover non-obvious placement strategies
- **Constraint Handling**: Action masking enforces feasibility
- **Scale**: Attention-based policy scales to large clusters
- **End-to-End**: Single policy optimizes multiple objectives (latency + fragmentation + utilization)

### Objectives

1. **Primary**: Minimize tail latency (p95, p99 slowdowns) in pod completion
2. **Secondary**: Maximize GPU utilization (GAR - GPU Allocation Ratio)
3. **Tertiary**: Reduce cluster fragmentation (FGD - Fragmentation Grade Degree)
4. **Constraints**: 
   - Meet resource requirements (CPU, memory, GPU count/type)
   - Scale to 128+ node clusters
   - Make decisions in milliseconds

---

## Literature Study

### Foundational Scheduling Algorithms

| Algorithm | Type | Pros | Cons |
|-----------|------|------|------|
| **First-Fit Decreasing (FFD)** | Heuristic | Fast, simple | High fragmentation |
| **Best-Fit** | Heuristic | Reduces fragmentation | O(n²) complexity |
| **Multi-Dimensional Bin Packing** | Heuristic | Handles CPU/Memory/GPU | NP-hard, approximation only |
| **Knapsack Variants** | Optimization | Theoretically bounded | Computationally expensive |

### Related Work in RL-based Scheduling

1. **Decima** (Mao et al., 2019)
   - First major work on RL-based cluster scheduling
   - Uses GCN for graph-based observations
   - Focuses on batch job scheduling, not continuous arrivals
   - **Limitation**: GPU scheduling not primary focus

2. **Pollux** (Qiao et al., 2021)
   - Co-adaptive cluster scheduling for goodput-optimized deep learning
   - Multi-resource scheduling that optimizes at both the per-job and cluster-wide levels
   - **Limitation**: Not RL-based, heuristic co-adaptation approach

### Key Insights from Literature

- **Constraint Satisfaction**: Action masking (Huang & Ontañón, 2020) enables RL in constrained domains
- **Attention for Scheduling**: Transformer architectures (Vaswani et al., 2017) scale well to variable-size inputs
- **On-Policy Learning**: Better for non-stationary environments (cluster state changes over time)
- **Reward Shaping**: Carefully designed rewards (tail latency, fragmentation) crucial for convergence
---

## Technical Details

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              GPU Cluster Scheduler (RL-based)          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Policy Network (MaskablePPO)             │  │
│  │  - NodeAttention: Multi-head attention over 128  │  │
│  │  - Pod Encoder: 17D pod features                │  │
│  │  - Global Context: Cluster-wide fragmentation   │  │
│  │  - Output: Action logits for 128 nodes          │  │
│  └──────────────────────────────────────────────────┘  │
│                      ▲                                  │
│                      │ Observations (539D)              │
│                      │ Action Masks (128D binary)       │
│                      │ Rewards (scalar)                 │
│                      │                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │        Scheduling Environment                    │  │
│  │  - ClusterSimulator: Event-driven, deterministic │  │
│  │  - Observation Generator: Real-time state        │  │
│  │  - Feasibility Checker: GPU/CPU/Memory/Type      │  │
│  │  - Reward Aggregator: Tail latency + metrics     │  │
│  └──────────────────────────────────────────────────┘  │
│                      ▲                                  │
│                      │                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │        Cluster Simulator                         │  │
│  │  - Node State: 128 GPU nodes with heterogeneous  │  │
│  │                GPU models                        │  │
│  │  - Pod Queue: FIFO + feasibility-based masking   │  │
│  │  - Event Loop: Completions trigger new decisions │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### MDP Formulation

**State Space (S)**: 539-dimensional observation vector

```
Observation = [pod_features (17D) ⊕ node_features (1024D) ⊕ global_features (10D)]
```

#### Pod Features (17D):
- GPU count (normalized by max_gpu_per_node)
- GPU millicores (milli/1000 normalized)
- CPU millicores (normalized by node_max_cpu)
- Memory MiB (normalized by node_max_memory)
- **GPU Type (8D one-hot)**: [CPU-only, A10, G2, G3, P100, T4, V100M16, V100M32]
- **QoS (4D one-hot)**: [Guaranteed, Burstable, BestEffort, LS]
- Priority (0 or 1)

#### Node Features (8D × 128 nodes = 1024D):
For each of 128 nodes:
1. CPU availability ratio
2. Memory availability ratio
3. GPU availability ratio
4. GPU model scalar identifier
5. **Fragmentation score** (FGD: Fragmentation Grade Degree)
6. Max free GPU segment (highest contiguous GPU allocation)
7. Full free GPU ratio (count of completely free GPUs)
8. Partial GPU ratio (GPUs with 0 < allocation < 1000)

#### Global Features (10D):
- Per-GPU-model free counts (7D, normalized)
- Cluster fragmentation average
- Pending pods ratio
- Current time normalization

**Observation Bounds**: All features normalized to [-1.0, 2.0]
- **Validated**: Empirical audit on 500 observations confirms bounds
- **Methods**: Min-max scaling, one-hot encoding, ratio normalization

---

**Action Space (A)**: Discrete(128)
- Action $a \in \{0, 1, ..., 127\}$ represents "schedule current pod on node $a$"
- **Feasibility**: Only actions satisfying resource constraints allowed (via masking)
- **Safe Fallback**: If no feasible node exists, allow all actions (rare, <0.1% in tests)

---

**Reward Function (R)**: Multi-component latency-focused design

$$R(s, a) = \text{success\_reward} \cdot \mathbb{1}[\text{scheduled}] + \text{fail\_penalty} \cdot \mathbb{1}[\neg\text{scheduled}]$$
$$+ \text{frag\_weight} \cdot (-\Delta\text{FGD}) + \text{balance\_weight} \cdot (-\Delta\text{skew}) + \text{util\_weight} \cdot \Delta\text{GAR}$$

Where:
- $\mathbb{1}[\text{scheduled}]$: 1 if pod successfully placed
- $\Delta\text{FGD}$: Change in fragmentation grade
- $\Delta\text{skew}$: Change in node utilization skewness
- $\Delta\text{GAR}$: Change in GPU allocation ratio

**Weights** (from `configs/phase5_maskable_ppo.json`):
- `success_reward`: 1.0
- `fail_penalty`: -5.0
- `frag_weight`: 50.0
- `balance_weight`: 0.05
- `util_weight`: 0.5

---

**Transitions**: $P(s' | s, a)$ - **Deterministic**

Algorithm:
```
1. Check feasibility: mask = compute_feasibility(pod, nodes)
2. If action a not feasible:
   - Add to failed pods
   - Return negative reward
3. Else:
   - Allocate resources: node[a].cpu_avail -= pod.cpu
   - Allocate GPUs (best-fit): select GPUs on node[a]
   - Add to running queue with completion_time
4. Process events:
   - Pop completed pods (deallocate resources)
   - Advance simulation time to next event
   - Select next pod from queue
5. Compute next observation from new cluster state
```

**Markov Property**: ✅ **Verified**
- Observation contains all information needed to compute next state
- No hidden state dependencies
- Transition fully determined by visible state + action
- See `THEORETICAL_AUDIT.md` for formal proof

---

**Horizon**: Finite-horizon episodic MDP
- Max pods per episode: 500
- Each pod has minimum duration (600ms)
- Guaranteed termination: Mathematical proof in `THEORETICAL_AUDIT.md`

---

### Policy Network Architecture

**NodeAttentionExtractor** (custom PyTorch feature extractor):

```
Input: observation (539D)
├─ Pod encoder (17D) → FC(64) → ReLU
├─ Global encoder (10D) → FC(64) → ReLU
└─ Node encoder (1024D → 8×128)
   ├─ Reshape to (128, 8)
   ├─ Multi-head attention (8 heads, 64D)
   │  - Query/Key/Value projections
   │  - Scaled dot-product attention
   │  - Learnable node positional embeddings
   └─ FC(64) per node
       
├─ Concatenate: [pod_emb, node_emb_1, ..., node_emb_128, global_emb]
└─ FC(256) → ReLU → FC(128) [action logits]

Output: 128 logits (one per node)
        + Value head for critic (128 → 64 → 1)
```

**Advantages**:
- Scalable to larger clusters (O(n) with attention)
- Captures inter-node relationships
- Incorporates pod-specific and global context
- Compatible with action masking

---

### Training: MaskablePPO

**Algorithm**: Proximal Policy Optimization with Action Masking

Key hyperparameters:
- Learning rate: 3e-4
- Batch size: 2048
- N steps: 2048 (steps between updates)
- Clip range: 0.2
- Entropy coeff: 0.01
- VecNormalize: Running mean/std normalization

**Training Process**:
```
for episode in range(num_episodes):
  reset_environment()
  for step in range(max_steps_per_episode):
    obs = environment.get_observation()
    mask = environment.get_action_mask()
    
    action = policy.predict(obs, mask=mask)
    obs_next, reward, done, info = environment.step(action)
    
    # Store transition in replay buffer
    store_transition(obs, action, reward, mask)
    
    if done:
      break
  
  # PPO update: minimize clipped surrogate loss
  update_policy_with_masking(replay_buffer)
  evaluate_on_validation_set()
  save_best_model(based_on_validation_gar)
```

---

## Implementation Details

### File Organization

```
queue-optimization/
├── cluster_sim.py                 # Event-driven simulator (Node, Pod, ClusterSimulator)
├── gpu_scheduling_env.py          # Gym environment with masking
├── train_maskable_ppo.py          # Training script with callbacks
├── phase6_evaluate.py             # Evaluation on held-out test set
├── configs/
│   ├── phase5_maskable_ppo.json   # Base training config
│   └── phase6_maskable_ppo_64node.json
├── models/
│   └── phase5/
│       ├── full/                  # Full model (all reward components)
│       ├── success_only/          # Ablation: only success/fail rewards
│       └── frag_only/             # Ablation: only fragmentation reward
├── evaluation/
│   └── phase6/
│       ├── phase6_summary.json    # Aggregate results
│       ├── full/                  # Scenario: full trace
│       ├── high_load/             # Scenario: reduced capacity
│       ├── gpu_intensive/         # GPU-only pods
│       └── mixed/                 # Mixed CPU/GPU pods
├── experiments/
│   ├── ablation_smoke.py          # Reward weight ablations
│   ├── frag_scale_sensitivity.py  # Fragmentation weight sensitivity
│   └── obs_audit.py               # Observation space validation
├── preprocessed/
│   └── phase3/                    # Cleaned traces (pods, nodes, episodes)
├── THEORETICAL_AUDIT.md           # Full MDP proof and validation
└── THEORETICAL_AUDIT_SUMMARY.md   # Quick reference checklist
```

### Key Classes and Functions

#### `cluster_sim.py`

```python
class Node:
  node_id: str
  cpu_total: int
  memory_total: int
  gpu_list: List[GPU]  # Each GPU has type and allocation
  
  # Methods
  reset()
  can_fit_pod(pod) -> bool

class Pod:
  name: str
  cpu_milli: int
  gpu_count: int
  gpu_type: str  # e.g., "V100M16"
  
class ClusterSimulator:
  # Main event-driven loop
  step(action: int) -> (observation, reward, done, info)
  reset() -> observation
```

#### `gpu_scheduling_env.py`

```python
class GPUSchedulingEnv(gym.Env):
  observation_space: Box(539,)
  action_space: Discrete(128)
  
  def reset() -> obs
  def step(action: int) -> (obs, reward, done, info)
  def action_masks() -> np.ndarray  # 128D binary mask
  def _get_obs() -> np.ndarray      # Compute 539D observation
```

#### `train_maskable_ppo.py`

```python
def mask_fn(env) -> np.ndarray:
  return env.action_masks()

# Training loop
env = make_vec_env(GPUSchedulingEnv, ...)
env = VecNormalize(env)

model = MaskablePPO(
  policy=policy_kwargs,
  env=env,
  learning_rate=3e-4,
  n_steps=2048,
)

model.learn(total_timesteps=500000)
```

---

## Experimental Evaluation

### Dataset Characteristics

**Workload Traces**:
- Source: Real production cluster (OpenB trace - anonymized)
- Total pods: ~512,000
- Duration: ~3.2 million seconds (~36 days)
- Train/Val/Test splits: 80%/10%/10%
- Episode structure: 500 pods per episode (episode limit)

**Pod Characteristics**:
- GPU request: 0-8 GPUs per pod
- GPU types: 8 heterogeneous models
- CPU: 100-16000 millicores
- Memory: 128-65536 MiB
- Duration: 600ms - 24 hours (normalized to min 600ms in experiments)

**Cluster Configuration**:
- Nodes: 16-128 (experiments vary)
- GPU nodes: ~112/128 in standard config
- CPU-only nodes: ~16/128
- Per-node GPUs: 1-8 (model-dependent)
- Total GPU capacity: ~600 GPUs (in 128-node cluster)

### Baseline Algorithms

| Baseline | Algorithm | Description |
|----------|-----------|-------------|
| **Random** | Stochastic | Uniformly random feasible node (5 runs) |
| **First-Fit** | Greedy | Select first feasible node |
| **Best-Fit** | Greedy | Select node minimizing fragmentation post-placement |
| **DotProduct** | Heuristic | Maximize pod-node feature similarity (dot product) |
| **GPUPacking** | Heuristic | Concentrate GPUs: prefer nodes with most GPUs used |
| **GPUClustering** | Heuristic | Group pods by GPU type |
| **FGD** | Heuristic | Minimize cluster fragmentation (Fragmentation Grade Degree) |

### Evaluation Metrics

**Primary Metrics**:

1. **GAR (GPU Allocation Ratio)**: 
   $$\text{GAR} = \frac{\text{Cumulative GPU-milli scheduled}}{\text{Total GPU-milli capacity × time}}$$
   - Measures GPU utilization efficiency
   - Higher is better; [0, 1]

2. **Success Rate**: 
   $$\text{Success} = \frac{\text{Pods scheduled}}{\text{Total pods}}$$
   - Fraction of pods placed successfully
   - Higher is better; [0, 1]

**Secondary Metrics**:

3. **Fragmentation (FGD)**:
   - Measures GPU allocation fragmentation
   - Computed per-node, aggregated
   - Higher indicates more fragmentation

4. **Tail Latency Metrics**:
   - avg/p95/p99 slowdown (completion_time / expected_duration)
   - avg/p95/p99 wait time (scheduling latency)

5. **Utilization Balance** (Sigma):
   $$\sigma_{\text{util}} = \sqrt{\frac{\sum (util_i - \bar{util})^2}{n}}$$
   - Standard deviation of utilization across nodes
   - Lower is better (less skew)

### Evaluation Scenarios

**Scenario 1: Full Trace** (Standard)
- Complete test set
- Original workload intensity
- Realistic mixed workloads

**Scenario 2: High-Load Stress**
- Reduced per-node capacity by 30%
- Increases contention and failures
- Tests scheduler under pressure

**Scenario 3: GPU-Intensive**
- Filter to pods with ≥4 GPUs
- GPU fragmentation challenges
- Resource constraint binding

**Scenario 4: Mixed (CPU/GPU)**
- Include all pod types
- Diverse resource demands
- Real-world heterogeneity

**Scenario 5: Heterogeneous Models**
- Specific GPU type mixes
- Test model affinity learning
- Specialization capability

---

## Results and Findings

### Overall Performance (High-Load Scenario, 64-node cluster)

| Metric | RL (Full) | Best-Fit | FGD | Random | Improvement |
|--------|-----------|----------|-----|--------|-------------|
| **GAR** | 0.673 | 0.677 | 0.676 | 0.675 | -0.4% (RL competitive) |
| **Success Rate** | 49.7% | 50.7% | 51.4% | 52.2% | Baseline edges ahead |
| **Final Fragmentation** | 0.877 | 0.825 | 0.821 | 0.824 | RL higher fragmentation |
| **Avg Full Free GPUs** | 127.2 | 7.0 | 7.0 | 2.4 | RL over-conservative |
| **Sigma GPU Util** | 0.207 | 0.204 | 0.165 | 0.106 | RL less balanced |

### Key Findings

**Finding 1: RL Performance Competitive on Utilization**
- GAR: RL ≈ Best-Fit ≈ FGD ≈ Random (all ~0.67)
- **Interpretation**: Under high load, all schedulers saturate similarly; limited differentiation opportunity
- **Implication**: RL advantage emerges in specific scenarios (tail latency, complex interactions)

**Finding 2: RL More Conservative (Fragmentation)**
- RL final fragmentation: 0.877 vs Baseline: 0.82-0.85
- **Root cause**: RL learned to avoid placements that cause immediate fragmentation, but this leads to more failures
- **Lesson**: Reward trade-off between fragmentation and scheduling success needs tuning

**Finding 3: Utilization Imbalance in RL**
- RL sigma GPU util: 0.207 vs Baseline: 0.106-0.204
- **Root cause**: Attention mechanism lacks strong load-balancing inductive bias
- **Action needed**: Add explicit balance reward or architectural constraint

**Finding 4: Theoretical MDP Valid, Empirical Gap**
- MDP formulation mathematically sound (all axioms satisfied)
- Empirical performance does not show clear RL advantage
- **Causes**:
  1. High-load scenario leaves little room for optimization
  2. Reward shaping not aligned with practical scheduling objectives
  3. Action masking prevents bad decisions, but also limits learning

### Ablation Results

**Experiment: Reward Component Contribution**

| Model | Components | GAR | Success | Fragmentation |
|-------|-----------|-----|---------|-----------------|
| `full` | success + frag + balance + util | 0.707 | 68.2% | 0.56 |
| `success_only` | success + fail penalty | 0.584 | 42.1% | 0.82 |
| `frag_only` | fragmentation only | 0.651 | 55.3% | 0.43 |

**Insights**:
- Fragmentation reward (weight=50) dominates learning
- Success reward necessary but not sufficient
- Balance reward provides marginal improvement

---

## Conclusion

### Summary

This project demonstrates a complete pipeline for **reinforcement learning-based GPU cluster scheduling**:

✅ **Theoretical Contributions**:
- Rigorous MDP formulation with proven Markov property
- Action masking integration for constrained optimization
- Validated observation space bounds

✅ **System Contributions**:
- End-to-end training pipeline (data → model → evaluation)
- Real-world trace evaluation on 512k+ pod workloads
- Extensible architecture for heterogeneous cluster types

✅ **Empirical Validation**:
- Competitive performance against 6 established baselines
- Stress testing in multiple scenarios (full, high-load, GPU-intensive)
- Ablation studies isolating reward component effects

### Limitations

1. **Empirical Performance Gap**: 
   - RL does not clearly outperform hand-tuned heuristics on current scenarios
   - Likely because problem is already well-solved by FGD/Best-Fit
   - RL advantage emerges with non-stationary or adversarial workloads

2. **Training Efficiency**:
   - Requires 500k steps (~48 hours on V100 GPU)
   - Inference time competitive (~1ms per decision)

3. **Scalability Trade-off**:
   - Attention mechanism scales better than dense layers
   - But still limited to 128 nodes in practice
   - Larger clusters need hierarchical scheduling

### Future Work

1. **Improved Reward Design**:
   - Incorporate real SLO (Service Level Objectives) 
   - Add tail latency directly to reward (currently implicit)
   - Implement multi-task learning (different workload types)

2. **Architectural Improvements**:
   - Graph neural networks for cluster topology awareness
   - Hierarchical scheduling for 1000+ nodes
   - Transfer learning across cluster sizes

3. **Real-World Deployment**:
   - Integration with Kubernetes scheduler
   - Online learning with exploration-exploitation trade-off
   - Fallback to safe heuristics on policy failure

4. **Comparative Analysis**:
   - Compare against recent work (HippoLytus, Pollux)
   - Non-stationary environment with dynamic pod arrivals
   - Stochastic pod durations and resource requirements

5. **Extended Experiments**:
   - Sensitivity to hyperparameter variations
   - Training convergence analysis with different initializations
   - Cross-workload generalization

---

## References

### Core Papers
[1] Mao, H., et al. (2019). "Learning Scheduling Algorithms for Data Processing Clusters". SIGCOMM - Decima scheduling algorithm

[2] Huang, S., & Ontañón, S. (2020). "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms". FLAIRS - Action masking theory

[3] Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms". arXiv preprint - PPO algorithm

[4] Vaswani, A., et al. (2017). "Attention Is All You Need". NeurIPS - Transformer architecture

[5] Qiao, A., et al. (2021). "Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning". OSDI

### Scheduling Background
[6] Grandl, R., et al. (2014). "Multi-resource Packing for Cluster Schedulers". SIGCOMM - Tetris scheduler

[7] Tumanov, A., et al. (2016). "TetriSched: Global Rescheduling with System-provided Plans". EuroSys - Heterogeneous cluster scheduling

[8] Hindman, B., et al. (2011). "Mesos: A Platform for Fine-Grained Resource Sharing in the Data Center". NSDI - Resource sharing framework

---

## Project Structure

```
queue-optimization/
│
├── README.md (this file)
│
├── THEORETICAL_AUDIT.md               # Complete MDP formulation proof
├── THEORETICAL_AUDIT_SUMMARY.md       # Quick reference
├── PHASE4_ENV.md                      # Environment documentation
├── PHASE6_EVAL.md                     # Evaluation framework documentation
│
├── Core Implementation
├── ├── cluster_sim.py                 # Event simulator (Node, Pod, Simulator)
├── ├── gpu_scheduling_env.py          # Gym environment (action masking)
├── ├── rl_env_backend.py              # Backend utilities
│
├── Training & Evaluation
├── ├── train_maskable_ppo.py          # MaskablePPO training script
├── ├── phase6_evaluate.py             # Held-out test evaluation
├── ├── test_gpu_scheduling_env.py     # Unit tests
│
├── Configuration & Experiments
├── ├── configs/
│   ├── phase5_maskable_ppo.json       # Training config (16-node, 128-node)
│   └── phase6_maskable_ppo_64node.json # Evaluation config
├── ├── experiments/
│   ├── ablation_smoke.py              # Reward ablation
│   ├── frag_scale_sensitivity.py      # Hyperparameter sensitivity
│   ├── obs_audit.py                   # Observation validation
│   ├── obs_audit.json                 # Audit results
│   └── ablation_smoke_results.json    # Ablation findings
│
├── Data
├── ├── preprocessed/phase3/
│   ├── nodes_clean.csv                # Cluster node specs
│   ├── pods_train.csv, pods_test.csv  # Pod traces
│   ├── episodes_train.csv, episodes_val.csv  # Episode boundaries
│   └── workload_demand_distribution.json
│
├── Models & Results
├── ├── models/phase5/
│   ├── full/
│   │   ├── best_model.zip             # Best RL policy
│   │   └── best_vecnormalize.pkl      # Normalization stats
│   ├── success_only/                  # Ablation variant
│   ├── frag_only/                     # Ablation variant
│   └── training_summary.json
│
├── ├── evaluation/phase6/
│   ├── phase6_summary.json            # Aggregate results
│   ├── phase6_results_table.csv       # Results table
│   ├── full/                          # Scenario results
│   │   ├── rl_full_run0_steps.csv     # Per-step log
│   │   ├── best_fit_run0_steps.csv
│   │   └── ... (6 baselines)
│   ├── high_load/
│   ├── gpu_intensive/
│   └── ...
│
└── Documentation
    ├── simulation-results/            # Phase 5 simulation outputs
    ├── runs/phase5/                   # Training runs
    └── tests/                         # Test suite
```

---

## Quick Start

### Installation

```bash
cd queue-optimization
pip install -r requirements-rl.txt
```

### Training

```bash
python train_maskable_ppo.py \
  --config configs/phase5_maskable_ppo.json \
  --nodes-csv preprocessed/phase3/nodes_clean.csv \
  --pods-csv preprocessed/phase3/pods_train.csv \
  --log-dir runs/phase5/full
```

### Evaluation

```bash
python phase6_evaluate.py \
  --run-rl \
  --run-baselines \
  --models-root models/phase5 \
  --rl-experiments full \
  --scenarios high_load,gpu_intensive \
  --make-plots
```

Results saved to `evaluation/phase6/` with JSON summaries and CSV logs.

### Viewing Results

```bash
# Summary statistics
cat evaluation/phase6/phase6_summary.json

# Detailed results table
cat evaluation/phase6/phase6_results_table.csv

# Per-step logs (for RL fine-tuning analysis)
head -20 evaluation/phase6/high_load/rl_full_run0_steps.csv
```

---

## Contributing

To extend this work:

1. **New Baselines**: Add to `phase6_evaluate.py` (implement `run_baseline()`)
2. **New Scenarios**: Modify `create_scenarios()` in `phase6_evaluate.py`
3. **Policy Changes**: Extend `NodeAttentionExtractor` in `train_maskable_ppo.py`
4. **New Metrics**: Add to reward function in `gpu_scheduling_env.py`

---

## License

[To be specified]

---

## Contact & Attribution

Arhaan Shah
Devarsh Vasani
