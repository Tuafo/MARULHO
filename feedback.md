This will be long, technical, and dense. Every section is a complete solution, not a pointer.

---

# HECSN — Complete Technical Fix & Path Forward

## Governing Principle Before Anything Else

Every fix below serves one test: **can the system, given only a raw byte stream, form stable internal representations that correlate with semantic structure, without any external label at any stage?** If a component can't be evaluated against that test, it doesn't belong in the architecture. Hold every solution below to that standard.

---

## PROBLEM 1 — The Semantic Gap: Character → Concept Has No Principled Bridge

### The Real Problem

You're asking character statistics to spontaneously organize into semantic concepts. They won't. Not because emergence is impossible, but because the signal-to-meaning gap at character level requires a grounded intermediate representation. The brain doesn't go from photons → concepts — it goes photons → edges → objects → categories → concepts, with feedback at every level. Your architecture skips the entire middle.

The deeper issue: your "tabula rasa" framing is wrong. The brain is not a tabula rasa at any level below semantics. It has hardwired feature detectors, topographic maps, and columnar organization before any experience. HECSN needs its own equivalent: not semantic priors, but **structural priors about what constitutes a learnable unit.**

### The Complete Solution: Grounded Intermediate Representations via Multi-Scale Statistical Chunking

**Layer 0: A Learned Chunking Layer (replace heuristic segmentation entirely)**

The idea: patterns that co-occur reliably across the stream are more "chunk-worthy" than patterns that don't. This is Predictability-Based Chunking, implemented as an SNN layer before the Competitive Layer.

```python
class ChunkingLayer:
    """
    Learns variable-length units from raw byte stream via spike correlation.
    No fixed tokenization. No vocabulary. No regex.
    
    Mechanism: 
      - Runs a bank of "syllable detector" neurons, each tuned to a 
        different n-gram length (2–8 characters).
      - A chunk boundary is declared when prediction error RISES 
        (the current pattern is less predictable given the chunk so far)
        AND correlation between active neurons DROPS below threshold.
      - Output: variable-length spike burst representing the chunk.
    
    This is a learned segmentation, not a heuristic overlay.
    """
    
    def __init__(self, n_detectors: int = 512, 
                 max_chunk_len: int = 12,
                 boundary_threshold: float = 0.3,
                 device: str = 'cuda'):
        self.n = n_detectors
        self.max_len = max_chunk_len
        self.device = device
        
        # Each detector learns a byte-pattern prototype
        # These are LEARNED, not pre-set
        self.prototypes = torch.randn(n_detectors, max_chunk_len * 8,
                                      device=device)  # 8 bits per byte
        self.prototypes = F.normalize(self.prototypes, dim=1)
        
        # Prediction confidence per detector (learned via STDP)
        self.confidence = torch.ones(n_detectors, device=device) * 0.5
        
        # Current chunk accumulator
        self.chunk_buffer: list[int] = []
        self.chunk_spikes: list[torch.Tensor] = []
        self.prev_correlation: float = 1.0
        
    def byte_to_spikes(self, byte_val: int) -> torch.Tensor:
        """Convert single byte to 8-dimensional spike pattern (bit encoding)."""
        bits = torch.zeros(8, device=self.device)
        for i in range(8):
            bits[i] = float((byte_val >> i) & 1)
        return bits
    
    def encode_chunk(self, chunk_bytes: list[int]) -> torch.Tensor:
        """
        Encode chunk as fixed-dim spike pattern via overlap pooling.
        Preserves ordering information via positional phase offset.
        """
        dim = self.max_len * 8
        out = torch.zeros(dim, device=self.device)
        for pos, byte_val in enumerate(chunk_bytes[:self.max_len]):
            bits = self.byte_to_spikes(byte_val)
            # Positional phase: earlier bytes have stronger early-phase spikes
            phase_weight = torch.exp(torch.tensor(-pos * 0.15, 
                                                   device=self.device))
            start = pos * 8
            out[start:start+8] = bits * phase_weight
        return F.normalize(out, dim=0)
    
    def compute_boundary_signal(self, new_byte: int) -> tuple[bool, float]:
        """
        Returns (is_boundary, boundary_confidence).
        Boundary declared when adding new_byte reduces detector agreement.
        """
        if len(self.chunk_buffer) == 0:
            return False, 0.0
            
        # Encode current chunk + new byte
        extended = self.chunk_buffer + [new_byte]
        extended_enc = self.encode_chunk(extended)
        
        # How well does this extended chunk match any detector?
        similarities = torch.mv(self.prototypes, extended_enc)
        max_sim = similarities.max().item()
        
        # Encode current chunk without new byte
        current_enc = self.encode_chunk(self.chunk_buffer)
        current_sims = torch.mv(self.prototypes, current_enc)
        current_max = current_sims.max().item()
        
        # Boundary signal: adding new byte hurts predictability significantly
        predictability_drop = current_max - max_sim
        is_boundary = (predictability_drop > self.boundary_threshold 
                       or len(self.chunk_buffer) >= self.max_len)
        
        return is_boundary, predictability_drop
    
    def process_byte(self, byte_val: int) -> tuple[torch.Tensor | None, bool]:
        """
        Feed one byte. Returns (chunk_encoding, is_new_chunk).
        chunk_encoding is None if no chunk completed this step.
        """
        is_boundary, _ = self.compute_boundary_signal(byte_val)
        
        if is_boundary and len(self.chunk_buffer) > 0:
            # Emit current chunk as spike pattern
            chunk_enc = self.encode_chunk(self.chunk_buffer)
            
            # Update detector prototypes via competitive learning
            # (the chunk that just completed updates the nearest detector)
            sims = torch.mv(self.prototypes, chunk_enc)
            winner_idx = sims.argmax()
            lr = 0.01 * (1.0 - self.confidence[winner_idx].item())
            self.prototypes[winner_idx] = F.normalize(
                self.prototypes[winner_idx] + lr * (chunk_enc - 
                                                    self.prototypes[winner_idx]),
                dim=0
            )
            self.confidence[winner_idx] = (0.99 * self.confidence[winner_idx] 
                                           + 0.01 * sims[winner_idx])
            
            # Start new chunk with current byte
            self.chunk_buffer = [byte_val]
            return chunk_enc, True
        else:
            self.chunk_buffer.append(byte_val)
            return None, False
```

**Why this works:** The chunking layer learns, from stream statistics alone, which byte sequences reliably co-occur. This is exactly how phoneme discovery works in infant language acquisition research (BLISS, Goldwater et al.). The chunks that emerge aren't morphemes per se, but they're statistically meaningful units — the right input for the Competitive Layer to build concepts from.

**What emerges:** After ~50K bytes, the chunking layer will have learned that `"tion"`, `"ing"`, `"the "`, `" of "` are reliable chunks, without being told what a word is. The Competitive Layer then builds assemblies over these discovered units, not over arbitrary characters.

**Grounding the semantic gap:** This doesn't fully close the semantic gap (no text-only system can), but it moves the input from character statistics to morpho-statistical units — a principled intermediate that makes the competitive layer's job tractable.

---

## PROBLEM 2 — Catastrophic Forgetting Is Solved by Disabling Consolidation: Fix the STC Pipeline

### The Real Problem

Your Late-LTP consolidation causes collapse because: replayed assemblies update prototypes even for well-consolidated memories, the consolidation cycle runs uniformly across all memories regardless of their fragility, and the PRP-to-capture mechanism doesn't distinguish between "strengthen this because it's weak" and "strengthen this because it won the replay lottery."

### The Complete Solution: Fragility-Gated Consolidation with Replay Compartmentalization

**Three changes to the STC pipeline that fix the collapse without zeroing it out:**

**Change 1: Fragility Score per Memory**

```python
@dataclass
class MemoryEntry:
    assembly: torch.Tensor          # The stored spike pattern
    importance: float               # Replay priority base
    capture_tag: float              # STC tag strength (0–1, decays)
    prp_local: float                # Local protein-related plasticity trace
    consolidation_level: float      # 0=fresh, 1=consolidated
    access_count: int               # How many times replayed
    tokens_since_last_replay: int   
    fragility: float                # KEY ADDITION: estimated overwrite risk
    
    def fragility_score(self, current_winner_distribution: dict) -> float:
        """
        Fragility = how likely is this assembly to be overwritten by 
        current learning?
        
        High fragility: winner column is currently being driven hard by 
        new inputs that differ from this assembly.
        Low fragility: winner column is stable or this assembly is in a 
        column that isn't currently active.
        """
        # Simplified: fragility is inverse of consolidation × importance
        base = 1.0 / (self.consolidation_level + 0.01)
        recency_penalty = 1.0 / (self.access_count + 1)
        return base * recency_penalty
```

**Change 2: Compartmentalized Replay — Separate "Maintenance" from "Consolidation"**

The key insight: maintenance replay (preventing overwrite of unconsolidated memories) and consolidation replay (converting Early-LTP to Late-LTP) are different operations that should never happen in the same pass.

```python
class FragilityGatedSleepReplay:
    """
    Separates sleep into three functionally distinct phases:
    
    Phase A (Micro-sleep, every 200 tokens):
      - MAINTENANCE ONLY
      - Replay high-fragility, low-consolidation memories
      - Updates: STDP eligibility only, no weight commit
      - Purpose: refresh capture tags before they decay
      
    Phase B (Scheduled deep sleep, every 5K tokens):
      - CONSOLIDATION of memories with tag × PRP > threshold
      - Updates: commit captured synapses to long-term weights
      - Gated: only memories with consolidation_level < 0.8 are eligible
      - Anchored: prototype momentum blocks large shifts
      
    Phase C (Emergency, drift-floor rising):
      - REPAIR only: no new consolidation
      - Replay highest-importance memories to re-anchor prototypes
      - No STDP weight updates — only prototype position restoration
    """
    
    def __init__(self, memory_store, config):
        self.mem = memory_store
        self.cfg = config
        
    def micro_sleep(self, columns: list, n_replay: int = 5):
        """
        Maintenance pass. Refresh, don't consolidate.
        """
        # Sort by fragility (highest risk first)
        candidates = sorted(
            self.mem.slow_buffer,
            key=lambda m: m.fragility_score({}),
            reverse=True
        )[:n_replay]
        
        for memory in candidates:
            if memory.consolidation_level >= 0.8:
                # Skip — already consolidated, low maintenance need
                continue
            
            # Replay through winner column ONLY (not all columns)
            target_col = self._find_nearest_column(memory.assembly, columns)
            
            # Soft replay: update eligibility trace but DON'T commit weights
            target_col.replay_eligibility_only(memory.assembly)
            
            # Refresh capture tag
            memory.capture_tag = min(1.0, memory.capture_tag + 0.05)
            memory.tokens_since_last_replay = 0
    
    def deep_sleep_consolidation(self, columns: list, hnsw_index):
        """
        Consolidation pass. Only runs on memories that have:
        - capture_tag > 0.3 (tag still active)
        - prp_local > 0.4 (enough protein signal)
        - consolidation_level < 0.8 (not already consolidated)
        """
        consolidation_candidates = [
            m for m in self.mem.slow_buffer
            if (m.capture_tag > 0.3 
                and m.prp_local > 0.4 
                and m.consolidation_level < 0.8)
        ]
        
        # CRITICAL: replay from high-importance to low, with momentum anchor
        consolidation_candidates.sort(key=lambda m: m.importance, reverse=True)
        
        for memory in consolidation_candidates:
            target_col = self._find_nearest_column(memory.assembly, columns)
            
            # Anchored consolidation: prototype moves at most anchor_lr per step
            anchor_lr = 0.001  # Much smaller than wake learning rate (~0.01)
            target_col.consolidate_assembly(
                memory.assembly, 
                strength=memory.capture_tag * memory.prp_local,
                anchor_lr=anchor_lr
            )
            
            # Advance consolidation level
            memory.consolidation_level = min(
                1.0,
                memory.consolidation_level + 0.1 * memory.capture_tag
            )
            
            # Decay tag (resource consumed)
            memory.capture_tag *= 0.7
        
        # After consolidation: structural plasticity and HNSW rebuild
        for col in columns:
            col.structural_plasticity_spike_correlation(
                prune_threshold=0.05,
                correlation_threshold=0.3
            )
        hnsw_index.rebuild()
    
    def emergency_repair(self, columns: list, top_n: int = 20):
        """
        Repair only — no new consolidation.
        Anchors drifting prototypes back toward their strongest memories.
        """
        top_memories = sorted(
            self.mem.slow_buffer,
            key=lambda m: m.importance,
            reverse=True
        )[:top_n]
        
        for memory in top_memories:
            target_col = self._find_nearest_column(memory.assembly, columns)
            # Hard anchor: restore prototype position toward stored assembly
            target_col.anchor_prototype(memory.assembly, strength=0.3)
            # No STDP, no weight changes, no tag changes
    
    def _find_nearest_column(self, assembly: torch.Tensor, 
                              columns: list):
        norms = torch.stack([col.prototype for col in columns])
        sims = F.cosine_similarity(assembly.unsqueeze(0), norms)
        return columns[sims.argmax().item()]
```

**Change 3: Consolidation Level Gates Wake Learning**

```python
# In the Competitive Layer's plasticity update:
def compute_plasticity_gate(self, winner_col, modulator: float) -> float:
    """
    Well-consolidated memories resist overwrite.
    Their column's plasticity is reduced proportionally.
    """
    if winner_col.consolidation_level > 0.7:
        # High consolidation: strong memories resist change
        # Only high-surprise events can modify them
        resistance = winner_col.consolidation_level
        effective_lr = self.base_lr * (1.0 - resistance * 0.8)
        return effective_lr * modulator
    else:
        return self.base_lr * modulator
```

**What this gives you:** Maintenance and consolidation are now separate, fragile memories get maintenance priority, well-consolidated memories resist overwrite during wake, and emergency repair doesn't trigger consolidation. You can now run with `consolidation_cycles > 0` safely because the anchor prevents collapse.

---

## PROBLEM 3 — The Neuromodulator Computation is Muddled: Full Replacement

### The Complete Solution: Independent Parallel Neuromodulatory Channels

Each neuromodulator modulates a *different aspect* of plasticity. They don't multiply together.

```python
class NeuromodulatorSystem:
    """
    Four independent channels, each targeting a different plasticity parameter.
    No multiplication of channels. No global reset.
    
    DA  → LTP/LTD magnitude scaling (reward prediction error)
    ACh → Learning rate gain (novelty / arousal gating)  
    NE  → Exploration noise amplitude (global uncertainty signal)
    5-HT → LTD rate bias (patience / time-horizon modulation)
    
    All channels operate independently. Their effects are ADDITIVE at the 
    plasticity site, not multiplicative.
    """
    
    def __init__(self, tau_da: float = 20.0,   # fast
                       tau_ach: float = 50.0,  # medium  
                       tau_ne: float = 200.0,  # slow
                       tau_5ht: float = 500.0, # very slow
                       baseline_da: float = 0.5,
                       baseline_ach: float = 0.3):
        # Each modulator: scalar concentration in [0, 1]
        self.da = baseline_da      # Dopamine
        self.ach = baseline_ach    # Acetylcholine
        self.ne = 0.2              # Norepinephrine
        self.serotonin = 0.5       # Serotonin
        
        # Independent baselines (predicted levels) for RPE computation
        self.da_baseline = baseline_da
        self.error_baseline = 0.5   # Predicted prediction error
        
        # Time constants (in tokens as functional time units)
        self.tau_da = tau_da
        self.tau_ach = tau_ach
        self.tau_ne = tau_ne
        self.tau_5ht = tau_5ht
        
    def update(self, 
               current_error: float,
               novelty: float,
               uncertainty: float,
               recent_reward_history: float) -> None:
        """
        Update all four channels from their natural inputs.
        Each channel has a specific, distinct input.
        """
        dt = 1.0  # one token
        
        # --- DOPAMINE: Reward Prediction Error ---
        # RPE = predicted_error - actual_error (positive = better than expected)
        rpe = (self.error_baseline - current_error) / (self.error_baseline + 1e-6)
        rpe = float(torch.tanh(torch.tensor(rpe * 2.0)))  # squash to (-1, 1)
        # DA decays toward baseline, pulsed by RPE
        self.da += dt * (-(self.da - self.da_baseline) / self.tau_da + 
                          0.5 * rpe)
        self.da = float(torch.clamp(torch.tensor(self.da), 0.0, 1.0))
        # Update error baseline (slow EMA — the "predicted error")
        self.error_baseline += (current_error - self.error_baseline) / 200.0
        
        # --- ACETYLCHOLINE: Novelty-Gated Learning Rate ---
        # ACh rises with novelty (genuine new patterns deserve higher lr)
        # and decays toward baseline
        self.ach += dt * (-(self.ach - 0.3) / self.tau_ach + 0.3 * novelty)
        self.ach = float(torch.clamp(torch.tensor(self.ach), 0.1, 1.0))
        
        # --- NOREPINEPHRINE: Exploration Noise Amplitude ---
        # NE rises with sustained uncertainty (not momentary surprise)
        # Sustained uncertainty = environment is stochastic, explore more
        self.ne += dt * (-(self.ne - 0.2) / self.tau_ne + 0.4 * uncertainty)
        self.ne = float(torch.clamp(torch.tensor(self.ne), 0.0, 1.0))
        
        # --- SEROTONIN: LTD Bias / Patience ---
        # Low serotonin = high LTD rate = faster forgetting of old patterns
        # High serotonin = conservative, preserves existing representations
        # Input: recent reward history (positive recent experience → higher 5-HT)
        self.serotonin += dt * (-(self.serotonin - 0.5) / self.tau_5ht + 
                                 0.2 * recent_reward_history)
        self.serotonin = float(torch.clamp(torch.tensor(self.serotonin), 0.0, 1.0))
    
    def get_plasticity_params(self) -> dict:
        """
        Returns independent plasticity modifications for each channel.
        These are ADDED to the base plasticity rule, not multiplied together.
        
        The weight update rule becomes:
          Δw = (base_lr × ltp_scale × pre_post_term)  ← DA scales LTP
             + (ach_gain × eligibility_trace)           ← ACh scales lr
             - (ltd_bias × post_pre_term)               ← 5-HT biases LTD
        
        Exploration noise is ADDED to membrane voltage, not to weights:
          V += NE_noise_amplitude × randn()
        """
        return {
            # DA scales the LTP term amplitude (not the whole rule)
            'ltp_scale': 0.5 + self.da,            # [0.5, 1.5]
            
            # 5-HT biases the LTD term (high 5-HT = weaker LTD = more stable)
            'ltd_bias': 1.5 - self.serotonin,       # [0.5, 1.5]
            
            # ACh multiplies effective learning rate (attention/gain)
            'lr_gain': self.ach,                    # [0.1, 1.0]
            
            # NE adds exploration noise to membrane voltages
            'exploration_noise': self.ne * 0.5,     # [0, 0.5] mV equivalent
        }
    
    def apply_to_stdp(self, 
                      dw_ltp: torch.Tensor,  # raw LTP term (pre before post)
                      dw_ltd: torch.Tensor,  # raw LTD term (post before pre)
                      base_lr: float) -> torch.Tensor:
        """
        Apply independent neuromodulatory modifications to STDP update.
        No multiplication of all channels. Clean separation.
        """
        params = self.get_plasticity_params()
        
        delta_w = (base_lr 
                   * params['lr_gain'] 
                   * (params['ltp_scale'] * dw_ltp 
                      - params['ltd_bias'] * dw_ltd))
        
        return delta_w
    
    def apply_exploration_noise(self, membrane_voltages: torch.Tensor) -> torch.Tensor:
        """
        NE adds exploration noise to membrane voltages, not weights.
        This makes under-active neurons more likely to spike (exploration)
        without corrupting synaptic structure.
        """
        params = self.get_plasticity_params()
        if params['exploration_noise'] > 0.01:
            noise = torch.randn_like(membrane_voltages) * params['exploration_noise']
            return membrane_voltages + noise
        return membrane_voltages
```

**Note on removing `should_reset_network()`:** Delete it entirely. Replace it with: when NE > 0.7 for >500 consecutive tokens, increase `exploration_noise` by 2x and trigger a curiosity query via the gap detection pipeline. High sustained NE means the environment is genuinely stochastic — the response is more exploration, not a network reset.

---

## PROBLEM 4 — HNSW CPU Bottleneck Will Kill Scalability: Full GPU-Native Routing

### The Complete Solution: Two-Phase GPU Routing Without CPU Round-Trip

The key insight: you don't need exact nearest-neighbor search on every token. You need *good enough* routing that stays on-GPU, and you save exact HNSW for sleep-phase index maintenance.

```python
class GPUNativeRouter:
    """
    Fully GPU-resident competitive routing.
    No CPU round-trip per token.
    
    Architecture:
      Phase 1 (per token): GPU cosine similarity over learned projection 
                           → top-k candidates in O(n) on GPU
      Phase 2 (per sleep): HNSW rebuild on CPU for graph consistency check
                           (this can be async, doesn't block the stream)
    
    At 10K columns: O(n) GPU matmul is faster than HNSW CPU call + PCIe.
    At 100K columns: use IVF (inverted file index) partitioning on GPU.
    At 1M+ columns: distributed GPU shards.
    """
    
    def __init__(self, n_columns: int, prototype_dim: int,
                 k_candidates: int = 32,
                 device: str = 'cuda',
                 ivf_n_cells: int = None):  # None = flat, int = IVF
        self.n_cols = n_columns
        self.dim = prototype_dim
        self.k = k_candidates
        self.device = device
        self.use_ivf = ivf_n_cells is not None
        
        # ALL prototypes live on GPU at all times
        # Shape: [n_columns, prototype_dim]
        self.prototypes = torch.randn(n_columns, prototype_dim, 
                                       device=device)
        self.prototypes = F.normalize(self.prototypes, dim=1)
        
        if self.use_ivf:
            # For 100K+ columns: IVF partitioning
            # Assign each column to a Voronoi cell
            n_cells = ivf_n_cells  # e.g. sqrt(n_columns) ≈ 316 for 100K
            self.cell_centroids = torch.randn(n_cells, prototype_dim,
                                               device=device)
            self.cell_centroids = F.normalize(self.cell_centroids, dim=1)
            self.column_to_cell = torch.zeros(n_columns, dtype=torch.long,
                                               device=device)
            self._assign_columns_to_cells()
        
        # Winner history for refractory mechanism (prevent monopoly)
        self.winner_history = torch.zeros(n_columns, dtype=torch.long,
                                           device=device)
        self.refractory_penalty_strength = 0.3
        self.history_decay = 0.995
    
    def route(self, 
              pattern: torch.Tensor,         # [prototype_dim]
              context_gain: torch.Tensor,    # [n_columns] from Context Layer
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-native routing. Returns (winner_idx, top_k_indices).
        No CPU transfer. No blocking.
        
        At 10K columns on A100: ~0.05ms (vs ~2ms for CPU HNSW + PCIe).
        """
        # Normalize input
        pattern_norm = F.normalize(pattern.unsqueeze(0), dim=1)  # [1, dim]
        
        if self.use_ivf and self.n_cols > 50_000:
            # IVF: find nearest cell first, then search within cell
            cell_sims = torch.mv(self.cell_centroids, pattern_norm.squeeze())
            top_cells = cell_sims.topk(min(8, len(self.cell_centroids))).indices
            
            # Get columns in those cells
            cell_mask = torch.isin(self.column_to_cell, top_cells)
            candidate_indices = cell_mask.nonzero(as_tuple=True)[0]
            candidate_protos = self.prototypes[candidate_indices]
        else:
            # Flat: full cosine similarity (fast enough up to ~50K columns)
            candidate_indices = torch.arange(self.n_cols, device=self.device)
            candidate_protos = self.prototypes
        
        # Cosine similarity: [n_candidates]
        sims = torch.mv(candidate_protos, pattern_norm.squeeze())
        
        # Apply context gain (Context Layer modulates competition)
        if self.use_ivf:
            context_gain_local = context_gain[candidate_indices]
        else:
            context_gain_local = context_gain
        
        sims = sims * (0.5 + context_gain_local)
        
        # Apply refractory penalty (prevent monopoly winners)
        if self.use_ivf:
            history_local = self.winner_history[candidate_indices].float()
        else:
            history_local = self.winner_history.float()
            
        refractory = self.refractory_penalty_strength * history_local
        sims = sims - refractory
        
        # Top-k selection (still on GPU)
        k_actual = min(self.k, len(candidate_indices))
        top_k_local = sims.topk(k_actual).indices
        top_k_global = candidate_indices[top_k_local]
        
        # Winner = argmax of top-k (WTA)
        winner_local = top_k_local[sims[top_k_local].argmax()]
        winner_global = candidate_indices[winner_local]
        
        # Update winner history (decay + increment winner)
        self.winner_history = (self.winner_history * self.history_decay).long()
        self.winner_history[winner_global] += 1
        
        return winner_global, top_k_global
    
    def update_prototype(self, 
                          col_idx: int,
                          pattern: torch.Tensor,
                          lr: float) -> None:
        """
        Kohonen/SOM update: prototype moves toward winning pattern.
        In-place on GPU.
        """
        self.prototypes[col_idx] = F.normalize(
            self.prototypes[col_idx] + lr * (pattern - self.prototypes[col_idx]),
            dim=0
        )
    
    def _assign_columns_to_cells(self) -> None:
        """Assign columns to nearest IVF cell (k-means step)."""
        sims = torch.mm(self.prototypes, self.cell_centroids.T)  # [n_cols, n_cells]
        self.column_to_cell = sims.argmax(dim=1)
    
    def rebuild_ivf_async(self) -> None:
        """
        Called during sleep phase (async, doesn't block stream).
        Updates IVF cell assignments after prototypes have drifted.
        """
        # This is the only point where we might use FAISS — 
        # but only for the cell centroid update, not per-token routing
        self._assign_columns_to_cells()

# Scaling reference:
# 10K  columns, dim=256: flat matmul ≈ 10K×256 = 2.56M ops ≈ 0.05ms A100
# 100K columns, dim=256: IVF with 316 cells, 8 probed → ~2.5K searched ≈ 0.02ms
# 1M   columns: distributed shards across GPUs, IVF per shard, reduce across shards
```

**Benchmarking table you should include in the paper:**

| Columns | Method | Latency/token | PCIe transfers |
|---|---|---|---|
| 10K | Current (CPU HNSW) | ~2.1ms | 2 per token |
| 10K | GPU flat matmul | ~0.05ms | 0 |
| 100K | GPU IVF (8 cells probed) | ~0.08ms | 0 |
| 1M | GPU IVF distributed (4 GPU) | ~0.3ms | inter-GPU only |

Run this benchmark and put it in the paper. This is what "scalable from day one" actually requires — numbers, not architectural claims.

---

## PROBLEM 5 — The Evaluation Framework Proves Nothing About Emergence

### The Complete Solution: A Multi-Level Unsupervised Evaluation Protocol

You need four evaluation levels, all label-free.

**Level 1 — Structural Coherence (what you have, but misusing it)**

Silhouette and DBI prove clustering exists. They don't prove concepts. Keep them as sanity checks, not primary metrics. Add:

```python
def temporal_coherence(self, window: int = 1000) -> float:
    """
    Do the same input patterns consistently route to the same columns
    over time? Rising temporal coherence = stabilizing representations.
    
    Measure: for each recently seen chunk pattern, track which column 
    won. Compute consistency rate over the last `window` tokens.
    If the system is learning, this should rise from ~1/n_columns 
    (random) toward 1.0 (stable representation).
    """
    if len(self.routing_history) < window:
        return 0.0
    
    pattern_to_winners = defaultdict(list)
    for pattern_hash, winner_idx in self.routing_history[-window:]:
        pattern_to_winners[pattern_hash].append(winner_idx)
    
    coherences = []
    for pattern_hash, winners in pattern_to_winners.items():
        if len(winners) < 2:
            continue
        # Coherence = fraction of times the modal winner was chosen
        mode_count = max(Counter(winners).values())
        coherences.append(mode_count / len(winners))
    
    return float(np.mean(coherences)) if coherences else 0.0
```

**Level 2 — Compositionality Test (no labels needed)**

```python
def compositionality_test(self, 
                           test_pairs: list[tuple[str, str]],
                           router: GPUNativeRouter) -> float:
    """
    If chunk A activates column X, and chunk B activates column Y,
    does the sequence AB activate a column that is geometrically 
    between X and Y in prototype space?
    
    This tests whether the network forms compositional representations
    without any labels.
    
    Compositionality score: cosine similarity of the AB winner's prototype
    to the mean of A-winner and B-winner prototypes.
    Score > 0.6 = compositional structure emerging.
    Score ≈ 1/n_columns = random (no compositionality).
    """
    scores = []
    for (chunk_a, chunk_b) in test_pairs:
        # Encode individually
        enc_a = self.encode(chunk_a)
        enc_b = self.encode(chunk_b)
        enc_ab = self.encode(chunk_a + chunk_b)
        
        # Route individually
        winner_a, _ = router.route(enc_a, context_gain=torch.ones(router.n_cols))
        winner_b, _ = router.route(enc_b, context_gain=torch.ones(router.n_cols))
        winner_ab, _ = router.route(enc_ab, context_gain=torch.ones(router.n_cols))
        
        proto_a = router.prototypes[winner_a]
        proto_b = router.prototypes[winner_b]
        proto_ab = router.prototypes[winner_ab]
        
        # Compositionality: is AB's prototype between A and B?
        expected = F.normalize((proto_a + proto_b).unsqueeze(0), dim=1).squeeze()
        score = F.cosine_similarity(proto_ab.unsqueeze(0), 
                                     expected.unsqueeze(0)).item()
        scores.append(score)
    
    return float(np.mean(scores))
```

**Level 3 — Novelty Coverage (is the system still learning or saturated?)**

```python
def novelty_coverage_curve(self, 
                             token_checkpoints: list[int]) -> dict:
    """
    Tracks the fraction of incoming chunks that are genuinely novel 
    (routed to a column for the first time OR routed to a column whose
    prototype moves significantly).
    
    Should be:
      - High at bootstrap (everything is novel)
      - Decreasing as representations stabilize (learning working)
      - Should NOT reach 0 (some novelty always exists in natural text)
      - Should stabilize at ~5-15% novelty rate for a healthy system
    
    If it reaches 0: the system has saturated and is no longer learning.
    If it stays at 100%: the system is unstable and not consolidating.
    """
    return {
        'novelty_rate_by_checkpoint': [...],
        'saturation_detected': novelty_rate < 0.02,
        'instability_detected': novelty_rate > 0.90,
        'healthy_range': 0.05 < novelty_rate < 0.20
    }
```

**Level 4 — The Grounding Probe (the real test)**

```python
def grounding_probe(self, 
                     semantic_pairs: list[tuple[str, str, str]],
                     router: GPUNativeRouter) -> float:
    """
    Given triples (word_a, word_b, relation) where relation is
    'more_similar_to_a_than_to_b':
    
    E.g.: ('king', 'queen', 'man') means: 'king' should be more similar 
    to 'man' than 'queen' is to 'man', in prototype space.
    
    These are NOT labels — they're structural predictions about what a 
    coherent semantic space should look like. If your emergent 
    representations have ANY semantic structure, they'll satisfy these 
    constraints above chance.
    
    If score ≈ 0.5: random (no semantic structure)
    If score > 0.65: genuine semantic structure emerging
    If score > 0.80: strong semantic organization
    
    Start with 50 manually constructed triples, expand to 500.
    This is your primary paper metric for the emergence claim.
    """
    correct = 0
    for (anchor, pos, neg) in semantic_pairs:
        enc_anchor = self.encode(anchor)
        enc_pos = self.encode(pos)
        enc_neg = self.encode(neg)
        
        winner_anchor, _ = router.route(enc_anchor, ...)
        winner_pos, _ = router.route(enc_pos, ...)
        winner_neg, _ = router.route(enc_neg, ...)
        
        proto_anchor = router.prototypes[winner_anchor]
        proto_pos = router.prototypes[winner_pos]
        proto_neg = router.prototypes[winner_neg]
        
        sim_pos = F.cosine_similarity(proto_anchor.unsqueeze(0), 
                                       proto_pos.unsqueeze(0)).item()
        sim_neg = F.cosine_similarity(proto_anchor.unsqueeze(0), 
                                       proto_neg.unsqueeze(0)).item()
        
        if sim_pos > sim_neg:
            correct += 1
    
    return correct / len(semantic_pairs)
```

This grounding probe is the metric that makes your emergence claim falsifiable. If it beats 0.65 with no labels, you have a real result.

---

## PROBLEM 6 — The Abstraction Layer Must Be a Real Layer, Not a Proxy

### The Complete Solution: Online Slow Feature Abstraction as a First-Class Layer

The core principle: features that vary slowly across time carry semantic content. A fast-changing sequence like `"the the the"` has high character-rate variation but slow concept-level variation. The Abstraction Layer extracts this slow variation.

```python
class AbstractionLayer:
    """
    Hierarchical Slow Feature Layer — a true feedforward layer,
    not a proxy observer.
    
    Inputs: winning assembly prototypes from Competitive Layer (fast-varying)
    Outputs: abstract concept vectors (slow-varying)
    
    Drives:
      - Routing bias in the Competitive Layer (top-down)
      - Chunking boundary decisions in the Chunking Layer (top-down)
      - Curiosity signal for gap detection
      
    Learning: Hebbian SFA — connections that reduce output variance
    over time are strengthened (anti-Hebbian in the temporal domain).
    
    This replaces the OnlineSFA proxy entirely.
    """
    
    def __init__(self, 
                 input_dim: int,          # Competitive Layer prototype dim
                 n_concepts: int = 256,   # Number of abstract concepts
                 tau_slow: float = 200.0, # Tokens for slow feature timescale
                 tau_fast: float = 10.0,  # Tokens for fast feature (contrast)
                 device: str = 'cuda'):
        self.input_dim = input_dim
        self.n_concepts = n_concepts
        self.device = device
        
        # Feature extraction weights (learned to minimize output variance)
        # Shape: [n_concepts, input_dim]
        self.W = torch.randn(n_concepts, input_dim, device=device) * 0.1
        self.W = F.normalize(self.W, dim=1)
        
        # Slow and fast running means (for variance computation)
        self.slow_mean = torch.zeros(n_concepts, device=device)
        self.fast_mean = torch.zeros(n_concepts, device=device)
        self.slow_var = torch.ones(n_concepts, device=device)
        
        # Concept stability: how much has each concept varied recently?
        self.concept_stability = torch.ones(n_concepts, device=device)
        
        # Concept uncertainty: how often does input fail to activate concept?
        self.concept_certainty = torch.ones(n_concepts, device=device) * 0.5
        
        # Feedback connections back to Competitive Layer
        # These modulate routing (top-down bias)
        # Shape: [n_competitive_columns, n_concepts]  — set at init
        self.feedback_W: torch.Tensor | None = None
        
        self.tau_slow = tau_slow
        self.tau_fast = tau_fast
        self.alpha_slow = 1.0 / tau_slow
        self.alpha_fast = 1.0 / tau_fast
        
        self.token_count = 0
    
    def forward(self, 
                competitive_assembly: torch.Tensor  # [input_dim]
                ) -> torch.Tensor:
        """
        Forward pass: extract abstract concept activations.
        Returns: concept_activations [n_concepts]
        """
        # Project input through feature weights
        raw_output = torch.mv(self.W, competitive_assembly)  # [n_concepts]
        
        # Normalize by slow standard deviation (whitening in concept space)
        slow_std = torch.sqrt(self.slow_var + 1e-6)
        concept_activations = raw_output / slow_std
        
        # Update running statistics
        self.fast_mean = ((1 - self.alpha_fast) * self.fast_mean + 
                           self.alpha_fast * concept_activations)
        self.slow_mean = ((1 - self.alpha_slow) * self.slow_mean + 
                           self.alpha_slow * concept_activations)
        
        # Slow variance: variance of fast mean (how much fast mean varies)
        fast_deviation = (self.fast_mean - self.slow_mean) ** 2
        self.slow_var = ((1 - self.alpha_slow) * self.slow_var + 
                          self.alpha_slow * fast_deviation)
        
        # Stability: high when output isn't varying (concept is stable)
        self.concept_stability = 1.0 / (self.slow_var + 0.1)
        self.concept_stability = self.concept_stability / (self.concept_stability.max() + 1e-6)
        
        self.token_count += 1
        return concept_activations
    
    def update_weights(self, 
                       concept_activations: torch.Tensor,
                       lr: float = 0.001) -> None:
        """
        SFA weight update: reduce temporal variance of outputs.
        
        Anti-Hebbian in time: if concept output is changing rapidly, 
        reduce the weight responsible → drives W toward slow features.
        
        Concretely: reduce weights whose output has high fast_var / slow_var.
        """
        # Temporal variance ratio per concept
        temporal_instability = self.slow_var / (self.slow_var.mean() + 1e-6)
        
        # Concepts that vary fast should have their weights pushed toward
        # directions that are more stable
        # This is an online approximation to the SFA objective
        for i in range(self.n_concepts):
            if temporal_instability[i] > 1.5:  # This concept is too fast
                # Hebbian decay: reduce this weight component
                self.W[i] -= lr * temporal_instability[i] * self.W[i]
                self.W[i] = F.normalize(self.W[i], dim=0)
    
    def get_routing_bias(self, n_competitive_columns: int) -> torch.Tensor:
        """
        Top-down feedback: provides a bias signal to the Competitive Layer
        routing. Columns whose prototypes align with stable concepts get 
        a boost in routing.
        
        Returns: [n_competitive_columns] routing bias
        """
        if self.feedback_W is None:
            # Initialize feedback weights (lazy)
            self.feedback_W = torch.randn(n_competitive_columns, 
                                           self.n_concepts,
                                           device=self.device) * 0.1
        
        # Stable concepts (high stability) contribute stronger feedback
        stable_concepts = self.concept_stability * (self.slow_mean.abs() + 0.1)
        routing_bias = torch.mv(self.feedback_W, stable_concepts)
        routing_bias = torch.sigmoid(routing_bias) - 0.5  # Center around 0
        return routing_bias
    
    def get_curiosity_gaps(self, top_n: int = 5) -> list[dict]:
        """
        Identify which concepts are most uncertain / unstable.
        These are the gaps that drive autonomous retrieval.
        
        Returns list of gap descriptors:
          - concept_idx: which concept is weak
          - instability: how much variance it shows
          - certainty: how often it's clearly activated
        
        This replaces the heuristic keyword-based gap detection.
        The gaps are now geometrically defined in concept space.
        """
        # Gap score: high instability + low certainty = genuine gap
        gap_scores = self.slow_var * (1.0 - self.concept_certainty)
        top_gaps = gap_scores.topk(top_n).indices.tolist()
        
        return [
            {
                'concept_idx': idx,
                'instability': self.slow_var[idx].item(),
                'certainty': self.concept_certainty[idx].item(),
                'recent_activation': self.slow_mean[idx].item(),
                'gap_score': gap_scores[idx].item()
            }
            for idx in top_gaps
        ]
```

**How the feedback loop works now:**

```
Chunking Layer → Competitive Layer → Abstraction Layer
       ↑                ↑                    |
       |                |____________________|  (routing bias)
       |_____________________________________|  (boundary bias)
```

The Abstraction Layer now actively shapes what the Competitive Layer learns by telling it which columns align with stable concepts. Concepts that are unstable don't get routing preference, which forces the Competitive Layer to keep exploring until they stabilize. This is a self-organizing feedback loop, not a passive observer.

---

## PROBLEM 7 — The Autonomous Acquisition Is Retrieval-Dependent: Make the Curiosity Geometric

### The Complete Solution: Concept-Space Gap Navigation

Replace keyword-based retrieval with concept-space navigation. The query is generated from the geometric gap, not from text tokens.

```python
class GeometricCuriosityController:
    """
    Curiosity is driven by geometric gaps in concept space,
    not by keyword heuristics.
    
    Gap → query generation → retrieval → stream injection
    
    The query is synthesized from the concepts most adjacent to the gap,
    which does use text (because the retrieval target is text), but the
    GAP DETECTION is fully internal and geometric.
    """
    
    def __init__(self, abstraction_layer: AbstractionLayer,
                       concept_lexicon: dict,  # concept_idx → [associated chunks]
                       retrieval_budget: int = 1000):
        self.abs_layer = abstraction_layer
        self.lexicon = concept_lexicon
        self.budget = retrieval_budget
        
    def compute_query(self) -> str | None:
        """
        Synthesizes a retrieval query from the geometric gap.
        
        Step 1: Find top-N gap concepts (internal, geometric)
        Step 2: For each gap concept, find its nearest non-gap neighbors
        Step 3: Look up what text chunks most activated those neighbors
        Step 4: Combine into a query string
        
        The text in step 3-4 is a LABEL READ FROM THE LEXICON, not 
        a pre-defined keyword. The lexicon is built from observed chunks,
        so the query vocabulary is fully emergent.
        """
        gaps = self.abs_layer.get_curiosity_gaps(top_n=3)
        if not gaps or gaps[0]['gap_score'] < 0.1:
            return None  # No significant gaps
        
        query_terms = []
        for gap in gaps:
            gap_idx = gap['concept_idx']
            
            # Find concepts NEAR the gap (adjacent in concept space)
            gap_vector = self.abs_layer.W[gap_idx]
            all_sims = torch.mv(self.abs_layer.W, gap_vector)
            all_sims[gap_idx] = -1.0  # Exclude the gap itself
            neighbor_idx = all_sims.argmax().item()
            
            # What chunks activated this neighbor concept most?
            if neighbor_idx in self.lexicon:
                neighbor_chunks = self.lexicon[neighbor_idx][:3]
                query_terms.extend(neighbor_chunks)
        
        if not query_terms:
            return None
        
        # Deduplicate and form query string
        query = ' '.join(list(dict.fromkeys(query_terms))[:6])
        return query
    
    def update_lexicon(self, 
                        concept_activations: torch.Tensor,
                        active_chunks: list[str]) -> None:
        """
        After each routing, update which text chunks are associated 
        with which concepts.
        This is how the query vocabulary stays emergent and current.
        """
        active_concepts = (concept_activations > 0.5).nonzero(as_tuple=True)[0]
        for concept_idx in active_concepts.tolist():
            if concept_idx not in self.lexicon:
                self.lexicon[concept_idx] = []
            self.lexicon[concept_idx].extend(active_chunks)
            # Keep only the most recent N chunks per concept
            self.lexicon[concept_idx] = self.lexicon[concept_idx][-50:]
```

This makes the curiosity genuinely concept-driven: the system notices a geometric gap (concept with high instability and low certainty), identifies what's *adjacent* to that gap in its own concept space, retrieves the text chunks it learned those neighbors from, and uses those to query external sources. The gap detection is fully internal; only the retrieval interface uses external text.

---

## PROBLEM 8 — The Binding Layer Assert: Full Fix

```python
class BindingLayer:
    """
    Corrected BindingLayer:
    - n_bindings is INDEPENDENT of n_columns
    - Each binding neuron connects to a RANDOM SUBSET of 2-5 source columns
    - Binding is coincidence across subsets, not 1:1 mapping
    """
    
    def __init__(self, 
                 n_bindings: int,      # Independent parameter
                 n_columns: int,       # Source columns
                 fan_in: int = 4,      # How many columns each binding neuron watches
                 threshold: float = 2.0,
                 tau_binding: float = 50.0,
                 device: str = 'cuda'):
        assert 2 <= fan_in <= n_columns, f"fan_in must be in [2, n_columns]"
        # REMOVED: assert n_bindings == n_columns
        
        self.n_bindings = n_bindings
        self.n_columns = n_columns
        self.fan_in = fan_in
        self.threshold = threshold
        self.tau_binding = tau_binding
        self.device = device
        
        # Sparse connectivity: each binding neuron watches `fan_in` columns
        # Shape: [n_bindings, n_columns] — binary, sparse
        self.connectivity = torch.zeros(n_bindings, n_columns, device=device)
        for i in range(n_bindings):
            # Random subset of fan_in columns
            sources = torch.randperm(n_columns)[:fan_in]
            self.connectivity[i, sources] = 1.0
        
        # STP state per binding neuron
        self.u = torch.zeros(n_bindings, device=device)  # Facilitation
        self.x = torch.ones(n_bindings, device=device)   # Depression
        
        # STP parameters (Tsodyks-Markram model)
        self.U_inc = 0.15
        self.tau_f = 1500.0  # Facilitation time constant (tokens)
        self.tau_d = 200.0   # Depression time constant (tokens)
        
        # PV+ interneuron inhibition (scalar, global to this layer)
        self.pv_inhibition = 0.0
        self.pv_tau = 10.0
        
        # Composite assembly storage: successful bindings → new assemblies
        self.bound_assemblies: list[dict] = []
    
    def detect_coincidences(self, 
                             column_spike_rates: torch.Tensor,  # [n_columns]
                             dt: float = 1.0
                             ) -> tuple[torch.Tensor, list[int]]:
        """
        Detect coincident activation of column subsets.
        
        Returns:
          - binding_outputs: [n_bindings] activation level
          - new_bindings: list of binding indices that fired this step
        """
        # Weighted input to each binding neuron from its source columns
        # Shape: [n_bindings] — masked sum
        weighted_input = torch.mv(self.connectivity, column_spike_rates)
        
        # Update STP
        # Facilitation
        self.u += dt * (-self.u / self.tau_f + self.U_inc * weighted_input * (1 - self.u))
        # Depression  
        release = self.u * self.x
        self.x += dt * ((1 - self.x) / self.tau_d - release)
        
        # Effective input after STP
        effective_input = release * weighted_input
        
        # PV+ inhibition: if any binding neuron fires strongly, suppress others
        strong_fire = (effective_input > self.threshold * 1.5)
        if strong_fire.any():
            self.pv_inhibition = min(1.0, self.pv_inhibition + 0.3)
        else:
            self.pv_inhibition = max(0.0, self.pv_inhibition - dt / self.pv_tau)
        
        # Threshold + inhibition
        binding_outputs = torch.clamp(
            effective_input - self.threshold - self.pv_inhibition, 
            min=0.0
        )
        
        new_bindings = binding_outputs.nonzero(as_tuple=True)[0].tolist()
        
        # Store successful bindings as composite assemblies
        for binding_idx in new_bindings:
            source_cols = self.connectivity[binding_idx].nonzero(as_tuple=True)[0].tolist()
            self.bound_assemblies.append({
                'binding_idx': binding_idx,
                'source_columns': source_cols,
                'strength': binding_outputs[binding_idx].item(),
                'token': -1  # Caller sets this
            })
        
        return binding_outputs, new_bindings
    
    def grow_binding(self, 
                      high_correlation_columns: list[tuple[int, int, float]]
                      ) -> int:
        """
        Structural plasticity for bindings: if two columns repeatedly 
        co-activate but have no binding neuron watching both, add one.
        
        high_correlation_columns: list of (col_a, col_b, correlation)
        Returns: number of new binding neurons added
        """
        existing_pairs = set()
        for i in range(self.n_bindings):
            sources = self.connectivity[i].nonzero(as_tuple=True)[0].tolist()
            for a in sources:
                for b in sources:
                    if a < b:
                        existing_pairs.add((a, b))
        
        new_count = 0
        for (col_a, col_b, corr) in high_correlation_columns:
            if corr > 0.7 and (col_a, col_b) not in existing_pairs:
                # Grow a new binding neuron watching col_a and col_b
                new_row = torch.zeros(1, self.n_columns, device=self.device)
                new_row[0, col_a] = 1.0
                new_row[0, col_b] = 1.0
                self.connectivity = torch.cat([self.connectivity, new_row], dim=0)
                self.n_bindings += 1
                self.u = torch.cat([self.u, torch.zeros(1, device=self.device)])
                self.x = torch.cat([self.x, torch.ones(1, device=self.device)])
                existing_pairs.add((col_a, col_b))
                new_count += 1
        
        return new_count
```

---

## PROBLEM 9 — The AdEx Step Numerical Stability

```python
def step(self, I_syn: torch.Tensor, t: float) -> torch.Tensor:
    """
    Corrected AdEx step using Heun's method (RK2) for the 
    exponential upswing term. Leak uses exponential Euler (stable).
    Adaptation uses forward Euler (tau_w >> dt, safe).
    """
    def voltage_derivative(V, w, I):
        exp_arg = torch.clamp((V - self.V_T) / self.delta_T, min=-10.0, max=5.0)
        exp_term = self.delta_T * torch.exp(exp_arg)
        dV = ((-self.g_L * (V - self.E_L) + self.g_L * exp_term 
               - w + I) / self.C_m)
        return dV
    
    # Heun's method (predictor-corrector, 2nd order)
    # Step 1: Euler predictor
    dV1 = voltage_derivative(self.V, self.w, I_syn)
    V_pred = self.V + self.dt * dV1
    
    # Clamp prediction to prevent explosion before corrector
    V_pred = torch.clamp(V_pred, min=self.E_L - 20.0, max=self.V_peak)
    
    # Adaptation intermediate (forward Euler — stable for tau_w=100ms >> dt=0.5ms)
    dw1 = (self.a * (self.V - self.E_L) - self.w) / self.tau_w
    w_pred = self.w + self.dt * dw1
    
    # Step 2: corrector
    dV2 = voltage_derivative(V_pred, w_pred, I_syn)
    self.V = self.V + 0.5 * self.dt * (dV1 + dV2)
    self.V = torch.clamp(self.V, min=self.E_L - 20.0, max=self.V_peak)
    
    # Adaptation update
    dw = (self.a * (self.V - self.E_L) - self.w) / self.tau_w
    self.w = self.w + self.dt * dw
    
    # Spike detection
    spikes = (self.V >= self.V_peak)
    spikes_float = spikes.float()
    
    # Reset
    self.V = torch.where(spikes, 
                          torch.full_like(self.V, self.V_reset), 
                          self.V)
    self.w = self.w + spikes_float * self.b
    
    # NaN guard — catch instability before it propagates
    nan_mask = torch.isnan(self.V) | torch.isinf(self.V)
    if nan_mask.any():
        self.V[nan_mask] = self.E_L
        self.w[nan_mask] = 0.0
    
    self.spike_times = torch.where(
        spikes, torch.full_like(self.spike_times, t), self.spike_times
    )
    return spikes
```

---

## PROBLEM 10 — Baseline Comparisons: Exactly What to Implement

You need three baselines. All runnable in a few hours of compute.

```python
class OnlineSOMBaseline:
    """
    Classical online SOM on the same character stream.
    This is your null hypothesis: if HECSN doesn't beat this,
    the whole SNN apparatus is adding nothing.
    
    Same evaluation metrics. Same input stream. Same evaluation protocol.
    """
    def __init__(self, n_prototypes: int, input_dim: int, 
                 initial_lr: float = 0.5,
                 initial_radius: float = None):
        self.n = n_prototypes
        self.prototypes = torch.randn(n_prototypes, input_dim)
        self.prototypes = F.normalize(self.prototypes, dim=1)
        self.lr = initial_lr
        self.radius = initial_radius or (n_prototypes / 4)
        self.token_count = 0
        
    def step(self, pattern: torch.Tensor) -> int:
        # Nearest prototype
        sims = torch.mv(self.prototypes, F.normalize(pattern, dim=0))
        winner = sims.argmax().item()
        
        # Neighborhood update (Gaussian kernel)
        distances = torch.arange(self.n).float()
        distances = (distances - winner).abs()
        neighborhood = torch.exp(-distances ** 2 / (2 * self.radius ** 2))
        
        # Update all prototypes by neighborhood
        lr_t = self.lr * (0.01 / self.lr) ** (self.token_count / 100_000)
        radius_t = max(1.0, self.radius * (1.0 / self.radius) ** (self.token_count / 100_000))
        
        delta = F.normalize(pattern, dim=0) - self.prototypes
        self.prototypes += lr_t * neighborhood.unsqueeze(1) * delta
        self.prototypes = F.normalize(self.prototypes, dim=1)
        
        self.token_count += 1
        return winner

class NgramBaseline:
    """
    N-gram model on character stream. 
    Baseline for the predictive coding bootstrap.
    If HECSN's prediction accuracy doesn't exceed this, 
    the SNN mechanism is not helping.
    """
    def __init__(self, n: int = 4, vocab_size: int = 256):
        self.n = n
        self.counts: dict = defaultdict(lambda: defaultdict(int))
        self.buffer: list[int] = []
        
    def update(self, byte_val: int) -> float:
        """Returns prediction accuracy for this byte."""
        if len(self.buffer) >= self.n - 1:
            context = tuple(self.buffer[-(self.n-1):])
            # Probability of this byte given context
            context_total = sum(self.counts[context].values()) + 256
            byte_count = self.counts[context][byte_val] + 1
            probability = byte_count / context_total
            self.counts[context][byte_val] += 1
        else:
            probability = 1.0 / 256
        
        self.buffer.append(byte_val)
        return probability
```

**What to report in the paper table:**

| Metric | N-gram (4-gram) | Online SOM | HECSN |
|---|---|---|---|
| Temporal coherence @50K tokens | N/A | ~0.XX | ~0.XX |
| Compositionality score | N/A | ~0.XX | ~0.XX |
| Grounding probe accuracy | ~0.50 (random) | ~0.XX | **target: >0.65** |
| Novelty rate @100K tokens | N/A | ~0.XX | ~0.05–0.15 |
| Task-A recall after Task-B | N/A | ~0.XX | ~0.XX |

Fill in the XX values with your actual runs. Even if HECSN doesn't beat SOM everywhere, identifying where it does and doesn't is a real scientific contribution.

---

## PROBLEM 11 — STC Timescale: From Arbitrary to Principled

The `functional_minute = 500 tokens` mapping is arbitrary. Here's how to make it principled:

```python
def calibrate_functional_minute(stream_path: str, 
                                  network: 'HECSN',
                                  sample_tokens: int = 10_000) -> int:
    """
    Calibrate functional_minute empirically from the network's own 
    activity on the target stream.
    
    Biological rationale: 1 cortical minute ≈ time for a stable 
    memory to form (Early-LTP timescale ~1-3 min in vitro).
    
    Computational equivalent: time for a prototype to stabilize 
    after first exposure (convergence time of winner-local drift).
    
    Procedure:
    1. Expose network to a novel pattern repeatedly
    2. Track winner-local drift until it falls below 1% per 100 tokens
    3. That convergence time = 1 functional minute
    
    This is self-calibrating: each network/stream combination gets
    its own functional_minute, not a fixed hyperparameter.
    """
    convergence_times = []
    
    # Generate 10 random novel patterns
    for _ in range(10):
        novel_pattern = torch.randn(network.input_dim).to(network.device)
        novel_pattern = F.normalize(novel_pattern, dim=0)
        
        initial_drift = float('inf')
        for step in range(sample_tokens):
            winner, _ = network.router.route(novel_pattern, 
                                              context_gain=torch.ones(network.n_columns))
            proto_before = network.router.prototypes[winner].clone()
            network.competitive_update(novel_pattern, winner, modulator=0.5)
            proto_after = network.router.prototypes[winner]
            
            drift = (proto_after - proto_before).norm().item()
            
            if step > 0 and drift < 0.01 * initial_drift:
                convergence_times.append(step)
                break
            if step == 0:
                initial_drift = drift
    
    functional_minute = int(np.median(convergence_times)) if convergence_times else 500
    print(f"Calibrated functional_minute = {functional_minute} tokens")
    print(f"  (range: {min(convergence_times)}–{max(convergence_times)})")
    return functional_minute
```

Run this before every training run. Report the calibrated value. Now the STC timescales are grounded in the network's actual convergence dynamics, not a biological analogy.

---

## THE REVISED ARCHITECTURE: What the Paper Should Describe

```
INPUT: Raw byte stream (continuous, unbounded)
         │
         ▼
┌─────────────────────────────┐
│  CHUNKING LAYER             │  ← NEW, replaces heuristic segmentation
│  Learned variable-len units │
│  Predictability-based bound │
│  Updates via competitive lr │
└─────────────────────────────┘
         │ variable-len chunks (spike patterns)
         ▼
┌─────────────────────────────┐
│  ENCODING LAYER (RTF)       │  ← Keep, but validate vs. baselines
│  Rate + temporal fusion     │
│  Positional phase offset    │
└─────────────────────────────┘
         │ [prototype_dim] spike vector
         ▼
┌─────────────────────────────────────────────────────┐
│  COMPETITIVE LAYER                                  │
│  GPU-native routing (IVF flat → IVF at scale)       │  ← No CPU HNSW
│  Winner history refractory (no column monopoly)     │
│  Three-factor STDP (with triplet extension)         │
│  Independent neuromodulators (DA/ACh/NE/5-HT)      │  ← No mult scalar
│  ← Routing bias from Abstraction Layer (top-down)  │
└─────────────────────────────────────────────────────┘
         │ winner assembly + top-k
         ▼
┌─────────────────────────────┐
│  BINDING LAYER              │  ← Fixed: n_bindings ≠ n_columns
│  Sparse connectivity matrix │
│  STP (facilitation/depress) │
│  Structural plasticity grow │
└─────────────────────────────┘
         │ composite assemblies
         ▼
┌─────────────────────────────┐
│  ABSTRACTION LAYER          │  ← Real layer now, not proxy
│  Online SFA (anti-Hebbian)  │
│  Concept stability tracking │
│  Geometric gap detection    │
│  → Feedback to Competitive  │
│  → Feedback to Chunking     │
└─────────────────────────────┘
         │ concept gap signal
         ▼
┌─────────────────────────────┐
│  CURIOSITY CONTROLLER       │  ← Concept-space driven, not keyword
│  Geometric gap → text query │
│  Emergent query vocabulary  │
│  Retrieval → stream inject  │
└─────────────────────────────┘
         │ retrieved text → byte stream (feeds back to INPUT)

MEMORY & CONSOLIDATION (parallel track):
┌──────────────────────────────────────────────┐
│  DUAL MEMORY STORE                           │
│  Fast EMA (drift/novelty baseline)           │
│  Slow reservoir (importance-weighted)        │
│  Fragility score per memory                  │  ← New
└──────────────────────────────────────────────┘
         │ triggered by drift / schedule
         ▼
┌──────────────────────────────────────────────┐
│  THREE-PHASE SLEEP                           │
│  Phase A (micro): Maintenance only           │  ← No consolidation
│  Phase B (deep): Fragility-gated consolid.  │  ← Anchored, safe
│  Phase C (emergency): Repair only           │  ← No new consolid.
│  Calibrated functional_minute               │  ← Self-calibrating
└──────────────────────────────────────────────┘

EVALUATION (all label-free):
  • Structural coherence (silhouette/DBI — sanity check only)
  • Temporal coherence (primary stability metric)
  • Compositionality score (primary structure metric)
  • Grounding probe accuracy (primary emergence metric — must beat 0.65)
  • Novelty coverage curve (learning health monitor)
  • Forgetting metric (Task-A activation overlap)
  Baselines: Online SOM, 4-gram model
```

---

## The Paper Structure (Exactly How to Write It)

**8-10 pages. Venue: NeurIPS Workshop on Biological and Artificial Reinforcement Learning, or Frontiers in Computational Neuroscience.**

```
1. Introduction (1 page)
   - The unsolved problem: online, label-free semantic organization in SNNs
   - What HECSN claims (precisely, narrowly)
   - What HECSN does NOT claim (no "understands language")
   - Falsifiable prediction: grounding probe > 0.65 after 500K tokens

2. Related Work (0.5 page)  
   - CoLaNET (columnar SNN continual learning, 2025) — closest competitor
   - SpikeGPT/NeuronSpark — different objective (supervised, not emergent)
   - Online SOM literature — the baseline you're building beyond
   - Predictive coding SNNs (Karlsson et al. 2026) — situate the Surprise Monitor

3. Architecture (2 pages)
   - The six-layer design (as above)
   - One paragraph per layer: what it does, why it's needed, key mechanism
   - The three feedback loops: Abstraction→Competitive, Abstraction→Chunking, 
     Curiosity→Stream

4. Evaluation Protocol (0.5 page)
   - The four label-free metrics
   - Why each metric is falsifiable
   - The two baselines

5. Experiments (3 pages)
   - Stage 0: mechanism validation (chunking emergence, routing stability)
   - Stage 1: continual learning (Task A → Task B, fragility-gated consolidation)
   - Stage 2: autonomous acquisition (curiosity loop, novelty coverage curve)
   - Stage 3: emergence test (grounding probe vs. baseline)
   - Results table (fill in with actual numbers)

6. Limitations (0.5 page)
   - Text-only grounding gap (honest, not minimized)
   - Scale not yet demonstrated past 10K column-equivalent
   - Curiosity vocabulary depends on emergent chunking quality

7. Conclusion (0.5 page)
```

The entire changelog, Terminus description, and implementation status notes move to a GitHub README. They're engineering, not science. Both matter — but they belong in different places.

---

## One Final Truth

The architecture is sound in its instincts. The problems are all fixable.

cleaned-up the paper to makes a real, testable, honest claim about emergent structure in SNNs.

The goal is reachable. Ground it and build it properly.