===========================================================================
FILE:    agents.py
PROJECT: Intent-Aware Multi-Agent AI Orchestration
         in the Edge-Cloud Continuum
===========================================================================

OVERVIEW
--------
agents.py is the intellectual core of the entire system. It defines six
independent AI agents, each implementing a named, published ML/AI algorithm.
Together they form a sequential pipeline that analyses, classifies, and
deploys IoT applications across the edge-cloud continuum in an intent-aware
manner.

Each agent performs one independent function and passes its output to the
next. All six outputs are bundled into a single AgentDecision object that
OrchestratorAgent.py reads to execute the final deployment.


===========================================================================
HOW ALL FILES CONNECT — THE MECHANICS
===========================================================================

OrchestratorAgent.py passes one shared dictionary through every module:

    shared_state = {}   ← one dict, passed through everything
    config       = {}   ← one config dict, controls all settings

Execution sequence when OrchestratorAgent.py runs:

    applications.run(config, shared_state)
        writes → shared_state["applications"]        list of ApplicationInstance

    computing_resources.run(config, shared_state)
        writes → shared_state["infrastructure"]      Infrastructure object

    agents.run(config, shared_state)
        reads  ← shared_state["applications"]        ← FROM applications.py
        reads  ← shared_state["infrastructure"]      ← FROM computing_resources.py
        writes → shared_state["agent_decisions"]     list of AgentDecision
        writes → shared_state["active_placements"]   dict of live placements

    baselines.run(config, shared_state)
        reads  ← shared_state["applications"]
        reads  ← shared_state["infrastructure"]
        reads  ← shared_state["agent_decisions"]
        writes → shared_state["comparison_table"]

    figures.run(config, shared_state)
        reads  ← everything above
        writes → IEEE-quality PDF graphs

The applications feeding the agents ALWAYS come from applications.py.
Nothing in agents.py generates its own data in production mode.

STANDALONE MODE (python agents.py in PyCharm):
    The __main__ block at the bottom calls applications.run() and
    computing_resources.run() automatically before running the pipeline.
    This is the same code path as OrchestratorAgent.py — no difference.
    Use standalone mode during development to test agents.py in isolation.


===========================================================================
CONFIGURATION SETTINGS  (top of agents.py — edit before each run)
===========================================================================

LLM_BACKEND : str
    "mock"       No model required. Uses deterministic keyword matching.
                 Covers all 9 subtypes. Fully reproducible output.
                 Use this first to verify the pipeline runs correctly.

    "ollama"     Uses a locally running Ollama server (recommended).
                 Install: https://ollama.com  (free, one .exe installer)
                 Pull model: ollama pull llama3.2:8b-instruct-q4_K_M
                 Ollama must be running before starting the program.

    "llama_cpp"  Uses llama-cpp-python directly (added in a later step).
                 For CPU-only inference on Windows without Ollama.

LLM_MODEL : str
    Ollama model tag. Used when LLM_BACKEND = "ollama".
    Default  : "llama3.2:8b-instruct-q4_K_M"
    Faster   : "llama3.2:3b-instruct-q8_0"    (less accurate)
    Strongest: "llama3.1:70b"                  (needs GPU, most accurate)

LLM_HOST : str
    Ollama server URL. Default: "http://localhost:11434"
    Change only if Ollama runs on a different machine on your network.

AGENT_VERBOSE : bool
    True  = print detailed per-agent reasoning to the console.
            Shows TF-IDF scores, fuzzy memberships, NB posteriors,
            EMA values, TOPSIS CC scores, and offload decisions.
            Recommended during development and debugging.
    False = silent. Only the final summary is printed.
            Use for large experiments (NUM_APPS > 100).

OFFLOAD_CPU_THR : float   (default 80.0)
    EMA CPU % threshold that triggers Agent 6 (OffloadAgent).
    If EMA_CPU of any node exceeds this, apps are migrated away.
    Uses EMA — not raw CPU — to prevent thrashing.
    Tunable range: 50.0 to 95.0. Lower = more aggressive offloading.

OFFLOAD_RAM_THR : float   (default 85.0)
    EMA RAM % threshold for offload trigger.
    Same mechanics as OFFLOAD_CPU_THR but for memory pressure.


===========================================================================
AGENT PIPELINE — EXECUTION ORDER PER APPLICATION
===========================================================================

    ApplicationInstance  (from applications.py)
            │
            ▼
    [Agent 1]  IntentParserAgent
               TF-IDF cosine similarity  +  LLM (Llama)
               Output → IntentProfile
            │
            ▼
    [Agent 2]  QoSProfilerAgent
               Fuzzy Logic  +  Weighted Multi-Criteria Scoring
               Output → QoSProfile
            │
            ▼
    [Agent 3]  ClassificationAgent
               Multinomial Naive Bayes  +  LLM Refinement
               Output → ClassificationResult
            │
            ▼
    [Agent 4]  ResourceMonitorAgent
               Exponential Moving Average (EMA)
               Output → ResourceSnapshot
            │
            ▼
    [Agent 5]  PlacementAgent
               TOPSIS  (Multi-Attribute Decision Making)
               Output → PlacementDecision
            │
            ▼
    [Agent 6]  OffloadAgent
               EMA Threshold Policy  +  Least-Loaded Routing
               Output → OffloadDecision
            │
            ▼
       AgentDecision
    (all six outputs bundled into one object)
            │
            ▼
    OrchestratorAgent.py
    (reads AgentDecision, executes the final deployment)


===========================================================================
AGENT 1 — IntentParserAgent
ALGORITHMS: TF-IDF Cosine Similarity  +  LLM (Llama) Semantic Analysis
===========================================================================

PURPOSE
    Reads the natural-language intent sentence of each application and
    extracts its full semantic meaning. This is the entry point of the
    intelligence pipeline.

TWO-STAGE DESIGN
    Stage 1 is fast and deterministic (TF-IDF, no model needed).
    Stage 2 is deep and flexible (LLM).
    The LLM receives Stage 1 output as a structured prior, which reduces
    hallucination and grounds the model with hard evidence.
    If LLM confidence < 0.5, Stage 1 result is used as fallback.

ALGORITHM 1A — TF-IDF + Cosine Similarity
    TF-IDF transforms the intent sentence into a weighted sparse vector.
    Domain-specific rare terms carry more weight than common words.

    Formulas:
        TF(t, d)  = freq(t, d) / total_terms(d)
        IDF(t)    = log( (N+1) / (df(t)+1) ) + 1     [smoothed]
        TF-IDF    = TF(t, d) × IDF(t)
        cos(A, B) = (A · B) / (‖A‖ × ‖B‖)

    Where:
        t     = term (word token)
        d     = document (intent sentence)
        N     = number of classes (9)
        df(t) = number of classes containing term t

    Process:
        1. At initialisation, build per-class TF-IDF centroid vectors
           from CORPUS (one canonical sentence per class, 9 classes).
        2. At runtime, vectorise the incoming intent sentence.
        3. Compute cosine similarity to every class centroid.
        4. Return best-matching class and similarity score.

    Outputs stored in IntentProfile:
        tfidf_best_class   e.g. "emergency_care"
        tfidf_cosine_sim   e.g. 0.7823
        tfidf_top_terms    top 6 most discriminative terms in the query

ALGORITHM 1B — LLM Semantic Analysis
    The LLM prompt includes:
        - Raw intent text
        - TF-IDF best class and cosine score (as prior)
        - Top discriminative terms
        - Extracted numeric constraints (ms, °C, bar, bpm, etc.)
        - Detected urgency signals

    The LLM returns JSON:
        domain, subtype, tier, urgency, confidence, reasoning

    Blending rule:
        LLM confidence >= 0.5 → use LLM result
        LLM confidence <  0.5 → fall back to TF-IDF result

ADDITIONAL EXTRACTIONS (always run regardless of LLM backend)
    Numeric constraints via regex:
        latency_ms, temperature_c, pressure_bar, heart_rate_bpm,
        glucose_mgdl, vibration_mms, bandwidth_kbps, storage_gb/mb

    Urgency signals (keyword lexicon, weights 0.3 – 1.0):
        immediately(1.0), emergency(1.0), critical(1.0), arrest(1.0),
        runaway(1.0), now(0.9), urgent(0.9), alert(0.8) ...

    Action verbs:
        classify, detect, predict, control, monitor, diagnose, alert ...

OUTPUT: IntentProfile
    detected_domain, detected_subtype, key_entities, action_verbs,
    numeric_constraints, urgency_signals, tfidf_best_class,
    tfidf_cosine_sim, tfidf_top_terms, llm_confidence,
    llm_reasoning, parse_time_ms


===========================================================================
AGENT 2 — QoSProfilerAgent
ALGORITHMS: Fuzzy Logic (Trapezoidal Membership Functions)
            + Weighted Multi-Criteria Scoring
===========================================================================

PURPOSE
    Converts the semantic intent profile into a formal, quantified
    Quality-of-Service contract. Provides Agent 5 (TOPSIS) with precise
    numerical targets rather than vague labels.

WHY FUZZY LOGIC?
    Classical crisp thresholds lose information at boundaries.
    A latency requirement of 48ms is "almost critical" — treating it as
    strictly "high" discards that nuance. Fuzzy logic models the inherent
    vagueness of urgency using graded membership degrees.

ALGORITHM 2A — Fuzzy Logic with Trapezoidal Membership Functions
    Each urgency level is a trapezoidal fuzzy set μ(x; a, b, c, d):

        μ(x) = 0              if x ≤ a or x ≥ d
             = (x − a)/(b − a)  if a < x < b    (rising slope)
             = 1              if b ≤ x ≤ c       (plateau)
             = (d − x)/(d − c)  if c < x < d    (falling slope)

    Five fuzzy sets defined over urgency score x ∈ [0, 1]:

        Level       a       b       c       d
        ──────────  ──────  ──────  ──────  ──────
        background  -0.10   0.00    0.15    0.25
        low          0.10   0.20    0.35    0.50
        medium       0.30   0.45    0.60    0.75
        high         0.55   0.70    0.80    0.92
        critical     0.75   0.88    1.00    1.10

    Fuzzification:
        raw_score    = max weight of matched urgency signals from lexicon
        memberships  = μ(raw_score) for each of the five sets above

    Defuzzification:
        urgency = argmax(memberships)   [highest membership wins]

    Stored in QoSProfile:
        fuzzy_memberships    e.g. {"critical":0.85, "high":0.30, ...}
        fuzzy_raw_score      e.g. 0.87
        urgency_level        e.g. "critical"

ALGORITHM 2B — Weighted Multi-Criteria Criticality Score
    Criticality C ∈ [0, 1]:

        C = (w₁×domain_weight + w₂×subtype_weight +
             w₃×llm_confidence + w₄×realtime_flag) × offload_penalty

    Weights (published for IEEE reproducibility):
        w₁ = 0.30  domain weight
        w₂ = 0.35  subtype weight (highest — most discriminating)
        w₃ = 0.15  LLM confidence
        w₄ = 0.20  real-time flag

    Domain weights:
        healthcare = 1.00,  industrial = 0.85,  general = 0.50

    Subtype weights:
        emergency_care=1.00, process_control=0.95,
        predictive_maintenance=0.75, quality_control=0.70,
        remote_monitoring=0.65, medical_imaging=0.60,
        digital_twin=0.50, federated_learning=0.35,
        smart_environment=0.25

    Offload penalty: 0.8 if app cannot be offloaded, else 1.0

LATENCY BUDGET (derived from urgency level):
    critical    →  10ms hard /   5ms target
    high        →  50ms hard /  20ms target
    medium      → 200ms hard / 100ms target
    low         →  1000ms    / 500ms target
    background  → 60000ms    / 10000ms target

TIER PREFERENCE:
    critical/high   → preferred=edge,  fallbacks=[fog, cloud]
    medium          → preferred=fog,   fallbacks=[edge, cloud]
    low/background  → preferred=cloud, fallbacks=[fog, edge]

OUTPUT: QoSProfile
    urgency_level, latency_budget_ms, latency_target_ms,
    criticality_score, fuzzy_memberships, fuzzy_raw_score,
    weight_domain, weight_subtype, weight_llm_conf, weight_realtime,
    real_time_required, gpu_required, offload_permitted,
    preferred_tier, fallback_tiers, sla_violation_cost, profiling_time_ms


===========================================================================
AGENT 3 — ClassificationAgent
ALGORITHMS: Multinomial Naive Bayes (Laplace-smoothed)
            + LLM Refinement (conditional)
===========================================================================

PURPOSE
    Removes application agnosticism. Assigns a precise taxonomy label
    and a compute fingerprint to each application. Determines which
    Llama model size should run and what resources are truly needed.

ALGORITHM 3A — Multinomial Naive Bayes
    Bayes theorem with naive conditional independence:

        P(c | text) ∝ P(c) × ∏ P(w | c)

    Prior:       P(c) = 1/9  (uniform over 9 classes)

    Likelihood with Laplace smoothing (α = 1):
        P(w | c) = (count(w, c) + 1) / (total_words(c) + |V|)

        Where |V| = vocabulary size. Laplace smoothing prevents
        zero probabilities for unseen words.

    Inference via log-sum (avoids float underflow):
        log P(c | text) = log P(c) + Σᵢ log P(wᵢ | c)

    Softmax normalisation gives full probability distribution:
        P(c | text) = exp(log_p(c)) / Σ exp(log_p(c'))

    This produces nb_posterior_probs — a probability for all 9 classes.
    Used by figures.py for the NB confidence histogram (IEEE Fig).

    Key outputs:
        nb_posterior_probs    probability over all 9 classes
        nb_top_class          winning class
        nb_confidence         probability of top class [0, 1]
        nb_confidence_gap     P(top1) − P(top2)

ALGORITHM 3B — LLM Refinement (conditional)
    Triggered ONLY when NB is uncertain:
        CONDITION 1: nb_confidence < 0.60
        CONDITION 2: nb_confidence_gap < 0.05  (top-2 too close)

    When triggered, LLM receives NB top-4 posteriors and TF-IDF result
    as context. It may override the class AND refine resource estimates:
        cpu_cores_refined, ram_mb_refined, gpu_mb_refined,
        bandwidth_kbps_refined

    All refinements are sanity-clamped:
        cpu_cores: [0.5, 128.0],  ram_mb: [64, 65536],  gpu_mb: [0, 81920]

    llm_refinement_used = True is logged for every refined decision.

COMPUTE FINGERPRINT (subtype → compute class → model size):
    emergency_care          → llm_heavy  → Llama 8B
    medical_imaging         → gpu_heavy  → Llama 8B
    remote_monitoring       → cpu_only   → Llama 1B
    predictive_maintenance  → cpu_only   → Llama 1B
    quality_control         → gpu_heavy  → Llama 8B
    process_control         → cpu_only   → Llama 1B
    federated_learning      → gpu_light  → Llama 3B
    digital_twin            → llm_heavy  → Llama 8B
    smart_environment       → cpu_only   → Llama 1B

OVERRIDE RULES:
    If app.gpu_memory_mb > 8192 and class ≠ gpu_heavy → upgrade
    If urgency = critical and class = gpu_heavy → downgrade to gpu_light
      (GPU latency overhead violates the hard deadline)

TAXONOMY PATH:
    Every application gets a full path string, e.g.:
        "healthcare/time_critical/emergency_care"
        "industrial/predictive/predictive_maintenance"
        "general/simulation/digital_twin"

OUTPUT: ClassificationResult
    primary_class, secondary_class, compute_class, taxonomy_path,
    nb_posterior_probs, nb_top_class, nb_confidence, nb_confidence_gap,
    estimated_cpu_cores, estimated_ram_mb, estimated_gpu_mb,
    estimated_bw_kbps, model_size_class, llm_refinement_used,
    classification_ms


===========================================================================
AGENT 4 — ResourceMonitorAgent
ALGORITHM: Exponential Moving Average (EMA)  +  SLA Watchdog
===========================================================================

PURPOSE
    Maintains a live, always-current view of all computing nodes across
    all tiers. Acts as the "eyes" of the system. Provides smoothed
    utilisation data to Agent 5 (placement) and Agent 6 (offload).

WHY EMA INSTEAD OF RAW CPU?
    Raw CPU readings are noisy due to bursty IoT workloads.
    A transient 95% spike should not trigger placement changes.
    EMA smooths the signal and prevents thrashing — a well-known
    problem in threshold-based cloud elasticity controllers.

ALGORITHM — Exponential Moving Average
    EMA_t = α × x_t + (1 − α) × EMA_{t−1}

    Where:
        x_t  = current raw CPU or RAM reading (%)
        α    = 0.3  (smoothing factor)
        EMA₀ = x₀   (initialised to first reading)

    α = 0.3 is the standard value used in AWS CloudWatch and
    the Kubernetes Horizontal Pod Autoscaler (HPA).

PERSISTENCE
    EMA state (self._ema_cpu, self._ema_ram) is maintained across ALL
    pipeline calls in one experiment run. Agent 4 is a singleton.

    Example with 3 calls:
        Call 1  raw=5%    EMA = 0.3×5  + 0.7×5  =  5.0%
        Call 2  raw=30%   EMA = 0.3×30 + 0.7×5  = 12.5%
        Call 3  raw=80%   EMA = 0.3×80 + 0.7×12.5 = 32.75%

    This builds a realistic rolling picture of increasing load
    as more applications are placed during the experiment.

SLA WATCHDOG
    Nodes where latency_to_next_ms > 200ms are flagged in sla_violations
    and passed to Agent 6 for reactive response.

UTILISATION LOGGING
    node._log_snapshot() is called on every node every monitoring call.
    This appends a timestamped utilisation record used by figures.py
    for the resource utilisation-over-time plots (IEEE Fig 7).

OUTPUT: ResourceSnapshot
    total_nodes, available_nodes, overloaded_nodes, offline_nodes,
    tier_avg_cpu_pct, tier_avg_ram_pct, tier_free_gpu_mb,
    tier_ema_cpu  ← used by Agent 5 and Agent 6
    tier_ema_ram  ← used by Agent 5 and Agent 6
    sla_violations, monitoring_ms


===========================================================================
AGENT 5 — PlacementAgent
ALGORITHM: TOPSIS
(Technique for Order of Preference by Similarity to Ideal Solution)
===========================================================================

PURPOSE
    The decision-making core. Selects the single best compute node for
    each application using a principled, published MADM algorithm.
    This is the primary algorithmic contribution of the paper.

BACKGROUND
    TOPSIS was published by Hwang & Yoon (1981). It is widely used in
    cloud resource management and edge computing placement research.
    It selects the alternative closest to the positive ideal solution
    and farthest from the negative ideal solution.

CRITERIA (5 dimensions, all benefit — higher is better)
    C1  Latency score      weight = 0.30  (latency is most critical in IoT)
    C2  CPU availability   weight = 0.25  (free CPU % of total)
    C3  RAM availability   weight = 0.20  (free RAM % of total)
    C4  GPU availability   weight = 0.15  (free VRAM if needed)
    C5  Tier-urgency fit   weight = 0.10  (tier alignment with urgency)
    ────────────────────────────────────
    Sum of weights         = 1.00

ALGORITHM — 6 Steps
    Step 1  Build decision matrix X  [n_candidates × 5_criteria]
    Step 2  Column-normalise:  r_ij = x_ij / sqrt(Σᵢ x_ij²)
    Step 3  Weighted matrix:   v_ij = w_j × r_ij
    Step 4  Ideal      A⁺ = { max(v_ij) for each criterion j }
            Anti-ideal A⁻ = { min(v_ij) for each criterion j }
    Step 5  Euclidean distances:
                D⁺ᵢ = sqrt( Σⱼ (v_ij − A⁺_j)² )
                D⁻ᵢ = sqrt( Σⱼ (v_ij − A⁻_j)² )
    Step 6  Closeness coefficient:
                CCᵢ = D⁻ᵢ / (D⁺ᵢ + D⁻ᵢ)   ∈ [0, 1]

            CC = 1  → node is the positive ideal
            CC = 0  → node is the negative ideal
            Higher CC = better placement candidate

PLACEMENT PROCESS
    1. Try preferred tier (from Agent 2) first.
    2. If no feasible node, try fallback tiers in order.
    3. Run TOPSIS on all feasible candidates in the tier.
    4. Select node with highest CC.
    5. Early stop if CC > 0.30 in preferred tier.

ADMISSION CONTROL
    A node is only a TOPSIS candidate if node.can_accommodate(app)
    passes — this checks CPU, RAM, GPU, bandwidth, and latency deadline
    simultaneously before the node enters the TOPSIS matrix.

TOPSIS RANKING LOG
    All candidates and their CC scores are saved in topsis_ranking.
    Used by figures.py for the TOPSIS score distribution plot and
    tier selection heatmap (IEEE comparative figures).

OUTPUT: PlacementDecision
    selected_node_id, selected_tier, selected_hw_class,
    topsis_cc  [0,1],  placement_score (= topsis_cc × 100),
    topsis_ranking, latency_estimate_ms, cpu_fit_pct, ram_fit_pct,
    admission_status, rejection_reason, alternatives_tried, placement_ms


===========================================================================
AGENT 6 — OffloadAgent
ALGORITHMS: EMA-Triggered Threshold Control Policy
            + Least-Loaded (LL) Routing
===========================================================================

PURPOSE
    Reactive, event-driven agent that checks for overload conditions
    after each placement and initiates workload migration to preserve SLA.
    This makes the system adaptive, not just reactive at placement time.

ALGORITHM 6A — EMA-Triggered Threshold Policy
    Trigger condition:
        IF  EMA_CPU(tier) > OFFLOAD_CPU_THR (default 80%)
        OR  EMA_RAM(tier) > OFFLOAD_RAM_THR (default 85%)
        AND the node hosts at least one active application
        THEN trigger offload from that node.

    EMA values come from Agent 4 — NOT raw instantaneous CPU.
    Using EMA prevents thrashing. A transient CPU spike will not
    trigger migration; only sustained high load triggers offload.
    This is equivalent to hysteresis in a threshold controller.

ALGORITHM 6B — Least-Loaded (LL) Routing
    Among all candidate target nodes in the escalation tier:
        Select the node with MINIMUM EMA_CPU.

    Least-Loaded is proven optimal for M/M/k queuing systems.
    For bursty IoT workloads, LL outperforms Round-Robin and Random
    because it responds to actual node load state.

ESCALATION PATH
    local → edge → fog → cloud
    If next tier has no available nodes → peer offload attempted
    (same tier, different node, also selected by LL).

MIGRATION LIMIT
    Up to 2 apps migrated per offload call to prevent over-migration.
    If target refuses allocation → rollback (source re-allocates app).

OUTPUT: OffloadDecision
    triggered, trigger_reason ("ema_cpu" | "ema_ram"),
    source_node_id, source_ema_cpu, source_ema_ram,
    target_node_id, target_tier, target_ema_cpu,
    apps_offloaded, offload_latency_ms, success


===========================================================================
OUTPUT FILES
===========================================================================

experiment_decisions.json
    Full AgentDecision record for every application.
    Contains all six agents' outputs in one JSON array.
    Read by baselines.py and figures.py.

experiment_applications.json    written by applications.py
experiment_infrastructure.json  written by computing_resources.py


===========================================================================
HOW TO RUN IN PYCHARM
===========================================================================

STANDALONE TEST (test agents.py alone without OrchestratorAgent):
    1. Put applications.py, computing_resources.py, agents.py
       in the same project folder.
    2. Set LLM_BACKEND = "mock"  (no model needed).
    3. Right-click agents.py → Run.
    4. __main__ calls applications.py and computing_resources.py
       automatically, then runs the 6-agent pipeline on 10 apps.
    5. Results printed to console + saved to experiment_decisions.json.

FULL SYSTEM RUN (via OrchestratorAgent.py):
    1. Configure settings in OrchestratorAgent.py.
    2. Run OrchestratorAgent.py only.
    3. It calls agents.run(config, shared_state) in the chain.

WITH OLLAMA (recommended for paper-quality results):
    1. Install Ollama: https://ollama.com
    2. ollama pull llama3.2:8b-instruct-q4_K_M
    3. Ollama starts automatically as a background service.
    4. Set LLM_BACKEND = "ollama" in agents.py.
    5. Run normally.


===========================================================================
DEPENDENCIES
===========================================================================

agents.py uses ONLY Python standard library:
    json, math, re, time, random, collections, dataclasses, enum, typing

    No pip install required for agents.py itself.

For LLM_BACKEND = "ollama":
    Install Ollama desktop app from https://ollama.com  (free, no pip).

For LLM_BACKEND = "llama_cpp"  (added in later step):
    pip install llama-cpp-python


===========================================================================
IEEE PAPER — CONTRIBUTION OF EACH AGENT
===========================================================================

Agent 1  Two-stage TF-IDF → LLM design removes need for hand-crafted
         intent classifiers. TF-IDF prior reduces LLM hallucination.
         Enables zero-shot generalisation to new application domains.

Agent 2  Fuzzy logic handles vagueness of IoT urgency classification
         scientifically — crisp thresholds cannot. Weighted scoring
         produces a formal, citable QoS contract per application.

Agent 3  NB proves compute-need inference from natural language —
         no manual application profiling required. Conditional LLM
         refinement adds accuracy only where NB is uncertain.

Agent 4  Persistent EMA gives a rolling infrastructure view rather than
         point-in-time snapshots. Prevents thrashing in placement and
         offload decisions. Standard in cloud monitoring literature.

Agent 5  TOPSIS with 5 IoT-specific criteria is the core placement
         contribution. Intent-aware (criteria derived from Agents 1–3),
         not just resource-aware. Directly compared to 4 baselines.

Agent 6  EMA-triggered, LL-routed offload provides adaptive SLA
         preservation post-placement. EMA hysteresis is a direct
         improvement over static threshold baselines (Baseline B3).

===========================================================================
END OF README
===========================================================================
