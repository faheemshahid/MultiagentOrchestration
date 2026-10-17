===========================================================================
FILE:    baselines.py
PROJECT: Intent-Aware Multi-Agent AI Orchestration
         in the Edge-Cloud Continuum
===========================================================================

OVERVIEW
--------
baselines.py implements four traditional IoT service orchestration
algorithms used as comparison benchmarks against the proposed multi-agent
system (agents.py) in the IEEE paper evaluation.

The core principle of fair comparison:
    - All four baselines use the IDENTICAL applications from applications.py
    - All four baselines use the IDENTICAL infrastructure from computing_resources.py
    - Each baseline gets a FRESH DEEP COPY of the infrastructure
      so one baseline's allocations never affect another's results
    - The only variable between systems is the orchestration algorithm itself

Results from all four baselines plus the multi-agent system are combined
into a single comparison_table dict that figures.py reads to generate
all IEEE comparative graphs.


===========================================================================
WHY THESE FOUR BASELINES?
===========================================================================

Each baseline represents a real category of IoT orchestration approach
that exists in practice or in the literature. Together they cover the
full spectrum from the simplest possible strategy to rule-based systems,
giving the paper a complete and convincing comparative evaluation.

B1  Cloud-Only
    Represents: Legacy centralised cloud IoT (pre-edge era).
    Real examples: AWS IoT Core + Lambda, Azure IoT Hub.
    Why include it: Shows the cost of ignoring edge computing entirely.
                    Expected to have worst latency for time-critical apps.

B2  Round-Robin
    Represents: Simple load distribution with no intelligence.
    Real examples: Basic MQTT broker routing, early k8s scheduling
                   without resource requests.
    Why include it: The zero-intelligence baseline. Any intelligent
                    system must beat random cycling to justify its cost.

B3  Static Threshold
    Represents: Rule-based IoT middleware with capacity management.
    Real examples: Eclipse Kura CPU threshold policies, OpenFog
                   reference architecture admission control.
    Why include it: Shows the limitation of fixed rules with no ML,
                    no EMA smoothing, no latency awareness.

B4  Greedy First-Fit
    Represents: Bin-packing heuristic from early edge computing papers.
    Real examples: First-Fit Decreasing (FFD) is the canonical baseline
                   in resource allocation research.
    Why include it: Has resource awareness but no optimisation or intent.
                    Direct comparison shows value of TOPSIS over greedy.


===========================================================================
HOW FILES CONNECT
===========================================================================

baselines.py is the fourth module called by OrchestratorAgent.py:

    applications.run()          → shared_state["applications"]
    computing_resources.run()   → shared_state["infrastructure"]
    agents.run()                → shared_state["agent_decisions"]
    baselines.run()             → reads all three above
                                  writes shared_state["baseline_results"]
                                  writes shared_state["comparison_table"]
    figures.run()               → reads shared_state["comparison_table"]
                                  generates all IEEE comparison graphs

When run STANDALONE (python baselines.py), the __main__ block calls
applications.run(), computing_resources.run(), and agents.run() first —
exactly what OrchestratorAgent.py does. Same code path.


===========================================================================
CONFIGURATION SETTINGS  (top of baselines.py)
===========================================================================

ACTIVE_BASELINES : list[str]
    Controls which baselines run in this experiment.
    Default: all four active.

    To run all four:
        ACTIVE_BASELINES = ["cloud_only","round_robin",
                            "static_threshold","greedy_first_fit"]

    To run only two for a quick test:
        ACTIVE_BASELINES = ["cloud_only", "greedy_first_fit"]

    To exclude one:
        Comment out any entry from the list.

STATIC_THRESHOLD_CPU : float   (default 70.0)
    The CPU utilisation percentage used by Baseline B3 (Static Threshold).
    B3 places an app on the first node whose raw CPU < this value.
    Changing this value simulates more or less aggressive admission control.
    Range: 50.0 – 90.0

    Note: B3 uses RAW CPU, not EMA. This is intentional — it demonstrates
    the weakness of threshold-based systems without smoothing.

BASELINE_VERBOSE : bool   (default True)
    True  = print each placement decision to the console as it happens.
    False = silent. Only the final comparison table is printed.
            Use False for large experiments (NUM_APPS > 100).


===========================================================================
BASELINE B1 — Cloud-Only
===========================================================================

STRATEGY
    Route every single application to the cloud datacenter node,
    regardless of urgency, latency requirements, domain, or resource fit.
    If the cloud node is full, the application is rejected.

REPRESENTS
    Legacy centralised cloud IoT orchestration before edge computing
    became mainstream. All processing happens in the datacenter.

ALGORITHM
    1. Get cloud_nodes from infrastructure (always just one datacenter).
    2. Call node.can_accommodate(app).
    3. If yes → allocate and mark placed.
    4. If no  → mark rejected with reason.

SCORING
    placement_score = 100.0 for all placed apps.
    Cloud-only always "succeeds by design" when the node has capacity.
    Score does not reflect quality — it reflects the strategy's intent.

EXPECTED RESULTS IN IEEE COMPARISON
    Highest avg_latency_ms   — cloud has 35ms+ WAN latency
    Lowest  edge_pct         — 0% edge placement
    Highest cloud_pct        — 100% cloud placement
    Worst   SLA compliance   — critical apps (10ms deadline) all violated
    High    rejection_pct    — cloud node saturates under heavy load

WHERE THIS APPEARS IN IEEE PAPER
    Fig 13 (grouped bar chart): worst latency bar
    Fig 14 (radar chart):       weakest on latency and edge utilisation axes
    Discussion: motivates why edge-cloud continuum is needed


===========================================================================
BASELINE B2 — Round-Robin
===========================================================================

STRATEGY
    Build a flat ordered list of ALL nodes across ALL active tiers.
    Distribute applications cyclically — node pointer advances by 1
    after each placement. No resource awareness, no QoS consideration,
    no intent parsing.

REPRESENTS
    Simple load distribution middleware. Treats all applications and
    all nodes as identical. No intelligence whatsoever.

ALGORITHM
    1. At initialise(), build flat node list: [local nodes, edge nodes,
       fog nodes, cloud nodes] in infrastructure order.
    2. For each app, try node at current pointer position.
    3. If node is offline or overloaded, skip and try next.
    4. Call node.allocate(app).
    5. Advance pointer by 1 (wraps around).
    6. If all nodes tried and failed → rejected.

SCORING
    placement_score = 50.0 for all placed apps.
    Round-Robin has no optimisation — the fixed score reflects that
    placement quality is arbitrary, not computed.

EXPECTED RESULTS IN IEEE COMPARISON
    High    rejection_pct     — places apps on nodes that cannot fit them
    Mixed   avg_latency_ms    — depends on which node pointer lands on
    Poor    SLA compliance    — critical apps may land on cloud by chance
    Low     avg_score         — no optimisation, arbitrary placement

WHERE THIS APPEARS IN IEEE PAPER
    Fig 13: worst or near-worst placement score bar
    Fig 14: weakest on score and SLA compliance axes
    Discussion: zero-intelligence lower bound — any ML system must beat this


===========================================================================
BASELINE B3 — Static Threshold
===========================================================================

STRATEGY
    Scan nodes in fixed tier order: edge → fog → cloud → local.
    Place on the first node whose RAW CPU utilisation < 70% AND has
    sufficient free CPU cores and RAM. No ML, no latency check,
    no urgency awareness, no GPU consideration.

REPRESENTS
    Rule-based IoT middleware with simple capacity management.
    The most common orchestration approach in deployed IoT systems today.

ALGORITHM
    1. Scan tiers in order: edge → fog → cloud → local.
    2. For each node, check:
         node.cpu_utilisation_pct < STATIC_THRESHOLD_CPU (70%)
         node.available_ram_mb >= app.ram_mb
         node.available_cpu_cores >= app.cpu_cores
    3. First node passing all checks → allocate and mark placed.
    4. If no node passes → rejected.

IMPORTANT DISTINCTION FROM AGENT 6 (OffloadAgent)
    B3 uses RAW instantaneous CPU — not EMA.
    This means B3 is susceptible to thrashing: a node that briefly
    spikes to 71% CPU will be skipped even if its average load is 30%.
    Agent 6 uses EMA (α=0.3) which smooths this out.
    This difference is a direct contribution claim in the paper.

SCORING
    placement_score = 100.0 - node.cpu_utilisation_pct
    Higher score = more CPU headroom on the chosen node.
    Simple heuristic — not a multi-criteria optimisation.

EXPECTED RESULTS IN IEEE COMPARISON
    Moderate placed_pct      — misses some valid placements due to threshold
    Edge-biased placement    — scans edge first so most apps land there
    No latency enforcement   — critical apps land wherever threshold passes
    Higher rejection_pct     — threshold too conservative in some cases

WHERE THIS APPEARS IN IEEE PAPER
    Fig 13: moderate score, no latency awareness bar
    Fig 15 (ablation): removing EMA from Agent 6 converges to this
    Discussion: shows why EMA threshold beats static threshold


===========================================================================
BASELINE B4 — Greedy First-Fit
===========================================================================

STRATEGY
    Scan ALL nodes across ALL tiers in fixed order: edge → fog → cloud → local.
    Place on the first node that passes node.can_accommodate(app) —
    which checks CPU, RAM, GPU, bandwidth, AND latency deadline.
    No scoring, no optimisation, no intent awareness.

REPRESENTS
    First-Fit bin-packing heuristic. The canonical baseline in resource
    allocation research. Has resource awareness but no intelligence
    about urgency, domain, or placement quality.

ALGORITHM
    1. Scan tiers in order: edge → fog → cloud → local.
    2. For each node, call node.can_accommodate(app).
       This checks all resource dimensions simultaneously:
         - available_cpu_cores >= app.cpu_cores
         - available_ram_mb >= app.ram_mb
         - available_gpu_vram_mb >= app.gpu_memory_mb
         - available_uplink_kbps >= app.bandwidth_kbps
         - latency_to_next_ms <= app.latency_ms_max
    3. First node passing → allocate and mark placed.
    4. If no node passes → rejected.

KEY DIFFERENCE FROM MULTI-AGENT TOPSIS
    Greedy First-Fit finds ANY feasible node.
    TOPSIS finds the BEST feasible node across all criteria.
    A node may be feasible (passes can_accommodate) but still be a
    poor choice — e.g. a fog node used for a critical app when an
    edge node with lower latency is available and equally feasible.
    TOPSIS would select the edge node; Greedy would take the fog node.

SCORING
    placement_score = (available_cpu_cores / total_cpu_cores) × 100
    Measures CPU availability fraction at the chosen node.
    Greedy tends to fill first nodes completely (fragmentation).

EXPECTED RESULTS IN IEEE COMPARISON
    High    placed_pct       — resource checks are comprehensive
    Mixed   avg_latency_ms   — latency checked but not optimised
    Low     avg_score        — first-fit causes fragmentation
    Moderate edge_pct        — edge scanned first, many apps land there

WHERE THIS APPEARS IN IEEE PAPER
    Fig 13: good admission rate but low score bar
    Fig 14: reasonable on placed_pct but weak on optimisation axes
    Discussion: shows why TOPSIS over greedy is necessary


===========================================================================
COMPARISON TABLE — METRICS EXPLAINED
===========================================================================

build_comparison_table() produces a unified dict with these metrics
for EVERY system (multi-agent + all 4 baselines):

placed_pct
    Percentage of applications successfully placed.
    Higher is better. Measures admission control effectiveness.

rejected_pct
    Percentage of applications that could not be placed.
    Lower is better. 100% - placed_pct.

avg_latency_ms
    Average one-way propagation latency of the chosen node.
    Lower is better. Key metric for time-critical IoT.
    This is the latency_to_next_ms of the selected node —
    the network latency from that node to the next tier.

avg_decision_ms
    Average time the algorithm took to make one placement decision.
    Lower is better. Measures computational overhead of the approach.
    Multi-agent will be highest (runs 6 agents) but produces best quality.

avg_score
    Average placement_score across all placed applications.
    Higher is better. Measures placement quality.
    Multi-agent: TOPSIS CC × 100  (principled optimisation score)
    B1: fixed 100 (no meaning)
    B2: fixed 50  (no meaning)
    B3: CPU headroom (simple heuristic)
    B4: CPU availability fraction (simple heuristic)

edge_pct
    Percentage of placed apps that landed on edge or local nodes.
    Higher = more edge utilisation = lower latency, lower cloud cost.
    Multi-agent expected highest — intent-aware tier selection.
    B1 expected lowest (0%) — everything goes to cloud.

cloud_pct
    Percentage of placed apps that landed on cloud.
    Lower is better for latency-sensitive workloads.
    B1 = 100%. Multi-agent = lowest (only complex/background jobs).

tier_distribution
    Full breakdown: {edge: N, fog: N, cloud: N, local: N}
    Used by figures.py for the stacked bar tier distribution chart.


===========================================================================
OUTPUT FILES
===========================================================================

experiment_baselines.json
    Per-baseline list of BaselineResult records for every application.
    Structure: { "cloud_only": [...], "round_robin": [...], ... }
    Used by figures.py for per-baseline CDF and distribution plots.

experiment_comparison_table.json
    The unified comparison table dict.
    Primary input for all IEEE comparative figures in figures.py.

Both files are saved to output_dir (default: current folder).


===========================================================================
HOW TO RUN IN PYCHARM
===========================================================================

STANDALONE TEST (test baselines.py alone):
    1. Place applications.py, computing_resources.py, agents.py,
       and baselines.py in the same project folder.
    2. Run baselines.py directly.
    3. __main__ calls all upstream modules automatically,
       runs all 4 baselines + the multi-agent pipeline,
       prints the comparison table to console,
       saves both output JSON files.

FULL SYSTEM RUN (via OrchestratorAgent.py):
    1. Configure settings in OrchestratorAgent.py.
    2. Run OrchestratorAgent.py only.
    3. It calls baselines.run(config, shared_state) automatically.

TO EXCLUDE A BASELINE:
    Comment out its name in ACTIVE_BASELINES at the top of baselines.py.
    Or pass active_baselines in the config dict from OrchestratorAgent.py.


===========================================================================
DEPENDENCIES
===========================================================================

baselines.py uses ONLY Python standard library:
    json, time, os, copy, dataclasses, typing

No pip install required.


===========================================================================
IEEE PAPER — HOW BASELINES APPEAR IN EACH FIGURE
===========================================================================

Fig 13 — Grouped Bar Chart (latency / score / placed_pct per system)
    X-axis: the 5 systems (B1, B2, B3, B4, Multi-Agent)
    Y-axis: each metric in a separate group of bars
    Multi-agent bar is highlighted / marked with ◄
    Expected outcome: multi-agent wins on latency and score

Fig 14 — Radar Chart (5-metric spider per system)
    5 axes: placed_pct, avg_latency (inverted), avg_score,
            edge_pct, avg_decision_ms (inverted)
    Multi-agent polygon should encompass the largest area
    B1 strong on placed_pct only, weak on all latency axes

Fig 15 — Ablation Study (remove one agent at a time)
    Compares full multi-agent vs versions with one agent removed
    B3 (Static Threshold) approximates removing Agent 6 (EMA offload)
    B4 (Greedy First-Fit) approximates removing Agent 5 (TOPSIS)
    This directly proves the contribution of each agent

Fig 1 — CDF of latency
    Cumulative distribution of avg_latency_ms across all placements
    Multi-agent curve furthest left (lowest latency)
    B1 curve furthest right (highest latency, all cloud)

Fig 6 — Escalation Rate
    Only multi-agent produces this — baselines have no escalation logic
    Used to show how the system adapts over time

===========================================================================
END OF README
===========================================================================
