===========================================================================
FILE:    OrchestratorAgent.py
PROJECT: Intent-Aware Multi-Agent AI Orchestration
         in the Edge-Cloud Continuum
===========================================================================

OVERVIEW
--------
OrchestratorAgent.py is the MAIN ENTRY POINT of the entire system.
This is the ONLY file you run. It calls all other modules in sequence,
deploys all applications to computing nodes, runs baseline comparisons,
and saves all results for IEEE graph generation.

YOU ONLY RUN THIS FILE:
    python OrchestratorAgent.py

All other files (applications.py, computing_resources.py, agents.py,
baselines.py) are called automatically by this file.


===========================================================================
WHAT HAPPENS WHEN YOU RUN THIS FILE
===========================================================================

Step 1 — applications.py is called
    Generates NUM_APPS application instances with realistic intent
    sentences, computing requirements, and QoS parameters.
    Writes → shared_state["applications"]

Step 2 — computing_resources.py is called
    Builds the full edge-cloud infrastructure: cloud datacenter,
    fog gateway nodes, edge compute nodes, and local Pi devices.
    Writes → shared_state["infrastructure"]

Step 3 — agents.py is called
    Runs the 6-agent ML pipeline on EVERY application instance.
    Each agent performs its individual AI/ML function:
        Agent 1  TF-IDF + LLM     → intent parsing
        Agent 2  Fuzzy Logic       → QoS profiling
        Agent 3  Naive Bayes + LLM → classification
        Agent 4  EMA monitoring    → resource snapshot
        Agent 5  TOPSIS            → optimal node selection
        Agent 6  Least-Loaded      → offload if overloaded
    DEPLOYS each application to the best computing node.
    Writes → shared_state["agent_decisions"]
    Writes → shared_state["active_placements"]

Step 4 — baselines.py is called
    Runs 4 traditional baselines on the SAME applications and
    the SAME infrastructure (deep copy — fair comparison):
        B1  Cloud-Only
        B2  Round-Robin
        B3  Static Threshold
        B4  Greedy First-Fit
    Writes → shared_state["baseline_results"]
    Writes → shared_state["comparison_table"]

Step 5 — Final results printed
    Full deployment table per application.
    Node utilisation after deployment.
    IEEE comparison table (proposed vs 4 baselines).

Step 6 — All JSON files saved to OUTPUT_DIR
    experiment_applications.json
    experiment_infrastructure.json
    experiment_decisions.json
    experiment_baselines.json
    experiment_comparison_table.json
    experiment_master.json


===========================================================================
HOW MANY APPLICATIONS ARE DEPLOYED
===========================================================================

Exactly NUM_APPS applications are deployed per run.

The multi-agent system deploys all NUM_APPS applications.
Each of the 4 baselines also processes the same NUM_APPS applications.

Total deployments per experiment run:
    NUM_APPS × 5  (1 proposed system + 4 baselines)

Example with NUM_APPS = 30:
    Multi-agent deploys   : 30 apps
    Cloud-Only deploys    : 30 apps (same 30)
    Round-Robin deploys   : 30 apps (same 30)
    Static-Threshold      : 30 apps (same 30)
    Greedy-First-Fit      : 30 apps (same 30)
    ─────────────────────────────
    Total decisions made  : 150


===========================================================================
SHARED_STATE MECHANICS
===========================================================================

One dictionary (shared_state) is passed through every module.
Each module reads from it and writes to it.

    shared_state = {}   ← starts empty

After Step 1:
    shared_state["applications"]      list[ApplicationInstance]
    shared_state["app_summary"]       dict  {domains, subtypes, total}

After Step 2:
    shared_state["infrastructure"]    Infrastructure object
    shared_state["infra_summary"]     dict  {tier: {node_count, cpu, ram...}}

After Step 3:
    shared_state["agent_decisions"]   list[AgentDecision]
    shared_state["active_placements"] dict  {instance_id: (app, ComputeNode)}

After Step 4:
    shared_state["baseline_results"]  dict  {name: list[BaselineResult]}
    shared_state["comparison_table"]  dict  (fed to figures.py)

The shared_state is also returned by main() so figures.py can read it
when called as the next step after OrchestratorAgent.py completes.


===========================================================================
CONFIGURATION SETTINGS  (top of OrchestratorAgent.py)
===========================================================================

This is the SINGLE place to configure the entire experiment.
All settings flow down to every module automatically via config dict.
You do NOT need to edit any other file for routine experiments.

── APPLICATION SETTINGS ────────────────────────────────────────────────

NUM_APPS : int   (default 30)
    Total application instances to generate and deploy.
    10  = quick test (runs in seconds)
    30  = standard experiment (recommended for paper results)
    100 = large-scale experiment
    200+ = stress test

APP_DOMAINS : list[str]
    Which domains to include.
    ["healthcare", "industrial", "general"]  = all three (default)
    ["healthcare"]                           = healthcare only
    ["healthcare", "industrial"]             = two domains

APP_SUBTYPES : dict
    Which subtypes within each domain.
    None = include ALL subtypes for that domain (default).
    ["emergency_care", "remote_monitoring"] = specific subtypes only.

REQUIREMENT_MODE : str
    "random"   Sample from [min, max] window — realistic heterogeneity
    "nominal"  Always use midpoint — controlled deterministic runs
    "worst"    Always use maximum — stress test / worst-case analysis

RANDOM_SEED : int | None
    42    = reproducible (same result every run — use for IEEE paper)
    None  = fully random (different every run)

── INFRASTRUCTURE SETTINGS ─────────────────────────────────────────────

ACTIVE_TIERS : list[str]
    Which computing tiers to include.
    ["cloud","fog","edge","local"]  = full four-tier (default)
    ["fog","edge"]                  = no cloud, no local
    ["cloud"]                       = cloud-only simulation

NUM_FOG_NODES : int   (default 2)
    Number of fog gateway server nodes.
    Increase to 4 or 8 for scalability experiments.

NUM_EDGE_NODES : int   (default 4)
    Number of edge compute nodes.
    Increase to 8 or 16 for dense edge experiments.

NUM_LOCAL_NODES : int   (default 2)
    Number of local Pi/Jetson devices.
    Set to 0 to exclude local tier without removing from ACTIVE_TIERS.

NODE_VARIATION : float   (default 0.15)
    ±15% random variation in each node's specs.
    0.0  = all nodes identical (controlled experiment)
    0.15 = realistic heterogeneous deployment (default)
    0.30 = highly heterogeneous nodes

── AGENT / LLM SETTINGS ────────────────────────────────────────────────

LLM_BACKEND : str
    "mock"       No model needed. Runs immediately offline.
                 Use this for development and first-time testing.

    "ollama"     Uses local Ollama server (recommended for paper results).
                 Install: https://ollama.com
                 Pull model: ollama pull llama3.2:8b-instruct-q4_K_M
                 Start Ollama before running OrchestratorAgent.py.

    "llama_cpp"  Direct llama-cpp-python (added in later step).

LLM_MODEL : str
    Ollama model tag (used when LLM_BACKEND = "ollama").
    Default: "llama3.2:8b-instruct-q4_K_M"

AGENT_VERBOSE : bool
    True  = print detailed per-agent reasoning for every app.
            Shows TF-IDF scores, fuzzy sets, NB posteriors,
            EMA values, TOPSIS rankings, offload decisions.
    False = silent agents, only deployment table printed.
            Recommended when NUM_APPS > 50.

── BASELINE SETTINGS ───────────────────────────────────────────────────

ACTIVE_BASELINES : list[str]
    Which baselines to run.
    Default: all four.
    Remove any entry to skip that baseline.

    "cloud_only"       B1 — all apps to cloud
    "round_robin"      B2 — cyclic node distribution
    "static_threshold" B3 — fixed 70% CPU threshold
    "greedy_first_fit" B4 — first node with enough resources

BASELINE_VERBOSE : bool
    True  = print each baseline placement decision.
    False = silent baselines, only comparison table printed (default).

── OUTPUT SETTINGS ─────────────────────────────────────────────────────

OUTPUT_DIR : str   (default "./experiment_results")
    Folder where all JSON output files are saved.
    Created automatically if it does not exist.
    Change for different experiment runs:
        "./experiment_results_healthcare_only"
        "./experiment_results_30apps_seed42"

PRINT_PER_APP : bool   (default True)
    True  = print one line per application in the deployment table.
    False = print aggregate stats only.
            Recommended when NUM_APPS > 100.


===========================================================================
HOW TO RUN IN PYCHARM — STEP BY STEP
===========================================================================

FIRST TIME SETUP:
    1. Create a new PyCharm project folder.
    2. Copy ALL these files into the same folder:
           OrchestratorAgent.py    ← main file
           applications.py
           computing_resources.py
           agents.py
           baselines.py
    3. Open OrchestratorAgent.py in PyCharm.

QUICK TEST (no model needed, runs in seconds):
    1. Set  LLM_BACKEND = "mock"   (already default)
    2. Set  NUM_APPS    = 10       (small for quick test)
    3. Set  AGENT_VERBOSE = False  (cleaner output)
    4. Right-click OrchestratorAgent.py → Run
    5. See deployment results and comparison table in terminal.
    6. Check ./experiment_results/ folder for JSON files.

STANDARD EXPERIMENT (for IEEE paper results):
    1. Install Ollama from https://ollama.com
    2. In terminal: ollama pull llama3.2:8b-instruct-q4_K_M
    3. Set  LLM_BACKEND = "ollama"
    4. Set  NUM_APPS    = 30
    5. Set  RANDOM_SEED = 42       (reproducible)
    6. Run OrchestratorAgent.py

HEALTHCARE-ONLY EXPERIMENT:
    1. Set  APP_DOMAINS   = ["healthcare"]
    2. Set  NUM_APPS      = 20
    3. Run OrchestratorAgent.py

SCALABILITY EXPERIMENT (vary number of nodes):
    Run 1:  NUM_FOG_NODES=1,  NUM_EDGE_NODES=2
    Run 2:  NUM_FOG_NODES=2,  NUM_EDGE_NODES=4  (default)
    Run 3:  NUM_FOG_NODES=4,  NUM_EDGE_NODES=8
    Compare results across runs using figures.py


===========================================================================
OUTPUT FILES — WHAT EACH CONTAINS
===========================================================================

All files saved to OUTPUT_DIR (default: ./experiment_results/)

experiment_applications.json
    Every generated ApplicationInstance object serialised to JSON.
    Contains: instance_id, domain, subtype, intent_text, cpu_cores,
    ram_mb, gpu_memory_mb, bandwidth_kbps, latency_ms_max,
    qos_priority, criticality_level, real_time, offload_eligible.

experiment_infrastructure.json
    Snapshot of all nodes at experiment start.
    Contains: per-tier node specs, capacity, latency, LLM model.

experiment_decisions.json
    Full AgentDecision for every application from the multi-agent system.
    Contains all six agents' outputs per app:
        intent_profile, qos_profile, classification,
        resource_snapshot, placement, offload, final_status,
        total_pipeline_ms, topsis_cc, nb_confidence.

experiment_baselines.json
    Per-baseline list of BaselineResult for every application.
    Structure: {"cloud_only": [...], "round_robin": [...], ...}

experiment_comparison_table.json
    Unified comparison dict with these metrics per system:
        placed_pct, rejected_pct, avg_latency_ms,
        avg_decision_ms, avg_score, edge_pct, cloud_pct,
        tier_distribution.
    This is the PRIMARY INPUT for figures.py IEEE graphs.

experiment_master.json
    Full experiment configuration + all summary statistics.
    Records: timestamp, all config settings, total elapsed time,
    app_summary, infra_summary, comparison_table.
    Use this to reproduce any experiment exactly.


===========================================================================
TERMINAL OUTPUT EXPLAINED
===========================================================================

When you run OrchestratorAgent.py you will see:

  ████ Banner ████
    Shows all config settings at a glance before the run starts.

  ══ STEP 1 — Generating applications ══
    Confirms how many apps were generated and which domains/subtypes.

  ══ STEP 2 — Building infrastructure ══
    Shows each tier: node count, total CPU/RAM/GPU, latency to next tier.

  ══ STEP 3 — Running 6-agent pipeline ══
    If AGENT_VERBOSE=True: detailed per-agent output for every app.
    Shows TF-IDF scores, fuzzy memberships, NB posteriors, TOPSIS CC.
    Ends with deployment table: one row per app with tier and node chosen.

  Node utilisation after deployment:
    Shows CPU% bar chart for every node — lets you see which nodes
    are busy and which are idle after all apps are deployed.

  ══ STEP 4 — Running baselines ══
    Each baseline processes all apps and prints a summary line.

  ══ STEP 5 — Final comparison table ══
    The key IEEE table:
        System            Placed%  Lat(ms)  Score  Edge%  Cloud%  Dec(ms)
        multi_agent_TOPSIS  93.3%    8.23ms  87.4   66.7%   6.7%   45.2ms ◄
        B1_Cloud-Only      100.0%   35.00ms  100.0   0.0% 100.0%    0.1ms
        B2_Round-Robin      80.0%   22.50ms   50.0  40.0%  25.0%    0.1ms
        B3_Static-Threshold 86.7%   12.10ms   72.3  53.3%  13.3%    0.2ms
        B4_Greedy-First-Fit 93.3%   10.40ms   61.8  60.0%   6.7%    0.1ms

  ████ EXPERIMENT COMPLETE ████
    Total elapsed time, placement rate, list of all saved files.


===========================================================================
PROJECT FILE STRUCTURE
===========================================================================

Your PyCharm project folder should look like this:

    MyProject/
    ├── OrchestratorAgent.py        ← RUN THIS FILE
    ├── applications.py             ← Step 1: app generation
    ├── computing_resources.py      ← Step 2: infrastructure
    ├── agents.py                   ← Step 3: 6-agent ML pipeline
    ├── baselines.py                ← Step 4: traditional baselines
    │
    ├── applications_README.txt
    ├── computing_resources_README.txt
    ├── agents_README.txt
    ├── baselines_README.txt
    ├── OrchestratorAgent_README.txt
    │
    └── experiment_results/         ← created automatically on first run
        ├── experiment_applications.json
        ├── experiment_infrastructure.json
        ├── experiment_decisions.json
        ├── experiment_baselines.json
        ├── experiment_comparison_table.json
        └── experiment_master.json


===========================================================================
DEPENDENCIES
===========================================================================

OrchestratorAgent.py uses ONLY Python standard library:
    os, sys, json, time, traceback, datetime, collections

No pip install required for OrchestratorAgent.py itself.

The full project (all 5 files) also only needs standard library
UNLESS you use LLM_BACKEND = "ollama" or "llama_cpp":

    LLM_BACKEND = "mock"      → no installation needed
    LLM_BACKEND = "ollama"    → install Ollama app (free, no pip)
                                 https://ollama.com
    LLM_BACKEND = "llama_cpp" → pip install llama-cpp-python
                                 (added in a later step)


===========================================================================
NEXT STEP — figures.py
===========================================================================

After running OrchestratorAgent.py, all results are saved to
experiment_results/. The next module to build is figures.py which reads
experiment_comparison_table.json and experiment_decisions.json and
generates all IEEE-quality publication graphs:

    Fig 1   CDF of latency per system
    Fig 2   Box plot of latency per tier
    Fig 3   Latency vs load (scalability)
    Fig 4   Routing decision heatmap
    Fig 5   Confusion matrix (intended vs actual tier)
    Fig 7   CPU/RAM utilisation over time (stacked area)
    Fig 13  Grouped bar chart (proposed vs 4 baselines)
    Fig 14  Radar chart (5-metric spider per system)
    Fig 15  Ablation study bar chart
    Fig 16  Sensor anomaly time series
    Fig 18  Agent decision Gantt chart

===========================================================================
END OF README
===========================================================================
