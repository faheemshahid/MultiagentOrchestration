===========================================================================
FILE:    applications.py
PROJECT: Intent-Aware Multi-Agent AI Orchestration
         in the Edge-Cloud Continuum
===========================================================================

OVERVIEW
--------
applications.py defines all IoT application types used in the experiment.
It generates realistic ApplicationInstance objects with natural-language
intent sentences, computing requirements, and QoS parameters.

This is the FIRST module called by OrchestratorAgent.py. All downstream
modules (computing_resources.py, agents.py, baselines.py, figures.py)
read the application instances it produces via shared_state.


===========================================================================
WHAT THIS FILE DOES
===========================================================================

1. DEFINES THREE APPLICATION DOMAINS
   Each domain has three subtypes — nine application types in total.

   HEALTHCARE
       emergency_care         Time-critical, life-safety, edge-preferred
       medical_imaging        Resource-intensive, GPU-heavy, cloud-preferred
       remote_monitoring      Continuous stream, balanced, fog-preferred

   INDUSTRIAL (IIoT)
       predictive_maintenance Real-time vibration/thermal analysis, edge
       quality_control        High-throughput vision inspection, fog/GPU
       process_control        Safety-critical closed-loop control, edge

   GENERAL (cross-domain)
       federated_learning     Distributed ML training, cloud/adaptive
       digital_twin           LLM-powered simulation, fog/adaptive
       smart_environment      Multi-sensor ambient monitoring, fog

2. GENERATES REALISTIC APPLICATION INSTANCES
   Each instance has:
       - A natural-language intent sentence (the human input the LLM reads)
       - Sampled computing requirements (CPU, GPU, RAM, storage)
       - Sampled network requirements (bandwidth, latency deadline)
       - QoS parameters (urgency level, tier preference, criticality)
       - IEEE metadata (criticality level 1-5, real_time flag, offload flag)

3. PROVIDES AN LLM PROMPT METHOD
   Each ApplicationInstance has a .llm_prompt() method that builds the
   structured prompt sent to the Llama model in Agent 1 and Agent 3.

4. SAVES TO JSON
   Saves all generated instances to experiment_applications.json
   for downstream agents and IEEE result logging.


===========================================================================
CONFIGURATION SETTINGS  (top of applications.py — edit before each run)
===========================================================================

APP_DOMAINS : list[str]
    Which domains to include in the experiment.
    Default: all three domains active.

    All three:
        APP_DOMAINS = ["healthcare", "industrial", "general"]

    Healthcare only:
        APP_DOMAINS = ["healthcare"]

    Two domains:
        APP_DOMAINS = ["healthcare", "industrial"]

APP_SUBTYPES : dict[str, list | None]
    Which subtypes to include within each domain.
    None = include ALL subtypes for that domain.

    All subtypes in all domains (default):
        APP_SUBTYPES = {
            "healthcare": None,
            "industrial": None,
            "general":    None,
        }

    Only specific subtypes:
        APP_SUBTYPES = {
            "healthcare": ["emergency_care", "remote_monitoring"],
            "industrial": ["predictive_maintenance"],
            "general":    None,
        }

NUM_APPS : int   (default 30)
    Total number of application instances to generate.
    The generator samples randomly from eligible specs.
    Increase for larger experiments: 50, 100, 200.
    Decrease for quick tests: 5, 10.

RANDOMISE_SEED : int | None   (default 42)
    Integer  = reproducible experiments (same seed = same apps every run).
               Use this for IEEE paper results so reviewers can reproduce.
    None     = fully random (different apps every run).

REQUIREMENT_MODE : str   (default "random")
    Controls how resource requirement values are sampled.

    "random"   Sample uniformly from [min, max] window.
               Use for realistic heterogeneous experiments.

    "nominal"  Always use the midpoint value.
               Use for controlled, deterministic baseline runs.

    "worst"    Always use the maximum value (stress test).
               Use for capacity planning experiments.


===========================================================================
APPLICATION DOMAINS AND SUBTYPES — FULL DETAILS
===========================================================================

──────────────────────────────────────────────────────────────────────────
DOMAIN 1: HEALTHCARE
──────────────────────────────────────────────────────────────────────────

SUBTYPE: emergency_care
    Description  : Life-critical triage requiring sub-10ms edge response
    QoS priority : ultra_low_latency
    Tier         : edge (offload NOT permitted)
    Criticality  : 5/5
    Real-time    : Yes
    CPU          : 2 – 8 cores  @ 2000 – 3800 MHz
    GPU VRAM     : 0 – 2048 MB
    RAM          : 512 MB – 4 GB
    Latency max  : 10 ms
    LLM model    : Llama 8B (llm_heavy)

    Example intent sentences:
        "Patient in cardiac arrest, automated defibrillator needs
         immediate rhythm analysis and shock advisory."
        "Stroke alert: patient showing left-side facial droop and
         slurred speech, CT scan priority needed."
        "ICU patient blood pressure 60/40 and falling, vasopressor
         protocol must activate immediately."

SUBTYPE: medical_imaging
    Description  : AI-assisted radiology requiring GPU acceleration
    QoS priority : high_throughput
    Tier         : cloud (offload permitted)
    Criticality  : 3/5
    Real-time    : No
    CPU          : 4 – 16 cores @ 2400 – 4500 MHz
    GPU VRAM     : 4 GB – 24 GB
    RAM          : 8 GB – 64 GB
    Latency max  : 5000 ms
    LLM model    : Llama 8B (gpu_heavy)

    Example intent sentences:
        "Chest CT scan uploaded, 512 DICOM slices, AI needs to flag
         pulmonary nodules above 6mm."
        "MRI brain scan shows possible glioblastoma, segmentation
         model should delineate tumour boundaries."

SUBTYPE: remote_monitoring
    Description  : Continuous vital sign streaming and anomaly detection
    QoS priority : balanced
    Tier         : fog (offload permitted)
    Criticality  : 3/5
    Real-time    : Yes
    CPU          : 1 – 4 cores @ 800 – 2000 MHz
    GPU VRAM     : 0 – 512 MB
    RAM          : 256 MB – 1 GB
    Latency max  : 200 ms
    LLM model    : Llama 1B (cpu_only)

    Example intent sentences:
        "My smartwatch is showing irregular heartbeat for the past
         20 minutes, heart rate jumping 110 to 160."
        "COPD patient peak flow meter reading 180 L/min, 40% of
         personal best, yellow zone protocol."

──────────────────────────────────────────────────────────────────────────
DOMAIN 2: INDUSTRIAL (IIoT)
──────────────────────────────────────────────────────────────────────────

SUBTYPE: predictive_maintenance
    Description  : Real-time vibration, thermal and acoustic analysis
    QoS priority : low_latency
    Tier         : edge (offload permitted)
    Criticality  : 4/5
    Real-time    : Yes
    CPU          : 2 – 8 cores @ 1600 – 3600 MHz
    GPU VRAM     : 0 – 4 GB
    RAM          : 512 MB – 8 GB
    Latency max  : 50 ms
    LLM model    : Llama 1B (cpu_only)

    Example intent sentences:
        "CNC spindle motor vibration at 12.4 mm/s RMS, baseline was
         3.2, bearing replacement window?"
        "Hydraulic pump pressure dropping 8 bar below set point
         intermittently, cavitation signature detected."

SUBTYPE: quality_control
    Description  : Real-time computer vision on production lines >60fps
    QoS priority : high_throughput
    Tier         : fog (offload NOT permitted)
    Criticality  : 3/5
    Real-time    : Yes
    CPU          : 4 – 16 cores @ 2400 – 4800 MHz
    GPU VRAM     : 2 GB – 16 GB
    RAM          : 4 GB – 32 GB
    Latency max  : 33 ms
    LLM model    : Llama 8B (gpu_heavy)

    Example intent sentences:
        "PCB surface mount inspection line running at 1200 boards/hour,
         solder joint defect rate above 0.3%."
        "Steel coil surface inspection for rolling marks, cracks and
         scale at 400 metres per minute."

SUBTYPE: process_control
    Description  : Closed-loop AI-augmented control for safety-critical processes
    QoS priority : ultra_low_latency
    Tier         : edge (offload NOT permitted)
    Criticality  : 5/5
    Real-time    : Yes
    CPU          : 4 – 16 cores @ 2000 – 4200 MHz
    GPU VRAM     : 0 – 8 GB
    RAM          : 2 GB – 32 GB
    Latency max  : 8 ms
    LLM model    : Llama 1B (cpu_only)

    Example intent sentences:
        "Reactor vessel exothermic runaway detected, coolant flow must
         increase 40% within 5 seconds."
        "Gas turbine blade temperature 1180°C and rising, fuel-air
         ratio adjustment required immediately."

──────────────────────────────────────────────────────────────────────────
DOMAIN 3: GENERAL (cross-domain)
──────────────────────────────────────────────────────────────────────────

SUBTYPE: smart_environment
    Description  : Multi-sensor ambient monitoring for smart buildings
    QoS priority : best_effort
    Tier         : fog (offload permitted)
    Criticality  : 1/5
    Real-time    : No
    CPU          : 1 – 2 cores @ 400 – 1600 MHz
    GPU VRAM     : 0 MB
    RAM          : 64 MB – 512 MB
    Latency max  : 5000 ms
    LLM model    : Llama 1B (cpu_only)

    Example intent sentences:
        "Meeting room CO2 concentration 1800ppm, occupancy sensors
         show 12 people, ventilation adjustment needed."
        "Campus smart grid solar panel output dropped 35% this
         afternoon despite clear skies, inverter check."

SUBTYPE: federated_learning
    Description  : Distributed model training with privacy-preserving aggregation
    QoS priority : high_throughput
    Tier         : adaptive
    Criticality  : 2/5
    Real-time    : No
    CPU          : 2 – 8 cores @ 1200 – 3200 MHz
    GPU VRAM     : 512 MB – 8 GB
    RAM          : 1 GB – 16 GB
    Latency max  : 30000 ms
    LLM model    : Llama 3B (gpu_light)

    Example intent sentences:
        "24 edge nodes ready, initiating federated round 47, global
         model accuracy stalled at 82.3%."
        "Non-IID data distribution across 16 manufacturing sites
         causing 6% accuracy drop versus centralised."

SUBTYPE: digital_twin
    Description  : LLM-powered digital twin synchronisation and what-if reasoning
    QoS priority : balanced
    Tier         : adaptive
    Criticality  : 2/5
    Real-time    : No
    CPU          : 2 – 12 cores @ 1600 – 4000 MHz
    GPU VRAM     : 1 GB – 16 GB
    RAM          : 2 GB – 32 GB
    Latency max  : 500 ms
    LLM model    : Llama 8B (llm_heavy)

    Example intent sentences:
        "Factory digital twin scenario: what happens to throughput
         if conveyor 3 goes down for 2 hours?"
        "Power grid twin simulating N-1 contingency: loss of
         substation 7 affects 14,000 customers, mitigation?"


===========================================================================
REQUIREMENT RANGES — HOW VALUES ARE SAMPLED
===========================================================================

Each application subtype defines a RequirementRange with three values
per resource dimension: minimum, nominal, and maximum.

Example for emergency_care CPU:
    cpu_cores_min = 2     (lightest possible)
    cpu_cores_nom = 4     (typical deployment)
    cpu_cores_max = 8     (peak demand)

At runtime, REQUIREMENT_MODE controls which value is used:
    "random"  → sample uniformly from [min, max]
    "nominal" → always use nom
    "worst"   → always use max

This means every generated instance has slightly different requirements,
simulating real heterogeneous IoT workload patterns.

Resources with explicit units (for IEEE paper clarity):
    cpu_cores         logical CPU cores
    cpu_freq_mhz      MHz clock frequency
    gpu_memory_mb     MB VRAM
    ram_mb            MB system RAM
    storage_mb        MB persistent storage
    bandwidth_kbps    kbps uplink to fog/cloud
    latency_ms_max    ms hard deadline
    latency_ms_target ms soft target
    packet_loss_max   % maximum tolerable packet loss


===========================================================================
INTENT SENTENCES — THE LLM INPUT
===========================================================================

Each subtype has 10 realistic natural-language intent sentences.
One sentence is selected at random for each generated instance.

These sentences simulate what a real human (clinician, operator,
or automated system) would send to the orchestration platform.

The Llama model in Agent 1 and Agent 3 reads these sentences and
must infer: domain, subtype, urgency, tier preference, and resource
needs — without being told explicitly. This is the "intent-aware"
part of the system title.

Sentence design principles:
    - Uses domain-specific vocabulary (medical, industrial, general)
    - Contains numeric values with units (87°C, 12.4 mm/s, 60/40 mmHg)
    - Uses urgency signals (immediately, now, required, alert)
    - Never explicitly states the domain or subtype label
    - Written as a real operator or patient would write it


===========================================================================
run() FUNCTION — ORCHESTRATOR INTEGRATION
===========================================================================

Called by OrchestratorAgent.py:

    result = applications.run(config, shared_state)

Config keys accepted:
    "num_apps"          int      — total instances to generate
    "domains"           list     — which domains to include
    "subtypes"          dict     — per-domain subtype filter
    "seed"              int|None — RNG seed
    "requirement_mode"  str      — "random" | "nominal" | "worst"
    "output_dir"        str      — where to save JSON output

Writes to shared_state:
    shared_state["applications"]  → list[ApplicationInstance]
    shared_state["app_summary"]   → summary dict for logging

Returns:
    {
      "instances":    list[ApplicationInstance],
      "count":        int,
      "output_path":  str,
      "status":       "ok" | "error",
      "error":        str   (only if status == "error")
    }


===========================================================================
OUTPUT FILE
===========================================================================

experiment_applications.json
    List of all generated ApplicationInstance objects serialised to JSON.
    Read by agents.py, baselines.py, and figures.py.
    Saved to output_dir (default: current folder).


===========================================================================
HOW TO RUN IN PYCHARM
===========================================================================

STANDALONE TEST:
    1. Open applications.py in PyCharm.
    2. Edit APP_DOMAINS, NUM_APPS, RANDOMISE_SEED at the top as needed.
    3. Run applications.py directly (right-click → Run).
    4. Terminal shows a summary table with domain/subtype distribution
       and a sample instance with its LLM prompt preview.
    5. experiment_applications.json saved to the project folder.

FULL SYSTEM RUN:
    Run OrchestratorAgent.py only. It calls applications.run()
    automatically as the first step.


===========================================================================
DEPENDENCIES
===========================================================================

applications.py uses ONLY Python standard library:
    random, time, uuid, json, dataclasses, enum, typing

No pip install required.


===========================================================================
END OF README
===========================================================================
