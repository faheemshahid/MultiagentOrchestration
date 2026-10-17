===========================================================================
FILE: computing_resources.py
PROJECT: Intent-Aware Multi-Agent AI Orchestration — Edge-Cloud Continuum
===========================================================================

WHAT THIS FILE IS:
------------------
This file defines the entire computing infrastructure of the system.
It describes every physical node (server, mini-PC, Raspberry Pi) across
all four tiers and tracks how resources are used during an experiment.

WHAT IT DOES:
-------------
1. DEFINES FOUR COMPUTING TIERS:

   CLOUD  — One large datacenter node (AMD EPYC + 8× A100 GPUs).
             Huge capacity, but 35ms+ latency to reach it.
             Runs the large Llama 70B model via Ollama.

   FOG    — N mid-tier gateway servers (Intel Xeon + RTX 4090).
             You set N with NUM_FOG_NODES (e.g. 2, 4, 8).
             Each has 64GB RAM, 24GB GPU, 1Gbps uplink to cloud.
             Runs Llama 8B quantised (4-bit) via Ollama.
             Realistic latency to cloud: ~35ms.

   EDGE   — N lightweight compute boxes (Intel NUC class, no GPU).
             You set N with NUM_EDGE_NODES (e.g. 4, 8, 16).
             Each has 32GB RAM, 100Mbps LAN uplink to fog.
             Runs Llama 3B (8-bit) via llama-cpp-python on CPU.
             Realistic latency to fog: ~8ms.

   LOCAL  — N minimal devices (Raspberry Pi 5 / Jetson Nano).
             You set N with NUM_LOCAL_NODES (e.g. 2, 4).
             Pi 5: 8GB RAM, no GPU, 10Mbps uplink to edge.
             Runs Llama 1B (smallest model) via llama-cpp-python.
             Realistic latency to edge: ~3ms.

2. MAKES EVERY NODE SLIGHTLY DIFFERENT:
   NODE_VARIATION_PCT = 0.15 means each node's CPU, RAM, GPU,
   bandwidth etc. is sampled randomly within ±15% of the nominal spec.
   This simulates real heterogeneous deployments.
   Set to 0.0 for identical nodes (controlled experiments).

3. TRACKS RESOURCE USAGE LIVE:
   Every node tracks used/available CPU cores, RAM, GPU VRAM,
   and bandwidth in real time. When the orchestrator places an
   application on a node, it calls node.allocate(app).
   When done, it calls node.release(app).
   After each change a snapshot is saved for the IEEE figures.

4. HELPS THE ORCHESTRATOR MAKE DECISIONS:
   node.can_accommodate(app)  — checks if the node has enough free
                                 resources for a given application.
   infra.best_node_for(app, tier) — finds the least-loaded node
                                     in a tier that fits the app.

HOW TO CONFIGURE (top of file):
--------------------------------
   ACTIVE_TIERS      = ["cloud","fog","edge","local"]  ← include/exclude tiers
   NUM_FOG_NODES     = 2    ← change to 1, 4, 8 etc.
   NUM_EDGE_NODES    = 4    ← change to 1, 8, 16 etc.
   NUM_LOCAL_NODES   = 2    ← change to 0 if no Pi devices
   NODE_VARIATION_PCT = 0.15  ← 0.0 = identical nodes
   RANDOMISE_SEED    = 42   ← change or set None for random

HOW IT INTEGRATES:
------------------
   OrchestratorAgent.py calls:  run(config, shared_state)
   This builds all nodes and puts the Infrastructure object into:
       shared_state["infrastructure"]
   All downstream agents (routing, LLM, figures) read from this.

   You can also run it alone in PyCharm to inspect the node specs:
       python computing_resources.py

OUTPUT FILE:
------------
   experiment_infrastructure.json  — snapshot of all nodes at start
   (used by figures.py to plot resource utilisation graphs)

FUTURE HARDWARE:
----------------
   When you add a Raspberry Pi or Nvidia Jetson physically,
   change "local_pi5" to "local_jetson_nano" in build_infrastructure().
   No other file needs to change.

===========================================================================
