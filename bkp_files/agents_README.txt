===========================================================================
FILE: agents.py
PROJECT: Intent-Aware Multi-Agent AI Orchestration — Edge-Cloud Continuum
===========================================================================

WHAT THIS FILE IS:
------------------
This is the intellectual core of the entire system. It defines six
independent AI agents that work together as a pipeline to analyse,
classify, and deploy IoT applications across the edge-cloud continuum.
This is what makes the system "multi-agent" and "intent-aware".

WHAT ARE THE SIX AGENTS:
-------------------------

AGENT 1 — IntentParserAgent
  What it does: Reads the natural-language sentence (intent) attached to
  each application and extracts its meaning using an LLM (Llama model).
  It finds numeric values (e.g. "87°C", "10ms"), action verbs (detect,
  control, monitor), urgency signals (immediately, critical, now), and
  named entities (reactor, patient, spindle motor).
  Why it matters: This is the "intelligence" layer — no hand-crafted
  rules. The LLM understands any sentence from any domain.

AGENT 2 — QoSProfilerAgent
  What it does: Converts the parsed intent into a formal Quality-of-Service
  contract: hard latency deadline (ms), criticality score (0–1), preferred
  computing tier, and SLA violation cost.
  Why it matters: Gives the placement agent precise numbers to optimise
  against, not vague labels.

AGENT 3 — ClassificationAgent
  What it does: Places each application in a precise taxonomy
  (e.g. healthcare/time_critical/emergency_care) and fingerprints its
  compute needs: CPU-only, GPU-light, GPU-heavy, or LLM-heavy. Also
  selects the right Llama model size (1B / 3B / 8B / 70B) for inference.
  Why it matters: Removes application agnosticism — the system knows
  WHAT the application does and WHAT compute it truly needs.

AGENT 4 — ResourceMonitorAgent
  What it does: Scans every compute node in every tier and takes a
  live snapshot of CPU%, RAM%, GPU free memory, available bandwidth,
  and latency. Detects overloaded and offline nodes. Raises SLA alerts.
  Why it matters: Placement decisions are based on the ACTUAL state of
  the infrastructure, not assumed values.

AGENT 5 — PlacementAgent
  What it does: This is the deployment decision-maker. It scores every
  candidate node using a 5-factor weighted function:
    - Latency score    (30%) — is the node close enough?
    - Resource fit     (25%) — does the node have enough free resources?
    - Criticality match(20%) — does the tier match the urgency level?
    - GPU match        (15%) — is GPU available if needed?
    - Load balance     (10%) — prefer less-loaded nodes
  The node with the highest score wins and the application is placed.
  Why it matters: This is the core algorithmic contribution of the paper.

AGENT 6 — OffloadAgent
  What it does: After every placement, checks if any node has become
  overloaded (CPU > 80% or RAM > 85%). If so, migrates the lowest-
  priority application from the overloaded node to the next tier up
  (local→edge→fog→cloud) or to a peer node in the same tier.
  Why it matters: Demonstrates adaptive, SLA-preserving behaviour —
  the system reacts to overload in real time, not just at placement time.

HOW THE AGENTS WORK TOGETHER:
------------------------------
For each application, the agents run in sequence:
  App instance → [1]Parse → [2]QoS → [3]Classify → [4]Monitor
              → [5]Place → [6]Offload check → AgentDecision

All six outputs are bundled into one AgentDecision object which
OrchestratorAgent.py reads to do the final deployment.

LLM BACKEND CONFIGURATION:
---------------------------
  LLM_BACKEND = "mock"      → No model needed. Runs fully offline.
                               Use this first to test the pipeline.
  LLM_BACKEND = "ollama"    → Uses local Ollama server (free, easy).
                               Run: ollama pull llama3.2:8b-instruct-q4_K_M
  LLM_BACKEND = "llama_cpp" → Uses llama-cpp-python (added later).

  Change LLM_BACKEND at the top of the file before running.

HOW TO TEST STANDALONE (PyCharm):
-----------------------------------
  Run agents.py directly. It will:
  1. Call applications.py to generate 10 test apps
  2. Call computing_resources.py to build the infrastructure
  3. Run all 6 agents for each app
  4. Print a full results summary table
  5. Save experiment_decisions.json

FILES CHANGED BY ADDING agents.py:
------------------------------------
  NO CHANGES needed to applications.py or computing_resources.py.
  agents.py imports them directly when run standalone.
  OrchestratorAgent.py will call all three via run(config, shared_state).

OUTPUT FILE:
------------
  experiment_decisions.json — full record of every agent's decision
  for every application. Used by figures.py for IEEE graphs.

===========================================================================
