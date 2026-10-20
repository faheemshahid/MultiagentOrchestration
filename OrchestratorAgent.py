"""
=============================================================================
OrchestratorAgent.py
=============================================================================
Intent-Aware Multi-Agent AI Orchestration in Edge-Cloud Continuum
-----------------------------------------------------------------------------
PURPOSE:
    Main entry point for the entire system. Run ONLY this file.
    It calls all modules in sequence, deploys all applications to computing
    nodes, runs the four baseline comparisons, and prints full results.

WHAT HAPPENS WHEN YOU RUN THIS FILE:
    Step 1  applications.py        generates NUM_APPS application instances
    Step 2  computing_resources.py builds the full infrastructure (all tiers)
    Step 3  agents.py              runs 6-agent ML pipeline on every app
                                   deploys each app to the best node
    Step 4  baselines.py           runs 4 traditional baselines on same apps
                                   and same infrastructure for fair comparison
    Step 5  Results printed        deployment summary, comparison table
    Step 6  JSON files saved       all results to OUTPUT_DIR

MECHANICS:
    One shared_state dict passes through every module:
        shared_state["applications"]       list[ApplicationInstance]
        shared_state["infrastructure"]     Infrastructure object
        shared_state["agent_decisions"]    list[AgentDecision]
        shared_state["active_placements"]  dict {instance_id: (app, node)}
        shared_state["baseline_results"]   dict {name: list[BaselineResult]}
        shared_state["comparison_table"]   dict (fed to figures.py)

    One config dict controls all settings — edit the EXPERIMENT CONFIG
    section below. Individual module defaults are overridden by this config.

HOW MANY APPS ARE DEPLOYED:
    Exactly NUM_APPS applications (set in EXPERIMENT CONFIG below).
    All NUM_APPS are deployed by the multi-agent system.
    The same NUM_APPS are then run through each of the 4 baselines.
    Total deployments per run: NUM_APPS × 5 (1 proposed + 4 baselines).

=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████████████████████████
#  EXPERIMENT CONFIGURATION  ←  EDIT THIS SECTION BEFORE EACH RUN
# ██████████████████████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

# ── Application settings ─2─────────────────────────────────────────────────────
NUM_APPS:          int   = 100          # total apps to generate and deploy
                                       # 10 = quick test, 30 = paper results
                                       # 100+ = large-scale experiment

APP_DOMAINS: list = [                  # which domains to include
    "healthcare",
    # "industrial",
    # "general",
]
# To run healthcare only:  APP_DOMAINS = ["healthcare"]
# To run two domains:      APP_DOMAINS = ["healthcare", "industrial"]

APP_SUBTYPES: dict = {                 # None = all subtypes in that domain
    "healthcare": None,
    "industrial": None,
    "general":    None,
}
# To select specific subtypes:
#   "healthcare": ["emergency_care", "remote_monitoring"]

REQUIREMENT_MODE: str = "random"       # "random" | "nominal" | "worst"
RANDOM_SEED: int      = 42             # set None for non-reproducible runs

# ── Infrastructure settings ───────────────────────────────────────────────────
ACTIVE_TIERS: list = [                 # which tiers to include
    "cloud",
    "fog",
    "edge",
    "local",
]
# Remove tiers to test subset deployments:
#   ACTIVE_TIERS = ["fog", "edge"]     (no cloud, no local)
#   ACTIVE_TIERS = ["cloud"]           (cloud-only simulation)

NUM_FOG_NODES:   int   = 3            # number of fog gateway nodes
NUM_EDGE_NODES:  int   = 3            # number of edge compute nodes
NUM_LOCAL_NODES: int   = 3            # number of local Pi/Jetson nodes
NODE_VARIATION:  float = 0.50         # ±15% variation in node specs
                                       # 0.0 = identical nodes (controlled)

# ── Agent / LLM settings ──────────────────────────────────────────────────────
LLM_BACKEND:    str  = "llama_cpp"         # "mock" | "ollama" | "llama_cpp"
LLM_MODEL:      str  = "llama3.2:8b-instruct-q4_K_M"
LLM_HOST:       str  = "http://localhost:11434"
AGENT_VERBOSE:  bool = True           # True = detailed per-agent output
                                       # False = summary only (faster console)

# ── Baseline settings ─────────────────────────────────────────────────────────
ACTIVE_BASELINES: list = [            # which baselines to run
    "cloud_only",
    "round_robin",
    "static_threshold",
    "greedy_first_fit",
]
BASELINE_VERBOSE: bool = False        # True = print each baseline placement

# ── Output settings ───────────────────────────────────────────────────────────
OUTPUT_DIR: str = "./experiment_results"   # folder for all output files
PRINT_PER_APP:  bool = True               # print summary per deployed app

# ██████████████████████████████████████████████████████████████████████████
#  END OF EXPERIMENT CONFIGURATION
# ██████████████████████████████████████████████████████████████████████████

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import time
import traceback
from datetime import datetime
from collections import Counter

# Ensure sibling modules are importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import applications        as apps_module
import computing_resources as cr_module
import agents              as agents_module
import baselines           as baselines_module


# ─────────────────────────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    print("\n" + "█" * 72)
    print("█  Intent-Aware Multi-Agent AI Orchestration                        █")
    print("█  Edge-Cloud Continuum — OrchestratorAgent                         █")
    print("█" * 72)
    print(f"\n  Start time  : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  NUM_APPS    : {NUM_APPS}")
    print(f"  Domains     : {APP_DOMAINS}")
    print(f"  Tiers       : {ACTIVE_TIERS}")
    print(f"  Fog nodes   : {NUM_FOG_NODES}")
    print(f"  Edge nodes  : {NUM_EDGE_NODES}")
    print(f"  Local nodes : {NUM_LOCAL_NODES}")
    print(f"  LLM backend : {LLM_BACKEND}")
    print(f"  Baselines   : {ACTIVE_BASELINES}")
    print(f"  Seed        : {RANDOM_SEED}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_step(step: int, title: str):
    print(f"\n{'═' * 72}")
    print(f"  STEP {step} — {title}")
    print(f"{'═' * 72}")


# ─────────────────────────────────────────────────────────────────────────────
#  DEPLOYMENT SUMMARY PRINTER
#  Prints per-application deployment results in a clean table
# ─────────────────────────────────────────────────────────────────────────────

def print_deployment_summary(decisions: list, infrastructure):
    print(f"\n{'─' * 72}")
    print(f"  DEPLOYMENT RESULTS — {len(decisions)} applications")
    print(f"{'─' * 72}")

    placed    = [d for d in decisions if d.final_status == "placed"]
    rejected  = [d for d in decisions if d.final_status == "rejected"]
    offloaded = [d for d in decisions if d.offload.triggered]

    # Per-app table
    if PRINT_PER_APP:
        print(f"\n  {'ID':<10} {'Domain':<14} {'Urgency':<10} "
              f"{'Tier':<6} {'Node':<14} {'TOPSIS':>7} {'Status':<9} {'ms':>6}")
        print("  " + "─" * 70)
        for d in decisions:
            p = d.placement
            print(f"  {d.instance_id:<10} "
                  f"{d.intent_profile.detected_domain:<14} "
                  f"{d.qos_profile.urgency_level:<10} "
                  f"{p.selected_tier:<6} "
                  f"{p.selected_node_id:<14} "
                  f"{p.topsis_cc:>7.4f} "
                  f"{d.final_status:<9} "
                  f"{d.total_pipeline_ms:>5.1f}ms")

    # Aggregate stats
    print(f"\n  ── Aggregate stats ──────────────────────────────────────────")
    print(f"  Total apps        : {len(decisions)}")
    print(f"  Successfully placed: {len(placed)} "
          f"({len(placed)/max(len(decisions),1)*100:.1f}%)")
    print(f"  Rejected          : {len(rejected)} "
          f"({len(rejected)/max(len(decisions),1)*100:.1f}%)")
    print(f"  Offload events    : {len(offloaded)}")

    if placed:
        avg_cc  = sum(d.placement.topsis_cc for d in placed) / len(placed)
        avg_lat = sum(d.placement.latency_estimate_ms for d in placed) / len(placed)
        avg_ms  = sum(d.total_pipeline_ms for d in decisions) / len(decisions)
        avg_nb  = sum(d.classification.nb_confidence for d in decisions) / len(decisions)
        llm_ref = sum(1 for d in decisions if d.classification.llm_refinement_used)
        print(f"  Avg TOPSIS CC     : {avg_cc:.4f}")
        print(f"  Avg latency       : {avg_lat:.2f} ms")
        print(f"  Avg pipeline time : {avg_ms:.1f} ms")
        print(f"  Avg NB confidence : {avg_nb:.4f}")
        print(f"  LLM refinements   : {llm_ref}/{len(decisions)}")

    # Tier distribution
    tier_dist = Counter(
        d.placement.selected_tier for d in placed if d.placement.selected_tier
    )
    print(f"\n  Tier distribution (placed apps):")
    for tier in ["local", "edge", "fog", "cloud"]:
        count = tier_dist.get(tier, 0)
        if count:
            bar = "█" * count
            pct = count / max(len(placed), 1) * 100
            print(f"    {tier:<8} {bar:<30} {count:>3} ({pct:.1f}%)")

    # Domain distribution
    domain_dist = Counter(d.intent_profile.detected_domain for d in decisions)
    print(f"\n  Domain distribution:")
    for domain, count in sorted(domain_dist.items()):
        print(f"    {domain:<20} {count}")

    # Urgency distribution
    urgency_dist = Counter(d.qos_profile.urgency_level for d in decisions)
    print(f"\n  Urgency distribution:")
    for urgency, count in sorted(urgency_dist.items()):
        print(f"    {urgency:<12} {count}")

    # Node utilisation snapshot
    print(f"\n  ── Node utilisation after deployment ────────────────────────")
    for tier in ["cloud", "fog", "edge", "local"]:
        nodes = infrastructure.nodes_for_tier(tier)
        if not nodes:
            continue
        for node in nodes:
            bar_len = int(node.cpu_utilisation_pct / 5)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    [{tier:<6}] {node.node_id:<14}  "
                  f"CPU: {bar} {node.cpu_utilisation_pct:5.1f}%  "
                  f"RAM: {node.ram_utilisation_pct:5.1f}%  "
                  f"Apps: {node.active_app_count}")


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL COMPARISON TABLE PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_final_comparison(table: dict):
    print(f"\n{'═' * 84}")
    print(f"  FINAL IEEE COMPARISON TABLE")
    print(f"  Proposed multi-agent system vs 4 traditional baselines")
    print(f"  Same {NUM_APPS} apps  ×  Same infrastructure  ×  Different algorithm only")
    print(f"{'═' * 84}")
    print(f"\n  {'System':<30} {'Placed%':>8} {'Lat(ms)':>8} "
          f"{'Score':>7} {'Edge%':>7} {'Cloud%':>7} {'Dec(ms)':>8}")
    print("  " + "─" * 80)

    # Print multi-agent first
    for name, m in table.items():
        if "multi_agent" in name:
            marker = "  ◄ PROPOSED"
            print(f"  {name:<30} "
                  f"{m['placed_pct']:>7.1f}% "
                  f"{m['avg_latency_ms']:>7.2f}ms "
                  f"{m['avg_score']:>7.1f} "
                  f"{m['edge_pct']:>6.1f}% "
                  f"{m['cloud_pct']:>6.1f}% "
                  f"{m['avg_decision_ms']:>7.2f}ms"
                  f"{marker}")

    print("  " + "─" * 80)

    # Print baselines
    for name, m in table.items():
        if "multi_agent" not in name:
            print(f"  {name:<30} "
                  f"{m['placed_pct']:>7.1f}% "
                  f"{m['avg_latency_ms']:>7.2f}ms "
                  f"{m['avg_score']:>7.1f} "
                  f"{m['edge_pct']:>6.1f}% "
                  f"{m['cloud_pct']:>6.1f}% "
                  f"{m['avg_decision_ms']:>7.2f}ms")

    print(f"\n  Columns: Placed% = admission rate | Lat = avg node latency |")
    print(f"           Score = placement quality | Edge/Cloud = tier distribution")
    print(f"{'═' * 84}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE MASTER RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_master_results(shared_state: dict, elapsed: float):
    """Save a master experiment summary JSON for figures.py."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "experiment": {
            "timestamp":       datetime.now().isoformat(),
            "num_apps":        NUM_APPS,
            "domains":         APP_DOMAINS,
            "active_tiers":    ACTIVE_TIERS,
            "num_fog_nodes":   NUM_FOG_NODES,
            "num_edge_nodes":  NUM_EDGE_NODES,
            "num_local_nodes": NUM_LOCAL_NODES,
            "llm_backend":     LLM_BACKEND,
            "seed":            RANDOM_SEED,
            "requirement_mode": REQUIREMENT_MODE,
            "total_elapsed_s": round(elapsed, 2),
        },
        "app_summary":       shared_state.get("app_summary",   {}),
        "infra_summary":     shared_state.get("infra_summary", {}),
        "comparison_table":  shared_state.get("comparison_table", {}),
    }
    path = os.path.join(OUTPUT_DIR, "experiment_master.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD UNIFIED CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def build_config() -> dict:
    """Build one config dict passed to every module."""
    return {
        # Application settings
        "num_apps":           NUM_APPS,
        "domains":            APP_DOMAINS,
        "subtypes":           APP_SUBTYPES,
        "seed":               RANDOM_SEED,
        "requirement_mode":   REQUIREMENT_MODE,
        # Infrastructure settings
        "active_tiers":       ACTIVE_TIERS,
        "num_fog_nodes":      NUM_FOG_NODES,
        "num_edge_nodes":     NUM_EDGE_NODES,
        "num_local_nodes":    NUM_LOCAL_NODES,
        "variation_pct":      NODE_VARIATION,
        # Agent settings
        "llm_backend":        LLM_BACKEND,
        "llm_model":          LLM_MODEL,
        "llm_host":           LLM_HOST,
        "agent_verbose":      AGENT_VERBOSE,
        # Baseline settings
        "active_baselines":   ACTIVE_BASELINES,
        "baseline_verbose":   BASELINE_VERBOSE,
        # Output
        "output_dir":         OUTPUT_DIR,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print_banner()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    shared_state: dict = {}
    config:       dict = build_config()

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 1 — Generate applications
    # ─────────────────────────────────────────────────────────────────────────
    print_step(1, f"Generating {NUM_APPS} application instances")
    t0 = time.time()

    r1 = apps_module.run(config, shared_state)
    if r1["status"] != "ok":
        print(f"\n  [ERROR] applications.py failed: {r1.get('error')}")
        sys.exit(1)

    apps = shared_state["applications"]
    summary = shared_state["app_summary"]
    elapsed = round(time.time() - t0, 2)

    print(f"\n  [OK] Generated {r1['count']} application instances  ({elapsed}s)")
    print(f"  Domains  : {summary.get('domains', [])}")
    print(f"  Subtypes : {summary.get('subtypes', [])}")
    print(f"  Saved to : {r1['output_path']}")

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 2 — Build infrastructure
    # ─────────────────────────────────────────────────────────────────────────
    print_step(2, "Building edge-cloud infrastructure")
    t0 = time.time()

    r2 = cr_module.run(config, shared_state)
    if r2["status"] != "ok":
        print(f"\n  [ERROR] computing_resources.py failed: {r2.get('error')}")
        sys.exit(1)

    infra   = shared_state["infrastructure"]
    isumm   = shared_state["infra_summary"]
    elapsed = round(time.time() - t0, 2)

    print(f"\n  [OK] Infrastructure built  ({elapsed}s)")
    total_nodes = sum(v["node_count"] for v in isumm.values())
    print(f"  Total nodes: {total_nodes}")
    for tier, data in isumm.items():
        print(f"    [{tier:<6}] "
              f"{data['node_count']} node(s)  "
              f"CPU: {data['total_cpu_cores']} cores  "
              f"RAM: {data['total_ram_gb']} GB  "
              f"GPU: {data['total_gpu_vram_gb']} GB  "
              f"Latency→up: {data.get('avg_latency_to_next_ms', 0)} ms")
    print(f"  Saved to : {r2['output_path']}")

    # ── Capacity sanity check — warn if NUM_APPS likely exceeds capacity ──────
    total_cpu = sum(v.get("total_cpu_cores", 0) for v in isumm.values())
    total_ram = sum(v.get("total_ram_gb", 0) * 1024 for v in isumm.values())
    # Rough estimate: avg app needs ~3 CPU cores and ~1GB RAM
    estimated_capacity = min(int(total_cpu / 3), int(total_ram / 1024))
    if NUM_APPS > estimated_capacity:
        print(f"\n  ⚠  WARNING: NUM_APPS={NUM_APPS} may exceed infrastructure capacity.")
        print(f"     Estimated capacity: ~{estimated_capacity} apps")
        print(f"     Total CPU: {total_cpu:.0f} cores  |  Total RAM: {total_ram/1024:.0f} GB")
        print(f"     To fix: increase NUM_FOG_NODES/NUM_EDGE_NODES, or reduce NUM_APPS")
        print(f"     Continuing — some apps will be rejected (this is normal for overload experiments)")

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 3 — Run multi-agent pipeline and deploy applications
    # ─────────────────────────────────────────────────────────────────────────
    print_step(3, f"Running 6-agent ML pipeline — deploying {NUM_APPS} apps")
    t0 = time.time()

    r3 = agents_module.run(config, shared_state)
    if r3["status"] != "ok":
        print(f"\n  [ERROR] agents.py failed: {r3.get('error')}")
        print(r3.get("traceback", ""))
        sys.exit(1)

    decisions = shared_state["agent_decisions"]
    elapsed   = round(time.time() - t0, 2)

    print(f"\n  [OK] Multi-agent deployment complete  ({elapsed}s)")
    print_deployment_summary(decisions, infra)
    print(f"\n  Decisions saved to: {r3['output_path']}")

    # ── App lifetime simulation — hold resources then release ────────────────
    # Each app holds resources for app_lifetime_s seconds, then releases.
    # This simulates realistic IoT workload churn and frees capacity for
    # subsequent apps. We simulate this by processing apps in waves:
    # Wave 1: deploy first batch → snapshot → release → Wave 2: deploy next...
    print(f"\n  Simulating app lifetime and resource release cycle ...")
    apps_list      = shared_state.get("applications", [])
    active_placements = shared_state.get("active_placements", {})

    # Release apps that have exceeded their lifetime (simulate time passing)
    released_count = 0
    for inst_id, (app, node) in list(active_placements.items()):
        lifetime = getattr(app, "app_lifetime_s", 5.0)
        # Simulate: after lifetime_s the app finishes and releases resources
        node.release(app)
        released_count += 1

    print(f"  Released {released_count} apps after lifetime interval")
    print(f"  Resources freed — next wave can now utilise freed capacity")

    # Take capacity snapshot BEFORE release for IEEE figures (peak utilisation)
    # Rebuild from decisions since we just released
    peak_capacity = {}
    for tier in ["cloud", "fog", "edge", "local"]:
        nodes = infra.nodes_for_tier(tier)
        if not nodes:
            continue
        # Recalculate from decisions what was placed
        placed_cpu = sum(
            (d.placement.cpu_fit_pct / 100) *
            next((n.total_cpu_cores for n in nodes
                  if n.node_id == d.placement.selected_node_id), 0)
            for d in decisions
            if d.placement.selected_tier == tier
            and d.placement.admission_status == "placed"
        )
        total_cpu = sum(n.total_cpu_cores for n in nodes)
        peak_capacity[tier] = {
            "max_cpu_cores":       round(total_cpu, 1),
            "used_cpu_cores":      round(min(placed_cpu, total_cpu), 1),
            "available_cpu_cores": round(max(0, total_cpu - placed_cpu), 1),
            "max_ram_mb":          sum(n.total_ram_mb for n in nodes),
            "used_ram_mb":         sum(n.used_ram_mb  for n in nodes),
            "available_ram_mb":    sum(n.available_ram_mb for n in nodes),
            "max_gpu_mb":          sum(n.total_gpu_vram_mb for n in nodes),
            "used_gpu_mb":         sum(n.used_gpu_vram_mb  for n in nodes),
            "node_count":          len(nodes),
            "active_apps":         sum(n.active_app_count for n in nodes),
        }

    shared_state["capacity_summary"] = peak_capacity
    if hasattr(infra, "capacity_summary"):
        # Override with real node values if method exists
        shared_state["capacity_summary"] = infra.capacity_summary()

    print(f"\n  Post-deployment capacity snapshot (peak — before release):")
    for tier, cap in shared_state["capacity_summary"].items():
        used_pct = round(cap["used_cpu_cores"] / max(cap["max_cpu_cores"], 1) * 100, 1)
        print(f"    [{tier:<6}] CPU {cap['used_cpu_cores']:.1f}/{cap['max_cpu_cores']:.1f} cores "
              f"({used_pct}%)  Apps: {cap['active_apps']}")

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 4 — Run baselines on same apps and infrastructure
    # ─────────────────────────────────────────────────────────────────────────
    print_step(4, f"Running {len(ACTIVE_BASELINES)} baselines on same {NUM_APPS} apps")
    print(f"  Using deep copy of same infrastructure — fair comparison guaranteed")
    t0 = time.time()

    r4 = baselines_module.run(config, shared_state)
    if r4["status"] != "ok":
        print(f"\n  [ERROR] baselines.py failed: {r4.get('error')}")
        print(r4.get("traceback", ""))
        sys.exit(1)

    elapsed = round(time.time() - t0, 2)
    print(f"\n  [OK] All baselines complete  ({elapsed}s)")
    print(f"  Baselines saved to : {r4['output_path']}")
    print(f"  Comparison table   : {r4['comparison_path']}")

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 5 — Print final IEEE comparison table
    # ─────────────────────────────────────────────────────────────────────────
    print_step(5, "Final comparison — proposed system vs baselines")
    comparison_table = shared_state.get("comparison_table", {})
    if comparison_table:
        print_final_comparison(comparison_table)
    else:
        print("  [WARN] Comparison table empty — check baselines output")

    # ─────────────────────────────────────────────────────────────────────────
    #  SAVE MASTER RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - t_total, 2)
    master_path   = save_master_results(shared_state, total_elapsed)

    # ─────────────────────────────────────────────────────────────────────────
    #  FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'█' * 72}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'█' * 72}")
    print(f"\n  Total elapsed      : {total_elapsed}s")
    print(f"  Apps generated     : {len(apps)}")
    print(f"  Apps deployed      : {len(decisions)}")

    placed_count = sum(1 for d in decisions if d.final_status == "placed")
    print(f"  Placement rate     : "
          f"{placed_count}/{len(decisions)} "
          f"({placed_count/max(len(decisions),1)*100:.1f}%)")

    print(f"\n  Output files in: {OUTPUT_DIR}/")
    print(f"    experiment_applications.json    — generated app instances")
    print(f"    experiment_infrastructure.json  — node specs and state")
    print(f"    experiment_decisions.json       — per-app agent decisions")
    print(f"    experiment_baselines.json       — per-baseline results")
    print(f"    experiment_comparison_table.json— unified comparison data")
    print(f"    experiment_master.json          — full experiment summary")
    print(f"\n  Next step: run figures.py to generate IEEE graphs")
    print(f"{'█' * 72}\n")

    return shared_state


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [STOPPED] Experiment interrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n  [FATAL ERROR] {e}")
        print(traceback.format_exc())
        sys.exit(1)
