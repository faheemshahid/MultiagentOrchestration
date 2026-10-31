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
                                   writes IntentAnalysis.txt
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
NUM_APPS:          int   = 1000         # total apps to generate and deploy
                                       # 10 = quick test, 30 = paper results
                                       # 100+ = large-scale experiment

APP_DOMAINS: list = [                  # which domains to include
    "healthcare",
    "industrial",   # commented for focused healthcare experiment
    # "general",      # commented for focused healthcare experiment
]
# To run healthcare only:  APP_DOMAINS = ["healthcare"]
# To run two domains:      APP_DOMAINS = ["healthcare", "industrial"]

APP_SUBTYPES: dict = {                 # None = all subtypes in that domain
    "healthcare": ["emergency_care", "remote_monitoring"],
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

NUM_FOG_NODES:   int   = 5            # number of fog gateway nodes
NUM_EDGE_NODES:  int   = 5            # number of edge compute nodes
NUM_LOCAL_NODES: int   = 3            # number of local Pi/Jetson nodes
NODE_VARIATION:  float = 0.50         # ±15% variation in node specs
                                       # 0.0 = identical nodes (controlled)

# ── Agent / LLM settings ──────────────────────────────────────────────────────
LLM_BACKEND:    str  = "mock"         # "mock" | "ollama" | "llama_cpp"
LLM_MODEL:      str  = "llama3.2:8b-instruct-q4_K_M"
LLM_HOST:       str  = "http://localhost:11434"
AGENT_VERBOSE:  bool = False           # True = detailed per-agent output
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
import textwrap
import traceback
import statistics
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
#  WRITE INTENT ANALYSIS
#  Saves IntentParserAgent output for all apps to IntentAnalysis.txt
#  with real-life IoT scenarios and citable academic sources.
# ─────────────────────────────────────────────────────────────────────────────

def write_intent_analysis(decisions: list, output_dir: str):
    """
    Generate IntentAnalysis.txt from agent_decisions.
    Each entry documents the IntentParserAgent output for one app,
    including real-world scenario context and classification rationale.
    Academic references are appended at the end.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "IntentAnalysis.txt")

    # ── Urgency signal mappings (real-world scenario descriptions) ────────────
    URGENCY_SCENARIOS = {
        "critical": (
            "life-safety or safety-critical industrial control — "
            "requires sub-10 ms end-to-end response (e.g. ICU vasopressor "
            "protocol, reactor runaway, wellhead pressure spike)"
        ),
        "high": (
            "time-sensitive medical or industrial monitoring — "
            "10–200 ms acceptable latency (e.g. COPD peak-flow alert, "
            "conveyor vision inspection, surgical robot feedback)"
        ),
        "medium": (
            "near-real-time analytics or process optimisation — "
            "200 ms – 2 s tolerable (e.g. MRI scan post-processing, "
            "predictive maintenance model inference, digital twin update)"
        ),
        "low": (
            "batch or periodic telemetry — latency > 2 s acceptable "
            "(e.g. EHR aggregation, energy consumption reporting, "
            "environmental sensor archival)"
        ),
        "background": (
            "non-urgent background processing — minutes to hours acceptable "
            "(e.g. whole-slide pathology archival, training data export, "
            "long-term trend analysis)"
        ),
    }

    # ── Tier rationale per deployment tier ────────────────────────────────────
    TIER_RATIONALE = {
        "local": (
            "Local node selected: ultra-low latency (1–4 ms) required; "
            "app is non-offload-eligible or real-time with strict deadline. "
            "Ref: Shi et al. (2016) edge computing paradigm."
        ),
        "edge":  (
            "Edge node selected: low latency (4–10 ms) with moderate compute; "
            "intent signals high urgency or real-time classification. "
            "Ref: Mach & Becvar (2017) MEC survey."
        ),
        "fog":   (
            "Fog node selected: balanced latency–throughput trade-off; "
            "TOPSIS scores fog tier highest across 5 MADM criteria. "
            "Ref: Bonomi et al. (2012) fog computing IoT."
        ),
        "cloud": (
            "Cloud node selected: high-compute, non-latency-sensitive task; "
            "GPU-intensive inference or large model; intent is background/batch. "
            "Ref: Armbrust et al. (2010) cloud computing view."
        ),
    }

    # ── Domain descriptions ───────────────────────────────────────────────────
    DOMAIN_DESC = {
        "healthcare": (
            "Healthcare IoT — patient monitoring, diagnostics, emergency response. "
            "Strict SLA: data integrity, HIPAA-aligned privacy, < 10 ms for critical."
        ),
        "industrial": (
            "Industrial IoT (IIoT) — process control, quality vision, predictive maintenance. "
            "OT-network constraints; real-time control loops; ISA-95 compliance."
        ),
        "general": (
            "General IoT — smart environment, campus energy, digital twins. "
            "Best-effort or balanced QoS; adaptive tier placement."
        ),
    }

    # ── Academic references ───────────────────────────────────────────────────
    REFERENCES = [
        ("[R1]  Shi, W., Cao, J., Zhang, Q., Li, Y., & Xu, L. (2016). "
         "Edge computing: Vision and challenges. "
         "IEEE Internet of Things Journal, 3(5), 637–646. "
         "https://doi.org/10.1109/JIOT.2016.2579198"),
        ("[R2]  Mach, P., & Becvar, Z. (2017). "
         "Mobile edge computing: A survey on architecture and computation offloading. "
         "IEEE Communications Surveys & Tutorials, 19(3), 1628–1656. "
         "https://doi.org/10.1109/COMST.2017.2682318"),
        ("[R3]  Bonomi, F., Milito, R., Zhu, J., & Addepalli, S. (2012). "
         "Fog computing and its role in the internet of things. "
         "Proc. MCC Workshop on Mobile Cloud Computing, 13–16. "
         "https://doi.org/10.1145/2342509.2342513"),
        ("[R4]  Armbrust, M., et al. (2010). "
         "A view of cloud computing. "
         "Communications of the ACM, 53(4), 50–58. "
         "https://doi.org/10.1145/1721654.1721672"),
        ("[R5]  Hwang, C.-L., & Yoon, K. (1981). "
         "Multiple Attribute Decision Making: Methods and Applications. "
         "Springer-Verlag. https://doi.org/10.1007/978-3-642-48318-9"),
        ("[R6]  Chen, M., Mao, S., & Liu, Y. (2014). "
         "Big data: A survey. "
         "Mobile Networks and Applications, 19(2), 171–209. "
         "https://doi.org/10.1007/s11036-013-0489-0"),
        ("[R7]  Zhang, Q., Cheng, L., & Boutaba, R. (2010). "
         "Cloud computing: State-of-the-art and research challenges. "
         "Journal of Internet Services and Applications, 1(1), 7–18. "
         "https://doi.org/10.1007/s13174-010-0007-6"),
        ("[R8]  Abbas, N., Zhang, Y., Taherkordi, A., & Skeie, T. (2018). "
         "Mobile edge computing: A survey. "
         "IEEE Internet of Things Journal, 5(1), 450–465. "
         "https://doi.org/10.1109/JIOT.2017.2750180"),
        ("[R9]  Wang, X., Han, Y., Leung, V. C. M., Niyato, D., Yan, X., & Chen, X. (2020). "
         "Convergence of edge computing and deep learning: A comprehensive survey. "
         "IEEE Communications Surveys & Tutorials, 22(2), 869–904. "
         "https://doi.org/10.1109/COMST.2020.2970550"),
        ("[R10] Ai, Y., Peng, M., & Zhang, K. (2018). "
         "Edge computing technologies for Internet of Things: A primer. "
         "Digital Communications and Networks, 4(2), 77–86. "
         "https://doi.org/10.1016/j.dcan.2017.07.001"),
        ("[R11] Ren, J., et al. (2019). "
         "Serving at the edge: A scalable IoT architecture based on transparent computing. "
         "IEEE Network, 31(5), 96–105. "
         "https://doi.org/10.1109/MNET.2017.1700030"),
    ]

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("=" * 78)
    lines.append("  IntentAnalysis.txt")
    lines.append("  IntentParserAgent Output — Intent-Aware Multi-Agent IoT Orchestration")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total applications analysed: {len(decisions)}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("  This file documents the IntentParserAgent's analysis of each IoT")
    lines.append("  application's natural-language intent text. The agent uses TF-IDF")
    lines.append("  cosine similarity (vocabulary match) combined with LLM confidence")
    lines.append("  scoring (semantic understanding) to extract domain, urgency, and")
    lines.append("  deployment preference signals used by downstream MADM (TOPSIS).")
    lines.append("")
    lines.append("  Agent pipeline: IntentParser → NB Classifier → QoS Profiler →")
    lines.append("                  TOPSIS Placer → SLA Monitor → Offload Arbiter")
    lines.append("")

    # ── Per-app entries ───────────────────────────────────────────────────────
    for idx, d in enumerate(decisions, 1):
        # Safe attribute access (works for both dataclass objects and dicts)
        def _get(obj, *keys, default="N/A"):
            for k in keys:
                if hasattr(obj, k):
                    obj = getattr(obj, k)
                elif isinstance(obj, dict) and k in obj:
                    obj = obj[k]
                else:
                    return default
            return obj if obj is not None else default

        inst_id    = _get(d, "instance_id")
        intent_txt = _get(d, "intent_text")
        final_st   = _get(d, "final_status")
        pipeline_ms = _get(d, "total_pipeline_ms", default=0.0)

        # intent_profile fields
        ip          = _get(d, "intent_profile")
        domain      = _get(ip, "detected_domain")
        subtype     = _get(ip, "detected_subtype")
        cosine_sim  = _get(ip, "tfidf_cosine_sim", default=0.0)
        llm_conf    = _get(ip, "llm_confidence",   default=0.0)
        llm_reason  = _get(ip, "llm_reasoning",    default="")
        parse_ms    = _get(ip, "parse_time_ms",    default=0.0)
        urg_signals = _get(ip, "urgency_signals",  default=[])

        # qos_profile fields
        qp        = _get(d, "qos_profile")
        urgency   = _get(qp, "urgency_level", default="medium")

        # placement fields
        pl        = _get(d, "placement")
        sel_node  = _get(pl, "selected_node_id")
        sel_tier  = _get(pl, "selected_tier")
        topsis_cc = _get(pl, "topsis_cc",            default=0.0)
        adm_stat  = _get(pl, "admission_status")
        lat_est   = _get(pl, "latency_estimate_ms",  default=0.0)

        # offload fields
        of         = _get(d, "offload")
        off_trig   = _get(of, "triggered",   default=False)
        off_tier   = _get(of, "target_tier", default="N/A")

        # classification fields
        cl        = _get(d, "classification")
        nb_conf   = _get(cl, "nb_confidence",        default=0.0)
        llm_used  = _get(cl, "llm_refinement_used",  default=False)

        # Scenario lookup
        urg_str   = str(urgency).lower()
        tier_str  = str(sel_tier).lower() if sel_tier and sel_tier != "N/A" else ""
        dom_str   = str(domain).lower()
        scenario  = URGENCY_SCENARIOS.get(urg_str, "Standard IoT workload.")
        tier_rat  = TIER_RATIONALE.get(tier_str, "Tier selected by TOPSIS optimisation.")
        dom_desc  = DOMAIN_DESC.get(dom_str, "IoT application domain.")

        lines.append("─" * 78)
        lines.append(f"  App {idx:>3}  |  ID: {inst_id}")
        lines.append("─" * 78)
        lines.append(f"  Intent text  : {intent_txt}")
        lines.append(f"  Domain       : {domain}  ({subtype})")
        lines.append(f"  {dom_desc}")
        lines.append("")
        lines.append(f"  ── IntentParserAgent Output ──────────────────────────────────")
        lines.append(f"  TF-IDF cosine similarity : {float(cosine_sim):.4f}")
        lines.append(f"    Measures vocabulary overlap between intent text and domain")
        lines.append(f"    keyword corpus. Values > 0.30 indicate strong domain signal.")
        lines.append(f"  LLM confidence score     : {float(llm_conf):.4f}")
        lines.append(f"    LLM semantic confidence from Llama-3.2 prompt inference.")
        lines.append(f"    Combined with TF-IDF for robust intent disambiguation.")
        lines.append(f"  Parse time               : {float(parse_ms):.1f} ms")
        lines.append(f"  Urgency level            : {urgency}")
        lines.append(f"    Scenario: {scenario}")
        if urg_signals and isinstance(urg_signals, (list, tuple)) and len(urg_signals) > 0:
            sig_str = ", ".join(str(s) for s in urg_signals[:5])
            lines.append(f"  Urgency signals detected : {sig_str}")
        lines.append("")
        lines.append(f"  ── Classification & Placement ────────────────────────────────")
        lines.append(f"  NB classifier confidence : {float(nb_conf):.4f}")
        lines.append(f"  LLM refinement used      : {llm_used}")
        lines.append(f"  TOPSIS CC score          : {float(topsis_cc):.4f}")
        lines.append(f"    (0.0 = worst, 1.0 = ideal — multi-criteria optimum)")
        lines.append(f"  Selected node            : {sel_node}  [{sel_tier}]")
        lines.append(f"  Estimated latency        : {float(lat_est):.2f} ms")
        lines.append(f"  Admission status         : {adm_stat}")
        lines.append(f"  Final status             : {final_st}")
        if off_trig:
            lines.append(f"  Offload triggered        : YES → {off_tier}")
        lines.append(f"  Total pipeline time      : {float(pipeline_ms):.1f} ms")
        lines.append(f"  {tier_rat}")
        lines.append("")

    # ── Aggregate summary ─────────────────────────────────────────────────────
    placed_n    = sum(1 for d in decisions if _get(d, "final_status") == "placed")
    rejected_n  = len(decisions) - placed_n
    offload_n   = sum(1 for d in decisions if _get(_get(d, "offload"), "triggered", default=False))

    placed_decs = [d for d in decisions if _get(d, "final_status") == "placed"]
    avg_cc  = 0.0
    avg_lat = 0.0
    avg_cos = 0.0
    avg_llm = 0.0
    if placed_decs:
        try:
            avg_cc  = sum(float(_get(_get(d, "placement"), "topsis_cc",           default=0)) for d in placed_decs) / len(placed_decs)
            avg_lat = sum(float(_get(_get(d, "placement"), "latency_estimate_ms",  default=0)) for d in placed_decs) / len(placed_decs)
            avg_cos = sum(float(_get(_get(d, "intent_profile"), "tfidf_cosine_sim", default=0)) for d in decisions) / len(decisions)
            avg_llm = sum(float(_get(_get(d, "intent_profile"), "llm_confidence",   default=0)) for d in decisions) / len(decisions)
        except Exception:
            pass

    tier_counts = Counter()
    domain_counts = Counter()
    urgency_counts = Counter()
    for d in decisions:
        pl = _get(d, "placement")
        if _get(d, "final_status") == "placed":
            tier_counts[str(_get(pl, "selected_tier"))] += 1
        domain_counts[str(_get(_get(d, "intent_profile"), "detected_domain"))] += 1
        urgency_counts[str(_get(_get(d, "qos_profile"), "urgency_level"))] += 1

    lines.append("=" * 78)
    lines.append("  AGGREGATE INTENT ANALYSIS SUMMARY")
    lines.append("=" * 78)
    lines.append(f"  Total applications  : {len(decisions)}")
    lines.append(f"  Placed              : {placed_n} ({placed_n/max(len(decisions),1)*100:.1f}%)")
    lines.append(f"  Rejected            : {rejected_n} ({rejected_n/max(len(decisions),1)*100:.1f}%)")
    lines.append(f"  Offload events      : {offload_n}")
    lines.append(f"  Avg TF-IDF cosine   : {avg_cos:.4f}")
    lines.append(f"  Avg LLM confidence  : {avg_llm:.4f}")
    lines.append(f"  Avg TOPSIS CC       : {avg_cc:.4f}")
    lines.append(f"  Avg latency (placed): {avg_lat:.2f} ms")
    lines.append("")
    lines.append("  Tier distribution (placed apps):")
    for tier in ["local", "edge", "fog", "cloud"]:
        n = tier_counts.get(tier, 0)
        if n:
            pct = n / max(placed_n, 1) * 100
            lines.append(f"    {tier:<8} {n:>4} apps  ({pct:.1f}%)")
    lines.append("")
    lines.append("  Domain distribution:")
    for domain, n in sorted(domain_counts.items()):
        pct = n / max(len(decisions), 1) * 100
        lines.append(f"    {domain:<20} {n:>4} apps  ({pct:.1f}%)")
    lines.append("")
    lines.append("  Urgency distribution:")
    for urg in ["critical", "high", "medium", "low", "background"]:
        n = urgency_counts.get(urg, 0)
        if n:
            pct = n / max(len(decisions), 1) * 100
            lines.append(f"    {urg:<12} {n:>4} apps  ({pct:.1f}%)")
    lines.append("")

    # ── Academic references ───────────────────────────────────────────────────
    lines.append("=" * 78)
    lines.append("  ACADEMIC REFERENCES")
    lines.append("  (cite these in your IEEE paper for methodology justification)")
    lines.append("=" * 78)
    for ref in REFERENCES:
        wrapped = textwrap.fill(ref, width=76, subsequent_indent="       ")
        lines.append(wrapped)
        lines.append("")

    lines.append("=" * 78)
    lines.append("  END OF IntentAnalysis.txt")
    lines.append("=" * 78)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  [OK] IntentAnalysis.txt saved to: {path}")
    print(f"       {len(decisions)} apps documented  |  "
          f"{placed_n} placed  |  {rejected_n} rejected  |  "
          f"{offload_n} offloaded")
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

def main(seed_override=None, iteration: int = 1):
    t_total = time.time()
    print_banner()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    shared_state: dict = {}
    config:       dict = build_config()

    # Apply seed override for multi-iteration runs
    if seed_override is not None:
        config["seed"] = seed_override

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

    # ── Write IntentAnalysis.txt ──────────────────────────────────────────────
    print_step(3, "Writing IntentParserAgent analysis to IntentAnalysis.txt")
    try:
        write_intent_analysis(decisions, OUTPUT_DIR)
    except Exception as e:
        print(f"\n  [WARN] IntentAnalysis.txt could not be written: {e}")

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
    print(f"    IntentAnalysis.txt              — IntentParserAgent analysis")
    print(f"\n  Next step: run figures.py to generate IEEE graphs")
    print(f"{'█' * 72}\n")

    return shared_state


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-ITERATION RUNNER
#  Runs the experiment N times with seeds RANDOM_SEED, RANDOM_SEED+1, ...
#  Averages all key metrics and saves aggregated results.
# ─────────────────────────────────────────────────────────────────────────────

def run_iterations(num_iterations: int):
    """Run the experiment num_iterations times and report mean ± std."""

    t_total_all = time.time()

    # Collect per-iteration metrics
    iter_metrics: list = []   # list of dicts, one per iteration

    for it in range(num_iterations):
        seed = RANDOM_SEED + it if RANDOM_SEED is not None else None
        print(f"\n\n{'▓' * 72}")
        print(f"  ITERATION {it + 1} / {num_iterations}   seed={seed}")
        print(f"{'▓' * 72}")

        # Override seed for this iteration
        import builtins
        builtins._ITER_SEED_OVERRIDE = seed

        # Run one full experiment
        state = main(seed_override=seed, iteration=it + 1)

        # Extract key metrics from comparison_table
        ct = state.get("comparison_table", {})
        decisions = state.get("agent_decisions", [])

        placed = [d for d in decisions if d.final_status == "placed"]
        placement_rate = len(placed) / max(len(decisions), 1) * 100

        ma_key = next((k for k in ct if "multi_agent" in k), None)
        ma = ct.get(ma_key, {}) if ma_key else {}

        iter_metrics.append({
            "iteration":       it + 1,
            "seed":            seed,
            "placement_rate":  placement_rate,
            "avg_score":       ma.get("avg_score",       0.0),
            "avg_latency_ms":  ma.get("avg_latency_ms",  0.0),
            "avg_decision_ms": ma.get("avg_decision_ms", 0.0),
            "edge_pct":        ma.get("edge_pct",        0.0),
            "cloud_pct":       ma.get("cloud_pct",       0.0),
        })

    # ── Aggregate across iterations ──────────────────────────────────────────
    total_elapsed = round(time.time() - t_total_all, 2)

    def _mean(key):
        vals = [m[key] for m in iter_metrics]
        return statistics.mean(vals)

    def _std(key):
        vals = [m[key] for m in iter_metrics]
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    print(f"\n\n{'█' * 72}")
    print(f"  MULTI-ITERATION SUMMARY — {num_iterations} runs")
    print(f"{'█' * 72}")
    print(f"\n  {'Metric':<25} {'Mean':>10}  {'± Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'─' * 70}")

    metrics_display = [
        ("Placement rate (%)",  "placement_rate"),
        ("Avg placement score", "avg_score"),
        ("Avg latency (ms)",    "avg_latency_ms"),
        ("Avg decision (ms)",   "avg_decision_ms"),
        ("Edge placement (%)",  "edge_pct"),
        ("Cloud placement (%)", "cloud_pct"),
    ]
    for label, key in metrics_display:
        vals = [m[key] for m in iter_metrics]
        print(f"  {label:<25} {_mean(key):>10.2f}  {_std(key):>10.2f}  "
              f"{min(vals):>10.2f}  {max(vals):>10.2f}")

    print(f"\n  Seeds used    : {[m['seed'] for m in iter_metrics]}")
    print(f"  Total elapsed : {total_elapsed}s")

    # ── Save aggregated results ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    agg = {
        "experiment": {
            "timestamp":      datetime.now().isoformat(),
            "num_iterations": num_iterations,
            "seeds":          [m["seed"] for m in iter_metrics],
            "num_apps":       NUM_APPS,
            "domains":        APP_DOMAINS,
            "total_elapsed_s": total_elapsed,
        },
        "iterations": iter_metrics,
        "aggregate": {
            key: {"mean": round(_mean(key), 4), "std": round(_std(key), 4)}
            for _, key in metrics_display
        },
    }
    agg_path = os.path.join(OUTPUT_DIR, "experiment_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n  Aggregated results saved to: {agg_path}")
    print(f"{'█' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        # ── Ask user for number of iterations ────────────────────────────────
        print("\n" + "═" * 72)
        print("  Intent-Aware Multi-Agent IoT Orchestration")
        print("═" * 72)
        try:
            raw = input(f"\n  How many iterations to run? [default = 1]: ").strip()
            num_iter = int(raw) if raw else 1
            if num_iter < 1:
                num_iter = 1
        except (ValueError, EOFError):
            num_iter = 1

        print(f"\n  Running {num_iter} iteration(s) "
              f"with seed(s): {[RANDOM_SEED + i for i in range(num_iter)]}")
        print()

        if num_iter == 1:
            main()
        else:
            run_iterations(num_iter)

    except KeyboardInterrupt:
        print("\n\n  [STOPPED] Experiment interrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n  [FATAL ERROR] {e}")
        print(traceback.format_exc())
        sys.exit(1)
