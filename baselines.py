"""
=============================================================================
baselines.py
=============================================================================
Intent-Aware Multi-Agent AI Orchestration in Edge-Cloud Continuum
-----------------------------------------------------------------------------
PURPOSE:
    Implements four traditional IoT service orchestration baselines used
    for IEEE comparative evaluation against the multi-agent system in agents.py.

    All four baselines run on the IDENTICAL set of ApplicationInstance objects
    and the IDENTICAL Infrastructure produced by applications.py and
    computing_resources.py. This ensures a fair comparison — the only
    variable is the orchestration algorithm.

BASELINES:
    B1  Cloud-Only           All applications routed to cloud regardless of urgency.
                             Represents legacy centralised cloud orchestration.
                             Common in early IoT deployments before edge computing.

    B2  Round-Robin          Applications distributed cyclically across all nodes
                             in all active tiers. No resource awareness, no intent
                             parsing, no QoS consideration.

    B3  Static-Threshold     Applications placed on first node with CPU < 70%.
                             Fixed threshold, no ML, no latency awareness.
                             Represents simple rule-based IoT middleware.

    B4  Greedy First-Fit     Applications placed on the first node (in tier order)
                             that has sufficient free resources. No scoring, no
                             optimisation. Represents naive edge-cloud placement.

METRICS COLLECTED (identical schema to agents.py AgentDecision):
    - placement latency (ms)
    - placement_score (0–100, where applicable)
    - tier chosen
    - admission_status (placed / rejected)
    - pipeline_ms (total decision time)
    - resource utilisation snapshots

    Results are saved to experiment_baselines.json and read by figures.py
    to generate the IEEE comparison bar charts, CDF plots, and radar charts.

INTEGRATION:
    OrchestratorAgent calls: run(config, shared_state)
    Standalone test:         python baselines.py
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  BASELINE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_BASELINES: list[str] = [
    "cloud_only",
    "round_robin",
    "static_threshold",
    "greedy_first_fit",
]
# Comment out any baseline to exclude it from this experiment run.

STATIC_THRESHOLD_CPU: float = 70.0   # B3 threshold (%)
BASELINE_VERBOSE:     bool  = True

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import json, time, os, copy
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  BASELINE RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineResult:
    """
    One placement decision from a baseline algorithm.
    Schema mirrors PlacementDecision in agents.py for direct comparison.
    """
    baseline_name:        str
    instance_id:          str
    domain:               str
    subtype:              str
    urgency_declared:     str   # from app.qos_priority
    criticality_level:    int

    selected_node_id:     str   = ""
    selected_tier:        str   = ""
    selected_hw_class:    str   = ""
    placement_score:      float = 0.0    # algorithm-specific score
    latency_estimate_ms:  float = 0.0
    cpu_fit_pct:          float = 0.0
    ram_fit_pct:          float = 0.0
    admission_status:     str   = "pending"
    rejection_reason:     str   = ""
    pipeline_ms:          float = 0.0
    timestamp:            float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        return (f"[{self.instance_id}] "
                f"{self.domain:<12} "
                f"tier={self.selected_tier:<6} "
                f"node={self.selected_node_id:<14} "
                f"score={self.placement_score:5.1f} "
                f"status={self.admission_status:<8} "
                f"{self.pipeline_ms:.1f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE B1 — Cloud-Only
#
#  STRATEGY: Route every application to the cloud datacenter node,
#  regardless of urgency, latency requirements, or resource availability.
#
#  REPRESENTS: Legacy centralised cloud IoT orchestration (pre-edge era).
#  Examples: AWS IoT Core with Lambda, Azure IoT Hub with cloud processing.
#
#  EXPECTED WEAKNESSES (for IEEE discussion):
#  - High latency for time-critical apps (emergency_care, process_control)
#  - Bandwidth-intensive (all data sent to cloud)
#  - Single point of failure
#  - High cloud cost at scale
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class CloudOnlyBaseline:
    NAME = "cloud_only"

    def place(self, app, infrastructure) -> BaselineResult:
        t0 = time.time()
        r  = BaselineResult(
            baseline_name     = self.NAME,
            instance_id       = app.instance_id,
            domain            = app.domain,
            subtype           = app.subtype,
            urgency_declared  = app.qos_priority,
            criticality_level = app.criticality_level,
        )

        cloud_nodes = infrastructure.nodes_for_tier("cloud")
        if not cloud_nodes:
            r.admission_status = "rejected"
            r.rejection_reason = "no_cloud_nodes"
            r.pipeline_ms = round((time.time() - t0) * 1000, 2)
            return r

        # Always pick the first (only) cloud node
        node = cloud_nodes[0]
        ok, reason = node.can_accommodate(app)

        if ok and node.allocate(app):
            r.selected_node_id  = node.node_id
            r.selected_tier     = "cloud"
            r.selected_hw_class = node.hw_class
            r.placement_score   = 100.0  # cloud-only always "succeeds" by design
            r.latency_estimate_ms = node.latency_to_next_ms
            r.cpu_fit_pct = round((app.cpu_cores / node.total_cpu_cores) * 100, 2)
            r.ram_fit_pct = round((app.ram_mb    / node.total_ram_mb)    * 100, 2)
            r.admission_status = "placed"
        else:
            r.admission_status = "rejected"
            r.rejection_reason = reason if not ok else "allocation_failed"

        r.pipeline_ms = round((time.time() - t0) * 1000, 2)
        return r


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE B2 — Round-Robin
#
#  STRATEGY: Distribute applications cyclically across ALL nodes in ALL
#  active tiers, regardless of resource availability or QoS requirements.
#  The node pointer advances by 1 on each placement call.
#
#  REPRESENTS: Simple load distribution middleware used in early IoT gateways.
#  Examples: MQTT broker with round-robin subscriber routing, basic k8s
#  scheduling without resource requests.
#
#  EXPECTED WEAKNESSES:
#  - Ignores latency requirements (may place critical apps on cloud)
#  - Ignores resource availability (may overload nodes)
#  - No intent awareness (all apps treated identically)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class RoundRobinBaseline:
    NAME = "round_robin"

    def __init__(self):
        self._pointer: int = 0
        self._node_list: list = []

    def initialise(self, infrastructure):
        """Build flat ordered node list across all tiers."""
        self._node_list = infrastructure.all_nodes()
        self._pointer   = 0

    def place(self, app, infrastructure) -> BaselineResult:
        t0 = time.time()
        r  = BaselineResult(
            baseline_name     = self.NAME,
            instance_id       = app.instance_id,
            domain            = app.domain,
            subtype           = app.subtype,
            urgency_declared  = app.qos_priority,
            criticality_level = app.criticality_level,
        )

        if not self._node_list:
            self.initialise(infrastructure)

        # Try up to N nodes starting from pointer
        N = len(self._node_list)
        for attempt in range(N):
            idx  = (self._pointer + attempt) % N
            node = self._node_list[idx]
            if node.status in ["offline", "overloaded"]:
                continue
            if node.allocate(app):
                self._pointer = (idx + 1) % N
                r.selected_node_id  = node.node_id
                r.selected_tier     = node.tier
                r.selected_hw_class = node.hw_class
                r.placement_score   = 50.0  # round-robin has no optimisation
                r.latency_estimate_ms = node.latency_to_next_ms
                r.cpu_fit_pct = round(
                    (app.cpu_cores / max(node.total_cpu_cores, 1)) * 100, 2)
                r.ram_fit_pct = round(
                    (app.ram_mb / max(node.total_ram_mb, 1)) * 100, 2)
                r.admission_status = "placed"
                r.pipeline_ms = round((time.time() - t0) * 1000, 2)
                return r

        r.admission_status = "rejected"
        r.rejection_reason = "all_nodes_full_or_offline"
        r.pipeline_ms = round((time.time() - t0) * 1000, 2)
        return r


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE B3 — Static Threshold
#
#  STRATEGY: Scan nodes in tier order (edge → fog → cloud → local).
#  Place on first node whose CPU utilisation < STATIC_THRESHOLD_CPU (70%).
#  No intent parsing, no latency check, no resource fit scoring.
#
#  REPRESENTS: Rule-based IoT middleware with simple capacity management.
#  Examples: Eclipse Kura with CPU threshold policies, OpenFog reference
#  architecture with static admission control.
#
#  EXPECTED WEAKNESSES:
#  - Threshold-crossing causes thrashing (no EMA smoothing)
#  - First-available-tier ignores urgency (places background jobs on edge)
#  - No GPU awareness
#  - No latency-SLA enforcement
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class StaticThresholdBaseline:
    NAME = "static_threshold"

    SCAN_ORDER: list[str] = ["edge", "fog", "cloud", "local"]

    def place(self, app, infrastructure) -> BaselineResult:
        t0 = time.time()
        r  = BaselineResult(
            baseline_name     = self.NAME,
            instance_id       = app.instance_id,
            domain            = app.domain,
            subtype           = app.subtype,
            urgency_declared  = app.qos_priority,
            criticality_level = app.criticality_level,
        )

        for tier in self.SCAN_ORDER:
            nodes = infrastructure.nodes_for_tier(tier)
            for node in nodes:
                if node.status == "offline":
                    continue
                # Static threshold check — raw CPU, no EMA
                if node.cpu_utilisation_pct >= STATIC_THRESHOLD_CPU:
                    continue
                # Minimum resource check (no scoring)
                if node.available_ram_mb < app.ram_mb:
                    continue
                if node.available_cpu_cores < app.cpu_cores:
                    continue
                if node.allocate(app):
                    r.selected_node_id  = node.node_id
                    r.selected_tier     = tier
                    r.selected_hw_class = node.hw_class
                    # Score = remaining CPU headroom (simple heuristic)
                    r.placement_score = round(
                        100.0 - node.cpu_utilisation_pct, 2)
                    r.latency_estimate_ms = node.latency_to_next_ms
                    r.cpu_fit_pct = round(
                        (app.cpu_cores / max(node.total_cpu_cores, 1)) * 100, 2)
                    r.ram_fit_pct = round(
                        (app.ram_mb / max(node.total_ram_mb, 1)) * 100, 2)
                    r.admission_status = "placed"
                    r.pipeline_ms = round((time.time() - t0) * 1000, 2)
                    return r

        r.admission_status = "rejected"
        r.rejection_reason = "all_nodes_above_threshold_or_full"
        r.pipeline_ms = round((time.time() - t0) * 1000, 2)
        return r


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE B4 — Greedy First-Fit
#
#  STRATEGY: Scan ALL nodes across ALL tiers in fixed order.
#  Place on the first node that has sufficient free CPU, RAM, and GPU.
#  No scoring, no optimisation, no QoS awareness.
#
#  REPRESENTS: Basic bin-packing heuristic used in early edge computing papers.
#  First-Fit Decreasing (FFD) is the canonical baseline for resource allocation
#  research. This version uses First-Fit (FF) without pre-sorting.
#
#  EXPECTED WEAKNESSES:
#  - Fragmentation — fills first nodes completely, leaves last nodes empty
#  - No latency-SLA enforcement
#  - No intent awareness (ignores urgency, domain, criticality)
#  - Sensitive to node scan order (edge-biased here)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class GreedyFirstFitBaseline:
    NAME = "greedy_first_fit"

    SCAN_ORDER: list[str] = ["edge", "fog", "cloud", "local"]

    def place(self, app, infrastructure) -> BaselineResult:
        t0 = time.time()
        r  = BaselineResult(
            baseline_name     = self.NAME,
            instance_id       = app.instance_id,
            domain            = app.domain,
            subtype           = app.subtype,
            urgency_declared  = app.qos_priority,
            criticality_level = app.criticality_level,
        )

        for tier in self.SCAN_ORDER:
            for node in infrastructure.nodes_for_tier(tier):
                if node.status == "offline":
                    continue
                ok, reason = node.can_accommodate(app)
                if ok and node.allocate(app):
                    r.selected_node_id  = node.node_id
                    r.selected_tier     = tier
                    r.selected_hw_class = node.hw_class
                    # Score = CPU availability fraction (greedy measure)
                    r.placement_score = round(
                        (node.available_cpu_cores /
                         max(node.total_cpu_cores, 1)) * 100, 2)
                    r.latency_estimate_ms = node.latency_to_next_ms
                    r.cpu_fit_pct = round(
                        (app.cpu_cores / max(node.total_cpu_cores, 1)) * 100, 2)
                    r.ram_fit_pct = round(
                        (app.ram_mb / max(node.total_ram_mb, 1)) * 100, 2)
                    r.admission_status = "placed"
                    r.pipeline_ms = round((time.time() - t0) * 1000, 2)
                    return r

        r.admission_status = "rejected"
        r.rejection_reason = "no_node_with_sufficient_resources"
        r.pipeline_ms = round((time.time() - t0) * 1000, 2)
        return r


# ─────────────────────────────────────────────────────────────────────────────
#  BASELINE RUNNER — run all four baselines on the same applications
# ─────────────────────────────────────────────────────────────────────────────

def run_all_baselines(apps: list, infrastructure_snapshot,
                      active_baselines: list[str]) -> dict[str, list]:
    """
    Run each active baseline on a FRESH COPY of the infrastructure.
    Returns dict: {baseline_name: list[BaselineResult]}

    Each baseline gets an independent deep copy of the infrastructure
    so that one baseline's allocations do not affect another's decisions.
    This ensures fair comparison.
    """
    results: dict = {}

    baselines_map = {
        "cloud_only":        CloudOnlyBaseline(),
        "round_robin":       RoundRobinBaseline(),
        "static_threshold":  StaticThresholdBaseline(),
        "greedy_first_fit":  GreedyFirstFitBaseline(),
    }

    for b_name in active_baselines:
        baseline = baselines_map.get(b_name)
        if baseline is None:
            print(f"  [baselines] Unknown baseline '{b_name}' — skipped")
            continue

        print(f"\n[baselines.py] Running baseline: {b_name.upper()} "
              f"on {len(apps)} apps ...")

        # Deep copy infrastructure so each baseline starts fresh
        infra_copy = copy.deepcopy(infrastructure_snapshot)

        # Initialise round-robin with fresh node list
        if hasattr(baseline, "initialise"):
            baseline.initialise(infra_copy)

        b_results: list = []
        for app in apps:
            result = baseline.place(app, infra_copy)
            b_results.append(result)
            if BASELINE_VERBOSE:
                print("  " + result.summary_line())

        placed   = sum(1 for r in b_results if r.admission_status == "placed")
        rejected = sum(1 for r in b_results if r.admission_status == "rejected")
        avg_lat  = (sum(r.latency_estimate_ms for r in b_results if r.selected_tier)
                    / max(placed, 1))
        avg_ms   = sum(r.pipeline_ms for r in b_results) / len(b_results)

        print(f"  [{b_name}] placed={placed}  rejected={rejected}  "
              f"avg_latency={avg_lat:.1f}ms  "
              f"avg_decision={avg_ms:.2f}ms")

        results[b_name] = b_results

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  COMPARISON SUMMARY — used by figures.py for IEEE tables and plots
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(baseline_results: dict[str, list],
                            agent_decisions:  list) -> dict:
    """
    Build a unified comparison table across all baselines + the
    multi-agent system. This dict is the primary input for figures.py.

    Returns:
        {
          "system_name": {
            "placed_pct":         float,   # % successfully placed
            "rejected_pct":       float,
            "avg_latency_ms":     float,
            "avg_decision_ms":    float,
            "avg_score":          float,
            "tier_distribution":  dict,    # {tier: count}
            "edge_pct":           float,   # % placed on edge/local
            "cloud_pct":          float,
          }
        }
    """
    table: dict = {}

    # Multi-agent results
    total = len(agent_decisions)
    if total > 0:
        placed   = [d for d in agent_decisions if d.final_status == "placed"]
        n_placed = len(placed)
        tiers    = {}
        for d in placed:
            t = d.placement.selected_tier
            tiers[t] = tiers.get(t, 0) + 1

        table["multi_agent_TOPSIS"] = {
            "placed_pct":       round(n_placed / total * 100, 2),
            "rejected_pct":     round((total - n_placed) / total * 100, 2),
            "avg_latency_ms":   round(
                sum(d.placement.latency_estimate_ms for d in placed) / max(n_placed, 1), 2),
            "avg_decision_ms":  round(
                sum(d.total_pipeline_ms for d in agent_decisions) / total, 2),
            "avg_score":        round(
                sum(d.placement.topsis_cc * 100 for d in placed) / max(n_placed, 1), 2),
            "tier_distribution": tiers,
            "edge_pct": round(
                (tiers.get("edge", 0) + tiers.get("local", 0)) / max(n_placed, 1) * 100, 2),
            "cloud_pct": round(tiers.get("cloud", 0) / max(n_placed, 1) * 100, 2),
        }

    # Baseline results
    baseline_labels = {
        "cloud_only":       "B1_Cloud-Only",
        "round_robin":      "B2_Round-Robin",
        "static_threshold": "B3_Static-Threshold",
        "greedy_first_fit": "B4_Greedy-First-Fit",
    }

    for b_name, b_results in baseline_results.items():
        n = len(b_results)
        if n == 0:
            continue
        placed_r   = [r for r in b_results if r.admission_status == "placed"]
        n_placed_r = len(placed_r)
        tiers_r    = {}
        for r in placed_r:
            tiers_r[r.selected_tier] = tiers_r.get(r.selected_tier, 0) + 1

        label = baseline_labels.get(b_name, b_name)
        table[label] = {
            "placed_pct":       round(n_placed_r / n * 100, 2),
            "rejected_pct":     round((n - n_placed_r) / n * 100, 2),
            "avg_latency_ms":   round(
                sum(r.latency_estimate_ms for r in placed_r) / max(n_placed_r, 1), 2),
            "avg_decision_ms":  round(
                sum(r.pipeline_ms for r in b_results) / n, 2),
            "avg_score":        round(
                sum(r.placement_score for r in placed_r) / max(n_placed_r, 1), 2),
            "tier_distribution": tiers_r,
            "edge_pct": round(
                (tiers_r.get("edge", 0) + tiers_r.get("local", 0))
                / max(n_placed_r, 1) * 100, 2),
            "cloud_pct": round(tiers_r.get("cloud", 0) / max(n_placed_r, 1) * 100, 2),
        }

    return table


def print_comparison_table(table: dict) -> None:
    print("\n" + "=" * 90)
    print("  IEEE COMPARISON TABLE — Multi-Agent vs Baselines")
    print("=" * 90)
    hdr = f"  {'System':<28} {'Placed%':>8} {'Rej%':>6} {'AvgLat':>8} {'AvgDec':>8} {'Score':>7} {'Edge%':>7} {'Cloud%':>7}"
    print(hdr)
    print("  " + "─" * 86)
    for sys_name, m in table.items():
        marker = " ◄" if "multi_agent" in sys_name else ""
        print(f"  {sys_name:<28} "
              f"{m['placed_pct']:>7.1f}% "
              f"{m['rejected_pct']:>5.1f}% "
              f"{m['avg_latency_ms']:>7.1f}ms "
              f"{m['avg_decision_ms']:>7.2f}ms "
              f"{m['avg_score']:>6.1f} "
              f"{m['edge_pct']:>6.1f}% "
              f"{m['cloud_pct']:>6.1f}%"
              f"{marker}")
    print("=" * 90 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR INTEGRATION INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def run(config: dict, shared_state: dict) -> dict:
    """
    Integration entry point for OrchestratorAgent.py.

    Reads from shared_state:
        shared_state["applications"]   → list[ApplicationInstance]
        shared_state["infrastructure"] → Infrastructure
        shared_state["agent_decisions"]→ list[AgentDecision]  (for comparison table)

    Writes to shared_state:
        shared_state["baseline_results"]    → dict {name: list[BaselineResult]}
        shared_state["comparison_table"]    → dict (for figures.py)

    Returns:
        {"baseline_results": dict, "comparison_table": dict,
         "output_path": str, "status": "ok"|"error"}
    """
    import traceback as tb
    try:
        global ACTIVE_BASELINES, BASELINE_VERBOSE
        ACTIVE_BASELINES = config.get("active_baselines", ACTIVE_BASELINES)
        BASELINE_VERBOSE = config.get("baseline_verbose", BASELINE_VERBOSE)

        apps            = shared_state.get("applications",   [])
        infrastructure  = shared_state.get("infrastructure", None)
        agent_decisions = shared_state.get("agent_decisions", [])

        if not apps:
            raise ValueError("shared_state['applications'] is empty.")
        if infrastructure is None:
            raise ValueError("shared_state['infrastructure'] is None.")

        baseline_results = run_all_baselines(apps, infrastructure, ACTIVE_BASELINES)
        comparison_table = build_comparison_table(baseline_results, agent_decisions)

        shared_state["baseline_results"] = baseline_results
        shared_state["comparison_table"] = comparison_table

        print_comparison_table(comparison_table)

        out_dir = config.get("output_dir", ".")
        os.makedirs(out_dir, exist_ok=True)

        # Save per-baseline results
        b_path = os.path.join(out_dir, "experiment_baselines.json")
        with open(b_path, "w") as f:
            json.dump(
                {k: [r.to_dict() for r in v] for k, v in baseline_results.items()},
                f, indent=2
            )

        # Save comparison table
        c_path = os.path.join(out_dir, "experiment_comparison_table.json")
        with open(c_path, "w") as f:
            json.dump(comparison_table, f, indent=2)

        return {
            "baseline_results": baseline_results,
            "comparison_table": comparison_table,
            "output_path":      b_path,
            "comparison_path":  c_path,
            "status":           "ok",
        }

    except Exception as e:
        return {
            "baseline_results": {},
            "comparison_table": {},
            "output_path":      "",
            "status":           "error",
            "error":            str(e),
            "traceback":        tb.format_exc(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[baselines.py] Standalone mode\n")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import applications        as apps_module
    import computing_resources as cr_module
    import agents              as agents_module

    _shared: dict = {}
    _config: dict = {
        "num_apps":          15,
        "domains":           ["healthcare", "industrial", "general"],
        "subtypes":          {"healthcare": None, "industrial": None, "general": None},
        "seed":              42,
        "requirement_mode":  "random",
        "active_tiers":      ["cloud", "fog", "edge", "local"],
        "num_fog_nodes":     2,
        "num_edge_nodes":    4,
        "num_local_nodes":   2,
        "variation_pct":     0.15,
        "llm_backend":       "mock",
        "agent_verbose":     False,    # suppress agent detail in baseline run
        "active_baselines":  ACTIVE_BASELINES,
        "baseline_verbose":  True,
        "output_dir":        ".",
    }

    r1 = apps_module.run(_config, _shared)
    if r1["status"] != "ok":
        print(f"ERROR apps: {r1.get('error')}"); sys.exit(1)

    r2 = cr_module.run(_config, _shared)
    if r2["status"] != "ok":
        print(f"ERROR infra: {r2.get('error')}"); sys.exit(1)

    r3 = agents_module.run(_config, _shared)
    if r3["status"] != "ok":
        print(f"ERROR agents: {r3.get('error')}"); sys.exit(1)

    r4 = run(_config, _shared)
    if r4["status"] != "ok":
        print(f"ERROR baselines: {r4.get('error')}")
        print(r4.get("traceback", ""))
        sys.exit(1)

    print(f"\n[OK] Baselines saved → {r4['output_path']}")
    print(f"[OK] Comparison table → {r4['comparison_path']}\n")
