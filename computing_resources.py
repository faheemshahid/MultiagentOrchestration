"""
=============================================================================
computing_resources.py
=============================================================================
Intent-Aware Multi-Agent AI Orchestration in Edge-Cloud Continuum
-----------------------------------------------------------------------------
PURPOSE:
    Defines all computing infrastructure across four tiers:
      - Cloud  : single high-capacity datacenter
      - Fog    : N configurable mid-tier gateway nodes
      - Edge   : N configurable lightweight compute nodes
      - Local  : N minimal devices (Raspberry Pi class, Jetson-ready)

    Each node has realistic CPU, GPU, RAM, storage, bandwidth, and
    network latency specifications. Resource usage is tracked live so
    the orchestrator can make placement decisions and the analysis
    module can plot utilisation graphs for IEEE figures.

EXPERIMENT CONFIGURATION (edit before each run):
    ACTIVE_TIERS        — which tiers participate in this experiment
    NUM_FOG_NODES       — how many fog nodes to instantiate
    NUM_EDGE_NODES      — how many edge nodes to instantiate
    NUM_LOCAL_NODES     — how many local (Pi/Jetson) nodes to instantiate
    NODE_VARIATION_PCT  — % random variation around nominal specs per node
    RANDOMISE_SEED      — reproducibility seed (None = fully random)
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT CONFIGURATION  ←  EDIT HERE BEFORE EACH RUN
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_TIERS: list[str] = [
    "cloud",
    "fog",
    "edge",
    "local",
]
# Options: "cloud" | "fog" | "edge" | "local"
# Single tier  : ACTIVE_TIERS = ["edge"]
# Two-tier     : ACTIVE_TIERS = ["fog", "cloud"]
# Full pipeline: ACTIVE_TIERS = ["local", "edge", "fog", "cloud"]

NUM_FOG_NODES:   int = 5      # 1–8  recommended (fog gateway servers)
NUM_EDGE_NODES:  int = 5      # 1–16 recommended (edge compute boxes)
NUM_LOCAL_NODES: int = 5      # 1–8  recommended (Pi / Jetson devices)

# Capacity window: each node's specs are sampled uniformly within
# [nominal × (1 - VARIATION), nominal × (1 + VARIATION)]
# Set to 0.0 for identical nodes (good for controlled experiments).
NODE_VARIATION_PCT: float = 0.15   # 15 % variation around nominal

RANDOMISE_SEED: int | None = 42    # int = reproducible; None = random

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import random
import time
import uuid
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class Tier(str, Enum):
    CLOUD = "cloud"
    FOG   = "fog"
    EDGE  = "edge"
    LOCAL = "local"


class NodeStatus(str, Enum):
    IDLE      = "idle"
    BUSY      = "busy"
    OVERLOADED = "overloaded"
    OFFLINE   = "offline"


class HardwareClass(str, Enum):
    """Maps to real hardware families for paper reproducibility."""
    # Cloud
    DATACENTER_X86   = "datacenter_x86"       # e.g. AMD EPYC 9654 rack
    # Fog
    EDGE_SERVER_X86  = "edge_server_x86"       # e.g. Dell PowerEdge R250
    EDGE_SERVER_ARM  = "edge_server_arm"       # e.g. Ampere Altra
    # Edge
    MINI_PC_X86      = "mini_pc_x86"           # e.g. Intel NUC 13 Pro
    JETSON_ORIN      = "jetson_orin"           # Nvidia Jetson Orin NX 16GB
    # Local
    RASPBERRY_PI_5   = "raspberry_pi_5"        # Pi 5 / 8GB
    RASPBERRY_PI_4   = "raspberry_pi_4"        # Pi 4B / 4GB
    JETSON_NANO      = "jetson_nano"           # Jetson Nano 4GB


# ─────────────────────────────────────────────────────────────────────────────
#  NODE SPECIFICATION DATACLASS  (static hardware template)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeSpec:
    """
    Static nominal specification for one node class.
    All capacity values are the nominal (middle) point.
    Actual node instances are sampled within ±NODE_VARIATION_PCT.
    """
    tier:             Tier
    hw_class:         HardwareClass
    description:      str

    # ── Compute ──────────────────────────────────────────────────────────────
    cpu_cores:        int          # logical cores
    cpu_freq_mhz:     int          # base clock MHz
    cpu_tdp_w:        float        # thermal design power (Watts) — for energy fig
    gpu_vram_mb:      int          # 0 = no GPU
    gpu_tflops_fp16:  float        # 0.0 = no GPU
    ram_mb:           int          # system RAM in MB
    storage_mb:       int          # fast storage (SSD/eMMC) in MB

    # ── Network (to the NEXT tier upward) ────────────────────────────────────
    uplink_kbps:      int          # uplink bandwidth to next tier (kbps)
    downlink_kbps:    int          # downlink bandwidth from next tier (kbps)
    latency_to_next_ms: float      # one-way propagation latency (ms) to fog/cloud
    jitter_ms:        float        # network jitter (ms)
    packet_loss_pct:  float        # baseline packet loss %

    # ── LLM model that runs ON this node ─────────────────────────────────────
    llm_model_name:   str          # e.g. "llama3.2:1b", "llama3.2:8b", "llama3.1:70b"
    llm_runtime:      str          # "llama_cpp" | "ollama" | "api"
    llm_tokens_per_sec: float      # estimated inference throughput


# ─────────────────────────────────────────────────────────────────────────────
#  NODE INSTANCE DATACLASS  (live runtime node with sampled capacity + tracking)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComputeNode:
    """
    A live node instance with:
      - sampled capacity (slightly varied from nominal)
      - real-time resource usage tracking
      - allocation/release methods used by the orchestrator
    """
    node_id:          str
    tier:             str
    hw_class:         str
    description:      str

    # ── Sampled total capacity ────────────────────────────────────────────────
    total_cpu_cores:      float
    total_cpu_freq_mhz:   int
    total_gpu_vram_mb:    int
    total_ram_mb:         int
    total_storage_mb:     int
    total_uplink_kbps:    int
    total_downlink_kbps:  int
    cpu_tdp_w:            float
    gpu_tflops_fp16:      float

    # ── Network characteristics ───────────────────────────────────────────────
    latency_to_next_ms:   float
    jitter_ms:            float
    packet_loss_pct:      float

    # ── LLM info ─────────────────────────────────────────────────────────────
    llm_model_name:       str
    llm_runtime:          str
    llm_tokens_per_sec:   float

    # ── Live usage (updated by orchestrator during experiment) ────────────────
    used_cpu_cores:       float = 0.0
    used_gpu_vram_mb:     int   = 0
    used_ram_mb:          int   = 0
    used_storage_mb:      int   = 0
    used_uplink_kbps:     int   = 0
    active_app_count:     int   = 0
    status:               str   = NodeStatus.IDLE.value
    created_at:           float = field(default_factory=time.time)

    # ── History for IEEE figures ──────────────────────────────────────────────
    utilisation_log: list = field(default_factory=list)  # list of snapshot dicts

    # ─────────────────────────────────────────────────────────────────────────
    #  DERIVED PROPERTIES
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def available_cpu_cores(self) -> float:
        return max(0.0, self.total_cpu_cores - self.used_cpu_cores)

    @property
    def available_gpu_vram_mb(self) -> int:
        return max(0, self.total_gpu_vram_mb - self.used_gpu_vram_mb)

    @property
    def available_ram_mb(self) -> int:
        return max(0, self.total_ram_mb - self.used_ram_mb)

    @property
    def available_storage_mb(self) -> int:
        return max(0, self.total_storage_mb - self.used_storage_mb)

    @property
    def available_uplink_kbps(self) -> int:
        return max(0, self.total_uplink_kbps - self.used_uplink_kbps)

    @property
    def cpu_utilisation_pct(self) -> float:
        if self.total_cpu_cores == 0:
            return 0.0
        return round((self.used_cpu_cores / self.total_cpu_cores) * 100, 2)

    @property
    def ram_utilisation_pct(self) -> float:
        if self.total_ram_mb == 0:
            return 0.0
        return round((self.used_ram_mb / self.total_ram_mb) * 100, 2)

    @property
    def gpu_utilisation_pct(self) -> float:
        if self.total_gpu_vram_mb == 0:
            return 0.0
        return round((self.used_gpu_vram_mb / self.total_gpu_vram_mb) * 100, 2)

    @property
    def effective_latency_ms(self) -> float:
        """Latency with jitter added (realistic for simulation)."""
        return round(self.latency_to_next_ms + random.gauss(0, self.jitter_ms), 3)

    # ─────────────────────────────────────────────────────────────────────────
    #  RESOURCE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def can_accommodate(self, app) -> tuple[bool, str]:
        """
        Check if this node has enough free resources for an ApplicationInstance.
        Returns (True, "ok") or (False, reason_string).

        NOTE: latency_to_next_ms is the uplink propagation latency to the
        next tier — it is NOT checked here because the orchestrator already
        uses it as a TOPSIS scoring criterion. Rejecting on latency here
        would cause near-100% rejection rates since app.latency_ms_max is
        an end-to-end SLA, not an uplink hop latency.
        """
        if self.status == NodeStatus.OFFLINE.value:
            return False, "node_offline"
        if self.available_cpu_cores < app.cpu_cores:
            return False, f"cpu_insufficient ({self.available_cpu_cores:.1f} < {app.cpu_cores})"
        if self.available_ram_mb < app.ram_mb:
            return False, f"ram_insufficient ({self.available_ram_mb} < {app.ram_mb} MB)"
        # GPU: only reject if app explicitly needs GPU and node has none at all
        if app.gpu_memory_mb > 0 and self.total_gpu_vram_mb == 0:
            return False, f"gpu_not_available (node has no GPU)"
        if app.gpu_memory_mb > 0 and self.available_gpu_vram_mb < app.gpu_memory_mb:
            return False, f"gpu_insufficient ({self.available_gpu_vram_mb} < {app.gpu_memory_mb} MB)"
        return True, "ok"

    def allocate(self, app, app_lifetime_s: float = 0.0) -> bool:
        """
        Reserve resources for one ApplicationInstance.
        app_lifetime_s: seconds to hold resources before auto-release.
                        0 = hold indefinitely (manual release required).
        Returns True if successful, False if resources unavailable.
        """
        ok, reason = self.can_accommodate(app)
        if not ok:
            return False

        self.used_cpu_cores   += app.cpu_cores
        self.used_gpu_vram_mb += app.gpu_memory_mb
        self.used_ram_mb      += app.ram_mb
        self.used_uplink_kbps += app.bandwidth_kbps
        self.active_app_count += 1
        self._update_status()
        self._log_snapshot(event=f"allocate:{app.instance_id}")
        return True

    def release(self, app) -> None:
        """Release resources after an application finishes."""
        self.used_cpu_cores   = max(0.0, self.used_cpu_cores   - app.cpu_cores)
        self.used_gpu_vram_mb = max(0,   self.used_gpu_vram_mb - app.gpu_memory_mb)
        self.used_ram_mb      = max(0,   self.used_ram_mb      - app.ram_mb)
        self.used_uplink_kbps = max(0,   self.used_uplink_kbps - app.bandwidth_kbps)
        self.active_app_count = max(0,   self.active_app_count - 1)
        self._update_status()
        self._log_snapshot(event=f"release:{app.instance_id}")

    def _update_status(self) -> None:
        if self.cpu_utilisation_pct >= 90 or self.ram_utilisation_pct >= 90:
            self.status = NodeStatus.OVERLOADED.value
        elif self.active_app_count > 0:
            self.status = NodeStatus.BUSY.value
        else:
            self.status = NodeStatus.IDLE.value

    def _log_snapshot(self, event: str = "") -> None:
        """Append a utilisation snapshot — used later by figures.py."""
        self.utilisation_log.append({
            "timestamp":        time.time(),
            "event":            event,
            "cpu_pct":          self.cpu_utilisation_pct,
            "ram_pct":          self.ram_utilisation_pct,
            "gpu_pct":          self.gpu_utilisation_pct,
            "active_apps":      self.active_app_count,
            "status":           self.status,
            "avail_cpu_cores":  self.available_cpu_cores,
            "avail_ram_mb":     self.available_ram_mb,
            "avail_uplink_kbps": self.available_uplink_kbps,
        })

    def snapshot(self) -> dict:
        """Current state dict — used by OrchestratorAgent for logging."""
        return {
            "node_id":              self.node_id,
            "tier":                 self.tier,
            "hw_class":             self.hw_class,
            "status":               self.status,
            "cpu_utilisation_pct":  self.cpu_utilisation_pct,
            "ram_utilisation_pct":  self.ram_utilisation_pct,
            "gpu_utilisation_pct":  self.gpu_utilisation_pct,
            "active_apps":          self.active_app_count,
            "available_cpu_cores":  self.available_cpu_cores,
            "available_ram_mb":     self.available_ram_mb,
            "available_gpu_vram_mb": self.available_gpu_vram_mb,
            "available_uplink_kbps": self.available_uplink_kbps,
            "latency_to_next_ms":   self.latency_to_next_ms,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["available_cpu_cores"]   = self.available_cpu_cores
        d["available_ram_mb"]      = self.available_ram_mb
        d["available_gpu_vram_mb"] = self.available_gpu_vram_mb
        d["cpu_utilisation_pct"]   = self.cpu_utilisation_pct
        d["ram_utilisation_pct"]   = self.ram_utilisation_pct
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  NODE SPECIFICATION CATALOGUE
#  Realistic specs based on real hardware datasheets (2024 values)
# ─────────────────────────────────────────────────────────────────────────────

NODE_SPECS: dict[str, NodeSpec] = {

    # ══════════════════════════════════════════════════════════════════════════
    # CLOUD TIER — Single large datacenter (AMD EPYC 9654 class rack)
    # Latency to fog: 20–80ms (realistic WAN, CDN-assisted)
    # ══════════════════════════════════════════════════════════════════════════
    "cloud_datacenter": NodeSpec(
        tier              = Tier.CLOUD,
        hw_class          = HardwareClass.DATACENTER_X86,
        description       = "Primary cloud datacenter — AMD EPYC 9654 + 8×A100 80GB cluster",
        cpu_cores         = 300,           # calibrated for 1000-app healthcare experiment
        cpu_freq_mhz      = 3700,         # boost clock
        cpu_tdp_w         = 720.0,        # 2× 360W TDP
        gpu_vram_mb       = 327680,       # 4× A100 80GB = 320GB VRAM
        gpu_tflops_fp16   = 1248.0,       # 4× 312 TFLOPS FP16
        ram_mb            = 1048576,      # 1 TB DDR5 ECC (calibrated)
        storage_mb        = 20480000,     # 20 TB NVMe pool
        uplink_kbps       = 10_000_000,   # 10 Gbps datacenter uplink
        downlink_kbps     = 10_000_000,
        latency_to_next_ms = 50.0,        # cloud access from IoT device: ~50ms WAN round-trip
        jitter_ms         = 0.5,
        packet_loss_pct   = 0.001,
        llm_model_name    = "llama3.1:70b",
        llm_runtime       = "ollama",
        llm_tokens_per_sec = 85.0,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # FOG TIER — Mid-tier gateway server (Dell PowerEdge R250 class)
    # Latency to cloud: 20–60ms (realistic metro/WAN link)
    # ══════════════════════════════════════════════════════════════════════════
    "fog_server": NodeSpec(
        tier              = Tier.FOG,
        hw_class          = HardwareClass.EDGE_SERVER_X86,
        description       = "Fog gateway server — Intel Xeon E-2400 + RTX 4090 24GB",
        cpu_cores         = 200,           # calibrated: 5 nodes × 200 = 1000 fog cores
        cpu_freq_mhz      = 3600,
        cpu_tdp_w         = 95.0,
        gpu_vram_mb       = 24576,        # RTX 4090 24 GB
        gpu_tflops_fp16   = 330.0,        # RTX 4090 FP16
        ram_mb            = 524288,       # 512 GB DDR5 ECC (calibrated)
        storage_mb        = 4096000,      # 2 TB NVMe SSD
        uplink_kbps       = 1_000_000,    # 1 Gbps to cloud
        downlink_kbps     = 1_000_000,
        latency_to_next_ms = 35.0,        # ~35ms WAN to cloud datacenter
        jitter_ms         = 8.0,
        packet_loss_pct   = 0.05,
        llm_model_name    = "llama3.2:8b-instruct-q4_K_M",
        llm_runtime       = "ollama",
        llm_tokens_per_sec = 42.0,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # EDGE TIER — Lightweight compute node (Intel NUC 13 Pro class)
    # Latency to fog: 2–15ms (LAN / local 5G)
    # ══════════════════════════════════════════════════════════════════════════
    "edge_mini_pc": NodeSpec(
        tier              = Tier.EDGE,
        hw_class          = HardwareClass.MINI_PC_X86,
        description       = "Edge compute node — Intel Core i7-1360P NUC, no discrete GPU",
        cpu_cores         = 120,           # calibrated: 5 nodes × 120 = 600 edge cores
        cpu_freq_mhz      = 2200,         # base clock (boost 5.0 GHz)
        cpu_tdp_w         = 28.0,
        gpu_vram_mb       = 0,            # integrated Iris Xe — not usable for LLM
        gpu_tflops_fp16   = 0.0,
        ram_mb            = 262144,       # 256 GB LPDDR5 (calibrated)
        storage_mb        = 1024000,       # 512 GB NVMe SSD
        uplink_kbps       = 100_000,      # 100 Mbps LAN / 5G uplink
        downlink_kbps     = 100_000,
        latency_to_next_ms = 8.0,         # ~8ms LAN to fog gateway
        jitter_ms         = 2.0,
        packet_loss_pct   = 0.1,
        llm_model_name    = "llama3.2:3b-instruct-q8_0",
        llm_runtime       = "llama_cpp",
        llm_tokens_per_sec = 12.0,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # LOCAL TIER — Raspberry Pi 5 (8GB) class device
    # Latency to edge: 1–5ms (wired LAN or local WiFi)
    # Jetson Nano variant also defined below for GPU-capable local nodes.
    # ══════════════════════════════════════════════════════════════════════════
    "local_pi5": NodeSpec(
        tier              = Tier.LOCAL,
        hw_class          = HardwareClass.RASPBERRY_PI_5,
        description       = "Local sensor node — Raspberry Pi 5 / 8 GB LPDDR4X",
        cpu_cores         = 40,            # calibrated: 5 nodes × 40 = 200 local cores
        cpu_freq_mhz      = 2400,         # 2.4 GHz
        cpu_tdp_w         = 12.0,         # full load ~12W
        gpu_vram_mb       = 0,            # VideoCore VII — not LLM capable
        gpu_tflops_fp16   = 0.0,
        ram_mb            = 65536,        # 64 GB shared (calibrated)
        storage_mb        = 64768,        # 32 GB microSD (class A2)
        uplink_kbps       = 10_000,       # 100 Mbps Ethernet, realistic ~10 Mbps used
        downlink_kbps     = 10_000,
        latency_to_next_ms = 3.0,         # ~3ms to edge node on same LAN
        jitter_ms         = 1.0,
        packet_loss_pct   = 0.2,
        llm_model_name    = "llama3.2:1b-instruct-q8_0",
        llm_runtime       = "llama_cpp",
        llm_tokens_per_sec = 3.5,
    ),

    # ── Jetson Nano variant (GPU-capable local node, activated later) ─────────
    "local_jetson_nano": NodeSpec(
        tier              = Tier.LOCAL,
        hw_class          = HardwareClass.JETSON_NANO,
        description       = "Local GPU node — Nvidia Jetson Nano 4GB (future hardware)",
        cpu_cores         = 40,            # calibrated: matches local_pi5 for consistency
        cpu_freq_mhz      = 1430,
        cpu_tdp_w         = 10.0,
        gpu_vram_mb       = 4096,         # 4 GB shared LPDDR4 (GPU-accessible)
        gpu_tflops_fp16   = 0.47,         # 472 GFLOPS FP16
        ram_mb            = 32768,        # 32 GB shared (calibrated)
        storage_mb        = 32384,        # 16 GB eMMC
        uplink_kbps       = 10_000,
        downlink_kbps     = 10_000,
        latency_to_next_ms = 4.0,
        jitter_ms         = 1.5,
        packet_loss_pct   = 0.3,
        llm_model_name    = "llama3.2:1b-instruct-q4_K_M",
        llm_runtime       = "llama_cpp",
        llm_tokens_per_sec = 5.0,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  NODE FACTORY — instantiates ComputeNode from a NodeSpec with variation
# ─────────────────────────────────────────────────────────────────────────────

def _vary(nominal: float, variation: float, rng: random.Random,
          floor: float = 0.0) -> float:
    """Sample a value within ±variation fraction of nominal."""
    lo = nominal * (1.0 - variation)
    hi = nominal * (1.0 + variation)
    return max(floor, round(rng.uniform(lo, hi), 2))

def _vary_int(nominal: int, variation: float, rng: random.Random,
              floor: int = 0) -> int:
    lo = int(nominal * (1.0 - variation))
    hi = int(nominal * (1.0 + variation))
    return max(floor, rng.randint(min(lo, hi), max(lo, hi)))


def make_node(spec: NodeSpec, index: int, rng: random.Random,
              variation: float = NODE_VARIATION_PCT) -> ComputeNode:
    """
    Instantiate one ComputeNode from a NodeSpec.
    Each node gets slightly different capacity to simulate real heterogeneity.
    """
    tier_prefix = spec.tier.value[0].upper()  # C / F / E / L
    node_id = f"{tier_prefix}{index:02d}-{str(uuid.uuid4())[:4].upper()}"

    return ComputeNode(
        node_id             = node_id,
        tier                = spec.tier.value,
        hw_class            = spec.hw_class.value,
        description         = spec.description,

        total_cpu_cores     = _vary(spec.cpu_cores,      variation, rng, floor=1.0),
        total_cpu_freq_mhz  = _vary_int(spec.cpu_freq_mhz, variation, rng, floor=400),
        total_gpu_vram_mb   = _vary_int(spec.gpu_vram_mb,  variation, rng, floor=0),
        total_ram_mb        = _vary_int(spec.ram_mb,        variation, rng, floor=256),
        total_storage_mb    = _vary_int(spec.storage_mb,    variation, rng, floor=1000),
        total_uplink_kbps   = _vary_int(spec.uplink_kbps,   variation, rng, floor=64),
        total_downlink_kbps = _vary_int(spec.downlink_kbps, variation, rng, floor=64),
        cpu_tdp_w           = _vary(spec.cpu_tdp_w,      variation, rng, floor=1.0),
        gpu_tflops_fp16     = _vary(spec.gpu_tflops_fp16, variation, rng, floor=0.0),

        latency_to_next_ms  = _vary(spec.latency_to_next_ms, variation, rng, floor=0.0),
        jitter_ms           = spec.jitter_ms,
        packet_loss_pct     = spec.packet_loss_pct,

        llm_model_name      = spec.llm_model_name,
        llm_runtime         = spec.llm_runtime,
        llm_tokens_per_sec  = _vary(spec.llm_tokens_per_sec, variation, rng, floor=0.5),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  INFRASTRUCTURE BUILDER — creates the full multi-tier node pool
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Infrastructure:
    """
    The complete computing infrastructure for one experiment run.
    Holds all nodes organised by tier.
    Provides helper methods used by the orchestrator.
    """
    cloud_nodes: list[ComputeNode] = field(default_factory=list)
    fog_nodes:   list[ComputeNode] = field(default_factory=list)
    edge_nodes:  list[ComputeNode] = field(default_factory=list)
    local_nodes: list[ComputeNode] = field(default_factory=list)

    def all_nodes(self) -> list[ComputeNode]:
        return (self.cloud_nodes + self.fog_nodes +
                self.edge_nodes  + self.local_nodes)

    def nodes_for_tier(self, tier: str) -> list[ComputeNode]:
        return {
            "cloud": self.cloud_nodes,
            "fog":   self.fog_nodes,
            "edge":  self.edge_nodes,
            "local": self.local_nodes,
        }.get(tier, [])

    def available_nodes(self, tier: str) -> list[ComputeNode]:
        """Return non-offline nodes — busy/overloaded nodes still accept apps."""
        return [
            n for n in self.nodes_for_tier(tier)
            if n.status != NodeStatus.OFFLINE.value
        ]

    def reset_all(self) -> None:
        """Reset all nodes to zero usage between experiment runs."""
        for node in self.all_nodes():
            node.used_cpu_cores   = 0.0
            node.used_gpu_vram_mb = 0
            node.used_ram_mb      = 0
            node.used_uplink_kbps = 0
            node.active_app_count = 0
            node.status           = NodeStatus.IDLE.value
            node.utilisation_log  = []

    def capacity_summary(self) -> dict:
        """Per-tier totals of max / used / available CPU, RAM, GPU."""
        summary = {}
        for tier in ["cloud", "fog", "edge", "local"]:
            nodes = self.nodes_for_tier(tier)
            if not nodes:
                continue
            summary[tier] = {
                "max_cpu_cores":       round(sum(n.total_cpu_cores    for n in nodes), 1),
                "used_cpu_cores":      round(sum(n.used_cpu_cores     for n in nodes), 1),
                "available_cpu_cores": round(sum(n.available_cpu_cores for n in nodes), 1),
                "max_ram_mb":          sum(n.total_ram_mb    for n in nodes),
                "used_ram_mb":         sum(n.used_ram_mb     for n in nodes),
                "available_ram_mb":    sum(n.available_ram_mb for n in nodes),
                "max_gpu_mb":          sum(n.total_gpu_vram_mb for n in nodes),
                "used_gpu_mb":         sum(n.used_gpu_vram_mb  for n in nodes),
                "node_count":          len(nodes),
                "active_apps":         sum(n.active_app_count for n in nodes),
            }
        return summary

    def best_node_for(self, app, tier: str) -> Optional[ComputeNode]:
        """
        Select the least-loaded node in a tier that can accommodate app.
        Returns None if no node is available.
        """
        candidates = [
            n for n in self.available_nodes(tier)
            if n.can_accommodate(app)[0]
        ]
        if not candidates:
            return None
        # Pick least CPU-loaded node
        return min(candidates, key=lambda n: n.cpu_utilisation_pct)

    def global_snapshot(self) -> dict:
        """Full infrastructure state — written to shared_state by run()."""
        return {
            tier: [n.snapshot() for n in self.nodes_for_tier(tier)]
            for tier in ["cloud", "fog", "edge", "local"]
        }

    def summary(self) -> dict:
        totals = {}
        for tier in ["cloud", "fog", "edge", "local"]:
            nodes = self.nodes_for_tier(tier)
            if not nodes:
                continue
            totals[tier] = {
                "node_count":       len(nodes),
                "total_cpu_cores":  sum(n.total_cpu_cores  for n in nodes),
                "total_ram_gb":     round(sum(n.total_ram_mb for n in nodes) / 1024, 1),
                "total_gpu_vram_gb": round(sum(n.total_gpu_vram_mb for n in nodes) / 1024, 1),
                "total_storage_tb": round(sum(n.total_storage_mb for n in nodes) / 1_000_000, 2),
                "avg_latency_to_next_ms": round(
                    sum(n.latency_to_next_ms for n in nodes) / len(nodes), 2),
                "llm_models":       list({n.llm_model_name for n in nodes}),
            }
        return totals


def build_infrastructure(
    active_tiers:     list[str]  = ACTIVE_TIERS,
    num_fog_nodes:    int        = NUM_FOG_NODES,
    num_edge_nodes:   int        = NUM_EDGE_NODES,
    num_local_nodes:  int        = NUM_LOCAL_NODES,
    variation_pct:    float      = NODE_VARIATION_PCT,
    seed:             int | None = RANDOMISE_SEED,
) -> Infrastructure:
    """
    Build the full infrastructure for one experiment run.
    Returns an Infrastructure object holding all ComputeNode instances.
    """
    rng = random.Random(seed)
    infra = Infrastructure()

    if "cloud" in active_tiers:
        infra.cloud_nodes.append(
            make_node(NODE_SPECS["cloud_datacenter"], 1, rng, variation_pct)
        )

    if "fog" in active_tiers:
        for i in range(1, num_fog_nodes + 1):
            infra.fog_nodes.append(
                make_node(NODE_SPECS["fog_server"], i, rng, variation_pct)
            )

    if "edge" in active_tiers:
        for i in range(1, num_edge_nodes + 1):
            infra.edge_nodes.append(
                make_node(NODE_SPECS["edge_mini_pc"], i, rng, variation_pct)
            )

    if "local" in active_tiers:
        for i in range(1, num_local_nodes + 1):
            # Default: Raspberry Pi 5. Later swap to "local_jetson_nano"
            infra.local_nodes.append(
                make_node(NODE_SPECS["local_pi5"], i, rng, variation_pct)
            )

    return infra


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_infrastructure_summary(infra: Infrastructure) -> None:
    print("\n" + "=" * 70)
    print("  INFRASTRUCTURE SUMMARY")
    print("=" * 70)

    tier_order = ["cloud", "fog", "edge", "local"]
    tier_labels = {
        "cloud": "CLOUD TIER",
        "fog":   "FOG TIER",
        "edge":  "EDGE TIER",
        "local": "LOCAL TIER",
    }

    for tier in tier_order:
        nodes = infra.nodes_for_tier(tier)
        if not nodes:
            continue
        print(f"\n  ── {tier_labels[tier]} ({len(nodes)} node(s)) ──")
        for n in nodes:
            print(f"\n    Node ID     : {n.node_id}")
            print(f"    Hardware    : {n.hw_class}")
            print(f"    Description : {n.description}")
            print(f"    CPU         : {n.total_cpu_cores} cores @ {n.total_cpu_freq_mhz} MHz  ({n.cpu_tdp_w}W TDP)")
            print(f"    GPU VRAM    : {n.total_gpu_vram_mb:,} MB  ({n.gpu_tflops_fp16} TFLOPS FP16)")
            print(f"    RAM         : {n.total_ram_mb:,} MB")
            print(f"    Storage     : {n.total_storage_mb:,} MB")
            print(f"    Uplink      : {n.total_uplink_kbps:,} kbps")
            print(f"    Latency→up  : {n.latency_to_next_ms} ms  (jitter ±{n.jitter_ms} ms)")
            print(f"    LLM model   : {n.llm_model_name}  [{n.llm_runtime}]  ~{n.llm_tokens_per_sec} tok/s")
            print(f"    Status      : {n.status}")

    print()
    summary = infra.summary()
    print("  AGGREGATE TOTALS BY TIER:")
    for tier, data in summary.items():
        print(f"    {tier.upper():<8}: "
              f"{data['node_count']} node(s)  |  "
              f"CPU {data['total_cpu_cores']} cores  |  "
              f"RAM {data['total_ram_gb']} GB  |  "
              f"GPU {data['total_gpu_vram_gb']} GB VRAM  |  "
              f"Storage {data['total_storage_tb']} TB  |  "
              f"Latency→up {data.get('avg_latency_to_next_ms',0)} ms")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR INTEGRATION INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def run(config: dict, shared_state: dict) -> dict:
    """
    Integration entry point for OrchestratorAgent.py.

    Reads from config:
        "active_tiers"      list[str]
        "num_fog_nodes"     int
        "num_edge_nodes"    int
        "num_local_nodes"   int
        "variation_pct"     float
        "seed"              int | None
        "output_dir"        str

    Writes to shared_state:
        shared_state["infrastructure"]  → Infrastructure object
        shared_state["infra_summary"]   → summary dict for logging

    Returns:
        { "infrastructure": Infrastructure, "summary": dict,
          "output_path": str, "status": "ok"|"error" }
    """
    import os
    try:
        infra = build_infrastructure(
            active_tiers   = config.get("active_tiers",    ACTIVE_TIERS),
            num_fog_nodes  = config.get("num_fog_nodes",   NUM_FOG_NODES),
            num_edge_nodes = config.get("num_edge_nodes",  NUM_EDGE_NODES),
            num_local_nodes= config.get("num_local_nodes", NUM_LOCAL_NODES),
            variation_pct  = config.get("variation_pct",   NODE_VARIATION_PCT),
            seed           = config.get("seed",            RANDOMISE_SEED),
        )

        shared_state["infrastructure"] = infra
        shared_state["infra_summary"]  = infra.summary()

        output_dir = config.get("output_dir", ".")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "experiment_infrastructure.json")
        with open(output_path, "w") as f:
            json.dump(infra.global_snapshot(), f, indent=2)

        return {
            "infrastructure": infra,
            "summary":        infra.summary(),
            "output_path":    output_path,
            "status":         "ok",
        }

    except Exception as e:
        return {
            "infrastructure": None,
            "summary":        {},
            "output_path":    "",
            "status":         "error",
            "error":          str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[computing_resources.py] Running in standalone mode ...\n")

    _config = {
        "active_tiers":    ACTIVE_TIERS,
        "num_fog_nodes":   NUM_FOG_NODES,
        "num_edge_nodes":  NUM_EDGE_NODES,
        "num_local_nodes": NUM_LOCAL_NODES,
        "variation_pct":   NODE_VARIATION_PCT,
        "seed":            RANDOMISE_SEED,
        "output_dir":      ".",
    }
    _shared_state: dict = {}

    result = run(_config, _shared_state)

    if result["status"] == "ok":
        print_infrastructure_summary(result["infrastructure"])
        print(f"  Saved infrastructure → {result['output_path']}\n")
    else:
        print(f"  [ERROR] computing_resources.py failed: {result['error']}\n")
