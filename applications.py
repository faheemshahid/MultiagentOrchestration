"""
=============================================================================
applications.py
=============================================================================
Intent-Aware Multi-Agent AI Orchestration in Edge-Cloud Continuum
-----------------------------------------------------------------------------
PURPOSE:
    Defines all application types, their computing/network requirements,
    and natural-language intent sentences used by the LLM (Llama) to
    classify, semantically analyse, and assign orchestration decisions.

EXPERIMENT CONFIGURATION (edit this section before running):
    - APP_DOMAINS       : which domains to include in the experiment
    - APP_SUBTYPES      : which subtypes within each domain (None = all)
    - NUM_APPS          : total number of application instances to generate
    - RANDOMISE_SEED    : set an integer for reproducible experiments, None for random
    - REQUIREMENT_MODE  : 'random'  → sample uniformly from (min, max)
                          'nominal' → always use the nominal midpoint value
                          'worst'   → always use the max (stress test)

=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION  ←  EDIT HERE BEFORE EACH EXPERIMENT RUN
# ─────────────────────────────────────────────────────────────────────────────

APP_DOMAINS: list[str] = [
    "healthcare",
    "industrial",
    "general",
]
# Options: "healthcare" | "industrial" | "general"
# Example single-domain run: APP_DOMAINS = ["healthcare"]
# Example two-domain mix:    APP_DOMAINS = ["healthcare", "industrial"]

APP_SUBTYPES: dict[str, list[str] | None] = {
    # Set to None to include ALL subtypes, or list specific subtype keys.
    "healthcare":  None,           # None = all three healthcare subtypes
    "industrial":  None,           # None = all three industrial subtypes
    "general":     None,           # None = all three general subtypes
    # Example: "healthcare": ["emergency", "remote_monitoring"]
}

NUM_APPS: int = 30               # total application instances to generate
RANDOMISE_SEED: int | None = 42  # integer = reproducible; None = fully random
REQUIREMENT_MODE: str = "random" # "random" | "nominal" | "worst"

# ── Application lifetime (how long each app holds resources before releasing)
# Each generated app will hold resources for a random time in [min, max] seconds.
# After this interval the orchestrator releases its resources — freeing space
# for new apps. This simulates realistic IoT workload churn.
# Set both to 0.0 to disable automatic release (manual release only).
APP_LIFETIME_MIN_S: float = 2.0   # minimum seconds an app holds resources
APP_LIFETIME_MAX_S: float = 8.0   # maximum seconds an app holds resources

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import random
import time
import uuid
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class Domain(str, Enum):
    HEALTHCARE = "healthcare"
    INDUSTRIAL = "industrial"
    GENERAL    = "general"


class QoSPriority(str, Enum):
    """Maps to orchestration tier selection logic."""
    ULTRA_LOW_LATENCY  = "ultra_low_latency"   # edge only, <10 ms
    LOW_LATENCY        = "low_latency"          # edge preferred, <50 ms
    BALANCED           = "balanced"             # fog layer, <200 ms
    HIGH_THROUGHPUT    = "high_throughput"      # fog/cloud, batch OK
    BEST_EFFORT        = "best_effort"          # cloud, no hard deadline


class TierPreference(str, Enum):
    EDGE  = "edge"
    FOG   = "fog"
    CLOUD = "cloud"
    ADAPTIVE = "adaptive"   # orchestrator decides at runtime


# ─────────────────────────────────────────────────────────────────────────────
# REQUIREMENT RANGE DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RequirementRange:
    """
    Defines a realistic window (min, nominal, max) for each resource dimension.
    The experiment runner samples from this range according to REQUIREMENT_MODE.
    All values carry explicit units for IEEE paper clarity.
    """
    # ── Compute ──────────────────────────────────────────────────────────────
    cpu_cores_min:      float  = 1.0       # logical CPU cores
    cpu_cores_nom:      float  = 2.0
    cpu_cores_max:      float  = 4.0

    cpu_freq_mhz_min:   int    = 800       # MHz
    cpu_freq_mhz_nom:   int    = 1500
    cpu_freq_mhz_max:   int    = 3200

    gpu_memory_mb_min:  int    = 0         # MB VRAM (0 = CPU-only path)
    gpu_memory_mb_nom:  int    = 0
    gpu_memory_mb_max:  int    = 0

    ram_mb_min:         int    = 256       # MB system RAM
    ram_mb_nom:         int    = 512
    ram_mb_max:         int    = 1024

    storage_mb_min:     int    = 100       # MB persistent storage
    storage_mb_nom:     int    = 500
    storage_mb_max:     int    = 2000

    # ── Network ──────────────────────────────────────────────────────────────
    bandwidth_kbps_min: int    = 128       # kbps uplink to fog/cloud
    bandwidth_kbps_nom: int    = 512
    bandwidth_kbps_max: int    = 2048

    latency_ms_max:     int    = 200       # hard deadline in milliseconds
    latency_ms_target:  int    = 100       # soft target

    packet_loss_max_pct: float = 1.0       # maximum tolerable packet loss %

    # ── LLM Inference ────────────────────────────────────────────────────────
    llm_tokens_max:     int    = 256       # max tokens for LLM response
    llm_context_window: int    = 1024      # context window size (tokens)


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION SPECIFICATION DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ApplicationSpec:
    """
    Full specification of one application type.
    This is the static template — instances are generated from this spec.
    """
    key:          str                  # unique identifier e.g. "emergency_care"
    domain:       Domain
    subtype:      str                  # human-readable subtype label
    description:  str                  # one-line description for logs / papers
    qos_priority: QoSPriority
    tier_pref:    TierPreference
    requirements: RequirementRange

    # Natural-language intent sentences the LLM will parse.
    # Each sentence simulates a real human input (patient/operator/system).
    # The LLM infers domain, urgency, resource needs from these.
    intent_sentences: list[str] = field(default_factory=list)

    # IEEE paper metadata
    criticality_level: int = 1         # 1=low … 5=ultra-critical
    real_time:         bool = False
    offload_eligible:  bool = True     # can this task go to cloud?


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION INSTANCE DATACLASS  (generated at runtime)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ApplicationInstance:
    """
    A single generated application instance with sampled requirements
    and one intent sentence for the LLM to parse.
    """
    instance_id:       str
    spec_key:          str
    domain:            str
    subtype:           str
    timestamp:         float
    intent_text:       str             # the human-language sentence fed to LLM

    # Sampled requirement values (single floats, not ranges)
    cpu_cores:         float
    cpu_freq_mhz:      int
    gpu_memory_mb:     int
    ram_mb:            int
    storage_mb:        int
    bandwidth_kbps:    int
    latency_ms_max:    int
    latency_ms_target: int
    packet_loss_max:   float
    llm_tokens_max:    int

    qos_priority:      str
    tier_preference:   str
    criticality_level: int
    real_time:         bool
    offload_eligible:  bool

    # Application lifetime — how long it holds resources before releasing
    app_lifetime_s:          float = 5.0   # seconds (set by generator from config)

    # Fields populated after LLM inference
    llm_predicted_domain:    Optional[str] = None
    llm_predicted_subtype:   Optional[str] = None
    llm_predicted_tier:      Optional[str] = None
    llm_confidence:          Optional[float] = None
    llm_reasoning:           Optional[str] = None
    llm_inference_ms:        Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def llm_prompt(self) -> str:
        """
        Constructs the structured prompt sent to the Llama model.
        The LLM must return: domain, subtype, tier, confidence, reasoning.
        """
        return f"""You are an intelligent orchestration agent in an edge-cloud continuum system.

Analyse the following application request and extract its intent.

USER INPUT:
"{self.intent_text}"

TASK:
1. Identify the APPLICATION DOMAIN from: [healthcare, industrial, general]
2. Identify the APPLICATION SUBTYPE (e.g. emergency_care, predictive_maintenance)
3. Recommend the COMPUTING TIER from: [edge, fog, cloud, adaptive]
4. Estimate a CONFIDENCE score between 0.0 and 1.0
5. Provide brief REASONING (one sentence)

Respond ONLY in this exact JSON format:
{{
  "domain": "<domain>",
  "subtype": "<subtype>",
  "tier": "<tier>",
  "confidence": <float>,
  "reasoning": "<one sentence>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION CATALOGUE
#  Three domains × three subtypes = 9 application types total
#  Each subtype: realistic requirements + 10 diverse intent sentences
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

APPLICATION_CATALOGUE: dict[str, ApplicationSpec] = {

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 1: HEALTHCARE
    # ══════════════════════════════════════════════════════════════════════════

    # ── HC-1: Emergency Care (Ultra-low latency, time-critical) ───────────────
    "emergency_care": ApplicationSpec(
        key          = "emergency_care",
        domain       = Domain.HEALTHCARE,
        subtype      = "Emergency care",
        description  = "Life-critical triage and resuscitation support requiring sub-10ms edge response",
        qos_priority = QoSPriority.ULTRA_LOW_LATENCY,
        tier_pref    = TierPreference.EDGE,
        criticality_level = 5,
        real_time    = True,
        offload_eligible  = False,
        requirements = RequirementRange(
            cpu_cores_min=1,      cpu_cores_nom=2,       cpu_cores_max=4,
            cpu_freq_mhz_min=2000, cpu_freq_mhz_nom=2800, cpu_freq_mhz_max=3800,
            gpu_memory_mb_min=0,  gpu_memory_mb_nom=0,   gpu_memory_mb_max=0,
            ram_mb_min=256,       ram_mb_nom=512,         ram_mb_max=2048,
            storage_mb_min=200,   storage_mb_nom=500,    storage_mb_max=1000,
            bandwidth_kbps_min=512,  bandwidth_kbps_nom=1024,  bandwidth_kbps_max=2048,
            latency_ms_max=10,    latency_ms_target=5,
            packet_loss_max_pct=0.01,
            llm_tokens_max=128,   llm_context_window=512,
        ),
        intent_sentences=[
            "Patient in cardiac arrest, automated defibrillator needs immediate rhythm analysis and shock advisory.",
            "Trauma patient arriving in 2 minutes, blood type O-negative, allergic to penicillin — prep OR now.",
            "Infant SpO2 dropped to 78%, ventilator settings require instant adjustment, nurse unresponsive.",
            "Stroke alert: patient showing left-side facial droop and slurred speech, CT scan priority needed.",
            "ICU patient blood pressure 60/40 and falling, vasopressor protocol must activate immediately.",
            "Severe anaphylaxis reaction, patient weight 72kg, epinephrine dose and airway management required now.",
            "Multi-vehicle accident: three critical patients inbound, triage classification and resource allocation needed.",
            "Premature neonate 28 weeks, oxygen saturation unstable, NICU incubator settings need emergency override.",
            "Post-surgical haemorrhage detected, patient losing 200mL per minute, surgical team emergency page required.",
            "Septic shock confirmed, lactate 6.2 mmol/L, initiate sepsis bundle protocol without delay.",
        ],
    ),

    # ── HC-2: Medical Imaging Diagnostics (Resource-intensive, high-throughput) ─
    "medical_imaging": ApplicationSpec(
        key          = "medical_imaging",
        domain       = Domain.HEALTHCARE,
        subtype      = "Medical imaging diagnostics",
        description  = "AI-assisted radiology and pathology requiring GPU acceleration and large model inference",
        qos_priority = QoSPriority.HIGH_THROUGHPUT,
        tier_pref    = TierPreference.CLOUD,
        criticality_level = 3,
        real_time    = False,
        offload_eligible  = True,
        requirements = RequirementRange(
            cpu_cores_min=2,      cpu_cores_nom=4,       cpu_cores_max=8,
            cpu_freq_mhz_min=2400, cpu_freq_mhz_nom=3200, cpu_freq_mhz_max=4500,
            gpu_memory_mb_min=2048, gpu_memory_mb_nom=4096, gpu_memory_mb_max=8192,
            ram_mb_min=2048,      ram_mb_nom=8192,       ram_mb_max=16384,
            storage_mb_min=2048,  storage_mb_nom=10240,  storage_mb_max=51200,
            bandwidth_kbps_min=2048, bandwidth_kbps_nom=5120, bandwidth_kbps_max=10240,
            latency_ms_max=5000,  latency_ms_target=2000,
            packet_loss_max_pct=0.1,
            llm_tokens_max=512,   llm_context_window=4096,
        ),
        intent_sentences=[
            "Chest CT scan uploaded, 512 DICOM slices, AI needs to flag pulmonary nodules above 6mm.",
            "Mammography batch of 240 patients, screening for microcalcifications, results needed by end of shift.",
            "MRI brain scan shows possible glioblastoma, segmentation model should delineate tumour boundaries.",
            "Retinal fundus image for diabetic retinopathy grading, patient has had diabetes for 12 years.",
            "Whole-slide pathology image 40x magnification, identify mitotic figures for cancer staging.",
            "Echocardiogram video 60fps, automated ejection fraction measurement and wall motion scoring required.",
            "Bone density DEXA scan, patient 68-year-old female, osteoporosis risk stratification needed.",
            "Colonoscopy video stream, real-time polyp detection model must highlight regions for endoscopist.",
            "Skin lesion dermoscopy image, ABCDE criteria evaluation and melanoma probability score required.",
            "Fetal ultrasound 20-week anatomy scan, automated biometric measurements and anomaly screening.",
        ],
    ),

    # ── HC-3: Remote Patient Monitoring (Balanced, continuous stream) ─────────
    "remote_monitoring": ApplicationSpec(
        key          = "remote_monitoring",
        domain       = Domain.HEALTHCARE,
        subtype      = "Remote patient monitoring",
        description  = "Continuous vital sign streaming and anomaly detection for chronic disease management",
        qos_priority = QoSPriority.BALANCED,
        tier_pref    = TierPreference.FOG,
        criticality_level = 3,
        real_time    = True,
        offload_eligible  = True,
        requirements = RequirementRange(
            cpu_cores_min=1,      cpu_cores_nom=2,       cpu_cores_max=4,
            cpu_freq_mhz_min=800, cpu_freq_mhz_nom=1200, cpu_freq_mhz_max=2000,
            gpu_memory_mb_min=0,  gpu_memory_mb_nom=0,   gpu_memory_mb_max=512,
            ram_mb_min=256,       ram_mb_nom=512,        ram_mb_max=1024,
            storage_mb_min=500,   storage_mb_nom=2048,   storage_mb_max=10240,
            bandwidth_kbps_min=64,   bandwidth_kbps_nom=256,   bandwidth_kbps_max=1024,
            latency_ms_max=200,   latency_ms_target=100,
            packet_loss_max_pct=0.5,
            llm_tokens_max=256,   llm_context_window=2048,
        ),
        intent_sentences=[
            "My smartwatch is showing irregular heartbeat for the past 20 minutes, heart rate jumping 110 to 160.",
            "Continuous glucose monitor alert: blood sugar has been above 280 mg/dL for three consecutive readings.",
            "Home oxygen concentrator patient SpO2 reading 91%, lower than usual, feeling slightly breathless.",
            "COPD patient peak flow meter reading 180 L/min, 40% of personal best, yellow zone protocol.",
            "Hypertensive patient blood pressure 165/105 at home, took medication an hour ago, still elevated.",
            "Post-discharge heart failure patient weight increased 2.3 kg in 48 hours, ankle swelling reported.",
            "Parkinson's patient wearable showing increased tremor frequency overnight, medication timing review needed.",
            "Diabetic patient missed insulin injection, CGM trending upward, ketone level requested remotely.",
            "Elderly patient fall detected by accelerometer, no response to voice prompt, emergency contact alert.",
            "Sleep apnoea patient CPAP compliance data: 4 apnoea events per hour last night, mask leak detected.",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 2: INDUSTRIAL AUTOMATION (IIoT)
    # ══════════════════════════════════════════════════════════════════════════

    # ── IND-1: Predictive Maintenance (Low latency, real-time vibration analysis)
    "predictive_maintenance": ApplicationSpec(
        key          = "predictive_maintenance",
        domain       = Domain.INDUSTRIAL,
        subtype      = "Predictive maintenance",
        description  = "Real-time vibration, thermal and acoustic analysis to predict machine failure before it occurs",
        qos_priority = QoSPriority.LOW_LATENCY,
        tier_pref    = TierPreference.EDGE,
        criticality_level = 4,
        real_time    = True,
        offload_eligible  = True,
        requirements = RequirementRange(
            cpu_cores_min=2,      cpu_cores_nom=4,       cpu_cores_max=8,
            cpu_freq_mhz_min=1600, cpu_freq_mhz_nom=2400, cpu_freq_mhz_max=3600,
            gpu_memory_mb_min=0,  gpu_memory_mb_nom=512, gpu_memory_mb_max=4096,
            ram_mb_min=512,       ram_mb_nom=2048,        ram_mb_max=8192,
            storage_mb_min=1024,  storage_mb_nom=5120,    storage_mb_max=20480,
            bandwidth_kbps_min=256,  bandwidth_kbps_nom=1024,  bandwidth_kbps_max=4096,
            latency_ms_max=50,    latency_ms_target=20,
            packet_loss_max_pct=0.1,
            llm_tokens_max=256,   llm_context_window=2048,
        ),
        intent_sentences=[
            "CNC spindle motor vibration at 12.4 mm/s RMS, baseline was 3.2, bearing replacement window?",
            "Hydraulic pump pressure dropping 8 bar below set point intermittently, cavitation signature detected.",
            "Conveyor gearbox temperature rising 12°C above ambient over last two hours, lubrication status unknown.",
            "Industrial compressor acoustic emission pattern changed at 4 kHz band, impeller erosion suspected.",
            "Servo motor current draw spiking 40% above nominal during homing cycle, encoder fault possible.",
            "Robot arm joint 3 showing increased backlash, repeatability degraded from 0.01mm to 0.08mm.",
            "Cooling tower fan blade imbalance detected at 1X running speed, vibration spectrum shows harmonics.",
            "Injection moulding press tonnage fluctuating ±15% from setpoint, tie-bar strain gauges unbalanced.",
            "Centrifugal pump impeller wear detected, flow rate reduced 18% at same head pressure and speed.",
            "High-voltage transformer partial discharge events increasing: 12 events/hour vs 2 events/hour baseline.",
        ],
    ),

    # ── IND-2: Quality Control Vision (Resource-intensive, high-performance GPU) ─
    "quality_control": ApplicationSpec(
        key          = "quality_control",
        domain       = Domain.INDUSTRIAL,
        subtype      = "Quality control vision",
        description  = "Real-time computer vision inspection on production lines requiring GPU inference at >60fps",
        qos_priority = QoSPriority.HIGH_THROUGHPUT,
        tier_pref    = TierPreference.FOG,
        criticality_level = 3,
        real_time    = True,
        offload_eligible  = False,
        requirements = RequirementRange(
            cpu_cores_min=2,      cpu_cores_nom=4,       cpu_cores_max=8,
            cpu_freq_mhz_min=2400, cpu_freq_mhz_nom=3400, cpu_freq_mhz_max=4800,
            gpu_memory_mb_min=1024, gpu_memory_mb_nom=4096, gpu_memory_mb_max=8192,
            ram_mb_min=2048,      ram_mb_nom=8192,       ram_mb_max=16384,
            storage_mb_min=2048,  storage_mb_nom=10240,  storage_mb_max=20480,
            bandwidth_kbps_min=2048,  bandwidth_kbps_nom=10240, bandwidth_kbps_max=20480,
            latency_ms_max=33,    latency_ms_target=16,
            packet_loss_max_pct=0.05,
            llm_tokens_max=256,   llm_context_window=2048,
        ),
        intent_sentences=[
            "PCB surface mount inspection line running at 1200 boards/hour, solder joint defect rate above 0.3%.",
            "Automotive body panel paint defect camera system detecting orange peel at 95cm/s line speed.",
            "Pharmaceutical blister pack seal integrity check, 98 cavities per pack, leak detection needed.",
            "Steel coil surface inspection for rolling marks, cracks and scale at 400 metres per minute.",
            "Food packaging label alignment verification, ±1.2mm tolerance, reject rate above acceptable threshold.",
            "Semiconductor wafer edge inspection, 300mm wafer, sub-micron defect detection required at 150 wafers/hour.",
            "Ceramic tile dimensional measurement and colour uniformity grading at 40 tiles per minute.",
            "Bottling line fill level and cap torque verification, 36,000 bottles per hour, zero false pass allowed.",
            "Weld bead geometry inspection on exhaust manifolds, 3D structured light scan at 120 parts per hour.",
            "Solar panel electroluminescence imaging for micro-crack detection, 500 panels per shift throughput.",
        ],
    ),

    # ── IND-3: Autonomous Process Control (Time-critical, safety-critical) ────
    "process_control": ApplicationSpec(
        key          = "process_control",
        domain       = Domain.INDUSTRIAL,
        subtype      = "Autonomous process control",
        description  = "Closed-loop PID and AI-augmented control for chemical, energy and safety-critical processes",
        qos_priority = QoSPriority.ULTRA_LOW_LATENCY,
        tier_pref    = TierPreference.EDGE,
        criticality_level = 5,
        real_time    = True,
        offload_eligible  = False,
        requirements = RequirementRange(
            cpu_cores_min=1,      cpu_cores_nom=2,       cpu_cores_max=4,
            cpu_freq_mhz_min=2000, cpu_freq_mhz_nom=3000, cpu_freq_mhz_max=4200,
            gpu_memory_mb_min=0,  gpu_memory_mb_nom=0,   gpu_memory_mb_max=0,
            ram_mb_min=512,       ram_mb_nom=2048,        ram_mb_max=8192,
            storage_mb_min=512,   storage_mb_nom=2048,    storage_mb_max=8192,
            bandwidth_kbps_min=512,   bandwidth_kbps_nom=2048,  bandwidth_kbps_max=5120,
            latency_ms_max=8,     latency_ms_target=4,
            packet_loss_max_pct=0.001,
            llm_tokens_max=128,   llm_context_window=512,
        ),
        intent_sentences=[
            "Reactor vessel exothermic runaway detected, coolant flow must increase 40% within 5 seconds.",
            "Gas turbine blade temperature 1180°C and rising, fuel-air ratio adjustment required immediately.",
            "Distillation column reflux ratio drifted 12% off setpoint, product purity falling below spec.",
            "High-pressure steam header at 92 bar, PRV cycling abnormally, de-superheater spray valve fault.",
            "Wind turbine pitch control losing synchronisation in gusts above 18m/s, yaw error accumulating.",
            "Electrolyser stack cell voltage imbalance detected, hydrogen purity dropping, shutdown sequence standby.",
            "Nuclear cooling loop pump cavitation noise at 3.2 kHz, backup pump start authorisation needed.",
            "Autonomous vehicle assembly line robot collision risk detected, safety fence breach, E-stop required.",
            "Ammonia synthesis loop converter temperature 512°C, catalyst bed hot spot forming, flow redistribution.",
            "Offshore platform wellhead pressure spike 340 bar, subsea tree control system response needed now.",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 3: GENERAL (cross-domain, fits any edge-cloud orchestration paper)
    # ══════════════════════════════════════════════════════════════════════════

    # ── GEN-1: Smart Environment Monitoring (Best-effort, low resource) ───────
    "smart_environment": ApplicationSpec(
        key          = "smart_environment",
        domain       = Domain.GENERAL,
        subtype      = "Smart environment monitoring",
        description  = "Multi-sensor ambient monitoring for smart buildings, campuses and cities",
        qos_priority = QoSPriority.BEST_EFFORT,
        tier_pref    = TierPreference.FOG,
        criticality_level = 1,
        real_time    = False,
        offload_eligible  = True,
        requirements = RequirementRange(
            cpu_cores_min=1,      cpu_cores_nom=1,       cpu_cores_max=2,
            cpu_freq_mhz_min=400, cpu_freq_mhz_nom=800,  cpu_freq_mhz_max=1600,
            gpu_memory_mb_min=0,  gpu_memory_mb_nom=0,   gpu_memory_mb_max=0,
            ram_mb_min=64,        ram_mb_nom=128,         ram_mb_max=512,
            storage_mb_min=100,   storage_mb_nom=500,    storage_mb_max=2000,
            bandwidth_kbps_min=16,   bandwidth_kbps_nom=64,   bandwidth_kbps_max=256,
            latency_ms_max=5000,  latency_ms_target=1000,
            packet_loss_max_pct=5.0,
            llm_tokens_max=256,   llm_context_window=1024,
        ),
        intent_sentences=[
            "Meeting room CO2 concentration 1800ppm, occupancy sensors show 12 people, ventilation adjustment needed.",
            "Campus smart grid solar panel output dropped 35% this afternoon despite clear skies, inverter check.",
            "Building lobby PM2.5 particulate reading 68 μg/m³, air quality index entering unhealthy range.",
            "Parking lot occupancy sensor data: 94% capacity on level 2, redirect signage update required.",
            "Smart irrigation system: soil moisture 18% in zone 4, weather forecast shows rain in 6 hours.",
            "Street lighting energy consumption 23% above monthly baseline, dimming schedule optimisation requested.",
            "Cold storage warehouse humidity at 87% for past 4 hours, setpoint is 65%, compressor alarm active.",
            "Data centre hot aisle temperature 38°C, CRAC unit performance degraded, capacity planning review.",
            "Noise pollution sensor near school: average 72 dB during lesson time, traffic routing analysis needed.",
            "Smart electricity meter cluster reporting demand spike 340kW above forecast, grid balancing needed.",
        ],
    ),

    # ── GEN-2: Federated Learning Coordination (Resource-intensive, periodic) ──
    "federated_learning": ApplicationSpec(
        key          = "federated_learning",
        domain       = Domain.GENERAL,
        subtype      = "Federated learning coordination",
        description  = "Distributed model training across edge nodes with privacy-preserving gradient aggregation",
        qos_priority = QoSPriority.HIGH_THROUGHPUT,
        tier_pref    = TierPreference.ADAPTIVE,
        criticality_level = 2,
        real_time    = False,
        offload_eligible  = True,
        requirements = RequirementRange(
            cpu_cores_min=2,      cpu_cores_nom=4,       cpu_cores_max=8,
            cpu_freq_mhz_min=1200, cpu_freq_mhz_nom=2000, cpu_freq_mhz_max=3200,
            gpu_memory_mb_min=512, gpu_memory_mb_nom=2048, gpu_memory_mb_max=8192,
            ram_mb_min=1024,      ram_mb_nom=4096,        ram_mb_max=16384,
            storage_mb_min=2048,  storage_mb_nom=10240,   storage_mb_max=51200,
            bandwidth_kbps_min=512,  bandwidth_kbps_nom=2048,  bandwidth_kbps_max=10240,
            latency_ms_max=30000, latency_ms_target=10000,
            packet_loss_max_pct=1.0,
            llm_tokens_max=512,   llm_context_window=4096,
        ),
        intent_sentences=[
            "24 edge nodes ready, initiating federated round 47, global model accuracy stalled at 82.3%.",
            "Differential privacy budget exhausted on 3 hospital nodes, gradient aggregation must exclude them.",
            "Model convergence slow after 200 rounds, learning rate scheduler and client selection review needed.",
            "New edge device joined federation, local dataset 15,000 samples, model version compatibility check.",
            "Federated round failed: 8 of 30 nodes disconnected mid-aggregation, Byzantine fault tolerance needed.",
            "Communication bottleneck on fog aggregator, model gradient size 420MB per round, compression needed.",
            "Non-IID data distribution across 16 manufacturing sites causing 6% accuracy drop versus centralised.",
            "Secure aggregation protocol overhead 34% of total round time, homomorphic encryption cost review.",
            "Concept drift detected in 5 retail nodes, local model performance dropped 11% in last 48 hours.",
            "Stragglers: 4 nodes took over 8x median time, asynchronous aggregation policy update required.",
        ],
    ),

    # ── GEN-3: Cognitive Digital Twin (Balanced, simulation + real-time sync) ──
    "digital_twin": ApplicationSpec(
        key          = "digital_twin",
        domain       = Domain.GENERAL,
        subtype      = "Cognitive digital twin",
        description  = "LLM-powered digital twin synchronisation and what-if scenario reasoning across domains",
        qos_priority = QoSPriority.BALANCED,
        tier_pref    = TierPreference.ADAPTIVE,
        criticality_level = 2,
        real_time    = False,
        offload_eligible  = True,
        requirements = RequirementRange(
            cpu_cores_min=2,      cpu_cores_nom=4,       cpu_cores_max=12,
            cpu_freq_mhz_min=1600, cpu_freq_mhz_nom=2400, cpu_freq_mhz_max=4000,
            gpu_memory_mb_min=1024, gpu_memory_mb_nom=4096, gpu_memory_mb_max=16384,
            ram_mb_min=2048,      ram_mb_nom=8192,        ram_mb_max=32768,
            storage_mb_min=5120,  storage_mb_nom=20480,   storage_mb_max=102400,
            bandwidth_kbps_min=1024,  bandwidth_kbps_nom=4096,  bandwidth_kbps_max=20480,
            latency_ms_max=500,   latency_ms_target=200,
            packet_loss_max_pct=0.5,
            llm_tokens_max=1024,  llm_context_window=8192,
        ),
        intent_sentences=[
            "Hospital building twin shows HVAC failure in wing C will affect 3 operating theatres in 40 minutes.",
            "Factory digital twin scenario: what happens to throughput if conveyor 3 goes down for 2 hours?",
            "Smart city twin: predict traffic flow impact if the northern bridge closes for maintenance next Monday.",
            "Wind farm twin real-time state diverged from physical by 4.2%, recalibration and root-cause needed.",
            "Power grid twin simulating N-1 contingency: loss of substation 7 affects 14,000 customers, mitigation?",
            "Supply chain digital twin flagging: component shortage will halt assembly line in 72 hours.",
            "Port logistics twin: berth allocation optimisation needed, 3 vessels arriving simultaneously tomorrow.",
            "Campus energy twin: heating demand forecast for next 24h based on weather, occupancy and schedule.",
            "Wastewater treatment twin: influent flow 40% above design, model overflow risk and diversion options.",
            "Aircraft maintenance twin: engine serial 7842 cycle count approaching limit, dock scheduling required.",
        ],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# REQUIREMENT SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def _sample(min_val: float, nom_val: float, max_val: float, rng: random.Random) -> float:
    """Sample a value from the requirement range according to REQUIREMENT_MODE."""
    if REQUIREMENT_MODE == "nominal":
        return nom_val
    elif REQUIREMENT_MODE == "worst":
        return max_val
    else:  # "random"
        return round(rng.uniform(min_val, max_val), 2)


def _sample_int(min_val: int, nom_val: int, max_val: int, rng: random.Random) -> int:
    if REQUIREMENT_MODE == "nominal":
        return nom_val
    elif REQUIREMENT_MODE == "worst":
        return max_val
    else:
        return rng.randint(min_val, max_val)


# ─────────────────────────────────────────────────────────────────────────────
# INSTANCE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_instance(spec: ApplicationSpec, rng: random.Random) -> ApplicationInstance:
    """
    Generate one ApplicationInstance from a spec by sampling its requirements
    and selecting one intent sentence at random.
    """
    r = spec.requirements
    import importlib, sys
    _apps_mod = sys.modules.get("applications") or sys.modules.get("__main__")
    lt_min = getattr(_apps_mod, "APP_LIFETIME_MIN_S", 2.0)
    lt_max = getattr(_apps_mod, "APP_LIFETIME_MAX_S", 8.0)
    lifetime = round(rng.uniform(lt_min, lt_max), 2)

    return ApplicationInstance(
        instance_id       = str(uuid.uuid4())[:8],
        spec_key          = spec.key,
        domain            = spec.domain.value,
        subtype           = spec.subtype,
        timestamp         = time.time(),
        intent_text       = rng.choice(spec.intent_sentences),
        app_lifetime_s    = lifetime,

        cpu_cores         = _sample(r.cpu_cores_min,      r.cpu_cores_nom,      r.cpu_cores_max,      rng),
        cpu_freq_mhz      = _sample_int(r.cpu_freq_mhz_min,  r.cpu_freq_mhz_nom,  r.cpu_freq_mhz_max,  rng),
        gpu_memory_mb     = _sample_int(r.gpu_memory_mb_min, r.gpu_memory_mb_nom, r.gpu_memory_mb_max, rng),
        ram_mb            = _sample_int(r.ram_mb_min,        r.ram_mb_nom,        r.ram_mb_max,        rng),
        storage_mb        = _sample_int(r.storage_mb_min,    r.storage_mb_nom,    r.storage_mb_max,    rng),
        bandwidth_kbps    = _sample_int(r.bandwidth_kbps_min, r.bandwidth_kbps_nom, r.bandwidth_kbps_max, rng),
        latency_ms_max    = r.latency_ms_max,
        latency_ms_target = r.latency_ms_target,
        packet_loss_max   = r.packet_loss_max_pct,
        llm_tokens_max    = r.llm_tokens_max,

        qos_priority      = spec.qos_priority.value,
        tier_preference   = spec.tier_pref.value,
        criticality_level = spec.criticality_level,
        real_time         = spec.real_time,
        offload_eligible  = spec.offload_eligible,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_experiment(
    num_apps: int               = NUM_APPS,
    domains:  list[str]         = APP_DOMAINS,
    subtypes: dict[str, list | None] = APP_SUBTYPES,
    seed:     int | None        = RANDOMISE_SEED,
) -> list[ApplicationInstance]:
    """
    Generate a list of ApplicationInstance objects for one experiment run.

    Args:
        num_apps : total instances to generate
        domains  : which domains to include
        subtypes : per-domain subtype filter (None = all)
        seed     : RNG seed for reproducibility

    Returns:
        List of ApplicationInstance objects, shuffled.
    """
    rng = random.Random(seed)

    # Build eligible spec pool
    eligible_specs: list[ApplicationSpec] = []
    for domain_key in domains:
        allowed_subtypes = subtypes.get(domain_key, None)
        for spec_key, spec in APPLICATION_CATALOGUE.items():
            if spec.domain.value != domain_key:
                continue
            if allowed_subtypes is not None and spec_key not in allowed_subtypes:
                continue
            eligible_specs.append(spec)

    if not eligible_specs:
        raise ValueError(
            f"No application specs matched the given domains/subtypes: "
            f"domains={domains}, subtypes={subtypes}"
        )

    instances: list[ApplicationInstance] = []
    for _ in range(num_apps):
        spec = rng.choice(eligible_specs)
        instances.append(generate_instance(spec, rng))

    rng.shuffle(instances)
    return instances


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER  (for quick CLI inspection)
# ─────────────────────────────────────────────────────────────────────────────

def print_experiment_summary(instances: list[ApplicationInstance]) -> None:
    from collections import Counter
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Total instances : {len(instances)}")
    print(f"  Requirement mode: {REQUIREMENT_MODE}")
    print(f"  RNG seed        : {RANDOMISE_SEED}")
    print()

    domain_counts  = Counter(i.domain   for i in instances)
    subtype_counts = Counter(i.subtype  for i in instances)
    tier_counts    = Counter(i.tier_preference for i in instances)
    qos_counts     = Counter(i.qos_priority    for i in instances)

    print("  Domain distribution:")
    for k, v in sorted(domain_counts.items()):
        bar = "█" * v
        print(f"    {k:<25} {bar} ({v})")

    print("\n  Subtype distribution:")
    for k, v in sorted(subtype_counts.items()):
        bar = "█" * v
        print(f"    {k:<35} {bar} ({v})")

    print("\n  Tier preference distribution:")
    for k, v in sorted(tier_counts.items()):
        print(f"    {k:<20} {v}")

    print("\n  QoS priority distribution:")
    for k, v in sorted(qos_counts.items()):
        print(f"    {k:<25} {v}")

    print("\n  Sample instance (first):")
    inst = instances[0]
    print(f"    ID           : {inst.instance_id}")
    print(f"    Domain       : {inst.domain}")
    print(f"    Subtype      : {inst.subtype}")
    print(f"    Intent       : \"{inst.intent_text[:80]}...\"")
    print(f"    CPU cores    : {inst.cpu_cores}")
    print(f"    GPU VRAM MB  : {inst.gpu_memory_mb}")
    print(f"    RAM MB       : {inst.ram_mb}")
    print(f"    Storage MB   : {inst.storage_mb}")
    print(f"    Bandwidth    : {inst.bandwidth_kbps} kbps")
    print(f"    Latency max  : {inst.latency_ms_max} ms")
    print(f"    Criticality  : {inst.criticality_level}/5")
    print(f"    Real-time    : {inst.real_time}")
    print(f"    Offloadable  : {inst.offload_eligible}")

    print("\n  LLM prompt preview (first instance):")
    print("  " + "-" * 66)
    for line in inst.llm_prompt().splitlines():
        print("  " + line)
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR INTEGRATION INTERFACE
# Called by OrchestratorAgent.py — do NOT rename this function.
# ─────────────────────────────────────────────────────────────────────────────

def run(config: dict, shared_state: dict) -> dict:
    """
    Integration entry point for OrchestratorAgent.py.

    Args:
        config : dict of overrides from OrchestratorAgent. Supported keys:
                   - "num_apps"          (int)
                   - "domains"           (list[str])
                   - "subtypes"          (dict)
                   - "seed"              (int | None)
                   - "requirement_mode"  (str)
                   - "output_dir"        (str)  path to save JSON results
        shared_state : mutable dict shared across all modules in the pipeline.
                       This function writes:
                         shared_state["applications"]  → list[ApplicationInstance]
                         shared_state["app_summary"]   → summary dict for logging

    Returns:
        result dict with keys:
            "instances"   : list[ApplicationInstance]
            "count"       : int
            "output_path" : str  (path of saved JSON file)
            "status"      : "ok" | "error"
            "error"       : str  (only present if status == "error")
    """
    import os

    try:
        # ── Apply config overrides (fall back to module-level defaults) ──────
        # global must be declared BEFORE any use of the variable in this scope
        global REQUIREMENT_MODE

        num_apps    = config.get("num_apps",          NUM_APPS)
        domains     = config.get("domains",           APP_DOMAINS)
        subtypes    = config.get("subtypes",          APP_SUBTYPES)
        seed        = config.get("seed",              RANDOMISE_SEED)
        output_dir  = config.get("output_dir",        ".")
        req_mode    = config.get("requirement_mode",  REQUIREMENT_MODE)

        # Allow OrchestratorAgent to override REQUIREMENT_MODE at runtime
        REQUIREMENT_MODE = req_mode

        # ── Generate instances ───────────────────────────────────────────────
        instances = generate_experiment(
            num_apps = num_apps,
            domains  = domains,
            subtypes = subtypes,
            seed     = seed,
        )

        # ── Write to shared pipeline state ───────────────────────────────────
        shared_state["applications"] = instances
        shared_state["app_summary"]  = {
            "total"   : len(instances),
            "domains" : list({i.domain  for i in instances}),
            "subtypes": list({i.subtype for i in instances}),
        }

        # ── Persist JSON for downstream agents and IEEE result logging ────────
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "experiment_applications.json")
        with open(output_path, "w") as f:
            json.dump([i.to_dict() for i in instances], f, indent=2)

        return {
            "instances"   : instances,
            "count"       : len(instances),
            "output_path" : output_path,
            "status"      : "ok",
        }

    except Exception as e:
        return {
            "instances"   : [],
            "count"       : 0,
            "output_path" : "",
            "status"      : "error",
            "error"       : str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT  (run this file directly in PyCharm to test it alone)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[applications.py] Running in standalone mode ...\n")

    # Simulate what OrchestratorAgent will pass
    _config = {
        "num_apps"         : NUM_APPS,
        "domains"          : APP_DOMAINS,
        "subtypes"         : APP_SUBTYPES,
        "seed"             : RANDOMISE_SEED,
        "requirement_mode" : REQUIREMENT_MODE,
        "output_dir"       : ".",
    }
    _shared_state: dict = {}

    result = run(_config, _shared_state)

    if result["status"] == "ok":
        print_experiment_summary(result["instances"])
        print(f"  Saved {result['count']} instances → {result['output_path']}\n")
    else:
        print(f"  [ERROR] applications.py failed: {result['error']}\n")
