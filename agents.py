"""
=============================================================================
agents.py
=============================================================================
Intent-Aware Multi-Agent AI Orchestration in Edge-Cloud Continuum
-----------------------------------------------------------------------------
PURPOSE:
    Defines six independent AI agents, each implementing a named, published
    ML/AI algorithm. Outputs are aggregated into AgentDecision consumed by
    OrchestratorAgent.py.

HOW FILES CONNECT (mechanics):
    OrchestratorAgent.py calls each module in sequence via run(config, shared_state):
        1. applications.run()        → shared_state["applications"]
        2. computing_resources.run() → shared_state["infrastructure"]
        3. agents.run()              → reads both above, writes decisions
        4. baselines.run()           → runs 4 baselines on same apps/infra
        5. figures.run()             → generates IEEE comparison graphs

    When agents.py is run STANDALONE (python agents.py), the __main__ block
    calls applications.run() and computing_resources.run() itself first —
    exactly what OrchestratorAgent will do. Same code path, no difference.
    Applications feeding the agents always come from applications.py.

AGENT → ML/AI ALGORITHM MAPPING:
    Agent 1  IntentParserAgent       TF-IDF cosine similarity + LLM (Llama/Ollama)
    Agent 2  QoSProfilerAgent        Fuzzy Logic (trapezoidal membership functions)
                                     + Weighted Multi-Criteria Decision Analysis
    Agent 3  ClassificationAgent     Multinomial Naive Bayes (Laplace-smoothed)
                                     + LLM refinement on low-confidence cases
    Agent 4  ResourceMonitorAgent    Exponential Moving Average (EMA) smoothing
                                     + SLA threshold watchdog
    Agent 5  PlacementAgent          TOPSIS — Technique for Order of Preference
                                     by Similarity to Ideal Solution (MADM)
    Agent 6  OffloadAgent            EMA-triggered threshold control policy
                                     + Least-Loaded (LL) routing heuristic

IEEE BASELINES (in baselines.py):
    B1  Cloud-Only          — all apps sent to cloud regardless of urgency
    B2  Round-Robin         — distribute apps cyclically across all nodes
    B3  Static Threshold    — fixed CPU threshold, no ML, no intent-awareness
    B4  Greedy First-Fit    — place on first node that has enough resources

CONFIGURATION:
    LLM_BACKEND   "mock" | "ollama" | "llama_cpp"
    AGENT_VERBOSE True = print per-agent reasoning to console
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  ←  EDIT HERE BEFORE EACH RUN
# ─────────────────────────────────────────────────────────────────────────────

LLM_BACKEND:     str   = "mock"       # "mock" | "ollama" | "llama_cpp"
LLM_MODEL:       str   = "llama3.2:8b-instruct-q4_K_M"
LLM_HOST:        str   = "http://localhost:11434"
AGENT_VERBOSE:   bool  = True
OFFLOAD_CPU_THR: float = 80.0         # EMA CPU % that triggers offload
OFFLOAD_RAM_THR: float = 85.0         # EMA RAM % that triggers offload

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import json, math, re, time, random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class DeploymentStatus(str, Enum):
    PENDING   = "pending"
    PLACED    = "placed"
    OFFLOADED = "offloaded"
    REJECTED  = "rejected"
    FAILED    = "failed"

class UrgencyLevel(str, Enum):
    CRITICAL   = "critical"
    HIGH       = "high"
    MEDIUM     = "medium"
    LOW        = "low"
    BACKGROUND = "background"


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT DATACLASSES — one per agent, all bundled into AgentDecision
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentProfile:
    """Output of Agent 1 — IntentParserAgent."""
    detected_domain:      str   = ""
    detected_subtype:     str   = ""
    key_entities:         list  = field(default_factory=list)
    action_verbs:         list  = field(default_factory=list)
    numeric_constraints:  dict  = field(default_factory=dict)
    urgency_signals:      list  = field(default_factory=list)
    # TF-IDF specific outputs
    tfidf_best_class:     str   = ""
    tfidf_cosine_sim:     float = 0.0
    tfidf_top_terms:      list  = field(default_factory=list)
    # LLM specific outputs
    llm_confidence:       float = 0.0
    llm_reasoning:        str   = ""
    parse_time_ms:        float = 0.0


@dataclass
class QoSProfile:
    """Output of Agent 2 — QoSProfilerAgent."""
    urgency_level:        str   = UrgencyLevel.MEDIUM.value
    latency_budget_ms:    float = 200.0
    latency_target_ms:    float = 100.0
    throughput_req_kbps:  int   = 256
    reliability_pct:      float = 99.0
    criticality_score:    float = 0.0
    # Fuzzy logic outputs
    fuzzy_memberships:    dict  = field(default_factory=dict)  # per urgency level
    fuzzy_raw_score:      float = 0.0
    # Weighted scoring outputs
    weight_domain:        float = 0.0
    weight_subtype:       float = 0.0
    weight_llm_conf:      float = 0.0
    weight_realtime:      float = 0.0
    real_time_required:   bool  = False
    gpu_required:         bool  = False
    offload_permitted:    bool  = True
    preferred_tier:       str   = "fog"
    fallback_tiers:       list  = field(default_factory=list)
    sla_violation_cost:   float = 1.0
    profiling_time_ms:    float = 0.0


@dataclass
class ClassificationResult:
    """Output of Agent 3 — ClassificationAgent."""
    primary_class:        str   = ""
    secondary_class:      str   = ""
    compute_class:        str   = ""
    taxonomy_path:        str   = ""
    # Naive Bayes outputs
    nb_posterior_probs:   dict  = field(default_factory=dict)
    nb_top_class:         str   = ""
    nb_confidence:        float = 0.0
    nb_confidence_gap:    float = 0.0    # gap between top-1 and top-2
    # Resource estimates
    estimated_cpu_cores:  float = 1.0
    estimated_ram_mb:     int   = 512
    estimated_gpu_mb:     int   = 0
    estimated_bw_kbps:    int   = 256
    model_size_class:     str   = ""     # "1b" | "3b" | "8b" | "70b"
    llm_refinement_used:  bool  = False
    classification_ms:    float = 0.0


@dataclass
class ResourceSnapshot:
    """Output of Agent 4 — ResourceMonitorAgent."""
    timestamp:            float = 0.0
    total_nodes:          int   = 0
    available_nodes:      dict  = field(default_factory=dict)   # tier → count
    overloaded_nodes:     list  = field(default_factory=list)
    offline_nodes:        list  = field(default_factory=list)
    # Raw utilisation
    tier_avg_cpu_pct:     dict  = field(default_factory=dict)
    tier_avg_ram_pct:     dict  = field(default_factory=dict)
    tier_free_gpu_mb:     dict  = field(default_factory=dict)
    # EMA-smoothed values (used by placement and offload agents)
    tier_ema_cpu:         dict  = field(default_factory=dict)
    tier_ema_ram:         dict  = field(default_factory=dict)
    sla_violations:       list  = field(default_factory=list)
    monitoring_ms:        float = 0.0


@dataclass
class PlacementDecision:
    """Output of Agent 5 — PlacementAgent."""
    instance_id:          str   = ""
    selected_node_id:     str   = ""
    selected_tier:        str   = ""
    selected_hw_class:    str   = ""
    # TOPSIS outputs
    topsis_cc:            float = 0.0   # closeness coefficient [0,1]
    topsis_ranking:       list  = field(default_factory=list)   # all candidates ranked
    placement_score:      float = 0.0   # alias = topsis_cc * 100
    # Fit metrics
    latency_estimate_ms:  float = 0.0
    cpu_fit_pct:          float = 0.0
    ram_fit_pct:          float = 0.0
    # Decision trail
    admission_status:     str   = DeploymentStatus.PENDING.value
    rejection_reason:     str   = ""
    alternatives_tried:   list  = field(default_factory=list)
    placement_ms:         float = 0.0


@dataclass
class OffloadDecision:
    """Output of Agent 6 — OffloadAgent."""
    triggered:            bool  = False
    trigger_reason:       str   = ""    # "ema_cpu" | "ema_ram"
    source_node_id:       str   = ""
    source_ema_cpu:       float = 0.0
    source_ema_ram:       float = 0.0
    target_node_id:       str   = ""
    target_tier:          str   = ""
    target_ema_cpu:       float = 0.0   # load at target before migration
    apps_offloaded:       list  = field(default_factory=list)
    offload_latency_ms:   float = 0.0
    success:              bool  = False


@dataclass
class AgentDecision:
    """
    Master aggregated result — all six agents bundled.
    This is what OrchestratorAgent.py reads to make the final deployment.
    """
    instance_id:          str
    intent_text:          str
    intent_profile:       IntentProfile       = field(default_factory=IntentProfile)
    qos_profile:          QoSProfile          = field(default_factory=QoSProfile)
    classification:       ClassificationResult = field(default_factory=ClassificationResult)
    resource_snapshot:    ResourceSnapshot    = field(default_factory=ResourceSnapshot)
    placement:            PlacementDecision   = field(default_factory=PlacementDecision)
    offload:              OffloadDecision     = field(default_factory=OffloadDecision)
    final_status:         str   = DeploymentStatus.PENDING.value
    total_pipeline_ms:    float = 0.0
    timestamp:            float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary_line(self) -> str:
        p = self.placement
        return (f"[{self.instance_id}] "
                f"{self.intent_profile.detected_domain:<12} "
                f"urgency={self.qos_profile.urgency_level:<10} "
                f"NB={self.classification.nb_confidence:.2f} "
                f"tier={p.selected_tier:<6} "
                f"node={p.selected_node_id:<14} "
                f"TOPSIS={p.topsis_cc:.4f} "
                f"status={self.final_status:<8} "
                f"{self.total_pipeline_ms:.1f}ms")


# ─────────────────────────────────────────────────────────────────────────────
#  LLM BACKEND  (mock / ollama / llama_cpp)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_llm(prompt: str) -> str:
    """
    Deterministic keyword-based mock LLM for offline/test runs.
    Covers all 9 subtypes across 3 domains with realistic keyword matching.
    """
    p = prompt.lower()
    # Healthcare — emergency
    if any(w in p for w in ["cardiac","arrest","resuscit","defibrillat","trauma",
                              "icu","septic","haemorrhage","anaphylax","neonate",
                              "stroke","vasopressor","intubat"]):
        d,s,t,u = "healthcare","emergency_care","edge","critical"
    # Healthcare — imaging
    elif any(w in p for w in ["ct scan","mri","dicom","mammograph","radiology",
                               "pathology","fundus","colonoscopy","ultrasound",
                               "biopsy","echocardiogram","retinal","dermoscopy"]):
        d,s,t,u = "healthcare","medical_imaging","cloud","medium"
    # Healthcare — remote monitoring
    elif any(w in p for w in ["glucose","smartwatch","spo2","copd","wearable",
                               "chronic","fall detect","sleep apnoea","parkinson",
                               "peak flow","blood pressure","heart failure"]):
        d,s,t,u = "healthcare","remote_monitoring","fog","medium"
    # Industrial — predictive maintenance
    elif any(w in p for w in ["vibration","bearing","gearbox","pump","spindle",
                               "compressor","servo","acoustic emission","impeller",
                               "lubrication","backlash","cnc"]):
        d,s,t,u = "industrial","predictive_maintenance","edge","high"
    # Industrial — quality control
    elif any(w in p for w in ["pcb","solder","paint defect","blister","weld bead",
                               "bottling","semiconductor wafer","solar panel",
                               "tile","label align","fill level"]):
        d,s,t,u = "industrial","quality_control","fog","high"
    # Industrial — process control
    elif any(w in p for w in ["reactor","turbine","coolant","pressure spike",
                               "e-stop","ammonia","offshore","electrolyser",
                               "distillation","fuel-air","wellhead"]):
        d,s,t,u = "industrial","process_control","edge","critical"
    # General — federated learning
    elif any(w in p for w in ["federated","gradient","aggregat","differential privacy",
                               "byzantine","stragglers","non-iid","convergence round"]):
        d,s,t,u = "general","federated_learning","cloud","low"
    # General — digital twin
    elif any(w in p for w in ["digital twin","twin","simulation scenario","what-if",
                               "berth","n-1 contingency","influent","diverged"]):
        d,s,t,u = "general","digital_twin","fog","medium"
    # General — smart environment
    else:
        d,s,t,u = "general","smart_environment","fog","low"

    return json.dumps({
        "domain": d, "subtype": s, "tier": t, "urgency": u,
        "confidence": round(random.uniform(0.78, 0.97), 3),
        "reasoning": (f"Keyword evidence in intent maps to {d}/{s}. "
                      f"Urgency={u} from temporal/safety signals."),
        "cpu_cores_refined": None,
        "ram_mb_refined": None,
        "gpu_mb_refined": None,
        "bandwidth_kbps_refined": None,
    })


def _llm_call(prompt: str, max_tokens: int = 256) -> str:
    if LLM_BACKEND == "mock":
        return _mock_llm(prompt)
    if LLM_BACKEND == "ollama":
        try:
            import urllib.request
            payload = json.dumps({
                "model": LLM_MODEL, "prompt": prompt, "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1},
            }).encode()
            req = urllib.request.Request(
                f"{LLM_HOST}/api/generate", data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read()).get("response", "")
        except Exception as e:
            if AGENT_VERBOSE:
                print(f"    [LLM] Ollama error: {e} — mock fallback")
            return _mock_llm(prompt)
    return _mock_llm(prompt)


def _parse_llm_json(raw: str) -> dict:
    raw = re.sub(r"```json|```", "", raw.strip()).strip()
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 1 — IntentParserAgent
#
#  ML ALGORITHM 1a: TF-IDF + Cosine Similarity
#  ─────────────────────────────────────────────
#  Term Frequency–Inverse Document Frequency (TF-IDF) transforms the raw
#  intent sentence into a weighted sparse vector. Terms that appear in many
#  classes (low IDF) are down-weighted; domain-specific terms (high IDF)
#  carry more discriminative weight.
#
#    TF(t,d)  = freq(t,d) / total_terms(d)
#    IDF(t)   = log((N+1) / (df(t)+1)) + 1     [Scikit-learn smoothing]
#    TF-IDF   = TF(t,d) × IDF(t)
#    cos(A,B) = (A · B) / (‖A‖ × ‖B‖)
#
#  A reference corpus of one canonical document per class is used to build
#  per-class TF-IDF centroids. The incoming intent is vectorised and cosine
#  distance to each centroid determines the best-matching class.
#
#  ML ALGORITHM 1b: LLM Semantic Analysis (Llama)
#  ────────────────────────────────────────────────
#  The TF-IDF result is passed as a structured prior to the LLM prompt.
#  The LLM performs: Named Entity Recognition, action verb extraction,
#  numeric constraint parsing (regex-assisted), and domain/subtype
#  classification with confidence scoring. Two-stage design reduces
#  hallucination by grounding the model with a hard prior.
#
#  If LLM confidence < 0.5, TF-IDF classification is used as fallback.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class IntentParserAgent:
    AGENT_NAME = "IntentParserAgent"

    # One canonical document per class — used to build TF-IDF centroids
    CORPUS: dict[str, str] = {
        "emergency_care":
            "cardiac arrest defibrillator shock trauma resuscitation icu septic "
            "haemorrhage anaphylaxis critical immediately emergency ventilator "
            "neonate stroke vasopressor intubation blood pressure falling",
        "medical_imaging":
            "mri ct scan dicom radiology pathology mammography ultrasound fundus "
            "colonoscopy segmentation biopsy echocardiogram retinal dermoscopy "
            "density bone fetal anatomy scan",
        "remote_monitoring":
            "glucose smartwatch spo2 wearable chronic copd blood pressure heart "
            "failure parkinson sleep apnoea peak flow fall detect weight gain "
            "compliance ketone ccp",
        "predictive_maintenance":
            "vibration bearing gearbox pump motor compressor servo acoustic cnc "
            "spindle hydraulic lubrication backlash impeller pressure discharge "
            "temperature rise current draw cooling fan",
        "quality_control":
            "pcb solder paint defect inspection weld bottling semiconductor wafer "
            "solar panel label blister vision camera tile fill level torque "
            "bead geometry electroluminescence",
        "process_control":
            "reactor turbine coolant pressure shutdown runaway ammonia offshore "
            "electrolyser e-stop valve distillation fuel air ratio coolant "
            "wellhead contingency cascade interlock",
        "federated_learning":
            "federated gradient aggregation differential privacy byzantine non-iid "
            "round convergence stragglers global model local dataset accuracy "
            "secure aggregation homomorphic encryption",
        "digital_twin":
            "digital twin simulation what-if scenario contingency forecast optimise "
            "berth influent diverged state N-1 power supply chain logistics "
            "wastewater aircraft maintenance",
        "smart_environment":
            "co2 occupancy pm2.5 parking irrigation lighting humidity noise demand "
            "meter campus grid temperature setpoint energy consumption hvac solar",
    }

    URGENCY_LEXICON: dict[str, float] = {
        "immediately": 1.0, "emergency": 1.0, "critical": 1.0, "arrest": 1.0,
        "runaway": 1.0, "now": 0.9, "instant": 0.9, "urgent": 0.9,
        "alert": 0.8, "shock": 0.9, "rupture": 0.9, "overload": 0.8,
        "falling": 0.7, "dropping": 0.7, "fault": 0.7, "anomaly": 0.6,
        "abnormal": 0.6, "required": 0.4, "needed": 0.3,
    }

    ACTION_VERBS: list[str] = [
        "classify","detect","predict","control","monitor","analyse","aggregate",
        "route","schedule","optimise","diagnose","alert","segment","measure",
        "track","identify","recommend","adjust","initiate","activate","reconstruct",
    ]

    # TF-IDF domain mapping (subtype → domain)
    DOMAIN_MAP: dict[str, str] = {
        "emergency_care":         "healthcare",
        "medical_imaging":        "healthcare",
        "remote_monitoring":      "healthcare",
        "predictive_maintenance": "industrial",
        "quality_control":        "industrial",
        "process_control":        "industrial",
        "federated_learning":     "general",
        "digital_twin":           "general",
        "smart_environment":      "general",
    }

    def __init__(self):
        self._vocab:  list  = []
        self._idf:    dict  = {}
        self._class_vectors: dict = {}
        self._build_tfidf_index()

    # ── Internal TF-IDF helpers ───────────────────────────────────────────────
    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def _tf(self, tokens: list[str]) -> dict[str, float]:
        counts: dict = defaultdict(float)
        for t in tokens:
            counts[t] += 1.0
        n = max(len(tokens), 1)
        return {t: c / n for t, c in counts.items()}

    def _build_tfidf_index(self):
        """Build per-class TF-IDF centroid vectors from CORPUS."""
        docs = {cls: self._tokenise(doc) for cls, doc in self.CORPUS.items()}
        N = len(docs)
        df: dict = defaultdict(int)
        for tokens in docs.values():
            for t in set(tokens):
                df[t] += 1
        self._vocab = sorted(df.keys())
        # Smoothed IDF (matches sklearn TfidfVectorizer default)
        self._idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0
                     for t in self._vocab}
        for cls, tokens in docs.items():
            tf = self._tf(tokens)
            vec = [tf.get(t, 0.0) * self._idf.get(t, 0.0) for t in self._vocab]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            self._class_vectors[cls] = [v / norm for v in vec]

    def _vectorise_query(self, text: str) -> list[str]:
        """Return L2-normalised TF-IDF vector for one query sentence."""
        tokens = self._tokenise(text)
        tf = self._tf(tokens)
        vec = [tf.get(t, 0.0) * self._idf.get(t, 0.0) for t in self._vocab]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _cosine(self, a: list, b: list) -> float:
        return round(sum(x * y for x, y in zip(a, b)), 6)

    def _top_tfidf_terms(self, text: str, n: int = 6) -> list[str]:
        """Return the top-n highest TF-IDF scoring terms in the query."""
        tokens = self._tokenise(text)
        tf = self._tf(tokens)
        scored = [(t, tf[t] * self._idf.get(t, 0.0))
                  for t in set(tokens) if t in self._idf]
        scored.sort(key=lambda x: -x[1])
        return [t for t, _ in scored[:n]]

    def _extract_numeric_constraints(self, text: str) -> dict:
        """Regex-based extraction of numeric values with units."""
        nc: dict = {}
        patterns = [
            (r"(\d+\.?\d*)\s*ms",    "latency_ms"),
            (r"(\d+\.?\d*)\s*MHz",   "freq_mhz"),
            (r"(\d+\.?\d*)\s*GB",    "storage_gb"),
            (r"(\d+\.?\d*)\s*MB",    "storage_mb"),
            (r"(\d+\.?\d*)\s*kbps",  "bandwidth_kbps"),
            (r"(\d+\.?\d*)\s*°C",    "temperature_c"),
            (r"(\d+\.?\d*)\s*bar",   "pressure_bar"),
            (r"(\d+\.?\d*)\s*Hz",    "frequency_hz"),
            (r"(\d+\.?\d*)\s*%",     "percentage"),
            (r"(\d+\.?\d*)\s*bpm",   "heart_rate_bpm"),
            (r"(\d+\.?\d*)\s*mg/dL", "glucose_mgdl"),
            (r"(\d+\.?\d*)\s*mm/s",  "vibration_mms"),
            (r"(\d+\.?\d*)\s*mmHg",  "blood_pressure_mmhg"),
            (r"(\d+\.?\d*)\s*kN",    "force_kn"),
        ]
        for pat, label in patterns:
            m = re.findall(pat, text, re.IGNORECASE)
            if m:
                nc[label] = [float(x) for x in m]
        return nc

    def run(self, app) -> IntentProfile:
        t0 = time.time()
        profile = IntentProfile()
        text = app.intent_text

        if AGENT_VERBOSE:
            print(f"\n  [{self.AGENT_NAME}]  ML: TF-IDF cosine similarity + LLM")
            print(f"    Intent: \"{text[:72]}...\"")

        # ── STEP 1: TF-IDF cosine similarity ─────────────────────────────────
        q_vec = self._vectorise_query(text)
        sims = {cls: self._cosine(q_vec, cv)
                for cls, cv in self._class_vectors.items()}
        tfidf_best = max(sims, key=sims.get)
        tfidf_sim  = sims[tfidf_best]
        top_terms  = self._top_tfidf_terms(text)

        if AGENT_VERBOSE:
            top3 = sorted(sims.items(), key=lambda x: -x[1])[:3]
            print(f"    TF-IDF top3: " +
                  " ".join(f"{c}={v:.3f}" for c, v in top3))
            print(f"    TF-IDF best={tfidf_best}  cosine={tfidf_sim:.4f}  "
                  f"top_terms={top_terms}")

        # ── STEP 2: Numeric + urgency pre-extraction ──────────────────────────
        nc         = self._extract_numeric_constraints(text)
        tl         = text.lower()
        urg_sigs   = [w for w in self.URGENCY_LEXICON if w in tl]
        action_vs  = [v for v in self.ACTION_VERBS if v in tl]
        entities   = list(set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)))[:8]

        # ── STEP 3: LLM semantic analysis (grounded by TF-IDF prior) ─────────
        prompt = (
            f"You are an IoT orchestration AI. Analyse this application intent.\n"
            f"INTENT: \"{text}\"\n\n"
            f"TF-IDF PRE-ANALYSIS:\n"
            f"  best_class={tfidf_best}  cosine={tfidf_sim:.3f}\n"
            f"  top_discriminative_terms={top_terms}\n"
            f"  numeric_constraints={json.dumps(nc)}\n"
            f"  urgency_signals={urg_sigs}\n\n"
            f"Respond ONLY in JSON:\n"
            f'{{"domain":"...","subtype":"...","tier":"...","urgency":"...",'
            f'"confidence":0.0,"reasoning":"one sentence"}}'
        )
        raw    = _llm_call(prompt, max_tokens=app.llm_tokens_max)
        parsed = _parse_llm_json(raw)

        # ── STEP 4: Blend TF-IDF and LLM ─────────────────────────────────────
        llm_conf = float(parsed.get("confidence", 0.0))
        if llm_conf < 0.5 or not parsed.get("domain"):
            # TF-IDF fallback
            parsed = {
                "domain":    self.DOMAIN_MAP.get(tfidf_best, app.domain),
                "subtype":   tfidf_best,
                "confidence": round(tfidf_sim * 0.90, 3),
                "reasoning": f"TF-IDF fallback: cosine={tfidf_sim:.3f} → {tfidf_best}",
            }

        profile.detected_domain     = parsed.get("domain",   app.domain)
        profile.detected_subtype    = parsed.get("subtype",  app.subtype)
        profile.key_entities        = entities
        profile.action_verbs        = action_vs
        profile.numeric_constraints = nc
        profile.urgency_signals     = urg_sigs
        profile.tfidf_best_class    = tfidf_best
        profile.tfidf_cosine_sim    = tfidf_sim
        profile.tfidf_top_terms     = top_terms
        profile.llm_confidence      = float(parsed.get("confidence", 0.75))
        profile.llm_reasoning       = parsed.get("reasoning", "")
        profile.parse_time_ms       = round((time.time() - t0) * 1000, 2)

        if AGENT_VERBOSE:
            print(f"    Final: domain={profile.detected_domain}  "
                  f"subtype={profile.detected_subtype}  "
                  f"llm_conf={profile.llm_confidence:.2f}  "
                  f"({profile.parse_time_ms:.1f}ms)")
        return profile


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 2 — QoSProfilerAgent
#
#  ML ALGORITHM 2a: Fuzzy Logic — Trapezoidal Membership Functions
#  ────────────────────────────────────────────────────────────────
#  Classical crisp thresholds (e.g. latency > 50ms → "high") lose
#  information at boundaries. Fuzzy logic models the inherent vagueness
#  of urgency categories using trapezoidal membership functions μ(x; a,b,c,d):
#
#    μ(x) = 0           if x ≤ a or x ≥ d
#         = (x-a)/(b-a) if a < x < b   (rising slope)
#         = 1           if b ≤ x ≤ c   (plateau)
#         = (d-x)/(d-c) if c < x < d   (falling slope)
#
#  Fuzzy sets defined for urgency ∈ [0,1]:
#    background : [0.00, 0.00, 0.15, 0.25]
#    low        : [0.10, 0.20, 0.35, 0.50]
#    medium     : [0.30, 0.45, 0.60, 0.75]
#    high       : [0.55, 0.70, 0.80, 0.92]
#    critical   : [0.75, 0.88, 1.00, 1.10]
#
#  Defuzzification: argmax of membership degrees (crisp label selection).
#
#  ML ALGORITHM 2b: Weighted Multi-Criteria Scoring
#  ─────────────────────────────────────────────────
#  Criticality score C ∈ [0,1]:
#    C = (w₁×domain_weight + w₂×subtype_weight +
#         w₃×llm_confidence + w₄×realtime_flag) × offload_penalty
#    w = [0.30, 0.35, 0.15, 0.20]   (empirically tuned)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class QoSProfilerAgent:
    AGENT_NAME = "QoSProfilerAgent"

    URGENCY_LATENCY: dict[str, tuple] = {
        "critical":   (10.0,   5.0),
        "high":       (50.0,  20.0),
        "medium":    (200.0, 100.0),
        "low":      (1000.0, 500.0),
        "background": (60000.0, 10000.0),
    }

    DOMAIN_WEIGHT: dict[str, float] = {
        "healthcare": 1.00, "industrial": 0.85, "general": 0.50
    }

    SUBTYPE_WEIGHT: dict[str, float] = {
        "emergency_care": 1.00, "process_control": 0.95,
        "predictive_maintenance": 0.75, "medical_imaging": 0.60,
        "remote_monitoring": 0.65, "quality_control": 0.70,
        "digital_twin": 0.50, "federated_learning": 0.35,
        "smart_environment": 0.25,
    }

    URGENCY_LEXICON: dict[str, float] = {
        "immediately": 1.0, "emergency": 1.0, "critical": 1.0, "arrest": 1.0,
        "runaway": 1.0, "now": 0.9, "instant": 0.9, "urgent": 0.9,
        "alert": 0.8, "shock": 0.9, "rupture": 0.9, "overload": 0.8,
        "falling": 0.7, "dropping": 0.7, "fault": 0.7, "anomaly": 0.6,
        "abnormal": 0.6, "required": 0.4, "needed": 0.3,
    }

    # Fuzzy set parameters (a, b, c, d) — trapezoidal
    FUZZY_SETS: dict[str, tuple] = {
        "background": (-0.10, 0.00, 0.15, 0.25),
        "low":         (0.10, 0.20, 0.35, 0.50),
        "medium":      (0.30, 0.45, 0.60, 0.75),
        "high":        (0.55, 0.70, 0.80, 0.92),
        "critical":    (0.75, 0.88, 1.00, 1.10),
    }

    @staticmethod
    def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function μ(x; a, b, c, d) ∈ [0, 1]."""
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        return (x - a) / (b - a) if x < b else (d - x) / (d - c)

    def _fuzzify(self, score: float) -> dict[str, float]:
        """Return membership degree in each urgency fuzzy set."""
        return {
            label: self._trapezoid(score, *params)
            for label, params in self.FUZZY_SETS.items()
        }

    def run(self, app, intent: IntentProfile) -> QoSProfile:
        t0 = time.time()
        p  = QoSProfile()

        if AGENT_VERBOSE:
            print(f"\n  [{self.AGENT_NAME}]  ML: Fuzzy Logic + Weighted Scoring")

        # ── Raw urgency score (max of matched signal weights) ─────────────────
        raw_score = max(
            (self.URGENCY_LEXICON.get(s, 0.3) for s in intent.urgency_signals),
            default=0.2
        )

        # ── Fuzzy inference ───────────────────────────────────────────────────
        memberships = self._fuzzify(raw_score)
        urgency     = max(memberships, key=memberships.get)

        if AGENT_VERBOSE:
            print(f"    Fuzzy raw_score={raw_score:.2f}  "
                  f"memberships={{ " +
                  " ".join(f"{k}:{v:.2f}" for k, v in memberships.items()) +
                  " }}")
            print(f"    Defuzzified → {urgency}")

        # Override with declared QoS priority if stronger
        if "ultra_low" in app.qos_priority:
            urgency = "critical"
        elif "low_latency" in app.qos_priority and urgency != "critical":
            urgency = "high"

        # ── Weighted multi-criteria criticality score ─────────────────────────
        w_domain  = self.DOMAIN_WEIGHT.get(intent.detected_domain, 0.5)
        w_subtype = self.SUBTYPE_WEIGHT.get(intent.detected_subtype, 0.5)
        w_llm     = intent.llm_confidence
        w_rt      = 1.0 if app.real_time else 0.6
        penalty   = 0.8 if not app.offload_eligible else 1.0

        criticality = round(
            (w_domain * 0.30 + w_subtype * 0.35 +
             w_llm * 0.15 + w_rt * 0.20) * penalty, 4
        )

        hard_dl, soft_dl = self.URGENCY_LATENCY[urgency]
        hard_dl = min(hard_dl, app.latency_ms_max)
        soft_dl = min(soft_dl, app.latency_ms_target)

        if urgency in ["critical", "high"]:
            preferred, fallbacks = "edge", ["fog", "cloud"]
        elif urgency == "medium":
            preferred, fallbacks = "fog",  ["edge", "cloud"]
        else:
            preferred, fallbacks = "cloud", ["fog", "edge"]

        p.urgency_level       = urgency
        p.latency_budget_ms   = hard_dl
        p.latency_target_ms   = soft_dl
        p.throughput_req_kbps = app.bandwidth_kbps
        p.reliability_pct     = 99.9 if criticality > 0.8 else 99.0
        p.criticality_score   = criticality
        p.fuzzy_memberships   = {k: round(v, 4) for k, v in memberships.items()}
        p.fuzzy_raw_score     = raw_score
        p.weight_domain       = w_domain
        p.weight_subtype      = w_subtype
        p.weight_llm_conf     = w_llm
        p.weight_realtime     = w_rt
        p.real_time_required  = app.real_time
        p.gpu_required        = app.gpu_memory_mb > 0
        p.offload_permitted   = app.offload_eligible
        p.preferred_tier      = preferred
        p.fallback_tiers      = fallbacks
        p.sla_violation_cost  = round(criticality * 10.0, 2)
        p.profiling_time_ms   = round((time.time() - t0) * 1000, 2)

        if AGENT_VERBOSE:
            print(f"    criticality={criticality:.4f}  "
                  f"budget={hard_dl}ms  "
                  f"tier={preferred}  ({p.profiling_time_ms:.1f}ms)")
        return p


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 3 — ClassificationAgent
#
#  ML ALGORITHM 3a: Multinomial Naive Bayes (Laplace-smoothed)
#  ─────────────────────────────────────────────────────────────
#  Bayes theorem with the naive independence assumption:
#    P(c | text) ∝ P(c) × ∏ P(w | c)
#
#  Training: per-class word frequency table from the same CORPUS as Agent 1.
#  Likelihood with Laplace smoothing (α=1, standard additive smoothing):
#    P(w | c) = (count(w,c) + α) / (total_words(c) + α × |V|)
#
#  Inference uses log-sum to avoid float underflow:
#    log P(c | text) = log P(c) + Σ log P(wᵢ | c)
#  Softmax-normalised posteriors provide a full probability distribution
#  over all 9 classes — nb_posterior_probs is logged for IEEE figures.
#
#  ML ALGORITHM 3b: LLM Refinement
#  ─────────────────────────────────
#  When NB confidence < 0.60 OR confidence_gap(top1-top2) < 0.05,
#  the LLM is called with the NB top-4 posterior probabilities as context.
#  It may override the classification and refine resource estimates
#  (CPU cores, RAM, GPU VRAM, bandwidth). Refinements are sanity-clamped.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationAgent:
    AGENT_NAME = "ClassificationAgent"

    # Same corpus as Agent 1 — trained independently (NB model, not TF-IDF)
    CORPUS: dict[str, str] = {
        "emergency_care":
            "cardiac arrest defibrillator shock trauma resuscitation icu septic "
            "haemorrhage anaphylaxis critical emergency ventilator neonate stroke "
            "vasopressor intubation blood pressure falling resuscitation protocol",
        "medical_imaging":
            "mri ct scan dicom radiology pathology mammography ultrasound fundus "
            "colonoscopy segmentation biopsy echocardiogram retinal dermoscopy "
            "density fetal anatomy scan pelvis spine",
        "remote_monitoring":
            "glucose smartwatch spo2 wearable chronic copd blood pressure heart "
            "failure parkinson sleep apnoea peak flow fall detect weight gain "
            "compliance ketone ccp continuous monitor",
        "predictive_maintenance":
            "vibration bearing gearbox pump motor compressor servo acoustic cnc "
            "spindle hydraulic lubrication backlash impeller pressure discharge "
            "temperature rise current draw cooling fan turbine",
        "quality_control":
            "pcb solder paint defect inspection weld bottling semiconductor wafer "
            "solar panel label blister vision camera tile fill level torque "
            "bead geometry electroluminescence crack",
        "process_control":
            "reactor turbine coolant pressure shutdown runaway ammonia offshore "
            "electrolyser e-stop valve distillation fuel air ratio coolant "
            "wellhead contingency cascade interlock safety",
        "federated_learning":
            "federated gradient aggregation differential privacy byzantine non-iid "
            "round convergence stragglers global model local dataset accuracy "
            "secure aggregation homomorphic encryption communication",
        "digital_twin":
            "digital twin simulation what-if scenario contingency forecast optimise "
            "berth influent diverged state N-1 supply chain logistics "
            "wastewater aircraft maintenance wind farm",
        "smart_environment":
            "co2 occupancy pm2.5 parking irrigation lighting humidity noise demand "
            "meter campus grid temperature setpoint energy consumption hvac solar",
    }

    TAXONOMY: dict = {
        "healthcare": {
            "time_critical":        ["emergency_care"],
            "resource_intensive":   ["medical_imaging"],
            "continuous_stream":    ["remote_monitoring"],
        },
        "industrial": {
            "time_critical":        ["process_control"],
            "resource_intensive":   ["quality_control"],
            "predictive":           ["predictive_maintenance"],
        },
        "general": {
            "distributed_learning": ["federated_learning"],
            "simulation":           ["digital_twin"],
            "monitoring":           ["smart_environment"],
        },
    }

    COMPUTE_CLASS: dict[str, str] = {
        "emergency_care":         "llm_heavy",
        "medical_imaging":        "gpu_heavy",
        "remote_monitoring":      "cpu_only",
        "predictive_maintenance": "cpu_only",
        "quality_control":        "gpu_heavy",
        "process_control":        "cpu_only",
        "federated_learning":     "gpu_light",
        "digital_twin":           "llm_heavy",
        "smart_environment":      "cpu_only",
    }

    MODEL_SIZE: dict[str, str] = {
        "cpu_only":   "1b",
        "gpu_light":  "3b",
        "llm_heavy":  "8b",
        "gpu_heavy":  "8b",
    }

    DOMAIN_MAP: dict[str, str] = {
        "emergency_care": "healthcare", "medical_imaging": "healthcare",
        "remote_monitoring": "healthcare", "predictive_maintenance": "industrial",
        "quality_control": "industrial", "process_control": "industrial",
        "federated_learning": "general", "digital_twin": "general",
        "smart_environment": "general",
    }

    def __init__(self):
        self._word_counts: dict = {}
        self._class_totals: dict = {}
        self._vocab: set = set()
        self._V: int = 0
        self._N: int = 0
        self._train_nb()

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def _train_nb(self):
        """Train Multinomial Naive Bayes on CORPUS."""
        all_tokens = {cls: self._tokenise(doc) for cls, doc in self.CORPUS.items()}
        for tokens in all_tokens.values():
            for t in tokens:
                self._vocab.add(t)
        self._V = len(self._vocab)
        self._N = len(self.CORPUS)
        for cls, tokens in all_tokens.items():
            counts: dict = defaultdict(int)
            for t in tokens:
                counts[t] += 1
            self._word_counts[cls]  = counts
            self._class_totals[cls] = sum(counts.values())

    def _log_likelihood(self, tokens: list[str], cls: str) -> float:
        """Log P(text | class) with Laplace smoothing (α=1)."""
        ll   = 0.0
        tot  = self._class_totals[cls]
        wc   = self._word_counts[cls]
        for t in tokens:
            if t in self._vocab:
                ll += math.log((wc.get(t, 0) + 1) / (tot + self._V))
        return ll

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return softmax-normalised P(class | text) for all classes."""
        tokens = self._tokenise(text)
        prior  = math.log(1.0 / self._N)
        log_p  = {cls: prior + self._log_likelihood(tokens, cls)
                  for cls in self.CORPUS}
        # Softmax normalisation (numerically stable)
        mx     = max(log_p.values())
        exp_p  = {cls: math.exp(lp - mx) for cls, lp in log_p.items()}
        total  = sum(exp_p.values())
        return {cls: round(v / total, 6) for cls, v in exp_p.items()}

    def _taxonomy_path(self, domain: str, subtype: str) -> str:
        for cat, subs in self.TAXONOMY.get(domain, {}).items():
            if subtype in subs:
                return f"{domain}/{cat}/{subtype}"
        return f"{domain}/unknown/{subtype}"

    def run(self, app, intent: IntentProfile, qos: QoSProfile) -> ClassificationResult:
        t0 = time.time()
        r  = ClassificationResult()

        if AGENT_VERBOSE:
            print(f"\n  [{self.AGENT_NAME}]  ML: Multinomial Naive Bayes + LLM refinement")

        # ── STEP 1: Naive Bayes posterior ─────────────────────────────────────
        posteriors   = self.predict_proba(app.intent_text)
        ranked       = sorted(posteriors.items(), key=lambda x: -x[1])
        nb_top, nb_conf = ranked[0]
        gap          = nb_conf - (ranked[1][1] if len(ranked) > 1 else 0.0)

        if AGENT_VERBOSE:
            print(f"    NB top3: " +
                  " ".join(f"{c}={p:.4f}" for c, p in ranked[:3]))
            print(f"    NB winner={nb_top}  conf={nb_conf:.4f}  "
                  f"gap={gap:.4f}")

        # ── STEP 2: LLM refinement when NB uncertain ──────────────────────────
        final_sub   = nb_top
        final_dom   = self.DOMAIN_MAP.get(nb_top, intent.detected_domain)
        llm_refined = False
        parsed      = {}

        if nb_conf < 0.60 or gap < 0.05:
            llm_refined = True
            prompt = (
                f"Classify this IoT application intent.\n"
                f"INTENT: \"{app.intent_text}\"\n\n"
                f"Naive Bayes top-4 posteriors:\n"
                f"{json.dumps({c: p for c, p in ranked[:4]}, indent=2)}\n\n"
                f"TF-IDF result: {intent.tfidf_best_class} "
                f"(cosine={intent.tfidf_cosine_sim:.3f})\n\n"
                f"Respond ONLY in JSON:\n"
                f'{{"domain":"...","subtype":"...",'
                f'"cpu_cores_refined":null,"ram_mb_refined":null,'
                f'"gpu_mb_refined":null,"bandwidth_kbps_refined":null,'
                f'"domain_confidence":0.0,"reasoning":"one sentence"}}'
            )
            parsed = _parse_llm_json(_llm_call(prompt, max_tokens=256))
            if parsed.get("subtype"):
                final_sub = parsed["subtype"]
                final_dom = parsed.get("domain", final_dom)
                if AGENT_VERBOSE:
                    print(f"    LLM override → {final_dom}/{final_sub}  "
                          f"conf={parsed.get('domain_confidence', '?')}")

        # ── STEP 3: Compute fingerprint ───────────────────────────────────────
        compute_cc = self.COMPUTE_CLASS.get(final_sub, "cpu_only")
        # Upgrade compute class if GPU declared
        if app.gpu_memory_mb > 8192 and compute_cc != "gpu_heavy":
            compute_cc = "gpu_heavy"
        # Downgrade for critical latency (GPU adds overhead)
        if qos.urgency_level == "critical" and compute_cc == "gpu_heavy":
            compute_cc = "gpu_light"

        # Refined resource estimates (LLM or declared)
        def _safe(val, declared, lo, hi, cast=float):
            if val is None:
                return cast(declared)
            return cast(min(max(cast(val), lo), hi))

        cpu_r = _safe(parsed.get("cpu_cores_refined"),   app.cpu_cores,      0.5, 128.0, float)
        ram_r = _safe(parsed.get("ram_mb_refined"),      app.ram_mb,         64,  65536, int)
        gpu_r = _safe(parsed.get("gpu_mb_refined"),      app.gpu_memory_mb,  0,   81920, int)
        bw_r  = _safe(parsed.get("bandwidth_kbps_refined"), app.bandwidth_kbps, 16, 102400, int)

        r.primary_class       = f"{final_dom}.{final_sub}"
        r.secondary_class     = f"{final_dom}.{nb_top}"
        r.compute_class       = compute_cc
        r.taxonomy_path       = self._taxonomy_path(final_dom, final_sub)
        r.nb_posterior_probs  = dict(ranked)
        r.nb_top_class        = nb_top
        r.nb_confidence       = nb_conf
        r.nb_confidence_gap   = gap
        r.estimated_cpu_cores = cpu_r
        r.estimated_ram_mb    = ram_r
        r.estimated_gpu_mb    = gpu_r
        r.estimated_bw_kbps   = bw_r
        r.model_size_class    = self.MODEL_SIZE.get(compute_cc, "3b")
        r.llm_refinement_used = llm_refined
        r.classification_ms   = round((time.time() - t0) * 1000, 2)

        if AGENT_VERBOSE:
            print(f"    → {r.primary_class}  compute={compute_cc}  "
                  f"model={r.model_size_class}B  "
                  f"LLM_refined={llm_refined}  ({r.classification_ms:.1f}ms)")
        return r


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 4 — ResourceMonitorAgent
#
#  ML ALGORITHM: Exponential Moving Average (EMA)
#  ─────────────────────────────────────────────────
#  Raw per-node CPU/RAM samples are noisy due to bursty IoT workloads.
#  EMA smooths the signal while weighting recent observations more heavily:
#
#    EMA_t = α × x_t + (1 − α) × EMA_{t−1}
#
#  Smoothing factor α = 0.3 (standard in cloud monitoring literature).
#  Small α → heavy smoothing, tracks long-term trends.
#  Large α → fast response, tracks transient spikes.
#
#  EMA state is maintained PERSISTENTLY across all agent pipeline calls
#  within one experiment run (singleton instance). This builds a rolling
#  picture of per-tier load that improves over time.
#
#  SLA WATCHDOG:
#  Nodes where EMA_CPU > OFFLOAD_CPU_THR OR latency > app deadline are
#  flagged in sla_violations and passed to the OffloadAgent.
#
#  IEEE note: EMA-based utilisation monitoring is standard in cloud elasticity
#  papers (e.g. Amazon CloudWatch, Kubernetes HPA). Using EMA here rather than
#  raw instantaneous CPU is a direct contribution claim.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class ResourceMonitorAgent:
    AGENT_NAME = "ResourceMonitorAgent"
    EMA_ALPHA  = 0.3     # smoothing factor α

    def __init__(self):
        # Persistent EMA state across all calls in one experiment run
        self._ema_cpu: dict[str, float] = {}
        self._ema_ram: dict[str, float] = {}
        self._call_count: int = 0

    def _ema_update(self, key: str, new_val: float, store: dict) -> float:
        """EMA_t = α × x_t + (1−α) × EMA_{t−1}"""
        prev    = store.get(key, new_val)
        updated = self.EMA_ALPHA * new_val + (1 - self.EMA_ALPHA) * prev
        store[key] = updated
        return round(updated, 4)

    def run(self, infrastructure) -> ResourceSnapshot:
        t0 = time.time()
        self._call_count += 1
        snap = ResourceSnapshot()
        snap.timestamp = time.time()

        if AGENT_VERBOSE:
            print(f"\n  [{self.AGENT_NAME}]  ML: EMA (α={self.EMA_ALPHA})  "
                  f"call #{self._call_count}")

        cpu_raw:  dict = {}
        ram_raw:  dict = {}
        gpu_free: dict = {}
        avail:    dict = {}
        overloaded: list = []
        offline:    list = []
        sla_viol:   list = []

        for tier in ["cloud", "fog", "edge", "local"]:
            nodes = infrastructure.nodes_for_tier(tier)
            if not nodes:
                continue
            cpu_list, ram_list = [], []
            free_gpu = 0
            avail_count = 0

            for node in nodes:
                node._log_snapshot(event=f"ema_monitor_call{self._call_count}")
                cpu_list.append(node.cpu_utilisation_pct)
                ram_list.append(node.ram_utilisation_pct)
                free_gpu += node.available_gpu_vram_mb

                if node.status == "offline":
                    offline.append(node.node_id)
                elif node.status == "overloaded":
                    overloaded.append(node.node_id)
                else:
                    avail_count += 1

                if node.latency_to_next_ms > 200:
                    sla_viol.append(node.node_id)

            rc = sum(cpu_list) / len(cpu_list)
            rr = sum(ram_list) / len(ram_list)
            ec = self._ema_update(tier, rc, self._ema_cpu)
            er = self._ema_update(tier, rr, self._ema_ram)

            cpu_raw[tier]  = round(rc, 3)
            ram_raw[tier]  = round(rr, 3)
            gpu_free[tier] = free_gpu
            avail[tier]    = avail_count

            if AGENT_VERBOSE:
                print(f"    [{tier:<6}] "
                      f"CPU_raw={rc:5.1f}%  EMA_CPU={ec:5.1f}%  "
                      f"RAM_raw={rr:5.1f}%  EMA_RAM={er:5.1f}%  "
                      f"GPU_free={free_gpu:,}MB  avail={avail_count}")

        snap.total_nodes      = len(infrastructure.all_nodes())
        snap.available_nodes  = avail
        snap.overloaded_nodes = overloaded
        snap.offline_nodes    = offline
        snap.tier_avg_cpu_pct = cpu_raw
        snap.tier_avg_ram_pct = ram_raw
        snap.tier_free_gpu_mb = gpu_free
        snap.tier_ema_cpu     = dict(self._ema_cpu)
        snap.tier_ema_ram     = dict(self._ema_ram)
        snap.sla_violations   = sla_viol
        snap.monitoring_ms    = round((time.time() - t0) * 1000, 2)

        if overloaded:
            print(f"    ⚠  Overloaded nodes: {overloaded}")
        if sla_viol:
            print(f"    ⚠  SLA latency violations: {sla_viol}")

        return snap


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 5 — PlacementAgent
#
#  ML ALGORITHM: TOPSIS
#  (Technique for Order of Preference by Similarity to Ideal Solution)
#  ────────────────────────────────────────────────────────────────────
#  TOPSIS is a canonical Multi-Attribute Decision Making (MADM) algorithm
#  used extensively in cloud/edge resource management research.
#  Published in: Hwang & Yoon, 1981. Widely adopted for IoT placement.
#
#  ALGORITHM (6 steps):
#    Step 1. Build decision matrix X [n_candidates × 5_criteria]
#    Step 2. Column-normalise: r_ij = x_ij / sqrt(Σᵢ x_ij²)
#    Step 3. Weight:           v_ij = w_j × r_ij
#    Step 4. Ideal     A⁺ = {max(v_ij)} for each criterion (all benefit)
#            Anti-ideal A⁻ = {min(v_ij)} for each criterion
#    Step 5. Euclidean distances:
#              D⁺ᵢ = ||vᵢ − A⁺||₂
#              D⁻ᵢ = ||vᵢ − A⁻||₂
#    Step 6. Closeness coefficient:
#              CCᵢ = D⁻ᵢ / (D⁺ᵢ + D⁻ᵢ)   ∈ [0, 1]
#            Higher CC = closer to ideal = better candidate.
#
#  CRITERIA (5 dimensions):
#    C1: Latency score     weight=0.30  (benefit — lower latency → higher score)
#    C2: CPU availability  weight=0.25  (benefit — more free CPU)
#    C3: RAM availability  weight=0.20  (benefit — more free RAM)
#    C4: GPU availability  weight=0.15  (benefit — GPU if needed, else bonus)
#    C5: Tier-urgency fit  weight=0.10  (benefit — tier matches urgency level)
#
#  Weights are published alongside results for IEEE reproducibility.
#  Sensitivity analysis on weights is part of the evaluation (baselines.py).
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class PlacementAgent:
    AGENT_NAME = "PlacementAgent"

    # TOPSIS criterion weights [latency, cpu_avail, ram_avail, gpu_avail, tier_fit]
    TOPSIS_WEIGHTS: list[float] = [0.30, 0.25, 0.20, 0.15, 0.10]

    TIER_LATENCY_SCORE: dict[str, float] = {
        "local": 100.0, "edge": 90.0, "fog": 60.0, "cloud": 30.0
    }

    URGENCY_TIER_FIT: dict[str, dict] = {
        "critical":   {"edge":100,"local":80,"fog":40,"cloud":10},
        "high":       {"edge": 90,"local":70,"fog":70,"cloud":30},
        "medium":     {"fog": 90,"edge":70,"cloud":70,"local":50},
        "low":        {"cloud":90,"fog":80,"edge":50,"local":30},
        "background": {"cloud":100,"fog":60,"edge":30,"local":10},
    }

    def _build_matrix(self, candidates: list, app,
                      qos: QoSProfile, cls: ClassificationResult) -> list[list[float]]:
        """
        Build raw decision matrix [n_candidates x 5_criteria].
        Uses ABSOLUTE values (cores, GB) not percentages so nodes are
        differentiated even when all are empty (avoids TOPSIS collapse).
        Node-level latency variation ensures unique rows per node.
        """
        M = []
        for node in candidates:
            # C1: Latency — base tier score adjusted by actual node latency
            base_lat = self.TIER_LATENCY_SCORE.get(node.tier, 50.0)
            lat = base_lat * (1.0 - min(node.latency_to_next_ms / 200.0, 0.5))
            if node.latency_to_next_ms > qos.latency_budget_ms:
                lat = 0.01

            # C2: Available CPU cores — ABSOLUTE (not %)
            cpu_a = max(node.available_cpu_cores, 0.01)

            # C3: Available RAM in GB — ABSOLUTE
            ram_a = max(node.available_ram_mb / 1024.0, 0.01)

            # C4: GPU score
            if qos.gpu_required and node.total_gpu_vram_mb > 0:
                gpu_a = max(node.available_gpu_vram_mb / 1024.0, 0.01)
            elif qos.gpu_required and node.total_gpu_vram_mb == 0:
                gpu_a = 0.01
            else:
                gpu_a = 50.0 if node.total_gpu_vram_mb == 0 else 30.0

            # C5: Tier-urgency fit, penalised by current load
            tier_fit = self.URGENCY_TIER_FIT.get(
                qos.urgency_level, {}).get(node.tier, 50.0)
            tier_fit = tier_fit * (1.0 - node.cpu_utilisation_pct / 100.0 * 0.3)

            M.append([lat, cpu_a, ram_a, gpu_a, tier_fit])
        return M

    def _topsis(self, M: list[list[float]]) -> list[float]:
        """
        Run TOPSIS on decision matrix M.
        Returns list of closeness coefficients CCᵢ ∈ [0,1].
        """
        if not M:
            return []
        n, m = len(M), len(M[0])

        # Step 2: Column-normalise
        col_norms = [
            math.sqrt(sum(M[i][j] ** 2 for i in range(n))) or 1e-9
            for j in range(m)
        ]
        R = [[M[i][j] / col_norms[j] for j in range(m)] for i in range(n)]

        # Step 3: Weighted normalised matrix
        V = [[R[i][j] * self.TOPSIS_WEIGHTS[j] for j in range(m)] for i in range(n)]

        # Step 4: Ideal A⁺ and Anti-ideal A⁻ (all criteria are benefit)
        A_pos = [max(V[i][j] for i in range(n)) for j in range(m)]
        A_neg = [min(V[i][j] for i in range(n)) for j in range(m)]

        # Step 5: Euclidean distances
        D_pos = [math.sqrt(sum((V[i][j] - A_pos[j]) ** 2 for j in range(m)))
                 for i in range(n)]
        D_neg = [math.sqrt(sum((V[i][j] - A_neg[j]) ** 2 for j in range(m)))
                 for i in range(n)]

        # Step 6: Closeness coefficient
        CC = [D_neg[i] / (D_pos[i] + D_neg[i] + 1e-9) for i in range(n)]
        return CC

    def run(self, app, qos: QoSProfile, cls: ClassificationResult,
            snap: ResourceSnapshot, infrastructure) -> PlacementDecision:
        t0 = time.time()
        d  = PlacementDecision()
        d.instance_id = app.instance_id
        alternatives_tried: list = []
        all_rankings:       list = []
        best_node = None
        best_cc   = -1.0

        if AGENT_VERBOSE:
            print(f"\n  [{self.AGENT_NAME}]  ML: TOPSIS (5-criteria, weights={self.TOPSIS_WEIGHTS})")
            print(f"    preferred={qos.preferred_tier}  urgency={qos.urgency_level}  "
                  f"gpu_req={qos.gpu_required}")

        # Try preferred tier first, then fallbacks
        tiers = [qos.preferred_tier] + [
            t for t in qos.fallback_tiers if t != qos.preferred_tier
        ]

        for tier in tiers:
            candidates = [
                n for n in infrastructure.available_nodes(tier)
                if n.can_accommodate(app)[0]
            ]
            if not candidates:
                reasons = [(n.node_id, n.can_accommodate(app)[1])
                           for n in infrastructure.nodes_for_tier(tier)[:3]]
                alternatives_tried.append({
                    "tier": tier, "reason": "no_feasible_node", "checked": reasons
                })
                continue

            M  = self._build_matrix(candidates, app, qos, cls)
            CC = self._topsis(M)
            ranked = sorted(zip(candidates, CC), key=lambda x: -x[1])

            if AGENT_VERBOSE:
                for nd, cc in ranked[:3]:
                    print(f"    [{tier}] {nd.node_id}  CC={cc:.6f}  "
                          f"CPU_avail={nd.available_cpu_cores:.1f}  "
                          f"RAM_avail={nd.available_ram_mb}MB")

            all_rankings.extend([
                {"node_id": nd.node_id, "tier": tier, "cc": round(cc, 6)}
                for nd, cc in ranked
            ])

            if ranked and ranked[0][1] > best_cc:
                best_cc   = ranked[0][1]
                best_node = ranked[0][0]

            # Stop if a good-enough placement found in the preferred tier
            if best_node and best_node.tier == tier and best_cc > 0.30:
                break

        # ── Admission control ─────────────────────────────────────────────────
        if best_node is None or best_cc <= 0:
            d.admission_status  = DeploymentStatus.REJECTED.value
            d.rejection_reason  = "no_feasible_node_all_tiers"
            d.alternatives_tried = alternatives_tried
            d.topsis_ranking    = all_rankings
            d.placement_ms      = round((time.time() - t0) * 1000, 2)
            if AGENT_VERBOSE:
                print(f"    ✗  REJECTED — no feasible node found")
            return d

        # ── Allocate on winning node ──────────────────────────────────────────
        if not best_node.allocate(app):
            d.admission_status = DeploymentStatus.FAILED.value
            d.rejection_reason = "allocation_race_condition"
            d.placement_ms     = round((time.time() - t0) * 1000, 2)
            return d

        d.selected_node_id   = best_node.node_id
        d.selected_tier      = best_node.tier
        d.selected_hw_class  = best_node.hw_class
        d.topsis_cc          = round(best_cc, 6)
        d.placement_score    = round(best_cc * 100, 4)
        d.latency_estimate_ms = best_node.latency_to_next_ms
        d.cpu_fit_pct        = round(
            (cls.estimated_cpu_cores / best_node.total_cpu_cores) * 100, 2)
        d.ram_fit_pct        = round(
            (cls.estimated_ram_mb / best_node.total_ram_mb) * 100, 2)
        d.topsis_ranking     = all_rankings
        d.admission_status   = DeploymentStatus.PLACED.value
        d.alternatives_tried = alternatives_tried
        d.placement_ms       = round((time.time() - t0) * 1000, 2)

        if AGENT_VERBOSE:
            print(f"    ✓  PLACED → {best_node.node_id}  "
                  f"tier={best_node.tier}  "
                  f"CC={best_cc:.6f}  "
                  f"cpu_fit={d.cpu_fit_pct:.1f}%  "
                  f"lat={d.latency_estimate_ms}ms  "
                  f"({d.placement_ms:.1f}ms)")
        return d


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 6 — OffloadAgent
#
#  ML ALGORITHM 6a: EMA-triggered Threshold Control Policy
#  ─────────────────────────────────────────────────────────
#  Trigger condition uses EMA values from Agent 4 (not raw CPU):
#    TRIGGER  IF  EMA_CPU(tier) > OFFLOAD_CPU_THR
#             OR  EMA_RAM(tier) > OFFLOAD_RAM_THR
#
#  Using EMA rather than instantaneous CPU prevents thrashing — a
#  well-known problem in threshold-based cloud elasticity controllers
#  (oscillation around the threshold). EMA smoothing introduces
#  hysteresis naturally.
#
#  ML ALGORITHM 6b: Least-Loaded (LL) Routing
#  ────────────────────────────────────────────
#  Among candidate target nodes, select the one with minimum EMA_CPU.
#  LL is the optimal routing algorithm for M/M/k queuing systems under
#  Poisson arrival (Little's Law). It outperforms Round-Robin and Random
#  for bursty IoT workloads because it responds to actual load state.
#
#  ESCALATION PATH:  local → edge → fog → cloud → (peer same-tier fallback)
#  Up to 2 apps migrated per call to avoid over-migration.
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class OffloadAgent:
    AGENT_NAME = "OffloadAgent"

    TIER_ESCALATION: dict[str, str] = {
        "local": "edge", "edge": "fog", "fog": "cloud", "cloud": "cloud"
    }

    def run(self, snap: ResourceSnapshot, infrastructure,
            active_placements: dict) -> OffloadDecision:
        t0 = time.time()
        d  = OffloadDecision()

        if AGENT_VERBOSE:
            print(f"\n  [{self.AGENT_NAME}]  ML: EMA threshold + Least-Loaded routing")

        # ── Identify overloaded nodes that host at least one app ──────────────
        overloaded_with_apps: list = []
        for node in infrastructure.all_nodes():
            ema_cpu = snap.tier_ema_cpu.get(node.tier, 0.0)
            ema_ram = snap.tier_ema_ram.get(node.tier, 0.0)
            hosts_apps = any(
                placed_node.node_id == node.node_id
                for _, placed_node in active_placements.values()
            )
            if (ema_cpu > OFFLOAD_CPU_THR or ema_ram > OFFLOAD_RAM_THR) \
                    and hosts_apps:
                overloaded_with_apps.append(node)

        if not overloaded_with_apps or not active_placements:
            d.triggered  = False
            d.offload_latency_ms = round((time.time() - t0) * 1000, 2)
            if AGENT_VERBOSE:
                print(f"    No offload needed (all nodes within EMA threshold)")
            return d

        src     = overloaded_with_apps[0]
        src_cpu = snap.tier_ema_cpu.get(src.tier, 0.0)
        src_ram = snap.tier_ema_ram.get(src.tier, 0.0)

        if AGENT_VERBOSE:
            print(f"    Overload on {src.node_id}  "
                  f"EMA_CPU={src_cpu:.1f}%  EMA_RAM={src_ram:.1f}%")

        d.triggered       = True
        d.source_node_id  = src.node_id
        d.source_ema_cpu  = src_cpu
        d.source_ema_ram  = src_ram
        d.trigger_reason  = "ema_cpu" if src_cpu >= OFFLOAD_CPU_THR else "ema_ram"

        # Apps hosted on source node
        hosted = [
            (iid, app, node)
            for iid, (app, node) in active_placements.items()
            if node.node_id == src.node_id
        ]

        # ── Least-Loaded target selection ─────────────────────────────────────
        tgt_tier  = self.TIER_ESCALATION.get(src.tier, "cloud")
        candidates = infrastructure.available_nodes(tgt_tier)
        if not candidates:
            # Peer offload fallback (same tier, different node)
            candidates = [
                n for n in infrastructure.available_nodes(src.tier)
                if n.node_id != src.node_id
            ]
            tgt_tier = src.tier

        # Sort by EMA_CPU → least loaded first
        candidates.sort(key=lambda n: snap.tier_ema_cpu.get(n.tier, 0.0))
        tgt = candidates[0] if candidates else None

        if tgt is None:
            d.success = False
            d.offload_latency_ms = round((time.time() - t0) * 1000, 2)
            if AGENT_VERBOSE:
                print(f"    ✗  No target node available for offload")
            return d

        tgt_cpu = snap.tier_ema_cpu.get(tgt.tier, 0.0)
        if AGENT_VERBOSE:
            print(f"    Least-Loaded target: {tgt.node_id}  "
                  f"(tier={tgt_tier}  EMA_CPU={tgt_cpu:.1f}%)")

        # ── Migrate up to 2 apps ──────────────────────────────────────────────
        migrated: list = []
        for iid, app, _ in hosted[:2]:
            src.release(app)
            if tgt.allocate(app):
                active_placements[iid] = (app, tgt)
                migrated.append(iid)
                if AGENT_VERBOSE:
                    print(f"    ↑  Migrated {iid} → {tgt.node_id}")
            else:
                src.allocate(app)   # rollback if target refused

        d.target_node_id   = tgt.node_id
        d.target_tier      = tgt_tier
        d.target_ema_cpu   = tgt_cpu
        d.apps_offloaded   = migrated
        d.success          = len(migrated) > 0
        d.offload_latency_ms = round((time.time() - t0) * 1000, 2)
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLETON AGENT INSTANCES
#  Reused across all applications in one experiment run.
#  EMA state in ResourceMonitorAgent and NB model in ClassificationAgent
#  persist across calls — intentional for correct rolling behaviour.
# ─────────────────────────────────────────────────────────────────────────────

_agent_intent   = IntentParserAgent()
_agent_qos      = QoSProfilerAgent()
_agent_classify = ClassificationAgent()
_agent_monitor  = ResourceMonitorAgent()
_agent_place    = PlacementAgent()
_agent_offload  = OffloadAgent()


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE — run all six agents for one ApplicationInstance
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_pipeline(app, infrastructure,
                       active_placements: dict) -> AgentDecision:
    """
    Execute all six agents in sequence for one ApplicationInstance.
    active_placements is updated in-place by PlacementAgent and OffloadAgent.
    Returns AgentDecision with all six agents' outputs aggregated.
    """
    t_start  = time.time()
    decision = AgentDecision(instance_id=app.instance_id, intent_text=app.intent_text)

    # Agent 1 — TF-IDF cosine + LLM
    decision.intent_profile   = _agent_intent.run(app)

    # Agent 2 — Fuzzy Logic + Weighted scoring
    decision.qos_profile      = _agent_qos.run(app, decision.intent_profile)

    # Agent 3 — Naive Bayes + LLM refinement
    decision.classification   = _agent_classify.run(
        app, decision.intent_profile, decision.qos_profile)

    # Agent 4 — EMA monitoring
    decision.resource_snapshot = _agent_monitor.run(infrastructure)

    # Agent 5 — TOPSIS placement
    decision.placement        = _agent_place.run(
        app, decision.qos_profile, decision.classification,
        decision.resource_snapshot, infrastructure)

    # Agent 6 — EMA threshold + Least-Loaded offload
    decision.offload          = _agent_offload.run(
        decision.resource_snapshot, infrastructure, active_placements)

    # Register placement in active tracking dict
    if decision.placement.admission_status == DeploymentStatus.PLACED.value:
        placed_node = next(
            (n for n in infrastructure.all_nodes()
             if n.node_id == decision.placement.selected_node_id), None
        )
        if placed_node:
            active_placements[app.instance_id] = (app, placed_node)

    decision.final_status      = decision.placement.admission_status
    decision.total_pipeline_ms = round((time.time() - t_start) * 1000, 2)
    return decision


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_pipeline_summary(decisions: list) -> None:
    from collections import Counter
    print("\n" + "=" * 84)
    print("  MULTI-AGENT PIPELINE — RESULTS SUMMARY")
    print("=" * 84)

    placed    = [d for d in decisions if d.final_status == "placed"]
    rejected  = [d for d in decisions if d.final_status == "rejected"]
    offloaded = [d for d in decisions if d.offload.triggered]

    print(f"\n  Total apps        : {len(decisions)}")
    print(f"  Placed            : {len(placed)}")
    print(f"  Rejected          : {len(rejected)}")
    print(f"  Offload events    : {len(offloaded)}")

    if placed:
        avg_cc  = sum(d.placement.topsis_cc for d in placed) / len(placed)
        avg_ms  = sum(d.total_pipeline_ms for d in decisions) / len(decisions)
        avg_nb  = sum(d.classification.nb_confidence for d in decisions) / len(decisions)
        llm_ref = sum(1 for d in decisions if d.classification.llm_refinement_used)
        print(f"  Avg TOPSIS CC     : {avg_cc:.6f}")
        print(f"  Avg pipeline ms   : {avg_ms:.1f}")
        print(f"  Avg NB confidence : {avg_nb:.4f}")
        print(f"  LLM refinements   : {llm_ref}/{len(decisions)}")

    print("\n  Tier distribution (placed):")
    for k, v in sorted(Counter(d.placement.selected_tier
                               for d in placed if d.placement.selected_tier).items()):
        print(f"    {k:<8} {'█' * v} ({v})")

    print("\n  Urgency distribution:")
    for k, v in sorted(Counter(d.qos_profile.urgency_level
                               for d in decisions).items()):
        print(f"    {k:<12} {v}")

    print("\n  Domain distribution:")
    for k, v in sorted(Counter(d.intent_profile.detected_domain
                               for d in decisions).items()):
        print(f"    {k:<20} {v}")

    print(f"\n  Per-application decisions (Agent 1–6 pipeline):")
    print("  " + "─" * 80)
    for d in decisions:
        print("  " + d.summary_line())
    print("=" * 84 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR INTEGRATION INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def run(config: dict, shared_state: dict) -> dict:
    """
    Integration entry point for OrchestratorAgent.py.

    Reads from shared_state:
        shared_state["applications"]   → list[ApplicationInstance]  from applications.py
        shared_state["infrastructure"] → Infrastructure              from computing_resources.py

    Writes to shared_state:
        shared_state["agent_decisions"]    → list[AgentDecision]
        shared_state["active_placements"]  → dict {instance_id: (app, ComputeNode)}

    Returns:
        {"decisions": list, "count": int, "output_path": str,
         "status": "ok"|"error", "error": str}
    """
    import os, traceback as tb

    try:
        global LLM_BACKEND, LLM_MODEL, AGENT_VERBOSE
        LLM_BACKEND   = config.get("llm_backend",   LLM_BACKEND)
        LLM_MODEL     = config.get("llm_model",     LLM_MODEL)
        AGENT_VERBOSE = config.get("agent_verbose",  AGENT_VERBOSE)

        apps  = shared_state.get("applications",   [])
        infra = shared_state.get("infrastructure", None)

        if not apps:
            raise ValueError(
                "shared_state['applications'] is empty. "
                "Run applications.run() first."
            )
        if infra is None:
            raise ValueError(
                "shared_state['infrastructure'] is None. "
                "Run computing_resources.run() first."
            )

        active_placements: dict = {}
        decisions:         list = []

        print(f"\n[agents.py] Running {len(apps)} applications through "
              f"6-agent ML pipeline ...")

        for i, app in enumerate(apps):
            if AGENT_VERBOSE:
                print(f"\n{'─' * 72}")
                print(f"  App {i+1}/{len(apps)} — {app.instance_id}  "
                      f"({app.domain}/{app.subtype})")
            decisions.append(
                run_agent_pipeline(app, infra, active_placements)
            )

        shared_state["agent_decisions"]   = decisions
        shared_state["active_placements"] = active_placements

        out_dir = config.get("output_dir", ".")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "experiment_decisions.json")
        with open(out_path, "w") as f:
            json.dump([d.to_dict() for d in decisions], f, indent=2)

        return {
            "decisions":   decisions,
            "count":       len(decisions),
            "output_path": out_path,
            "status":      "ok",
        }

    except Exception as e:
        return {
            "decisions":   [],
            "count":       0,
            "output_path": "",
            "status":      "error",
            "error":       str(e),
            "traceback":   tb.format_exc(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE ENTRY POINT
#  Run this file directly in PyCharm to test the full pipeline.
#  Calls applications.py and computing_resources.py automatically —
#  same code path as OrchestratorAgent.py.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[agents.py] Standalone mode — loading applications + infrastructure\n")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import applications        as apps_module
    import computing_resources as cr_module

    _shared: dict = {}
    _config: dict = {
        # Application settings
        "num_apps":          10,
        "domains":           ["healthcare", "industrial", "general"],
        "subtypes":          {"healthcare": None, "industrial": None, "general": None},
        "seed":              42,
        "requirement_mode":  "random",
        # Infrastructure settings
        "active_tiers":      ["cloud", "fog", "edge", "local"],
        "num_fog_nodes":     2,
        "num_edge_nodes":    4,
        "num_local_nodes":   2,
        "variation_pct":     0.15,
        # Agent settings
        "llm_backend":       LLM_BACKEND,
        "llm_model":         LLM_MODEL,
        "agent_verbose":     True,
        "output_dir":        ".",
    }

    # Step 1: Generate applications (from applications.py)
    r1 = apps_module.run(_config, _shared)
    if r1["status"] != "ok":
        print(f"ERROR in applications.py: {r1.get('error')}")
        sys.exit(1)
    print(f"[OK] Generated {r1['count']} application instances\n")

    # Step 2: Build infrastructure (from computing_resources.py)
    r2 = cr_module.run(_config, _shared)
    if r2["status"] != "ok":
        print(f"ERROR in computing_resources.py: {r2.get('error')}")
        sys.exit(1)
    summary = r2.get("summary", {})
    print(f"[OK] Built infrastructure: " +
          " | ".join(f"{t}: {v['node_count']} node(s)"
                     for t, v in summary.items()))
    print()

    # Step 3: Run 6-agent pipeline (this file)
    r3 = run(_config, _shared)
    if r3["status"] != "ok":
        print(f"ERROR in agents.py: {r3.get('error')}")
        print(r3.get("traceback", ""))
        sys.exit(1)

    print_pipeline_summary(r3["decisions"])
    print(f"[OK] Saved decisions → {r3['output_path']}\n")
