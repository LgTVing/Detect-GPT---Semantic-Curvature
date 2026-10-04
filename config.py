# =============================================================================
# config.py - Centralized configuration for the Vietnamese DetectGPT project
# =============================================================================

# [1] MODELS
# -----------------------------------------------------------------------------
# Scoring model used to compute log-likelihood and extract hidden states
# Examples: "google/gemma-3-4b-pt", "Qwen/Qwen2.5-1.5B"
SCORING_MODEL_NAME = "google/gemma-3-4b-pt"

# Mask-filling model used to generate perturbations
# Must be a Seq2Seq model (e.g., T5 / ViT5)
MASK_FILLING_MODEL_NAME = "VietAI/vit5-large"

# Model precision to optimize VRAM usage
SCORING_MODEL_DTYPE = "bfloat16"
MASK_MODEL_DTYPE    = "bfloat16"


# [2] DEVICE SETTINGS
# -----------------------------------------------------------------------------
DEVICE_SCORING = "cuda:0"
DEVICE_MASK    = "cuda:1"


# [3] DATASET SETTINGS
# -----------------------------------------------------------------------------
DATASET_NAME = "ICCIES-2025-DetectAI/vietnamese_news_human_ai"
N_SAMPLES = 1000
DATASET_AI_LABEL = 1
DATASET_HUM_LABEL = 0
RANDOM_SEED = 42
MAX_TOKENS = 512


# [4] PERTURBATION PARAMETERS
# -----------------------------------------------------------------------------
# Number of perturbed variants generated per text
N_PERTURBATIONS = 25
N_PERTURBATION_ROUNDS = 1
SPAN_LENGTH = 2
PCT_WORDS_MASKED = 0.15
BUFFER_SIZE = 1
MASK_TOP_P = 1.0
CHUNK_SIZE = 10


# [5] EVALUATION SETTINGS
# -----------------------------------------------------------------------------
# Target FPR thresholds to evaluate TPR
TPR_AT_FPR_TARGETS = [0.01, 0.05]

# Multi-layer strategy ("last", "weighted", "concat", "sweep")
MULTI_LAYER_STRATEGY = "weighted"
K_LAYERS = 4
LAYER_WEIGHTS = None


# [6] STORAGE & PATHS
# -----------------------------------------------------------------------------
CACHE_DIR = "./.cache"
SAVE_DIR  = "./results"
SAVE_RAW_RESULTS = True
SAVE_PLOTS = True


# [7] DEBUG MODE
# -----------------------------------------------------------------------------
# Enable quick sanity check before running full experiments
DEBUG_MODE = False
DEBUG_N_SAMPLES = 10
DEBUG_N_PERTURBATIONS = 5
