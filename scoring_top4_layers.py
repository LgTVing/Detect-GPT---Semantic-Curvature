# =============================================================================
# scoring_top4_layers.py — Tính Semantic Curvature dựa trên 4 Layer AUROC cao nhất
# Chạy trên Kaggle — Cần add notebook output "detect-full-benchmark" và model Gemma-3 làm Input
# =============================================================================
# Kết quả từ Layer Sweep cho thấy 4 lớp có AUROC cao nhất là:
#   Layer 25: 0.9314  |  Layer 18: 0.9301  |  Layer 19: 0.9274  |  Layer 21: 0.9256
# Chiến lược mới: Nối vector (Concat) tại 4 lớp tối ưu này thay vì 4 lớp đều (8, 18, 25, 34)
# Kỳ vọng: AUROC >= 0.9284 (vượt Concat cũ) vì tập trung vào các tầng ngữ nghĩa mạnh nhất
# =============================================================================

# ─── Cell 1: Cài thư viện ────────────────────────────────────────────────────
# !pip install -q transformers datasets accelerate sentencepiece scipy

# ─── Cell 2: Import ──────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import cosine as cosine_distance
import os
import json
import time
import datetime
import concurrent.futures


# ─── Cell 3: Cấu hình ───────────────────────────────────────────────────────

# === MÔ HÌNH ===
SCORING_MODEL_NAME = "/kaggle/input/models/google/gemma-3/transformers/gemma-3-4b-pt/1"
SCORING_MODEL_DTYPE = "bfloat16"

# === 4 LAYER TỐT NHẤT (từ kết quả Layer Sweep) ===
# Thứ tự AUROC giảm dần: 25 > 18 > 19 > 21
TOP4_LAYERS = [18, 19, 21, 25]   # <-- Đổi ở đây nếu muốn thử tổ hợp khác

# === DỮ LIỆU ===
N_SAMPLES = 1000
N_PERTURBATIONS = 25
MAX_TOKENS = 512
RANDOM_SEED = 42

# === LƯU TRỮ ===
CACHE_DIR = "/root/.cache"
SAVE_DIR = "/kaggle/working/results"
CHECKPOINT_EVERY = 50
TPR_AT_FPR_TARGETS = [0.01, 0.05]

# === ĐƯỜNG DẪN INPUT (output từ datapipeline) ===
DATAPIPELINE_OUTPUT_DIR = "/kaggle/input/notebooks/lngthnhvinh/detect-full-benchmark"

# === RESUME PHA 2 (để rỗng nếu chạy mới) ===
RESUME_SCORING_DIR = ""

# Tạo thư mục lưu kết quả
START_TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
SAVE_FOLDER = os.path.join(
    SAVE_DIR,
    f"{START_TIME}_{SCORING_MODEL_NAME.replace('/','_')}_top4layers"
)
os.makedirs(SAVE_FOLDER, exist_ok=True)
print(f"📁 Kết quả sẽ được lưu tại: {SAVE_FOLDER}")
print(f"🎯 Chiến lược: Concat Top-4 Layers = {TOP4_LAYERS}")


# ─── Cell 4: Load Model ─────────────────────────────────────────────────────

def get_dtype(dtype_str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}.get(dtype_str, torch.float32)

print(f"\n⏳ Loading scoring model: {SCORING_MODEL_NAME}")
scoring_model_0 = transformers.AutoModelForCausalLM.from_pretrained(
    SCORING_MODEL_NAME,
    torch_dtype=get_dtype(SCORING_MODEL_DTYPE),
    cache_dir=CACHE_DIR,
).to("cuda:0").eval()

try:
    scoring_model_1 = transformers.AutoModelForCausalLM.from_pretrained(
        SCORING_MODEL_NAME,
        torch_dtype=get_dtype(SCORING_MODEL_DTYPE),
        cache_dir=CACHE_DIR,
    ).to("cuda:1").eval()
    SCORING_MODELS = [scoring_model_0, scoring_model_1]
    print("✅ Đã load trên cả 2 GPU (cuda:0, cuda:1).")
except RuntimeError:
    SCORING_MODELS = [scoring_model_0, scoring_model_0]
    print("⚠️ Chỉ có 1 GPU. Fallback single GPU.")

scoring_tokenizer = transformers.AutoTokenizer.from_pretrained(
    SCORING_MODEL_NAME, cache_dir=CACHE_DIR
)
if scoring_tokenizer.pad_token is None:
    scoring_tokenizer.pad_token = scoring_tokenizer.eos_token

h_size = getattr(scoring_model_0.config, "hidden_size", "Unknown")
n_transformer_layers = getattr(scoring_model_0.config, "num_hidden_layers", "Unknown")
print(f"✅ Model loaded | hidden_size={h_size}, n_layers={n_transformer_layers}")
print(f"   Vector Concat Top-4 sẽ có chiều: {h_size} × {len(TOP4_LAYERS)} = {h_size * len(TOP4_LAYERS)}")


# ─── Cell 5: Core Functions ─────────────────────────────────────────────────

def _pool(hidden, mask):
    """Mean pooling có attention mask."""
    summed = (hidden * mask).sum(dim=1)
    count  = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / count).squeeze(0).cpu().float().numpy()


def _batch_semantic_vectors_top4(texts, batch_size=8):
    """
    Tính vector biểu diễn Concat Top-4 cho danh sách văn bản.
    Chỉ trích xuất hidden states tại TOP4_LAYERS, nối lại thành vector duy nhất.
    Sử dụng Dual GPU (chunk chẵn/lẻ).
    """
    if not texts:
        return []

    chunks = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    def process_chunk(chunk_idx, chunk_texts):
        gpu_id = chunk_idx % len(SCORING_MODELS)
        model = SCORING_MODELS[gpu_id]
        device = f"cuda:{gpu_id}"

        enc = scoring_tokenizer(
            chunk_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)

        all_hidden = outputs.hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).float()

        def _pool_batch(hidden, mask):
            summed = (hidden * mask).sum(dim=1)
            count  = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / count).cpu().float().numpy()
            return [pooled[j] for j in range(pooled.shape[0])]

        # Trích xuất và nối vector tại TOP4_LAYERS
        parts = [np.array(_pool_batch(all_hidden[layer_idx], mask)) for layer_idx in TOP4_LAYERS]
        concat_arr = np.concatenate(parts, axis=1)
        chunk_results = [concat_arr[j] for j in range(concat_arr.shape[0])]

        del enc, outputs, all_hidden, mask
        torch.cuda.empty_cache()
        return chunk_idx, chunk_results

    all_results_unsorted = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SCORING_MODELS)) as executor:
        futures = [executor.submit(process_chunk, i, chunks[i]) for i in range(len(chunks))]
        for future in concurrent.futures.as_completed(futures):
            all_results_unsorted.append(future.result())

    all_results_unsorted.sort(key=lambda x: x[0])
    return [v for _, chunk_res in all_results_unsorted for v in chunk_res]


# ─── Cell 6: Evaluation Metrics ─────────────────────────────────────────────

def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0]*len(real_preds) + [1]*len(sample_preds), real_preds + sample_preds)
    return fpr.tolist(), tpr.tolist(), float(auc(fpr, tpr))

def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0]*len(real_preds) + [1]*len(sample_preds), real_preds + sample_preds)
    return precision.tolist(), recall.tolist(), float(auc(recall, precision))

def get_accuracy_f1(real_preds, sample_preds):
    labels = [0]*len(real_preds) + [1]*len(sample_preds)
    scores = real_preds + sample_preds
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = int(np.argmax(np.array(tpr) - np.array(fpr)))
    optimal_threshold = float(thresholds[optimal_idx])
    preds = [1 if s >= optimal_threshold else 0 for s in scores]
    return float(accuracy_score(labels, preds)), float(f1_score(labels, preds)), optimal_threshold

def get_tpr_at_fpr(real_preds, sample_preds, target_fpr=0.01):
    fpr_arr, tpr_arr, _ = roc_curve([0]*len(real_preds) + [1]*len(sample_preds), real_preds + sample_preds)
    idx = int(np.argmin(np.abs(np.array(fpr_arr) - target_fpr)))
    return float(tpr_arr[idx])


# ─── Cell 7: Checkpoint Helpers ──────────────────────────────────────────────

CHECKPOINT_SCORE = os.path.join(SAVE_FOLDER, "ckpt_score_progress.json")

def ckpt_save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)
    print(f"   💾 Checkpoint saved → {os.path.basename(path)}")

def ckpt_load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ─── Cell 8: Tìm file dữ liệu ──────────────────────────────────────────────

def find_datapipeline_files(base_dir):
    raw_data_path = ckpt_perturb_path = None
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "raw_data.json" and raw_data_path is None:
                raw_data_path = os.path.join(root, f)
            if f == "ckpt_perturb_texts.json" and ckpt_perturb_path is None:
                ckpt_perturb_path = os.path.join(root, f)
        if raw_data_path and ckpt_perturb_path:
            break
    return raw_data_path, ckpt_perturb_path


# ─── Cell 9: MAIN — Scoring Loop ────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print(f"🧠 [SCORING] Concat Top-4 Best Layers: {TOP4_LAYERS}")
    print("="*70)

    import shutil
    raw_data_path, ckpt_perturb_path = find_datapipeline_files(DATAPIPELINE_OUTPUT_DIR)

    if raw_data_path is None:
        raise FileNotFoundError(f"Không tìm thấy raw_data.json trong {DATAPIPELINE_OUTPUT_DIR}")
    if ckpt_perturb_path is None:
        raise FileNotFoundError(f"Không tìm thấy ckpt_perturb_texts.json trong {DATAPIPELINE_OUTPUT_DIR}")

    print(f"📂 raw_data: {raw_data_path}")
    print(f"📂 perturb:  {ckpt_perturb_path}")

    with open(raw_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ {len(data['original'])} Người + {len(data['sampled'])} AI")

    with open(ckpt_perturb_path, "r", encoding="utf-8") as f:
        perturb_ckpt = json.load(f)
    p_sampled_text  = perturb_ckpt["p_sampled"]
    p_original_text = perturb_ckpt["p_original"]
    print(f"✅ Perturbation texts: {len(p_original_text)} entries")

    results = []
    for idx in range(len(data["original"])):
        results.append({
            "original":           data["original"][idx],
            "sampled":            data["sampled"][idx],
            "perturbed_original": p_original_text[idx*N_PERTURBATIONS:(idx+1)*N_PERTURBATIONS],
            "perturbed_sampled":  p_sampled_text[idx*N_PERTURBATIONS:(idx+1)*N_PERTURBATIONS],
        })

    if RESUME_SCORING_DIR:
        old_ckpt = os.path.join(RESUME_SCORING_DIR, "ckpt_score_progress.json")
        if os.path.exists(old_ckpt):
            shutil.copy2(old_ckpt, CHECKPOINT_SCORE)

    score_ckpt = ckpt_load(CHECKPOINT_SCORE)
    if score_ckpt is not None:
        done_results = score_ckpt["done_results"]
        start_idx = len(done_results)
        print(f"⏭️  Resume: đã xong {start_idx}/{len(results)} mẫu.")
        for i, done in enumerate(done_results):
            results[i].update(done)
    else:
        done_results = []
        start_idx = 0

    print(f"\n🧠 Tính Semantic Curvature (Concat Top-4: {TOP4_LAYERS})...")
    t_start = time.time()

    for idx in tqdm.tqdm(range(start_idx, len(results)),
                         desc="SC Concat-Top4", initial=start_idx, total=len(results)):
        res = results[idx]
        all_texts = (
            [res["original"], res["sampled"]]
            + res["perturbed_original"]
            + res["perturbed_sampled"]
        )
        all_vecs = _batch_semantic_vectors_top4(all_texts)

        v_orig      = all_vecs[0]
        v_samp      = all_vecs[1]
        p_orig_vecs = all_vecs[2 : 2 + N_PERTURBATIONS]
        p_samp_vecs = all_vecs[2 + N_PERTURBATIONS : 2 + 2*N_PERTURBATIONS]

        sc_orig = float(np.mean([cosine_distance(v_orig, v) for v in p_orig_vecs]))
        sc_samp = float(np.mean([cosine_distance(v_samp, v) for v in p_samp_vecs]))

        scored = {"sc_concat_top4_original": sc_orig, "sc_concat_top4_sampled": sc_samp}
        res.update(scored)
        done_results.append(scored)

        if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == len(results):
            ckpt_save(CHECKPOINT_SCORE, {"done_results": done_results})

    elapsed = time.time() - t_start
    print(f"\n✓ Chấm xong {len(results)} mẫu trong {elapsed/60:.1f} phút.")

    # ─── Cell 10: Tính metrics và so sánh ────────────────────────────────────

    sem_real    = [r["sc_concat_top4_original"] for r in results]
    sem_samples = [r["sc_concat_top4_sampled"]  for r in results]

    fpr, tpr, roc_auc = get_roc_metrics(sem_real, sem_samples)
    p, r, pr_auc      = get_precision_recall_metrics(sem_real, sem_samples)
    acc, f1, threshold = get_accuracy_f1(sem_real, sem_samples)
    tpr_at_fprs = {f"tpr@fpr={t:.0%}": get_tpr_at_fpr(sem_real, sem_samples, t)
                   for t in TPR_AT_FPR_TARGETS}

    print("\n" + "="*70)
    print(f"⭐ KẾT QUẢ: Concat Top-4 Layers {TOP4_LAYERS}")
    print("="*70)
    print(f"   AUROC     = {roc_auc:.6f}")
    print(f"   AUPRC     = {pr_auc:.6f}")
    print(f"   Accuracy  = {acc:.4f}")
    print(f"   F1 Score  = {f1:.4f}")
    print(f"   Threshold = {threshold:.8f}")
    for name, val in tpr_at_fprs.items():
        print(f"   {name} = {val:.4f}")

    print("\n📊 SO SÁNH VỚI CÁC CHIẾN LƯỢC TRƯỚC ĐÓ:")
    prev_results = {
        "SC Last (Lớp 34)":          0.9138,
        "SC Weighted (4 lớp cuối)":  0.9102,
        "SC Concat (8,18,25,34)":    0.9284,
        "SC Sweep Layer 25":         0.9314,
    }
    for name, prev_auroc in prev_results.items():
        delta = roc_auc - prev_auroc
        symbol = "🟢" if delta > 0 else ("🔴" if delta < 0 else "🟡")
        print(f"   {symbol} vs {name}: {prev_auroc:.4f} → {roc_auc:.4f} ({delta:+.4f})")

    output = {
        'name': 'semantic_curvature_concat_top4',
        'top4_layers': TOP4_LAYERS,
        'predictions': {'real': sem_real, 'samples': sem_samples},
        'metrics':     {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics':  {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'extra_metrics': {'accuracy': acc, 'f1': f1, 'threshold': threshold, **tpr_at_fprs},
    }
    with open(os.path.join(SAVE_FOLDER, "semantic_curvature_concat_top4.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in output.items() if k != 'raw_results'}, f, indent=2)

    # Vẽ ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#E63946', lw=2.5, label=f'Concat Top-4 {TOP4_LAYERS}, AUROC={roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve — Concat Top-4 Best Layers\nLayers {TOP4_LAYERS}', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FOLDER, "roc_concat_top4.png"), dpi=150)

    # Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(sem_real,    bins=30, alpha=0.65, color='steelblue', label='Human (người viết)')
    ax.hist(sem_samples, bins=30, alpha=0.65, color='tomato',    label='AI (máy viết)')
    ax.axvline(threshold, color='green', linestyle='--', lw=2, label=f'Ngưỡng tối ưu = {threshold:.6f}')
    ax.set_xlabel('Semantic Curvature (Concat Top-4)', fontsize=11)
    ax.set_ylabel('Số lượng mẫu', fontsize=11)
    ax.set_title(f'Phân bố SC Concat Top-4 Layers {TOP4_LAYERS}\nAUROC = {roc_auc:.4f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); fig.tight_layout()
    plt.savefig(os.path.join(SAVE_FOLDER, "histogram_concat_top4.png"), dpi=150)

    print(f"\n✅ Hoàn thành! Kết quả tại: {SAVE_FOLDER}")
