# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import datasets as hf_datasets
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
import functools
import time
import datetime

from config import (
    SCORING_MODEL_NAME, MASK_FILLING_MODEL_NAME,
    SCORING_MODEL_DTYPE, MASK_MODEL_DTYPE,
    DEVICE_SCORING, DEVICE_MASK,
    DATASET_NAME, N_SAMPLES, DATASET_AI_LABEL, DATASET_HUM_LABEL,
    RANDOM_SEED, MAX_TOKENS,
    N_PERTURBATIONS, N_PERTURBATION_ROUNDS, SPAN_LENGTH,
    PCT_WORDS_MASKED, BUFFER_SIZE, MASK_TOP_P, CHUNK_SIZE,
    TPR_AT_FPR_TARGETS,
    CACHE_DIR, SAVE_DIR, SAVE_RAW_RESULTS, SAVE_PLOTS,
    DEBUG_MODE, DEBUG_N_SAMPLES, DEBUG_N_PERTURBATIONS,
    MULTI_LAYER_STRATEGY, K_LAYERS, LAYER_WEIGHTS,
)

if DEBUG_MODE:
    print("⚠️  DEBUG MODE: chạy với dữ liệu nhỏ để kiểm tra pipeline")
    N_SAMPLES       = DEBUG_N_SAMPLES
    N_PERTURBATIONS = DEBUG_N_PERTURBATIONS

COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
          "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
          "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]
pattern = re.compile(r"<extra_id_\d+>")

START_TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
SAVE_FOLDER = os.path.join(SAVE_DIR, f"{START_TIME}_{SCORING_MODEL_NAME.replace('/','_')}")
os.makedirs(SAVE_FOLDER, exist_ok=True)
print(f"📁 Kết quả sẽ được lưu tại: {SAVE_FOLDER}")

def get_dtype(dtype_str):
    """Convert a string representation to a PyTorch dtype."""
    
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}.get(dtype_str, torch.float32)

print(f"⏳ Loading scoring model: {SCORING_MODEL_NAME} → Dual GPUs (cuda:0, cuda:1)")
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
    print("✅ Đã load thành công trên cả 2 GPU (cuda:0, cuda:1).")
except RuntimeError as e:
    print(f"⚠️ Lỗi khi load cuda:1 (có thể máy chỉ có 1 GPU). Fallback về 1 GPU.\nChi tiết lỗi: {e}")
    SCORING_MODELS = [scoring_model_0, scoring_model_0]

scoring_model = scoring_model_0  # Backward compatibility cho các hàm cũ
scoring_tokenizer = transformers.AutoTokenizer.from_pretrained(
    SCORING_MODEL_NAME, cache_dir=CACHE_DIR
)
if scoring_tokenizer.pad_token is None:
    scoring_tokenizer.pad_token = scoring_tokenizer.eos_token
h_size = getattr(scoring_model_0.config, "hidden_size", getattr(scoring_model_0.config, "hidden_dim", getattr(scoring_model_0.config, "d_model", "Unknown")))
print(f"✅ Scoring model loaded | hidden_size = {h_size}")

def _forward_scoring(text, need_hidden=False):
    
    tokenized = scoring_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
    ).to(DEVICE_SCORING)
    outputs = scoring_model(**tokenized, output_hidden_states=need_hidden)
    return tokenized, outputs

def get_rank(text, log=False):
    
    with torch.no_grad():
        tokenized, outputs = _forward_scoring(text, need_hidden=False)
        logits = outputs.logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]
        matches   = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
        ranks, timesteps = matches[:, -1], matches[:, -2]
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all()
        ranks = ranks.float() + 1
        if log:
            ranks = torch.log(ranks)
        return ranks.float().mean().item()

def get_entropy(text):
    
    with torch.no_grad():
        tokenized, outputs = _forward_scoring(text, need_hidden=False)
        logits      = outputs.logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()

def _pool(hidden, mask):
    
    summed = (hidden * mask).sum(dim=1)
    count  = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / count).squeeze(0).cpu().float().numpy()

def get_semantic_vector(text):
    
    with torch.no_grad():
        tokenized = scoring_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
        ).to(DEVICE_SCORING)
        outputs = scoring_model(**tokenized, output_hidden_states=True)

    all_hidden = outputs.hidden_states      # tuple len = n_transformer_layers + 1
    n_layers   = len(all_hidden)            # bao gồm embedding layer (index 0)
    mask = tokenized["attention_mask"].unsqueeze(-1).float()   # (1, seq_len, 1)

    if MULTI_LAYER_STRATEGY == "last":
        return _pool(all_hidden[-1], mask)

    elif MULTI_LAYER_STRATEGY == "weighted":
        k = min(K_LAYERS, n_layers - 1)   # bỏ embedding (index 0)
        layers = [all_hidden[-(i+1)] for i in range(k-1, -1, -1)]  # tăng dần
        if LAYER_WEIGHTS is None:
            weights = [1.0 / k] * k
        else:
            assert len(LAYER_WEIGHTS) == k, (
                f"LAYER_WEIGHTS phải có độ dài K_LAYERS={k}, "
                f"đang là {len(LAYER_WEIGHTS)}"
            )
            s = sum(LAYER_WEIGHTS)
            weights = [w / s for w in LAYER_WEIGHTS]
        pooled = [_pool(h, mask) for h in layers]
        vector = sum(w * p for w, p in zip(weights, pooled))
        return vector

    elif MULTI_LAYER_STRATEGY == "concat":
        tl = n_layers - 1   # số transformer layer thực
        checkpoints = sorted(set([
            max(1, tl // 4),
            max(1, tl // 2),
            max(1, 3 * tl // 4),
            tl,
        ]))
        pooled = [_pool(all_hidden[i], mask) for i in checkpoints]
        return np.concatenate(pooled, axis=0)

    elif MULTI_LAYER_STRATEGY == "sweep":
        return {i: _pool(all_hidden[i], mask) for i in range(1, n_layers)}

    else:
        raise ValueError(
            f"MULTI_LAYER_STRATEGY không hợp lệ: '{MULTI_LAYER_STRATEGY}'. "
            "Chọn một trong: last, weighted, concat, sweep"
        )

def get_semantic_vectors(texts):
    
    return [get_semantic_vector(t) for t in texts]

import concurrent.futures

def _batch_semantic_vectors(texts, batch_size=8):
    
    if not texts:
        return []
        
    chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    
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
            
        all_hidden = outputs.hidden_states   # tuple, len = n_layers+1
        n_layers   = len(all_hidden)
        mask = enc["attention_mask"].unsqueeze(-1).float() # (batch, seq_len, 1)

        def _pool_batch(hidden, mask):
            summed = (hidden * mask).sum(dim=1)
            count  = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / count).cpu().float().numpy()
            return [pooled[j] for j in range(pooled.shape[0])]

        last_pooled = _pool_batch(all_hidden[-1], mask)

        k = min(K_LAYERS, n_layers - 1)
        layers = [all_hidden[-(j+1)] for j in range(k-1, -1, -1)]
        weights = ([1.0/k]*k if LAYER_WEIGHTS is None else [w/sum(LAYER_WEIGHTS) for w in LAYER_WEIGHTS])
        result_w = None
        for w, h in zip(weights, layers):
            batch_pooled = np.array(_pool_batch(h, mask))
            result_w = (batch_pooled * w) if result_w is None else (result_w + batch_pooled * w)
        weighted_pooled = [result_w[j] for j in range(result_w.shape[0])]

        tl = n_layers - 1
        cps = sorted(set([max(1,tl//4), max(1,tl//2), max(1,3*tl//4), tl]))
        parts = [np.array(_pool_batch(all_hidden[j], mask)) for j in cps]
        concat_arr = np.concatenate(parts, axis=1)
        concat_pooled = [concat_arr[j] for j in range(concat_arr.shape[0])]

        sweep_layers = {}
        for layer_idx in range(1, n_layers):
            sweep_layers[layer_idx] = _pool_batch(all_hidden[layer_idx], mask)

        chunk_results = []
        for j in range(len(chunk_texts)):
            chunk_results.append({
                "last": last_pooled[j],
                "weighted": weighted_pooled[j],
                "concat": concat_pooled[j],
                "sweep": {lid: sweep_layers[lid][j] for lid in sweep_layers}
            })
            
        del enc, outputs, all_hidden, mask, sweep_layers
        torch.cuda.empty_cache()
        return chunk_idx, chunk_results

    all_results_unsorted = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SCORING_MODELS)) as executor:
        futures = [executor.submit(process_chunk, i, chunks[i]) for i in range(len(chunks))]
        for future in concurrent.futures.as_completed(futures):
            all_results_unsorted.append(future.result())
            
    all_results_unsorted.sort(key=lambda x: x[0])
    
    all_results = []
    for _, chunk_res in all_results_unsorted:
        all_results.extend(chunk_res)

    return all_results

def get_ll_and_semantic_vector(text):
    
    with torch.no_grad():
        tokenized = scoring_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
        ).to(DEVICE_SCORING)
        outputs = scoring_model(**tokenized, output_hidden_states=True)

    logits       = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokenized.input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    ll = -loss.item()

    all_hidden = outputs.hidden_states
    n_layers   = len(all_hidden)
    mask = tokenized["attention_mask"].unsqueeze(-1).float()

    if MULTI_LAYER_STRATEGY == "last":
        vec = _pool(all_hidden[-1], mask)
    elif MULTI_LAYER_STRATEGY == "weighted":
        k = min(K_LAYERS, n_layers - 1)
        layers = [all_hidden[-(i+1)] for i in range(k-1, -1, -1)]
        weights = ([1.0/k]*k if LAYER_WEIGHTS is None
                   else [w/sum(LAYER_WEIGHTS) for w in LAYER_WEIGHTS])
        pooled = [_pool(h, mask) for h in layers]
        vec = sum(w * p for w, p in zip(weights, pooled))
    elif MULTI_LAYER_STRATEGY == "concat":
        tl = n_layers - 1
        cps = sorted(set([max(1,tl//4), max(1,tl//2), max(1,3*tl//4), tl]))
        vec = np.concatenate([_pool(all_hidden[i], mask) for i in cps])
    elif MULTI_LAYER_STRATEGY == "sweep":
        vec = {i: _pool(all_hidden[i], mask) for i in range(1, n_layers)}
    else:
        raise ValueError(f"MULTI_LAYER_STRATEGY không hợp lệ: '{MULTI_LAYER_STRATEGY}'")

    return ll, vec

def get_ll(text):
    """Calculate the log-likelihood of a text using the scoring model."""
    
    ll, _ = get_ll_and_semantic_vector(text)
    return ll

def get_lls(texts):
    return [get_ll(t) for t in texts]

def get_roc_metrics(real_preds, sample_preds):
    
    fpr, tpr, _ = roc_curve(
        [0] * len(real_preds) + [1] * len(sample_preds),
        real_preds + sample_preds
    )
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_preds, sample_preds):
    
    precision, recall, _ = precision_recall_curve(
        [0] * len(real_preds) + [1] * len(sample_preds),
        real_preds + sample_preds
    )
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def get_accuracy_f1(real_preds, sample_preds):
    
    labels = [0] * len(real_preds) + [1] * len(sample_preds)
    scores = real_preds + sample_preds
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx       = int(np.argmax(np.array(tpr) - np.array(fpr)))
    optimal_threshold = float(thresholds[optimal_idx])
    preds  = [1 if s >= optimal_threshold else 0 for s in scores]
    acc    = float(accuracy_score(labels, preds))
    f1     = float(f1_score(labels, preds, zero_division=0))
    return acc, f1, optimal_threshold

def get_tpr_at_fpr(real_preds, sample_preds, target_fpr=0.01):
    
    fpr_arr, tpr_arr, _ = roc_curve(
        [0] * len(real_preds) + [1] * len(sample_preds),
        real_preds + sample_preds
    )
    idx = int(np.argmin(np.abs(np.array(fpr_arr) - target_fpr)))
    return float(tpr_arr[idx])

def compute_full_metrics(real_preds, sample_preds):
    
    fpr, tpr, roc_auc = get_roc_metrics(real_preds, sample_preds)
    p, r, pr_auc      = get_precision_recall_metrics(real_preds, sample_preds)
    acc, f1, _        = get_accuracy_f1(real_preds, sample_preds)
    tpr_at_fprs       = {f"tpr@fpr={t:.0%}": get_tpr_at_fpr(real_preds, sample_preds, t)
                         for t in TPR_AT_FPR_TARGETS}
    return {
        "roc_auc": roc_auc, "fpr": fpr, "tpr": tpr,
        "pr_auc": pr_auc, "precision": p, "recall": r,
        "accuracy": acc, "f1": f1,
        **tpr_at_fprs,
    }

def save_roc_curves(experiments):
    
    if not SAVE_PLOTS:
        return
    plt.clf()
    for experiment, color in zip(experiments, COLORS):
        m = experiment["metrics"]
        plt.plot(m["fpr"], m["tpr"],
                 label=f"{experiment['name']}, AUROC={m['roc_auc']:.3f}", color=color)
        print(f"  {experiment['name']}: AUROC={m['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves\n{SCORING_MODEL_NAME} + {MASK_FILLING_MODEL_NAME}')
    plt.legend(loc="lower right", fontsize=6)
    path = os.path.join(SAVE_FOLDER, "roc_curves.png")
    plt.savefig(path, dpi=150)
    print(f"💾 ROC curve saved: {path}")

def save_ll_histograms(experiments):
    
    if not SAVE_PLOTS:
        return
    for experiment in experiments:
        try:
            results = experiment.get("raw_results", [])
            if not results:
                continue
            plt.clf()
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results],    alpha=0.5, bins='auto', label='AI (sampled)')
            plt.hist([r["perturbed_sampled_ll"] for r in results], alpha=0.5, bins='auto', label='AI perturbed')
            plt.xlabel("log likelihood"); plt.ylabel('count'); plt.legend()
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results],   alpha=0.5, bins='auto', label='Human (original)')
            plt.hist([r["perturbed_original_ll"] for r in results], alpha=0.5, bins='auto', label='Human perturbed')
            plt.xlabel("log likelihood"); plt.ylabel('count'); plt.legend()
            path = os.path.join(SAVE_FOLDER, f"ll_hist_{experiment['name']}.png")
            plt.savefig(path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"  ⚠️  Không thể vẽ histogram cho {experiment.get('name','?')}: {e}")

def save_score_histograms(perturbation_results, semantic_output, detectgpt_outputs):
    
    if not SAVE_PLOTS or not perturbation_results:
        return

    import matplotlib.pyplot as plt
    fig_configs = [
        {
            "name": "likelihood",
            "human_vals": [r["original_ll"] for r in perturbation_results],
            "ai_vals":    [r["sampled_ll"]  for r in perturbation_results],
            "xlabel":     "Log-Likelihood",
            "title":      "Phân bố Log-Likelihood (AI vs Human)",
            "ai_higher":  True,
        },
        {
            "name": "dscore",
            "human_vals": [r["original_ll"] - r["perturbed_original_ll"] for r in perturbation_results],
            "ai_vals":    [r["sampled_ll"]  - r["perturbed_sampled_ll"]  for r in perturbation_results],
            "xlabel":     "d-score = LL(x) - Mean(LL(x̃))",
            "title":      "Phân bố DetectGPT d-score (AI vs Human)",
            "ai_higher":  True,
        },
        {
            "name": "zscore",
            "human_vals": [(r["original_ll"] - r["perturbed_original_ll"]) / (r["perturbed_original_ll_std"] or 1)
                           for r in perturbation_results],
            "ai_vals":    [(r["sampled_ll"]  - r["perturbed_sampled_ll"])  / (r["perturbed_sampled_ll_std"]  or 1)
                           for r in perturbation_results],
            "xlabel":     "z-score = d-score / Std(LL(x̃))",
            "title":      "Phân bố DetectGPT z-score (AI vs Human)",
            "ai_higher":  True,
        },
        {
            "name": "semantic_curvature",
            "human_vals": [r["semantic_curvature_original"] for r in perturbation_results],
            "ai_vals":    [r["semantic_curvature_sampled"]  for r in perturbation_results],
            "xlabel":     "Semantic Curvature = Mean(cosine_dist(v(x), v(x̃)))",
            "title":      "Phân bố Semantic Curvature (AI vs Human)",
            "ai_higher":  True,
        },
    ]

    for cfg in fig_configs:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(cfg["human_vals"], bins=30, alpha=0.65, color='steelblue', label='Human (người viết)')
        ax.hist(cfg["ai_vals"],    bins=30, alpha=0.65, color='tomato',    label='AI (máy viết)')
        ax.set_xlabel(cfg["xlabel"], fontsize=11)
        ax.set_ylabel('Số lượng mẫu', fontsize=11)
        ax.set_title(cfg["title"], fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        decision_note = "→ Giá trị CAO hơn → dự đoán là AI" if cfg["ai_higher"] else "→ Giá trị THẤP hơn → dự đoán là AI"
        ax.annotate(decision_note, xy=(0.97, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))
        fig.tight_layout()
        path = os.path.join(SAVE_FOLDER, f"histogram_{cfg['name']}.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"💾 Histogram saved: {path}")

def print_comparison_table(all_outputs):
    
    header = f"{'Phương pháp':<30} | {'AUROC':>7} | {'AUPRC':>7} | {'Acc':>7} | {'F1':>7}"
    for t in TPR_AT_FPR_TARGETS:
        header += f" | {'TPR@' + f'{t:.0%}':>8}"
    sep = "─" * len(header)
    lines = [f"\n{sep}", header, sep]
    for output in all_outputs:
        name    = output['name']
        m       = output['metrics']
        roc_auc = m['roc_auc']
        pr_auc  = m.get('pr_auc', output.get('pr_metrics', {}).get('pr_auc', 0))
        preds   = output['predictions']
        acc, f1, _ = get_accuracy_f1(preds['real'], preds['samples'])
        row = f"{name:<30} | {roc_auc:>7.4f} | {pr_auc:>7.4f} | {acc:>6.1%} | {f1:>7.4f}"
        for t in TPR_AT_FPR_TARGETS:
            tpr_val = get_tpr_at_fpr(preds['real'], preds['samples'], t)
            row += f" | {tpr_val:>7.1%}"
        lines.append(row)
    lines.append(sep)
    table_str = "\n".join(lines)
    print(table_str)
    table_path = os.path.join(SAVE_FOLDER, "summary_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"💾 Summary table saved: {table_path}")

CHECKPOINT_PERTURB = os.path.join(SAVE_FOLDER, "ckpt_perturb_texts.json")
CHECKPOINT_SCORE   = os.path.join(SAVE_FOLDER, "ckpt_score_progress.json")
CHECKPOINT_EVERY   = 50   # Lưu sau mỗi N mẫu đã chấm xong

def ckpt_save(path, obj):
    """Save progress to a checkpoint file."""
    
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)   # atomic rename
    print(f"   💾 Checkpoint saved → {os.path.basename(path)}")

def ckpt_load(path):
    """Load progress from a checkpoint file."""
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def run_baseline_threshold_experiment(criterion_fn, name, n_samples=N_SAMPLES):
    
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    results = []
    original_text = data["original"]
    sampled_text  = data["sampled"]
    for idx in tqdm.tqdm(range(min(n_samples // 2, len(original_text))),
                         desc=f"Baseline [{name}]"):
        results.append({
            "original":      original_text[idx],
            "original_crit": criterion_fn(original_text[idx]),
            "sampled":       sampled_text[idx],
            "sampled_crit":  criterion_fn(sampled_text[idx]),
        })
    predictions = {
        'real':    [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"]  for x in results],
    }
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc      = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"  {name}: AUROC={roc_auc:.4f}, AUPRC={pr_auc:.4f}")
    return {
        'name': f'{name}_threshold',
        'predictions': predictions,
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'pr_auc': pr_auc,
        'loss': 1 - pr_auc,
    }

def run_perturbation_experiment(results, criterion, span_length=SPAN_LENGTH,
                                n_perturbations=N_PERTURBATIONS):
    
    predictions = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            std_orig = res['perturbed_original_ll_std'] or 1
            std_samp = res['perturbed_sampled_ll_std']  or 1
            predictions['real'].append(
                (res['original_ll'] - res['perturbed_original_ll']) / std_orig)
            predictions['samples'].append(
                (res['sampled_ll'] - res['perturbed_sampled_ll']) / std_samp)
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc      = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"  {name}: AUROC={roc_auc:.4f}, AUPRC={pr_auc:.4f}")
    return {
        'name': name,
        'predictions': predictions,
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'pr_auc': pr_auc,
        'loss': 1 - pr_auc,
    }

def get_perturbation_results(n_perturbations=N_PERTURBATIONS):
    
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    perturb_ckpt = ckpt_load(CHECKPOINT_PERTURB)
    if perturb_ckpt is None:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint perturbation: {CHECKPOINT_PERTURB}\n"
            "Hãy kiểm tra lại DATAPIPELINE_OUTPUT_DIR hoặc chạy datapipeline.py trước!"
        )
    print(f"\n⏭️  [PHA 2] Phục hồi perturbation texts từ checkpoint "
          f"({len(perturb_ckpt['p_original'])} entries).")
    p_sampled_text  = perturb_ckpt["p_sampled"]
    p_original_text = perturb_ckpt["p_original"]

    results = []
    for idx in range(len(data["original"])):
        results.append({
            "original":           data["original"][idx],
            "sampled":            data["sampled"][idx],
            "perturbed_sampled":  p_sampled_text [idx * n_perturbations:(idx + 1) * n_perturbations],
            "perturbed_original": p_original_text[idx * n_perturbations:(idx + 1) * n_perturbations],
        })

    score_ckpt = ckpt_load(CHECKPOINT_SCORE)
    if score_ckpt is not None:
        done_results = score_ckpt["done_results"]
        if len(done_results) > len(results):
            print(f"⚠️  Checkpoint có {len(done_results)} entries > {len(results)} mẫu "
                  "→ bỏ qua, chạy lại từ đầu.")
            done_results = []
            start_idx    = 0
        else:
            start_idx = len(done_results)
            print(f"\n⏭️  [PHA 2] Phục hồi chấm điểm: đã xong {start_idx}/{len(results)} mẫu.")
            for i, done in enumerate(done_results):
                results[i].update(done)
    else:
        done_results = []
        start_idx    = 0
        print(f"\n🧠 [PHA 2] Tính Semantic Vectors (Multi-Layer: {MULTI_LAYER_STRATEGY})...")

    for idx in tqdm.tqdm(range(start_idx, len(results)), desc="Semantic All Strategies",
                         initial=start_idx, total=len(results)):
        res = results[idx]

        scored = {}

        all_texts = (
            [res["original"], res["sampled"]]
            + res["perturbed_original"]
            + res["perturbed_sampled"]
        )
        all_vecs = _batch_semantic_vectors(all_texts)
        v_orig       = all_vecs[0]
        v_samp       = all_vecs[1]
        p_orig_vecs  = all_vecs[2 : 2 + n_perturbations]
        p_samp_vecs  = all_vecs[2 + n_perturbations : 2 + 2 * n_perturbations]

        for strategy in ["last", "weighted", "concat"]:
            scored[f"semantic_curvature_{strategy}_original"] = float(
                np.mean([cosine_distance(v_orig[strategy], v[strategy]) for v in p_orig_vecs])
            )
            scored[f"semantic_curvature_{strategy}_sampled"] = float(
                np.mean([cosine_distance(v_samp[strategy], v[strategy]) for v in p_samp_vecs])
            )

        v_orig_sw = v_orig["sweep"]
        v_samp_sw = v_samp["sweep"]
        p_orig_sw = [v["sweep"] for v in p_orig_vecs]
        p_samp_sw = [v["sweep"] for v in p_samp_vecs]

        layer_ids = sorted(
            set(v_orig_sw.keys())
            & set(v_samp_sw.keys())
            & set.intersection(*[set(v.keys()) for v in p_orig_sw])
            & set.intersection(*[set(v.keys()) for v in p_samp_sw])
        )
        if not layer_ids:
            raise RuntimeError(f"Mẫu {idx}: không có layer_id chung giữa các văn bản!")

        sc_orig_per_layer = {
            str(i): float(np.mean([cosine_distance(v_orig_sw[i], v[i]) for v in p_orig_sw]))
            for i in layer_ids
        }
        sc_samp_per_layer = {
            str(i): float(np.mean([cosine_distance(v_samp_sw[i], v[i]) for v in p_samp_sw]))
            for i in layer_ids
        }
        scored["semantic_curvature_sweep_original_per_layer"] = sc_orig_per_layer
        scored["semantic_curvature_sweep_sampled_per_layer"]  = sc_samp_per_layer

        res.update(scored)
        done_results.append(scored)

        if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == len(results):
            ckpt_save(CHECKPOINT_SCORE, {"done_results": done_results})

    print(f"\n✓ Chấm xong {len(results)} mẫu (Semantic Only).")
    return results

DATAPIPELINE_OUTPUT_DIR = "./detect-full-benchmark"

RESUME_SCORING_DIR = ""

def find_datapipeline_files(base_dir):
    """Locate output files from the data pipeline phase."""
    
    raw_data_path = None
    ckpt_perturb_path = None
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "raw_data.json" and raw_data_path is None:
                raw_data_path = os.path.join(root, f)
            if f == "ckpt_perturb_texts.json" and ckpt_perturb_path is None:
                ckpt_perturb_path = os.path.join(root, f)
        if raw_data_path and ckpt_perturb_path:
            break
    return raw_data_path, ckpt_perturb_path

if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"🧠 [SCORING] Multi-Layer Semantic Curvature (All Strategies: last, weighted, concat, sweep)")
    print("="*60)
    
    import shutil
    raw_data_path, ckpt_perturb_path = find_datapipeline_files(DATAPIPELINE_OUTPUT_DIR)
    
    if raw_data_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy raw_data.json trong {DATAPIPELINE_OUTPUT_DIR}\n"
            "Hãy chạy datapipeline.py trước hoặc kiểm tra lại đường dẫn DATAPIPELINE_OUTPUT_DIR!"
        )
    if ckpt_perturb_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy ckpt_perturb_texts.json trong {DATAPIPELINE_OUTPUT_DIR}\n"
            "Hãy chạy datapipeline.py trước hoặc kiểm tra lại đường dẫn DATAPIPELINE_OUTPUT_DIR!"
        )
    
    print(f"📂 raw_data.json: {raw_data_path}")
    print(f"📂 ckpt_perturb_texts.json: {ckpt_perturb_path}")
    
    if RESUME_SCORING_DIR:
        old_score_ckpt = os.path.join(RESUME_SCORING_DIR, "ckpt_score_progress.json")
        if os.path.exists(old_score_ckpt):
            print(f"📂 TÌM THẤY CHECKPOINT CHẤM ĐIỂM CŨ: {old_score_ckpt}")
            shutil.copy2(old_score_ckpt, CHECKPOINT_SCORE)
            print(f"   => Đã copy sang working directory để chạy tiếp (resume).")
        else:
            print(f"⚠️ Không tìm thấy ckpt_score_progress.json trong {RESUME_SCORING_DIR}")
    
    with open(raw_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data['original'])} Người + {len(data['sampled'])} AI từ raw_data.json")
    
    CHECKPOINT_PERTURB = ckpt_perturb_path
        
    perturbation_results = get_perturbation_results(N_PERTURBATIONS)

    print(f"\n🌀 Phân tích Semantic Curvature cho tất cả các chiến lược (last, weighted, concat, sweep)...")
    
    all_outputs = []

    for strategy in ["last", "weighted", "concat"]:
        sem_real    = [r[f"semantic_curvature_{strategy}_original"] for r in perturbation_results]
        sem_samples = [r[f"semantic_curvature_{strategy}_sampled"]  for r in perturbation_results]
        
        fpr, tpr, roc_auc = get_roc_metrics(sem_real, sem_samples)
        p, r, pr_auc      = get_precision_recall_metrics(sem_real, sem_samples)
        acc, f1, threshold = get_accuracy_f1(sem_real, sem_samples)
        tpr_at_fprs = {f"tpr@fpr={t:.0%}": get_tpr_at_fpr(sem_real, sem_samples, t)
                       for t in TPR_AT_FPR_TARGETS}
                       
        print(f"\n⭐ Chiến lược [{strategy.upper()}]: AUROC={roc_auc:.4f}, AUPRC={pr_auc:.4f}")
        print(f"   Accuracy={acc:.4f}, F1={f1:.4f}, Threshold={threshold:.6f}")
        for name, val in tpr_at_fprs.items():
            print(f"   {name} = {val:.4f}")

        semantic_output = {
            'name':        f'semantic_curvature_{strategy}',
            'predictions': {'real': sem_real, 'samples': sem_samples},
            'metrics':     {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
            'pr_metrics':  {'pr_auc': pr_auc, 'precision': p, 'recall': r},
            'extra_metrics': {'accuracy': acc, 'f1': f1, **tpr_at_fprs},
            'pr_auc': pr_auc, 'loss': 1 - pr_auc,
        }
        all_outputs.append(semantic_output)
        
        sc_filename = f"semantic_curvature_{strategy}.json"
        with open(os.path.join(SAVE_FOLDER, sc_filename), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in semantic_output.items() if k != 'raw_results'}, f, indent=2)
        print(f"   💾 Saved: {sc_filename}")

    if perturbation_results:
        first = perturbation_results[0]
        if "semantic_curvature_sweep_original_per_layer" in first:
            layer_ids = sorted(first["semantic_curvature_sweep_original_per_layer"].keys())
            print(f"\n📊 Phân tích Per-Layer AUROC Sweep ({len(layer_ids)} layers):")
            layer_aurocs = {}
            for lid in layer_ids:
                lr = [r["semantic_curvature_sweep_original_per_layer"][lid] for r in perturbation_results]
                ls = [r["semantic_curvature_sweep_sampled_per_layer"][lid]  for r in perturbation_results]
                _, _, auroc = get_roc_metrics(lr, ls)
                layer_aurocs[lid] = auroc
                print(f"   Layer {lid:>3s}: AUROC = {auroc:.4f}")
            best = max(layer_aurocs, key=layer_aurocs.get)
            print(f"   🏆 Best layer: {best} (AUROC={layer_aurocs[best]:.4f})")
            
            with open(os.path.join(SAVE_FOLDER, "sweep_per_layer_auroc.json"), "w") as f:
                json.dump(layer_aurocs, f, indent=2)
            print(f"   💾 Saved: sweep_per_layer_auroc.json")

    print(f"\n✅ Hoàn thành! Kết quả tại: {SAVE_FOLDER}")