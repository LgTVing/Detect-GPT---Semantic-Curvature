# =============================================================================
# generate_all_figures.py — Tạo toàn bộ 12 hình minh họa cho báo cáo final
# Chạy trên Kaggle — Cần add 2 notebook outputs làm Input:
#   1. "detect-full-benchmark" (kết quả 500 mẫu, Pha 1 — có perturbation_raw.json)
#   2. "detect-scoring-multilayer" (kết quả Multi-Layer — có ckpt_score_progress.json)
# =============================================================================

# ─── Cell 1: Import ──────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import json
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

# ─── Font & Style Configuration ─────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})

# Output directory
SAVE_DIR = "/kaggle/working/figures"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📁 Figures sẽ lưu tại: {SAVE_DIR}")

# ─── Cell 2: Tự động tìm đường dẫn Input ─────────────────────────────────────
# Tìm kiếm trong cả /kaggle/working (chạy cùng lúc) và /kaggle/input (add data)
SEARCH_DIRS = [
    "/kaggle/working",
    "/kaggle/input/notebooks/lngthnhvinh/scoring",
    "/kaggle/input/notebooks/lngthnhvinh/hidden-layers",
    "/kaggle/input/notebooks/lngthnhvinh/detect-full-benchmark",
    "/kaggle/input"
]

def find_dir_containing(filename):
    """Tìm thư mục chứa file cụ thể."""
    for base_dir in SEARCH_DIRS:
        if not os.path.exists(base_dir): continue
        for root, dirs, files in os.walk(base_dir):
            if filename in files:
                return root
    return None

def load_json(base_dir, filename):
    if base_dir is None:
        print(f"⚠️  Không có base_dir để load {filename}")
        return None
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️  Không tìm thấy {filename} trong {base_dir}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("🔍 Đang tìm kiếm các thư mục kết quả...")
PHA1_DIR = find_dir_containing("detectgpt_25_d.json")
print(f"   Pha 1 (Baseline): {PHA1_DIR}")

PHA2_DIR = find_dir_containing("semantic_curvature_concat.json")
print(f"   Pha 2 (Multi-Layer): {PHA2_DIR}")

PHA3_DIR = find_dir_containing("semantic_curvature_concat_top4.json")
print(f"   Pha 3 (Top-4): {PHA3_DIR}")


# ─── Cell 3: Load dữ liệu ───────────────────────────────────────────────────

print("📂 Loading dữ liệu...")

# Pha 1: Kết quả cơ bản
pha1_ckpt = load_json(PHA1_DIR, "ckpt_score_progress.json")
sc_single = load_json(PHA1_DIR, "semantic_curvature.json")
dg_d      = load_json(PHA1_DIR, "detectgpt_25_d.json")
dg_z      = load_json(PHA1_DIR, "detectgpt_25_z.json")

# Pha 2: Multi-Layer results
pha2_ckpt    = load_json(PHA2_DIR, "ckpt_score_progress.json")
sc_last      = load_json(PHA2_DIR, "semantic_curvature_last.json")
sc_weighted  = load_json(PHA2_DIR, "semantic_curvature_weighted.json")
sc_concat    = load_json(PHA2_DIR, "semantic_curvature_concat.json")
sweep_auroc  = load_json(PHA2_DIR, "sweep_per_layer_auroc.json")

# Pha 3: Concat Top-4 Best Layers
sc_concat_top4 = load_json(PHA3_DIR, "semantic_curvature_concat_top4.json")

print("✅ Dữ liệu loaded.")


# ─── Cell 4: Helper Functions ────────────────────────────────────────────────

def get_auroc(real, samples):
    fpr, tpr, _ = roc_curve([0]*len(real) + [1]*len(samples), real + samples)
    return fpr, tpr, float(auc(fpr, tpr))

def get_tpr_at_fpr(real, samples, target_fpr):
    fpr_arr, tpr_arr, _ = roc_curve([0]*len(real) + [1]*len(samples), real + samples)
    idx = int(np.argmin(np.abs(np.array(fpr_arr) - target_fpr)))
    return float(tpr_arr[idx])

def get_optimal_threshold(real, samples):
    labels = [0]*len(real) + [1]*len(samples)
    scores = real + samples
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = int(np.argmax(np.array(tpr) - np.array(fpr)))
    return float(thresholds[idx])


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE A1: Đường cong ROC tổng hợp (tất cả phương pháp cơ bản)
# ═════════════════════════════════════════════════════════════════════════════

def fig_a1_roc_baseline():
    """ROC Curves cho tất cả phương pháp baseline + SC đơn tầng."""
    print("\n🖼️  Vẽ Figure A1: ROC Curves tổng hợp...")
    fig, ax = plt.subplots(figsize=(9, 7))

    methods = []
    # Load baseline methods nếu có
    for fname, label, color in [
        ("semantic_curvature.json",   "Semantic Curvature (Last)", "#E63946"),
        ("detectgpt_25_d.json",       "DetectGPT d-score",        "#457B9D"),
        ("detectgpt_25_z.json",       "DetectGPT z-score",        "#1D3557"),
    ]:
        data = load_json(PHA1_DIR, fname)
        if data and "metrics" in data:
            m = data["metrics"]
            ax.plot(m["fpr"], m["tpr"], label=f'{label}, AUROC={m["roc_auc"]:.3f}',
                    color=color, lw=2)

    # Multi-Layer methods
    for fname, label, color, ls in [
        ("semantic_curvature_last.json",     "SC Last (Multi-run)", "#2A9D8F", "-"),
        ("semantic_curvature_weighted.json", "SC Weighted",         "#E9C46A", "-"),
        ("semantic_curvature_concat.json",   "SC Concat (12k-D)",   "#F4A261", "-"),
    ]:
        data = load_json(PHA2_DIR, fname)
        if data and "metrics" in data:
            m = data["metrics"]
            ax.plot(m["fpr"], m["tpr"], label=f'{label}, AUROC={m["roc_auc"]:.3f}',
                    color=color, lw=2, linestyle=ls)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title('Đường cong ROC — So sánh toàn bộ các phương pháp\nGemma-3-4B-PT + ViT5-large | 1.000 mẫu',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_a1_roc_all_methods.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE A2–A5: Histogram phân bố điểm số (LL, d-score, z-score, SC)
# ═════════════════════════════════════════════════════════════════════════════

def fig_a2_a5_histograms():
    """4 Histogram phân bố điểm số AI vs Human."""
    print("\n🖼️  Vẽ Figure A2–A5: Histograms phân bố điểm số...")

    if pha1_ckpt is None:
        print("   ⚠️ Không có pha1_ckpt, bỏ qua.")
        return

    done = pha1_ckpt["done_results"]

    configs = [
        {
            "name": "fig_a2_histogram_likelihood",
            "human": [r["original_ll"] for r in done],
            "ai":    [r["sampled_ll"] for r in done],
            "xlabel": "Log-Likelihood",
            "title":  "Phân bố Log-Likelihood (AI vs Human)",
        },
        {
            "name": "fig_a3_histogram_dscore",
            "human": [r["original_ll"] - r["perturbed_original_ll"] for r in done],
            "ai":    [r["sampled_ll"] - r["perturbed_sampled_ll"] for r in done],
            "xlabel": "d-score = LL(x) − Mean(LL(x̃))",
            "title":  "Phân bố DetectGPT d-score (AI vs Human)",
        },
        {
            "name": "fig_a4_histogram_zscore",
            "human": [(r["original_ll"] - r["perturbed_original_ll"]) / max(r["perturbed_original_ll_std"], 1e-9)
                      for r in done],
            "ai":    [(r["sampled_ll"] - r["perturbed_sampled_ll"]) / max(r["perturbed_sampled_ll_std"], 1e-9)
                      for r in done],
            "xlabel": "z-score = d-score / Std(LL(x̃))",
            "title":  "Phân bố DetectGPT z-score (AI vs Human)",
        },
        {
            "name": "fig_a5_histogram_semantic_curvature",
            "human": [r["semantic_curvature_original"] for r in done],
            "ai":    [r["semantic_curvature_sampled"] for r in done],
            "xlabel": "Semantic Curvature = Mean(cosine_dist(v(x), v(x̃)))",
            "title":  "Phân bố Semantic Curvature (AI vs Human)",
        },
    ]

    for cfg in configs:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.hist(cfg["human"], bins=35, alpha=0.65, color='steelblue', label='Human (người viết)', edgecolor='white', linewidth=0.5)
        ax.hist(cfg["ai"],    bins=35, alpha=0.65, color='tomato',    label='AI (máy viết)',     edgecolor='white', linewidth=0.5)
        ax.set_xlabel(cfg["xlabel"], fontsize=11)
        ax.set_ylabel('Số lượng mẫu', fontsize=11)
        ax.set_title(cfg["title"], fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.annotate('→ Giá trị CAO hơn → dự đoán là AI', xy=(0.97, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))
        fig.tight_layout()
        path = os.path.join(SAVE_DIR, f"{cfg['name']}.png")
        fig.savefig(path); plt.close(fig)
        print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE B1: AUROC theo Layer (Layer Sweep Curve)
# ═════════════════════════════════════════════════════════════════════════════

def fig_b1_layer_sweep():
    """Biểu đồ AUROC vs Layer Index (1→34)."""
    print("\n🖼️  Vẽ Figure B1: AUROC theo Layer Sweep...")

    if sweep_auroc is None:
        print("   ⚠️ Không có sweep_per_layer_auroc.json, bỏ qua.")
        return

    layers = sorted(sweep_auroc.keys(), key=lambda x: int(x))
    layer_nums = [int(l) for l in layers]
    aurocs = [sweep_auroc[l] for l in layers]

    best_idx = np.argmax(aurocs)
    best_layer = layer_nums[best_idx]
    best_auroc = aurocs[best_idx]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(layer_nums, aurocs, 'o-', color='#2A9D8F', lw=2, markersize=6, label='AUROC per Layer')

    # Đánh dấu top 4
    top4_indices = np.argsort(aurocs)[-4:]
    for i in top4_indices:
        ax.plot(layer_nums[i], aurocs[i], 'o', color='#E63946', markersize=12, zorder=5)
        ax.annotate(f'L{layer_nums[i]}\n{aurocs[i]:.4f}',
                    xy=(layer_nums[i], aurocs[i]), xytext=(0, 15),
                    textcoords='offset points', ha='center', fontsize=8, fontweight='bold',
                    color='#E63946',
                    arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5))

    # Đường tham chiếu
    ax.axhline(y=0.9138, color='gray', linestyle=':', alpha=0.7, label=f'SC Last (Lớp 34) = 0.9138')
    ax.axhline(y=0.9284, color='#F4A261', linestyle='--', alpha=0.7, label=f'SC Concat (8,18,25,34) = 0.9284')

    ax.set_xlabel('Layer Index (Lớp ẩn trong Gemma-3-4B-PT)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('AUROC theo từng lớp ẩn (Layer Sweep)\nĐiểm đỏ = Top 4 Layer tốt nhất',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(layer_nums)
    ax.set_xticklabels(layer_nums, fontsize=8)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_ylim([0.65, 0.95])
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_b1_layer_sweep_auroc.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE B2: ROC Multi-Layer (Last vs Weighted vs Concat vs Sweep L25)
# ═════════════════════════════════════════════════════════════════════════════

def fig_b2_roc_multilayer():
    """ROC curves cho 4 chiến lược Multi-Layer."""
    print("\n🖼️  Vẽ Figure B2: ROC Multi-Layer strategies...")

    fig, ax = plt.subplots(figsize=(8, 7))

    strategies = [
        (sc_last,     "SC Last (Lớp 34)",             "#1D3557", "-",  2),
        (sc_weighted, "SC Weighted (4 lớp cuối)",      "#E9C46A", "-",  2),
        (sc_concat,   "SC Concat (8,18,25,34, 12k-D)", "#E63946", "-",  2.5),
    ]

    for data, label, color, ls, lw in strategies:
        if data and "metrics" in data:
            m = data["metrics"]
            ax.plot(m["fpr"], m["tpr"], label=f'{label}, AUROC={m["roc_auc"]:.4f}',
                    color=color, lw=lw, linestyle=ls)

    # Sweep Layer 25 (tính từ pha2_ckpt)
    if pha2_ckpt and sweep_auroc:
        done2 = pha2_ckpt["done_results"]
        best_layer = max(sweep_auroc, key=sweep_auroc.get)
        real_l25 = [r["semantic_curvature_sweep_original_per_layer"][best_layer] for r in done2]
        samp_l25 = [r["semantic_curvature_sweep_sampled_per_layer"][best_layer] for r in done2]
        fpr25, tpr25, auroc25 = get_auroc(real_l25, samp_l25)
        ax.plot(fpr25, tpr25, label=f'SC Sweep Layer {best_layer}, AUROC={auroc25:.4f}',
                color='#2A9D8F', lw=2.5, linestyle='--')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title('So sánh ROC — Các chiến lược Multi-Layer SC\nGemma-3-4B-PT | 1.000 mẫu',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_b2_roc_multilayer.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE B3: Histogram SC Multi-Layer (Concat)
# ═════════════════════════════════════════════════════════════════════════════

def fig_b3_histogram_concat():
    """Histogram SC Concat (12k-D)."""
    print("\n🖼️  Vẽ Figure B3: Histogram SC Concat...")

    if pha2_ckpt is None:
        print("   ⚠️ Không có pha2_ckpt, bỏ qua.")
        return

    done = pha2_ckpt["done_results"]
    human_vals = [r["semantic_curvature_concat_original"] for r in done]
    ai_vals    = [r["semantic_curvature_concat_sampled"]  for r in done]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(human_vals, bins=35, alpha=0.65, color='steelblue', label='Human (người viết)', edgecolor='white', linewidth=0.5)
    ax.hist(ai_vals,    bins=35, alpha=0.65, color='tomato',    label='AI (máy viết)',     edgecolor='white', linewidth=0.5)
    ax.set_xlabel('SC Concat (12.288 chiều)', fontsize=11)
    ax.set_ylabel('Số lượng mẫu', fontsize=11)
    ax.set_title('Phân bố Semantic Curvature — Chiến lược Concat (12k-D)\nAI vs Human | 1.000 mẫu',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_b3_histogram_sc_concat.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE B4: Bar chart AUROC tổng hợp
# ═════════════════════════════════════════════════════════════════════════════

def fig_b4_bar_auroc():
    """Biểu đồ cột AUROC tất cả phương pháp."""
    print("\n🖼️  Vẽ Figure B4: Bar chart AUROC tổng hợp...")

    # Thu thập AUROC
    methods = []
    aurocs = []

    # Baseline methods (hardcoded nếu không load được)
    baseline_data = {
        "Likelihood":  0.8670,
        "Rank":        0.8720,
        "Log-Rank":    0.8780,
        "Entropy":     0.2050,
        "d-score":     0.8520,
        "z-score":     0.7720,
    }
    # Try to load actual values
    for fname, key in [("detectgpt_25_d.json", "d-score"), ("detectgpt_25_z.json", "z-score")]:
        data = load_json(PHA1_DIR, fname)
        if data and "metrics" in data:
            baseline_data[key] = data["metrics"]["roc_auc"]

    for name, val in baseline_data.items():
        methods.append(name)
        aurocs.append(val)

    # SC methods
    for data, label in [
        (sc_last or sc_single, "SC Last"),
        (sc_weighted, "SC Weighted"),
        (sc_concat,   "SC Concat"),
    ]:
        if data and "metrics" in data:
            methods.append(label)
            aurocs.append(data["metrics"]["roc_auc"])

    # Sweep best
    if sweep_auroc:
        best_layer = max(sweep_auroc, key=sweep_auroc.get)
        methods.append(f"SC Sweep\nL{best_layer}")
        aurocs.append(sweep_auroc[best_layer])

    # Sort by AUROC
    pairs = sorted(zip(methods, aurocs), key=lambda x: x[1])
    methods, aurocs = zip(*pairs)

    # Color scheme
    colors = []
    for m in methods:
        if "SC" in m:
            colors.append('#E63946')
        elif m == "Entropy":
            colors.append('#CCCCCC')
        else:
            colors.append('#457B9D')

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(methods)), aurocs, color=colors, edgecolor='white', height=0.7)

    for i, (bar, val) in enumerate(zip(bars, aurocs)):
        ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('AUROC', fontsize=12)
    ax.set_title('So sánh AUROC — Tất cả phương pháp phát hiện\nGemma-3-4B-PT + ViT5-large | 1.000 mẫu',
                 fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.grid(axis='x', alpha=0.2)

    # Legend
    legend_patches = [
        mpatches.Patch(color='#457B9D', label='Baseline / DetectGPT'),
        mpatches.Patch(color='#E63946', label='Semantic Curvature (đề xuất)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_b4_bar_auroc_all_methods.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE C1: Sơ đồ Pipeline tổng thể
# ═════════════════════════════════════════════════════════════════════════════

def fig_c1_pipeline():
    """Sơ đồ Pipeline tổng thể dạng Flowchart."""
    print("\n🖼️  Vẽ Figure C1: Pipeline flowchart...")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    ax.axis('off')

    boxes = [
        (1, 2.5,   "Văn bản\ngốc x",           "#264653", "white"),
        (3.5, 2.5, "ViT5\nMasking\n& Fill",     "#2A9D8F", "white"),
        (6, 2.5,   "25 Dị bản\n{x̃₁,...,x̃₂₅}",  "#E9C46A", "black"),
        (8.5, 3.5, "Log-Likelihood\nLL(x), LL(x̃ᵢ)", "#F4A261", "black"),
        (8.5, 1.5, "Hidden States\nh₁,...,h₃₄",     "#E76F51", "white"),
        (11.5, 3.5, "d-score\nz-score",         "#457B9D", "white"),
        (11.5, 1.5, "Semantic\nCurvature",       "#E63946", "white"),
    ]

    for x, y, text, fc, tc in boxes:
        bbox = dict(boxstyle='round,pad=0.5', facecolor=fc, edgecolor='white', alpha=0.9)
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                fontweight='bold', color=tc, bbox=bbox)

    # Arrows
    arrows = [(2, 2.5, 2.5, 2.5), (4.5, 2.5, 5, 2.5), (7, 2.5, 7.5, 3.5),
              (7, 2.5, 7.5, 1.5), (9.5, 3.5, 10.5, 3.5), (9.5, 1.5, 10.5, 1.5)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#264653', lw=2))

    # Label
    ax.text(7.7, 2.5, "Gemma-3\n4B-PT", ha='center', va='center', fontsize=9,
            style='italic', color='#264653',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

    ax.set_title('Pipeline tổng thể: Phát hiện văn bản AI sinh bằng Semantic Curvature',
                 fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_c1_pipeline_overview.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE C2: Sơ đồ Masking → Prediction → Ghép lại
# ═════════════════════════════════════════════════════════════════════════════

def fig_c2_masking_illustration():
    """Minh hoạ quá trình Masking → ViT5 Fill → Perturbation."""
    print("\n🖼️  Vẽ Figure C2: Masking illustration...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), gridspec_kw={'hspace': 0.4})

    steps = [
        {
            "ax": axes[0],
            "title": "Bước 1: Câu gốc (Văn bản AI)",
            "parts": [
                ("Tổng Bí thư Đỗ Mười không chỉ là một ", "#F5F5F5", "black"),
                ("nhà lãnh đạo xuất sắc", "#FFE0B2", "black"),
                (" mà còn là biểu tượng của lòng kiên trung ", "#F5F5F5", "black"),
                ("và sự tận tụy", "#FFE0B2", "black"),
                (" với lý tưởng cách mạng...", "#F5F5F5", "black"),
            ],
        },
        {
            "ax": axes[1],
            "title": "Bước 2: Sau khi Masking (~15% từ bị che)",
            "parts": [
                ("Tổng Bí thư Đỗ Mười không chỉ là một ", "#F5F5F5", "black"),
                ("<extra_id_0>", "#FFCDD2", "#C62828"),
                (" mà còn là biểu tượng của lòng kiên trung ", "#F5F5F5", "black"),
                ("<extra_id_1>", "#FFCDD2", "#C62828"),
                (" với lý tưởng cách mạng...", "#F5F5F5", "black"),
            ],
        },
        {
            "ax": axes[2],
            "title": "Bước 3: ViT5 điền từ → Dị bản Perturbation",
            "parts": [
                ("Tổng Bí thư Đỗ Mười không chỉ là một ", "#F5F5F5", "black"),
                ("người lãnh đạo kiên cường", "#C8E6C9", "#1B5E20"),
                (" mà còn là biểu tượng của lòng kiên trung ", "#F5F5F5", "black"),
                ("và hết lòng cống hiến", "#C8E6C9", "#1B5E20"),
                (" với lý tưởng cách mạng...", "#F5F5F5", "black"),
            ],
        },
    ]

    for step in steps:
        ax = step["ax"]
        ax.axis('off')
        ax.set_title(step["title"], fontsize=11, fontweight='bold', loc='left')

        x = 0.02
        for text, bg, fg in step["parts"]:
            bbox = dict(boxstyle='round,pad=0.3', facecolor=bg, edgecolor='#BDBDBD', linewidth=0.5)
            ax.text(x, 0.5, text, transform=ax.transAxes, fontsize=9.5,
                    va='center', color=fg, bbox=bbox, fontfamily='serif')
            x += len(text) * 0.0065 + 0.01

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor='#FFE0B2', edgecolor='#BDBDBD', label='Cụm từ sẽ bị che'),
        mpatches.Patch(facecolor='#FFCDD2', edgecolor='#BDBDBD', label='Token mask <extra_id>'),
        mpatches.Patch(facecolor='#C8E6C9', edgecolor='#BDBDBD', label='ViT5 điền từ mới'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Minh hoạ quá trình Masking → ViT5 Prediction → Perturbation',
                 fontsize=14, fontweight='bold', y=1.02)
    path = os.path.join(SAVE_DIR, "fig_c2_masking_illustration.png")
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE C3: Sơ đồ kiến trúc Multi-Layer
# ═════════════════════════════════════════════════════════════════════════════

def fig_c3_multilayer_architecture():
    """Minh hoạ cách trích xuất hidden states từ nhiều lớp."""
    print("\n🖼️  Vẽ Figure C3: Multi-Layer architecture...")

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7)
    ax.axis('off')

    # Transformer stack
    layer_colors = {8: '#BBDEFB', 18: '#90CAF9', 25: '#42A5F5', 34: '#1565C0'}
    for i in range(1, 35):
        y = 0.5 + i * 0.16
        color = layer_colors.get(i, '#E3F2FD')
        alpha = 0.9 if i in layer_colors else 0.3
        rect = plt.Rectangle((1, y), 2.5, 0.14, facecolor=color, edgecolor='white',
                              alpha=alpha, linewidth=0.5)
        ax.add_patch(rect)
        if i in layer_colors:
            ax.text(2.25, y+0.07, f'Layer {i}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

    ax.text(2.25, 0.2, 'Gemma-3-4B-PT\n(34 Transformer Layers)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#264653')

    # Arrows to strategies
    strategies = [
        (5.5, 5.8, "Last\n(Lớp 34)", "v₃₄ ∈ ℝ³⁰⁷²", "#1565C0"),
        (7.5, 4.5, "Weighted\n(31–34)", "Σwᵢhᵢ ∈ ℝ³⁰⁷²", "#E9C46A"),
        (9.5, 3.2, "Sweep\n(Best: L25)", "h₂₅ ∈ ℝ³⁰⁷²", "#2A9D8F"),
        (11.5, 1.9, "Concat\n(8,18,25,34)", "[h₈‖h₁₈‖h₂₅‖h₃₄]\n∈ ℝ¹²²⁸⁸", "#E63946"),
    ]

    for x, y, title, formula, color in strategies:
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='white', alpha=0.9)
        ax.text(x, y, f'{title}\n{formula}', ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', bbox=bbox)
        # Arrow from stack to strategy
        ax.annotate('', xy=(x-0.8, y), xytext=(3.7, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, connectionstyle='arc3,rad=0.1'))

    ax.set_title('Kiến trúc Multi-Layer Semantic Curvature\n4 chiến lược trích xuất đặc trưng ngữ nghĩa',
                 fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_c3_multilayer_architecture.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE C4: Bar chart so sánh 200 mẫu vs 1000 mẫu
# ═════════════════════════════════════════════════════════════════════════════

def fig_c4_200_vs_1000():
    """Grouped bar chart so sánh AUROC 200 vs 1000 mẫu."""
    print("\n🖼️  Vẽ Figure C4: 200 vs 1000 mẫu...")

    # Dữ liệu từ báo cáo
    methods = ['LL', 'Rank', 'Log-Rank', 'Entropy', 'd-score', 'z-score', 'SC Last']
    auroc_200  = [0.8890, 0.8838, 0.8909, 0.1762, 0.8684, 0.8157, 0.9143]
    auroc_1000 = [0.8670, 0.8720, 0.8780, 0.2050, 0.8520, 0.7720, 0.9138]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars1 = ax.bar(x - width/2, auroc_200,  width, label='200 mẫu (sơ bộ)',  color='#457B9D', alpha=0.85)
    bars2 = ax.bar(x + width/2, auroc_1000, width, label='1.000 mẫu (chính thức)', color='#E63946', alpha=0.85)

    # Labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('So sánh kết quả: 200 mẫu (sơ bộ) vs 1.000 mẫu (chính thức)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_c4_200_vs_1000_comparison.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE D1: Confusion Matrix
# ═════════════════════════════════════════════════════════════════════════════

def fig_d1_confusion_matrix():
    """Confusion matrix tại ngưỡng tối ưu Youden's J cho SC Concat."""
    print("\n🖼️  Vẽ Figure D1: Confusion Matrix...")

    if sc_concat is None:
        print("   ⚠️ Không có sc_concat, bỏ qua.")
        return

    real = sc_concat["predictions"]["real"]
    samp = sc_concat["predictions"]["samples"]
    threshold = get_optimal_threshold(real, samp)

    labels = [0]*len(real) + [1]*len(samp)
    scores = real + samp
    preds = [1 if s >= threshold else 0 for s in scores]

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Người viết', 'AI sinh'], fontsize=11)
    ax.set_yticklabels(['Người viết', 'AI sinh'], fontsize=11)
    ax.set_xlabel('Dự đoán', fontsize=12)
    ax.set_ylabel('Thực tế', fontsize=12)

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=20, fontweight='bold', color=color)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    ax.set_title(f'Confusion Matrix — SC Concat (12k-D)\nAcc={acc:.2%}, F1={f1:.4f}, τ={threshold:.6f}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_d1_confusion_matrix.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE D4: TPR@FPR=5% bar chart
# ═════════════════════════════════════════════════════════════════════════════

def fig_d4_tpr_at_fpr5():
    """So sánh TPR@FPR=5% giữa các phương pháp."""
    print("\n🖼️  Vẽ Figure D4: TPR@FPR=5%...")

    methods_data = []

    for data_obj, label in [
        (sc_single or sc_last, "SC Last"),
        (sc_weighted, "SC Weighted"),
        (sc_concat,   "SC Concat"),
        (dg_d,        "d-score"),
        (dg_z,        "z-score"),
    ]:
        if data_obj and "predictions" in data_obj:
            real = data_obj["predictions"]["real"]
            samp = data_obj["predictions"]["samples"]
            tpr5 = get_tpr_at_fpr(real, samp, 0.05)
            methods_data.append((label, tpr5))

    if not methods_data:
        print("   ⚠️ Không đủ dữ liệu, bỏ qua.")
        return

    methods_data.sort(key=lambda x: x[1])
    names, vals = zip(*methods_data)

    colors = ['#E63946' if 'SC' in n else '#457B9D' for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='white', height=0.6)

    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(val + 0.01, i, f'{val:.2%}', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('TPR (True Positive Rate)', fontsize=12)
    ax.set_title('TPR tại FPR = 5% — Tỷ lệ phát hiện đúng AI khi chấp nhận 5% false alarm',
                 fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1.1])
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_d4_tpr_at_fpr5.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE B5: ROC so sánh Concat gốc vs Concat Top-4 Best Layers
# ═════════════════════════════════════════════════════════════════════════════

def fig_b5_roc_concat_top4():
    """ROC curve so sánh Concat (8,18,25,34) vs Concat Top-4 (18,19,21,25)."""
    print("\n🖼️  Vẽ Figure B5: ROC Concat vs Concat Top-4...")

    fig, ax = plt.subplots(figsize=(8, 7))

    # Concat gốc (8, 18, 25, 34)
    if sc_concat and "metrics" in sc_concat:
        m = sc_concat["metrics"]
        ax.plot(m["fpr"], m["tpr"],
                label=f'Concat gốc (8,18,25,34), AUROC={m["roc_auc"]:.4f}',
                color='#457B9D', lw=2)

    # Concat Top-4 (18, 19, 21, 25)
    if sc_concat_top4 and "metrics" in sc_concat_top4:
        m = sc_concat_top4["metrics"]
        top4_layers = sc_concat_top4.get("top4_layers", [18, 19, 21, 25])
        ax.plot(m["fpr"], m["tpr"],
                label=f'Concat Top-4 ({",".join(map(str,top4_layers))}), AUROC={m["roc_auc"]:.4f}',
                color='#E63946', lw=2.5)

    # Sweep Layer 25 (tham chiếu)
    if pha2_ckpt and sweep_auroc:
        done2 = pha2_ckpt["done_results"]
        best_layer = max(sweep_auroc, key=sweep_auroc.get)
        try:
            real_l = [r["semantic_curvature_sweep_original_per_layer"][best_layer] for r in done2]
            samp_l = [r["semantic_curvature_sweep_sampled_per_layer"][best_layer] for r in done2]
            fpr_l, tpr_l, auroc_l = get_auroc(real_l, samp_l)
            ax.plot(fpr_l, tpr_l, label=f'Sweep Layer {best_layer}, AUROC={auroc_l:.4f}',
                    color='#2A9D8F', lw=2, linestyle='--')
        except (KeyError, TypeError):
            pass

    # SC Last (tham chiếu)
    if sc_last and "metrics" in sc_last:
        m = sc_last["metrics"]
        ax.plot(m["fpr"], m["tpr"], label=f'SC Last (Lớp 34), AUROC={m["roc_auc"]:.4f}',
                color='#999999', lw=1.5, linestyle=':')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title('ROC — Concat gốc (đều) vs Concat Top-4 (tối ưu)\nGemma-3-4B-PT | 1.000 mẫu',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_b5_roc_concat_vs_top4.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE B6: Histogram SC Concat Top-4 (18, 19, 21, 25)
# ═════════════════════════════════════════════════════════════════════════════

def fig_b6_histogram_concat_top4():
    """Histogram phân bố SC Concat Top-4 (AI vs Human)."""
    print("\n🖼️  Vẽ Figure B6: Histogram SC Concat Top-4...")

    if sc_concat_top4 is None or "predictions" not in sc_concat_top4:
        print("   ⚠️ Không có sc_concat_top4, bỏ qua.")
        return

    human_vals = sc_concat_top4["predictions"]["real"]
    ai_vals    = sc_concat_top4["predictions"]["samples"]
    top4_layers = sc_concat_top4.get("top4_layers", [18, 19, 21, 25])
    auroc = sc_concat_top4["metrics"]["roc_auc"]

    # Tính ngưỡng tối ưu
    threshold = get_optimal_threshold(human_vals, ai_vals)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(human_vals, bins=35, alpha=0.65, color='steelblue',
            label='Human (người viết)', edgecolor='white', linewidth=0.5)
    ax.hist(ai_vals,    bins=35, alpha=0.65, color='tomato',
            label='AI (máy viết)', edgecolor='white', linewidth=0.5)
    ax.axvline(threshold, color='#2D6A4F', linestyle='--', lw=2,
               label=f'Ngưỡng tối ưu = {threshold:.6f}')

    ax.set_xlabel(f'SC Concat Top-4 ({",".join(map(str,top4_layers))})', fontsize=11)
    ax.set_ylabel('Số lượng mẫu', fontsize=11)
    ax.set_title(f'Phân bố Semantic Curvature — Concat Top-4 Best Layers\n'
                 f'Layers {top4_layers} | AUROC = {auroc:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.annotate('→ Giá trị CAO hơn → dự đoán là AI', xy=(0.97, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig_b6_histogram_sc_concat_top4.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾 {path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN: Chạy tất cả
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎨 GENERATE ALL FIGURES FOR FINAL REPORT")
    print("="*70)

    # Nhóm A: Histograms + ROC (từ dữ liệu Pha 1)
    fig_a1_roc_baseline()
    fig_a2_a5_histograms()

    # Nhóm B: Multi-Layer analysis
    fig_b1_layer_sweep()
    fig_b2_roc_multilayer()
    fig_b3_histogram_concat()
    fig_b4_bar_auroc()

    # Nhóm B (Top-4): Concat Top-4 Best Layers
    fig_b5_roc_concat_top4()
    fig_b6_histogram_concat_top4()

    # Nhóm C: Sơ đồ kiến trúc
    fig_c1_pipeline()
    fig_c2_masking_illustration()
    fig_c3_multilayer_architecture()
    fig_c4_200_vs_1000()

    # Nhóm D: Supplementary
    fig_d1_confusion_matrix()
    fig_d4_tpr_at_fpr5()

    print("\n" + "="*70)
    n_files = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.png')])
    print(f"✅ Hoàn thành! Tổng cộng {n_files} hình đã lưu tại: {SAVE_DIR}")
    print("="*70)
    print("\nDanh sách file:")
    for f in sorted(os.listdir(SAVE_DIR)):
        if f.endswith('.png'):
            print(f"   📊 {f}")
