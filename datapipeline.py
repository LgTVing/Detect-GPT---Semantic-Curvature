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

print(f"🔄 Loading scoring_tokenizer for exact length filtering: {SCORING_MODEL_NAME}")
scoring_tokenizer = transformers.AutoTokenizer.from_pretrained(
    SCORING_MODEL_NAME, cache_dir=CACHE_DIR
)
if scoring_tokenizer.pad_token is None:
    scoring_tokenizer.pad_token = scoring_tokenizer.eos_token

print(f"🔄 Loading mask-filling model: {MASK_FILLING_MODEL_NAME} ➔ {DEVICE_MASK}")
mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    MASK_FILLING_MODEL_NAME,
    torch_dtype=get_dtype(MASK_MODEL_DTYPE),
    cache_dir=CACHE_DIR,
)
mask_model = mask_model.to(DEVICE_MASK).eval()
import tokenizers
if hasattr(tokenizers.models, "Unigram"):
    _orig_unigram = tokenizers.models.Unigram
    def _patched_unigram(vocab=None, *args, **kwargs):
        if isinstance(vocab, dict):
            vocab = [(k, 0.0) for k in vocab.keys()]
        return _orig_unigram(vocab, *args, **kwargs)
    tokenizers.models.Unigram = _patched_unigram

mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
    MASK_FILLING_MODEL_NAME, model_max_length=MAX_TOKENS, cache_dir=CACHE_DIR
)
print(f"✅ Mask model loaded")

def strip_newlines(text):
    """Remove newlines and extra whitespaces from text."""
    
    return ' '.join(text.split())

def load_vietnamese_dataset(n_samples=N_SAMPLES):
    """Load and sample real and AI-generated texts from the dataset."""
    
    print(f"⏳ Loading dataset: {DATASET_NAME}")
    ds = hf_datasets.load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)
    total = len(ds)
    mid   = total // 2

    n_each = n_samples // 2
    buffer = min(n_each * 4, mid)

    random.seed(RANDOM_SEED)
    ai_indices  = random.sample(range(0, mid), buffer)
    hum_indices = random.sample(range(mid, total), buffer)

    ai_subset  = ds.select(ai_indices)
    hum_subset = ds.select(hum_indices)
    print(f"   Đã chọn {buffer} mẫu AI và {buffer} mẫu Người (từ tổng {total:,})")

    ai_texts, hum_texts = [], []
    
    for row in ai_subset:
        t = strip_newlines(row["Text"])
        tok_len = len(scoring_tokenizer.encode(t, truncation=False))
        if tok_len <= MAX_TOKENS:
            ai_texts.append(t)
            if len(ai_texts) >= n_each:
                break

    for row in hum_subset:
        t = strip_newlines(row["Text"])
        tok_len = len(scoring_tokenizer.encode(t, truncation=False))
        if tok_len <= MAX_TOKENS:
            hum_texts.append(t)
            if len(hum_texts) >= n_each:
                break

    print(f"✅ Dataset ready: {len(hum_texts)} Người + {len(ai_texts)} AI")
    return {
        "original": hum_texts,   # văn bản người (label=0, như paper gốc gọi là "original")
        "sampled":  ai_texts,    # văn bản AI   (label=1, như paper gốc gọi là "sampled")
    }

def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    """Tokenize text and randomly replace words with mask tokens."""
    
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'
    n_spans = pct * len(tokens) / (span_length + BUFFER_SIZE * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, max(1, len(tokens) - span_length))
        end   = start + span_length
        search_start = max(0, start - BUFFER_SIZE)
        search_end   = min(len(tokens), end + BUFFER_SIZE)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks
    return ' '.join(tokens)

def count_masks(texts):
    """Count the number of mask tokens present in a text."""
    
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

_case_study_log  = []        # Danh sách case study entries đã thu thập
_CASE_STUDY_LIMIT = 5        # Chỉ log chi tiết cho N mẫu đầu tiên
_case_study_sample_idx = 0   # Index mẫu gốc đang xử lý (cập nhật từ get_perturbation_results)

def replace_masks(texts, original_tokens_list=None, original_texts_list=None):
    """Replace mask tokens in text with model-generated fills."""
    
    global _case_study_log, _case_study_sample_idx

    n_expected = count_masks(texts)
    stop_id    = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens     = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE_MASK)

    do_case_study = (_case_study_sample_idx < _CASE_STUDY_LIMIT) and (original_tokens_list is not None)

    if do_case_study:
        gen_out = mask_model.generate(
            **tokens, max_length=150, do_sample=True,
            top_p=MASK_TOP_P, num_return_sequences=1, eos_token_id=stop_id,
            output_scores=True, return_dict_in_generate=True
        )
        decoded = mask_tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=False)

        TOP_K = 5
        for batch_idx, (masked_text, orig_tokens) in enumerate(zip(texts, original_tokens_list or [])):
            orig_text = (original_texts_list or [])[batch_idx] if original_texts_list and batch_idx < len(original_texts_list) else ""
            top_k_steps = []
            for step_scores in gen_out.scores:
                probs = torch.softmax(step_scores[batch_idx], dim=-1)
                topk_probs, topk_ids = torch.topk(probs, TOP_K)
                top_k_steps.append([
                    {
                        "token": mask_tokenizer.decode([int(tid)]).strip(),
                        "prob":  round(float(p), 4)
                    }
                    for tid, p in zip(topk_ids.tolist(), topk_probs.tolist())
                ])

            top1_hits, top5_hits, total = 0, 0, 0
            for mask_pos_idx, orig_tok in enumerate(orig_tokens or []):
                if mask_pos_idx >= len(top_k_steps):
                    break
                step_preds = top_k_steps[mask_pos_idx]
                pred_tokens_top1 = [step_preds[0]["token"]] if step_preds else []
                pred_tokens_top5 = [s["token"] for s in step_preds]
                total += 1
                if orig_tok in pred_tokens_top1:
                    top1_hits += 1
                if orig_tok in pred_tokens_top5:
                    top5_hits += 1

            entry = {
                "sample_idx":              _case_study_sample_idx,
                "original_text":           orig_text,
                "masked_text":             masked_text,
                "filled_text":             decoded[batch_idx] if batch_idx < len(decoded) else "",
                "original_tokens":         orig_tokens,
                "top_k_predictions":       top_k_steps[:len(orig_tokens or [])],
                "top1_accuracy":           round(top1_hits / total, 4) if total > 0 else None,
                "top5_accuracy":           round(top5_hits / total, 4) if total > 0 else None,
            }
            _case_study_log.append(entry)

        return decoded
    else:
        outputs = mask_model.generate(
            **tokens, max_length=150, do_sample=True,
            top_p=MASK_TOP_P, num_return_sequences=1, eos_token_id=stop_id
        )
        return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    """Extract filled text segments generated by the mask-filling model."""
    
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    """Apply extracted fills back into the original masked text."""
    
    tokens = [x.split(' ') for x in masked_texts]
    n_expected = count_masks(masked_texts)
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
    return [" ".join(x) for x in tokens]

def perturb_texts_(texts, span_length, pct, ceil_pct=False, original_tokens_list=None, original_texts_list=None):
    """Core function to generate perturbed text variants using the mask-filling model."""
    
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills    = replace_masks(masked_texts, original_tokens_list=original_tokens_list,
                                 original_texts_list=original_texts_list)
    extracted    = extract_fills(raw_fills)
    perturbed    = apply_extracted_fills(masked_texts, extracted)
    MAX_ATTEMPTS = 10  # Giới hạn số lần retry để tránh treo vô hạn
    attempts = 1
    while '' in perturbed and attempts < MAX_ATTEMPTS:
        idxs = [i for i, x in enumerate(perturbed) if x == '']
        print(f'⚠️  {len(idxs)} văn bản không có fills. Thử lại [lần {attempts}/{MAX_ATTEMPTS}].')
        masked_texts_retry = [tokenize_and_mask(texts[i], span_length, pct, ceil_pct) for i in idxs]
        raw_fills_retry    = replace_masks(masked_texts_retry)  # retry không cần log
        new_perturbed      = apply_extracted_fills(masked_texts_retry, extract_fills(raw_fills_retry))
        for i, x in zip(idxs, new_perturbed):
            if x:
                perturbed[i] = x
        attempts += 1
    for i, x in enumerate(perturbed):
        if x == '':
            perturbed[i] = texts[i]
            print(f'⚠️  Fallback: giữ nguyên câu gốc tại vị trí {i} sau {MAX_ATTEMPTS} lần thử.')
    return perturbed

def perturb_texts(texts, span_length, pct, ceil_pct=False, original_texts_for_log=None):
    """Generate multiple perturbed variants for a list of texts (batch processing)."""
    
    global _case_study_sample_idx
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), CHUNK_SIZE), desc="Applying perturbations"):
        chunk = texts[i:i + CHUNK_SIZE]
        orig_tokens_chunk = None
        if original_texts_for_log is not None and _case_study_sample_idx < _CASE_STUDY_LIMIT:
            orig_tokens_chunk = [
                [w for w in original_texts_for_log[i + j].split()
                 if w not in texts[i + j].split()]  # từ có trong gốc nhưng bị thay/mask
                for j in range(len(chunk))
                if i + j < len(original_texts_for_log)
            ]
        outputs.extend(perturb_texts_(chunk, span_length, pct, ceil_pct=ceil_pct,
                                      original_tokens_list=orig_tokens_chunk,
                                      original_texts_list=original_texts_for_log[i:i+len(chunk)] if original_texts_for_log else None))
        _case_study_sample_idx += 1
    return outputs

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

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📝 [DATAPIPELINE] Tạo Perturbation Texts...")
    print("="*60)
    
    data = load_vietnamese_dataset(N_SAMPLES)
    if SAVE_RAW_RESULTS:
        with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    perturb_fn = functools.partial(perturb_texts, span_length=SPAN_LENGTH, pct=PCT_WORDS_MASKED)
    
    checkpoint_data = ckpt_load(CHECKPOINT_PERTURB)
    if checkpoint_data is None:
        checkpoint_data = {"p_sampled": [], "p_original": []}
        start_idx = 0
    else:
        start_idx = len(checkpoint_data["p_sampled"]) // N_PERTURBATIONS
        print(f"\n⏭️  Phục hồi Phase 1 từ checkpoint: đã tạo perturbations cho {start_idx}/{len(data['original'])} mẫu.")

    BATCH_SIZE = 10  # Xử lý mỗi lần 10 mẫu (tương đương 10 * N_PERTURBATIONS = 500 bản sao) để lưu checkpoint

    for i in tqdm.tqdm(range(start_idx, len(data["original"]), BATCH_SIZE), desc="Perturbing in batches",
                       initial=start_idx // BATCH_SIZE, total=(len(data["original"]) + BATCH_SIZE - 1) // BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(data["original"]))
        
        batch_sampled = data["sampled"][i:end_idx]
        batch_original = data["original"][i:end_idx]
        
        sampled_expanded  = [x for x in batch_sampled   for _ in range(N_PERTURBATIONS)]
        original_expanded = [x for x in batch_original  for _ in range(N_PERTURBATIONS)]
        
        if i == 0:
            p_sampled_text  = perturb_fn(sampled_expanded,  original_texts_for_log=sampled_expanded)
            p_original_text = perturb_fn(original_expanded, original_texts_for_log=original_expanded)
        else:
            p_sampled_text  = perturb_fn(sampled_expanded)
            p_original_text = perturb_fn(original_expanded)
            
        checkpoint_data["p_sampled"].extend(p_sampled_text)
        checkpoint_data["p_original"].extend(p_original_text)
        
        ckpt_save(CHECKPOINT_PERTURB, checkpoint_data)
        
        if i == 0 and _case_study_log:
            with open(os.path.join(SAVE_FOLDER, "case_study_log.json"), "w", encoding="utf-8") as f:
                json.dump(_case_study_log, f, ensure_ascii=False, indent=2)
            
    print(f"\n✅ Hoàn tất Data Pipeline! Perturbation texts đã lưu. Hãy chuyển sang chạy scoring.py")