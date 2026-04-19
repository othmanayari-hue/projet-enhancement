import os, json, re
import numpy as np

# ===================== PATHS (Lightning) =====================
PRED_DIR = "/teamspace/studios/this_studio/CodeFuse-CGM/reranker/reranker_outputs/stage_2_5/relevant_files"
GROUND_TRUTH_PATH = "/teamspace/studios/this_studio/ground_truth.json"
# ============================================================

def normalize_path(p: str) -> str:
    return p.replace("\\", "/").strip()

def is_match(pred_file: str, gold_file: str) -> bool:
    pred_file = normalize_path(pred_file)
    gold_file = normalize_path(gold_file)
    return pred_file.endswith(gold_file) or gold_file.endswith(pred_file)

def load_predictions(pred_dir: str):
    preds = {}
    if not os.path.exists(pred_dir):
        raise FileNotFoundError(pred_dir)

    files = [f for f in os.listdir(pred_dir) if f.endswith(".json")]
    print("Prediction json files found:", len(files))
    print("Sample files:", files[:5])

    for fn in files:
        instance_id = fn[:-5]
        path = os.path.join(pred_dir, fn)
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            continue

        # Stage-2 format: {"relevant_file_score": {...}, "selected_relevant_files": [...]}
        if isinstance(data, dict) and "selected" in data:
            preds[instance_id] = data["selected"]
        # Stage-1 format: list[str]
        elif isinstance(data, list):
            preds[instance_id] = data
        else:
            # unknown format
            continue

    return preds

def load_ground_truth(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return json.load(open(path, "r", encoding="utf-8"))

def eval_metrics(preds: dict, gt: dict, k: int):
    keys = sorted(set(preds.keys()) & set(gt.keys()))
    if not keys:
        return {"n": 0, f"Recall@{k}": 0.0, f"Precision@{k}": 0.0, f"MRR@{k}": 0.0}

    recalls, precisions, mrrs = [], [], []

    for iid in keys:
        gold = gt[iid]
        pred = (preds.get(iid, []) or [])[:k]

        gold_norm = [normalize_path(x) for x in gold]
        pred_norm = [normalize_path(x) for x in pred]

        hits = 0
        first_hit_rank = 0

        for rank, pf in enumerate(pred_norm, start=1):
            hit = any(is_match(pf, gf) for gf in gold_norm)
            if hit:
                hits += 1
                if first_hit_rank == 0:
                    first_hit_rank = rank

        recalls.append(hits / max(len(gold_norm), 1))
        precisions.append(hits / k)
        mrrs.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

    return {
        "n": len(keys),
        f"Recall@{k}": float(np.mean(recalls)),
        f"Precision@{k}": float(np.mean(precisions)),
        f"MRR@{k}": float(np.mean(mrrs)),
    }

def main():
    print("Loading predictions...")
    preds = load_predictions(PRED_DIR)
    print("Loaded predictions:", len(preds))

    if preds:
        sample_id = next(iter(preds.keys()))
        print("Sample pred instance:", sample_id)
        print("Sample pred files:", preds[sample_id])

    print("\nLoading ground truth...")
    gt = load_ground_truth(GROUND_TRUTH_PATH)
    print("Loaded ground truth:", len(gt))

    m1 = eval_metrics(preds, gt, k=1)
    m5 = eval_metrics(preds, gt, k=5)

    print("\n==============================")
    print("       EVALUATION RESULTS     ")
    print("==============================")
    print("Evaluated instances (intersection):", m5["n"])
    print(f"Recall@1    : {m1['Recall@1']:.2%}")
    print(f"Recall@5    : {m5['Recall@5']:.2%}")
    print("-" * 30)
    print(f"Precision@1 : {m1['Precision@1']:.2%}")
    print(f"Precision@5 : {m5['Precision@5']:.2%}")
    print("-" * 30)
    print(f"MRR@5       : {m5['MRR@5']:.4f}")
    print("==============================")

if __name__ == "__main__":
    main()