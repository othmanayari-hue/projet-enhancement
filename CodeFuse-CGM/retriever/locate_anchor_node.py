"""
Anchor node localization (Lightning version, safe, works with partial PKLs).
- Extractor: rapidfuzz fuzzy match
- Inferer: faiss dense retrieval
- SAFE: if embedding dims mismatch, it skips inferer for that instance instead of crashing.

Processes instance_ids from NODE_EMBEDDING_DIR (*.pkl).
Resolves graph file by repo prefix from flat graph filenames owner#repo#sha.graph.json.

Output: retriever/anchor_node.json
"""

import os, json, pickle, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import process, fuzz
import faiss

from codegraph_parser.python.codegraph_python_local import parse, NodeType

# ===================== PATHS (EDIT IF NEEDED) =====================
GRAPH_DATA_PATH = "/teamspace/studios/this_studio/travail/data/swe-bench-lite"

# Node embeddings downloaded from HF (check actual folder; common is tmp_node_embedding/tmp_node_embedding/*.pkl)
NODE_EMBEDDING_DIR = "/teamspace/studios/this_studio/travail/CodeFuse-CGM/preprocess_embedding/tmp_node_embedding/tmp_node_embedding"

REWRITER_OUTPUT_PATH = "/teamspace/studios/this_studio/travail/CodeFuse-CGM/rewriter/test_rewriter_output.json"
QUERY_EMBEDDING_PATH = "/teamspace/studios/this_studio/travail/CodeFuse-CGM/rewriter/rewriter_embedding.pkl"

OUT_PATH = "/teamspace/studios/this_studio/travail/CodeFuse-CGM/retriever/anchor_node.json"
# ================================================================

# Params
EXTRACT_LIMIT = 3
INFER_TOPK = 15
SAVE_EVERY = 1  # save after each instance (safe)

def extract_info(item):
    return item[1]

def build_graph_index(graph_dir: str):
    """
    Map repo_prefix (owner__repo) -> list of graph files
    """
    idx = {}
    files = [f for f in os.listdir(graph_dir) if f.endswith(".json")]
    for f in files:
        parts = f.split("#")
        if len(parts) < 3:
            continue
        owner, repo = parts[0], parts[1]
        key = f"{owner}__{repo}"
        idx.setdefault(key, []).append(f)
    for k in idx:
        idx[k].sort()
    return idx

def instance_to_repo_prefix(instance_id: str):
    # instance_id like astropy__astropy-6938 -> astropy__astropy
    if "__" not in instance_id:
        return None
    return instance_id.split("-", 1)[0]

def get_extractor_anchor(graph, entity_query, keywords_query):
    all_nodes = graph.get_nodes()

    cand_name_list = []
    cand_path_name_list = []

    for node in all_nodes:
        node_type = node.get_type()
        if node_type in [NodeType.REPO, NodeType.PACKAGE]:
            continue

        if not hasattr(node, "name"):
            continue

        cand_name_list.append((node.node_id, node.name))

        if node_type == NodeType.FILE:
            if getattr(node, "path", None):
                name_with_path = node.path + "/" + node.name
            else:
                name_with_path = node.name
            cand_path_name_list.append((node.node_id, name_with_path))

    res = set()
    full_queries = (entity_query or []) + (keywords_query or [])
    if not full_queries:
        return res

    for query in full_queries:
        if not isinstance(query, str) or not query.strip():
            continue

        if "/" in query:
            cand_path_name = process.extract(
                (-1, query),
                cand_path_name_list,
                scorer=fuzz.WRatio,
                limit=EXTRACT_LIMIT,
                processor=extract_info,
            )
            for item in cand_path_name:
                res.add(item[0][0])

        query_wo_path = query.split("/")[-1]
        cand_name = process.extract(
            (-1, query_wo_path),
            cand_name_list,
            scorer=fuzz.WRatio,
            limit=EXTRACT_LIMIT,
            processor=extract_info,
        )
        for item in cand_name:
            res.add(item[0][0])

    return res

def get_inferer_anchor(query_emb, node_embedding, k=15):
    raw = node_embedding.get("code", {})
    if not raw:
        return []

    node_ids = list(raw.keys())
    cand_vec = np.stack([raw[nid] for nid in node_ids], axis=0).astype("float32")

    q = np.array(query_emb, dtype="float32")
    if q.ndim == 1:
        q = q.reshape(1, -1)

    # SAFE dim check
    if q.shape[1] != cand_vec.shape[1]:
        print(f"[WARN] Dim mismatch: query {q.shape[1]} vs nodes {cand_vec.shape[1]} -> skip inferer")
        return []

    d = cand_vec.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(cand_vec)

    k = min(k, cand_vec.shape[0])
    if k <= 0:
        return []

    _, I = index.search(q, k)
    # flatten first query
    return [int(node_ids[i]) for i in I[0]]

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    if not os.path.exists(NODE_EMBEDDING_DIR):
        raise FileNotFoundError(f"NODE_EMBEDDING_DIR not found: {NODE_EMBEDDING_DIR}")
    if not os.path.exists(GRAPH_DATA_PATH):
        raise FileNotFoundError(f"GRAPH_DATA_PATH not found: {GRAPH_DATA_PATH}")

    pkl_files = sorted([f for f in os.listdir(NODE_EMBEDDING_DIR) if f.endswith(".pkl")])
    instance_ids = [f[:-4] for f in pkl_files]
    print("Instances (from PKLs):", len(instance_ids))

    graph_idx = build_graph_index(GRAPH_DATA_PATH)

    with open(REWRITER_OUTPUT_PATH, "r", encoding="utf-8") as f:
        rewriter_output = json.load(f)

    with open(QUERY_EMBEDDING_PATH, "rb") as f:
        query_embedding = pickle.load(f)

    out = {}

    for n, instance_id in enumerate(tqdm(instance_ids, desc="anchor_node"), 1):
        if instance_id not in query_embedding:
            continue

        repo_prefix = instance_to_repo_prefix(instance_id)
        if repo_prefix not in graph_idx:
            continue

        graph_file = graph_idx[repo_prefix][0]
        graph_path = os.path.join(GRAPH_DATA_PATH, graph_file)

        node_pkl = os.path.join(NODE_EMBEDDING_DIR, f"{instance_id}.pkl")
        with open(node_pkl, "rb") as f:
            node_embedding = pickle.load(f)

        query_emb = query_embedding[instance_id]

        # parse graph
        graph = parse(graph_path)

        entity_query = rewriter_output.get(instance_id, {}).get("code_entity", [])
        keyword_query = rewriter_output.get(instance_id, {}).get("keyword", [])

        res_extractor = get_extractor_anchor(graph, entity_query, keyword_query)
        res_inferer = get_inferer_anchor(query_emb, node_embedding, k=INFER_TOPK)

        out[instance_id] = {
            "extractor_anchor_nodes": list(res_extractor),
            "inferer_anchor_nodes": list(res_inferer),
            "graph_file_used": graph_file,
        }

        if SAVE_EVERY and (n % SAVE_EVERY == 0):
            with open(OUT_PATH, "w", encoding="utf-8") as wf:
                json.dump(out, wf)

    with open(OUT_PATH, "w", encoding="utf-8") as wf:
        json.dump(out, wf)

    print("Saved:", OUT_PATH, "| instances:", len(out))

if __name__ == "__main__":
    main()