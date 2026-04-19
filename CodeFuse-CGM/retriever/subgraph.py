"""
Subgraph generation (Lightning version, partial-PKL friendly) - FIXED None nodes

Inputs:
- anchor_node.json
- flat graph files owner#repo#sha.graph.json
- node PKLs folder (to know which instance_ids)

Output:
- subgraph_nodes.json : dict[instance_id] -> list[file_node_id]
"""

import os
import json
from tqdm import tqdm

from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType
from utils import codegraph_to_nxgraph

# ===================== PATHS (EDIT IF NEEDED) =====================
GRAPH_DATA_PATH = "/teamspace/studios/this_studio/travail/data/swe-bench-lite"
NODE_EMBEDDING_DIR = "/teamspace/studios/this_studio/travail/CodeFuse-CGM/preprocess_embedding/tmp_node_embedding/tmp_node_embedding"
ANCHOR_NODE_PATH = "/teamspace/studios/this_studio/travail/CodeFuse-CGM/retriever/anchor_node.json"
OUT_PATH = "/teamspace/studios/this_studio/travail/CodeFuse-CGM/retriever/subgraph_nodes.json"
# ================================================================

def build_graph_index(graph_dir: str):
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
    if "__" not in instance_id:
        return None
    return instance_id.split("-", 1)[0]

def get_path_to_repo(node, pre_node_dict, graph_nx):
    if node is None:
        return []
    if node.get_type() == NodeType.REPO:
        return [node]

    if node.node_id in pre_node_dict:
        return pre_node_dict[node.node_id]

    pre_nodes = []
    for pre_node in graph_nx.predecessors(node):
        if graph_nx[pre_node][node][0]["type"] == EdgeType.CONTAINS:
            pre_nodes.append(pre_node)
            if pre_node.get_type() != NodeType.REPO:
                pre_nodes.extend(get_path_to_repo(pre_node, pre_node_dict, graph_nx))
            break

    pre_node_dict[node.node_id] = pre_nodes
    return pre_nodes

def reconstruct_graph(subgraph_nodes, graph_nx, pre_node_dict):
    nodes = [n for n in subgraph_nodes if n is not None]
    all_nodes = set(nodes)
    for node in nodes:
        pre_nodes = get_path_to_repo(node, pre_node_dict, graph_nx)
        all_nodes |= set([n for n in pre_nodes if n is not None])
    return graph_nx.subgraph(list(all_nodes))

def bfs_expand_file(graph_nx, seed_nodes, hops=2):
    seed = [n for n in seed_nodes if n is not None]
    visited = set()
    nhops = set([n.node_id for n in seed])

    for _ in range(hops):
        nxt = []
        for node in seed:
            if node is None:
                continue
            if node.node_id in visited:
                continue
            visited.add(node.node_id)

            for nb in graph_nx.successors(node):
                if nb is None:
                    continue
                if nb.get_type() == NodeType.FILE:
                    nxt.append(nb)
                nhops.add(nb.node_id)

            for nb in graph_nx.predecessors(node):
                if nb is None:
                    continue
                if nb.get_type() == NodeType.FILE:
                    nxt.append(nb)
                nhops.add(nb.node_id)

        seed = nxt

    return nhops

def safe_get_node(graph, node_id):
    try:
        n = graph.get_node_by_id(node_id)
        return n
    except Exception:
        return None

def main():
    if not os.path.exists(ANCHOR_NODE_PATH):
        raise FileNotFoundError(f"Missing {ANCHOR_NODE_PATH}")
    if not os.path.exists(GRAPH_DATA_PATH):
        raise FileNotFoundError(f"Missing {GRAPH_DATA_PATH}")
    if not os.path.exists(NODE_EMBEDDING_DIR):
        raise FileNotFoundError(f"Missing {NODE_EMBEDDING_DIR}")

    pkl_files = sorted([f for f in os.listdir(NODE_EMBEDDING_DIR) if f.endswith(".pkl")])
    instance_ids = [f[:-4] for f in pkl_files]
    print("Instances (from PKLs):", len(instance_ids))

    with open(ANCHOR_NODE_PATH, "r", encoding="utf-8") as f:
        anchor_node_dict = json.load(f)

    graph_idx = build_graph_index(GRAPH_DATA_PATH)
    subgraph_id_dict = {}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    for instance_id in tqdm(instance_ids, desc="subgraph"):
        if instance_id not in anchor_node_dict:
            continue

        repo_prefix = instance_to_repo_prefix(instance_id)
        if repo_prefix not in graph_idx:
            continue

        graph_file = graph_idx[repo_prefix][0]
        graph_path = os.path.join(GRAPH_DATA_PATH, graph_file)

        graph = parse(graph_path)
        graph_nx = codegraph_to_nxgraph(graph)

        anchors_raw = anchor_node_dict[instance_id]
        extractor_anchors = anchors_raw.get("extractor_anchor_nodes", [])
        inferer_anchors = anchors_raw.get("inferer_anchor_nodes", [])
        anchor_ids = list(set(extractor_anchors + inferer_anchors))

        anchor_nodes = [safe_get_node(graph, nid) for nid in anchor_ids]
        anchor_nodes = [n for n in anchor_nodes if n is not None]

        if not anchor_nodes:
            subgraph_id_dict[instance_id] = []
            continue

        expanded_ids = bfs_expand_file(graph_nx, anchor_nodes, hops=2)

        expanded_nodes = [safe_get_node(graph, nid) for nid in expanded_ids]
        expanded_nodes = [n for n in expanded_nodes if n is not None]

        pre_node_dict = {}
        subgraph = reconstruct_graph(expanded_nodes, graph_nx, pre_node_dict)

        file_node_ids = [n.node_id for n in subgraph.nodes() if n is not None and n.get_type() == NodeType.FILE]
        subgraph_id_dict[instance_id] = list(set(file_node_ids))

        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(subgraph_id_dict, f)

    print("Saved:", OUT_PATH, "| instances:", len(subgraph_id_dict))

if __name__ == "__main__":
    main()