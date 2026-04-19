import os, json, time
from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType
from utils import codegraph_to_nxgraph

BASE = "/teamspace/studios/this_studio/travail/CodeFuse-CGM"
GRAPH_DATA_PATH = "/teamspace/studios/this_studio/travail/data/swe-bench-lite"
SUBGRAPH_DICT_PATH = f"{BASE}/retriever/subgraph_nodes.json"
SAVE_DIR = f"{BASE}/retriever/subgraph"
os.makedirs(SAVE_DIR, exist_ok=True)

PRINT_EVERY_SEC = 1
MAX_INNER_NODES_PER_FILE = 50000  # cap to prevent explosions

def should_skip(out_path: str) -> bool:
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return False
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False

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

def serialize_subgraph(graph_nx, out_json_path):
    node_list = [n.to_dict() for n in graph_nx.nodes() if n is not None]
    edge_list = []
    for a, b in graph_nx.edges():
        et = graph_nx[a][b][0]["type"]
        edge_list.append({"edgeType": et.name.lower(), "source": a.node_id, "target": b.node_id})
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": node_list, "edges": edge_list}, f)

def safe_get_node(graph, node_id):
    try:
        return graph.get_node_by_id(node_id)
    except Exception:
        return None

def get_contained_node(graph_nx, node):
    out = []
    for suc in graph_nx.successors(node):
        if graph_nx[node][suc][0]["type"] == EdgeType.CONTAINS:
            out.append(suc)
    return out

def get_inner_nodes_capped(graph_nx, node, cap=50000):
    inner = get_contained_node(graph_nx, node)
    all_inner = []
    seen = set()
    while inner:
        tmp = inner
        inner = []
        for nd in tmp:
            if nd in seen:
                continue
            seen.add(nd)
            all_inner.append(nd)
            if len(all_inner) >= cap:
                return list(set(all_inner))
            inner.extend(get_contained_node(graph_nx, nd))
    return list(set(all_inner))

def main():
    graph_idx = build_graph_index(GRAPH_DATA_PATH)
    subgraph_nodes_dict = json.load(open(SUBGRAPH_DICT_PATH, "r", encoding="utf-8"))
    instance_ids = sorted(subgraph_nodes_dict.keys())

    # missing-only
    missing = []
    for iid in instance_ids:
        out_path = os.path.join(SAVE_DIR, f"{iid}.json")
        if not should_skip(out_path):
            missing.append(iid)

    print("Total instances:", len(instance_ids))
    print("Already done:", len(instance_ids) - len(missing))
    print("Missing:", len(missing))

    cache = {}  # graph_file -> (graph, graph_nx)

    t0 = time.time()
    last_print = t0

    for idx_i, instance_id in enumerate(missing, 1):
        out_path = os.path.join(SAVE_DIR, f"{instance_id}.json")
        repo_prefix = instance_to_repo_prefix(instance_id)
        if repo_prefix not in graph_idx:
            continue

        graph_file = graph_idx[repo_prefix][0]
        graph_path = os.path.join(GRAPH_DATA_PATH, graph_file)
        if not os.path.exists(graph_path):
            continue

        if graph_file not in cache:
            g = parse(graph_path)
            gnx = codegraph_to_nxgraph(g)
            cache[graph_file] = (g, gnx)
        else:
            g, gnx = cache[graph_file]

        file_node_ids = subgraph_nodes_dict.get(instance_id, [])
        if not file_node_ids:
            serialize_subgraph(gnx.subgraph([]), out_path)
            continue

        all_node_ids = set()
        for nid in file_node_ids:
            n = safe_get_node(g, nid)
            if n is None:
                continue
            all_node_ids.add(n.node_id)
            if n.get_type() == NodeType.FILE:
                inner = get_inner_nodes_capped(gnx, n, cap=MAX_INNER_NODES_PER_FILE)
                for inner_node in inner:
                    if inner_node is not None:
                        all_node_ids.add(inner_node.node_id)

        node_objs = [safe_get_node(g, nid) for nid in all_node_ids]
        node_objs = [x for x in node_objs if x is not None]
        subg = gnx.subgraph(list(node_objs))
        serialize_subgraph(subg, out_path)

        now = time.time()
        if now - last_print >= PRINT_EVERY_SEC:
            speed = idx_i / max(now - t0, 1e-6)
            print(f"progress {idx_i}/{len(missing)} ({speed:.2f} inst/s) | current={instance_id}", flush=True)
            last_print = now

    print("DONE. Files in subgraph/:", len([f for f in os.listdir(SAVE_DIR) if f.endswith('.json')]))

if __name__ == "__main__":
    main()