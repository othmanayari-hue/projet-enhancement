import json
import re
import os
import pandas as pd
import argparse

from qwen_api import QwenAPI
from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType

stage_1_system_prompt = """
You are an experienced software developer who specializes in extracting the most relevant files for solving issues from many reference files.

Task:
Based on the information received about the issue from a repository, find the most likely few files from among those that may be able to resolve the issue.

Instructions:
1. Analysis:
- Analyze the provided issue description and files, and pay attention to the relevance of the provided files with the given issue, especially those might be modified during fixing the issue.
- Determine the specific problem or error mentioned in the issue and note any clues that could help your judgment.
2. Extraction:
- Based on your analysis, choose the Top **{}** relevant files which might be used in fixing the issue.
- You should choose files from the provided files, and should not modify their name in any way.

Respond in the following format:
[start_of_analysis]
<detailed_analysis> 
[end_of_analysis] 

[start_of_relevant_files] 
1. <file_with_its_path>
2. <file_with_its_path>
3. ...
[end_of_relevant_files] 

Notes:
- You can refer to to the information in the error logs (if exists).
- The relevant file usually exists in the project described in the issue (e.g., django, sklearn). File need modification is usually not in the tests files or external packages.
- The file you choose should be contained in the provided files.
- Provide the file path with files. Do not include redundant suffix like '/home/username/', '/etc/service/' or '/tree/master'.
- Do not include any additional information such as line numbers or explanations in your extraction result.
- Files for initialization and configuration might be modified during changing the code.

Preferred extraction Examples of Related Files:
1. src/utils/file_handler.py
2. core/services/service_manager.py
3. ...
""".strip()

stage_1_user_prompt_template = """
<repository>
{}
</repository>

<issue>
{}
</issue>
 
<reference_python_file_list>
{}
</reference_python_file_list>

<other_reference_file_list>
{}
</other_reference_file_list>
"""

stage_2_system_prompt_v3 = """
You are an experienced software developer who specializes in assessing the relevance of the file for solving the issue in software repositories.

Task:
For a file provided, evaluate the likelihood that modifying this file would resolve the given issue, and assign a score based on specific criteria.

Instructions:
1. Analysis:
- Analyze the provided issue description and the content of the single relevant file, pay attention to any keywords, error messages, or specific functionalities mentioned that relate to the file.
- Determine how closely the contents and functionality of the file are tied to the problem or error described in the issue.
- Consider the role of the file in the overall project structure (e.g., configuration files, core logic files versus test files, or utility scripts).
2. Scoring:
- Based on your analysis, assign a score from 1 to 5 that represents the relevance of modifying the given file in order to solve the issue.

Score Specifications:
1. **Score 1**: The file is almost certainly unrelated to the issue, with no apparent connection to the functionality or error described in the issue.
2. **Score 2**: The file may be tangentially related, but modifying it is unlikely to resolve the issue directly; possible in rare edge cases.
3. **Score 3**: The file has some relevance to the issue; it might interact with the affected functionality indirectly and tweaking it could be part of a broader fix.
4. **Score 4**: The file is likely related to the issue; it includes code that interacts directly with the functionality in question and could plausibly contain bugs that lead to the issue.
5. **Score 5**: The file is very likely the root cause or heavily involved in the issue and modifying it should directly address the error or problem mentioned.

Respond in the following format:
[start_of_analysis]
<detailed_analysis>
[end_of_analysis]

[start_of_score]
Score <number>
[end_of_score]

Notes:
- The content of the file shows only the structure of this file, including the names of the classes and functions defined in this file.
- You can refer to to the information in the error logs (if exists).
""".strip()

stage_2_user_prompt_template = """
<repository>
{}
</repository>

<issue>
{}
</issue>

<file_name>
{}
</file_name>

<file_content>
{}
</file_content>
"""


# ===================== HYBRID RERANKER =====================

def graph_distance_score(graph, node, max_depth=4):
    if node is None:
        return 0.0
    visited = set()
    queue = [(node.node_id, 0)]
    while queue:
        nid, dist = queue.pop(0)
        if dist > max_depth:
            break
        if nid in visited:
            continue
        visited.add(nid)
        current = graph.get_node_by_id(nid)
        if current.get_type() == NodeType.FUNCTION:
            return 1 / (1 + dist)
        for nxt in graph.get_out_nodes(nid):
            queue.append((nxt, dist + 1))
    return 0.0


def heuristic_score(file_name):
    score = 0
    name = file_name.lower()
    if "test" in name:
        score -= 1
    if "doc" in name:
        score -= 0.5
    if "example" in name:
        score -= 0.5
    if "core" in name:
        score += 0.3
    return score


# ===================== UTILS =====================

def get_python_inner_class_and_function(graph, node_id, layer_cnt=0):
    ret_list = []
    if layer_cnt > 5:
        return ret_list
    node = graph.get_node_by_id(node_id)
    inner_node_ids = graph.get_out_nodes(node_id)
    for inner_node_id in inner_node_ids:
        inner_node = graph.get_node_by_id(inner_node_id)
        if inner_node.get_type() == NodeType.FUNCTION and "def " + inner_node.name in node.text:
            ret_list.append((layer_cnt, inner_node))
            ret_list.extend(get_python_inner_class_and_function(graph, inner_node.node_id, layer_cnt + 1))
        elif inner_node.get_type() == NodeType.CLASS and "class " + inner_node.name in node.text:
            ret_list.append((layer_cnt, inner_node))
            ret_list.extend(get_python_inner_class_and_function(graph, inner_node.node_id, layer_cnt + 1))
    return ret_list


def parse_reranker_stage_1(response):
    pattern = r"\[start_of_relevant_files\]\s*(.*?)\s*\[end_of_relevant_files\]"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        pattern = r"<start_of_relevant_files>\s*(.*?)\s*<end_of_relevant_files>"
        match = re.search(pattern, response, re.DOTALL)
    relevant_files = match.group(1).strip().split("\n") if match else []
    for idx, relevant_file in enumerate(relevant_files):
        new_relevant_file = relevant_file
        if new_relevant_file.startswith("- "):
            new_relevant_file = new_relevant_file[2:]
        pattern = r"\d+ *\.(.+)"
        match = re.search(pattern, new_relevant_file)
        if match:
            new_relevant_file = match.group(1).strip()
        relevant_files[idx] = new_relevant_file
    return relevant_files


def parse_reranker_stage_2(response):
    pattern = r"\[start_of_score\]\s*(.*?)\s*\[end_of_score\]"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        pattern = r"<start_of_score>\s*(.*?)\s*<end_of_score>"
        match = re.search(pattern, response, re.DOTALL)
    score = match.group(1).strip().split("\n")[0] if match else "0"
    if score.startswith("- "):
        score = score[2:]
    match = re.search(r"Score (\d+)", score)
    return int(match.group(1)) if match else 0


def extract_files_from_subgraph(subgraph_path, output_path):
    subgraph_list = os.listdir(subgraph_path)
    for subgraph in subgraph_list:
        if not subgraph.endswith(".json"):
            continue
        try:
            with open(os.path.join(subgraph_path, subgraph), "r", encoding="utf-8") as f:
                subgraph_json = json.load(f)
        except:
            continue
        subgraph_nodes = subgraph_json.get("nodes", [])
        file_nodes = [node for node in subgraph_nodes if node.get("nodeType") == "File"]
        pred_files = []
        for node in file_nodes:
            file_path = node.get("filePath")
            file_name = node.get("fileName")
            file = file_name if file_path is None else os.path.join(file_path, file_name)
            pred_files.append(file)
        subgraph_name = subgraph.split(".")[0]
        with open(os.path.join(output_path, subgraph_name + ".json"), "w", encoding="utf-8") as f:
            json.dump(pred_files, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Reranker.")
    parser.add_argument('--stage_1_k', type=int, default=10)
    parser.add_argument('--stage_2_k', type=int, default=5)
    return parser.parse_args()


# ===================== MAIN =====================

if __name__ == "__main__":
    llm = QwenAPI("Qwen/Qwen2.5-Coder-32B-Instruct")
    output_dir = "/teamspace/studios/this_studio/CodeFuse-CGM/reranker/reranker_outputs"
    subgraph_file_dir = "/teamspace/studios/this_studio/CodeFuse-CGM/retriever/subgraph"

    retriever_filtered_files_dir = "/teamspace/studios/this_studio/CodeFuse-CGM/reranker/subgraph_extracted_files/"
    os.makedirs(retriever_filtered_files_dir, exist_ok=True)
    extract_files_from_subgraph(subgraph_file_dir, retriever_filtered_files_dir)

    df = pd.read_json("/teamspace/studios/this_studio/CodeFuse-CGM/preprocess_embedding/test_lite_basic_info.json", orient="index")
    args = parse_args()

    # ------------------ STAGE 1 ------------------
    stage_1_output_dir = os.path.join(output_dir, f"stage_1_top_{args.stage_1_k}")
    os.makedirs(os.path.join(stage_1_output_dir, "relevant_files"), exist_ok=True)
    os.makedirs(os.path.join(stage_1_output_dir, "response"), exist_ok=True)
    stage_1_system_prompt = stage_1_system_prompt.format(args.stage_1_k)

    for i, data in df.iterrows():
        repo = data["repo"]
        instance_id = data["instance_id"]
        problem_statement = data["problem_statement"]

        filtered_file_path = os.path.join(retriever_filtered_files_dir, instance_id + ".json")
        if not os.path.exists(filtered_file_path):
            continue

        stage_1_relevant_out = os.path.join(stage_1_output_dir, "relevant_files", instance_id + ".json")
        stage_1_response_out = os.path.join(stage_1_output_dir, "response", instance_id + ".txt")
        if os.path.exists(stage_1_relevant_out) and os.path.exists(stage_1_response_out):
            print(f"Stage 1 skipped {instance_id}")
            continue

        with open(filtered_file_path, "r") as f:
            filtered_files = json.load(f)

        python_files = [item for item in filtered_files if item.endswith(".py")]
        other_files = [item for item in filtered_files if not item.endswith(".py")]
        user_prompt = stage_1_user_prompt_template.format(repo, problem_statement, "\n".join(python_files), "\n".join(other_files))
        response = llm.get_response(stage_1_system_prompt, user_prompt)
        relevant_files = parse_reranker_stage_1(response)

        with open(stage_1_relevant_out, "w") as f:
            json.dump(relevant_files, f, indent=4)
        with open(stage_1_response_out, "w", encoding="utf-8") as f:
            f.write(response)

    # ------------------ STAGE 2 ------------------
    stage_2_output_dir = os.path.join(output_dir, f"stage_2_{args.stage_2_k}")
    os.makedirs(os.path.join(stage_2_output_dir, "relevant_files"), exist_ok=True)
    os.makedirs(os.path.join(stage_2_output_dir, "response"), exist_ok=True)

    for i, data in df.iterrows():
        repo = data["repo"]
        instance_id = data["instance_id"]
        problem_statement = data["problem_statement"]

        stage_1_file = os.path.join(stage_1_output_dir, "relevant_files", instance_id + ".json")
        if not os.path.exists(stage_1_file):
            continue

        subgraph_path = os.path.join(subgraph_file_dir, instance_id + ".json")
        if not os.path.exists(subgraph_path):
            continue

        stage_2_relevant_out = os.path.join(stage_2_output_dir, "relevant_files", instance_id + ".json")
        stage_2_response_out = os.path.join(stage_2_output_dir, "response", instance_id + ".json")
        if os.path.exists(stage_2_relevant_out) and os.path.exists(stage_2_response_out):
            print(f"Stage 2 skipped {instance_id}")
            continue

        with open(stage_1_file, "r") as f:
            stage_1_files = json.load(f)

        graph = parse(subgraph_path)

        scores = {}
        responses = {}

        for file in stage_1_files:
            found_node = None
            content = ""
            for node in graph.get_nodes_by_type(NodeType.FILE):
                full = os.path.join(node.path or "", node.name).replace("\\", "/")
                if full == file:
                    found_node = node
                    class_and_functions = get_python_inner_class_and_function(graph, node.node_id)
                    for layer, n in class_and_functions:
                        if n.get_type() == NodeType.CLASS:
                            content += "    " * layer + "class " + n.name + "\n"
                        elif n.get_type() == NodeType.FUNCTION and n.name != "<anonymous>":
                            content += "    " * layer + "def " + n.name + "\n"
                    break

            if not content:
                scores[file] = 0
                responses[file] = ""
                continue

            user_prompt = stage_2_user_prompt_template.format(repo, problem_statement, file, content)
            response = llm.get_response(stage_2_system_prompt_v3, user_prompt)

            # ------------------ HYBRID SCORING ------------------
            llm_score = parse_reranker_stage_2(response)
            if llm_score == 3:
                llm_score -= 0.3
            g_score = graph_distance_score(graph, found_node)
            h_score = heuristic_score(file)
            final_score = 0.75 * llm_score + 0.2 * g_score * 5 + 0.05 * h_score
            scores[file] = final_score
            responses[file] = response

        # ------------------ ADAPTIVE TOP-K ------------------
        sorted_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        k = args.stage_2_k
        if len(sorted_files) >= k:
            top_scores = [x[1] for x in sorted_files[:k]]
            if max(top_scores) - min(top_scores) < 0.5:
                k = min(len(sorted_files), k + 2)
        selected = [x[0] for x in sorted_files[:k]]

        with open(stage_2_relevant_out, "w") as f:
            json.dump({"scores": scores, "selected": selected}, f, indent=4)
        with open(stage_2_response_out, "w") as f:
            json.dump(responses, f, indent=4)

        print(f"Done {instance_id}")