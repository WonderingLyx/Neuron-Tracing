import re
from collections import deque

def parse_log(log_file_path):
    """解析log文件，提取节点父子关系（修复adding匹配问题）"""
    # 正则表达式：匹配Current coord（提取三个整数坐标）
    current_coord_pattern = re.compile(r'Current coord: \[(\d+), (\d+), (\d+)\]')
    # 匹配adding后的完整内容（从[开始到最后一个]结束，使用贪婪匹配）
    adding_pattern = re.compile(r'adding: (\[\[.*\]\])')  # 关键修正：匹配整个嵌套列表
    # 匹配单个子节点坐标（[x,y,z]）
    child_coord_pattern = re.compile(r'\[(\d+), (\d+), (\d+)\]')
    
    node_relations = {}  # 存储 {当前节点坐标: 子节点坐标列表}
    
    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 只处理包含"Current coord"和"adding"的行
            if "Current coord" in line and "adding" in line:
                # 提取当前节点坐标
                current_match = current_coord_pattern.search(line)
                if not current_match:
                    continue  # 跳过格式错误的行
                x, y, z = map(int, current_match.groups())
                current_coord = (x, y, z)
                
                # 提取adding后的完整子节点列表（修复匹配逻辑）
                adding_match = adding_pattern.search(line)
                if not adding_match:
                    continue  # 跳过没有子节点的行
                children_str = adding_match.group(1)  # 获取[[...], [...]]部分
                
                # 解析每个子节点坐标
                children = []
                for child_match in child_coord_pattern.finditer(children_str):
                    cx, cy, cz = map(int, child_match.groups())
                    children.append((cx, cy, cz))
                
                # 存储关系
                node_relations[current_coord] = children
    
    return node_relations

def find_root_node(node_relations):
    """找到根节点（没有父节点的节点）"""
    current_nodes = set(node_relations.keys())
    child_nodes = set()
    for children in node_relations.values():
        child_nodes.update(children)
    # 根节点是当前节点中不在子节点集合中的节点
    root_candidates = current_nodes - child_nodes
    if len(root_candidates) != 1:
        raise ValueError(f"无法确定唯一根节点，候选：{root_candidates}")
    return root_candidates.pop()

def generate_swc_from_log(log_file_path, output_swc_path):
    """从log文件生成SWC文件"""
    node_relations = parse_log(log_file_path)
    if not node_relations:
        raise ValueError("log文件中未找到有效节点信息")
    
    root_coord = find_root_node(node_relations)
    
    # 坐标到ID的映射和SWC节点列表
    coord_to_id = {}
    swc_nodes = []
    id_counter = 1
    
    # 处理根节点
    rx, ry, rz = root_coord
    coord_to_id[root_coord] = id_counter
    swc_nodes.append((id_counter, 1, rx, ry, rz, 1.0, -1))
    id_counter += 1
    
    # 层次遍历处理所有子节点
    queue = deque()
    queue.append((root_coord, 1))  # (当前节点坐标, 父节点ID)
    
    while queue:
        current_coord, parent_id = queue.popleft()
        if current_coord in node_relations:
            for child in node_relations[current_coord]:
                child_id = id_counter
                coord_to_id[child] = child_id
                cx, cy, cz = child
                swc_nodes.append((child_id, 3, cx, cy, cz, 1.0, parent_id))
                id_counter += 1
                queue.append((child, child_id))
    
    # 写入SWC文件
    with open(output_swc_path, 'w') as f:
        for node in swc_nodes:
            f.write(f"{node[0]} {node[1]} {node[2]} {node[3]} {node[4]} {node[5]} {node[6]}\n")
    
    print(f"SWC文件已生成：{output_swc_path}")

if __name__ == "__main__":
    # 输入log文件路径（请替换为实际的log文件路径）
    log_file = "/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Results/Predict/process/trace_log.log"
    # 输出SWC文件路径
    output_swc = "/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Results/Predict/process/output.swc"
    
    # 生成SWC文件
    generate_swc_from_log(log_file, output_swc)