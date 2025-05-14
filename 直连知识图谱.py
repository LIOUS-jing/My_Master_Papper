import pandas as pd
import numpy as np
import re
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyvis.network import Network
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置输出路径
output_dir = "D:\\AI CODE\\情感分析\\知识图谱"
os.makedirs(output_dir, exist_ok=True)

# 设置TOP数量
TOP_N = 15

def read_csv_file():
    """读取CSV文件"""
    file_path = "C:\\Users\\15058\\OneDrive - 南京大学\\桌面\\数据处理文件\\知识图谱.csv"
    try:
        # 尝试不同编码读取
        for encoding in ['utf-8', 'gbk', 'latin1', 'utf-8-sig']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用{encoding}编码读取文件")
                return df
            except UnicodeDecodeError:
                continue
        print("无法读取文件，请检查文件路径和编码")
        return None
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None

def extract_relationships(df):
    """提取并解析关系数据"""
    # 假设列名为A、B、C、D
    relationships = {
        'space_to_cognition': [],  # 空间特征->认知
        'cognition_to_emotion': [], # 认知->情感
        'emotion_to_behavior': [],  # 情感->行为
        'space_to_emotion': []      # 空间特征->情感
    }
    
    # 正则表达式提取[A]->[B]格式的内容
    pattern = r'\[(.*?)\]\s*->\s*\[(.*?)\]'
    
    # 分别解析每一列
    for col_idx, category in enumerate(['space_to_cognition', 'cognition_to_emotion', 
                                       'emotion_to_behavior', 'space_to_emotion']):
        if col_idx < len(df.columns):
            col_data = df.iloc[:, col_idx].dropna()
            for item in col_data:
                matches = re.findall(pattern, str(item))
                for match in matches:
                    if len(match) == 2:
                        source, target = match
                        relationships[category].append((source.strip(), target.strip()))
    
    # 打印关系统计信息
    for category, items in relationships.items():
        print(f"{category} 关系数量: {len(items)}")
    
    return relationships

def get_top_entities(relationships, top_n=TOP_N):
    """获取各类型实体中频次最高的TOP_N个"""
    # 提取各类型的实体
    spaces = []
    cognitions = []
    emotions = []
    behaviors = []
    
    # 从不同关系中提取节点
    for source, target in relationships['space_to_cognition']:
        spaces.append(source)
        cognitions.append(target)
    
    for source, target in relationships['cognition_to_emotion']:
        cognitions.append(source)
        emotions.append(target)
    
    for source, target in relationships['emotion_to_behavior']:
        emotions.append(source)
        behaviors.append(target)
    
    for source, target in relationships['space_to_emotion']:
        spaces.append(source)
        emotions.append(target)
    
    # 计算各实体出现频率
    space_counts = Counter(spaces)
    cognition_counts = Counter(cognitions)
    emotion_counts = Counter(emotions)
    behavior_counts = Counter(behaviors)
    
    # 获取TOP_N的实体
    top_spaces = dict(space_counts.most_common(top_n))
    top_cognitions = dict(cognition_counts.most_common(top_n))
    top_emotions = dict(emotion_counts.most_common(top_n))
    top_behaviors = dict(behavior_counts.most_common(top_n))
    
    # 打印TOP实体统计信息
    print(f"\nTOP {top_n} 空间特征:")
    for item, count in top_spaces.items():
        print(f"  - {item}: {count}次")
    
    print(f"\nTOP {top_n} 认知评价:")
    for item, count in top_cognitions.items():
        print(f"  - {item}: {count}次")
    
    print(f"\nTOP {top_n} 情感类型:")
    for item, count in top_emotions.items():
        print(f"  - {item}: {count}次")
    
    print(f"\nTOP {top_n} 行为意图:")
    for item, count in top_behaviors.items():
        print(f"  - {item}: {count}次")
    
    top_entities = {
        'space': top_spaces,
        'cognition': top_cognitions,
        'emotion': top_emotions,
        'behavior': top_behaviors
    }
    
    return top_entities

def filter_top_relationships(relationships, top_entities, min_weight=2):
    """筛选TOP实体之间的关系，并过滤低频关系"""
    # 首先计算每对关系的频次
    relation_counts = {
        'space_to_cognition': Counter(relationships['space_to_cognition']),
        'cognition_to_emotion': Counter(relationships['cognition_to_emotion']),
        'emotion_to_behavior': Counter(relationships['emotion_to_behavior']),
        'space_to_emotion': Counter(relationships['space_to_emotion'])
    }
    
    # 筛选TOP实体之间且频次>=min_weight的关系
    top_relationships = {
        'space_to_cognition': [],
        'cognition_to_emotion': [],
        'emotion_to_behavior': [],
        'space_to_emotion': []
    }
    
    # 筛选空间->认知关系
    for (source, target), count in relation_counts['space_to_cognition'].items():
        if source in top_entities['space'] and target in top_entities['cognition'] and count >= min_weight:
            top_relationships['space_to_cognition'].append((source, target, count))
    
    # 筛选认知->情感关系
    for (source, target), count in relation_counts['cognition_to_emotion'].items():
        if source in top_entities['cognition'] and target in top_entities['emotion'] and count >= min_weight:
            top_relationships['cognition_to_emotion'].append((source, target, count))
    
    # 筛选情感->行为关系
    for (source, target), count in relation_counts['emotion_to_behavior'].items():
        if source in top_entities['emotion'] and target in top_entities['behavior'] and count >= min_weight:
            top_relationships['emotion_to_behavior'].append((source, target, count))
    
    # 筛选空间->情感关系
    for (source, target), count in relation_counts['space_to_emotion'].items():
        if source in top_entities['space'] and target in top_entities['emotion'] and count >= min_weight:
            top_relationships['space_to_emotion'].append((source, target, count))
    
    # 打印TOP关系统计信息
    for category, items in top_relationships.items():
        print(f"TOP{TOP_N}实体间 {category} 关系数量(频次≥{min_weight}): {len(items)}")
    
    return top_relationships

def create_layered_direct_graph(top_entities, top_relationships, output_path):
    """创建分层布局的直线连接静态图谱图像"""
    plt.figure(figsize=(24, 20))
    
    # 使用YlGnBu颜色映射
    cmap = plt.cm.YlGnBu
    
    # 定义节点类型的颜色
    color_map = {
        'space': '#1f77b4',    # 蓝色
        'cognition': '#ff7f0e', # 黄色/橙色
        'emotion': '#2ca02c',   # 绿色
        'behavior': '#d62728'   # 红色
    }
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    for category, nodes in top_entities.items():
        for node, weight in nodes.items():
            G.add_node(node, type=category, weight=weight)
    
    # 添加边
    for rel_type, edges in top_relationships.items():
        for source, target, weight in edges:
            G.add_edge(source, target, type=rel_type, weight=weight)
    
    # 计算每个关系类型的最大权重，用于归一化
    max_weights = {}
    for rel_type, edges in top_relationships.items():
        if edges:
            max_weights[rel_type] = max(weight for _, _, weight in edges)
        else:
            max_weights[rel_type] = 1
    
    # 计算节点权重的最大值，用于归一化节点大小
    category_max_weights = {
        'space': max(top_entities['space'].values()),
        'cognition': max(top_entities['cognition'].values()),
        'emotion': max(top_entities['emotion'].values()),
        'behavior': max(top_entities['behavior'].values())
    }
    
    # 创建分层布局
    # 分配每个节点的位置
    pos = {}
    
    # 获取各类型节点列表
    space_nodes = [(node, data['weight']) for node, data in G.nodes(data=True) if data['type'] == 'space']
    cognition_nodes = [(node, data['weight']) for node, data in G.nodes(data=True) if data['type'] == 'cognition']
    emotion_nodes = [(node, data['weight']) for node, data in G.nodes(data=True) if data['type'] == 'emotion']
    behavior_nodes = [(node, data['weight']) for node, data in G.nodes(data=True) if data['type'] == 'behavior']
    
    # 根据节点权重排序
    space_nodes.sort(key=lambda x: x[1], reverse=True)
    cognition_nodes.sort(key=lambda x: x[1], reverse=True)
    emotion_nodes.sort(key=lambda x: x[1], reverse=True)
    behavior_nodes.sort(key=lambda x: x[1], reverse=True)
    
    # 为每层分配位置，增大层间距
    layer_height = 1.5  # 增大层间距
    
    # 空间特征节点 - 最上层
    space_y = 3 * layer_height
    spacing_factor = 1.5  # 增大节点间的水平间距
    for i, (node, _) in enumerate(space_nodes):
        # 增加水平间距，防止标签重叠
        pos[node] = (spacing_factor * (i - len(space_nodes)/2), space_y)
    
    # 认知评价节点 - 第二层
    cognition_y = 2 * layer_height
    for i, (node, _) in enumerate(cognition_nodes):
        # 交错排列认知节点，使相邻节点水平位置错开
        offset = 0.5 if i % 2 == 1 else 0  # 奇数索引的节点向右偏移0.5
        pos[node] = (spacing_factor * (i - len(cognition_nodes)/2) + offset, cognition_y)
    
    # 情感类型节点 - 第三层
    emotion_y = 1 * layer_height
    for i, (node, _) in enumerate(emotion_nodes):
        # 交错排列情感节点
        offset = 0.5 if i % 2 == 0 else 0  # 偶数索引的节点向右偏移0.5
        pos[node] = (spacing_factor * (i - len(emotion_nodes)/2) + offset, emotion_y)
    
    # 行为意图节点 - 最下层
    behavior_y = 0
    for i, (node, _) in enumerate(behavior_nodes):
        # 增加水平间距
        pos[node] = (spacing_factor * (i - len(behavior_nodes)/2), behavior_y)
    
    # 使用YlGnBu颜色映射为所有边着色
    # 收集所有边的权重
    all_weights = []
    for rel_type, edges in top_relationships.items():
        for _, _, weight in edges:
            all_weights.append(weight)
    
    # 如果没有边，设置默认值
    if not all_weights:
        max_weight = 1
    else:
        max_weight = max(all_weights)
    
    # 绘制所有边，使用直线和YlGnBu颜色映射，增强粗细差异
    for rel_type, edges in top_relationships.items():
        for source, target, weight in edges:
            # 计算边的相对粗细，使差异更加明显
            width = 1.0 + 6.0 * (weight / max_weights[rel_type]) ** 0.8  # 增大系数和使用幂运算增强差异
            
            # 计算边的颜色，基于权重的相对大小
            color_val = 0.2 + 0.8 * (weight / max_weight)  # 扩大颜色范围到0.2-1.0
            edge_color = cmap(color_val)
            
            # 绘制直线边
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=[(source, target)], 
                width=width,
                edge_color=[edge_color], 
                arrows=True,
                arrowsize=18,  # 增大箭头
                connectionstyle='arc3,rad=0',  # rad=0 表示直线
                alpha=0.85  # 增加不透明度
            )
    
    # 绘制不同类型的节点，大小依据词频，增强差异
    for category, color in color_map.items():
        nodes = [node for node, data in G.nodes(data=True) if data['type'] == category]
        max_weight = category_max_weights[category]
        
        # 根据词频计算节点大小，增强差异
        min_size = 1000  # 最小节点大小
        max_size = 8000  # 最大节点大小（增大最大值）
        
        weights = [G.nodes[node]['weight'] for node in nodes]
        # 使用指数为0.4的比例计算节点大小，使差异更加明显
        sizes = [min_size + (max_size - min_size) * (w / max_weight) ** 0.4 for w in weights]
        
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=sizes,
                              node_color=color, alpha=0.8)
    
    # 绘制节点标签，使用更大的字体，并为长标签添加换行
    label_font_size = 21  # 将字体大小增大1.5倍（从14增加到21）
    
    # 为长标签添加换行符，采用更激进的换行策略
    labels = {}
    for node in G.nodes():
        # 改进的换行策略
        if len(node) <= 4:
            # 4个字以下不换行
            labels[node] = node
        elif len(node) <= 6:
            # 5-6个字在中间换行
            mid = len(node) // 2
            labels[node] = node[:mid] + '\n' + node[mid:]
        elif len(node) <= 9:
            # 7-9个字分成三行
            third = len(node) // 3
            labels[node] = node[:third] + '\n' + node[third:third*2] + '\n' + node[third*2:]
        else:
            # 10个字以上分成多行，确保每行不超过3个字
            chars_per_line = 3
            lines = []
            for i in range(0, len(node), chars_per_line):
                lines.append(node[i:i+chars_per_line])
            labels[node] = '\n'.join(lines)
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=label_font_size, font_family='SimHei', font_weight='bold')
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['space'], 
                  label='空间特征', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['cognition'], 
                  label='认知评价', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['emotion'], 
                  label='情感类型', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['behavior'], 
                  label='行为意图', markersize=15)
    ]
    
    # 添加表示边颜色和粗细的图例，与实际边粗细保持一致
    weight_legend = []
    for i, w in enumerate([0.2, 0.5, 0.8, 1.0]):
        color_val = 0.2 + 0.8 * w  # 与边颜色计算保持一致
        weight_legend.append(
            plt.Line2D([0], [0], color=cmap(color_val), 
                      linewidth=1.0 + 6.0 * w ** 0.8,  # 与边粗细计算保持一致
                      label=f'关系强度: {int(w * 100)}%')
        )
    
    # 使用更大的字体大小，将图例放置在左下角空白区域
    plt.legend(handles=legend_elements + weight_legend, 
              loc='lower left', bbox_to_anchor=(-0.1, -0.1), fontsize=16)
    
    plt.title(f'空间-认知-情感-行为 直连知识图谱 (各维度TOP{TOP_N})', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"直连静态知识图谱已保存至: {output_path}")

def create_layered_direct_interactive_graph(top_entities, top_relationships, output_path):
    """创建分层布局的直线连接交互式图谱HTML"""
    # 创建交互式网络
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    
    # 关闭物理模拟，以保持节点位置固定
    net.barnes_hut(gravity=0, central_gravity=0, spring_length=0)
    
    # 定义节点类型的颜色
    color_map = {
        'space': '#1f77b4',    # 蓝色
        'cognition': '#ff7f0e', # 橙色/黄色
        'emotion': '#2ca02c',   # 绿色
        'behavior': '#d62728'   # 红色
    }
    
    # 使用YlGnBu颜色映射的几个代表颜色
    ylgnbu_colors = ["#ffffd9", "#c7e9b4", "#41b6c4", "#225ea8"]
    
    # 计算每个关系类型的最大权重，用于归一化
    max_weights = {}
    for rel_type, edges in top_relationships.items():
        if edges:
            max_weights[rel_type] = max(weight for _, _, weight in edges)
        else:
            max_weights[rel_type] = 1
    
    # 计算所有关系中的最大权重
    all_weights = [weight for rel_type, edges in top_relationships.items() for _, _, weight in edges]
    max_weight = max(all_weights) if all_weights else 1
    
    # 计算节点权重的最大值，用于归一化节点大小
    max_weights_by_type = {
        'space': max(top_entities['space'].values()),
        'cognition': max(top_entities['cognition'].values()),
        'emotion': max(top_entities['emotion'].values()),
        'behavior': max(top_entities['behavior'].values())
    }
    
    # 获取各类型节点列表及权重
    space_nodes = [(node, weight) for node, weight in top_entities['space'].items()]
    cognition_nodes = [(node, weight) for node, weight in top_entities['cognition'].items()]
    emotion_nodes = [(node, weight) for node, weight in top_entities['emotion'].items()]
    behavior_nodes = [(node, weight) for node, weight in top_entities['behavior'].items()]
    
    # 根据节点权重排序
    space_nodes.sort(key=lambda x: x[1], reverse=True)
    cognition_nodes.sort(key=lambda x: x[1], reverse=True)
    emotion_nodes.sort(key=lambda x: x[1], reverse=True)
    behavior_nodes.sort(key=lambda x: x[1], reverse=True)
    
    # 设置层级和位置，增大层间距和水平间距
    layer_spacing = 250  # 增大垂直层间距
    horizontal_spacing = 150  # 增大水平间距
    space_y = layer_spacing * 3
    cognition_y = layer_spacing * 2
    emotion_y = layer_spacing
    behavior_y = 0
    
    # 添加空间特征节点
    for i, (node, weight) in enumerate(space_nodes):
        x_pos = (i - len(space_nodes)/2) * horizontal_spacing  # 使用更大的横向间距
        title = f"类型: 空间特征<br>频次: {weight}"
        
        # 节点大小基于权重，增强差异性
        min_size = 25  # 最小节点大小
        max_size = 100  # 最大节点大小
        size = min_size + (max_size - min_size) * (weight / max_weights_by_type['space']) ** 0.4
        
        # 处理较长的标签文本，使用更激进的换行策略
        display_label = node
        if len(node) <= 4:
            # 4个字以下不换行
            display_label = node
        elif len(node) <= 6:
            # 5-6个字在中间换行
            mid = len(node) // 2
            display_label = node[:mid] + '\n' + node[mid:]
        elif len(node) <= 9:
            # 7-9个字分成三行
            third = len(node) // 3
            display_label = node[:third] + '\n' + node[third:third*2] + '\n' + node[third*2:]
        else:
            # 10个字以上分成多行，确保每行不超过3个字
            chars_per_line = 3
            lines = []
            for i in range(0, len(node), chars_per_line):
                lines.append(node[i:i+chars_per_line])
            display_label = '\n'.join(lines)
        
        net.add_node(node, title=title, label=display_label, color=color_map['space'],
                    size=size, x=x_pos, y=-space_y, physics=False)
    
    # 添加认知评价节点
    for i, (node, weight) in enumerate(cognition_nodes):
        # 交错排列认知节点，使相邻节点水平位置错开
        offset = horizontal_spacing/3 if i % 2 == 1 else 0  # 奇数索引的节点向右偏移
        x_pos = (i - len(cognition_nodes)/2) * horizontal_spacing + offset
        title = f"类型: 认知评价<br>频次: {weight}"
        size = min_size + (max_size - min_size) * (weight / max_weights_by_type['cognition']) ** 0.4
        
        # 处理较长的标签文本，使用更激进的换行策略
        display_label = node
        if len(node) <= 4:
            # 4个字以下不换行
            display_label = node
        elif len(node) <= 6:
            # 5-6个字在中间换行
            mid = len(node) // 2
            display_label = node[:mid] + '\n' + node[mid:]
        elif len(node) <= 9:
            # 7-9个字分成三行
            third = len(node) // 3
            display_label = node[:third] + '\n' + node[third:third*2] + '\n' + node[third*2:]
        else:
            # 10个字以上分成多行，确保每行不超过3个字
            chars_per_line = 3
            lines = []
            for i in range(0, len(node), chars_per_line):
                lines.append(node[i:i+chars_per_line])
            display_label = '\n'.join(lines)
        
        net.add_node(node, title=title, label=display_label, color=color_map['cognition'],
                    size=size, x=x_pos, y=-cognition_y, physics=False)
    
    # 添加情感类型节点
    for i, (node, weight) in enumerate(emotion_nodes):
        # 交错排列情感节点
        offset = horizontal_spacing/3 if i % 2 == 0 else 0  # 偶数索引的节点向右偏移
        x_pos = (i - len(emotion_nodes)/2) * horizontal_spacing + offset
        title = f"类型: 情感类型<br>频次: {weight}"
        size = min_size + (max_size - min_size) * (weight / max_weights_by_type['emotion']) ** 0.4
        
        # 处理较长的标签文本，使用更激进的换行策略
        display_label = node
        if len(node) <= 4:
            # 4个字以下不换行
            display_label = node
        elif len(node) <= 6:
            # 5-6个字在中间换行
            mid = len(node) // 2
            display_label = node[:mid] + '\n' + node[mid:]
        elif len(node) <= 9:
            # 7-9个字分成三行
            third = len(node) // 3
            display_label = node[:third] + '\n' + node[third:third*2] + '\n' + node[third*2:]
        else:
            # 10个字以上分成多行，确保每行不超过3个字
            chars_per_line = 3
            lines = []
            for i in range(0, len(node), chars_per_line):
                lines.append(node[i:i+chars_per_line])
            display_label = '\n'.join(lines)
        
        net.add_node(node, title=title, label=display_label, color=color_map['emotion'],
                    size=size, x=x_pos, y=-emotion_y, physics=False)
    
    # 添加行为意图节点
    for i, (node, weight) in enumerate(behavior_nodes):
        # 使用更大的横向间距
        x_pos = (i - len(behavior_nodes)/2) * horizontal_spacing
        title = f"类型: 行为意图<br>频次: {weight}"
        size = min_size + (max_size - min_size) * (weight / max_weights_by_type['behavior']) ** 0.4
        
        # 处理较长的标签文本，使用更激进的换行策略
        display_label = node
        if len(node) <= 4:
            # 4个字以下不换行
            display_label = node
        elif len(node) <= 6:
            # 5-6个字在中间换行
            mid = len(node) // 2
            display_label = node[:mid] + '\n' + node[mid:]
        elif len(node) <= 9:
            # 7-9个字分成三行
            third = len(node) // 3
            display_label = node[:third] + '\n' + node[third:third*2] + '\n' + node[third*2:]
        else:
            # 10个字以上分成多行，确保每行不超过3个字
            chars_per_line = 3
            lines = []
            for i in range(0, len(node), chars_per_line):
                lines.append(node[i:i+chars_per_line])
            display_label = '\n'.join(lines)
        
        net.add_node(node, title=title, label=display_label, color=color_map['behavior'],
                    size=size, x=x_pos, y=-behavior_y, physics=False)
    
    # 添加边，使用YlGnBu颜色映射
    for rel_type, edges in top_relationships.items():
        for source, target, weight in edges:
            # 设置边标题为关系类型和权重
            if rel_type == 'space_to_cognition':
                rel_name = '空间→认知'
            elif rel_type == 'cognition_to_emotion':
                rel_name = '认知→情感'
            elif rel_type == 'emotion_to_behavior':
                rel_name = '情感→行为'
            else:
                rel_name = '空间→情感'
                
            title = f"关系: {rel_name}<br>频次: {weight}"
            
            # 计算边的粗细，增强差异
            width = 1.0 + 8.0 * (weight / max_weights[rel_type]) ** 0.8
            
            # 计算边的颜色
            # 将权重映射到YlGnBu颜色数组的索引
            color_idx = min(int(3 * weight / max_weight), 3)
            edge_color = ylgnbu_colors[color_idx]
            
            # 添加直线边
            net.add_edge(source, target, title=title, width=width, 
                        color=edge_color, 
                        arrowStrikethrough=True,
                        smooth={'enabled': False})  # 禁用平滑，使用直线
    
    # 添加图例
    legend_x = -800
    legend_y = -600
    net.add_node("空间特征_图例", shape="box", label="空间特征", color=color_map['space'], 
                physics=False, x=legend_x, y=legend_y, size=30)
    net.add_node("认知评价_图例", shape="box", label="认知评价", color=color_map['cognition'], 
                physics=False, x=legend_x + 150, y=legend_y, size=30)
    net.add_node("情感类型_图例", shape="box", label="情感类型", color=color_map['emotion'], 
                physics=False, x=legend_x + 300, y=legend_y, size=30)
    net.add_node("行为意图_图例", shape="box", label="行为意图", color=color_map['behavior'], 
                physics=False, x=legend_x + 450, y=legend_y, size=30)
    
    # 边强度的图例
    for i, (color, strength) in enumerate(zip(ylgnbu_colors, ["弱", "中", "强", "极强"])):
        net.add_node(f"强度_{strength}", shape="box", label=f"关系强度: {strength}", 
                    color=color, physics=False, 
                    x=legend_x + i*150, y=legend_y + 50, size=30)
    
    # 添加标题
    title = f'<h2>空间-认知-情感-行为 直连知识图谱 (各维度TOP{TOP_N})</h2>'
    net.heading = title
    
    # 设置选项，增大字体大小
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 27,
          "face": "Microsoft YaHei",
          "bold": true
        },
        "scaling": {
          "label": {
            "enabled": true,
            "min": 27,
            "max": 36
          }
        }
      },
      "edges": {
        "color": {
          "inherit": false
        },
        "smooth": {
          "enabled": false
        },
        "width": 2,
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1
          }
        }
      },
      "physics": {
        "enabled": false
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "zoomView": true,
        "dragView": true
      }
    }
    """)
    
    # 保存为HTML文件
    net.save_graph(output_path)
    print(f"直连交互式知识图谱已保存至: {output_path}")

def main():
    """主函数"""
    print(f"开始创建直连知识图谱 (TOP{TOP_N})...")
    
    # 读取数据
    df = read_csv_file()
    if df is None:
        return
    
    # 提取关系
    relationships = extract_relationships(df)
    
    # 获取TOP实体
    top_entities = get_top_entities(relationships, TOP_N)
    
    # 筛选TOP关系 (频次≥2的关系)
    top_relationships = filter_top_relationships(relationships, top_entities, min_weight=2)
    
    # 创建直连静态图谱
    create_layered_direct_graph(top_entities, top_relationships, 
                             os.path.join(output_dir, f"直连知识图谱_TOP{TOP_N}.png"))
    
    # 创建直连交互式图谱
    create_layered_direct_interactive_graph(top_entities, top_relationships, 
                                         os.path.join(output_dir, f"直连知识图谱_TOP{TOP_N}.html"))
    
    print(f"直连知识图谱创建完成！")

if __name__ == "__main__":
    main()