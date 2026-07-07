#!/usr/bin/env python3
"""Generate Module 0 (Query Compiler) UML activity diagram as PNG using graphviz."""

import graphviz

dot = graphviz.Digraph('module0', format='png')
dot.attr(rankdir='TB', dpi='150', bgcolor='#FEFEFE')
dot.attr('node', fontname='PingFang SC', fontsize='11', style='filled')
dot.attr('edge', fontname='PingFang SC', fontsize='10')

# Start
dot.node('start', '', shape='circle', width='0.3', fillcolor='#202124', fontcolor='white')

# Input
dot.node('input', '用户问题清单\n（自然语言描述）', shape='rectangle', fillcolor='#E8F0FE', color='#4285F4')

# Call 1
dot.node('call1_label', 'Call 1 — LLM 调用 #1', shape='rectangle', fillcolor='#4285F4', fontcolor='white', style='filled,bold')
dot.node('part1', 'Part 1: 子问题分解\n按逗号/分号/语义转折切分\n产出: id, raw_text, failure_summary', shape='rectangle', fillcolor='#E8F0FE', color='#4285F4')

# N sub-problems
dot.node('n_sub', 'N 条子问题（batch）', shape='parallelogram', fillcolor='#FEF7E0', color='#F9AB00')

# Call 2
dot.node('call2_label', 'Call 2 — LLM 调用 #2（批量 + 置信度自评）', shape='rectangle', fillcolor='#4285F4', fontcolor='white', style='filled,bold')

# Call 2 sub-parts (parallel)
dot.node('part2', 'Part 2\n能力标签映射\ntarget_capability', shape='rectangle', fillcolor='#E8F0FE', color='#4285F4')
dot.node('part3', 'Part 3\n轨迹自报信号\ntrajectory_signal', shape='rectangle', fillcolor='#E8F0FE', color='#4285F4')
dot.node('part4', 'Part 4\nHyDE 正例生成\nhyde_positive ×2-3', shape='rectangle', fillcolor='#E8F0FE', color='#4285F4')
dot.node('part5', 'Part 5\n关键词+结构化过滤\nkeywords, structured_filters', shape='rectangle', fillcolor='#E8F0FE', color='#4285F4')
dot.node('part6', 'Part 6\n置信度自评\nconfidence', shape='rectangle', fillcolor='#E8F0FE', color='#4285F4')

# Merge after parallel
dot.node('merge', '', shape='point', width='0.1')

# Decision
dot.node('decision', 'confidence ≥ 0.8 ?', shape='diamond', fillcolor='#FEF7E0', color='#F9AB00', width='2')

# Pass path
dot.node('pass_action', '加入 Problem Spec', shape='rectangle', fillcolor='#E6F4EA', color='#34A853')

# Drop path
dot.node('drop_action', '筛除该子问题', shape='rectangle', fillcolor='#FCE8E6', color='#EA4335')

# Output
dot.node('output', 'Problem Spec 输出\n（仅含 pass 的子问题）\n携带: target_capability / trajectory_signal\nhyde_positive / keywords / structured_filters', shape='rectangle', fillcolor='#E6F4EA', color='#34A853', style='filled,bold')

# Deliver
dot.node('deliver', '交付下游召回模块\n（模块 1→2）', shape='rectangle', fillcolor='#F1F3F4', color='#5F6368')

# End
dot.node('end', '', shape='doublecircle', width='0.3', fillcolor='#202124')

# Edges
dot.edge('start', 'input')
dot.edge('input', 'call1_label')
dot.edge('call1_label', 'part1')
dot.edge('part1', 'n_sub')
dot.edge('n_sub', 'call2_label')

# Call 2 fan-out
dot.edge('call2_label', 'part2')
dot.edge('call2_label', 'part3')
dot.edge('call2_label', 'part4')
dot.edge('call2_label', 'part5')
dot.edge('call2_label', 'part6')

# Merge
dot.edge('part2', 'merge')
dot.edge('part3', 'merge')
dot.edge('part4', 'merge')
dot.edge('part5', 'merge')
dot.edge('part6', 'merge')

dot.edge('merge', 'decision')

# Decision branches
dot.edge('decision', 'pass_action', label=' ≥ 0.8 (pass)', color='#34A853')
dot.edge('decision', 'drop_action', label=' < 0.8 (drop)', color='#EA4335')

dot.edge('pass_action', 'output')
dot.edge('output', 'deliver')
dot.edge('deliver', 'end')
dot.edge('drop_action', 'end')

# Layout hints - keep parts at same rank
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('part2')
    s.node('part3')
    s.node('part4')
    s.node('part5')
    s.node('part6')

# Render
dot.render('/Users/deng/开发/UXFlow/module0_query_compiler', cleanup=True)
print("Done! PNG saved to /Users/deng/开发/UXFlow/module0_query_compiler.png")
