# SPDX-License-Identifier: Apache-2.0
from service.colors import assign_colors


def test_same_labels_same_colors_deterministic():
    labels = ["self_verification", "valid_syntax_in_toolcall"]
    a = assign_colors(labels)
    b = assign_colors(labels)
    assert a == b  # 稳定：同输入同输出
    assert set(a.keys()) == set(labels)
    for v in a.values():
        assert v.startswith("#") and len(v) == 7


def test_distinct_labels_get_distinct_colors_until_palette_exhausted():
    labels = [f"cap_{i}" for i in range(3)]
    colors = assign_colors(labels)
    assert len(set(colors.values())) == 3  # 前几个各不相同


def test_palette_wraps_when_more_labels_than_colors():
    labels = [f"cap_{i}" for i in range(50)]
    colors = assign_colors(labels)
    assert len(colors) == 50  # 每个 label 都有色，超出调色板则循环复用
    assert all(c.startswith("#") for c in colors.values())


def test_order_independent_assignment():
    # 同一组 label，顺序不同，各 label 拿到的颜色应一致（按 label 排序分配）
    forward = assign_colors(["b_cap", "a_cap"])
    backward = assign_colors(["a_cap", "b_cap"])
    assert forward == backward
