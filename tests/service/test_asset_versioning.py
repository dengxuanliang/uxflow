# SPDX-License-Identifier: Apache-2.0
"""静态资源缓存失效：index.html 的 ?v= 由内容 hash 自动注入。

手工维护 ?v=N 会漏（改了 app.js 忘记升版本），而漏了不报错 —— 浏览器静默使用
缓存的旧脚本配新 HTML，顶层绑定引用不存在的元素/函数，整个前端初始化中断。
"""
import pathlib

from fastapi.testclient import TestClient

from service.app import _ASSET_PLACEHOLDER, _render_index, create_app
from service.runstore import MemoryRunStore


def _client():
    return TestClient(create_app(store=MemoryRunStore(), deps=object()))


# ── 端到端：两个入口都必须拿到渲染后的 HTML ────────────────────────────
def test_index_route_has_no_unreplaced_placeholder():
    body = _client().get("/").text
    assert _ASSET_PLACEHOLDER not in body, "占位符未被替换，缓存失效未生效"
    assert "/app.js?v=" in body
    assert "/style.css?v=" in body


def test_index_html_alias_is_rendered_too():
    """直接访问 /index.html 也要走渲染版。

    不注册这个别名的话，StaticFiles 会原样返回模板，?v=__ASSET_V__ 变成字面量。
    """
    body = _client().get("/index.html").text
    assert _ASSET_PLACEHOLDER not in body
    assert "/app.js?v=" in body


def test_static_assets_still_served():
    """动态 index 路由不得挡住静态资源本身。"""
    c = _client()
    assert c.get("/app.js").status_code == 200
    assert c.get("/style.css").status_code == 200


def test_both_assets_share_one_version_token():
    """两个资源共用同一个 hash —— 任一变化都刷新两者。"""
    import re
    body = _client().get("/").text
    versions = set(re.findall(r"\?v=([0-9a-f]+)", body))
    assert len(versions) == 1, f"版本号不一致: {versions}"
    assert len(versions.pop()) == 8


# ── _render_index 单元行为 ────────────────────────────────────────────
def test_version_changes_when_asset_content_changes(tmp_path):
    (tmp_path / "index.html").write_text(f"<p>?v={_ASSET_PLACEHOLDER}</p>",
                                         encoding="utf-8")
    (tmp_path / "app.js").write_text("// v1", encoding="utf-8")
    (tmp_path / "style.css").write_text("/*c*/", encoding="utf-8")
    first = _render_index(tmp_path)

    (tmp_path / "app.js").write_text("// v2 —— 内容变了", encoding="utf-8")
    assert _render_index(tmp_path) != first, "app.js 变了但版本号没变"


def test_version_is_stable_when_content_unchanged(tmp_path):
    """内容没变 → hash 不变 → 浏览器缓存继续有效（这正是不用 mtime 的理由）。"""
    (tmp_path / "index.html").write_text(f"<p>?v={_ASSET_PLACEHOLDER}</p>",
                                         encoding="utf-8")
    (tmp_path / "app.js").write_text("// same", encoding="utf-8")
    (tmp_path / "style.css").write_text("/*c*/", encoding="utf-8")
    assert _render_index(tmp_path) == _render_index(tmp_path)


def test_missing_asset_degrades_instead_of_crashing(tmp_path, capsys):
    """缺个静态资源不该让 create_app 崩掉、服务起不来。"""
    (tmp_path / "index.html").write_text(f"<p>?v={_ASSET_PLACEHOLDER}</p>",
                                         encoding="utf-8")
    (tmp_path / "app.js").write_text("// only js", encoding="utf-8")
    # 没有 style.css
    out = _render_index(tmp_path)
    assert _ASSET_PLACEHOLDER not in out
    assert "style.css" in capsys.readouterr().out      # 但要出声


def test_missing_placeholder_warns(tmp_path, capsys):
    """模板里没有占位符时必须告警 —— 静默失效正是本机制要消灭的东西。"""
    (tmp_path / "index.html").write_text("<p>?v=9</p>", encoding="utf-8")
    (tmp_path / "app.js").write_text("// js", encoding="utf-8")
    (tmp_path / "style.css").write_text("/*c*/", encoding="utf-8")
    out = _render_index(tmp_path)
    assert out == "<p>?v=9</p>"                        # 原样返回，不破坏页面
    assert _ASSET_PLACEHOLDER in capsys.readouterr().out


def test_real_index_html_carries_the_placeholder():
    """仓库里的 index.html 必须用占位符，不能被手工改回固定版本号。"""
    web = pathlib.Path(__file__).resolve().parents[2] / "src" / "service" / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    assert html.count(_ASSET_PLACEHOLDER) == 2, "app.js 和 style.css 各需一个"
