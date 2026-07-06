# ============================================================
# 路径工具 — 自动定位项目根目录，将相对路径转为绝对路径
# 确保无论从何处运行都能正确找到文件
# ============================================================
import os


def get_project_root():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(current_dir)
    return project_root


def get_abs_path(relative_path: str) -> str:
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)
