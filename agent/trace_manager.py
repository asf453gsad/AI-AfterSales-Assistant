# ============================================================
# 对话追踪管理器 — 持久化保存对话记录到本地 traces/ 目录
# 每个 thread_id 对应一个 JSON 文件，支持多对话切换
# ============================================================
import json
import os
from datetime import datetime
from utils.path_tool import get_abs_path


class TraceManager:

    def __init__(self):
        self.trace_dir = get_abs_path("traces")
        os.makedirs(self.trace_dir, exist_ok=True)

    # ============================================================
    # 内部：获取 thread 对应的 JSON 文件路径
    # ============================================================
    def _get_thread_path(self, thread_id):
        return os.path.join(self.trace_dir, f"{thread_id}.json")

    # ============================================================
    # 创建新对话：写入初始结构（thread_id, messages[], 时间戳）
    # ============================================================
    def create_thread(self, thread_id):
        path = self._get_thread_path(thread_id)
        if os.path.exists(path):
            return
        data = {
            "thread_id": thread_id,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 添加消息：追加到 messages 数组，更新 updated_at
    # ============================================================
    def add_message(self, thread_id, role, content):
        path = self._get_thread_path(thread_id)
        if not os.path.exists(path):
            self.create_thread(thread_id)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        data["updated_at"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 获取指定对话的全部消息
    # ============================================================
    def get_messages(self, thread_id):
        path = self._get_thread_path(thread_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["messages"]

    # ============================================================
    # 列出所有对话：按 updated_at 降序排列
    # ============================================================
    def list_threads(self):
        threads = []
        if not os.path.isdir(self.trace_dir):
            return threads
        for fname in os.listdir(self.trace_dir):
            if fname.endswith(".json"):
                path = os.path.join(self.trace_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    threads.append({
                        "thread_id": data["thread_id"],
                        "created_at": data["created_at"],
                        "updated_at": data["updated_at"],
                        "msg_count": len([m for m in data["messages"] if m["role"] == "user"]),
                    })
                except:
                    continue
        threads.sort(key=lambda x: x["updated_at"], reverse=True)
        return threads

    # ============================================================
    # 统计用户消息条数（用于判断是否需要压缩）
    # ============================================================
    def count_user_messages(self, thread_id):
        msgs = self.get_messages(thread_id)
        return len([m for m in msgs if m["role"] == "user"])

    # ============================================================
    # 聚合所有对话文本（用于生成用户画像）
    # ============================================================
    def get_conversation_text(self):
        all_text = ""
        for fname in os.listdir(self.trace_dir):
            if fname.endswith(".json"):
                path = os.path.join(self.trace_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for m in data["messages"]:
                        all_text += f"[{m['role']}]: {m['content']}\n"
                except:
                    continue
        return all_text

    # ============================================================
    # 压缩摘要的存取（存储为 summary_{thread_id}.txt）
    # ============================================================
    def get_summary(self, thread_id):
        path = os.path.join(self.trace_dir, f"summary_{thread_id}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def save_summary(self, thread_id, summary):
        path = os.path.join(self.trace_dir, f"summary_{thread_id}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary)
