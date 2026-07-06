# ============================================================
# 智扫通机器人智能客服 — Streamlit 前端主入口
# 功能：多对话管理、agent路由、流式显示、用户画像展示
# ============================================================
import uuid
import streamlit as st
from agent.react_agent import ReactAgent
from agent.report_agent import ReportAgent
from agent.trace_manager import TraceManager
from agent.user_profile import UserProfile

st.set_page_config(layout="wide")
st.title("智扫通机器人智能客服")

# 全局服务实例
trace_manager = TraceManager()
user_profile_manager = UserProfile()

# ============================================================
# Session 初始化：agent 实例 + 当前对话 thread_id
# 首次启动时读取已有对话或创建新对话
# ============================================================
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "current_thread_id" not in st.session_state:
    threads = trace_manager.list_threads()
    if threads:
        st.session_state["current_thread_id"] = threads[0]["thread_id"]
    else:
        new_id = str(uuid.uuid4())
        st.session_state["current_thread_id"] = new_id
        trace_manager.create_thread(new_id)

# ============================================================
# 侧边栏：对话历史列表 + 用户画像
# ============================================================
with st.sidebar:
    st.header("对话历史")
    if st.button("新对话", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state["current_thread_id"] = new_id
        trace_manager.create_thread(new_id)
        st.rerun()

    threads = trace_manager.list_threads()
    for t in threads:
        label = f"对话({t['msg_count']}条) {t['created_at'][:16]}"
        if st.button(label, key=t["thread_id"], use_container_width=True):
            st.session_state["current_thread_id"] = t["thread_id"]
            st.rerun()

    st.divider()
    profile = user_profile_manager.get_profile()
    if profile:
        st.header("用户画像")
        st.text_area("", profile.get("profile", ""), height=300, disabled=True, label_visibility="collapsed")

# ============================================================
# 主区域：渲染历史消息
# ============================================================
messages = trace_manager.get_messages(st.session_state["current_thread_id"])
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# 用户输入处理：路由到主Agent或报告Agent + 流式显示 + 持久化
# ============================================================
prompt = st.chat_input("请输入您的问题...")

if prompt and st.session_state["current_thread_id"]:
    with st.chat_message("user"):
        st.markdown(prompt)
    trace_manager.add_message(st.session_state["current_thread_id"], "user", prompt)

    is_report = any(kw in prompt for kw in ["报告", "report", "使用报告", "生成报告"])

    with st.chat_message("assistant"):
        if is_report:
            report_agent = ReportAgent()
            response = st.write_stream(
                report_agent.execute_stream(prompt, st.session_state["current_thread_id"])
            )
        else:
            response = st.write_stream(
                st.session_state["agent"].execute_stream(prompt, st.session_state["current_thread_id"])
            )

    trace_manager.add_message(st.session_state["current_thread_id"], "assistant", response)

    if not is_report:
        st.session_state["agent"].check_and_compress(st.session_state["current_thread_id"])

    user_profile_manager.check_and_generate()
