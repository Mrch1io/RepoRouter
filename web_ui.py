
import streamlit as st
import time
import logging
from base64 import b64encode
from pathlib import Path

from src.agent.core import RepoRouter 

#图片路径跟着当前文件走，从其他目录启动也能找到
AGENT_ICON = str(Path(__file__).resolve().parent / "assets" / "agent.png")

@st.cache_data
def load_agent_image()->str:
    #1.缓存图片编码，聊天刷新时不用重复读取和转换
    return "data:image/png;base64," + b64encode(Path(AGENT_ICON).read_bytes()).decode()

AGENT_IMAGE = load_agent_image()


def show_trace(trace)->float:
    #1.只把执行记录显示为调用，tool_context是回传给模型的内容，不再重复显示
    # 🔍 把每一步显示出来：派单、调了哪个工具、命中哪些文件、有没有软熔断
    cost = 0.0
    for ev in trace:
        if ev.get("event") == "router_decision":
            d = ev["decision"]
            st.write(f"路由决策：{d.get('decision_type')} {d.get('tool_name') or ''}")
        elif "result_preview" in ev:
            if ev.get("error"):
                st.write(f"工具处理失败：`{ev['tool_name']}` · {ev['error']}")
            else:
                st.write(f"调用工具：`{ev['tool_name']}`")
            sp = ev.get("search_payload") or {}
            if sp.get("soft_fuse_triggered"):
                st.write("检索结果和问题不够像，已触发软熔断，不使用检索片段")
            for r in sp.get("results", [])[:3]:
                st.write(f"命中：{r['source']} {('第 ' + r['lines'] + ' 行') if r.get('lines') else ''}")
        elif ev.get("event") == "usage":
            cost = ev["cost_usd"]
    return cost


# 1. 网页标题与极客风配置
st.set_page_config(page_title="RepoRouter · 代码库助手", page_icon=AGENT_ICON, layout="centered")
#只调整间距和灰色边框，保留原来的深色主题与组件配色
st.html("""
<style>
.stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp button, .stApp textarea {
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}
.stMainBlockContainer { max-width: 56rem; padding-top: 3.5rem; padding-bottom: 6rem; }
[data-testid="stBottomBlockContainer"] { max-width: 56rem; }
.rr-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 1.75rem; }
.rr-brand img { width: 44px; height: 44px; object-fit: contain; flex-shrink: 0; }
.rr-brand strong { font-size: 18px; font-weight: 600; letter-spacing: -0.5px; }
.rr-brand small { display: block; font-size: 11px; opacity: 0.55; margin-top: 3px; }
.rr-header { display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 1rem; border-bottom: 1px solid rgb(128 128 128 / 18%); }
.rr-header strong { font-size: 14px; font-weight: 500; }
.rr-header span { font-size: 12px; opacity: 0.5; }
.rr-hero { text-align: center; padding: 1.5rem 0; }
.rr-hero img { display: block; width: 144px; height: 144px; object-fit: contain; margin: 0 auto 1rem;
    filter: drop-shadow(0 8px 18px rgb(0 0 0 / 15%)); }
.stApp .rr-hero h2 { font-size: 30px; font-weight: 500; letter-spacing: -0.8px; padding: 0; margin: 0; }
.rr-hero p { font-size: 14px; opacity: 0.6; margin: 14px 0 0; line-height: 1.8; }
.rr-card { border: 1px solid rgb(128 128 128 / 22%); border-radius: 16px;
    padding: 20px; min-height: 150px; background: rgb(128 128 128 / 4%); }
.rr-card img { width: 21px; height: 21px; opacity: 0.8; margin-bottom: 12px; }
.stApp .rr-card h3 { font-size: 14px; font-weight: 500; padding: 0; margin: 0 0 8px; }
.rr-card p { font-size: 12px; line-height: 1.8; opacity: 0.55; margin: 0; }
[data-testid="stChatMessage"] { border-radius: 16px; padding: 1.25rem; }
[data-testid="stChatMessage"] p { font-size: 15px; line-height: 1.8; }
[data-testid="stChatMessageAvatarCustom"] { object-fit: contain; border-radius: 0; }
[data-testid="stChatInput"] > div {
    border-radius: 20px; border: 1px solid rgb(128 128 128 / 30%);
    box-shadow: 0 6px 24px rgb(0 0 0 / 10%);
}
[data-testid="stChatInput"] > div:focus-within { border-color: #85858b; }
[data-testid="stChatInputTextArea"] { font-size: 14px; }
[data-testid="stSidebar"] { border-right: 1px solid rgb(128 128 128 / 15%); }
[data-testid="stSidebar"] [data-testid="stButton"] button { border-radius: 12px; min-height: 44px; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { font-size: 12px; line-height: 1.8; }
@media (max-width: 640px) {
    .stMainBlockContainer { padding-top: 3.5rem; }
    .rr-header span { display: none; }
    .rr-hero { padding: 2rem 0 1rem; }
    .rr-hero img { width: 120px; height: 120px; }
    .stApp .rr-hero h2 { font-size: 25px; }
    .rr-card { min-height: 0; padding: 16px; }
    .rr-card img { float: left; margin: 4px 14px 20px 0; }
    [data-testid="stChatMessage"] { padding: 0.75rem; }
}
</style>
""")
st.html('<div class="rr-header"><strong>Agent 工作台</strong><span>代码理解 / 检索 / 分析</span></div>')

# 2. 会话状态管理 (Session State) —— 极其重要！
# 因为 Streamlit 每次用户点击都会重新从上往下跑一遍代码
# 我们必须把 Agent 实例和历史聊天记录“锁”在内存里
if "agent" not in st.session_state:
    # 初始化你的 Agent（包含向量库预热）
    with st.spinner("正在准备代码库助手…"):
        st.session_state.agent = RepoRouter(max_history_turns=4)
    st.session_state.messages = [] # 用于前端显示的聊天记录

# 清空对话按钮（放在侧边栏）：前端记录和 Agent 记忆一起清掉
with st.sidebar:
    st.html(f'<div class="rr-brand"><img src="{AGENT_IMAGE}" alt=""><div><strong>RepoRouter</strong><small>私有代码库助手</small></div></div>')
    st.caption("当前会话")
    if st.button("清空对话", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.agent.clear_memory()
        st.session_state.messages = []
    st.divider()
    with st.expander("提问小提示"):
        st.caption("带上文件名、函数名或完整报错，更容易定位问题。")
        st.caption("回答中的「处理详情」可以展开查看检索与工具记录。")
    st.caption("清空对话会同时清除聊天记录和对话记忆。")

#空白页给几个例子，发送问题后收起，把空间留给回答
welcome = st.empty()
if not st.session_state.messages:
    with welcome.container():
        with st.container(key="welcome"):
            st.html(f'<div class="rr-hero"><img src="{AGENT_IMAGE}" alt="RepoRouter Agent"><h2>让代码，更容易理解。</h2><p>定位实现、理清逻辑，或一起排查问题。<br>从一个问题开始，RepoRouter 帮你找到线索。</p></div>')
            with st.container(key="suggestions"):
                examples = (
                    ("定位实现", "聊天接口在哪个文件里注册？", '<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/>'),
                    ("理解代码", "解释 core.py 中的对话流程。", '<path d="m7 7-5 5 5 5m10-10 5 5-5 5m-3-13-4 20"/>'),
                    ("梳理概念", "BM25 和向量检索有什么区别？", '<path d="M9 18h6m-5 3h4M8 14a6 6 0 1 1 8 0l-1 2H9z"/>'),
                )
                for column, (title, example, icon) in zip(st.columns(3), examples):
                    with column:
                        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#b8bdc8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">{icon}</svg>'
                        icon_image = "data:image/svg+xml;base64," + b64encode(svg.encode()).decode()
                        st.html(f'<div class="rr-card"><img src="{icon_image}" alt=""><h3>{title}</h3><p>{example}</p></div>')

# 3. 渲染历史聊天气泡
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else AGENT_ICON):
        #1.旧消息只有正文；新回答把处理详情一起保存，下次刷新也能展开
        if "trace" in msg:
            with st.status(f"处理详情 · {msg['elapsed_time']:.2f} 秒 · 约 ${msg['cost']:.4f}",state="complete",expanded=False):
                show_trace(msg["trace"])
        st.markdown(msg["content"])

# 4. 监听用户输入
if prompt := st.chat_input("描述你的问题，或粘贴代码与报错…"):
    welcome.empty()
    
    # 立即在前端显示用户的提问
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    # 5. 呼叫后端 Agent 并展示“思考特效”
    with st.chat_message("assistant", avatar=AGENT_ICON):
        # 👑 炫酷的 Agent 思考状态框
        with st.status("正在处理你的问题…", expanded=False) as status:
            st.write("正在分析问题并准备回答。")
            
            # 记录开始时间
            start_time = time.perf_counter()
            
            # 🔥 呼叫你的底层逻辑！trace 会记下 Agent 每一步做了什么
            trace = []
            try:
                final_response = st.session_state.agent.chat(prompt, trace=trace)
            except Exception:
                #1.后端失败会回滚，本轮也不加入前端历史。
                logging.exception("聊天请求失败")
                status.update(label="本次请求失败",state="error",expanded=True)
                st.error("这次没有收到完整回答，请稍后重新发送。之前的对话记录已保留。")
                st.stop()

            # 计算耗时
            elapsed_time = time.perf_counter() - start_time

            cost = show_trace(trace)

            # 状态框变绿打钩
            status.update(label=f"处理详情 · {elapsed_time:.2f} 秒 · 约 ${cost:.4f}", state="complete", expanded=False)
        
        # 打印最终的人话
        st.markdown(final_response)
        
    # 把大模型的回答存入前端记忆区
    #2.成功后再一起保存提问和回答，避免只留下失败的提问。
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": final_response,
                                     "trace": trace,"elapsed_time": elapsed_time,"cost": cost})
