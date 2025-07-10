import logging
import time

import streamlit as st
from qwen_agent.agents import Assistant  # TODO 更换agent

from scripts.pipeline import singleton_pipeline

# Streamlit App Initialization
st.title("🤖 Lobar Law  RAG")


# -------------------------
# 数据集字典
def local_dataset(name):
    data = [{"collection": "labor_law", "name": "中国劳动法"}]
    if name != "":
        for d in data:
            if d['name'] == name:
                return d
    else:
        return data


# Session State Initialization
if 'model_version' not in st.session_state:
    st.session_state.model_version = "qwen-plus-latest"
if 'model_api_key' not in st.session_state:
    st.session_state.model_api_key = "YOUR_DASHSCOPE_API_KEY"
if 'travily_key' not in st.session_state:
    st.session_state.travily_key = "YOUR_TAVILY_API_KEY"
if 'history' not in st.session_state:
    st.session_state.history = []
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False
if 'embedding' not in st.session_state:
    st.session_state.embedding = "text-embedding-v4"
if 'dataset' not in st.session_state:
    st.session_state.dataset = local_dataset("")[0]['name']
if 'recalls' not in st.session_state:
    st.session_state.recalls = 50
if 'topk' not in st.session_state:
    st.session_state.topk = 10

# Sidebar Configuration
st.sidebar.header("⚙️ 设置")

# Model Selection
st.sidebar.header("🧠 模型选择")
st.session_state.model_version = st.sidebar.radio(
    "Select Model Version",
    options=["qwen-plus-latest", "qwen-max-latest"],
)

# RAG 存储(Milvus)
st.sidebar.header("📚 RAG")
st.session_state.embedding = st.sidebar.radio(
    "向量模型",
    options=["text-embedding-v4", "text-embedding-v3"],
)
st.session_state.dataset = st.sidebar.radio(
    "数据集(Milvus)",
    options=[item['name'] for item in local_dataset("")],
)
st.session_state.recalls = st.sidebar.slider(
    "召回数(关键词、向量检索)",
    min_value=10,
    max_value=200,
    value=50,
)
st.session_state.topk = st.sidebar.slider(
    "重排序数量",
    min_value=2,
    max_value=50,
    value=10,
)

# Clear Chat Button
if st.sidebar.button("✨ 清除问答"):
    st.session_state.history = []
    st.rerun()


def get_qwen_agent(ref_docs):
    try:
        if st.session_state.use_web_search:
            use_web_prompt = "使用 Tavily MCP 搜索相关内容"
        else:
            use_web_prompt = "不使用 Tavily MCP 搜索相关内容"
        # ====== 助手 system prompt 和函数描述 ======
        system_prompt = """请充分理解以下参考资料内容，组织出满足用户提问的条理清晰的回复。
{}

# 参考资料：(文档的相关度作为参考的权重)
{}""".format(use_web_prompt, ref_docs)
        # TODO BUG
        # functions_desc = [
        #     {"mcpServers": {
        #         "Tavily": {
        #             "command": "npx",
        #             "args": [
        #                 "-y",
        #                 "tavily-mcp@0.1.4"
        #             ],
        #             "env": {
        #                 "TAVILY_API_KEY": st.session_state.travily_key
        #             }
        #         }
        #     }}
        # ]
        functions_desc = []
        llm_cfg = {
            # 使用 DashScope 提供的模型服务
            'model': st.session_state.model_version,
            'model_server': 'dashscope',
            'api_key': st.session_state.model_api_key,
            'timeout': 30,
            'retry_count': 3,
            'generate_cfg': {
                'top_p': 0.1
            }
        }
        bot = Assistant(
            llm=llm_cfg,
            name='RAG检索助手',
            description='用户提出问题，根据从知识库中检索的某个相关细节来回答。',
            system_message=system_prompt,
            function_list=functions_desc,
        )
        logging.info("创建助手成功")
        return bot
    except Exception as e:
        logging.error(f"创建助手失败: {str(e)}")
        raise


chat_col, toggle_col = st.columns([0.9, 0.1])

with chat_col:
    prompt = st.chat_input("请输入查询内容...")

with toggle_col:
    st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

if prompt:
    # Add user message to history
    # st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    logging.info(f"recall/topk: {st.session_state.recalls}, {st.session_state.topk}")
    # Existing RAG flow remains unchanged
    with st.spinner("🔍 Searching"):
        try:
            rewritten_query = prompt
            t0 = time.time()
            coll_name = local_dataset(st.session_state.dataset)['collection']
            search_results = singleton_pipeline.search(coll_name, rewritten_query,
                                                       count=st.session_state.recalls,
                                                       top_k=st.session_state.topk)
            t1 = time.time()
            with st.expander(f"检索结果： 查询到 {len(search_results)} 个文档，耗时: {t1 - t0:.2f} 秒"):
                with st.chat_message("user"):
                    st.markdown(
                        f"用户问题: {rewritten_query}\n\n---\n\n{singleton_pipeline.format_search_results(search_results)}")
        except Exception as e:
            st.error(f"❌ Error query: {str(e)}")
            rewritten_query = ""

    try:
        with st.spinner("🤔Thinking..."):
            bot = get_qwen_agent(singleton_pipeline.format_search_results(search_results))
            logging.info("正在处理您的请求...")
            messages = [{'role': 'user', 'content': rewritten_query}]

            # 运行助手并处理响应
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in bot.run(messages):
                    resp_json = response[0]
                    if resp_json['role'] == 'assistant' and resp_json['content'] != '':
                        full_response = resp_json['content']
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
    except Exception as e:
        st.error(f"❌ Error Agent: {str(e)}")

