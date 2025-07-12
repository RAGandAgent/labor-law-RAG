# RAG项目

本项目基于检索增强生成（Retrieval-Augmented Generation, RAG）技术架构，致力于开发一款智能法律咨询助手。该系统通过构建本地法律文档知识库，
利用Milvus向量数据库实现高效语义检索，并结合大语言模型生成专业法律回复。前端采用Streamlit框架开发，为用户提供简洁友好的交互界面，实现从文档检索到智能回复的一体化服务。
## 项目简介

该项目主要功能包括：

## 核心功能

### 📄 文档解析与向量化处理
- 支持多格式文档解析（PDF/TXT等）
- 采用先进的嵌入模型（Embedding Model）提取语义特征
- 数据存储于Milvus向量数据库，构建高效可检索的法律知识库

### 🔍 智能语义检索
- 基于自然语言理解处理用户查询
- 从向量数据库中精准匹配相关法律条文和案例
- 确保检索结果的准确性和法律相关性

### 💡 Agent增强问答系统
- 集成Qwen Agent等大语言模型
- 结合RAG（检索增强生成）技术
- 生成专业、合规且易于理解的法律解答

### 🖥️ 交互式可视化界面
- 基于Streamlit框架开发
- 提供简洁直观的Web交互界面
- 支持：
  - 实时法律咨询
  - 检索结果可视化展示
  - 系统参数配置
## 项目结构

```
. 
├── documents/              # 存储原始法律文档       
│   └── labor_law/          # 专门存放与劳动法相关的法律文件（如劳动合同法、劳动争议调解仲裁法等）,支持PDF/TXT等常见文档格式
├── sreams.py               # Streamlit前端交互界面实现
├── main.py                 # 项目主入口，用于文档处理和测试检索
├── parsed_documents/       # 解析后的文档和向量化数据存放目录
├── scripts/                # 核心逻辑代码
│   ├── document_parser.py  # 文档解析模块
│   ├── pipeline.py         # RAG核心处理流水线，用来协调检索与生成流程
│   ├── utils.py            # 常用工具函数
│   └── vector_processor.py # 向量嵌入生成与Milvus数据库交互 
└── README.md               # 项目说明文件
```

## 安装步骤

1. **克隆仓库**：

   ```bash
   git clone https://github.com/RAGandAgent/labor-law-RAG.git
   cd labor-law-RAG
   ```

2. **创建虚拟环境并安装依赖**：

   建议使用  `conda` 创建虚拟环境。

   ```bash
   # 使用 conda
   conda create -n rag python=3.9
   ```

   ```bash
   # 安装依赖
   pip install streamlit qwen-agent
   ```
3. **配置 Milvus**：   本项目依赖 Milvus 作为向量数据库。您需要确保 Milvus 服务正在运行。 

   1、在管理员模式下右击并选择以**管理员身份运行**，打开 Docker Desktop。

   2、下载安装脚本并将其保存为`standalone.bat` 。
   ```
   C:\>Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
   ```
   3、运行下载的脚本，将 Milvus 作为 Docker 容器启动。
   ```
   C:\>standalone.bat start
   Wait for Milvus starting...
   Start successfully.
   To change the default Milvus configuration, edit user.yaml and restart the service.
   ```
4. **配置 API 密钥**：
   在 `sreams.py` 中，模型 API 密钥和 Tavily API 密钥被硬编码。在实际使用中，您可以将其替换为环境变量。
## 使用方法

### 1. 文档处理

在运行 sreams.py 之前，您需要先处理文档，将其解析并向量化存储到 Milvus 中。可以通过运行 `main.py` 来完成此步骤：

```bash
python main.py
```

`main.py` 中的 `docs_path` 变量指定了要处理的文档路径。您可以修改此路径以处理不同的文档集。

### 2. 启动 Streamlit 应用

文档处理完成后，您可以启动 Streamlit 应用：

```bash
streamlit run sreams.py
```

应用将在您的浏览器中打开。您可以通过左侧的侧边栏配置模型版本、向量模型、数据集、召回数和重排序数量。

### 3. 进行查询

在聊天输入框中输入您的问题，系统将从配置的数据集中检索相关信息，并由 Agent 生成回答。

- **数据集选择**：在侧边栏选择您想要查询的数据集（你可以选择处理上传不同的数据集）。
- **强制网络搜索**：可以通过界面上的切换按钮选择是否强制进行网络搜索（如果 Agent 配置了相关工具）。

