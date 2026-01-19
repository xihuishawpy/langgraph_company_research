# Company Research Agent

基于 LangGraph 的 AI 智能体系统，自动进行公司深度调研，生成结构化的研究报告。

## 架构概览

```
输入 (公司名、URL、行业、总部位置)
    |
    v
[Grounding] --> 4个并行研究节点:
    |          - FinancialAnalyst (财务分析)
    |          - NewsScanner (新闻扫描)
    |          - IndustryAnalyzer (行业分析)
    |          - CompanyAnalyzer (公司分析)
    |
    v
[Collector]  ---> 收集并验证各研究数据
    |
    v
[Curator]   ---> 去重、筛选、相关性评分
    |
    v
[Enricher]  ---> 获取原始网页内容
    |
    v
[Briefing]  ---> 生成4类简报
    |
    v
[Editor]    ---> 整合为最终报告
    |
    v
输出 (Markdown格式研究报告 + 参考文献)
```

## 节点详解

| 节点 | 文件 | 职责 |
|------|------|------|
| **Grounding** | `nodes/grounding.py` | 爬取公司官网，收集初始信息作为研究基础 |
| **CompanyAnalyzer** | `nodes/researchers/company.py` | 研究公司业务、产品服务、合规体系、核心团队 |
| **FinancialAnalyst** | `nodes/researchers/financial.py` | 分析财务投入、研发成本、召回/罚款等财务相关事件 |
| **NewsScanner** | `nodes/researchers/news.py` | 扫描产品检测公告、合规动态、合作伙伴新闻 |
| **IndustryAnalyzer** | `nodes/researchers/industry.py` | 分析行业TIC需求、法规标准、市场竞争格局 |
| **Collector** | `nodes/collector.py` | 验证各类研究数据是否存在，汇总研究结果 |
| **Curator** | `nodes/curator.py` | URL去重、规范化，按相关性评分筛选高质量内容 |
| **Enricher** | `nodes/enricher.py` | 并行获取文档原始网页内容，丰富研究素材 |
| **Briefing** | `nodes/briefing.py` | 生成公司/行业/财务/新闻四类结构化简报 |
| **Editor** | `nodes/editor.py` | 整合所有简报为最终Markdown研究报告 |

## 数据流向

```
ResearchState 状态定义:

输入字段:
  - company: 公司名称
  - company_url: 公司官网URL
  - hq_location: 总部位置
  - industry: 所属行业
  - job_id: 任务ID

中间状态:
  - site_scrape: 官网爬取内容
  - xxx_data: 4类研究原始数据 (company/financial/news/industry)
  - curated_xxx_data: 整理后的研究数据
  - raw_content: 原始网页内容
  - xxx_briefing: 4类简报

输出:
  - report: 最终研究报告
  - references: Top 10 参考文献
```

## 技术栈

| 类别 | 技术/库 |
|------|--------|
| 工作流框架 | LangGraph (StateGraph) |
| LLM框架 | LangChain + LangChain-OpenAI |
| LLM提供商 | DashScope (Qwen系列) / OpenAI兼容API |
| 搜索爬取 | Tavily (AsyncTavilyClient) |
| Web框架 | FastAPI + Uvicorn |
| 数据库 | MongoDB (PyMongo) |
| 配置管理 | python-dotenv |

## 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| LLM模型 | qwen-plus / qwen-max | DashScope模型 |
| 搜索结果数 | 5条/次 | Tavily搜索限制 |
| 相关性阈值 | 0.4 | Tavily评分过滤 |
| 文档上限 | 30个/类 | 每类文档最大数量 |
| 参考文献 | Top 10 | 最终报告引用数 |
| Enricher并发 | 3批次×20URL | 并行获取网页内容 |

## 项目结构

```
backend/
├── graph.py              # LangGraph工作流定义
├── __init__.py           # 应用入口配置
├── classes/
│   └── state.py          # InputState/ResearchState定义
├── nodes/
│   ├── grounding.py
│   ├── collector.py
│   ├── curator.py
│   ├── enricher.py
│   ├── briefing.py
│   ├── editor.py
│   └── researchers/
│       ├── base.py       # BaseResearcher通用功能
│       ├── company.py
│       ├── financial.py
│       ├── industry.py
│       └── news.py
├── prompts.py            # 所有LLM提示词模板
└── utils/
    └── references.py     # 参考文献处理工具
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量 (.env)
cp .env.example .env
# 编辑 .env 设置 API keys

# 启动服务
python backend/__init__.py
```
