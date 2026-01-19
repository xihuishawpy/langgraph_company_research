from typing import Any


from langchain_core.messages import AIMessage

from ...classes import ResearchState
from ...prompts import NEWS_SCANNER_QUERY_PROMPT
from .base import BaseResearcher


class NewsScanner(BaseResearcher):
    """新闻扫描器，负责收集和分析公司相关的最新新闻动态。

    研究产品检测公告、合规动态、处罚信息、合作新闻等。
    """

    def __init__(self) -> None:
        """初始化 NewsScanner。"""
        super().__init__()
        self.analyst_type = "news_analyzer"

    async def analyze(self, state: ResearchState):
        """分析新闻信息并产出事件。

        执行以下操作：
        1. 生成新闻相关的搜索查询
        2. 搜索并收集相关文档
        3. 更新状态中的 news_data

        Args:
            state: 当前研究状态

        Yields:
            事件字典（query_generating、queries_complete、search_complete、analysis_complete）
        """
        company = state.get('company', 'Unknown Company')
        
        # Generate search queries and yield events
        queries = []
        async for event in self.generate_queries(state, NEWS_SCANNER_QUERY_PROMPT):
            yield event
            if event.get("type") == "queries_complete":
                queries = event.get("queries", [])
        
        # Log subqueries
        subqueries_msg = "🔍 Subqueries for news analysis:\n" + "\n".join([f"• {query}" for query in queries])
        state.setdefault('messages', []).append(AIMessage(content=subqueries_msg))
        
        # Start with site scrape data
        news_data = dict[str, Any](state.get('site_scrape', {}))
        
        # Search and merge documents, yielding events
        documents = {}
        async for event in self.search_documents(state, queries):
            yield event
            if event.get("type") == "search_complete":
                documents = event.get("merged_docs", {})
        
        news_data.update(documents)
        
        # Update state
        completion_msg = f"📰 News Scanner found {len(news_data)} documents for {company}"
        state.setdefault('messages', []).append(AIMessage(content=completion_msg))
        state['news_data'] = news_data
        
        yield {"type": "analysis_complete", "data_type": "news_data", "count": len(news_data)}
        yield {'message': [completion_msg], 'news_data': news_data}

    async def run(self, state: ResearchState):
        """运行扫描器并产出所有事件。

        Args:
            state: 当前研究状态

        Yields:
            事件字典和最终结果
        """
        result = None
        async for event in self.analyze(state):
            yield event
            if "message" in event or "news_data" in event:
                result = event
        yield result or {} 