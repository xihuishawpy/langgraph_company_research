from langchain_core.messages import AIMessage

from ...classes import ResearchState
from ...prompts import INDUSTRY_ANALYZER_QUERY_PROMPT
from .base import BaseResearcher


class IndustryAnalyzer(BaseResearcher):
    """行业分析师，负责收集和分析行业相关信息。

    研究行业 TIC 需求格局、法规标准、检测场景、TIC 服务商等。
    """

    def __init__(self) -> None:
        """初始化 IndustryAnalyzer。"""
        super().__init__()
        self.analyst_type = "industry_analyzer"

    async def analyze(self, state: ResearchState):
        """分析行业信息并产出事件。

        执行以下操作：
        1. 生成行业分析相关的搜索查询
        2. 搜索并收集相关文档
        3. 更新状态中的 industry_data

        Args:
            state: 当前研究状态

        Yields:
            事件字典（query_generating、queries_complete、search_complete、analysis_complete）
        """
        company = state.get('company', 'Unknown Company')
        industry = state.get('industry', 'Unknown Industry')
        
        # Generate search queries and yield events
        queries = []
        async for event in self.generate_queries(state, INDUSTRY_ANALYZER_QUERY_PROMPT):
            yield event
            if event.get("type") == "queries_complete":
                queries = event.get("queries", [])
        
        # Log subqueries
        subqueries_msg = "🔍 Subqueries for industry analysis:\n" + "\n".join([f"• {query}" for query in queries])
        state.setdefault('messages', []).append(AIMessage(content=subqueries_msg))
        
        # Start with site scrape data
        industry_data = dict(state.get('site_scrape', {}))
        
        # Search and merge documents, yielding events
        documents = {}
        async for event in self.search_documents(state, queries):
            yield event
            if event.get("type") == "search_complete":
                documents = event.get("merged_docs", {})
        
        industry_data.update(documents)
        
        # Update state
        completion_msg = f"🏭 Industry Analyzer found {len(industry_data)} documents for {company} in {industry}"
        state.setdefault('messages', []).append(AIMessage(content=completion_msg))
        state['industry_data'] = industry_data
        
        yield {"type": "analysis_complete", "data_type": "industry_data", "count": len(industry_data)}
        yield {'message': [completion_msg], 'industry_data': industry_data}

    async def run(self, state: ResearchState):
        """运行分析器并产出所有事件。

        Args:
            state: 当前研究状态

        Yields:
            事件字典和最终结果
        """
        result = None
        async for event in self.analyze(state):
            yield event
            if "message" in event or "industry_data" in event:
                result = event
        yield result or {} 