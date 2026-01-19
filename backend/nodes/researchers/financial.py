from langchain_core.messages import AIMessage

from ...classes import ResearchState
from ...prompts import FINANCIAL_ANALYZER_QUERY_PROMPT
from .base import BaseResearcher


class FinancialAnalyst(BaseResearcher):
    """财务分析师，负责收集和分析公司财务与合规相关信息。

    研究质量投入、检测费用、认证成本、召回/罚款等信息。
    """

    def __init__(self) -> None:
        """初始化 FinancialAnalyst。"""
        super().__init__()
        self.analyst_type = "financial_analyzer"

    async def analyze(self, state: ResearchState):
        """分析财务信息并产出事件。

        执行以下操作：
        1. 生成财务分析相关的搜索查询
        2. 搜索并收集相关文档
        3. 更新状态中的 financial_data

        Args:
            state: 当前研究状态

        Yields:
            事件字典（query_generating、queries_complete、search_complete、analysis_complete）
        """
        company = state.get('company', 'Unknown Company')
        
        # Generate search queries and yield events
        queries = []
        async for event in self.generate_queries(state, FINANCIAL_ANALYZER_QUERY_PROMPT):
            yield event
            if event.get("type") == "queries_complete":
                queries = event.get("queries", [])
        
        # Log subqueries
        subqueries_msg = "🔍 Subqueries for financial analysis:\n" + "\n".join([f"• {query}" for query in queries])
        state.setdefault('messages', []).append(AIMessage(content=subqueries_msg))
        
        # Start with site scrape data
        financial_data = dict(state.get('site_scrape', {}))
        
        # Search and merge documents, yielding events
        documents = {}
        async for event in self.search_documents(state, queries):
            yield event
            if event.get("type") == "search_complete":
                documents = event.get("merged_docs", {})
        
        financial_data.update(documents)
        
        # Update state
        completion_msg = f"💰 Financial Analyst found {len(financial_data)} documents for {company}"
        state.setdefault('messages', []).append(AIMessage(content=completion_msg))
        state['financial_data'] = financial_data
        
        yield {"type": "analysis_complete", "data_type": "financial_data", "count": len(financial_data)}
        yield {'message': [completion_msg], 'financial_data': financial_data}

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
            if "message" in event or "financial_data" in event:
                result = event
        yield result or {}