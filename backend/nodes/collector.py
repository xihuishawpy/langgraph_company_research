from langchain_core.messages import AIMessage

from ..classes import ResearchState


class Collector:
    """收集节点，负责收集和验证所有研究数据。

    在整理(curation)之前检查各类研究数据是否存在。
    """

    async def collect(self, state: ResearchState) -> ResearchState:
        """收集并验证所有研究数据是否存在。

        检查财务、新闻、行业和公司数据四类研究数据，
        并更新状态消息记录收集结果。

        Args:
            state: 当前研究状态

        Returns:
            ResearchState: 更新后的研究状态，包含收集消息
        """
        company = state.get('company', 'Unknown Company')
        msg = [f"📦 Collecting research data for {company}:"]
        
        # Check each type of research data
        research_types = {
            'financial_data': '💰 Financial',
            'news_data': '📰 News',
            'industry_data': '🏭 Industry',
            'company_data': '🏢 Company'
        }
        
        for data_field, label in research_types.items():
            data = state.get(data_field, {})
            if data:
                msg.append(f"• {label}: {len(data)} documents collected")
            else:
                msg.append(f"• {label}: No data found")
        
        # Update state with collection message
        state.setdefault('messages', []).append(AIMessage(content="\n".join(msg)))
        
        return state

    async def run(self, state: ResearchState) -> ResearchState:
        """运行收集节点。

        执行数据收集操作。

        Args:
            state: 当前研究状态

        Returns:
            ResearchState: 更新后的研究状态
        """
        return await self.collect(state)
