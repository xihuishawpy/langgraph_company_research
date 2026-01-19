import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tavily import AsyncTavilyClient

from ...classes import ResearchState
from ...classes.state import job_status
from ...utils.references import clean_title
from ...prompts import QUERY_FORMAT_GUIDELINES

logger = logging.getLogger(__name__)

# Qwen (DashScope) API 配置
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus"  # 可选: qwen-turbo, qwen-plus, qwen-max


class BaseResearcher:
    """基础研究者类，提供研究节点的通用功能。

    子类包括：CompanyAnalyzer、FinancialAnalyst、IndustryAnalyzer、NewsScanner

    通用功能：
    - 生成搜索查询
    - 执行 Tavily 搜索
    - 处理搜索结果
    """

    def __init__(self):
        tavily_key = os.getenv("TAVILY_API_KEY")
        dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")

        if not tavily_key:
            raise ValueError("Missing TAVILY_API_KEY")
        if not dashscope_key:
            raise ValueError("Missing DASHSCOPE_API_KEY or OPENAI_API_KEY")

        self.tavily_client = AsyncTavilyClient(api_key=tavily_key)
        self.llm = ChatOpenAI(
            model=QWEN_MODEL,
            temperature=0,
            streaming=True,
            api_key=dashscope_key,
            base_url=DASHSCOPE_BASE_URL
        )
        self.analyst_type = "base_researcher"

    @property
    def analyst_type(self) -> str:
        if not hasattr(self, '_analyst_type'):
            raise ValueError("Analyst type not set by subclass")
        return self._analyst_type

    @analyst_type.setter
    def analyst_type(self, value: str):
        self._analyst_type = value

    async def generate_queries(self, state: Dict, prompt: str):
        """生成搜索查询并实时产出事件。

        使用 LLM 根据提示生成最多 4 个搜索查询。

        Args:
            state: 包含公司、行业、总部位置等信息的状态
            prompt: 生成查询的提示模板

        Yields:
            事件字典（query_generating、query_generated、queries_complete）
        """
        company = state.get("company", "Unknown Company")
        industry = state.get("industry", "Unknown Industry")
        hq_location = state.get("hq_location", "Unknown")
        current_year = datetime.now().year
        job_id = state.get("job_id")
        
        logger.info(f"=== GENERATE_QUERIES START: job_id={job_id}, analyst={self.analyst_type} ===")
        if not job_id:
            logger.warning(f"⚠️ NO JOB_ID in state! Keys: {list(state.keys())}")
        
        try:
            logger.info(f"Generating queries for {company} as {self.analyst_type}, job_id={job_id}")
            
            # Create prompt template using LangChain
            query_prompt = ChatPromptTemplate.from_messages([
                ("system", "你正在研究{company}，所属行业为{industry}，总部位于{hq_location}。请用中文输出。"),
                ("user", """研究对象：{company}（{year}，截至{date}）。
{task_prompt}
{format_guidelines}""")
            ])
            
            # Create LCEL chain
            chain = query_prompt | self.llm
            
            queries = []
            current_query = ""
            current_query_number = 1

            # Stream queries using LangChain's astream
            async for chunk in chain.astream({
                "company": company,
                "industry": industry,
                "hq_location": hq_location,
                "year": current_year,
                "date": datetime.now().strftime("%B %d, %Y"),
                "task_prompt": prompt,
                "format_guidelines": QUERY_FORMAT_GUIDELINES.format(company=company)
            }):
                current_query += chunk.content
                
                # Yield query generation progress
                event = {
                    "type": "query_generating",
                    "query": current_query,
                    "query_number": current_query_number,
                    "category": self.analyst_type
                }
                
                # Update job status if job_id provided
                if job_id:
                    try:
                        logger.info(f"job_id={job_id}, job_id in job_status={job_id in job_status}")
                        if job_id in job_status:
                            job_status[job_id]["events"].append(event)

                        else:
                            logger.warning(f"job_id {job_id} not found in job_status. Available keys: {list(job_status.keys())[:3]}")
                    except Exception as e:
                        logger.error(f"Error appending event: {e}")
                
                yield event
                
                # Parse completed queries on newline
                if '\n' in current_query:
                    parts = current_query.split('\n')
                    current_query = parts[-1]
                    
                    for query in parts[:-1]:
                        query = query.strip()
                        if query:
                            queries.append(query)
                            event = {
                                "type": "query_generated",
                                "query": query,
                                "query_number": len(queries),
                                "category": self.analyst_type
                            }
                            
                            # Update job status if job_id provided
                            if job_id:
                                try:
                                    if job_id in job_status:
                                        job_status[job_id]["events"].append(event)
                                    else:
                                        logger.warning(f"job_id {job_id} not found in job_status for query_generated")
                                except Exception as e:
                                    logger.error(f"Error appending query_generated event: {e}")
                            
                            yield event
                            current_query_number += 1

            # Add remaining query
            if current_query.strip():
                queries.append(current_query.strip())
                yield {
                    "type": "query_generated",
                    "query": current_query.strip(),
                    "query_number": len(queries),
                    "category": self.analyst_type
                }
            
            if not queries:
                raise ValueError(f"No queries generated for {company}")

            queries = queries[:4]  # Limit to 4 queries
            logger.info(f"Final queries for {self.analyst_type}: {queries}")
            
            yield {"type": "queries_complete", "queries": queries, "count": len(queries)}
            
        except Exception as e:
            logger.error(f"Error generating queries for {company}: {e}")
            raise RuntimeError(f"Fatal API error - query generation failed: {str(e)}") from e

    def _get_search_params(self) -> Dict[str, Any]:
        """根据分析器类型获取搜索参数。

        Returns:
            Dict: 搜索参数字典（search_depth、include_raw_content、max_results、topic）
        """
        params = {
            "search_depth": "basic",
            "include_raw_content": False,
            "max_results": 5
        }
        
        topic_map = {
            "news_analyzer": "news",
            "financial_analyzer": "finance"
        }
        
        if topic := topic_map.get(self.analyst_type):
            params["topic"] = topic
            
        return params
    
    def _process_search_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """将单个搜索结果处理为标准化格式。

        Args:
            result: 原始搜索结果
            query: 关联的搜索查询

        Returns:
            Dict: 标准化后的文档，包含 title、content、url、query、source、score
        """
        if not result.get("content") or not result.get("url"):
            return {}
            
        url = result.get("url")
        title = clean_title(result.get("title", "")) if result.get("title") else ""
        
        # Reset empty or invalid titles
        if not title or title.lower() == url.lower():
            title = ""
        
        return {
            "title": title,
            "content": result.get("content", ""),
            "query": query,
            "url": url,
            "source": "web_search",
            "score": result.get("score", 0.0)
        }

    async def search_documents(self, state: ResearchState, queries: List[str]):
        """并行执行所有 Tavily 搜索并产出事件。

        Args:
            state: 研究状态
            queries: 搜索查询列表

        Yields:
            事件字典（search_started、query_error、search_complete）
        """
        if not queries:
            logger.error("No valid queries to search")
            yield {"type": "error", "error": "No valid queries to search"}
            return

        # Yield start event
        yield {
            "type": "search_started",
            "message": f"Searching {len(queries)} queries",
            "total_queries": len(queries)
        }

        # Execute all searches in parallel
        search_params = self._get_search_params()
        search_tasks = [self.tavily_client.search(query, **search_params) for query in queries]

        try:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during parallel search execution: {e}")
            yield {"type": "error", "error": str(e)}
            return

        # Process and merge results
        merged_docs = {}
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for query '{query}': {result}")
                yield {"type": "query_error", "query": query, "error": str(result)}
                continue
                
            for item in result.get("results", []):
                if doc := self._process_search_result(item, query):
                    merged_docs[doc["url"]] = doc

        # Yield completion event
        yield {
            "type": "search_complete",
            "message": f"Found {len(merged_docs)} documents",
            "total_documents": len(merged_docs),
            "queries_processed": len(queries),
            "merged_docs": merged_docs
        }
