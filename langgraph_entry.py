# langgraph_entry.py - 直接运行 LangGraph Agent
import asyncio
from dotenv import load_dotenv

load_dotenv()

from backend.graph import Graph
from backend.classes.state import job_status


async def main():
    print("=" * 50)
    print("启动公司研究 Agent (LangGraph + Qwen)")
    print("=" * 50)

    # 初始化 Graph
    graph = Graph(
        company="Tesla",
        url="https://www.tesla.com",
        industry="Automotive",
        hq_location="Austin, TX",
        job_id="test-001"
    )
    # Register job_id for event tracking in CLI runs
    if graph.input_state.get("job_id"):
        job_id = graph.input_state["job_id"]
        job_status[job_id]  # touch to create entry
        job_status[job_id]["company"] = graph.input_state.get("company")
        job_status[job_id]["status"] = "running"

    # 编译 graph
    compiled_graph = graph.compile()

    print("\n开始执行工作流...\n")

    # 运行 graph
    last_state = None
    async for state in compiled_graph.astream(graph.input_state, thread={}):
        node_name = list(state.keys())[0] if state else "unknown"
        print(f"✅ 完成节点: {node_name}")
        if state and node_name in state:
            last_state = state[node_name]

    print("\n" + "=" * 50)
    print("研究完成!")
    print("=" * 50)
    if isinstance(last_state, dict):
        report = last_state.get("report") or (last_state.get("editor") or {}).get("report")
        if report:
            print("\n" + "=" * 50)
            print("最终报告输出:")
            print("=" * 50)
            print(report)
        else:
            print("\n未在最终状态中找到 report 字段，请检查节点输出或日志。")


if __name__ == "__main__":
    asyncio.run(main())
