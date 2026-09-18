import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from market_agent import graph


def test_market_agent():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "messages": [HumanMessage(content="2026 latest AI agent trends and market opportunities")]
    }

    print(f"--- 开始运行市场情报 Agent (Thread ID: {thread_id}) ---")

    try:
        # 1. 第一次运行，会在 human_review 处中断
        graph.invoke(initial_input, config=config)
    except Exception as e:
        print(f"工作流在审核节点暂停: {e}")

    # 2. 检查当前状态并自动模拟“同意（Approved）”
    current_state = graph.get_state(config)
    if current_state.next:
        print("--- 自动模拟人工审核：批准交付并发送邮件 ---")
        resume_payload = {"action": "approved", "comment": "Auto approved by test script"}

        # 3. 恢复图的运行，这会直接触发 deliver_node 发邮件
        final_output = graph.invoke(Command(resume=resume_payload), config=config)
        print("--- 工作流执行完毕，邮件应该已经发出 ---")


if __name__ == "__main__":
    test_market_agent()