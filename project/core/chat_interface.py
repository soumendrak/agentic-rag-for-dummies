from langchain_core.messages import HumanMessage

# ── RAGWatch: root span for every user query ──────────────────────────────────
from ragwatch.adapters.langgraph import workflow
# ─────────────────────────────────────────────────────────────────────────────


class ChatInterface:

    def __init__(self, rag_system):
        self.rag_system = rag_system

    @workflow("query-execution")
    def chat(self, message, history):
        """Each call becomes the root span; all @node spans nest inside it,
        giving exactly one trace per user query in Jaeger and Phoenix."""

        if not self.rag_system.agent_graph:
            return "⚠️ System not initialized!"

        try:
            result = self.rag_system.agent_graph.invoke(
                {"messages": [HumanMessage(content=message.strip())]},
                self.rag_system.get_config()
            )
            return result["messages"][-1].content

        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def clear_session(self):
        self.rag_system.reset_thread()