from typing import List
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager

from ragwatch import SpanKind, trace
from ragwatch.adapters.langgraph import tool as rw_tool


class ToolFactory:

    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()

    @trace("search-child-chunks", span_kind=SpanKind.RETRIEVER, auto_track_io=False)
    def _search_child_chunks(self, query: str, limit: int):
        """Search for the top K most relevant child chunks.

        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            results = self.collection.similarity_search_with_relevance_scores(
                query, k=limit, score_threshold=0.7
            )
            if not results:
                return "NO_RELEVANT_CHUNKS"
            return results
        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

    @rw_tool("retrieve-parent-chunks", auto_track_io=False)
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]):
        """Retrieve full parent chunks by their IDs.

        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"
            return raw_parents
        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"

    @rw_tool("retrieve-parent-chunk", auto_track_io=False)
    def _retrieve_parent_chunks(self, parent_id: str):
        """Retrieve a full parent chunk by its ID.

        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"
            return parent
        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"

    def create_tools(self) -> List:
        """Create and return the list of LangChain tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        return [search_tool, retrieve_tool]
