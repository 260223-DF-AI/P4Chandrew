"""
ResearchFlow — Cross-Thread Memory (Store Interface)

Manages user preferences and query history across threads
using the LangGraph Store interface with namespaces and scopes.
"""
from langgraph.store.memory import InMemoryStore


# Module-level singleton — one Store across the whole process.
# In Lambda you would swap this for PostgresStore so memory survives
# between invocations.
_store = InMemoryStore()

DEFAULT_PREFERENCES = {
    "verbosity": "normal",         # "concise" | "normal" | "verbose"
    "trusted_sources": [],
}


def get_user_preferences(user_id: str) -> dict:
    """
    Retrieve stored preferences for a user from the Store.

    TODO:
    - Use the Store interface with namespace = ("users", user_id).
    - Return a dict of preferences (verbosity, trusted sources, etc.).
    - Return sensible defaults if no preferences exist.
    """
    namespace = ("users", user_id)
    item = _store.get(namespace, "preferences")
    return item.value if item else dict(DEFAULT_PREFERENCES)


def save_user_preferences(user_id: str, preferences: dict) -> None:
    """
    Persist user preferences to the Store.

    TODO:
    - Write to the Store under the user's namespace.
    """
    _store.put(("users", user_id), "preferences", preferences)


def get_query_history(user_id: str, limit: int = 5) -> list[str]:
    """
    Retrieve recent query history for dynamic few-shot prompting.

    TODO:
    - Read from the Store under a "history" scope.
    - Return the most recent `limit` queries.
    """
    raise NotImplementedError


def append_query(user_id: str, question: str) -> None:
    """
    Append a query to the user's history in the Store.

    TODO:
    - Write the new query to the Store.
    """
    raise NotImplementedError
