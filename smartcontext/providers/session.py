from smartcontext.core.models import ConversationTurn


class SessionStore:
    """In-memory conversation history per session. Swap for Redis in production."""

    def __init__(self):
        self._sessions: dict[str, list[ConversationTurn]] = {}

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        return self._sessions.get(session_id, [])

    def add_turn(self, session_id: str, turn: ConversationTurn):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(turn)

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


session_store = SessionStore()
