import pytest
from smartcontext.hooks import HookManager, EventType, Event


class TestHookManager:
    @pytest.mark.asyncio
    async def test_sync_callback(self):
        events = []
        manager = HookManager()
        manager.on(EventType.INTENT_PREDICTED, lambda e: events.append(e))

        await manager.emit(Event(type=EventType.INTENT_PREDICTED, data={"test": True}))

        assert len(events) == 1
        assert events[0].data["test"] is True

    @pytest.mark.asyncio
    async def test_async_callback(self):
        events = []
        manager = HookManager()

        async def async_handler(e: Event):
            events.append(e)

        manager.on(EventType.RESPONSE_GENERATED, async_handler)
        await manager.emit(Event(type=EventType.RESPONSE_GENERATED, data={"resp": "ok"}))

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        count = [0]
        manager = HookManager()
        manager.on(EventType.INTENT_PREDICTED, lambda e: count.__setitem__(0, count[0] + 1))
        manager.on(EventType.INTENT_PREDICTED, lambda e: count.__setitem__(0, count[0] + 1))

        await manager.emit(Event(type=EventType.INTENT_PREDICTED))

        assert count[0] == 2

    @pytest.mark.asyncio
    async def test_off_removes_callback(self):
        events = []
        manager = HookManager()
        handler = lambda e: events.append(e)
        manager.on(EventType.INTENT_PREDICTED, handler)
        manager.off(EventType.INTENT_PREDICTED, handler)

        await manager.emit(Event(type=EventType.INTENT_PREDICTED))

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_wrong_event_type_ignored(self):
        events = []
        manager = HookManager()
        manager.on(EventType.INTENT_PREDICTED, lambda e: events.append(e))

        await manager.emit(Event(type=EventType.ERROR))

        assert len(events) == 0

    def test_chaining(self):
        manager = HookManager()
        result = manager.on(EventType.INTENT_PREDICTED, lambda e: None)
        assert result is manager

    @pytest.mark.asyncio
    async def test_clear(self):
        events = []
        manager = HookManager()
        manager.on(EventType.INTENT_PREDICTED, lambda e: events.append(e))
        manager.clear()

        await manager.emit(Event(type=EventType.INTENT_PREDICTED))
        assert len(events) == 0
