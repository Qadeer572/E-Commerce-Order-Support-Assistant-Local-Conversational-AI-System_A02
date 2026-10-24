from backend.conversation_manager.session import Session
from backend import history_store


def test_reset_clears_summary_cursor_and_entities():
    s=Session(conversation_summary='old',last_order_id='ORD-10001')
    s.summarized_turn_count=4
    s.reset()
    assert s.conversation_summary is None
    assert s.summarized_turn_count==0
    assert s.last_order_id is None


def test_persist_and_restore_working_state_without_losing_transcript(tmp_path,monkeypatch):
    monkeypatch.setattr(history_store,'HISTORY_FILE',tmp_path/'history.json')
    s=Session(last_order_id='ORD-10001',conversation_summary='summary')
    s.add_turn('user','order'); s.add_turn('assistant','reply')
    s.summarized_turn_count=2
    history_store.save_session(s)
    restored=Session(session_id=s.session_id)
    history_store.restore_session(restored)
    assert len(restored.history)==2
    assert restored.conversation_summary=='summary'
    assert restored.last_order_id=='ORD-10001'
    assert restored.summarized_turn_count==2
def test_history_replace_retries_transient_windows_file_lock(tmp_path, monkeypatch):
    from pathlib import Path
    from backend import history_store
    from backend.conversation_manager.session import Session
    target = tmp_path / 'history.json'
    monkeypatch.setattr(history_store, 'HISTORY_FILE', target)
    original = Path.replace
    attempts = []
    def replace(path, destination):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError('Windows reader temporarily holds file')
        return original(path, destination)
    monkeypatch.setattr(Path, 'replace', replace)
    s = Session()
    s.add_turn('user', 'hello')
    history_store.save_session(s)
    assert history_store.get_session_messages(s.session_id)[0]['content'] == 'hello'
    assert len(attempts) == 2

