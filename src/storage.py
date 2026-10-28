"""SQLite持久化：短连接、参数化SQL、成功的一轮问答原子写入。"""
import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions(
                    id TEXT PRIMARY KEY, model_mode TEXT NOT NULL CHECK(model_mode IN ('base','lora')),
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS messages_session_order ON messages(session_id,id);
                CREATE TABLE IF NOT EXISTS character_cards(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, card_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS voice_profiles(
                    character_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
                    voice_style TEXT NOT NULL, reference_text TEXT NOT NULL DEFAULT '',
                    reference_audio_path TEXT, language TEXT NOT NULL DEFAULT 'zh',
                    updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS session_memories(
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS session_memory_versions(
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(session_id,revision));
            ''')
            columns={r['name'] for r in db.execute('PRAGMA table_info(sessions)')}
            if 'character_id' not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN character_id TEXT NOT NULL DEFAULT 'assistant'")
            if 'model_id' not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN model_id TEXT NOT NULL DEFAULT 'Qwen/Qwen3-0.6B'")
            db.execute('''INSERT OR IGNORE INTO voice_profiles
                          (character_id,display_name,voice_style,reference_text,reference_audio_path,language,updated_at)
                          VALUES(?,?,?,?,?,?,?)''',
                       ('companion','小满 · 日常音色','温柔、自然、轻柔，语速适中','',None,'zh',now()))

    @contextmanager
    def connect(self):
        # 每次操作在自己的线程创建连接，不跨线程共享SQLite连接。
        db=sqlite3.connect(self.path,timeout=5)
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create(self,mode,character_id='assistant',model_id='Qwen/Qwen3-0.6B'):
        sid=str(uuid4());timestamp=now()
        with self.connect() as db:
            db.execute('INSERT INTO sessions(id,model_mode,created_at,updated_at,character_id,model_id) VALUES(?,?,?,?,?,?)',
                       (sid,mode,timestamp,timestamp,character_id,model_id))
        return self.get(sid)

    def get(self,sid):
        with self.connect() as db:
            row=db.execute('SELECT * FROM sessions WHERE id=?',(sid,)).fetchone()
            if row is None:
                raise KeyError(sid)
            result=dict(row)
            result['messages']=[dict(r) for r in db.execute(
                'SELECT id,role,content,created_at FROM messages WHERE session_id=? ORDER BY id',(sid,))]
            return result

    def list(self,limit=50,offset=0,nonempty=False,query=''):
        conditions=[];params=[]
        if nonempty:
            conditions.append('EXISTS (SELECT 1 FROM messages WHERE session_id=sessions.id)')
        if query:
            # instr treats %, _ and quotes as literal text, not LIKE patterns or SQL.
            conditions.append('EXISTS (SELECT 1 FROM messages WHERE session_id=sessions.id AND instr(lower(content),lower(?))>0)')
            params.append(query)
        where=' WHERE '+' AND '.join(conditions) if conditions else ''
        with self.connect() as db:
            rows=[dict(r) for r in db.execute('''
                SELECT sessions.*,
                    (SELECT count(*) FROM messages WHERE session_id=sessions.id) AS message_count,
                    (SELECT substr(content,1,160) FROM messages
                     WHERE session_id=sessions.id AND role='user' ORDER BY id LIMIT 1) AS title
                FROM sessions'''+where+' ORDER BY updated_at DESC,id LIMIT ? OFFSET ?',(*params,limit,offset))]
            for row in rows:
                row['title']=' '.join((row['title'] or '新对话').split())[:40]
            return rows

    def save_turn(self,sid,question,answer):
        timestamp=now()
        with self.connect() as db:
            db.execute('INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)',
                       (sid,'user',question,timestamp))
            db.execute('INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)',
                       (sid,'assistant',answer,timestamp))
            db.execute('UPDATE sessions SET updated_at=? WHERE id=?',(timestamp,sid))

    def clear(self,sid):
        with self.connect() as db:
            if db.execute('SELECT id FROM sessions WHERE id=?',(sid,)).fetchone() is None:
                raise KeyError(sid)
            db.execute('DELETE FROM messages WHERE session_id=?',(sid,))
            db.execute('DELETE FROM session_memories WHERE session_id=?',(sid,))
            db.execute('DELETE FROM session_memory_versions WHERE session_id=?',(sid,))
            db.execute('UPDATE sessions SET updated_at=? WHERE id=?',(now(),sid))

    def delete_session(self,sid):
        with self.connect() as db:
            cursor=db.execute('DELETE FROM sessions WHERE id=?',(sid,))
            if cursor.rowcount == 0:
                raise KeyError(sid)

    def get_memory(self,sid):
        with self.connect() as db:
            if db.execute('SELECT id FROM sessions WHERE id=?',(sid,)).fetchone() is None:
                raise KeyError(sid)
            row=db.execute('SELECT payload,revision FROM session_memories WHERE session_id=?',(sid,)).fetchone()
            result=json.loads(row['payload']) if row else dict(items=[],covered_message_id=0,notes='',auto_enabled=True)
            result['revision']=row['revision'] if row else 0
            return result

    def save_memory(self,sid,payload,expected_revision=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT id FROM sessions WHERE id=?',(sid,)).fetchone() is None:
                raise KeyError(sid)
            row=db.execute('SELECT payload,revision FROM session_memories WHERE session_id=?',(sid,)).fetchone()
            revision=row['revision'] if row else 0
            if expected_revision is not None and revision!=expected_revision:
                raise ValueError('记忆已更新，请重新打开后再保存。')
            previous=json.loads(row['payload']) if row else {}
            result={**dict(items=[],covered_message_id=0,notes='',auto_enabled=True),**previous,**payload}
            result.pop('revision',None)
            encoded=json.dumps(result,ensure_ascii=False)
            if row:
                db.execute('INSERT OR REPLACE INTO session_memory_versions VALUES(?,?,?)',(sid,revision,row['payload']))
            db.execute('INSERT INTO session_memories VALUES(?,?,?) ON CONFLICT(session_id) DO UPDATE SET payload=excluded.payload,revision=excluded.revision',
                       (sid,encoded,revision+1))
            db.execute('DELETE FROM session_memory_versions WHERE session_id=? AND revision<?',(sid,max(0,revision-9)))
        return {**result,'revision':revision+1}

    def restore_memory(self,sid):
        current=self.get_memory(sid)
        with self.connect() as db:
            row=db.execute('SELECT payload FROM session_memory_versions WHERE session_id=? ORDER BY revision DESC LIMIT 1',(sid,)).fetchone()
        if row is None:
            raise ValueError('没有可恢复的旧摘要。')
        return self.save_memory(sid,json.loads(row['payload']),expected_revision=current['revision'])

    def get_voice_profile(self,character_id):
        with self.connect() as db:
            row=db.execute('SELECT * FROM voice_profiles WHERE character_id=?',(character_id,)).fetchone()
            return dict(row) if row else None

    def save_voice_profile(self,character_id,display_name,voice_style,reference_text,language='zh'):
        timestamp=now()
        with self.connect() as db:
            db.execute('''INSERT INTO voice_profiles
                          (character_id,display_name,voice_style,reference_text,reference_audio_path,language,updated_at)
                          VALUES(?,?,?,?,NULL,?,?) ON CONFLICT(character_id) DO UPDATE SET
                          display_name=excluded.display_name,voice_style=excluded.voice_style,
                          reference_text=excluded.reference_text,language=excluded.language,updated_at=excluded.updated_at''',
                       (character_id,display_name,voice_style,reference_text,language,timestamp))
            row=db.execute('SELECT * FROM voice_profiles WHERE character_id=?',(character_id,)).fetchone()
            return dict(row)

    def set_voice_reference_audio(self,character_id,path):
        with self.connect() as db:
            cursor=db.execute('UPDATE voice_profiles SET reference_audio_path=?,updated_at=? WHERE character_id=?',
                               (str(path),now(),character_id))
            if cursor.rowcount == 0:
                raise KeyError(character_id)
            row=db.execute('SELECT * FROM voice_profiles WHERE character_id=?',(character_id,)).fetchone()
            return dict(row)

    def save_character_card(self,card):
        """保存已由作者审核的自定义角色卡。"""
        payload=json.dumps(card,ensure_ascii=False,separators=(',',':'))
        timestamp=now()
        with self.connect() as db:
            db.execute('''INSERT INTO character_cards(id,name,card_json,created_at,updated_at)
                          VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                          name=excluded.name,card_json=excluded.card_json,updated_at=excluded.updated_at''',
                       (card['id'],card['name'],payload,timestamp,timestamp))
        return self.get_character_card(card['id'])

    def get_character_card(self,character_id):
        with self.connect() as db:
            row=db.execute('SELECT card_json FROM character_cards WHERE id=?',(character_id,)).fetchone()
            if row is None:
                raise KeyError(character_id)
            return json.loads(row['card_json'])

    def list_character_cards(self):
        with self.connect() as db:
            rows=db.execute('SELECT card_json FROM character_cards ORDER BY created_at,id').fetchall()
            return [json.loads(row['card_json']) for row in rows]

    def delete_character_card(self,character_id):
        with self.connect() as db:
            cursor=db.execute('DELETE FROM character_cards WHERE id=?',(character_id,))
            if cursor.rowcount == 0:
                raise KeyError(character_id)
