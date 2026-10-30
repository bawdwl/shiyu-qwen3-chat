"""SQLite持久化：短连接、参数化SQL、成功的一轮问答原子写入。"""
import sqlite3
import json
from contextlib import contextmanager
from datetime import date, datetime, timezone
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
                    content TEXT NOT NULL,
                    display_json TEXT,
                    created_at TEXT NOT NULL);
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
                CREATE TABLE IF NOT EXISTS role_conversations(
                    character_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS app_settings(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS model_usage(
                    request_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    session_id TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_tokens INTEGER NOT NULL DEFAULT 0,
                    usage_day TEXT NOT NULL,
                    created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS model_usage_day
                    ON model_usage(provider,usage_day);
                CREATE TABLE IF NOT EXISTS uploaded_images(
                    id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                    message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                    file_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS character_state(
                    character_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS long_term_memories(
                    id TEXT PRIMARY KEY,
                    character_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_message_ids TEXT NOT NULL DEFAULT '[]',
                    confidence REAL,
                    importance REAL,
                    valid_from TEXT,
                    valid_to TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS long_term_memories_character
                    ON long_term_memories(character_id,status,updated_at);
                CREATE TABLE IF NOT EXISTS state_transitions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    old_payload TEXT,
                    new_payload TEXT,
                    reason TEXT NOT NULL,
                    source_message_ids TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS character_assets(
                    id TEXT PRIMARY KEY,
                    character_id TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    prompt TEXT NOT NULL DEFAULT '',
                    seed INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS character_assets_character
                    ON character_assets(character_id,asset_type,created_at);
            ''')
            columns={r['name'] for r in db.execute('PRAGMA table_info(sessions)')}
            if 'character_id' not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN character_id TEXT NOT NULL DEFAULT 'assistant'")
            if 'model_id' not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN model_id TEXT NOT NULL DEFAULT 'Qwen/Qwen3-0.6B'")
            if 'provider' not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN provider TEXT NOT NULL DEFAULT 'local'")
            message_columns={r['name'] for r in db.execute('PRAGMA table_info(messages)')}
            if 'display_json' not in message_columns:
                db.execute("ALTER TABLE messages ADD COLUMN display_json TEXT")
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

    def create(self,mode,character_id='assistant',model_id='Qwen/Qwen3-0.6B',provider='local'):
        sid=str(uuid4());timestamp=now()
        with self.connect() as db:
            db.execute('''INSERT INTO sessions
                          (id,model_mode,created_at,updated_at,character_id,model_id,provider)
                          VALUES(?,?,?,?,?,?,?)''',
                       (sid,mode,timestamp,timestamp,character_id,model_id,provider))
        return self.get(sid)

    def get_or_create_role_conversation(self,character_id,mode='base',
                                        model_id='deepseek-flash',provider='deepseek'):
        """Return the single persistent conversation for one character.

        Older versions could create many sessions per character. They remain
        readable through the legacy session APIs, while this mapping is the
        authoritative one-conversation-per-role record used by new clients.
        """
        timestamp=now()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('''SELECT session_id FROM role_conversations
                              WHERE character_id=?''',(character_id,)).fetchone()
            if row:
                sid=row['session_id']
                db.execute(
                    '''UPDATE sessions
                       SET provider=?,model_id=?,updated_at=?
                       WHERE id=?''',
                    (provider,model_id,timestamp,sid),
                )
                db.execute('UPDATE role_conversations SET updated_at=? WHERE character_id=?',
                           (timestamp,character_id))
            else:
                sid=str(uuid4())
                db.execute('''INSERT INTO sessions
                              (id,model_mode,created_at,updated_at,character_id,model_id,provider)
                              VALUES(?,?,?,?,?,?,?)''',
                           (sid,mode,timestamp,timestamp,character_id,model_id,provider))
                db.execute('''INSERT INTO role_conversations
                              (character_id,session_id,created_at,updated_at)
                              VALUES(?,?,?,?)''',
                           (character_id,sid,timestamp,timestamp))
        return self.get(sid)

    def get_role_conversation(self,character_id):
        with self.connect() as db:
            row=db.execute('''SELECT session_id FROM role_conversations
                              WHERE character_id=?''',(character_id,)).fetchone()
            if row is None:
                return None
            return self.get(row['session_id'])

    def clear_role_conversation(self,character_id):
        current=self.get_role_conversation(character_id)
        if current is None:
            raise KeyError(character_id)
        with self.connect() as db:
            db.execute('DELETE FROM messages WHERE session_id=?',(current['id'],))
            db.execute('DELETE FROM session_memories WHERE session_id=?',(current['id'],))
            db.execute('DELETE FROM session_memory_versions WHERE session_id=?',(current['id'],))
            db.execute('DELETE FROM uploaded_images WHERE session_id=?',(current['id'],))
            db.execute('DELETE FROM character_state WHERE character_id=?',(character_id,))
            db.execute('DELETE FROM long_term_memories WHERE character_id=?',(character_id,))
            db.execute('DELETE FROM state_transitions WHERE character_id=?',(character_id,))
            db.execute('DELETE FROM character_assets WHERE character_id=?',(character_id,))
            db.execute('UPDATE sessions SET updated_at=? WHERE id=?',(now(),current['id']))
            db.execute('UPDATE role_conversations SET updated_at=? WHERE character_id=?',
                       (now(),character_id))
        return self.get(current['id'])

    def get(self,sid):
        with self.connect() as db:
            row=db.execute('SELECT * FROM sessions WHERE id=?',(sid,)).fetchone()
            if row is None:
                raise KeyError(sid)
            result=dict(row)
            result['messages']=[dict(r) for r in db.execute(
                '''SELECT m.id,m.role,m.content,m.display_json,m.created_at,
                          i.id AS image_id,i.mime_type AS image_mime_type,
                          i.width AS image_width,i.height AS image_height
                   FROM messages m
                   LEFT JOIN uploaded_images i ON i.message_id=m.id
                   WHERE m.session_id=? ORDER BY m.id''',(sid,))]
            for message in result['messages']:
                raw_display=message.pop('display_json',None)
                if raw_display:
                    try:
                        message['display']=json.loads(raw_display)
                    except (TypeError,ValueError,json.JSONDecodeError):
                        message['display']=None
                else:
                    message['display']=None
                if message.get('image_id'):
                    message['image']={
                        'id':message.pop('image_id'),
                        'mime_type':message.pop('image_mime_type'),
                        'width':message.pop('image_width'),
                        'height':message.pop('image_height'),
                    }
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

    def save_turn(self,sid,question,answer,image_id=None,display=None):
        timestamp=now()
        with self.connect() as db:
            user_cursor=db.execute(
                'INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)',
                (sid,'user',question,timestamp))
            if image_id:
                cursor=db.execute(
                    '''UPDATE uploaded_images SET message_id=?
                       WHERE id=? AND session_id=? AND message_id IS NULL''',
                    (user_cursor.lastrowid,image_id,sid))
                if cursor.rowcount!=1:
                    raise ValueError('图片附件不存在、已使用或不属于当前对话')
            assistant_cursor=db.execute(
                '''INSERT INTO messages(
                       session_id,role,content,display_json,created_at)
                   VALUES(?,?,?,?,?)''',
                (
                    sid,'assistant',answer,
                    json.dumps(display,ensure_ascii=False) if display else None,
                    timestamp,
                ),
            )
            db.execute('UPDATE sessions SET updated_at=? WHERE id=?',(timestamp,sid))
            return {
                'user_message_id':user_cursor.lastrowid,
                'assistant_message_id':assistant_cursor.lastrowid,
            }

    def ensure_opening_message(self,sid,content):
        """Persist one role-play opening only when the conversation is empty."""
        opening=str(content or '').strip()
        if not opening:
            return False
        timestamp=now()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute(
                'SELECT 1 FROM messages WHERE session_id=? LIMIT 1',
                (sid,),
            ).fetchone() is not None:
                return False
            if db.execute(
                'SELECT 1 FROM sessions WHERE id=?',
                (sid,),
            ).fetchone() is None:
                raise KeyError(sid)
            db.execute(
                '''INSERT INTO messages(session_id,role,content,created_at)
                   VALUES(?,?,?,?)''',
                (sid,'assistant',opening,timestamp),
            )
            db.execute(
                'UPDATE sessions SET updated_at=? WHERE id=?',
                (timestamp,sid),
            )
            return True

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
            db.execute('DELETE FROM role_conversations WHERE session_id=?',(sid,))
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
            db.execute('DELETE FROM role_conversations WHERE character_id=?',(character_id,))
            db.execute('DELETE FROM character_state WHERE character_id=?',(character_id,))
            db.execute('DELETE FROM long_term_memories WHERE character_id=?',(character_id,))
            db.execute('DELETE FROM state_transitions WHERE character_id=?',(character_id,))
            db.execute('DELETE FROM character_assets WHERE character_id=?',(character_id,))
            cursor=db.execute('DELETE FROM character_cards WHERE id=?',(character_id,))
            if cursor.rowcount == 0:
                raise KeyError(character_id)

    def get_setting(self,key,default=None):
        with self.connect() as db:
            row=db.execute('SELECT value FROM app_settings WHERE key=?',(key,)).fetchone()
            if row is None:
                return default
            try:
                return json.loads(row['value'])
            except json.JSONDecodeError:
                return row['value']

    def set_setting(self,key,value):
        encoded=json.dumps(value,ensure_ascii=False,separators=(',',':'))
        with self.connect() as db:
            db.execute('''INSERT INTO app_settings(key,value,updated_at)
                          VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET
                          value=excluded.value,updated_at=excluded.updated_at''',
                       (key,encoded,now()))
        return value

    def delete_setting(self,key):
        with self.connect() as db:
            cursor=db.execute('DELETE FROM app_settings WHERE key=?',(key,))
            return cursor.rowcount>0

    def record_usage(self,request_id,provider,model,session_id,prompt_tokens,
                     completion_tokens,cached_tokens=0,usage_day=None):
        usage_day=usage_day or date.today().isoformat()
        with self.connect() as db:
            db.execute('''INSERT INTO model_usage
                          (request_id,provider,model,session_id,prompt_tokens,
                           completion_tokens,cached_tokens,usage_day,created_at)
                          VALUES(?,?,?,?,?,?,?,?,?)''',
                       (request_id,provider,model,session_id,int(prompt_tokens),
                        int(completion_tokens),int(cached_tokens),usage_day,now()))
        return self.usage_for_day(provider,usage_day)

    def usage_for_day(self,provider,usage_day=None):
        usage_day=usage_day or date.today().isoformat()
        with self.connect() as db:
            row=db.execute('''SELECT
                                COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,
                                COALESCE(SUM(completion_tokens),0) AS completion_tokens,
                                COALESCE(SUM(cached_tokens),0) AS cached_tokens,
                                COUNT(*) AS calls
                              FROM model_usage WHERE provider=? AND usage_day=?''',
                           (provider,usage_day)).fetchone()
            result=dict(row)
            result['total_tokens']=result['prompt_tokens']+result['completion_tokens']
            result['usage_day']=usage_day
            return result

    def save_image_metadata(self,image):
        with self.connect() as db:
            db.execute('''INSERT INTO uploaded_images
                          (id,session_id,message_id,file_path,mime_type,byte_size,
                           sha256,width,height,description,created_at)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                       (image['id'],image['session_id'],image.get('message_id'),
                        image['file_path'],image['mime_type'],int(image['byte_size']),
                        image['sha256'],int(image['width']),int(image['height']),
                        image.get('description',''),now()))
        return self.get_image(image['id'],session_id=image['session_id'])

    def get_image(self,image_id,session_id=None):
        query='SELECT * FROM uploaded_images WHERE id=?'
        params=[image_id]
        if session_id is not None:
            query+=' AND session_id=?';params.append(session_id)
        with self.connect() as db:
            row=db.execute(query,tuple(params)).fetchone()
            if row is None:
                raise KeyError(image_id)
            return dict(row)

    def list_images_for_session(self,session_id):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                'SELECT * FROM uploaded_images WHERE session_id=? ORDER BY created_at',
                (session_id,))]

    def save_character_asset(self,asset):
        with self.connect() as db:
            db.execute('''INSERT INTO character_assets
                          (id,character_id,asset_type,file_path,mime_type,sha256,
                           prompt,seed,metadata_json,created_at)
                          VALUES(?,?,?,?,?,?,?,?,?,?)''',
                       (asset['id'],asset['character_id'],asset['asset_type'],
                        asset['file_path'],asset.get('mime_type','image/webp'),
                        asset['sha256'],asset.get('prompt',''),asset.get('seed'),
                        json.dumps(asset.get('metadata',{}),ensure_ascii=False),
                        now()))
        return self.get_character_asset(asset['id'])

    def get_character_asset(self,asset_id):
        with self.connect() as db:
            row=db.execute('SELECT * FROM character_assets WHERE id=?',(asset_id,)).fetchone()
            if row is None:
                raise KeyError(asset_id)
            result=dict(row)
            result['metadata']=json.loads(result.pop('metadata_json') or '{}')
            return result

    def latest_character_asset(self,character_id,asset_type=None):
        query='SELECT * FROM character_assets WHERE character_id=?'
        params=[character_id]
        if asset_type:
            query+=' AND asset_type=?';params.append(asset_type)
        query+=' ORDER BY created_at DESC LIMIT 1'
        with self.connect() as db:
            row=db.execute(query,tuple(params)).fetchone()
            if row is None:
                return None
            result=dict(row)
            result['metadata']=json.loads(result.pop('metadata_json') or '{}')
            return result

    def list_character_assets(self,character_id):
        with self.connect() as db:
            rows=[]
            for row in db.execute(
                    'SELECT * FROM character_assets WHERE character_id=? ORDER BY created_at',
                    (character_id,)):
                item=dict(row)
                item['metadata']=json.loads(item.pop('metadata_json') or '{}')
                rows.append(item)
            return rows
