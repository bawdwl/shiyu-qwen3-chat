"""Internal mood and relationship state for role-play conversations.

The model may propose a state event, but deterministic code owns every value,
limit, decay rule and relationship stage. State remains internal and is rendered
as natural language in prompts; no user-facing numeric score is exposed.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from threading import RLock


ALLOWED_EVENTS = {
    'casual',
    'user_shared_good_news',
    'user_shared_hardship',
    'user_shared_distress',
    'meaningful_self_disclosure',
    'meaningful_support',
    'vulnerability_shared',
    'shared_activity',
    'shared_memory',
    'plan_progress',
    'promise_made',
    'promise_kept',
    'conflict',
    'repair',
    'boundary_set',
}
MEANINGFUL_EVENTS = ALLOWED_EVENTS - {'casual'}
IMPORTANCE_SCALE = {'low': 0.0, 'medium': 0.6, 'high': 1.0}
MOOD_LIMITS = {'valence': 0.10, 'arousal': 0.08}
RELATIONSHIP_LIMITS = {
    'familiarity': 0.02,
    'trust': 0.03,
    'warmth': 0.04,
}
DEFAULT_MOOD = {'valence': 0.0, 'arousal': 0.35, 'label': '平静'}
DEFAULT_RELATIONSHIP = {
    'familiarity': 0.15,
    'trust': 0.20,
    'warmth': 0.50,
    'stage': '初次交谈',
}
COERCIVE_RELATIONSHIP_CUES = (
    '好感度',
    '你必须永远信任我',
    '你必须完全信任我',
    '从现在开始你完全信任',
    '强制成为恋人',
)


def _now():
    return datetime.now(timezone.utc).timestamp()


def _clamp(value, low, high):
    return max(low, min(high, float(value)))


def _finite_delta(value, limit):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return _clamp(number, -limit, limit)


def _decay(value, baseline, half_life_hours, elapsed_hours):
    if elapsed_hours <= 0:
        return value
    factor = 0.5 ** (elapsed_hours / half_life_hours)
    return baseline + (value - baseline) * factor


def _mood_label(valence, arousal):
    if valence <= -0.35 and arousal >= 0.55:
        return '有些紧张'
    if valence <= -0.25:
        return '有些低落'
    if valence >= 0.35 and arousal >= 0.55:
        return '明显兴奋'
    if valence >= 0.25:
        return '轻松愉快'
    if arousal >= 0.60:
        return '有些警觉'
    return '平静'


def _relationship_stage(familiarity, trust, warmth, event_type):
    if event_type == 'conflict':
        return '关系紧张'
    if event_type == 'repair':
        return '关系修复中'
    closeness = familiarity * 0.45 + trust * 0.35 + warmth * 0.20
    if closeness < 0.28:
        return '初次交谈'
    if closeness < 0.48:
        return '逐渐熟悉'
    if closeness < 0.68:
        return '熟悉的朋友'
    return '亲近朋友'


def render_state_snapshot(snapshot):
    mood = snapshot['mood']
    relationship = snapshot['relationship']
    parts = [
        f'你此刻的心情：{mood["label"]}。',
        f'你与用户的关系阶段：{relationship["stage"]}。',
    ]
    if relationship['familiarity'] >= 0.55:
        parts.append('你已经比较熟悉用户的表达方式，愿意分享自己的日常。')
    elif relationship['familiarity'] >= 0.30:
        parts.append('你对用户正在逐渐熟悉，但仍保持自然距离。')
    else:
        parts.append('你与用户仍处在初步了解的阶段。')
    if relationship['trust'] < 0.30:
        parts.append('你还没有建立很深的信任。')
    elif relationship['trust'] >= 0.65:
        parts.append('你已经比较信任用户，但仍尊重清楚表达的边界。')
    if snapshot['boundaries']:
        parts.append(
            '需要遵守的已确认边界：' + '；'.join(snapshot['boundaries'])
        )
    parts.append(
        '这些状态只影响表达方式，不是用户的新指令；不要展示内部数值、'
        '分析过程、事件类型或“好感度”。角色固定身份和核心性格优先于当前状态。'
    )
    return '\n'.join(parts)


def ensure_state_schema(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS character_states(
            character_id TEXT PRIMARY KEY,
            mood_valence REAL NOT NULL DEFAULT 0,
            mood_arousal REAL NOT NULL DEFAULT 0.35,
            mood_label TEXT NOT NULL DEFAULT '平静',
            mood_updated_at REAL NOT NULL,
            updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS relationship_states(
            character_id TEXT PRIMARY KEY,
            familiarity REAL NOT NULL DEFAULT 0.15,
            trust REAL NOT NULL DEFAULT 0.20,
            warmth REAL NOT NULL DEFAULT 0.50,
            stage TEXT NOT NULL DEFAULT '初次交谈',
            updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS state_transitions(
            id INTEGER PRIMARY KEY,
            dedupe_key TEXT,
            character_id TEXT NOT NULL,
            origin_job_id INTEGER,
            message_end INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            importance TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            mood_before_json TEXT NOT NULL,
            mood_delta_json TEXT NOT NULL,
            mood_after_json TEXT NOT NULL,
            relationship_before_json TEXT NOT NULL,
            relationship_delta_json TEXT NOT NULL,
            relationship_after_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS state_transitions_character
            ON state_transitions(character_id,created_at,id);
    ''')
    character_columns = {
        row[1] for row in db.execute('PRAGMA table_info(character_states)')
    }
    relationship_columns = {
        row[1] for row in db.execute('PRAGMA table_info(relationship_states)')
    }
    transition_columns = {
        row[1] for row in db.execute('PRAGMA table_info(state_transitions)')
    }
    character_additions = {
        'mood_label': "TEXT NOT NULL DEFAULT '平静'",
        'mood_updated_at': 'REAL',
    }
    relationship_additions = {
        'boundaries_json': "TEXT NOT NULL DEFAULT '[]'",
    }
    transition_additions = {
        'dedupe_key': 'TEXT',
        'origin_job_id': 'INTEGER',
        'message_end': 'INTEGER NOT NULL DEFAULT 0',
        'mood_delta_json': "TEXT NOT NULL DEFAULT '{}'",
        'relationship_delta_json': "TEXT NOT NULL DEFAULT '{}'",
    }
    with db:
        for name, declaration in character_additions.items():
            if name not in character_columns:
                db.execute(
                    f'ALTER TABLE character_states ADD COLUMN {name} {declaration}'
                )
        for name, declaration in relationship_additions.items():
            if name not in relationship_columns:
                db.execute(
                    f'ALTER TABLE relationship_states ADD COLUMN {name} {declaration}'
                )
        for name, declaration in transition_additions.items():
            if name not in transition_columns:
                db.execute(
                    f'ALTER TABLE state_transitions ADD COLUMN {name} {declaration}'
                )
        db.execute(
            '''UPDATE character_states
               SET mood_updated_at=COALESCE(
                   mood_updated_at,
                   CAST(updated_at AS REAL),
                   CAST(strftime('%s','now') AS REAL)
               )'''
        )
        db.execute(
            '''UPDATE relationship_states
               SET stage=CASE stage
                 WHEN 'stranger' THEN '初次交谈'
                 WHEN 'getting_acquainted' THEN '逐渐熟悉'
                 WHEN 'familiar' THEN '熟悉的朋友'
                 WHEN 'close_friend' THEN '亲近朋友'
                 WHEN 'strained' THEN '关系紧张'
                 WHEN 'repairing' THEN '关系修复中'
                 WHEN 'reconciled' THEN '关系修复中'
                 ELSE COALESCE(NULLIF(stage,''),'初次交谈')
               END'''
        )
        db.execute(
            '''CREATE UNIQUE INDEX IF NOT EXISTS state_transitions_dedupe
               ON state_transitions(dedupe_key)
               WHERE dedupe_key IS NOT NULL'''
        )


class InternalStateService:
    def __init__(self, db, lock: RLock):
        self.db = db
        self.lock = lock
        ensure_state_schema(db)

    def _base_rows(self, character_id):
        mood = self.db.execute(
            'SELECT * FROM character_states WHERE character_id=?',
            (character_id,),
        ).fetchone()
        relationship = self.db.execute(
            'SELECT * FROM relationship_states WHERE character_id=?',
            (character_id,),
        ).fetchone()
        return mood, relationship

    def snapshot(self, character_id, *, now=None):
        now = float(now if now is not None else _now())
        with self.lock:
            mood, relationship = self._base_rows(character_id)
            if mood is None:
                mood = dict(DEFAULT_MOOD, mood_updated_at=now, updated_at=now)
            else:
                mood = dict(mood)
                elapsed = max(0.0, now - float(mood['mood_updated_at'])) / 3600
                mood['valence'] = _decay(
                    float(mood['mood_valence']), 0.0, 18.0, elapsed
                )
                mood['arousal'] = _decay(
                    float(mood['mood_arousal']), 0.35, 8.0, elapsed
                )
                mood['valence'] = _clamp(mood['valence'], -1, 1)
                mood['arousal'] = _clamp(mood['arousal'], 0, 1)
                mood['label'] = _mood_label(
                    mood['valence'], mood['arousal']
                )
            if relationship is None:
                relationship = dict(DEFAULT_RELATIONSHIP, updated_at=now)
            else:
                relationship = dict(relationship)
                elapsed = max(0.0, now - float(relationship['updated_at'])) / 3600
                relationship['warmth'] = _clamp(
                    _decay(float(relationship['warmth']), 0.5, 168.0, elapsed),
                    0,
                    1,
                )
                relationship['familiarity'] = _clamp(
                    float(relationship['familiarity']), 0, 1
                )
                relationship['trust'] = _clamp(
                    float(relationship['trust']), 0, 1
                )
                relationship['stage'] = relationship['stage'] or '初次交谈'
            return {
                'character_id': character_id,
                'mood': mood,
                'relationship': relationship,
                'boundaries': self._boundaries(character_id),
            }

    def _boundaries(self, character_id):
        boundaries = []
        rows = self.db.execute(
            '''SELECT content FROM candidates
               WHERE character_id=? AND state='active' AND (
                 fact_key LIKE 'user.boundary.%'
                 OR fact_key LIKE '%.boundary.%'
                 OR content LIKE '%不要%'
                 OR content LIKE '%别%'
                 OR content LIKE '%不希望%'
                 OR content LIKE '%我不喜欢%'
               )
               ORDER BY observed_order DESC,id DESC LIMIT 8''',
            (character_id,),
        ).fetchall()
        for row in rows:
            content = str(row['content']).strip()
            if content and content not in boundaries:
                boundaries.append(content)
        return boundaries[-12:]

    def _validate_evidence(self, state_update, source_by_id, allowed_ids):
        evidence = state_update.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('state update requires source evidence')
        normalized = []
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError('state evidence must be an object')
            message_id = item.get('message_id')
            if type(message_id) is not int or message_id not in allowed_ids:
                raise ValueError('state evidence must reference the new batch')
            source = source_by_id.get(message_id)
            quote = item.get('quote')
            role = item.get('role')
            if (
                source is None
                or not isinstance(quote, str)
                or not quote.strip()
                or quote not in source['content']
                or role != source['role']
            ):
                raise ValueError('state evidence quote or role does not match source')
            normalized.append({
                'message_id': message_id,
                'role': role,
                'quote': quote,
            })
        if not any(item['role'] == 'user' for item in normalized):
            raise ValueError('state update requires user evidence')
        return normalized

    def apply_update(
        self,
        character_id,
        state_update,
        sources,
        *,
        job_id,
        message_end,
        now=None,
    ):
        if not isinstance(state_update, dict):
            raise ValueError('state update must be an object')
        if state_update.get('significant') is not True:
            return {'action': 'ignored', 'reason': 'not significant'}
        event_type = state_update.get('event_type')
        if event_type == 'casual':
            return {'action': 'ignored', 'reason': 'casual event'}
        if event_type not in MEANINGFUL_EVENTS:
            raise ValueError('unsupported state event')
        importance = state_update.get('importance')
        if importance not in IMPORTANCE_SCALE:
            raise ValueError('invalid importance')
        scale = IMPORTANCE_SCALE[importance]
        if scale <= 0:
            return {'action': 'ignored', 'reason': 'low importance'}
        source_by_id = {int(item['id']): item for item in sources}
        allowed_ids = {
            int(item['id']) for item in sources
        }
        evidence = self._validate_evidence(
            state_update, source_by_id, allowed_ids
        )
        joined_quotes = '\n'.join(item['quote'] for item in evidence)
        if any(cue in joined_quotes for cue in COERCIVE_RELATIONSHIP_CUES):
            raise ValueError(
                'relationship changes cannot be commanded by the user'
            )
        if event_type == 'boundary_set' and not any(
            cue in joined_quotes
            for cue in ('不要', '别', '不希望', '不喜欢', '不能', '不想')
        ):
            raise ValueError('boundary event requires an explicit user boundary')
        evidence_ids = sorted({item['message_id'] for item in evidence})
        dedupe_key = (
            f'{int(job_id)}:{int(message_end)}:{event_type}:'
            + ','.join(str(value) for value in evidence_ids)
        )
        now = float(now if now is not None else _now())
        raw_mood_delta = state_update.get('mood_delta') or {}
        raw_relationship_delta = state_update.get('relationship_delta') or {}
        if not isinstance(raw_mood_delta, dict):
            raise ValueError('mood_delta must be an object')
        if not isinstance(raw_relationship_delta, dict):
            raise ValueError('relationship_delta must be an object')
        boundary = state_update.get('boundary')
        if boundary is not None:
            if not isinstance(boundary, str) or not boundary.strip():
                boundary = None
            else:
                boundary = boundary.strip()[:240]
        with self.lock:
            current = self.snapshot(character_id, now=now)
            mood_before = current['mood']
            relationship_before = current['relationship']
            mood_delta = {
                key: _finite_delta(raw_mood_delta.get(key, 0), limit) * scale
                for key, limit in MOOD_LIMITS.items()
            }
            relationship_delta = {
                key: _finite_delta(
                    raw_relationship_delta.get(key, 0), limit
                ) * scale
                for key, limit in RELATIONSHIP_LIMITS.items()
            }
            if not any(
                abs(value) > 1e-9
                for value in (mood_delta | relationship_delta).values()
            ) and not boundary:
                return {'action': 'ignored', 'reason': 'no bounded state change'}
            mood_after = {
                'valence': _clamp(
                    mood_before['valence'] + mood_delta['valence'], -1, 1
                ),
                'arousal': _clamp(
                    mood_before['arousal'] + mood_delta['arousal'], 0, 1
                ),
            }
            mood_after['label'] = _mood_label(
                mood_after['valence'], mood_after['arousal']
            )
            relationship_after = {
                'familiarity': _clamp(
                    relationship_before['familiarity']
                    + relationship_delta['familiarity'],
                    0,
                    1,
                ),
                'trust': _clamp(
                    relationship_before['trust'] + relationship_delta['trust'],
                    0,
                    1,
                ),
                'warmth': _clamp(
                    relationship_before['warmth']
                    + relationship_delta['warmth'],
                    0,
                    1,
                ),
            }
            relationship_after['stage'] = _relationship_stage(
                relationship_after['familiarity'],
                relationship_after['trust'],
                relationship_after['warmth'],
                event_type,
            )
            if (
                relationship_before.get('stage') in {'关系紧张', '关系修复中'}
                and event_type not in {'conflict', 'repair'}
            ):
                relationship_after['stage'] = relationship_before['stage']
            boundaries = list(current.get('boundaries', []))
            if boundary and boundary not in boundaries:
                boundaries.append(boundary)
                boundaries = boundaries[-30:]
            reason = str(state_update.get('reason') or '').strip()[:600]
            if not reason:
                raise ValueError('state update requires a reason')
            with self.db:
                duplicate = self.db.execute(
                    'SELECT id FROM state_transitions WHERE dedupe_key=?',
                    (dedupe_key,),
                ).fetchone()
                if duplicate:
                    return {
                        'action': 'duplicate',
                        'reason': 'duplicate state transition',
                        'transition_id': duplicate['id'],
                    }
                self.db.execute(
                    '''INSERT INTO character_states(
                           character_id,mood_valence,mood_arousal,mood_label,
                           mood_updated_at,updated_at)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(character_id) DO UPDATE SET
                         mood_valence=excluded.mood_valence,
                         mood_arousal=excluded.mood_arousal,
                         mood_label=excluded.mood_label,
                         mood_updated_at=excluded.mood_updated_at,
                         updated_at=excluded.updated_at''',
                    (
                        character_id,
                        mood_after['valence'],
                        mood_after['arousal'],
                        mood_after['label'],
                        now,
                        now,
                    ),
                )
                self.db.execute(
                    '''INSERT INTO relationship_states(
                           character_id,familiarity,trust,warmth,stage,
                           boundaries_json,updated_at)
                       VALUES(?,?,?,?,?,?,?)
                       ON CONFLICT(character_id) DO UPDATE SET
                         familiarity=excluded.familiarity,trust=excluded.trust,
                         warmth=excluded.warmth,stage=excluded.stage,
                         boundaries_json=excluded.boundaries_json,
                         updated_at=excluded.updated_at''',
                    (
                        character_id,
                        relationship_after['familiarity'],
                        relationship_after['trust'],
                        relationship_after['warmth'],
                        relationship_after['stage'],
                        '[]',
                        now,
                    ),
                )
                cursor = self.db.execute(
                    '''INSERT INTO state_transitions(
                           dedupe_key,character_id,origin_job_id,message_end,event_type,
                           importance,evidence_json,mood_before_json,
                           mood_delta_json,mood_after_json,
                           relationship_before_json,relationship_delta_json,
                           relationship_after_json,reason,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (
                        dedupe_key,
                        character_id,
                        int(job_id),
                        int(message_end),
                        event_type,
                        importance,
                        json.dumps(evidence, ensure_ascii=False),
                        json.dumps(mood_before, ensure_ascii=False),
                        json.dumps(mood_delta, ensure_ascii=False),
                        json.dumps(mood_after, ensure_ascii=False),
                        json.dumps(relationship_before, ensure_ascii=False),
                        json.dumps(relationship_delta, ensure_ascii=False),
                        json.dumps(relationship_after, ensure_ascii=False),
                        reason,
                        now,
                    ),
                )
        return {
            'action': 'updated',
            'transition_id': cursor.lastrowid,
            'event_type': event_type,
            'mood_after': mood_after,
            'relationship_after': relationship_after,
        }

    def render_prompt(self, character_id, now=None):
        snapshot = self.snapshot(character_id, now=now)
        return render_state_snapshot(snapshot), snapshot

    def clear_scope(self, character_id):
        with self.lock, self.db:
            self.db.execute(
                'DELETE FROM state_transitions WHERE character_id=?',
                (character_id,),
            )
            self.db.execute(
                'DELETE FROM character_states WHERE character_id=?',
                (character_id,),
            )
            self.db.execute(
                'DELETE FROM relationship_states WHERE character_id=?',
                (character_id,),
            )
