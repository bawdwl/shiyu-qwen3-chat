"""Evidence validation for automatically adjudicated character memory.

Exact evidence verifies attribution, not semantic entailment of the summary.
The memory store applies a conservative operation policy after this check.
"""
from datetime import date
import re

KINDS = {'user_report', 'character_detail', 'shared_event', 'belief', 'plan'}
SUBJECTS = {'user', 'character', 'shared'}
OPERATIONS = {'add', 'merge', 'supersede', 'contested', 'ignore'}
_FACT_KEY = re.compile(r'[a-z][a-z0-9_.-]{2,119}\Z')


def _infer_subject(kind, evidence):
    roles = {item['role'] for item in evidence}
    if kind == 'user_report' or roles == {'user'}:
        return 'user'
    if kind == 'character_detail' or roles == {'assistant'}:
        return 'character'
    return 'shared'


def validate_candidate(candidate, messages):
    if not isinstance(candidate, dict):
        raise ValueError('candidate must be an object')
    sources = {m['id']: m for m in messages}
    kind = candidate.get('kind')
    if kind not in KINDS:
        raise ValueError('unsupported memory kind')
    evidence = candidate.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('memory needs source evidence')
    checked = []
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError('invalid evidence')
        mid, quote = item.get('message_id'), item.get('quote')
        source = sources.get(mid) if type(mid) is int else None
        if not source or not isinstance(quote, str) or not quote.strip() or quote not in source['content']:
            raise ValueError('evidence does not match source')
        if kind == 'user_report' and source['role'] != 'user':
            raise ValueError('assistant text cannot establish user report')
        if kind == 'character_detail' and source['role'] != 'assistant':
            raise ValueError('user text cannot establish character detail')
        checked.append({'message_id': mid, 'role': source['role'], 'quote': quote})
    content = candidate.get('content')
    if not isinstance(content, str) or not 1 <= len(content.strip()) <= 600:
        raise ValueError('invalid memory content')
    subject = candidate.get('subject') or _infer_subject(kind, checked)
    if subject not in SUBJECTS:
        raise ValueError('unsupported memory subject')
    if kind == 'user_report' and subject != 'user':
        raise ValueError('user reports must use the user subject')
    if kind == 'character_detail' and subject != 'character':
        raise ValueError('character details must use the character subject')
    fact_key = candidate.get('fact_key')
    if fact_key is not None:
        if not isinstance(fact_key, str) or not _FACT_KEY.fullmatch(fact_key.strip().lower()):
            raise ValueError('invalid fact key')
        fact_key = fact_key.strip().lower()
    value = candidate.get('value')
    if value is not None:
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200:
            raise ValueError('invalid fact value')
        value = value.strip()
        if not any(value in item['quote'] for item in checked):
            raise ValueError('fact value requires an exact source substring; paraphrase separately')
    event_date = candidate.get('event_date')
    if event_date is not None:
        if not isinstance(event_date, str):
            raise ValueError('invalid event date')
        date.fromisoformat(event_date)
        if not any(event_date in item['quote'] for item in checked):
            raise ValueError('calendar date requires explicit source; relative dates stay unresolved')
    operation = candidate.get('operation', 'add')
    if operation not in OPERATIONS:
        raise ValueError('unsupported memory operation')
    target_item_id = candidate.get('target_item_id')
    if target_item_id is not None and (
        type(target_item_id) is not int or target_item_id <= 0
    ):
        raise ValueError('invalid memory target')
    if operation in {'merge', 'supersede', 'contested'} and target_item_id is None:
        raise ValueError('memory operation requires a target')
    return {'kind': kind, 'subject': subject, 'fact_key': fact_key, 'value': value,
            'content': content.strip(), 'evidence': checked,
            'event_date': event_date, 'status': 'candidate',
            'semantic_verified': False, 'operation': operation,
            'target_item_id': target_item_id}


def propose_revision(current, candidate, *, same_subject_and_fact=False):
    """Prepare a reviewable change; caller must establish the shared fact key.

    Quote validation does not establish that a correction is semantically true,
    so no current record is mutated and no proposal is automatically approved.
    """
    same_subject_and_fact = same_subject_and_fact or (
        current.get('kind') == candidate.get('kind')
        and current.get('subject') == candidate.get('subject')
        and current.get('fact_key') is not None
        and current.get('fact_key') == candidate.get('fact_key')
    )
    if not same_subject_and_fact:
        return {'action': 'review_separately', 'current': current, 'candidate': candidate}
    if not current or not candidate or current.get('kind') != candidate.get('kind'):
        raise ValueError('revision needs same-kind current and candidate records')
    if not current.get('evidence') or not candidate.get('evidence'):
        raise ValueError('revision needs evidence for both versions')
    if current.get('value') is not None and candidate.get('value') == current.get('value'):
        return {'action': 'duplicate_value', 'current': current, 'candidate': candidate}
    old_ids = [item['message_id'] for item in current['evidence']]
    new_ids = [item['message_id'] for item in candidate['evidence']]
    if min(new_ids) <= max(old_ids):
        return {'action': 'review_order', 'current': current, 'candidate': candidate}
    return {'action': 'review_supersession', 'current': current, 'candidate': candidate,
            'reason': 'newer source, same fact proposed; semantic review required'}
