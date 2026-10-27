"""Freeze the unseen-role and capability evaluation beside a complete data build."""
import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from prepare_shared_role_data import write_json
from shared_role_data import (audit_final_evaluation, audit_shared_dataset,
                              load_registry)
from training_data import file_hash


def freeze_evaluation(base_dir, source_eval_dir, output_dir):
    base = Path(base_dir).resolve()
    source_eval = Path(source_eval_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('拒绝覆盖已有冻结包：' + str(output))
    if not base.is_dir() or not source_eval.is_dir() or not output.parent.is_dir():
        raise FileNotFoundError('完整候选包、最终评测源目录及输出父目录必须存在')

    base_audit = audit_shared_dataset(base, require_training_ready=False)
    if (base_audit['source_dialogues'] != {'train': 400, 'validation': 50}
            or base_audit['counts'] != {'train': 932, 'validation': 126}):
        raise ValueError('输入包不是完整的400/50训练与验证候选')
    main_registry = load_registry(base / 'role-cards.json')
    training_dialogues = json.loads((base / 'dialogues.json').read_text(encoding='utf-8'))
    final_audit = audit_final_evaluation(
        source_eval, main_registry, training_dialogues,
        protocol_path=source_eval / 'protocol.json')

    with tempfile.TemporaryDirectory(prefix='.shared-role-freeze-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'payload'
        shutil.copytree(base, stage)
        shutil.copytree(source_eval, stage / 'final-evaluation')
        manifest_path = stage / 'freeze-manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest.update(
            stage='full_data_and_holdout_frozen_pending_gpu_preflight',
            created_at=datetime.now().astimezone().isoformat(),
            training_ready=False,
            final_evaluation_frozen=True,
            reserved_final_role_ids=final_audit['final_role_ids'],
            final_evaluation=final_audit,
            evaluation_files={
                'role_cards': dict(path='final-evaluation/role-cards.json',
                                   sha256=file_hash(stage / 'final-evaluation/role-cards.json')),
                'final_chat': dict(path='final-evaluation/chat-cases.json',
                                   sha256=file_hash(stage / 'final-evaluation/chat-cases.json')),
                'capability_regression': dict(path='final-evaluation/capability-regression.json',
                                              sha256=file_hash(stage / 'final-evaluation/capability-regression.json')),
                'protocol': dict(path='final-evaluation/protocol.json',
                                 sha256=file_hash(stage / 'final-evaluation/protocol.json')),
            },
            remaining=[
                'Run the locked 1024-token full-data preflight and longest-sample GPU memory check',
                'Train and reload the first shared-role LoRA candidate',
                'Screen development outputs before the one-time frozen comparison'
            ],
            note='Train/validation candidates and final evaluation are frozen. Training remains disabled until full-data and GPU preflight is recorded.')
        manifest.setdefault('provenance', {}).setdefault('source_files', {})
        for name in ('role-cards.json', 'chat-cases.json',
                     'capability-regression.json', 'protocol.json'):
            manifest['provenance']['source_files'][
                str(Path('data/shared-role-chat-v1/final-evaluation') / name)] = file_hash(source_eval / name)
        manifest['code_sha256'] = {
            name: file_hash(name) for name in (
                'shared_role_data.py', 'training_data.py',
                'prepare_shared_role_data.py', 'freeze_shared_role_evaluation.py')
        }
        write_json(manifest_path, manifest)

        preflight_path = stage / 'preflight.json'
        preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
        preflight.update(
            status='passed_data_and_evaluation_freeze_pending_gpu_preflight',
            training_ready=False,
            final_evaluation=final_audit)
        write_json(preflight_path, preflight)
        frozen_audit = audit_shared_dataset(stage, require_training_ready=False)
        if frozen_audit.get('evaluation') != final_audit:
            raise ValueError('输出包中的最终评测与预检报告不一致')
        stage.rename(output)
    return dict(output_dir=str(output), training_ready=False,
                final_evaluation_frozen=True, evaluation=final_audit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='outputs/shared-role-build-v11')
    parser.add_argument('--source-eval-dir', default='data/shared-role-chat-v1/final-evaluation')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_evaluation(args.base_dir, args.source_eval_dir, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
