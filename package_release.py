"""按明确清单打包展示材料，不包含权重、原始计划书或本地环境。"""
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath


# Only project-authored examples may be redistributed. .gitignore does not
# apply to zipfile, and newly downloaded corpora must not enter by recursion.
PUBLIC_DATA_FILES = frozenset({
    'data/README.md', 'data/train.jsonl', 'data/validation.jsonl', 'data/eval.jsonl',
    'data/chat_scenarios_v1.json', 'data/im_chat_development_v2.json',
    'data/v2/train.jsonl', 'data/v2/validation.jsonl', 'data/v2/test.jsonl',
    'data/v2/freeze-manifest.json', 'data/v2/training-manifest.json',
    'data/quality-holdout-v1/test.jsonl', 'data/quality-holdout-v1/freeze-manifest.json',
})

# Documentation allowed in a public candidate. Do not recurse through docs/:
# it contains private review sheets and historical material that is not meant
# to be shipped with the runnable project.
PUBLIC_DOC_FILES = frozenset({
    'docs/architecture.md', 'docs/api-guide.md', 'docs/demo-guide.md',
    'docs/求职项目目标与岗位差距.md', 'docs/IM岗位项目主计划与验收标准.md',
    'docs/角色聊天实验A-D执行计划.md', 'docs/独立环境复现记录.md',
    'docs/LCCC-base首轮LoRA实验记录.md', 'docs/训练与评估报告.md',
    'docs/MNN本机实跑记录.md', 'docs/MNN手机端部署验证记录.md',
    'docs/手机语音体验最小记录表.md', 'docs/语音输入接入记录.md',
    'docs/语音朗读接入记录.md', 'docs/training_report.md',
    'docs/qwen3-1.7b-source.json', 'docs/mnn-model-source.json',
    'docs/asr-model-source.json', 'docs/tts-model-source.json',
    'docs/mnn-cpu-smoke.json', 'docs/api-smoke-result.json',
    'docs/streaming-smoke-result.json',
})


def allowed_data_path(relative):
    return PurePosixPath(relative).as_posix() in PUBLIC_DATA_FILES


def main():
    root=Path(__file__).resolve().parent
    names=['README.md','LICENSE','LICENSE-NOTICE.md','.gitignore','requirements-inference.txt','requirements-lock-windows-py311.txt',
           'requirements-train.txt','app.py','conversation.py','inference.py','evaluate.py',
           'compare_results.py','train_lora.py','training_data.py','prepare_data.py',
           'verify_adapter.py','package_release.py','audit_release.py','test_model_switch.py',
           'prepare_evaluation_v2.py','audit_evaluation_v2.py',
           'prepare_training_v2.py','evaluate_v2.py','summarize_review_v2.py',
           'compare_evaluation_v2.py','backend.py','storage.py','smoke_api.py','requirements-api.txt',
           'api_client.py','stream_job.py','smoke_streaming.py','local_app.py',
           'benchmark_performance.py','prepare_quality_holdout.py',
           'prepare_blind_review.py','summarize_blind_review.py',
           'fetch_oasst2_source.py','prepare_oasst2_candidates.py',
           'generation_options.py','context_window.py','smoke_context_window.py','smoke_chat_controls.py','speech.py',
           'onnx_speech_worker.py','fetch_tts_model.py','smoke_speech.py','requirements-speech.txt',
           'recognition.py','onnx_asr_worker.py','fetch_asr_model.py','smoke_recognition.py','requirements-media.txt',
           'characters.py','launch_app.py','runtime_models.py','fetch_chat_model.py','smoke_model_profile.py',
           'same_chat_memory.py','role_knowledge.py','audit_training_pipeline.py',
           'requirements-mnn.txt','fetch_mnn_model.py','smoke_mnn.py','verify_clean_install.py','fetch_torch_wheel.py',
           'compare_chat_prompts.py','inspect_lccc.py','prepare_lccc.py','prepare_naturalconv.py',
           'evaluate_lccc_pilot.py','preflight_role_training.py','evaluate_role_ablation.py',
           'evaluate_lora_learning_rate.py',
           'prepare_lccc_chat_review.py','evaluate_frozen_chat.py','evaluate_character_context.py',
           'prepare_naturalconv_review.py','evaluate_naturalconv_review.py']
    files=[root/n for n in names]
    files.extend(p for p in (root/'web').rglob('*') if p.is_file() and p.suffix in {'.html','.js','.css','.svg','.webmanifest','.png','.md'})
    files.extend(root / name for name in PUBLIC_DATA_FILES if (root / name).is_file())
    files.extend(root / name for name in PUBLIC_DOC_FILES if (root / name).is_file())
    files.extend(p for p in (root/'tests').rglob('*') if p.is_file() and p.suffix in {'.py','.cjs'} and '__pycache__' not in p.parts)
    dest=root/'dist'
    dest.mkdir(exist_ok=True)
    target=dest/('qwen3-im-chat-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.zip')
    manifest=[]
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            relative=path.relative_to(root).as_posix()
            assert path.suffix not in {'.safetensors','.bin','.docx'}
            content=path.read_bytes()
            # 发布副本隐藏个人绝对路径；本地原始记录不改动。
            if path.suffix in {'.md','.json','.jsonl','.txt'}:
                text=content.decode('utf-8-sig')
                text=text.replace(str(Path.home()).replace('\\','\\\\'),'<USERPROFILE>')
                text=text.replace(str(Path.home()),'<USERPROFILE>')
                text=text.replace(Path.home().as_posix(),'<USERPROFILE>')
                text=text.replace('D:\\Others\\Study\\Anaconda3','<CONDA_ROOT>')
                text=text.replace('D:\\Others\\Study\\Anaconda3'.replace('\\','\\\\'),'<CONDA_ROOT>')
                content=text.encode('utf-8')
            archive.writestr('qwen3-im-chat/'+relative,content)
            manifest.append(dict(path=relative,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        archive.writestr('qwen3-im-chat/RELEASE-MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert '/outputs/' not in name and '/__pycache__/' not in name
            if name.endswith(('.json','.jsonl')):
                content=archive.read(name).decode('utf-8')
                if name.endswith('.jsonl'):
                    for line in content.splitlines():
                        json.loads(line)
                else:
                    json.loads(content)
    print(json.dumps(dict(archive=str(target),file_count=len(manifest)+1,bytes=target.stat().st_size,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         note='MIT项目材料包；第三方许可单独适用，未在线发布。'),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
