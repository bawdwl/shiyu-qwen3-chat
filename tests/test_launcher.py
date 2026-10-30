import unittest
from unittest.mock import patch

import httpx
import launch_app


class LauncherTests(unittest.TestCase):
    def test_deepseek_service_can_be_reused_before_api_key_is_configured(self):
        launch_app.check_reusable(
            dict(
                status="model_unavailable",
                default_provider="deepseek",
                providers={"deepseek": {"configured": False}},
            ),
            "deepseek",
        )

    def test_matching_service_is_reused_without_starting_another_model(self):
        health=dict(status='ready',model_id='Qwen/Qwen3-1.7B',
                    same_chat_memory_enabled=False,
                    structured_memory_enabled=True,
                    structured_memory_mode='automatic',
                    memory_daily_token_limit=30000,
                    memory_retrieval={
                        'fts5': True,
                        'embedding_model': 'BAAI/bge-small-zh-v1.5',
                    })
        health['default_provider']='local'
        with patch('sys.argv',['launch_app.py','--no-open','--provider','local','--model-profile','qwen3-1.7b']), \
             patch('launch_app.running_health',return_value=health), \
             patch('launch_app.subprocess.Popen') as process:
            launch_app.main()
            process.assert_not_called()

    def test_explicit_model_mismatch_is_not_silently_ignored(self):
        health=dict(status='ready',model_id='Qwen/Qwen3-1.7B',same_chat_memory_enabled=False)
        health['default_provider']='local'
        with patch('sys.argv',['launch_app.py','--no-open','--provider','local','--model-profile','qwen3-0.6b']), \
             patch('launch_app.running_health',return_value=health), \
             patch('launch_app.subprocess.Popen') as process:
            with self.assertRaisesRegex(RuntimeError,'不一致'): launch_app.main()
            process.assert_not_called()

    def test_unavailable_existing_service_is_not_restarted(self):
        with self.assertRaisesRegex(RuntimeError,'未就绪'):
            launch_app.check_reusable(dict(status='model_unavailable',default_provider='local'),'local')

    def test_memory_mode_must_match_requested_experiment_condition(self):
        health=dict(status='ready',model_id='Qwen/Qwen3-1.7B',same_chat_memory_enabled=True)
        with self.assertRaisesRegex(RuntimeError,'记忆为开启'):
            health['default_provider']='local'
            launch_app.check_reusable(health,'local','qwen3-1.7b',same_chat_memory=False)

    def test_speech_provider_must_match_when_explicitly_requested(self):
        health=dict(status='ready',model_id='Qwen/Qwen3-1.7B',same_chat_memory_enabled=False,
                    speech_provider='本地中文语音（MeloTTS，CPU；非GPT-SoVITS）')
        with self.assertRaisesRegex(RuntimeError,'语音提供方'):
            health['default_provider']='local'
            launch_app.check_reusable(health,'local','qwen3-1.7b',same_chat_memory=False,speech_provider='profiles')

    def test_custom_memory_budget_must_round_trip_through_launcher(self):
        health=dict(
            status='ready',model_id='Qwen/Qwen3-1.7B',
            same_chat_memory_enabled=False,
            structured_memory_enabled=True,
            structured_memory_mode='automatic',
            memory_daily_token_limit=12500,
            memory_retrieval={
                'fts5': True,
                'embedding_model': 'BAAI/bge-small-zh-v1.5',
            },
            default_provider='local',
        )
        launch_app.check_reusable(
            health,'local','qwen3-1.7b',False,None,True,12500,
        )
        with self.assertRaisesRegex(RuntimeError,'后台记忆预算'):
            launch_app.check_reusable(
                health,'local','qwen3-1.7b',False,None,True,20000,
            )

    def test_old_service_without_memory_status_is_not_mistaken_for_current_version(self):
        health=dict(status='ready',model_id='Qwen/Qwen3-1.7B')
        with self.assertRaisesRegex(RuntimeError,'不支持本轮记忆对照'):
            health['default_provider']='local'
            launch_app.check_reusable(health,'local','qwen3-1.7b')

    def test_old_service_without_hybrid_retrieval_is_not_reused(self):
        health=dict(
            status='ready',
            model_id='Qwen/Qwen3-1.7B',
            same_chat_memory_enabled=False,
            structured_memory_enabled=True,
            structured_memory_mode='automatic',
            memory_daily_token_limit=30000,
            default_provider='local',
        )
        with self.assertRaisesRegex(RuntimeError,'FTS5/BGE'):
            launch_app.check_reusable(
                health,'local','qwen3-1.7b',False,None,True,30000,
            )

    def test_other_application_on_port_is_rejected(self):
        response=httpx.Response(200,json={'status':'ok'})
        with patch('launch_app.httpx.get',return_value=response):
            with self.assertRaisesRegex(RuntimeError,'端口'): launch_app.running_health()

    def test_unoccupied_port_returns_none_but_timeout_is_not_unoccupied(self):
        with patch('launch_app.httpx.get',side_effect=httpx.ConnectError('refused')):
            self.assertIsNone(launch_app.running_health())
        with patch('launch_app.httpx.get',side_effect=httpx.ConnectTimeout('closed port')):
            self.assertIsNone(launch_app.running_health())
        with patch('launch_app.httpx.get',side_effect=httpx.ReadTimeout('hung')):
            with self.assertRaisesRegex(RuntimeError,'不会重复'): launch_app.running_health()


if __name__=='__main__': unittest.main()
