"""Local Stable Diffusion 1.5 image generation for character avatars."""
from __future__ import annotations

import hashlib
import io
import logging
import re
import sys
from pathlib import Path
from threading import Lock


logger = logging.getLogger(__name__)

_runtime = Path(__file__).resolve().parent.parent / "outputs/image-runtime"
if _runtime.is_dir() and str(_runtime) not in sys.path:
    # Append after the main environment so the CUDA torch installation remains
    # authoritative. The runtime directory exists for diffusers/accelerate only.
    sys.path.append(str(_runtime))


class ImageGenerationUnavailable(RuntimeError):
    pass


_CJK_TEXT = re.compile(r'[\u3400-\u9fff]')
_PROMPT_NOISE = re.compile(
    r'(?i)\b(masterpiece|best quality|high quality|ultra detailed|8k|4k)\b[, ]*'
)
_STYLE_NOISE = re.compile(
    r'(?i)\b(photograph|photorealistic|raw photo|photo|hyperrealistic|'
    r'realistic portrait|3d render|cgi|game render|digital painting|'
    r'anime|illustration|cinematic)\b[, ]*'
)
_DEFAULT_AVATAR_PROMPT = (
    "polished semi-realistic anime portrait, one original character, "
    "head and shoulders, face fully visible, centered composition, "
    "expressive eyes, natural detailed face, simple fitting clothes, "
    "soft warm background, clean cinematic lighting, high detail"
)
_DEFAULT_NEGATIVE_PROMPT = (
    "photograph, photorealistic, raw photo, 3d render, CGI, game render, "
    "plastic skin, flat colors, noisy, bad anatomy, bad hands, "
    "extra fingers, missing fingers, deformed face, distorted eyes, "
    "asymmetrical eyes, text, watermark, logo, multiple people, cropped head"
)
_PORTRAIT_PREFIX = (
    "polished semi-realistic anime portrait, "
    "head and shoulders, face fully visible, centered composition, "
)


def _compact_sd_prompt(value: str, max_words: int = 45) -> str:
    """Keep the prompt below SD1.5's effective 77-token CLIP window.

    CLIP tokenization is not one token per word, so 45 ordinary English words
    leaves room for punctuation and character names. Prompts containing CJK
    text are rejected: the base SD1.5 text encoder cannot understand them.
    """
    text = " ".join(str(value or "").replace("\n", " ").split()).strip(" `\"'")
    if not text or _CJK_TEXT.search(text):
        return ""
    text = _PROMPT_NOISE.sub("", text).strip(" ,")
    text = _STYLE_NOISE.sub("", text).strip(" ,")
    words = text.split()
    if len(words) <= max_words:
        return text
    # Prefer complete comma-delimited visual clauses instead of a hard cut in
    # the middle of "shoulder-length brown hair".
    selected = []
    used = 0
    for clause in text.split(","):
        clause_words = clause.strip().split()
        if not clause_words:
            continue
        if used + len(clause_words) > max_words:
            break
        selected.append(" ".join(clause_words))
        used += len(clause_words)
    return ", ".join(selected) if selected else " ".join(words[:max_words])


class StableDiffusionAvatarGenerator:
    def __init__(
        self,
        model_path,
        *,
        # The base model's earlier 24-step/flat-illustration setting produced
        # noisy, rudimentary portraits. A 40-step run remained under 7 seconds
        # on the project's RTX 4060 while preserving the face and clothing.
        steps: int = 40,
        width: int = 512,
        height: int = 512,
    ):
        self.model_path = Path(model_path)
        self.steps = int(steps)
        self.width = int(width)
        self.height = int(height)
        self._pipeline = None
        self._lock = Lock()

    def available(self) -> bool:
        return (self.model_path / "model_index.json").is_file()

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        if not self.available():
            raise ImageGenerationUnavailable(
                "本地 SD1.5 模型尚未准备，请运行 scripts/setup_image_runtime.ps1"
            )
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise ImageGenerationUnavailable(
                "本地图片运行库尚未安装，请运行 scripts/setup_image_runtime.ps1"
            ) from exc
        if not torch.cuda.is_available():
            raise ImageGenerationUnavailable("本地 SD1.5 需要可用的 CUDA 显卡")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
        pipe = pipe.to("cuda")
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        pipe.set_progress_bar_config(disable=True)
        self._pipeline = pipe
        return pipe

    @staticmethod
    def build_avatar_prompt(card: dict) -> str:
        del card
        # Chinese card prose is useful to the chat model but not to SD1.5's
        # English CLIP encoder. The configured cloud model supplies a compact
        # English prompt when available; this deterministic fallback keeps the
        # local path usable without sending character data to another service.
        return _DEFAULT_AVATAR_PROMPT

    @staticmethod
    def prepare_prompt(prompt: str | None, card: dict) -> str:
        candidates = [prompt, StableDiffusionAvatarGenerator.build_avatar_prompt(card)]
        for candidate in candidates:
            # Reserve roughly half of SD1.5's 77-token CLIP window for the
            # fixed quality/framing prefix. Long prompts were previously
            # truncated before important visual instructions.
            compact = _compact_sd_prompt(candidate or "", max_words=32)
            if compact:
                if compact.lower().startswith("polished semi-realistic anime portrait"):
                    return compact
                return _PORTRAIT_PREFIX + compact
        return _DEFAULT_AVATAR_PROMPT

    def generate_avatar(self, card: dict, prompt: str | None = None) -> tuple[bytes, dict]:
        prompt = self.prepare_prompt(prompt, card)
        seed = int(
            hashlib.sha256(str(card.get("id") or card.get("name")).encode("utf-8")).hexdigest()[:8],
            16,
        )
        with self._lock:
            try:
                import torch
                pipe = self._load()
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe(
                    prompt=prompt,
                    negative_prompt=_DEFAULT_NEGATIVE_PROMPT,
                    width=self.width,
                    height=self.height,
                    num_inference_steps=self.steps,
                    guidance_scale=7.0,
                    generator=generator,
                ).images[0]
            except ImageGenerationUnavailable:
                raise
            except Exception as exc:
                logger.exception("SD1.5 avatar generation failed")
                raise RuntimeError("本地 SD1.5 生成角色头像失败") from exc
        output = io.BytesIO()
        image.save(output, format="WEBP", quality=90, method=6)
        return output.getvalue(), {
            "prompt": prompt,
            "seed": seed,
            "width": image.width,
            "height": image.height,
            "steps": self.steps,
            "guidance_scale": 7.0,
            "negative_prompt": _DEFAULT_NEGATIVE_PROMPT,
            "scheduler": "pndm",
            "model": "stable-diffusion-v1-5-fp16",
        }

    def unload(self):
        with self._lock:
            self._pipeline = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
