import math
import re

import torch

import comfy.utils


VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# Experimental only: MOSS-Image was not trained with a fixed system instruction.
MOSS_SYSTEM_PROMPT = (
    "You are preparing a prompt for MOSS-Image. If reference images are supplied, use them as "
    "visual context: identify their subjects, composition, colors, materials, lighting and style, "
    "and carry the relevant details into the requested image. Resolve conflicts in favor of the "
    "user's explicit instructions. Return only the prompt, in the format the user requests."
)


class TextEncodeMossImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Text prompt. With reference images connected, the vision tokens are "
                               "placed before or after this text according to vision_position.",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Negative prompt. The same reference images and template are used for "
                               "the negative conditioning.",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional system instruction. MOSS-Image has no trained default system "
                               "prompt, so leaving this unconnected sends no system message. Connect "
                               "MOSS-Image System Prompt (or any STRING) to add one.",
                }),
                "image1": ("IMAGE",),
                "mask1": ("MASK",),
                "resolution": ("INT", {
                    "default": 1024, "min": 0, "max": 4096, "step": 32,
                    "tooltip": "Reference images are resized to about resolution x resolution pixels, "
                               "at multiples of 32, preserving aspect ratio. 0 keeps each reference at "
                               "its own size, rounded to a multiple of 32.",
                }),
                "mask_padding": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02,
                    "tooltip": "Context kept around the mask before cropping, as a fraction of the "
                               "image size added on EACH side. 0 = tight crop to the mask; 0.1 = ~10% "
                               "margin of surroundings. Only applies when a mask is connected.",
                }),
                "vision_position": (["before prompt", "after prompt"], {
                    "default": "before prompt",
                    "tooltip": "Where the image (vision) tokens sit in the user turn relative to your "
                               "text. 'before prompt' = image then text (default); 'after prompt' = text "
                               "then image. No effect without an image.",
                }),
                "print_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print the full assembled prompt sent to the MOSS-Image text encoder "
                               "(system instruction + vision blocks + your text) to the console.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "model/conditioning/moss image"
    DESCRIPTION = ("MOSS-Image text conditioning with optional vision prompting. Reference images are "
                   "passed as repeated MOSS vision blocks through the Qwen3.5-9B text encoder, with an "
                   "optional per-image mask crop. No VAE or latent input is needed because MOSS-Image "
                   "conditions only through the text encoder.")

    @staticmethod
    def _collect_indexed(kwargs, prefix):
        pattern = re.compile(r"^{}(\d+)$".format(prefix))
        out = {}
        for key, value in kwargs.items():
            match = pattern.match(key)
            if match is not None and value is not None:
                out[int(match.group(1))] = value
        return out

    @staticmethod
    def _crop_to_mask(image, mask, padding=0.0):
        if mask is None:
            return image

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])

        h, w = image.shape[1], image.shape[2]
        if mask.shape[-2:] != (h, w):
            resized = comfy.utils.common_upscale(
                mask.unsqueeze(1), w, h, "bilinear", "disabled"
            )
            mask = resized[:, 0]

        presence = (mask > 0.5).any(dim=0)
        if not bool(presence.any()):
            return image

        rows = torch.where(torch.any(presence, dim=1))[0]
        cols = torch.where(torch.any(presence, dim=0))[0]
        y0, y1 = int(rows[0]), int(rows[-1])
        x0, x1 = int(cols[0]), int(cols[-1])

        if padding > 0.0:
            pad_x = round(padding * w)
            pad_y = round(padding * h)
            x0 = max(0, x0 - pad_x)
            x1 = min(w - 1, x1 + pad_x)
            y0 = max(0, y0 - pad_y)
            y1 = min(h - 1, y1 + pad_y)

        return image[:, y0:y1 + 1, x0:x1 + 1, :]

    @staticmethod
    def _resize_vision(image, resolution):
        samples = image.movedim(-1, 1)
        if resolution > 0:
            ratio = samples.shape[3] / samples.shape[2]
            width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
            height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
        else:
            width = round(samples.shape[3] / 32) * 32
            height = round(samples.shape[2] / 32) * 32
        width, height = max(32, width), max(32, height)
        if (width, height) == (samples.shape[3], samples.shape[2]):
            return image
        resized = comfy.utils.common_upscale(
            samples, width, height, "lanczos", "disabled"
        )
        return resized.movedim(1, -1)

    @classmethod
    def _prepare_vision(cls, kwargs, resolution, mask_padding):
        images = cls._collect_indexed(kwargs, "image")
        masks = cls._collect_indexed(kwargs, "mask")

        prepared = []
        for n in sorted(images.keys()):
            cropped = cls._crop_to_mask(
                images[n][:1], masks.get(n), padding=mask_padding
            )
            resized = cls._resize_vision(cropped, resolution)
            rgb = resized[:, :, :, :3]
            if resized.shape[-1] > 3:
                alpha = resized[:, :, :, 3:]
                rgb = rgb * alpha + (1.0 - alpha)
            prepared.append(rgb)
        return prepared

    @staticmethod
    def _build_text(prompt, image_prompt, vision_position):
        if image_prompt:
            text = (
                prompt + image_prompt
                if vision_position == "after prompt"
                else image_prompt + prompt
            )
        else:
            text = prompt
        template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        return text, template

    @staticmethod
    def _render_prompt(system_prompt, text):
        system = (system_prompt or "").strip()
        prefix = ""
        if system:
            prefix = "<|im_start|>system\n" + system + "<|im_end|>\n"
        return prefix + (
            "<|im_start|>user\n" + text + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )

    @staticmethod
    def _tokenize(clip, text, images, template, system_prompt):
        kwargs = {
            "images": images,
            "llama_template": template,
            "prevent_empty_text": True,
        }
        system = (system_prompt or "").strip()
        if system:
            kwargs["system_prompt"] = system
        return clip.tokenize(text, **kwargs)

    @staticmethod
    def _fp8_hint(exc, images):
        if images and isinstance(exc, NotImplementedError) and "Float8" in str(exc):
            return RuntimeError(
                "MOSS Image: the Qwen3.5-9B text encoder is loaded in FP8, which ComfyUI's vision "
                "tower cannot run on the image path ('add_stub not implemented for Float8_e4m3fn'). "
                "Load a non-FP8 text encoder (bf16/fp16 or the supported int8 checkpoint) with "
                "CLIPLoader type 'moss_image' when using image references. The FP8 encoder works "
                "only text-only."
            )
        return None

    def encode(
        self,
        clip,
        prompt,
        negative_prompt,
        resolution=1024,
        mask_padding=0.0,
        system_prompt="",
        vision_position="before prompt",
        print_prompt=False,
        **kwargs,
    ):
        images = self._prepare_vision(kwargs, resolution, mask_padding)
        image_prompt = VISION_BLOCK * len(images)

        text_pos, template = self._build_text(
            (prompt or "").strip(), image_prompt, vision_position
        )
        text_neg, _ = self._build_text(
            (negative_prompt or "").strip(), image_prompt, vision_position
        )

        if print_prompt:
            print("\n========== Text Encode (MOSS Image) -> Qwen3.5 prompt ==========")
            print(self._render_prompt(system_prompt, text_pos))
            print(
                "---- references: {} ({} vision block(s)) ----".format(
                    len(images), len(images)
                )
            )
            print("===============================================================\n")

        tokens_pos = self._tokenize(clip, text_pos, images, template, system_prompt)
        tokens_neg = self._tokenize(clip, text_neg, images, template, system_prompt)

        try:
            positive = clip.encode_from_tokens_scheduled(tokens_pos)
            negative = clip.encode_from_tokens_scheduled(tokens_neg)
        except NotImplementedError as exc:
            hint = self._fp8_hint(exc, images)
            if hint is not None:
                raise hint from exc
            raise

        return (positive, negative)


class MossImageSystemPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": MOSS_SYSTEM_PROMPT,
                    "tooltip": "Experimental system instruction for MOSS-Image. The model has no "
                               "trained default system prompt; use this to describe how reference "
                               "images and the user prompt should be combined. Clear the field to "
                               "send no system message.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "build"
    CATEGORY = "model/conditioning/moss image"
    DESCRIPTION = "Optional experimental system instruction for MOSS-Image conditioning."

    def build(self, text):
        return ((text or "").strip(),)


NODE_CLASS_MAPPINGS = {
    "TextEncodeMossImage": TextEncodeMossImage,
    "MossImageSystemPrompt": MossImageSystemPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeMossImage": "Text Encode (MOSS Image)",
    "MossImageSystemPrompt": "MOSS-Image System Prompt",
}

__all__ = [
    "TextEncodeMossImage",
    "MossImageSystemPrompt",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

