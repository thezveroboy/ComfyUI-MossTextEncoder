# ComfyUI Moss Text Encoder

Vision-aware text conditioning for **MOSS-Image**.

MOSS-Image conditions only through its Qwen3.5-9B text encoder: there is no reference
latent, no VAE input and no second conditioning channel. This package exposes that path
as a normal text-encode node that can also feed reference images into the encoder, so
they can guide the prompt instead of merely being described.

## Installation

Clone into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thezveroboy/ComfyUI-MossTextEncoder
```

No extra Python dependencies are required.

## Nodes

### Text Encode (MOSS Image)

Category: `model/conditioning/moss image`.

Inputs:

- `prompt` / `negative_prompt` — text (dynamic prompts supported).
- `system_prompt` (optional, STRING, `forceInput`) — extra system instruction. Connect
  **MOSS-Image System Prompt** or any string node.
- `image1` / `mask1` (optional) — reference image and its mask. More `imageN`/`maskN`
  pairs appear as slots get connected (up to 16).
- `resolution` — reference images are resized to about `resolution × resolution`
  pixels, preserving aspect ratio and rounding both sides to a multiple of 32.
  `0` keeps each reference at its own size (also rounded to 32).
- `mask_padding` — context kept around the mask before cropping, as a fraction of the
  image size added on **each** side. `0` = tight crop, `0.1` = about a 10% margin.
  Ignored when no mask is connected for that image.
- `vision_position` — `before prompt` (image tokens, then your text) or
  `after prompt`.
- `print_prompt` — print the assembled system + vision + user prompt to the console.

Outputs: `positive`, `negative` (`CONDITIONING`).

Both outputs use the same reference images and the same chat template; only the text
differs. Masks are used per image for cropping, not as a separate conditioning signal.

### MOSS-Image System Prompt

Emits an experimental system instruction that tells the model how to read reference
images and merge them with the user prompt. MOSS-Image has no trained default system
prompt, so treat this as an optional experiment: clear the text field (or do not connect
the node) to send no system message.

## Notes

- The text encoder is selected in **CLIPLoader** with type `moss_image`.
- Vision prompting needs a non-FP8 text encoder. A FP8 encoder cannot run the vision
  tower and raises `add_stub not implemented for Float8_e4m3fn`; load bf16/fp16 (or the
  supported int8 checkpoint). Text-only prompts do work with FP8.
- Prompts describing a reference image work best as structured JSON, for example:

```json
{
  "summary": {"text": "Studio portrait on a dark background"},
  "items": [
    {"label": "subject", "caption": "woman", "text": "short dark hair, neutral expression", "bbox": [120, 80, 640, 900]}
  ]
}
```

## License

MIT
