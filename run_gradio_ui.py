import os
import random
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps


ROOT = Path(__file__).parent
SD_BIN_DIR = ROOT / "sd_bin"
MODEL_CONFIG_PATH = ROOT / "models" / "zimage" / "selected_model.txt"
MODEL_NAME = os.environ.get("ZIMAGE_MODEL_NAME")
if not MODEL_NAME and MODEL_CONFIG_PATH.exists():
    MODEL_NAME = MODEL_CONFIG_PATH.read_text(encoding="utf-8").strip()
if not MODEL_NAME:
    MODEL_NAME = "z_image_turbo_Q4_0.gguf"
MODEL_PATH = str(ROOT / "models" / "zimage" / MODEL_NAME)
LORA_DIR = ROOT / "models" / "loras"
OUTDIR = ROOT / "outputs"
TEMP_INPUT_DIR = ROOT / "tmp_inputs"

DEFAULT_VAE_PATH = str(ROOT / "models" / "vae" / "ae.safetensors")
DEFAULT_LLM_PATH = str(ROOT / "models" / "llm" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")

OUTDIR.mkdir(exist_ok=True)
LORA_DIR.mkdir(exist_ok=True)
TEMP_INPUT_DIR.mkdir(exist_ok=True)

current_proc = None
FIRST_RUN = True
LAST_SEED = None

RES_PRESETS = [
    ("1:1 (256x256)", 256, 256),
    ("1:1 (512x512)", 512, 512),
    ("1:1 (768x768)", 768, 768),
    ("1:1 (1024x1024)", 1024, 1024),
    ("16:9 (640x384)", 640, 384),
    ("16:9 (896x512)", 896, 512),
    ("16:9 (1024x576)", 1024, 576),
    ("9:16 (384x640)", 384, 640),
    ("9:16 (512x896)", 512, 896),
    ("9:16 (576x1024)", 576, 1024),
    ("4:3 (640x480)", 640, 480),
    ("4:3 (768x576)", 768, 576),
    ("3:2 (768x512)", 768, 512),
    ("2:3 (512x768)", 512, 768),
]

SIZE_OPTIONS = sorted({s for _, w, h in RES_PRESETS for s in (w, h)})
VRAM_PRESETS = [
    "4GB (safest)",
    "6-8GB (balanced)",
    "10GB+ (fastest)",
]
LORA_APPLY_MODES = ["auto", "immediately", "at_runtime"]


def find_sd_executable():
    """Auto-detect available stable-diffusion executable."""
    candidates = [
        ("sd-cli.exe", "sd-cli.exe (recommended)"),
        ("sd.exe", "sd.exe (legacy)"),
    ]
    for exe_name, label in candidates:
        exe_path = SD_BIN_DIR / exe_name
        if exe_path.exists():
            return str(exe_path), label
    return None, None


SD_EXE, SD_EXE_LABEL = find_sd_executable()


def get_lora_list():
    """List available LoRA files in the loras directory."""
    if not LORA_DIR.exists():
        return []
    return [f.name for f in sorted(LORA_DIR.glob("*.safetensors"))]


def get_recent_outputs(limit=12):
    images = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        images.extend(OUTDIR.glob(pattern))
    return [str(p.absolute()) for p in sorted(images, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]]


def apply_preset(preset_label):
    for name, w, h in RES_PRESETS:
        if name == preset_label:
            return w, h
    return gr.update(), gr.update()


def random_seed():
    return random.randint(0, 2_147_483_647)


def reuse_last_seed():
    if LAST_SEED is None:
        return gr.update()
    return LAST_SEED


def refresh_loras():
    return gr.update(choices=get_lora_list())


def refresh_gallery():
    return get_recent_outputs()


def set_unlocked(enabled):
    return gr.update(interactive=bool(enabled)), gr.update(interactive=bool(enabled))


def set_img2img_enabled(enabled):
    return gr.update(interactive=bool(enabled)), gr.update(interactive=bool(enabled))


def stop_gen():
    global current_proc
    if current_proc and current_proc.poll() is None:
        print("Stopping generation...")
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(current_proc.pid)], capture_output=True)
        else:
            current_proc.terminate()
        return "Generation stopped by user."
    return "No active generation to stop."


def normalize_seed(seed_value):
    try:
        seed_int = int(seed_value)
    except (TypeError, ValueError):
        seed_int = -1
    if seed_int < 0:
        return random_seed()
    return seed_int


def append_lora_tags(prompt, selected_loras, lora_strength):
    final_prompt = prompt or ""
    if selected_loras:
        for lora in selected_loras:
            lora_name = Path(lora).stem
            final_prompt += f" <lora:{lora_name}:{lora_strength}>"
    return final_prompt


def low_vram_flags(vram_mode, clip_on_cpu, balanced_vae_tiling):
    flags = []
    if vram_mode == "4GB (safest)":
        flags.extend(["--offload-to-cpu", "--diffusion-fa", "--vae-tiling", "--vae-conv-direct"])
        if clip_on_cpu:
            flags.append("--clip-on-cpu")
    elif vram_mode == "6-8GB (balanced)":
        flags.extend(["--offload-to-cpu", "--diffusion-fa"])
        if balanced_vae_tiling:
            flags.append("--vae-tiling")
    elif vram_mode == "10GB+ (fastest)":
        flags.append("--diffusion-fa")
    return flags


def prepare_init_image(init_image_path, width, height):
    if not init_image_path:
        return None, None

    src = Path(init_image_path)
    if not src.exists():
        return None, f"Img2img input image was not found: {src}"

    try:
        image = Image.open(src)
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        dest = TEMP_INPUT_DIR / f"init_{uuid.uuid4().hex[:8]}.png"
        image.save(dest)
        return str(dest.absolute()), None
    except Exception as exc:
        return None, f"Could not prepare img2img input image: {exc}"


def format_command(cmd):
    return subprocess.list2cmdline([str(part) for part in cmd])


def write_metadata(out_file, metadata):
    meta_path = Path(out_file).with_suffix(Path(out_file).suffix + ".txt")
    lines = [
        "Z-Image Turbo generation metadata",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for key, value in metadata.items():
        if value is None or value == "":
            continue
        lines.append(f"{key}: {value}")
    try:
        meta_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        print(f"Could not write metadata file: {exc}")


def gen_image(
    prompt,
    width,
    height,
    steps,
    seed,
    cfg_scale,
    vae_path,
    llm_path,
    selected_loras,
    lora_strength,
    lora_apply_mode,
    vram_mode,
    clip_on_cpu,
    balanced_vae_tiling,
    img2img_enabled,
    init_image_path,
    img2img_strength,
):
    global current_proc

    if SD_EXE is None:
        yield None, "Error: No stable-diffusion executable found.", "", ""
        return

    if img2img_enabled and not init_image_path:
        yield None, "Img2img is enabled, but no input image was uploaded.", "0s", ""
        return

    init_file = None
    init_error = None
    if img2img_enabled:
        init_file, init_error = prepare_init_image(init_image_path, width, height)
        if init_error:
            yield None, init_error, "0s", ""
            return

    uid = uuid.uuid4().hex[:8]
    out_file = str((OUTDIR / f"out_{uid}.png").absolute())
    final_prompt = append_lora_tags(prompt, selected_loras, lora_strength)

    cmd = [
        SD_EXE,
        "--diffusion-model",
        MODEL_PATH,
        "--vae",
        vae_path,
        "--llm",
        llm_path,
        "--lora-model-dir",
        str(LORA_DIR),
        "--lora-apply-mode",
        lora_apply_mode,
        "-p",
        final_prompt,
        "--cfg-scale",
        str(cfg_scale),
        "--steps",
        str(steps),
        "-H",
        str(height),
        "-W",
        str(width),
        "-o",
        out_file,
        "--seed",
        str(seed),
        "--rng",
        "cuda",
    ]
    cmd.extend(low_vram_flags(vram_mode, clip_on_cpu, balanced_vae_tiling))

    if img2img_enabled:
        cmd.extend(["--init-img", init_file, "--strength", str(img2img_strength)])

    cmd_str = format_command(cmd)
    yield None, f"Starting generation...\nCommand: {cmd_str}", "0s", cmd_str

    t_start = time.perf_counter()
    current_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )

    full_log = ""
    try:
        for line in current_proc.stdout:
            print(line, end="")
            full_log += line
            elapsed = int(time.perf_counter() - t_start)
            yield None, full_log.strip(), f"{elapsed}s", cmd_str
    except Exception as exc:
        yield None, f"Error during logging: {exc}", "0s", cmd_str

    current_proc.wait()
    total_time = f"{time.perf_counter() - t_start:.1f}s"

    if current_proc.returncode != 0:
        if current_proc.returncode in [-1, 1, 3221225786, 15]:
            yield None, f"Generation stopped.\n\n{full_log.strip()}", total_time, cmd_str
        else:
            yield (
                None,
                f"sd.exe exited with code {current_proc.returncode}\n\n{full_log.strip()}",
                total_time,
                cmd_str,
            )
        return

    if not os.path.exists(out_file):
        imgs = sorted(OUTDIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not imgs:
            yield None, f"No image was produced.\n\n{full_log.strip()}", total_time, cmd_str
            return
        out_file = str(imgs[0].absolute())

    write_metadata(
        out_file,
        {
            "mode": "img2img" if img2img_enabled else "txt2img",
            "prompt": prompt,
            "final_prompt": final_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "vram_mode": vram_mode,
            "lora_files": ", ".join(selected_loras or []),
            "lora_strength": lora_strength,
            "lora_apply_mode": lora_apply_mode,
            "init_image": init_file,
            "img2img_strength": img2img_strength if img2img_enabled else None,
            "command": cmd_str,
            "generation_time": total_time,
            "log": full_log.strip(),
        },
    )
    yield out_file, full_log.strip(), total_time, cmd_str


with gr.Blocks() as demo:
    gr.Markdown("# Z-Image Turbo - Low VRAM UI")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Basic"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="A large orange octopus on an ocean floor, cinematic, 8k",
                        lines=3,
                    )

                    with gr.Row():
                        preset = gr.Dropdown(
                            [n for n, _, _ in RES_PRESETS],
                            value="1:1 (512x512)",
                            label="Resolution Preset",
                        )
                        steps = gr.Slider(1, 50, value=8, step=1, label="Steps")

                    with gr.Row():
                        width = gr.Dropdown(SIZE_OPTIONS, value=512, label="Width")
                        height = gr.Dropdown(SIZE_OPTIONS, value=512, label="Height")

                    with gr.Row():
                        cfg_scale = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="CFG Scale")
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

                    with gr.Row():
                        random_seed_btn = gr.Button("Random Seed", variant="secondary")
                        reuse_seed_btn = gr.Button("Reuse Last Seed", variant="secondary")

                    with gr.Group():
                        gr.Markdown("### Low VRAM Mode")
                        vram_mode = gr.Radio(
                            VRAM_PRESETS,
                            value="4GB (safest)",
                            label="VRAM Preset",
                        )
                        with gr.Row():
                            clip_on_cpu = gr.Checkbox(
                                value=False,
                                label="4GB extra: keep text encoder on CPU",
                            )
                            balanced_vae_tiling = gr.Checkbox(
                                value=False,
                                label="6-8GB extra: VAE tiling",
                            )

                    with gr.Group():
                        gr.Markdown("### LoRA Support")
                        with gr.Row():
                            lora_list = gr.CheckboxGroup(choices=get_lora_list(), label="Select LoRAs")
                            refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")
                        with gr.Row():
                            lora_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="LoRA Strength")
                            lora_apply_mode = gr.Dropdown(
                                LORA_APPLY_MODES,
                                value="auto",
                                label="LoRA Apply Mode",
                            )

                with gr.Tab("Experimental Img2Img"):
                    gr.Markdown(
                        "Experimental for Z-Image Turbo GGUF; results depend on backend support."
                    )
                    img2img_enabled = gr.Checkbox(value=False, label="Enable img2img")
                    init_image = gr.Image(
                        label="Input image",
                        type="filepath",
                        interactive=False,
                    )
                    img2img_strength = gr.Slider(
                        0.1,
                        1.0,
                        value=0.6,
                        step=0.05,
                        label="Img2Img Strength",
                        interactive=False,
                    )

                with gr.Tab("Advanced"):
                    unlock = gr.Checkbox(value=False, label="Allow editing advanced paths")
                    with gr.Row():
                        vae_path = gr.Textbox(label="VAE path", value=DEFAULT_VAE_PATH, interactive=False)
                        llm_path = gr.Textbox(label="LLM (Qwen) path", value=DEFAULT_LLM_PATH, interactive=False)

            with gr.Row():
                btn = gr.Button("Generate", variant="primary", scale=2)
                stop_btn = gr.Button("Stop", variant="stop", scale=1)

        with gr.Column(scale=2):
            img = gr.Image(label="Result", interactive=False, type="filepath")
            timer_display = gr.Markdown("Generation Time: **0s**")
            gallery = gr.Gallery(label="Recent Outputs", value=get_recent_outputs(), columns=3, height=260)
            refresh_gallery_btn = gr.Button("Refresh Gallery", variant="secondary")
            command_box = gr.Textbox(label="Last Command", interactive=False, lines=3)
            status = gr.Textbox(label="Status / Logs", interactive=False, lines=14)

    preset.change(apply_preset, inputs=[preset], outputs=[width, height])
    refresh_btn.click(refresh_loras, outputs=[lora_list])
    random_seed_btn.click(random_seed, outputs=[seed])
    reuse_seed_btn.click(reuse_last_seed, outputs=[seed])
    refresh_gallery_btn.click(refresh_gallery, outputs=[gallery])
    unlock.change(set_unlocked, inputs=[unlock], outputs=[vae_path, llm_path])
    img2img_enabled.change(set_img2img_enabled, inputs=[img2img_enabled], outputs=[init_image, img2img_strength])

    def run_and_return(
        p,
        w,
        h,
        st,
        sd,
        cfg,
        vae,
        llm,
        l_list,
        l_str,
        l_apply_mode,
        low_vram_mode,
        keep_clip_cpu,
        use_balanced_vae_tiling,
        use_img2img,
        input_image,
        strength,
    ):
        global FIRST_RUN, LAST_SEED

        run_seed = normalize_seed(sd)
        LAST_SEED = run_seed
        status_msg = "Generating... (first run can take longer)" if FIRST_RUN else "Generating..."
        FIRST_RUN = False

        yield (
            None,
            status_msg,
            gr.update(interactive=False),
            gr.update(interactive=True),
            "Generation Time: **0s**",
            "",
            get_recent_outputs(),
            run_seed,
        )

        last_img = None
        last_log = ""
        last_time = "0s"
        last_command = ""
        for out_img, log, time_str, cmd_str in gen_image(
            p,
            int(w),
            int(h),
            int(st),
            run_seed,
            float(cfg),
            vae,
            llm,
            l_list,
            l_str,
            l_apply_mode,
            low_vram_mode,
            keep_clip_cpu,
            use_balanced_vae_tiling,
            use_img2img,
            input_image,
            strength,
        ):
            if out_img is not None:
                last_img = out_img
            last_log = log
            last_time = time_str
            last_command = cmd_str
            image_update = out_img if out_img is not None else gr.update()
            yield (
                image_update,
                log,
                gr.update(interactive=False),
                gr.update(interactive=True),
                f"Generation Time: **{time_str}**",
                cmd_str,
                get_recent_outputs(),
                run_seed,
            )

        final_image = last_img if last_img is not None else gr.update()
        yield (
            final_image,
            last_log,
            gr.update(interactive=True),
            gr.update(interactive=False),
            f"Generation Time: **{last_time}**",
            last_command,
            get_recent_outputs(),
            run_seed,
        )

    btn.click(
        run_and_return,
        inputs=[
            prompt,
            width,
            height,
            steps,
            seed,
            cfg_scale,
            vae_path,
            llm_path,
            lora_list,
            lora_strength,
            lora_apply_mode,
            vram_mode,
            clip_on_cpu,
            balanced_vae_tiling,
            img2img_enabled,
            init_image,
            img2img_strength,
        ],
        outputs=[img, status, btn, stop_btn, timer_display, command_box, gallery, seed],
    )
    stop_btn.click(stop_gen, outputs=[status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=9000, share=False)
