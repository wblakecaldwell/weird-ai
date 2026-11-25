"""Shared art-direction bible for the Class of 2025 Gumball-style Meme Yearbook."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleSpec:
    id: str
    label: str
    count_in_20: int
    llm_guidance: str
    prompt_prefix: str
    prompt_suffix: str
    use_pixel_lora: bool = False
    pixelate_grid: int | None = None  # Nearest-neighbor downscale width for crisp pixel look


# Shared framing rules so every student looks like they sat on the exact same studio stool
STUDIO_FRAMING = (
    "centered head-and-shoulders high school senior yearbook portrait photograph, "
    "facing slightly angled toward camera, looking directly into the lens, "
    "classic mottled slate-blue and gray Lifetouch photography studio canvas backdrop, "
    "soft front key light with subtle catchlight, crisp focus, high quality, no text, no watermark"
)

STYLES: dict[str, StyleSpec] = {
    "pixel_16bit": StyleSpec(
        id="pixel_16bit",
        label="16-Bit Pixel Sprite",
        count_in_20=4,
        llm_guidance=(
            "Crisp 16-bit SNES/GBA pixel-art character sprite (bold pixel outlines, goofy iconic "
            "silhouette, expressive blocky face) wearing high school clothes"
        ),
        prompt_prefix="pixel art portrait, 16-bit retro video game sprite character, crisp chunky pixels, bold dark pixel outlines, ",
        prompt_suffix=(
            ", wearing high school attire, centered head-and-shoulders yearbook portrait framing, "
            "solid mottled blue-gray retro studio gradient backdrop, clean pixel art aesthetic, no text, no watermark"
        ),
        use_pixel_lora=True,
        pixelate_grid=96,
    ),
    "ms_paint_doodle": StyleSpec(
        id="ms_paint_doodle",
        label="MS Paint Ink Cryptid",
        count_in_20=3,
        llm_guidance=(
            "Crude hand-drawn 2008 MS Paint / black-marker meme character with jagged ink lines, "
            "wildly exaggerated facial expression (like an original cousin of classic rage-comic line art, "
            "NOT an existing meme), weird warped proportions"
        ),
        prompt_prefix=(
            "mixed-media yearbook photo of a bizarre hand-drawn black-ink MS Paint doodle character "
            "with thick wobbly black marker lines, stark white paper fill, absurd exaggerated comic facial features, "
        ),
        prompt_suffix=f", {STUDIO_FRAMING}",
    ),
    "cursed_3d_2002": StyleSpec(
        id="cursed_3d_2002",
        label="Early-2000s 3D CGI",
        count_in_20=3,
        llm_guidance=(
            "Slightly uncanny early-2000s low-poly glossy 3D CGI character (bowling-alley screen or "
            "PS2 mascot energy, shiny plastic textures, goofy proportions, intense expression)"
        ),
        prompt_prefix=(
            "early-2000s glossy 3D CGI rendered character with shiny plastic shaders and goofy low-poly charm, "
        ),
        prompt_suffix=f", {STUDIO_FRAMING}",
    ),
    "felt_puppet": StyleSpec(
        id="felt_puppet",
        label="Fuzzy Felt Puppet",
        count_in_20=2,
        llm_guidance=(
            "Handmade fuzzy felt/fleece puppet creature with mismatched googly or button eyes, "
            "visible yarn stitching, chaotic eyebrow expression, wearing a real collared shirt or hoodie"
        ),
        prompt_prefix=(
            "real studio photograph of a handmade fuzzy felt and fleece puppet character with visible fabric fibers, "
            "button or plastic eyes, quirky stitched mouth, "
        ),
        prompt_suffix=f", {STUDIO_FRAMING}",
    ),
    "claymation": StyleSpec(
        id="claymation",
        label="Stop-Motion Clay",
        count_in_20=2,
        llm_guidance=(
            "Lumpy stop-motion plasticine clay figurine with visible thumbprints, crooked sculpted "
            "eyebrows, odd proportions, wearing a tiny knit sweater or varsity jacket"
        ),
        prompt_prefix=(
            "studio macro photograph of a handmade stop-motion plasticine clay character figurine "
            "with visible clay texture and thumbprints, sculpted expressive face, "
        ),
        prompt_suffix=f", {STUDIO_FRAMING}",
    ),
    "cutout_collage": StyleSpec(
        id="cutout_collage",
        label="Paper Cutout Collage",
        count_in_20=2,
        llm_guidance=(
            "Surreal handmade cardboard and magazine-cutout character with mismatched glued eyes, "
            "masking tape details, paper layers (Gumball / Terry Gilliam mixed-media vibe)"
        ),
        prompt_prefix=(
            "studio photograph of a quirky handmade cardboard and paper-cutout collage character "
            "with layered torn paper edges, taped-on expressive eyes and mouth, "
        ),
        prompt_suffix=f", {STUDIO_FRAMING}",
    ),
    "thick_cartoon_2d": StyleSpec(
        id="thick_cartoon_2d",
        label="Flat 2D Cartoon",
        count_in_20=2,
        llm_guidance=(
            "Bold flat-colored 2D cartoon character with thick clean black outlines and an absurd "
            "non-human body/head (living object, weird geometric creature, or surreal food item)"
        ),
        prompt_prefix=(
            "mixed-media yearbook portrait of a flat-shaded 2D animated cartoon character with bold black ink outlines "
            "and vivid saturated flat colors, composited seamlessly into a real photograph, "
        ),
        prompt_suffix=f", {STUDIO_FRAMING}",
    ),
    "uncanny_photo_creature": StyleSpec(
        id="uncanny_photo_creature",
        label="Photorealistic Absurd",
        count_in_20=2,
        llm_guidance=(
            "Completely photorealistic DSLR portrait of an impossible absurd creature or sentient everyday "
            "object with a bizarrely human expression, wearing real high school clothes"
        ),
        prompt_prefix=(
            "ultra-realistic DSLR studio photograph of a bizarre surreal sentient creature or living object "
            "with deadpan human-like personality, real fabric textures, "
        ),
        prompt_suffix=f", {STUDIO_FRAMING}",
    ),
}


def get_style_plan(total: int = 20) -> list[str]:
    """Return a balanced list of `total` style IDs matching the Gumball mixed-media ratio."""
    base_plan: list[str] = []
    for style_id, spec in STYLES.items():
        base_plan.extend([style_id] * spec.count_in_20)
    if total <= len(base_plan):
        return base_plan[:total]
    # Repeat proportionally if user scales past 20
    result: list[str] = []
    while len(result) < total:
        result.extend(base_plan)
    return result[:total]


def build_image_prompt(style_id: str, visual_description: str) -> str:
    """Combine style bible framing with a student's unique visual blueprint."""
    spec = STYLES[style_id]
    desc = visual_description.strip().rstrip(".")
    return f"{spec.prompt_prefix}{desc}{spec.prompt_suffix}"
