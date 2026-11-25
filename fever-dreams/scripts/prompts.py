"""Art direction and prompt builders for Gumball mixed-media 1950s Little Golden Books."""

from __future__ import annotations

# Explicit medium-lock instructions so multi-reference edit models preserve each
# character's native yearbook medium inside a 1950s painted storybook background.
MEDIUM_PRESERVATION: dict[str, str] = {
    "pixel_16bit": (
        "remains a crisp 16-bit retro pixel-art video game sprite with chunky visible square pixels "
        "and dark pixel outlines, composited right into the painted world"
    ),
    "ms_paint_doodle": (
        "remains a stark hand-drawn black-ink MS Paint doodle with wobbly thick black marker lines "
        "and flat white paper fill, standing out inside the painted scene"
    ),
    "cursed_3d_2002": (
        "remains a shiny glossy early-2000s low-poly 3D CGI character with specular plastic shaders "
        "and goofy bowling-alley animation charm"
    ),
    "felt_puppet": (
        "remains a tangible handmade fuzzy felt and fleece puppet with visible fabric fibers, "
        "stitched seams, yarn hair, and button or plastic eyes"
    ),
    "claymation": (
        "remains a lumpy handmade stop-motion plasticine clay figurine with visible thumbprints "
        "and sculpted matte clay texture"
    ),
    "cutout_collage": (
        "remains a layered cardboard and torn paper-cutout collage character with masking tape "
        "and flat glued-on features"
    ),
    "thick_cartoon_2d": (
        "remains a flat-shaded 2D animated cartoon character with bold clean black ink outlines "
        "and vivid saturated flat colors"
    ),
    "uncanny_photo_creature": (
        "remains a startlingly photorealistic real-world object/creature with lifelike studio "
        "textures and real tailored fabric clothing"
    ),
}

GOLDEN_BOOK_WORLD_STYLE = (
    "1950s Little Golden Book mid-century children's storybook illustration background, "
    "lush hand-painted matte gouache and tempera brushwork, warm nostalgic mid-century color "
    "palette (mustard gold, dusty teal, coral red, sage green, warm cream), subtle vintage "
    "printed picture-book paper grain, charmingly earnest 1950s suburban and high-school Americana "
    "environment, mixed-media Gumball-style visual contrast where the bizarre multi-medium characters "
    "physically inhabit and cast natural shadows inside the painted 1950s storybook world, "
    "crisp storytelling composition, zero text, zero speech bubbles, zero watermarks"
)


def medium_lock_phrase(style_id: str) -> str:
    return MEDIUM_PRESERVATION.get(
        style_id,
        "keeps their exact distinctive visual medium, face, colors, and clothing from their reference image",
    )


def build_turnaround_prompt(char: dict) -> str:
    """Prompt to expand a head-and-shoulders yearbook portrait into a 3-view full-body character sheet."""
    lock = medium_lock_phrase(char["style_id"])
    return (
        f"Character design turnaround reference sheet of {char['name']} based directly on Image 1. "
        f"Keep the exact same head, face, colors, expression style, and upper-body outfit from Image 1, "
        f"and {lock}. "
        f"Show THREE full-body standing views side-by-side on a clean solid warm-cream studio background: "
        f"(left) full-body front view, (center) full-body three-quarter view walking in motion, "
        f"(right) full-body side profile view. "
        f"Extend their upper-body clothes naturally into matching simple trousers/skirt and shoes that "
        f"fit a quirky high school senior ({char['visual_description']}). "
        f"Clean even lighting, crisp full-body proportions from head to toe, no text, no labels, no watermark."
    )


def build_scene_prompt(
    cast_in_scene: list[dict],
    scene_description: str,
    camera_angle: str,
    lighting_mood: str = "",
    *,
    is_cover: bool = False,
) -> str:
    """Build a multi-reference edit prompt mapping Image 1..N to cast members in a 1950s Golden Book scene."""
    ref_lines: list[str] = []
    for idx, char in enumerate(cast_in_scene, start=1):
        lock = medium_lock_phrase(char["style_id"])
        ref_lines.append(
            f"Image {idx} is {char['name']} ({char['style_label']}), who MUST keep their exact face, "
            f"silhouette, colors, and outfit from Image {idx} and {lock}."
        )
    refs_block = " ".join(ref_lines)

    mood_clause = f" Lighting and mood: {lighting_mood.strip().rstrip('.')}." if lighting_mood else ""
    cover_clause = (
        " Classic Little Golden Book front cover hero illustration: leave clean uncluttered painted "
        "sky or backdrop breathing room along the top 22% of the frame for a printed book title banner, "
        "center the characters dynamically in the lower two-thirds, absolutely NO letters, NO title text, "
        "and NO border frame baked into the artwork."
        if is_cover
        else ""
    )

    return (
        f"{GOLDEN_BOOK_WORLD_STYLE}.{cover_clause} "
        f"{refs_block} "
        f"Camera angle and framing: {camera_angle.strip().rstrip('.')}. "
        f"Scene action: {scene_description.strip().rstrip('.')}.{mood_clause} "
        f"Every character from the reference images appears clearly recognizable, full of expressive "
        f"body language, interacting naturally with the painted 1950s environment. No text, no words, no captions."
    )
