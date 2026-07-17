import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from html_chunker import split_html_into_chunks

MAX_CONCURRENT_CHUNK_REQUESTS = 10

# Body chunks may emit hover/keyframe rules and font @import statements here since
# those can't be expressed as inline style="..." attributes; collected and
# assembled into one <style> block.
_STYLE_COMMENT_RE = re.compile(r"<!--STYLE:(.*?)-->", re.DOTALL)
# (?<![\w-]) rather than \b: \b alone also matches inside "data-src=" (the
# '-'->'s' transition is a word boundary), which would validate the img-src
# safety check below against the wrong attribute on lazy-loaded images.
_IMG_SRC_RE = re.compile(r'<img\b[^>]*(?<![\w-])src=["\']([^"\']+)["\']', re.IGNORECASE)
# Identifiers the model may introduce inside a <!--STYLE:...--> comment for a
# hover/animation hook — used to namespace them per chunk (see
# _namespace_chunk_style_identifiers) so two independently-regenerated chunks
# can never collide on the same class/keyframe name.
_STYLE_CLASS_SELECTOR_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_-]*)")
_KEYFRAMES_NAME_RE = re.compile(r"@keyframes\s+([A-Za-z_][A-Za-z0-9_-]*)")


def _extract_img_srcs(html: str) -> set[str]:
    return set(_IMG_SRC_RE.findall(html))


def _namespace_chunk_style_identifiers(regenerated_html: str, chunk_index: int) -> str:
    """Renames model-introduced hook classes and @keyframes names found inside
    the trailing <!--STYLE:...--> comment by suffixing them with the chunk
    index, so two independently-regenerated chunks can never collide on the
    same hook class/animation name. Renames every occurrence of each
    identifier throughout regenerated_html (class= attributes, animation/
    animation-name references, and the STYLE comment itself)."""
    match = _STYLE_COMMENT_RE.search(regenerated_html)
    if not match:
        return regenerated_html

    style_content = match.group(1)
    identifiers = set(_STYLE_CLASS_SELECTOR_RE.findall(style_content)) | set(
        _KEYFRAMES_NAME_RE.findall(style_content)
    )
    if not identifiers:
        return regenerated_html

    renamed = regenerated_html
    for name in identifiers:
        renamed = re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])", f"{name}-c{chunk_index}", renamed)
    return renamed


def _build_style_block(style_rules: list[str]) -> str:
    # @import must precede all other rules per the CSS spec. Rules are deduped
    # (preserving first-seen order) since independently-regenerated chunks can
    # emit identical imports or hover/keyframe rules.
    import_rules = list(dict.fromkeys(r for r in style_rules if r.startswith("@import")))
    other_rules = list(dict.fromkeys(r for r in style_rules if not r.startswith("@import")))
    joined_rules = "\n".join(import_rules + other_rules)
    return f"<style>\n{joined_rules}\n</style>"


def _regenerate_chunk(
    client: OpenAI,
    chunk: str,
    theme_prompt: str,
    regeneration_theme: str | None,
    chunk_index: int,
    total_chunks: int,
) -> str:
    theme_label = regeneration_theme or "a clean modern redesign"
    system_msg = (
        f"""You are an HTML and web design expert specializing in dramatic visual transformations.

            You will receive a fragment of an HTML page whose elements already carry inline style="..." attributes
            (the CSS has already been inlined per-element upstream). Those inline styles currently reflect the
            ORIGINAL site's look. Rewrite every element's style="..." attribute so its styling matches this theme:
            {theme_label}. Replace the existing values — do not layer new declarations on top of the old ones.

            You should try to change the following if a change fits the theme— not just colors:

            TYPOGRAPHY:
            Set font-family in each element's inline style to theme-appropriate fonts,
            Change font-size, font-weight, letter-spacing, and line-height inline to match the theme,
            If you use a distinctive font that isn't a standard system font (e.g. a Google Font), it must be loaded
            or it will silently fall back to the browser default — include an
            @import url('https://fonts.googleapis.com/...'); as the content of a trailing HTML comment of the exact
            form <!--STYLE:@import url('...');--> (the same comment convention used for hover/keyframe rules below),

            COLORS:
            Replace every background-color, color, and border-color in each style attribute,
            Build a cohesive color palette — do not just swap one color for another,
            Apply the palette consistently across all elements,

            BORDERS & SHAPES:
            Change border and border-radius values inline,
            A futuristic theme might use sharp corners; organic themes use rounded ones,

            SPACING & LAYOUT:
            Change padding and margin values inline to reflect the theme's density,
            Compact themes feel tight; luxurious themes use generous whitespace,

            DECORATIVE EFFECTS:
            Add or rewrite box-shadow, text-shadow, and background-image gradients inline,

            HOVER EFFECTS & ANIMATIONS:
            Inline style="..." attributes cannot express :hover or @keyframes. Where a hover effect or animation
            fits the theme, add a class name to the element purely as an animation/hover hook (do not use that
            class for any other styling), and emit the corresponding CSS rule(s) in a single trailing HTML
            comment of the exact form: <!--STYLE:.your-class:hover{{...}} @keyframes your-anim{{...}}-->
            placed at the very end of your output, after all the HTML. Omit this comment entirely if you added
            no hover/keyframe rules or font imports. Multiple such comments are not allowed — combine everything
            into a single trailing <!--STYLE:...--> comment.

            IT IS VERY IMPORTANT THAT THE WEBSITE LOOKS CLEAN AND NOT CLUNKY/MESSY

            RULES:
            Return ONLY the HTML fragment — no explanations, no markdown, no code fences, no <html>/<head>/<body> wrapper,
            Do not remove any HTML elements — every original element must appear in your output,
            Preserve all existing class and id attributes exactly as they are (except for hover/animation hook classes you add),
            Preserve every <img> src attribute exactly as given — do not invent, replace, or omit image URLs,
            Do not change width, height, max-width, max-height, object-fit, or aspect-ratio on img, picture, svg, video, or canvas elements,
            Preserve all factual content (text, product names, prices) and emojis exactly as they appear,
            The transformation must be immediately obvious at a glance"""
    )
    user_msg = (
        f"{theme_prompt}\n\n"
        f"This is chunk {chunk_index + 1} of {total_chunks} from the full page. "
        f"Regenerate this HTML: {chunk}"
    )
    print(f"Processing chunk {chunk_index + 1}/{total_chunks} ({len(chunk)} chars)...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=16384,
    )

    regenerated_html = response.choices[0].message.content
    if not regenerated_html or not regenerated_html.strip():
        print(
            f"Chunk {chunk_index + 1}/{total_chunks}: model returned empty output; "
            f"keeping original chunk unchanged so content isn't lost"
        )
        return chunk

    original_img_srcs = _extract_img_srcs(chunk)
    if original_img_srcs and not original_img_srcs.issubset(_extract_img_srcs(regenerated_html)):
        print(
            f"Chunk {chunk_index + 1}/{total_chunks}: model dropped or altered an <img> src; "
            f"keeping original chunk unchanged so images don't break"
        )
        return chunk

    return _namespace_chunk_style_identifiers(regenerated_html, chunk_index)


def regenerate_html(
    client: OpenAI,
    html: str,
    theme_prompt: str,
    regeneration_theme: str | None,
    on_chunk_complete: Callable[[int, int], None] | None = None,
) -> str:
    """Regenerates HTML with inline per-element styling using GPT-4o, processing
    chunks in parallel. on_chunk_complete(chunk_index, total_chunks) is called
    (thread-safely by the caller's responsibility) after each chunk finishes —
    use it to publish per-chunk status updates."""
    chunks, labels = split_html_into_chunks(html)
    print(f"Split HTML into {len(chunks)} chunk(s) for processing")

    # The head needs no AI regeneration — it carries no visual styling to re-theme.
    # Chunks labeled "body" (or "raw", when head/body couldn't be parsed) go to the model.
    regen_indices = [i for i, label in enumerate(labels) if label != "head"]
    results = {}
    if labels[0] == "head":
        results[0] = chunks[0]

    with ThreadPoolExecutor(max_workers=min(len(regen_indices), MAX_CONCURRENT_CHUNK_REQUESTS) or 1) as executor:
        futures = {
            executor.submit(_regenerate_chunk, client, chunks[i], theme_prompt, regeneration_theme, i, len(chunks)): i
            for i in regen_indices
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            if on_chunk_complete:
                on_chunk_complete(idx, len(chunks))

    if labels[0] == "raw":
        # split_html_into_chunks labels every chunk "raw" together, never
        # mixed with "head"/"body" — there can be one or more of them.
        style_rules = []
        cleaned_parts = []
        for i in range(len(chunks)):
            part = results[i]
            style_rules.extend(m.strip() for m in _STYLE_COMMENT_RE.findall(part) if m.strip())
            cleaned_parts.append(_STYLE_COMMENT_RE.sub("", part))
        cleaned_html = "".join(cleaned_parts)
        if style_rules:
            return f"{_build_style_block(style_rules)}\n{cleaned_html}"
        return cleaned_html

    head_content = results[0]
    body_parts = [results[i] for i in range(1, len(chunks))]

    style_rules = []
    cleaned_body_parts = []
    for part in body_parts:
        style_rules.extend(m.strip() for m in _STYLE_COMMENT_RE.findall(part) if m.strip())
        cleaned_body_parts.append(_STYLE_COMMENT_RE.sub("", part))

    style_block = f"\n{_build_style_block(style_rules)}" if style_rules else ""

    return (
        f"<!DOCTYPE html>\n<html>\n"
        f"<head>{head_content}{style_block}</head>\n"
        f"<body>{''.join(cleaned_body_parts)}</body>\n"
        f"</html>"
    )
