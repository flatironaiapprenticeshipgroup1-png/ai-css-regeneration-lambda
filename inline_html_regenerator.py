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
# Lazy-loaded images often carry their real URL in data-src (with src holding
# a placeholder) — protected separately from src by _extract_img_data_srcs.
_IMG_DATA_SRC_RE = re.compile(r'<img\b[^>]*(?<![\w-])data-src=["\']([^"\']+)["\']', re.IGNORECASE)
# Identifiers the model may introduce inside a <!--STYLE:...--> comment for a
# hover/animation hook — used to namespace them per chunk (see
# _namespace_chunk_style_identifiers) so two independently-regenerated chunks
# can never collide on the same class/keyframe name.
_STYLE_CLASS_SELECTOR_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_-]*)")
_KEYFRAMES_NAME_RE = re.compile(r"@keyframes\s+([A-Za-z_][A-Za-z0-9_-]*)")
_IMPORT_STATEMENT_RE = re.compile(r"@import[^;]*;", re.IGNORECASE)
_RULE_HEADER_RE = re.compile(r"([^{}]+)\{")
_SELECTOR_COMBINATOR_SPLIT_RE = re.compile(r"[\s>+~]+")
# Each head chunk from split_node_into_parts already carries its own
# <head>...</head> wrapper (str(head) includes the tag itself); stripped so
# regenerate_html's own <head>{...}</head> wrapping doesn't nest.
_HEAD_OPEN_TAG_RE = re.compile(r"^<head\b[^>]*>", re.IGNORECASE)
_HEAD_CLOSE_TAG_RE = re.compile(r"</head>$", re.IGNORECASE)


def _strip_head_wrapper(head_chunk: str) -> str:
    return _HEAD_CLOSE_TAG_RE.sub("", _HEAD_OPEN_TAG_RE.sub("", head_chunk, count=1))


def _extract_ordered_img_srcs(html: str) -> list[str]:
    """Left-to-right ordered list of <img> src values (may contain
    duplicates) — used both for the drop/alter check (via set()) and for the
    position-preserved check below, which a plain set comparison can't do."""
    return _IMG_SRC_RE.findall(html)


def _extract_img_srcs(html: str) -> set[str]:
    return set(_extract_ordered_img_srcs(html))


def _extract_img_data_srcs(html: str) -> set[str]:
    return set(_IMG_DATA_SRC_RE.findall(html))


def _img_order_preserved(original_ordered: list[str], new_ordered: list[str]) -> bool:
    """True if every original <img> src still appears, unchanged, at the same
    left-to-right position as a PREFIX of the regenerated output's <img>
    srcs. Catches a same-URL-set swap between two images (e.g. cat.jpg and
    dog.jpg trading places) that a set-based subset check can't see, since
    chunks are already-parsed HTML fragments regenerated in isolation —
    element reordering isn't an expected/legitimate transformation here.
    Tolerates the model appending brand-new <img> tags after all originals;
    (conservatively) rejects a new image inserted mid-sequence, which shifts
    every later original out of its prefix position — a deliberate,
    low-cost false-positive trade (falls back to the original chunk, never
    corrupts) in exchange for closing the swap-detection hole."""
    return new_ordered[: len(original_ordered)] == original_ordered


def _hook_class_candidates(style_content: str) -> set[str]:
    """Returns class names that are the model's own invented hook class(es) —
    i.e. appear in the leftmost compound-selector token of some rule — as
    opposed to a class referenced only in a later, descendant-combinator
    token (e.g. the .icon in ".card-hover:hover .icon{...}"), which points at
    an existing/reused class on a *different* element and must not be
    renamed. If the leftmost token is itself a chained compound selector with
    no combinator (e.g. ".card.hover-lift:hover"), both classes are treated
    as candidates — the system prompt only allows the model one hook class
    per rule, so a compliant model shouldn't produce this shape."""
    content = _IMPORT_STATEMENT_RE.sub("", style_content)
    candidates: set[str] = set()
    for header_match in _RULE_HEADER_RE.finditer(content):
        for selector in header_match.group(1).split(","):
            selector = selector.strip()
            if selector:
                first_token = _SELECTOR_COMBINATOR_SPLIT_RE.split(selector, maxsplit=1)[0]
                candidates.update(_STYLE_CLASS_SELECTOR_RE.findall(first_token))
    return candidates


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
    identifiers = _hook_class_candidates(style_content) | set(_KEYFRAMES_NAME_RE.findall(style_content))
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


_DEFAULT_STYLE_GUIDE = """COLOR PALETTE:
1. Primary: #2563EB - main brand color, primary buttons and links
2. Secondary: #1E293B - headings and dark surfaces
3. Accent: #F59E0B - highlights, calls to action
4. Background: #F8FAFC - page and section backgrounds
5. Text: #0F172A - body text

TYPOGRAPHY:
Headings: 'Inter', sans-serif. Body: 'Inter', sans-serif.

BORDERS & SHAPES:
Rounded corners (border-radius: 8px-12px), subtle 1px borders.

SPACING:
Comfortable, moderately generous spacing.

EFFECTS:
Soft, low-opacity box-shadows; no gradients."""


def generate_style_guide(client: OpenAI, theme_prompt: str, regeneration_theme: str | None) -> str:
    """Generates one shared design style guide for the whole website, called
    once per regeneration job (not per chunk). Threading this same guide into
    every chunk's prompt (see _regenerate_chunk) is what keeps independently-
    regenerated chunks visually consistent with each other, instead of each
    chunk improvising its own color palette/fonts as before."""
    theme_label = regeneration_theme or "a clean modern redesign"
    system_msg = (
        "You are a web design expert. Produce a concise design style guide for a website "
        "theme, to be handed to several independent designers who must each style a different "
        "section of the same page — the guide is the only thing keeping their work consistent, "
        "so it must be specific and unambiguous. Return PLAIN TEXT only (no markdown, no code "
        "fences), following exactly this structure and nothing else:\n\n"
        "COLOR PALETTE:\n"
        "1. Primary: #RRGGBB - <role/usage>\n"
        "2. Secondary: #RRGGBB - <role/usage>\n"
        "3. Accent: #RRGGBB - <role/usage>\n"
        "4. Background: #RRGGBB - <role/usage>\n"
        "5. Text: #RRGGBB - <role/usage>\n\n"
        "TYPOGRAPHY:\n"
        "<heading font family, body font family; name a Google Font if a distinctive look fits "
        "the theme>\n\n"
        "BORDERS & SHAPES:\n"
        "<corner radius and border style direction>\n\n"
        "SPACING:\n"
        "<density direction: compact vs. generous>\n\n"
        "EFFECTS:\n"
        "<shadow/gradient direction>\n\n"
        "Provide EXACTLY 5 colors, no more, no fewer. Every hex code must be a complete, valid "
        "6-digit hex color."
    )
    user_msg = f"{theme_prompt}\n\nGenerate the style guide for this theme: {theme_label}."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=600,
        )
        style_guide = response.choices[0].message.content
    except Exception as e:
        print(f"Style guide generation failed ({e}); falling back to default style guide")
        return _DEFAULT_STYLE_GUIDE

    if not style_guide or not style_guide.strip():
        print("Style guide generation returned empty output; falling back to default style guide")
        return _DEFAULT_STYLE_GUIDE

    return style_guide.strip()


def _regenerate_chunk(
    client: OpenAI,
    chunk: str,
    theme_prompt: str,
    regeneration_theme: str | None,
    style_guide: str,
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

            STYLE GUIDE — every chunk of this page is being regenerated independently by a different
            call like this one, so this shared guide is the only thing keeping them visually consistent.
            Conform to it exactly rather than inventing your own colors/fonts/style direction:
            {style_guide}

            You should try to change the following if a change fits the theme— not just colors:

            TYPOGRAPHY:
            Set font-family in each element's inline style to the fonts specified in the STYLE GUIDE above,
            Change font-size, font-weight, letter-spacing, and line-height inline to match the theme,
            If the STYLE GUIDE names a distinctive font that isn't a standard system font (e.g. a Google Font),
            it must be loaded or it will silently fall back to the browser default — include an
            @import url('https://fonts.googleapis.com/...'); as the content of a trailing HTML comment of the exact
            form <!--STYLE:@import url('...');--> (the same comment convention used for hover/keyframe rules below),

            COLORS:
            Replace every background-color, color, and border-color in each style attribute,
            Use ONLY the 5 colors given in the STYLE GUIDE's COLOR PALETTE above — do not invent or introduce
            any other colors, and apply each of the 5 colors according to the role the STYLE GUIDE assigns it,

            BORDERS & SHAPES:
            Change border and border-radius values inline to match the STYLE GUIDE's BORDERS & SHAPES direction,

            SPACING & LAYOUT:
            Change padding and margin values inline to reflect the STYLE GUIDE's SPACING direction,

            DECORATIVE EFFECTS:
            Add or rewrite box-shadow, text-shadow, and background-image gradients inline to match the STYLE GUIDE's EFFECTS direction,

            HOVER EFFECTS & ANIMATIONS:
            Inline style="..." attributes cannot express :hover or @keyframes. Where a hover effect or animation
            fits the theme, add a class name to the element purely as an animation/hover hook (do not use that
            class for any other styling), and emit the corresponding CSS rule(s) in a single trailing HTML
            comment of the exact form: <!--STYLE:.your-class:hover{{...}} @keyframes your-anim{{...}}-->
            placed at the very end of your output, after all the HTML. Omit this comment entirely if you added
            no hover/keyframe rules or font imports. Multiple such comments are not allowed — combine everything
            into a single trailing <!--STYLE:...--> comment.

            LAYOUT & POSITIONING — DO NOT CHANGE:
            Never modify: display, position, top, right, bottom, left, flex-direction, flex-wrap,
            justify-content, align-items, align-content, align-self, order, float, clear,
            grid-template-columns, grid-template-rows, grid-template-areas, grid-column, grid-row, grid-area.
            If an element is currently a horizontal flex row, a vertical flex column, or part of a CSS grid, it
            must remain exactly that mechanism after your changes — restyle colors, fonts, borders, spacing, and
            effects on top of the existing layout, never the layout itself. (padding and margin may still change
            per SPACING & LAYOUT above — those affect density, not positioning structure.)

            IT IS VERY IMPORTANT THAT THE WEBSITE LOOKS CLEAN AND NOT CLUNKY/MESSY

            RULES:
            Return ONLY the HTML fragment — no explanations, no markdown, no code fences, no <html>/<head>/<body> wrapper,
            Do not remove any HTML elements — every original element must appear in your output,
            Preserve all existing class and id attributes exactly as they are (except for hover/animation hook classes you add),
            Preserve every <img> src attribute exactly as given — do not invent, replace, or omit image URLs,
            Do not change width, height, max-width, max-height, object-fit, or aspect-ratio on img, picture, svg, video, or canvas elements,
            Preserve all factual content (text, product names, prices) and emojis exactly as they appear,
            Do not change the sizes of images
            Do not change svg's to images or vice versa
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

    original_ordered_srcs = _extract_ordered_img_srcs(chunk)
    new_ordered_srcs = _extract_ordered_img_srcs(regenerated_html)
    original_img_srcs = set(original_ordered_srcs)
    original_data_srcs = _extract_img_data_srcs(chunk)

    img_src_broken = original_img_srcs and (
        not original_img_srcs.issubset(set(new_ordered_srcs))
        or not _img_order_preserved(original_ordered_srcs, new_ordered_srcs)
    )
    img_data_src_broken = original_data_srcs and not original_data_srcs.issubset(
        _extract_img_data_srcs(regenerated_html)
    )

    if img_src_broken or img_data_src_broken:
        print(
            f"Chunk {chunk_index + 1}/{total_chunks}: model dropped, reordered, or altered an "
            f"<img> src/data-src; keeping original chunk unchanged so images don't break"
        )
        return chunk

    return _namespace_chunk_style_identifiers(regenerated_html, chunk_index)


def regenerate_html(
    client: OpenAI,
    html: str,
    theme_prompt: str,
    regeneration_theme: str | None,
    style_guide: str,
    on_chunk_complete: Callable[[int, int], None] | None = None,
) -> str:
    """Regenerates HTML with inline per-element styling using GPT-4o, processing
    chunks in parallel. style_guide (see generate_style_guide) is passed identically
    to every chunk so independently-regenerated chunks stay visually consistent.
    on_chunk_complete(chunk_index, total_chunks) is called (thread-safely by the
    caller's responsibility) after each chunk finishes — use it to publish
    per-chunk status updates."""
    chunks, labels = split_html_into_chunks(html)
    print(f"Split HTML into {len(chunks)} chunk(s) for processing")

    # The head needs no AI regeneration — it carries no visual styling to re-theme.
    # Chunks labeled "body" (or "raw", when head/body couldn't be parsed) go to the model.
    # There can be more than one leading "head" chunk if the head exceeds max_chars.
    head_chunk_count = labels.count("head")
    regen_indices = [i for i, label in enumerate(labels) if label != "head"]
    results = {}
    for i in range(head_chunk_count):
        results[i] = chunks[i]

    with ThreadPoolExecutor(max_workers=min(len(regen_indices), MAX_CONCURRENT_CHUNK_REQUESTS) or 1) as executor:
        futures = {
            executor.submit(
                _regenerate_chunk, client, chunks[i], theme_prompt, regeneration_theme, style_guide, i, len(chunks)
            ): i
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

    head_content = "".join(_strip_head_wrapper(results[i]) for i in range(head_chunk_count))
    body_parts = [results[i] for i in range(head_chunk_count, len(chunks))]

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
