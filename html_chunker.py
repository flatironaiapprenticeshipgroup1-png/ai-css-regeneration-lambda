from bs4 import BeautifulSoup, Tag

MAX_CHARS_PER_CHUNK = 30_000


def _tag_open_close(node: Tag) -> tuple[str, str]:
    """Returns (opening_tag_str, closing_tag_str) for node's own tag, e.g.
    ('<div class="a">', '</div>'), without serializing its children."""
    wrapper = BeautifulSoup("", "html.parser").new_tag(node.name, attrs=node.attrs)
    wrapper_str = str(wrapper)
    close = f"</{node.name}>"
    open_str = wrapper_str[: -len(close)] if wrapper_str.endswith(close) else wrapper_str
    return open_str, close


def split_node_into_parts(node, max_chars: int) -> list[str]:
    """Returns a list of HTML strings derived from node, each <= max_chars.
    Recursively descends into child nodes for oversized elements, re-wrapping
    each resulting group in the element's own opening/closing tag so structure
    (e.g. a <script> or a styled <div>) survives the split.
    Falls back to string truncation only for irreducible leaf nodes."""
    node_str = str(node)
    if len(node_str) <= max_chars:
        return [node_str]
    if isinstance(node, Tag) and node.name:
        open_tag, close_tag = _tag_open_close(node)
        child_budget = max(max_chars - len(open_tag) - len(close_tag), 1)

        parts: list[str] = []
        current: list[str] = []
        current_size = 0
        for child in node.children:
            for part in split_node_into_parts(child, child_budget):
                if current and current_size + len(part) > child_budget:
                    parts.append(open_tag + "".join(current) + close_tag)
                    current = [part]
                    current_size = len(part)
                else:
                    current.append(part)
                    current_size += len(part)
        if current:
            parts.append(open_tag + "".join(current) + close_tag)
        return parts if parts else [node_str[:max_chars]]
    print(f"Leaf node too large to split ({len(node_str)} chars), truncating to {max_chars}")
    return [node_str[:max_chars]]


def _pack_top_level_children(children, max_chars: int) -> list[str]:
    """Packs a sequence of sibling nodes into groups of HTML strings, each
    <= max_chars where possible (subject to the same irreducible-leaf
    exception as split_node_into_parts)."""
    packed: list[str] = []
    current: list[str] = []
    current_size = 0
    for child in children:
        for part in split_node_into_parts(child, max_chars):
            if current and current_size + len(part) > max_chars:
                packed.append("".join(current))
                current = [part]
                current_size = len(part)
            else:
                current.append(part)
                current_size += len(part)
    if current:
        packed.append("".join(current))
    return packed


def split_html_into_chunks(html: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> tuple[list[str], list[str]]:
    """Returns (chunks, labels) where labels are 'head'/'body', or 'raw' for
    every chunk when the document can't be parsed into head/body.
    Chunk 0 is the <head> element when one is found; subsequent chunks are
    groups of top-level children (of <body>, or of the whole document when
    there's no head/body) packed to max_chars each."""
    soup = BeautifulSoup(html, "html.parser")

    # The old CSS-file pipeline wired in a stylesheet link that no longer
    # applies now that styling is inlined per-element. Strip it regardless of
    # whether the document parses into a clean head/body shape.
    for link in soup.find_all("link", rel="stylesheet"):
        link.decompose()

    head = soup.find("head")
    body = soup.find("body")

    if not head or not body:
        raw_chunks = _pack_top_level_children(soup.children, max_chars)
        if not raw_chunks:
            raw_chunks = [html]
        return raw_chunks, ["raw"] * len(raw_chunks)

    head_str = str(head)
    if len(head_str) > max_chars:
        print(f"Head element exceeds {max_chars} chars ({len(head_str)}) — truncating")
        head_str = head_str[:max_chars]

    body_chunks = _pack_top_level_children(body.children, max_chars)
    return [head_str] + body_chunks, ["head"] + ["body"] * len(body_chunks)
