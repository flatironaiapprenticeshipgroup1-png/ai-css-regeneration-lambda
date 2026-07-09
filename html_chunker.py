from bs4 import BeautifulSoup, Tag

MAX_CHARS_PER_CHUNK = 30_000


def split_node_into_parts(node, max_chars: int) -> list[str]:
    """Returns a list of HTML strings derived from node, each <= max_chars.
    Recursively descends into child nodes for oversized elements.
    Falls back to string truncation only for irreducible leaf nodes."""
    node_str = str(node)
    if len(node_str) <= max_chars:
        return [node_str]
    if isinstance(node, Tag) and node.name:
        parts: list[str] = []
        current: list[str] = []
        current_size = 0
        for child in node.children:
            for part in split_node_into_parts(child, max_chars):
                if current and current_size + len(part) > max_chars:
                    parts.append("".join(current))
                    current = [part]
                    current_size = len(part)
                else:
                    current.append(part)
                    current_size += len(part)
        if current:
            parts.append("".join(current))
        return parts if parts else [node_str[:max_chars]]
    print(f"Leaf node too large to split ({len(node_str)} chars), truncating to {max_chars}")
    return [node_str[:max_chars]]


def split_html_into_chunks(html: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> tuple[list[str], list[str]]:
    """Returns (chunks, labels) where labels are 'head' or 'body'.
    Chunk 0 is always the <head> element; subsequent chunks are groups of
    top-level <body> children packed to max_chars each.
    Falls back to a single raw chunk if the document can't be parsed."""
    soup = BeautifulSoup(html, "html.parser")
    head = soup.find("head")
    body = soup.find("body")

    if not head or not body:
        return [html], ["raw"]

    # The old CSS-file pipeline wired in a stylesheet link that no longer
    # applies now that styling is inlined per-element.
    for link in head.find_all("link", rel="stylesheet"):
        link.decompose()

    head_str = str(head)
    if len(head_str) > max_chars:
        print(f"Head element exceeds {max_chars} chars ({len(head_str)}) — truncating")
        head_str = head_str[:max_chars]

    chunks = [head_str]
    labels = ["head"]

    current_parts: list[str] = []
    current_size = 0

    for child in body.children:
        for part in split_node_into_parts(child, max_chars):
            if current_parts and current_size + len(part) > max_chars:
                chunks.append("".join(current_parts))
                labels.append("body")
                current_parts = [part]
                current_size = len(part)
            else:
                current_parts.append(part)
                current_size += len(part)

    if current_parts:
        chunks.append("".join(current_parts))
        labels.append("body")

    return chunks, labels
