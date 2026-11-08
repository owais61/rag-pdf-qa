import re


def safe_filename(raw_name):
    """Turn a user-supplied filename into a safe one, or raise ValueError.

    Strips any folder part (blocks '../' path traversal), replaces odd characters,
    and only allows .pdf files.
    """
    name = (raw_name or "").replace("\\", "/").split("/")[-1].strip()
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).lstrip(".")
    if not name.lower().endswith(".pdf") or len(name) <= len(".pdf"):
        raise ValueError("Only .pdf files are supported.")
    if len(name) > 100:
        name = name[-100:]
    return name


def collapse_whitespace(text):
    return re.sub(r"\s+", " ", text or "").strip()
