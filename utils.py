def trim(text: str) -> str:
    if not text or type(text) != str:
        return ""

    return text.strip()
