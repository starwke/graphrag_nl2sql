import re
from typing import List

from llama_index.core.node_parser import TextSplitter


class LineTextSpliter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []

        return re.split(r"\n|\\n", text, flags=re.I)
