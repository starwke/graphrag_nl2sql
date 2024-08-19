import os
from typing import Any, List, Tuple

from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent
from pyparsing import line
from tqdm import tqdm

import utils


class CustomizeTripletExtractor(TransformComponent):
    directory: str
    rel_column: str
    rel_foreign_key: str

    def __init__(self, directory: str = "data", rel_column: str = "", rel_foreign_key: str = "") -> None:
        directory = utils.trim(directory)

        rel_column = utils.trim(rel_column) or "include"
        rel_foreign_key = utils.trim(rel_foreign_key) or "foreign_key"

        super().__init__(directory=directory, rel_column=rel_column, rel_foreign_key=rel_foreign_key)

    def __extract_columns(self, text: str) -> List[Tuple[str, str, str]]:
        if not self.directory:
            raise ValueError(f"directory cannot be empty for columns")

        table = utils.trim(text)
        column_file = os.path.join(self.directory, "columns", f"{table}.csv")
        if not os.path.exists(column_file) or not os.path.isfile(column_file):
            return []

        relationships = list()

        # extract relationships between table and column
        with open(column_file, "r", encoding="utf8") as f:
            lines = f.readlines()

            # skip table header
            for i in range(1, len(lines)):
                column = utils.trim(lines[i])
                if not column or column.startswith("#") or column.startswith(";"):
                    continue

                relationships.append((table, self.rel_column, self.__generate_name(table, column)))

        return relationships

    def __extract_foreign_keys(self, text: str) -> List[Tuple[str, str, str]]:
        words = utils.trim(text).split(",", maxsplit=3)
        if len(words) != 4:
            return []

        # forigen key relationship
        t1, t2 = utils.trim(words[0]), utils.trim(words[2])
        c1, c2 = utils.trim(words[1]), utils.trim(words[3])

        if not t1 or not t2 or not c1 or not c2:
            return []

        subj = self.__generate_name(t1, c1)
        obj = self.__generate_name(t2, c2)
        return [(subj, self.rel_foreign_key, obj)]

    def __extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        content = utils.trim(text)

        if "," in content:
            return self.__extract_foreign_keys(content)

        return self.__extract_columns(content)

    def __generate_name(self, table: str, column: str, delimiter: str = "::") -> str:
        return column if not table else f"{table}{delimiter}{column}"

    @classmethod
    def class_name(cls) -> str:
        return "CustomizeTripletExtractor"

    def __call__(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        """Extract triples from a node."""
        if show_progress:
            nodes = tqdm(nodes, desc="Extracting triplets")

        result = list()
        for node in nodes:
            # extract triplets with table name
            triples = self.__extract_triplets(node.get_content(metadata_mode=MetadataMode.NONE))

            existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
            existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

            metadata = node.metadata.copy()
            for subj, rel, obj in triples:
                subj_node = EntityNode(name=subj, properties=metadata, label="table")
                obj_node = EntityNode(name=obj, properties=metadata, label="column")
                rel_node = Relation(
                    label=rel,
                    source_id=subj_node.id,
                    target_id=obj_node.id,
                    properties=metadata,
                )

                existing_nodes.extend([subj_node, obj_node])
                existing_relations.append(rel_node)

            node.metadata[KG_NODES_KEY] = existing_nodes
            node.metadata[KG_RELATIONS_KEY] = existing_relations

            result.append(node)

        return result
