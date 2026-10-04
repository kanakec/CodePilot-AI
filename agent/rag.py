from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ============================================================
# RAG DOCUMENT
# ============================================================

@dataclass
class DocumentChunk:
    source: str
    content: str
    chunk_id: int


# ============================================================
# LOCAL RAG VECTOR STORE
# ============================================================

class LocalVectorStore:
    """
    Lightweight local RAG retrieval system.

    Uses:
    - TF-IDF-style weighting
    - cosine similarity
    - keyword overlap
    - filename/path matching
    - simple code-aware normalization

    No external API or embedding model is required.
    """

    STOP_WORDS = {
        "the", "and", "for", "with", "this", "that", "from", "into",
        "then", "when", "where", "which", "will", "must", "should",
        "have", "has", "are", "was", "were", "been", "being", "use",
        "using", "only", "not", "can", "all", "each", "their", "your",
        "our", "its", "is", "a", "an", "to", "of", "in", "on", "as", "by",
        "how", "what", "why", "does", "do", "did", "this", "these", "those",
        "application", "project", "code", "file", "files",
    }

    # Common programming aliases.
    SYNONYMS = {
        "delete": {
            "delete",
            "deletion",
            "remove",
            "removal",
            "destroy",
            "discard",
            "erase",
        },
        "add": {
            "add",
            "create",
            "insert",
            "new",
            "append",
        },
        "update": {
            "update",
            "edit",
            "modify",
            "change",
            "alter",
        },
        "task": {
            "task",
            "todo",
            "todos",
            "item",
            "items",
        },
        "expense": {
            "expense",
            "expenses",
            "cost",
            "costs",
            "spending",
            "transaction",
            "transactions",
        },
        "search": {
            "search",
            "filter",
            "find",
            "query",
        },
        "login": {
            "login",
            "signin",
            "sign",
            "authentication",
            "auth",
        },
    }

    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.vectors: list[dict[str, float]] = []
        self.idf: dict[str, float] = {}

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """
        Convert natural language and source code into useful tokens.

        Handles:
        - camelCase
        - PascalCase
        - snake_case
        - kebab-case
        - normal English
        - common programming symbols
        """

        if not text:
            return []

        text = str(text)

        # Split camelCase / PascalCase.
        text = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1 \2",
            text,
        )

        # Split acronym + normal word.
        text = re.sub(
            r"([A-Z]+)([A-Z][a-z])",
            r"\1 \2",
            text,
        )

        # Treat programming separators as spaces.
        text = re.sub(
            r"[_\-/\\.:()[\]{}<>+=*!,;?\"'`|@#$%^&~]+",
            " ",
            text,
        )

        # Keep alphanumeric tokens.
        tokens = re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )

        result: list[str] = []

        for token in tokens:
            if len(token) <= 1:
                continue

            if token in cls.STOP_WORDS:
                continue

            result.append(token)

        return result

    @classmethod
    def expand_tokens(cls, tokens: Iterable[str]) -> set[str]:
        """
        Expand important concepts into lightweight synonym groups.
        """

        expanded: set[str] = set()

        for token in tokens:
            expanded.add(token)

            for canonical, synonyms in cls.SYNONYMS.items():
                if token in synonyms:
                    expanded.add(canonical)
                    expanded.update(synonyms)

        return expanded

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> list[str]:

        if not text.strip():
            return []

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

        chunks: list[str] = []

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(
                start + chunk_size,
                text_length,
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - overlap

        return chunks

    # --------------------------------------------------------
    # DOCUMENT ADDITION
    # --------------------------------------------------------

    def add_document(
        self,
        source: str,
        content: str,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> int:

        chunks = self.chunk_text(
            content,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        start_id = len(self.chunks)

        for offset, chunk in enumerate(chunks):
            self.chunks.append(
                DocumentChunk(
                    source=source,
                    content=chunk,
                    chunk_id=start_id + offset,
                )
            )

        self._rebuild_index()

        return len(chunks)

    def add_documents(
        self,
        documents: Iterable[tuple[str, str]],
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> int:

        documents = list(documents)

        if not documents:
            return 0

        start_index = len(self.chunks)

        for source, content in documents:

            chunks = self.chunk_text(
                content,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for chunk in chunks:
                self.chunks.append(
                    DocumentChunk(
                        source=source,
                        content=chunk,
                        chunk_id=len(self.chunks),
                    )
                )

        # Rebuild only once instead of once per document.
        if len(self.chunks) > start_index:
            self._rebuild_index()

        return len(self.chunks) - start_index

    # --------------------------------------------------------
    # TF-IDF INDEX
    # --------------------------------------------------------

    def _rebuild_index(self) -> None:

        if not self.chunks:
            self.vectors = []
            self.idf = {}
            return

        document_tokens: list[list[str]] = []

        for chunk in self.chunks:
            tokens = self.tokenize(chunk.content)
            document_tokens.append(tokens)

        document_frequency: dict[str, int] = {}

        for tokens in document_tokens:

            unique_tokens = set(tokens)

            for token in unique_tokens:
                document_frequency[token] = (
                    document_frequency.get(token, 0) + 1
                )

        total_documents = len(document_tokens)

        self.idf = {
            token: math.log(
                (1 + total_documents)
                / (1 + frequency)
            ) + 1.0
            for token, frequency in document_frequency.items()
        }

        self.vectors = []

        for tokens in document_tokens:

            term_frequency: dict[str, float] = {}

            for token in tokens:
                term_frequency[token] = (
                    term_frequency.get(token, 0.0) + 1.0
                )

            total_tokens = len(tokens)

            if total_tokens == 0:
                self.vectors.append({})
                continue

            vector: dict[str, float] = {}

            for token, count in term_frequency.items():

                if token not in self.idf:
                    continue

                tf = count / total_tokens

                vector[token] = (
                    tf * self.idf[token]
                )

            self.vectors.append(vector)

    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    @staticmethod
    def cosine_similarity(
        vector_a: dict[str, float],
        vector_b: dict[str, float],
    ) -> float:

        if not vector_a or not vector_b:
            return 0.0

        smaller, larger = (
            (vector_a, vector_b)
            if len(vector_a) <= len(vector_b)
            else (vector_b, vector_a)
        )

        dot_product = sum(
            value * larger.get(token, 0.0)
            for token, value in smaller.items()
        )

        magnitude_a = math.sqrt(
            sum(
                value * value
                for value in vector_a.values()
            )
        )

        magnitude_b = math.sqrt(
            sum(
                value * value
                for value in vector_b.values()
            )
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )

    # --------------------------------------------------------
    # QUERY VECTOR
    # --------------------------------------------------------

    def _query_vector(
        self,
        query: str,
    ) -> dict[str, float]:

        tokens = self.tokenize(query)

        if not tokens:
            return {}

        term_frequency: dict[str, float] = {}

        for token in tokens:

            term_frequency[token] = (
                term_frequency.get(token, 0.0) + 1.0
            )

        total_tokens = len(tokens)

        vector: dict[str, float] = {}

        for token, count in term_frequency.items():

            if token not in self.idf:
                continue

            tf = count / total_tokens

            vector[token] = (
                tf * self.idf[token]
            )

        return vector

    # --------------------------------------------------------
    # KEYWORD MATCHING
    # --------------------------------------------------------

    def _keyword_score(
        self,
        query: str,
        chunk: DocumentChunk,
    ) -> float:

        query_tokens = set(
            self.tokenize(query)
        )

        if not query_tokens:
            return 0.0

        content_tokens = set(
            self.tokenize(
                f"{chunk.source} {chunk.content}"
            )
        )

        expanded_query_tokens = self.expand_tokens(
            query_tokens
        )

        expanded_content_tokens = self.expand_tokens(
            content_tokens
        )

        direct_matches = (
            query_tokens & content_tokens
        )

        semantic_matches = (
            expanded_query_tokens
            & expanded_content_tokens
        )

        direct_score = (
            len(direct_matches)
            / max(len(query_tokens), 1)
        )

        semantic_score = (
            len(semantic_matches)
            / max(len(expanded_query_tokens), 1)
        )

        return (
            direct_score * 0.6
            + semantic_score * 0.4
        )

    # --------------------------------------------------------
    # SOURCE/PATH MATCHING
    # --------------------------------------------------------

    def _source_score(
        self,
        query: str,
        chunk: DocumentChunk,
    ) -> float:

        query_tokens = set(
            self.tokenize(query)
        )

        source_tokens = set(
            self.tokenize(chunk.source)
        )

        if not query_tokens or not source_tokens:
            return 0.0

        matches = query_tokens & source_tokens

        return len(matches) / len(query_tokens)

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:

        if top_k <= 0:
            return []

        if not self.chunks:
            return []

        query_vector = self._query_vector(query)

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scored_results: list[
            tuple[DocumentChunk, float]
        ] = []

        for index, document_vector in enumerate(
            self.vectors
        ):

            chunk = self.chunks[index]

            cosine_score = self.cosine_similarity(
                query_vector,
                document_vector,
            )

            keyword_score = self._keyword_score(
                query,
                chunk,
            )

            source_score = self._source_score(
                query,
                chunk,
            )

            # Combined retrieval score.
            #
            # Keyword matching is deliberately strong because
            # source-code questions often use terms such as
            # delete/remove, task/todo, login/auth, etc.
            final_score = (
                cosine_score * 0.50
                + keyword_score * 0.40
                + source_score * 0.10
            )

            if final_score >= min_score:
                scored_results.append(
                    (chunk, final_score)
                )

        scored_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_results[:top_k]

    # --------------------------------------------------------
    # CONTEXT BUILDER
    # --------------------------------------------------------

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        max_chars: int = 6000,
    ) -> str:

        results = self.search(
            query=query,
            top_k=top_k,
        )

        if not results:
            return ""

        context_parts: list[str] = []
        current_length = 0

        for chunk, score in results:

            section = (
                f"\n--- SOURCE: {chunk.source} "
                f"(relevance: {score:.3f}) ---\n"
                f"{chunk.content}\n"
            )

            if (
                current_length + len(section)
                > max_chars
            ):

                remaining = (
                    max_chars - current_length
                )

                if remaining > 100:
                    context_parts.append(
                        section[:remaining]
                    )

                break

            context_parts.append(section)
            current_length += len(section)

        return "".join(
            context_parts
        ).strip()

    # --------------------------------------------------------
    # STORE MANAGEMENT
    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()
        self.vectors.clear()
        self.idf.clear()


# ============================================================
# PROJECT CODEBASE INDEXER
# ============================================================

class ProjectIndexer:
    """
    Reads a generated project and indexes useful source files.
    """

    DEFAULT_EXTENSIONS = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".html",
        ".css",
        ".json",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".sql",
    }

    DEFAULT_IGNORED_DIRECTORIES = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
    }

    def __init__(
        self,
        root_path: str | Path,
        vector_store: LocalVectorStore | None = None,
    ) -> None:

        self.root_path = Path(root_path)

        self.vector_store = (
            vector_store
            if vector_store is not None
            else LocalVectorStore()
        )

    # --------------------------------------------------------
    # INDEX PROJECT
    # --------------------------------------------------------

    def index_project(self) -> int:

        if not self.root_path.exists():
            raise FileNotFoundError(
                "Project directory does not exist: "
                f"{self.root_path}"
            )

        if not self.root_path.is_dir():
            raise NotADirectoryError(
                "Project path is not a directory: "
                f"{self.root_path}"
            )

        documents: list[
            tuple[str, str]
        ] = []

        for path in self.root_path.rglob("*"):

            if not path.is_file():
                continue

            if self._should_ignore(path):
                continue

            if (
                path.suffix.lower()
                not in self.DEFAULT_EXTENSIONS
            ):
                continue

            try:
                content = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except OSError:
                continue

            if not content.strip():
                continue

            relative_path = str(
                path.relative_to(
                    self.root_path
                )
            )

            documents.append(
                (
                    relative_path,
                    content,
                )
            )

        self.vector_store.clear()

        return self.vector_store.add_documents(
            documents
        )

    # --------------------------------------------------------
    # IGNORE RULES
    # --------------------------------------------------------

    def _should_ignore(
        self,
        path: Path,
    ) -> bool:

        try:
            relative_parts = (
                path.relative_to(
                    self.root_path
                ).parts
            )
        except ValueError:
            return True

        return any(
            part in self.DEFAULT_IGNORED_DIRECTORIES
            for part in relative_parts
        )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:

        return self.vector_store.search(
            query=query,
            top_k=top_k,
        )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        max_chars: int = 6000,
    ) -> str:

        return self.vector_store.build_context(
            query=query,
            top_k=top_k,
            max_chars=max_chars,
        )


# ============================================================
# RAG FACTORY
# ============================================================

def build_project_rag(
    project_path: str | Path,
) -> ProjectIndexer:

    indexer = ProjectIndexer(
        project_path
    )

    indexer.index_project()

    return indexer


# ============================================================
# TEST / CLI
# ============================================================

if __name__ == "__main__":

    project_path = Path(
        "generated_project"
    )

    print("=" * 60)
    print("CodePilot-Ai Local RAG Test")
    print("=" * 60)

    if not project_path.exists():

        print(
            f"\nProject directory not found: "
            f"{project_path}"
        )

        print(
            "\nGenerate a project first, "
            "then run this test again."
        )

        raise SystemExit(1)

    try:

        indexer = build_project_rag(
            project_path
        )

    except Exception as error:

        print(
            "\nRAG indexing failed:"
        )

        print(error)

        raise SystemExit(1)

    print(
        f"\nIndexed chunks: "
        f"{len(indexer.vector_store)}"
    )

    query = input(
        "\nEnter a question about the project: "
    ).strip()

    if not query:

        print(
            "\nNo query provided."
        )

        raise SystemExit(0)

    results = indexer.search(
        query=query,
        top_k=5,
    )

    if not results:

        print(
            "\nNo relevant project information found."
        )

        raise SystemExit(0)

    print(
        "\nRetrieved context:"
    )

    print("-" * 60)

    for chunk, score in results:

        print(
            f"\n[{score:.3f}] "
            f"{chunk.source}"
        )

        print(
            chunk.content[:1200]
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "RAG retrieval test completed."
    )

    print("=" * 60)