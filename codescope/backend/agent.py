"""
Codebase Understanding Agent
Core agent logic: AST parsing, chunking, embedding, retrieval, and LLM answering.
"""

import os
import ast
import json
import hashlib
import textwrap
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from openai import OpenAI

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    chunk_type: str          # "function" | "class" | "module" | "import_block"
    name: str
    start_line: int
    end_line: int
    source_code: str
    docstring: str
    imports: list[str]
    called_functions: list[str]
    # Simple TF-style keyword vector (no external embeddings needed for demo)
    keywords: list[str]

    def to_dict(self):
        return asdict(self)


# ── AST Parser ───────────────────────────────────────────────────────────────

class ASTParser:
    """Parses Python files and extracts structural chunks."""

    SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}

    def parse_file(self, filepath: str) -> list[CodeChunk]:
        chunks = []
        try:
            source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except SyntaxError:
            return []

        lines = source.splitlines()

        # Module-level imports
        imports = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]

        # Functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                end_line = node.end_lineno if hasattr(node, "end_lineno") else node.lineno
                snippet = "\n".join(lines[node.lineno - 1 : end_line])
                docstring = ast.get_docstring(node) or ""
                called = self._extract_calls(node)
                keywords = self._extract_keywords(node.name, docstring, snippet)

                chunk = CodeChunk(
                    chunk_id=hashlib.md5(f"{filepath}:{node.name}:{node.lineno}".encode()).hexdigest()[:12],
                    file_path=filepath,
                    chunk_type="class" if isinstance(node, ast.ClassDef) else "function",
                    name=node.name,
                    start_line=node.lineno,
                    end_line=end_line,
                    source_code=snippet[:2000],
                    docstring=docstring,
                    imports=imports[:10],
                    called_functions=called[:15],
                    keywords=keywords,
                )
                chunks.append(chunk)

        # If no functions/classes, add a module chunk
        if not chunks:
            snippet = source[:1500]
            chunk = CodeChunk(
                chunk_id=hashlib.md5(filepath.encode()).hexdigest()[:12],
                file_path=filepath,
                chunk_type="module",
                name=Path(filepath).stem,
                start_line=1,
                end_line=len(lines),
                source_code=snippet,
                docstring="",
                imports=imports[:10],
                called_functions=[],
                keywords=self._extract_keywords(Path(filepath).stem, "", snippet),
            )
            chunks.append(chunk)

        return chunks

    def parse_repository(self, repo_path: str) -> list[CodeChunk]:
        all_chunks = []
        repo = Path(repo_path)
        for py_file in repo.rglob("*.py"):
            if any(skip in py_file.parts for skip in self.SKIP_DIRS):
                continue
            all_chunks.extend(self.parse_file(str(py_file)))
        return all_chunks

    def _extract_calls(self, node) -> list[str]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(set(calls))

    def _extract_keywords(self, name: str, doc: str, code: str) -> list[str]:
        text = f"{name} {doc} {code}".lower()
        stopwords = {"self", "cls", "return", "def", "class", "import", "from", "if",
                     "else", "for", "while", "in", "not", "and", "or", "true", "false",
                     "none", "pass", "with", "as", "try", "except", "raise", "yield"}
        words = [w.strip("()[]{}:,.'\"") for w in text.split()]
        keywords = [w for w in words if len(w) > 3 and w.isalpha() and w not in stopwords]
        # Frequency-ranked deduplication
        freq = {}
        for w in keywords:
            freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=lambda x: -freq[x])[:20]


# ── Retriever ─────────────────────────────────────────────────────────────────

class StructuralRetriever:
    """BM25-style keyword retrieval over code chunks."""

    def __init__(self, chunks: list[CodeChunk]):
        self.chunks = chunks

    def query(self, question: str, top_k: int = 5) -> list[CodeChunk]:
        q_words = set(question.lower().split())
        scored = []
        for chunk in self.chunks:
            chunk_words = set(chunk.keywords) | {chunk.name.lower(), chunk.chunk_type}
            # Add file path terms
            for part in Path(chunk.file_path).parts:
                chunk_words.update(part.lower().replace(".py", "").split("_"))
            overlap = len(q_words & chunk_words)
            # Boost exact name matches
            if any(w in chunk.name.lower() for w in q_words):
                overlap += 3
            # Boost docstring matches
            if chunk.docstring and any(w in chunk.docstring.lower() for w in q_words):
                overlap += 2
            if overlap > 0:
                scored.append((overlap, chunk))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]


# ── LLM Agent ─────────────────────────────────────────────────────────────────

class CodebaseAgent:
    """Main agent: retrieves relevant chunks and answers with Groq LLM."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.parser = ASTParser()
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        self.chunks: list[CodeChunk] = []
        self.retriever: Optional[StructuralRetriever] = None
        self._index_stats = {}

    def index(self) -> dict:
        print(f"[Agent] Indexing repository: {self.repo_path}")
        self.chunks = self.parser.parse_repository(self.repo_path)
        self.retriever = StructuralRetriever(self.chunks)

        files = list({c.file_path for c in self.chunks})
        func_count = sum(1 for c in self.chunks if c.chunk_type == "function")
        class_count = sum(1 for c in self.chunks if c.chunk_type == "class")

        self._index_stats = {
            "total_chunks": len(self.chunks),
            "total_files": len(files),
            "functions": func_count,
            "classes": class_count,
            "files": [str(Path(f).relative_to(self.repo_path)) for f in files],
        }
        print(f"[Agent] Indexed {len(self.chunks)} chunks from {len(files)} files.")
        return self._index_stats

    def query(self, question: str) -> dict:
        if not self.retriever:
            raise RuntimeError("Call .index() before .query()")

        retrieved = self.retriever.query(question, top_k=5)

        # Build context for LLM
        context_parts = []
        for i, chunk in enumerate(retrieved, 1):
            rel_path = str(Path(chunk.file_path).relative_to(self.repo_path))
            context_parts.append(
                f"### Chunk {i}: `{chunk.name}` ({chunk.chunk_type}) — {rel_path} L{chunk.start_line}-{chunk.end_line}\n"
                f"```python\n{chunk.source_code[:800]}\n```"
            )
        context = "\n\n".join(context_parts)

        system_prompt = textwrap.dedent("""
            You are a Codebase Understanding Agent. You help developers understand large codebases.
            You are given retrieved code chunks from a repository and must answer the developer's question.
            
            Rules:
            - Be precise and reference specific file paths and function names.
            - Explain HOW things work, not just WHERE they are.
            - If data flows through multiple files/functions, trace the flow step by step.
            - If the retrieved context is insufficient, say so honestly.
            - Format your answer in clear Markdown with headings and code references.
        """).strip()

        user_prompt = f"**Developer Question:** {question}\n\n**Retrieved Code Context:**\n\n{context}"

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response.choices[0].message.content

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "name": c.name,
                    "type": c.chunk_type,
                    "file": str(Path(c.file_path).relative_to(self.repo_path)),
                    "lines": f"L{c.start_line}-{c.end_line}",
                    "preview": c.source_code[:200],
                }
                for c in retrieved
            ],
            "model_used": "llama-3.3-70b-versatile",
        }

    def get_stats(self) -> dict:
        return self._index_stats

    def keyword_search_baseline(self, question: str) -> dict:
        """Naive keyword search baseline for comparison."""
        words = question.lower().split()
        results = []
        for chunk in self.chunks:
            hits = sum(1 for w in words if w in chunk.source_code.lower())
            if hits:
                results.append((hits, chunk))
        results.sort(key=lambda x: -x[0])
        return {
            "method": "keyword_search",
            "results": [
                {"name": c.name, "file": str(Path(c.file_path).relative_to(self.repo_path)), "hits": h}
                for h, c in results[:5]
            ],
        }
