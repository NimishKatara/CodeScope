"""
Flask API server for the Codebase Understanding Agent dashboard.
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import CodebaseAgent

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": ["Content-Type"]}})

# Global agent instance
_agent: CodebaseAgent = None
# Navigate up from backend/codebaseagent/ to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
SAMPLE_REPO = str(PROJECT_ROOT / "sample_repo")


def get_agent() -> CodebaseAgent:
    global _agent
    if _agent is None:
        try:
            _agent = CodebaseAgent(SAMPLE_REPO)
            _agent.index()
        except Exception as e:
            print(f"[Error] Failed to initialize agent: {e}")
            raise
    return _agent


@app.route("/api/index", methods=["POST"])
def index_repo():
    data = request.json or {}
    repo_path = data.get("repo_path", SAMPLE_REPO)
    global _agent
    _agent = CodebaseAgent(repo_path)
    stats = _agent.index()
    return jsonify({"status": "ok", "stats": stats})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    agent = get_agent()
    return jsonify(agent.get_stats())


@app.route("/api/query", methods=["POST"])
def query():
    data = request.json or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        agent = get_agent()
        result = agent.query(question)
        baseline = agent.keyword_search_baseline(question)
        result["baseline"] = baseline
        return jsonify(result)
    except Exception as e:
        print(f"[Error] Query failed: {e}")
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


@app.route("/api/chunks", methods=["GET"])
def list_chunks():
    agent = get_agent()
    chunks_data = [
        {
            "id": c.chunk_id,
            "name": c.name,
            "type": c.chunk_type,
            "file": str(Path(c.file_path).relative_to(agent.repo_path)),
            "lines": f"L{c.start_line}-{c.end_line}",
            "keywords": c.keywords[:8],
        }
        for c in agent.chunks
    ]
    return jsonify({"chunks": chunks_data})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("[Server] Starting Codebase Agent API on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
