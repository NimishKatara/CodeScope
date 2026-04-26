"""
api/routes.py
REST API route handlers. All routes are protected by token-based authentication.
"""

from flask import Flask, request, jsonify
from auth.auth_manager import AuthManager
from db.database import Database
from api.data_pipeline import DataPipeline


app = Flask(__name__)
db = Database()
auth = AuthManager(db)
pipeline = DataPipeline(db)


def _get_token() -> str:
    """Extract Bearer token from Authorization header."""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:]
    return ""


@app.route("/api/login", methods=["POST"])
def login():
    body = request.json or {}
    token = auth.login(body.get("username", ""), body.get("password", ""))
    if token is None:
        return jsonify({"error": "Invalid credentials"}), 401
    return jsonify({"token": token})


@app.route("/api/logout", methods=["POST"])
def logout():
    token = _get_token()
    auth.logout(token)
    return jsonify({"status": "logged out"})


@app.route("/api/data", methods=["GET"])
def get_data():
    """Fetch processed data records. Requires viewer role."""
    token = _get_token()
    if not auth.require_role(token, "viewer"):
        return jsonify({"error": "Unauthorized"}), 403

    records = pipeline.fetch_and_process()
    return jsonify({"data": records, "count": len(records)})


@app.route("/api/data", methods=["POST"])
def create_data():
    """Create a new data record. Requires editor role."""
    token = _get_token()
    if not auth.require_role(token, "editor"):
        return jsonify({"error": "Forbidden"}), 403

    payload = request.json or {}
    record_id = pipeline.ingest(payload)
    return jsonify({"id": record_id}), 201


@app.route("/api/admin/users", methods=["GET"])
def list_users():
    """List all users. Requires admin role."""
    token = _get_token()
    if not auth.require_role(token, "admin"):
        return jsonify({"error": "Admin only"}), 403
    users = db.get_all_users()
    return jsonify({"users": users})