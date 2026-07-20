#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Leaderboard Viewer - A web application to visualize the agent optimization leaderboard.

Run with: python3 serve.py [--workspace PATH]
Then open: http://localhost:8000

Options:
    --workspace, -w PATH    Path to the workspace folder containing leaderboard.json
                            (default: ./workspaces)
    --host HOST             Address to bind (default: 127.0.0.1; use 0.0.0.0 for LAN)
    --port, -p PORT         Port to run the server on (default: 8000)
    --user USER             HTTP Basic Auth username (required with --password for LAN)
    --password PASS         HTTP Basic Auth password (or set LEADERBOARD_PASSWORD)

No external dependencies required - uses Python's built-in http.server.

/api/data omits candidate source code (small payload); the UI loads code on demand
from /api/candidate_code (reads workspace/<id>/solution.py when present, else JSON).
Leaderboard JSON is re-read only when leaderboard.json changes (mtime cache).
"""

import argparse
import base64
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import math
import os
from pathlib import Path
import secrets
import sys
from typing import Optional
from urllib.parse import parse_qs, urlparse


WEB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_ROOT.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workspace_fs import file_exists_under_root, read_text_under_root

LEADERBOARD_PATH = None
WORKSPACE_ROOT = None

CONTENT_TYPES = {
    "html": "text/html; charset=utf-8",
    "js": "application/javascript",
    "css": "text/css",
}

_lb_mtime = None
_lb_raw = None
_lb_api_json_bytes = None


def _is_safe_workspace_id(workspace_id: str) -> bool:
    """Reject path traversal; workspace dirs are single segment under WORKSPACE_ROOT."""
    if not workspace_id or not isinstance(workspace_id, str):
        return False
    if workspace_id != os.path.normpath(workspace_id):
        return False
    sep = os.path.sep
    if sep in workspace_id:
        return False
    alt = os.path.altsep
    if alt and alt in workspace_id:
        return False
    if workspace_id in (".", "..") or workspace_id.startswith(".."):
        return False
    return True


def _refresh_leaderboard_cache():
    """Load leaderboard.json if needed; build slim API JSON bytes (no embedded code)."""
    global _lb_mtime, _lb_raw, _lb_api_json_bytes
    mtime = os.path.getmtime(LEADERBOARD_PATH)
    if _lb_mtime == mtime and _lb_raw is not None and _lb_api_json_bytes is not None:
        return
    with open(LEADERBOARD_PATH, "r", encoding="utf-8") as f:
        _lb_raw = json.load(f)
    api_light = build_api_data(_lb_raw, include_candidate_code=False)
    _lb_api_json_bytes = json.dumps(api_light).encode("utf-8")
    _lb_mtime = mtime


def get_leaderboard_raw():
    """Return parsed leaderboard JSON; uses file mtime cache."""
    _refresh_leaderboard_cache()
    return _lb_raw


def load_leaderboard():
    """Load the leaderboard JSON file (via mtime cache)."""
    return get_leaderboard_raw()


def code_from_disk(workspace_id: str) -> Optional[str]:
    """Read solution.py from WORKSPACE_ROOT/workspace_id if present. Returns None if missing."""
    if not WORKSPACE_ROOT or not _is_safe_workspace_id(workspace_id):
        return None
    ws_path = Path(WORKSPACE_ROOT) / workspace_id
    if not file_exists_under_root(ws_path, "solution.py"):
        return None
    try:
        return read_text_under_root(ws_path, "solution.py", errors="replace")
    except OSError:
        return None


def code_from_leaderboard(workspace_id: str) -> Optional[str]:
    """Find candidate code in cached raw leaderboard. None if workspace_id not found."""
    data = get_leaderboard_raw()
    for cluster_candidates in data.get("clusters", {}).values():
        for candidate in cluster_candidates:
            if candidate.get("workspace_id") == workspace_id:
                c = candidate.get("code", "")
                return "" if c is None else str(c)
    return None


def sanitize_for_json(obj):
    """Recursively convert float('inf') and float('-inf') to strings for JSON."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def _compute_pareto_front(points, higher_is_better):
    """Return Pareto-front indices from a list of (metric, complexity) pairs."""
    indexed = list(enumerate(points))
    indexed.sort(key=lambda t: -t[1][0] if higher_is_better else t[1][0])
    front_indices = []
    best_c = float("inf")
    for idx, (m, c) in indexed:
        if c < best_c:
            front_indices.append(idx)
            best_c = c
    return front_indices


def build_api_data(data, include_candidate_code: bool = True):
    """Build all computed data for the API response.

    When include_candidate_code is False, the ``code`` field is omitted from
    each candidate (smaller JSON for the browser; use /api/candidate_code).
    """
    higher_is_better = data.get("higher_is_better", False)

    candidates = {}
    for cluster_name, cluster_candidates in data.get("clusters", {}).items():
        for candidate in cluster_candidates:
            if include_candidate_code:
                entry = {**candidate, "cluster": cluster_name}
            else:
                entry = {
                    k: v for k, v in candidate.items() if k != "code"
                }
                entry["cluster"] = cluster_name
            candidates[candidate["workspace_id"]] = entry

    # Build scatter points from hp_results
    scatter_points = []
    for wid, cand in candidates.items():
        for hp in cand.get("hp_results", []):
            scatter_points.append({
                "x": hp["metric"],
                "y": hp["complexity"],
                "workspace_id": wid,
                "hp_index": hp["hp_index"],
                "params": hp.get("params", {}),
                "cluster": cand.get("cluster", ""),
                "generation": cand.get("generation", 0),
            })

    # Compute Pareto front over scatter points
    pareto_indices = []
    if scatter_points:
        pts = [(p["x"], p["y"]) for p in scatter_points]
        pareto_indices = _compute_pareto_front(pts, higher_is_better)
    pareto_points = [scatter_points[i] for i in pareto_indices]

    # Per-generation bests (within that generation only).
    from collections import defaultdict
    gen_data = defaultdict(lambda: {"total": 0, "successful": 0, "best_metric": None, "best_complexity": None})
    for cand in candidates.values():
        gen = cand.get("generation", 0)
        gen_data[gen]["total"] += 1
        if cand.get("success", False) and cand.get("hp_results"):
            gen_data[gen]["successful"] += 1
            for hp in cand["hp_results"]:
                m = hp.get("metric")
                if m is None:
                    continue
                cur = gen_data[gen]["best_metric"]
                is_better = (
                    cur is None
                    or (higher_is_better and m > cur)
                    or (not higher_is_better and m < cur)
                )
                if is_better:
                    gen_data[gen]["best_metric"] = m
                    gen_data[gen]["best_complexity"] = hp.get("complexity")

    # Evolution chart: cumulative best over generations 0..n (metric + paired complexity).
    generation_stats = []
    cumulative_best_metric = None
    cumulative_best_complexity = None
    for gen in sorted(gen_data.keys()):
        stats = gen_data[gen]
        gen_best = stats["best_metric"]
        if gen_best is not None:
            if cumulative_best_metric is None:
                cumulative_best_metric = gen_best
                cumulative_best_complexity = stats["best_complexity"]
            elif (higher_is_better and gen_best > cumulative_best_metric) or (
                not higher_is_better and gen_best < cumulative_best_metric
            ):
                cumulative_best_metric = gen_best
                cumulative_best_complexity = stats["best_complexity"]
        generation_stats.append({
            "generation": gen,
            "total": stats["total"],
            "successful": stats["successful"],
            "failed": stats["total"] - stats["successful"],
            "best_metric": cumulative_best_metric,
            "best_complexity": cumulative_best_complexity,
        })

    # Best metric overall
    best_metric = None
    for p in scatter_points:
        m = p["x"]
        if best_metric is None:
            best_metric = m
        elif higher_is_better and m > best_metric:
            best_metric = m
        elif not higher_is_better and m < best_metric:
            best_metric = m

    return sanitize_for_json({
        "higher_is_better": higher_is_better,
        "total_candidates": data.get("total_candidates", 0),
        "successful_candidates": data.get("successful_candidates", 0),
        "num_clusters": data.get("num_clusters", 0),
        "last_updated": data.get("last_updated", "N/A"),
        "query": data.get("query", ""),
        "cluster_descriptions": data.get("cluster_descriptions", {}),
        "best_metric": best_metric,
        "candidates": candidates,
        "generation_stats": generation_stats,
        "scatter_points": scatter_points,
        "pareto_points": pareto_points,
    })


def list_journal_agents():
    """Scan workspace for gen*-* directories containing journal.log files."""
    import re
    workspace_dir = os.path.dirname(LEADERBOARD_PATH)
    agents = []
    pattern = re.compile(r'^gen(\d+)-(\d+)$')
    try:
        for entry in os.listdir(workspace_dir):
            m = pattern.match(entry)
            if m and file_exists_under_root(Path(workspace_dir) / entry, "journal.log"):
                agents.append({
                    "id": entry,
                    "generation": int(m.group(1)),
                    "agent_num": int(m.group(2)),
                })
    except OSError:
        pass
    agents.sort(key=lambda a: (a["generation"], a["agent_num"]))
    return agents


def read_journal(workspace_id):
    """Read journal.log for a given workspace_id, returning the content."""
    import re
    if not re.match(r'^gen\d+-\d+$', workspace_id):
        return None
    workspace_dir = os.path.dirname(LEADERBOARD_PATH)
    ws_path = Path(workspace_dir) / workspace_id
    if not file_exists_under_root(ws_path, "journal.log"):
        return None
    try:
        return read_text_under_root(ws_path, "journal.log", errors="replace")
    except OSError:
        return None


def _is_public_bind(host: str) -> bool:
    """True when the server listens on a non-loopback interface."""
    return host in ("0.0.0.0", "::", "")


class LeaderboardHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for serving the leaderboard."""

    auth_user: str = ""
    auth_password: str = ""
    auth_required: bool = False

    def _authorized(self) -> bool:
        if not self.auth_required:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
            user, sep, password = decoded.partition(":")
            if not sep:
                return False
        except (ValueError, UnicodeDecodeError):
            return False
        return (
            secrets.compare_digest(user, self.auth_user)
            and secrets.compare_digest(password, self.auth_password)
        )

    def _send_unauthorized(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Leaderboard"')
        self.end_headers()

    def do_GET(self):
        if not self._authorized():
            return self._send_unauthorized()

        parsed = urlparse(self.path)
        path_only = parsed.path

        if path_only == "/" or path_only in ("/index.html", "/leaderboard.js", "/leaderboard.css"):
            rel = "index.html" if path_only == "/" else path_only.lstrip("/")

            try:
                content = (WEB_ROOT / rel).read_text()
                self.send_response(200)
                self.send_header("Content-type", CONTENT_TYPES[rel.split(".")[-1]])
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Error: {str(e)}".encode("utf-8"))

        elif path_only == "/api/data":
            try:
                _refresh_leaderboard_cache()
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(_lb_api_json_bytes)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif path_only == "/api/candidate_code":
            try:
                qs = parse_qs(parsed.query)
                wid_list = qs.get("workspace_id") or qs.get("wid")
                workspace_id = wid_list[0] if wid_list else ""
                if not _is_safe_workspace_id(workspace_id):
                    self.send_response(400)
                    self.send_header("Content-type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"error": "invalid or missing workspace_id"}).encode("utf-8")
                    )
                    return
                text = code_from_disk(workspace_id)
                source = "disk"
                if text is None:
                    text = code_from_leaderboard(workspace_id)
                    source = "leaderboard"
                if text is None:
                    self.send_response(404)
                    self.send_header("Content-type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"error": "unknown workspace_id", "workspace_id": workspace_id}
                        ).encode("utf-8")
                    )
                    return
                payload = json.dumps(
                    {"workspace_id": workspace_id, "code": text, "source": source}
                )
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(payload.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif path_only == "/api/journals":
            try:
                agents = list_journal_agents()
                payload = json.dumps(agents)
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(payload.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif path_only == "/api/journal":
            try:
                params = parse_qs(parsed.query)
                workspace_id = params.get("id", [None])[0]
                if not workspace_id:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing 'id' parameter"}).encode("utf-8"))
                    return
                content = read_journal(workspace_id)
                if content is None:
                    self.send_response(404)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Journal not found"}).encode("utf-8"))
                    return
                payload = json.dumps({"id": workspace_id, "content": content})
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(payload.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        else:
            # Do not serve files other than the ones explicitly supported above.
            return self.send_error(404)

    def log_message(self, format, *args):
        pass


def make_handler(auth_user: str, auth_password: str, auth_required: bool):
    """Return a handler class configured with optional HTTP Basic Auth."""

    class ConfiguredHandler(LeaderboardHandler):
        pass

    ConfiguredHandler.auth_user = auth_user
    ConfiguredHandler.auth_password = auth_password
    ConfiguredHandler.auth_required = auth_required
    return ConfiguredHandler


def main():
    global LEADERBOARD_PATH, WORKSPACE_ROOT

    parser = argparse.ArgumentParser(
        description="Leaderboard Viewer - Visualize agent optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workspace", "-w", type=str,
        default=PROJECT_ROOT / "workspaces",
        help="Path to the workspace folder containing leaderboard.json (default: ./workspaces)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Address to bind (default: 127.0.0.1; use 0.0.0.0 to listen on all interfaces)",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--user", type=str, default=os.environ.get("LEADERBOARD_USER", ""),
        help="HTTP Basic Auth username (or set LEADERBOARD_USER)",
    )
    parser.add_argument(
        "--password", type=str, default=os.environ.get("LEADERBOARD_PASSWORD", ""),
        help="HTTP Basic Auth password (or set LEADERBOARD_PASSWORD)",
    )
    args = parser.parse_args()

    if _is_public_bind(args.host) and not (args.user and args.password):
        parser.error(
            "Network exposure (--host 0.0.0.0) requires --user and --password "
            "(or LEADERBOARD_USER / LEADERBOARD_PASSWORD)"
        )
    if bool(args.user) != bool(args.password):
        parser.error("--user and --password must be provided together")

    auth_required = bool(args.user and args.password)
    handler_cls = make_handler(args.user, args.password, auth_required)

    workspace_path = os.path.abspath(args.workspace)
    WORKSPACE_ROOT = workspace_path
    LEADERBOARD_PATH = os.path.join(workspace_path, "leaderboard.json")

    print(f"Loading leaderboard from: {LEADERBOARD_PATH}")

    if not os.path.exists(LEADERBOARD_PATH):
        print(f"Error: Leaderboard file not found at {LEADERBOARD_PATH}")
        return

    data = load_leaderboard()
    print(f"  - Total candidates: {data.get('total_candidates', 0)}")
    print(f"  - Successful: {data.get('successful_candidates', 0)}")
    print(f"  - Ideas: {data.get('num_clusters', 0)}")
    print()
    print(f"Starting server at http://{args.host}:{args.port}")
    if auth_required:
        print(f"HTTP Basic Auth enabled (user: {args.user})")
    print("Press Ctrl+C to stop")

    httpd = HTTPServer((args.host, args.port), handler_cls)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    main()
