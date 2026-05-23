"""
VectorDB — Full Ollama Backend
Embeddings : ollama nomic-embed-text  (local, free)
Generation : ollama llama3.2          (local, free)
Server     : Flask port 5000
Run        : python server_final.py
"""

import math, time, random, threading, os
import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── OLLAMA CONFIG ─────────────────────────────────────────────
OLLAMA_HOST   = "http://127.0.0.1:11434"
EMBED_MODEL   = "nomic-embed-text"      # ollama pull nomic-embed-text
GEN_MODEL     = "llama3.2"              # ollama pull llama3.2
EMBED_DIMS    = 768                     # nomic-embed-text output dims

# ── APP CONFIG ────────────────────────────────────────────────
DEMO_DIMS     = 16
HNSW_M        = 16
HNSW_EF       = 200
CHUNK_WORDS   = 250
OVERLAP_WORDS = 30
PORT          = 5000

app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════════════
# DISTANCE METRICS
# ══════════════════════════════════════════════════════════════
def dist_euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def dist_cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)

def dist_manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric):
    return {"cosine": dist_cosine, "euclidean": dist_euclidean,
            "manhattan": dist_manhattan}.get(metric, dist_euclidean)


# ══════════════════════════════════════════════════════════════
# BRUTE FORCE
# ══════════════════════════════════════════════════════════════
class BruteForce:
    def __init__(self):
        self.items = {}

    def insert(self, item):
        self.items[item["id"]] = item

    def remove(self, vid):
        self.items.pop(vid, None)

    def knn(self, q, k, dist_fn):
        scored = [(dist_fn(q, v["emb"]), v["id"]) for v in self.items.values()]
        scored.sort(key=lambda x: x[0])
        return scored[:k]


# ══════════════════════════════════════════════════════════════
# KD-TREE
# ══════════════════════════════════════════════════════════════
class KDNode:
    __slots__ = ("item", "left", "right")
    def __init__(self, item):
        self.item  = item
        self.left  = None
        self.right = None

class KDTree:
    def __init__(self, dims):
        self.dims = dims
        self.root = None

    def _insert(self, node, item, depth):
        if node is None:
            return KDNode(item)
        axis = depth % self.dims
        if item["emb"][axis] < node.item["emb"][axis]:
            node.left  = self._insert(node.left,  item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, item):
        self.root = self._insert(self.root, item, 0)

    def rebuild(self, items):
        self.root = None
        for it in items:
            self.insert(it)

    def _search(self, node, q, k, depth, dist_fn, heap):
        import heapq
        if node is None:
            return
        d = dist_fn(q, node.item["emb"])
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.item["id"]))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.item["id"]))
        axis = depth % self.dims
        diff = q[axis] - node.item["emb"][axis]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._search(closer,  q, k, depth + 1, dist_fn, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._search(farther, q, k, depth + 1, dist_fn, heap)

    def knn(self, q, k, dist_fn):
        import heapq
        heap = []
        self._search(self.root, q, k, 0, dist_fn, heap)
        return sorted((-nd, vid) for nd, vid in heap)


# ══════════════════════════════════════════════════════════════
# HNSW
# ══════════════════════════════════════════════════════════════
class HNSW:
    def __init__(self, M=16, ef_construction=200):
        self.M           = M
        self.M0          = 2 * M
        self.ef          = ef_construction
        self.mL          = 1.0 / math.log(M)
        self.graph       = {}
        self.entry_point = None
        self.top_layer   = -1
        self._rng        = random.Random(42)

    def _rand_level(self):
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q_emb, ep_id, ef, layer, dist_fn):
        import heapq
        visited = {ep_id}
        d0      = dist_fn(q_emb, self.graph[ep_id]["item"]["emb"])
        cands   = [(d0, ep_id)]
        found   = [(-d0, ep_id)]
        while cands:
            cd, cid = heapq.heappop(cands)
            if cd > -found[0][0] and len(found) >= ef:
                break
            nbrs = self.graph[cid]["nbrs"][layer] if layer < len(self.graph[cid]["nbrs"]) else []
            for nid in nbrs:
                if nid not in visited and nid in self.graph:
                    visited.add(nid)
                    nd = dist_fn(q_emb, self.graph[nid]["item"]["emb"])
                    if len(found) < ef or nd < -found[0][0]:
                        heapq.heappush(cands, (nd, nid))
                        heapq.heappush(found, (-nd, nid))
                        if len(found) > ef:
                            heapq.heappop(found)
        return sorted((-nd, vid) for nd, vid in found)

    def insert(self, item, dist_fn):
        vid = item["id"]
        lvl = self._rand_level()
        self.graph[vid] = {"item": item, "maxLyr": lvl, "nbrs": [[] for _ in range(lvl + 1)]}
        if self.entry_point is None:
            self.entry_point = vid
            self.top_layer   = lvl
            return
        ep = self.entry_point
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.graph[ep]["nbrs"]):
                W = self._search_layer(item["emb"], ep, 1, lc, dist_fn)
                if W: ep = W[0][1]
        for lc in range(min(self.top_layer, lvl), -1, -1):
            W    = self._search_layer(item["emb"], ep, self.ef, lc, dist_fn)
            maxm = self.M0 if lc == 0 else self.M
            sel  = [vid2 for _, vid2 in W[:maxm]]
            self.graph[vid]["nbrs"][lc] = sel
            for nid in sel:
                if nid not in self.graph: continue
                while len(self.graph[nid]["nbrs"]) <= lc:
                    self.graph[nid]["nbrs"].append([])
                conn = self.graph[nid]["nbrs"][lc]
                conn.append(vid)
                if len(conn) > maxm:
                    pairs = [(dist_fn(self.graph[nid]["item"]["emb"],
                                     self.graph[c]["item"]["emb"]), c)
                             for c in conn if c in self.graph]
                    pairs.sort()
                    self.graph[nid]["nbrs"][lc] = [c for _, c in pairs[:maxm]]
            if W: ep = W[0][1]
        if lvl > self.top_layer:
            self.top_layer   = lvl
            self.entry_point = vid

    def knn(self, q_emb, k, ef, dist_fn):
        if self.entry_point is None: return []
        ep = self.entry_point
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.graph[ep]["nbrs"]):
                W = self._search_layer(q_emb, ep, 1, lc, dist_fn)
                if W: ep = W[0][1]
        return self._search_layer(q_emb, ep, max(ef, k), 0, dist_fn)[:k]

    def remove(self, vid):
        if vid not in self.graph: return
        for node in self.graph.values():
            for layer in node["nbrs"]:
                if vid in layer: layer.remove(vid)
        if self.entry_point == vid:
            self.entry_point = next((i for i in self.graph if i != vid), None)
        del self.graph[vid]
        self.top_layer = max((n["maxLyr"] for n in self.graph.values()), default=-1)

    def info(self):
        max_layer       = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_layer
        edges_per_layer = [0] * max_layer
        edges, nodes    = [], []
        for vid, node in self.graph.items():
            nodes.append({"id": vid, "metadata": node["item"].get("metadata", ""),
                          "category": node["item"].get("category", ""), "maxLyr": node["maxLyr"]})
            for lc, nbrs in enumerate(node["nbrs"]):
                if lc < max_layer: nodes_per_layer[lc] += 1
                for nid in nbrs:
                    if vid < nid and lc < max_layer:
                        edges_per_layer[lc] += 1
                        edges.append({"src": vid, "dst": nid, "lyr": lc})
        return {
            "topLayer": self.top_layer, "nodeCount": len(self.graph),
            "total_nodes": len(self.graph), "max_layer": self.top_layer,
            "M": self.M, "ef_construction": self.ef, "avg_degree": "—",
            "layers": [{"level": i, "node_count": nodes_per_layer[i],
                        "edge_count": edges_per_layer[i]} for i in range(max_layer)],
            "nodes": nodes, "edges": edges,
        }


# ══════════════════════════════════════════════════════════════
# VECTOR DB — 16D demo index
# ══════════════════════════════════════════════════════════════
class VectorDB:
    def __init__(self, dims=DEMO_DIMS):
        self.dims   = dims
        self._store = {}
        self._bf    = BruteForce()
        self._kdt   = KDTree(dims)
        self._hnsw  = HNSW(HNSW_M, HNSW_EF)
        self._lock  = threading.Lock()
        self._next  = 1

    def insert(self, metadata, category, emb, dist_fn=None):
        df = dist_fn or dist_cosine
        with self._lock:
            item = {"id": self._next, "metadata": metadata, "category": category, "emb": emb}
            self._next += 1
            self._store[item["id"]] = item
            self._bf.insert(item)
            self._kdt.insert(item)
            self._hnsw.insert(item, df)
            return item["id"]

    def remove(self, vid):
        with self._lock:
            if vid not in self._store: return False
            self._store.pop(vid)
            self._bf.remove(vid)
            self._hnsw.remove(vid)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, q, k, metric="cosine", algo="hnsw"):
        df = get_dist_fn(metric)
        t0 = time.perf_counter()
        with self._lock:
            if algo == "bruteforce": raw = self._bf.knn(q, k, df)
            elif algo == "kdtree":   raw = self._kdt.knn(q, k, df)
            else:                    raw = self._hnsw.knn(q, k, 50, df)
        us = int((time.perf_counter() - t0) * 1_000_000)
        results = []
        for d, vid in raw:
            item = self._store.get(vid)
            if item:
                results.append({"id": vid, "label": item["metadata"],
                                 "category": item["category"],
                                 "score": round(1.0 - d, 6),
                                 "distance": round(d, 6)})
        return results, us, algo, metric

    def benchmark(self, q, k, metric="cosine"):
        df = get_dist_fn(metric)
        def t(fn):
            s = time.perf_counter(); fn()
            return int((time.perf_counter() - s) * 1_000_000)
        with self._lock:
            return {
                "brute":  {"time_us": t(lambda: self._bf.knn(q, k, df))},
                "kdtree": {"time_us": t(lambda: self._kdt.knn(q, k, df))},
                "hnsw":   {"time_us": t(lambda: self._hnsw.knn(q, k, 50, df))},
                "items":  len(self._store),
            }

    def all_items(self):
        with self._lock: return list(self._store.values())
    def hnsw_info(self):
        with self._lock: return self._hnsw.info()
    def size(self): return len(self._store)


# ══════════════════════════════════════════════════════════════
# DOCUMENT DB — 768D Ollama nomic-embed-text
# ══════════════════════════════════════════════════════════════
class DocumentDB:
    def __init__(self):
        self._store = {}
        self._hnsw  = HNSW(HNSW_M, HNSW_EF)
        self._bf    = BruteForce()
        self._lock  = threading.Lock()
        self._next  = 1
        self._dims  = 0

    def insert(self, title, text, emb, chunk_idx=0, total_chunks=1):
        with self._lock:
            if self._dims == 0: self._dims = len(emb)
            doc = {"id": self._next, "title": title, "text": text, "emb": emb,
                   "chunk_index": chunk_idx, "total_chunks": total_chunks}
            self._next += 1
            self._store[doc["id"]] = doc
            vi = {"id": doc["id"], "metadata": title, "category": "doc", "emb": emb}
            self._bf.insert(vi)
            self._hnsw.insert(vi, dist_cosine)
            return doc["id"]

    def search(self, q_emb, k=3, max_dist=0.85):
        with self._lock:
            if not self._store: return []
            # BruteForce.knn(q, k, dist_fn)  — 3 args
            # HNSW.knn(q, k, ef, dist_fn)    — 4 args
            if len(self._store) < 10:
                raw = self._bf.knn(q_emb, k, dist_cosine)
            else:
                raw = self._hnsw.knn(q_emb, k, 50, dist_cosine)
        return [(d, self._store[vid]) for d, vid in raw
                if vid in self._store and d <= max_dist]

    def remove(self, vid):
        with self._lock:
            if vid not in self._store: return False
            self._store.pop(vid)
            self._bf.remove(vid)
            self._hnsw.remove(vid)
            return True

    def all_docs(self):
        with self._lock:
            docs = sorted(self._store.values(), key=lambda d: d["id"])
        return [{"id": d["id"], "title": d["title"],
                 "preview": d["text"][:120] + ("…" if len(d["text"]) > 120 else ""),
                 "word_count": len(d["text"].split()),
                 "chunk_index": d["chunk_index"],
                 "total_chunks": d["total_chunks"]}
                for d in docs]

    def size(self): return len(self._store)

    @property
    def dims(self): return self._dims


# ══════════════════════════════════════════════════════════════
# TEXT CHUNKER
# ══════════════════════════════════════════════════════════════
def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    if not words: return []
    if len(words) <= chunk_words: return [text]
    step, chunks, i = chunk_words - overlap, [], 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words): break
        i += step
    return chunks


# ══════════════════════════════════════════════════════════════
# OLLAMA — embeddings (nomic-embed-text)
# ══════════════════════════════════════════════════════════════
def ollama_embed(text: str) -> list:
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json().get("embedding", [])
        print(f"[Ollama Embed] Error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[Ollama Embed] Exception: {e}")
    return []


# ══════════════════════════════════════════════════════════════
# OLLAMA — generation (llama3.2)
# ══════════════════════════════════════════════════════════════
def ollama_generate(prompt: str) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
            timeout=240,
        )
        if r.status_code == 200:
            return r.json().get("response", "No response from model.")
        print(f"[Ollama Gen] Error {r.status_code}: {r.text[:300]}")
        return f"Ollama error {r.status_code} — is ollama serve running?"
    except Exception as e:
        print(f"[Ollama Gen] Exception: {e}")
        return "Ollama is not running. Start it with: ollama serve"


# ══════════════════════════════════════════════════════════════
# OLLAMA — availability check
# ══════════════════════════════════════════════════════════════
def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# DATABASE INSTANCES
# ══════════════════════════════════════════════════════════════
db     = VectorDB(DEMO_DIMS)
doc_db = DocumentDB()


# ══════════════════════════════════════════════════════════════
# DEMO DATA — 20 pre-loaded 16D vectors
# ══════════════════════════════════════════════════════════════
def load_demo():
    def ins(meta, cat, emb):
        db.insert(meta, cat, emb, dist_cosine)
    ins("Linked List: nodes connected by pointers",                     "cs",     [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06])
    ins("Binary Search Tree: O(log n) search and insert",              "cs",     [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10])
    ins("Dynamic Programming: memoization overlapping subproblems",    "cs",     [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07])
    ins("Graph BFS and DFS: breadth and depth first traversal",        "cs",     [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07])
    ins("Hash Table: O(1) average lookup with chaining",               "cs",     [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09])
    ins("Calculus: derivatives integrals and limits",                  "math",   [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10])
    ins("Linear Algebra: matrices eigenvalues eigenvectors",           "math",   [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09])
    ins("Probability: distributions random variables Bayes theorem",   "math",   [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08])
    ins("Number Theory: primes modular arithmetic RSA cryptography",   "math",   [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06])
    ins("Combinatorics: permutations combinations generating functions","math",   [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07])
    ins("Neapolitan Pizza: wood-fired dough San Marzano tomatoes",     "food",   [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07])
    ins("Sushi: vinegared rice raw fish and nori rolls",               "food",   [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08])
    ins("Ramen: noodle soup with chashu pork and soft-boiled egg",     "food",   [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06])
    ins("Tacos: corn tortillas with carnitas salsa and cilantro",      "food",   [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09])
    ins("Croissant: laminated pastry with buttery flaky layers",       "food",   [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06])
    ins("Basketball: shooting dribbling fast breaks slam dunks",       "sports", [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72])
    ins("American Football: tackles touchdowns field goals strategy",  "sports", [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76])
    ins("Tennis: racket volleys groundstrokes Wimbledon serves",       "sports", [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82])
    ins("Chess: openings endgames tactics strategic board game",       "sports", [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90])
    ins("Swimming: butterfly freestyle backstroke Olympic competition", "sports", [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80])

load_demo()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def parse_vec(s):
    try:    return [float(x) for x in s.split(",") if x.strip()]
    except: return []


# ══════════════════════════════════════════════════════════════
# SERVE FRONTEND
# ══════════════════════════════════════════════════════════════
@app.route("/")
def serve_index():
    p = os.path.join(os.path.dirname(__file__), "index.html")
    return send_file(p) if os.path.exists(p) else \
           ("<h1>index.html not found — place it next to server_final.py</h1>", 404)


# ══════════════════════════════════════════════════════════════
# DEMO VECTOR ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/search")
def search():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DEMO_DIMS:
        return jsonify({"error": f"vector must be {DEMO_DIMS}D"}), 400
    k      = max(1, min(50, int(request.args.get("k", 5))))
    metric = request.args.get("metric", "cosine")
    algo   = request.args.get("algo",   "hnsw")
    results, us, algo_used, metric_used = db.search(q, k, metric, algo)
    return jsonify({"results": results, "latencyUs": us,
                    "algo": algo_used, "metric": metric_used})

@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(force=True) or {}
    meta = body.get("metadata", "")
    cat  = body.get("category", "")
    emb  = body.get("embedding", [])
    if not meta or len(emb) != DEMO_DIMS:
        return jsonify({"error": f"need metadata and {DEMO_DIMS}D embedding"}), 400
    return jsonify({"id": db.insert(meta, cat, emb, dist_cosine)})

@app.route("/delete/<int:vid>", methods=["DELETE"])
def delete(vid):
    return jsonify({"ok": db.remove(vid)})

@app.route("/items")
def items():
    return jsonify([{"id": v["id"], "metadata": v["metadata"], "category": v["category"]}
                    for v in db.all_items()])

@app.route("/benchmark")
def benchmark():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DEMO_DIMS:
        return jsonify({"error": f"need {DEMO_DIMS}D vector"}), 400
    k      = max(1, min(50, int(request.args.get("k", 5))))
    metric = request.args.get("metric", "cosine")
    return jsonify(db.benchmark(q, k, metric))

@app.route("/hnsw-info")
def hnsw_info():
    return jsonify(db.hnsw_info())

@app.route("/stats")
def stats():
    return jsonify({"total_vectors": db.size(), "doc_chunks": doc_db.size(),
                    "dims": DEMO_DIMS, "doc_dims": doc_db.dims})

@app.route("/status")
def status():
    up = ollama_available()
    return jsonify({
        "ollama_online": up,
        "embed_model":   EMBED_MODEL,
        "gen_model":     GEN_MODEL,
        "vector_count":  db.size(),
        "doc_chunks":    doc_db.size(),
    })


# ══════════════════════════════════════════════════════════════
# DOCUMENT + RAG ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body  = request.get_json(force=True) or {}
    title = body.get("title", "").strip()
    text  = body.get("text",  "").strip()
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400
    if not ollama_available():
        return jsonify({"error": "Ollama not running — start with: ollama serve"}), 503

    chunks = chunk_text(text, CHUNK_WORDS, OVERLAP_WORDS)
    total  = len(chunks)
    ids    = []

    for i, chunk in enumerate(chunks):
        emb = ollama_embed(chunk)
        if not emb:
            return jsonify({"error": f"Embedding failed for chunk {i+1}. Is nomic-embed-text pulled?"}), 503
        chunk_title = f"{title} [{i+1}/{total}]" if total > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb, i, total))

    return jsonify({"ids": ids, "chunks_inserted": total, "dims": doc_db.dims})

@app.route("/doc/list")
def doc_list():
    return jsonify({"documents": doc_db.all_docs()})

@app.route("/doc/delete/<int:vid>", methods=["DELETE"])
def doc_delete(vid):
    return jsonify({"ok": doc_db.remove(vid)})

@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body     = request.get_json(force=True) or {}
    question = body.get("question", "").strip()
    k        = max(1, min(10, int(body.get("k", 3))))
    if not question:
        return jsonify({"error": "need question"}), 400

    if not ollama_available():
        return jsonify({
            "answer": "Ollama is not running. Open a terminal and run: ollama serve",
            "context_chunks": []
        }), 503

    # Step 1 — embed question
    q_emb = ollama_embed(question)
    if not q_emb:
        return jsonify({"answer": "Embedding failed — is nomic-embed-text pulled?",
                        "context_chunks": []}), 503

    # Step 2 — HNSW retrieval
    hits = doc_db.search(q_emb, k)
    if not hits:
        return jsonify({
            "answer": "No relevant documents found. Insert some documents in the Documents tab first.",
            "context_chunks": [],
        })

    # Step 3 — build RAG prompt
    context = ""
    for idx, (d, doc) in enumerate(hits):
        context += f"[{idx+1}] {doc['title']}:\n{doc['text']}\n\n"

    prompt = (
        "You are a helpful assistant. Answer the question based ONLY on the context below. "
        "Be concise and accurate. If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}"
        f"Question: {question}\n\nAnswer:"
    )

    # Step 4 — generate with llama3.2
    answer = ollama_generate(prompt)

    # Step 5 — return response
    context_chunks = [
        {"id": doc["id"], "title": doc["title"], "distance": round(d, 4),
         "text": doc["text"][:200], "text_preview": doc["text"][:200]}
        for d, doc in hits
    ]
    return jsonify({"answer": answer, "context_chunks": context_chunks})


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    up = ollama_available()
    print("\n" + "="*52)
    print("  VectorDB — Full Ollama Backend")
    print(f"  http://localhost:{PORT}")
    print(f"  {db.size()} demo vectors | {DEMO_DIMS}D | HNSW+KD-Tree+Brute")
    print(f"  Ollama   : {'ONLINE' if up else 'OFFLINE — run: ollama serve'}")
    if up:
        print(f"  Embed    : {EMBED_MODEL}")
        print(f"  Generate : {GEN_MODEL}")
    print("="*52 + "\n")
    if not up:
        print("  ⚠  Ollama offline — Search tab works but RAG needs Ollama.")
        print("  Run in another terminal:  ollama serve\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)