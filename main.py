"""
VectorDB Engine — Python port of the original C++ implementation.
HNSW + KD-Tree + Brute Force search with Ollama RAG pipeline.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import requests
from flask import Flask, Response, request, send_from_directory

DIMS = 16  # demo vectors
CHUNK_WORDS = 400
CHUNK_OVERLAP = 40
DEFAULT_RAG_K = 2

DistFn = Callable[[List[float], List[float]], float]


# =====================================================================
#  DATA TYPES
# =====================================================================


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]


# =====================================================================
#  DISTANCE METRICS
# =====================================================================


def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================


class BruteForce:
    def __init__(self) -> None:
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        results = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort(key=lambda x: x[0])
        return results[:k]

    def remove(self, item_id: int) -> None:
        self.items = [v for v in self.items if v.id != item_id]


# =====================================================================
#  KD-TREE
# =====================================================================


@dataclass
class KDNode:
    item: VectorItem
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTree:
    def __init__(self, dims: int) -> None:
        self.root: Optional[KDNode] = None
        self.dims = dims

    def _destroy(self, n: Optional[KDNode]) -> None:
        if not n:
            return
        self._destroy(n.left)
        self._destroy(n.right)

    def _ins(self, n: Optional[KDNode], v: VectorItem, d: int) -> KDNode:
        if not n:
            return KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < n.item.emb[ax]:
            n.left = self._ins(n.left, v, d + 1)
        else:
            n.right = self._ins(n.right, v, d + 1)
        return n

    def _knn(
        self,
        n: Optional[KDNode],
        q: List[float],
        k: int,
        d: int,
        dist: DistFn,
        heap: List[Tuple[float, int]],
    ) -> None:
        if not n:
            return
        dn = dist(q, n.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, n.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax = d % self.dims
        diff = q[ax] - n.item.emb[ax]
        closer = n.left if diff < 0 else n.right
        farther = n.right if diff < 0 else n.left
        self._knn(closer, q, k, d + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, d + 1, dist, heap)

    def insert(self, v: VectorItem) -> None:
        self.root = self._ins(self.root, v, 0)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        heap: List[Tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, heap)
        results = [(-d, i) for d, i in heap]
        results.sort(key=lambda x: x[0])
        return results

    def rebuild(self, items: List[VectorItem]) -> None:
        self._destroy(self.root)
        self.root = None
        for v in items:
            self.insert(v)


# =====================================================================
#  HNSW
# =====================================================================


@dataclass
class HNSWNode:
    item: VectorItem
    max_lyr: int
    nbrs: List[List[int]] = field(default_factory=list)


class HNSW:
    @dataclass
    class GraphInfo:
        top_layer: int
        node_count: int
        nodes_per_layer: List[int]
        edges_per_layer: List[int]

        @dataclass
        class NV:
            id: int
            metadata: str
            category: str
            max_lyr: int

        @dataclass
        class EV:
            src: int
            dst: int
            lyr: int

        nodes: List["HNSW.GraphInfo.NV"] = field(default_factory=list)
        edges: List["HNSW.GraphInfo.EV"] = field(default_factory=list)

    def __init__(self, m: int = 16, ef_build: int = 200) -> None:
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.G: Dict[int, HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt = -1
        self.rng = random.Random(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self.rng.random()) * self.mL))

    def _search_layer(
        self, q: List[float], ep: int, ef: int, lyr: int, dist: DistFn
    ) -> List[Tuple[float, int]]:
        vis: Dict[int, bool] = {}
        cands: List[Tuple[float, int]] = []
        found: List[Tuple[float, int]] = []

        d0 = dist(q, self.G[ep].item.emb)
        vis[ep] = True
        heapq.heappush(cands, (d0, ep))
        heapq.heappush(found, (-d0, ep))

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            if lyr >= len(self.G[cid].nbrs):
                continue
            for nid in self.G[cid].nbrs[lyr]:
                if vis.get(nid) or nid not in self.G:
                    continue
                vis[nid] = True
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        res = [(-d, i) for d, i in found]
        res.sort(key=lambda x: x[0])
        return res

    def _select_nbrs(self, cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [c[1] for c in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        item_id = item.id
        lvl = self._rand_level()
        self.G[item_id] = HNSWNode(item=item, max_lyr=lvl, nbrs=[[] for _ in range(lvl + 1)])

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            w = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(w, max_m)
            self.G[item_id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                if len(self.G[nid].nbrs) <= lc:
                    self.G[nid].nbrs.extend([[] for _ in range(lc + 1 - len(self.G[nid].nbrs))])
                conn = self.G[nid].nbrs[lc]
                conn.append(item_id)
                if len(conn) > max_m:
                    ds = [
                        (dist(self.G[nid].item.emb, self.G[c].item.emb), c)
                        for c in conn
                        if c in self.G
                    ]
                    ds.sort(key=lambda x: x[0])
                    self.G[nid].nbrs[lc] = [c for _, c in ds[:max_m]]

            if w:
                ep = w[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item_id

    def knn(
        self, q: List[float], k: int, ef: int, dist: DistFn
    ) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]
        w = self._search_layer(q, ep, max(ef, k), 0, dist)
        return w[:k]

    def remove(self, item_id: int) -> None:
        if item_id not in self.G:
            return
        for nd in self.G.values():
            for layer in nd.nbrs:
                if item_id in layer:
                    layer[:] = [x for x in layer if x != item_id]
        if self.entry_pt == item_id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != item_id:
                    self.entry_pt = nid
                    break
        del self.G[item_id]

    def get_info(self) -> GraphInfo:
        gi = HNSW.GraphInfo(
            top_layer=self.top_layer,
            node_count=len(self.G),
            nodes_per_layer=[],
            edges_per_layer=[],
        )
        max_l = max(self.top_layer + 1, 1)
        gi.nodes_per_layer = [0] * max_l
        gi.edges_per_layer = [0] * max_l
        for item_id, nd in self.G.items():
            gi.nodes.append(
                HNSW.GraphInfo.NV(
                    id=item_id,
                    metadata=nd.item.metadata,
                    category=nd.item.category,
                    max_lyr=nd.max_lyr,
                )
            )
            for lc in range(min(nd.max_lyr, max_l - 1) + 1):
                gi.nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if item_id < nid:
                            gi.edges_per_layer[lc] += 1
                            gi.edges.append(HNSW.GraphInfo.EV(src=item_id, dst=nid, lyr=lc))
        return gi

    def size(self) -> int:
        return len(self.G)


# =====================================================================
#  VECTOR DATABASE
# =====================================================================


@dataclass
class Hit:
    id: int
    meta: str
    cat: str
    emb: List[float]
    dist: float


@dataclass
class SearchOut:
    hits: List[Hit]
    us: int
    algo: str
    metric: str


@dataclass
class BenchOut:
    bf_us: int
    kd_us: int
    hnsw_us: int
    n: int


class VectorDB:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist: DistFn) -> int:
        with self.mu:
            v = VectorItem(id=self.next_id, metadata=meta, category=cat, emb=emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            rem = list(self.store.values())
            self.kdt.rebuild(rem)
            return True

    def search(
        self, q: List[float], k: int, metric: str, algo: str
    ) -> SearchOut:
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)

            us = int((time.perf_counter() - t0) * 1_000_000)
            hits = []
            for d, item_id in raw:
                if item_id in self.store:
                    s = self.store[item_id]
                    hits.append(
                        Hit(id=item_id, meta=s.metadata, cat=s.category, emb=s.emb, dist=d)
                    )
            return SearchOut(hits=hits, us=us, algo=algo, metric=metric)

    def benchmark(self, q: List[float], k: int, metric: str) -> BenchOut:
        with self.mu:

            def timed(fn: Callable[[], None]) -> int:
                t0 = time.perf_counter()
                fn()
                return int((time.perf_counter() - t0) * 1_000_000)

            dfn = get_dist_fn(metric)
            return BenchOut(
                bf_us=timed(lambda: self.bf.knn(q, k, dfn)),
                kd_us=timed(lambda: self.kdt.knn(q, k, dfn)),
                hnsw_us=timed(lambda: self.hnsw.knn(q, k, 50, dfn)),
                n=len(self.store),
            )

    def all(self) -> List[VectorItem]:
        with self.mu:
            return list(self.store.values())

    def hnsw_info(self) -> HNSW.GraphInfo:
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.mu:
            return len(self.store)


# =====================================================================
#  TEXT CHUNKER
# =====================================================================


def chunk_text(
    text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP
) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks: List[str] = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
    return chunks


# =====================================================================
#  OLLAMA CLIENT
# =====================================================================


class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434) -> None:
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2:1b"
        self._embed_cache: Dict[str, List[float]] = {}

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> List[float]:
        key = text.strip()
        if key in self._embed_cache:
            return self._embed_cache[key]
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.embed_model,
                    "prompt": key,
                    "keep_alive": "30m",
                },
                timeout=30,
            )
            if r.status_code != 200:
                return []
            data = r.json()
            emb = data.get("embedding", [])
            if emb:
                self._embed_cache[key] = emb
            return emb
        except (requests.RequestException, json.JSONDecodeError, KeyError):
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.gen_model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "30m",
                },
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            data = r.json()
            return data.get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
#  DOCUMENT DATABASE
# =====================================================================


class DocumentDB:
    def __init__(self) -> None:
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.mu = Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(id=self.next_id, title=title, text=text, emb=emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(id=item.id, metadata=title, category="doc", emb=emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(
        self, q: List[float], k: int, max_dist: float = 0.7
    ) -> List[Tuple[float, DocItem]]:
        with self.mu:
            if not self.store:
                return []
            raw = (
                self.bf.knn(q, k, cosine)
                if len(self.store) < 10
                else self.hnsw.knn(q, k, 50, cosine)
            )
            out: List[Tuple[float, DocItem]] = []
            for d, item_id in raw:
                if item_id in self.store and d <= max_dist:
                    out.append((d, self.store[item_id]))
            return out

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all(self) -> List[DocItem]:
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims


# =====================================================================
#  DEMO DATA
# =====================================================================


def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    demo = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
    ]
    for meta, cat, emb in demo:
        db.insert(meta, cat, emb, dist)


# =====================================================================
#  JSON / HTTP HELPERS
# =====================================================================


def parse_vec(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def cors_headers() -> Dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def json_response(data: object, status: int = 200) -> Response:
    resp = Response(json.dumps(data), status=status, mimetype="application/json")
    for k, v in cors_headers().items():
        resp.headers[k] = v
    return resp


def extract_str(body: str, key: str) -> str:
    pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    m = re.search(pattern, body)
    if not m:
        return ""
    return bytes(m.group(1), "utf-8").decode("unicode_escape")


def extract_int(body: str, key: str, default: int = 0) -> int:
    pattern = rf'"{re.escape(key)}"\s*:\s*(-?\d+)'
    m = re.search(pattern, body)
    if not m:
        return default
    try:
        return int(m.group(1))
    except ValueError:
        return default


def parse_body(body: str) -> Tuple[str, str, List[float]]:
    meta = extract_str(body, "metadata")
    cat = extract_str(body, "category")
    arr_match = re.search(r'"embedding"\s*:\s*\[([^\]]*)\]', body)
    emb = parse_vec(arr_match.group(1)) if arr_match else []
    return meta, cat, emb


# =====================================================================
#  FLASK APP
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)


@app.after_request
def add_cors(response: Response) -> Response:
    for k, v in cors_headers().items():
        response.headers[k] = v
    return response


@app.route("/", methods=["GET"])
def index() -> Response:
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/asset/<path:filename>", methods=["GET"])
def asset(filename: str) -> Response:
    return send_from_directory(os.path.join(BASE_DIR, "asset"), filename)


@app.route("/search", methods=["GET"])
def search() -> Response:
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return json_response({"error": f"need {DIMS}D vector"}, 400)
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"
    out = db.search(q, k, metric, algo)
    return json_response(
        {
            "results": [
                {
                    "id": h.id,
                    "metadata": h.meta,
                    "category": h.cat,
                    "distance": round(h.dist, 6),
                    "embedding": [round(x, 4) for x in h.emb],
                }
                for h in out.hits
            ],
            "latencyUs": out.us,
            "algo": out.algo,
            "metric": out.metric,
        }
    )


@app.route("/insert", methods=["POST"])
def insert() -> Response:
    meta, cat, emb = parse_body(request.get_data(as_text=True))
    if not meta or not emb or len(emb) != DIMS:
        return json_response({"error": "invalid body"}, 400)
    item_id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return json_response({"id": item_id})


@app.route("/delete/<int:item_id>", methods=["DELETE"])
def delete_item(item_id: int) -> Response:
    ok = db.remove(item_id)
    return json_response({"ok": ok})


@app.route("/items", methods=["GET"])
def items() -> Response:
    return json_response(
        [
            {
                "id": v.id,
                "metadata": v.metadata,
                "category": v.category,
                "embedding": [round(x, 4) for x in v.emb],
            }
            for v in db.all()
        ]
    )


@app.route("/benchmark", methods=["GET"])
def benchmark() -> Response:
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return json_response({"error": f"need {DIMS}D vector"}, 400)
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric") or "cosine"
    b = db.benchmark(q, k, metric)
    return json_response(
        {
            "bruteforceUs": b.bf_us,
            "kdtreeUs": b.kd_us,
            "hnswUs": b.hnsw_us,
            "itemCount": b.n,
        }
    )


@app.route("/hnsw-info", methods=["GET"])
def hnsw_info() -> Response:
    gi = db.hnsw_info()
    return json_response(
        {
            "topLayer": gi.top_layer,
            "nodeCount": gi.node_count,
            "nodesPerLayer": gi.nodes_per_layer,
            "edgesPerLayer": gi.edges_per_layer,
            "nodes": [
                {
                    "id": n.id,
                    "metadata": n.metadata,
                    "category": n.category,
                    "maxLyr": n.max_lyr,
                }
                for n in gi.nodes
            ],
            "edges": [{"src": e.src, "dst": e.dst, "lyr": e.lyr} for e in gi.edges],
        }
    )


@app.route("/doc/insert", methods=["POST"])
def doc_insert() -> Response:
    body = request.get_data(as_text=True)
    title = extract_str(body, "title")
    text = extract_str(body, "text")
    if not title or not text:
        return json_response({"error": "need title and text"}, 400)

    chunks = chunk_text(text)
    ids: List[int] = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return json_response(
                {
                    "error": (
                        "Ollama unavailable. Install from https://ollama.com then run: "
                        "ollama pull nomic-embed-text && ollama pull llama3.2:1b"
                    )
                },
                503,
            )
        chunk_title = (
            f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
        )
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return json_response({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.route("/doc/delete/<int:item_id>", methods=["DELETE"])
def doc_delete(item_id: int) -> Response:
    ok = doc_db.remove(item_id)
    return json_response({"ok": ok})


@app.route("/doc/list", methods=["GET"])
def doc_list() -> Response:
    docs = doc_db.all()
    result = []
    for d in docs:
        preview = d.text[:120]
        if len(d.text) > 120:
            preview += "…"
        result.append(
            {
                "id": d.id,
                "title": d.title,
                "preview": preview,
                "words": len(d.text.split()),
            }
        )
    return json_response(result)


@app.route("/doc/search", methods=["POST"])
def doc_search() -> Response:
    body = request.get_data(as_text=True)
    question = extract_str(body, "question")
    k = extract_int(body, "k", DEFAULT_RAG_K)
    if not question:
        return json_response({"error": "need question"}, 400)

    q_emb = ollama.embed(question)
    if not q_emb:
        return json_response({"error": "Ollama unavailable"}, 503)

    hits = doc_db.search(q_emb, k)
    return json_response(
        {
            "contexts": [
                {
                    "id": item.id,
                    "title": item.title,
                    "distance": round(dist, 4),
                }
                for dist, item in hits
            ]
        }
    )


@app.route("/doc/ask", methods=["POST"])
def doc_ask() -> Response:
    body = request.get_data(as_text=True)
    question = extract_str(body, "question")
    k = extract_int(body, "k", DEFAULT_RAG_K)
    if not question:
        return json_response({"error": "need question"}, 400)

    q_emb = ollama.embed(question)
    if not q_emb:
        return json_response({"error": "Ollama unavailable"}, 503)

    hits = doc_db.search(q_emb, k)

    ctx_parts = []
    for i, (dist, item) in enumerate(hits):
        ctx_parts.append(f"[{i + 1}] {item.title}:\n{item.text}\n")
    ctx = "\n".join(ctx_parts)

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    answer = ollama.generate(prompt)

    return json_response(
        {
            "answer": answer,
            "model": ollama.gen_model,
            "contexts": [
                {
                    "id": item.id,
                    "title": item.title,
                    "text": item.text,
                    "distance": round(dist, 4),
                }
                for dist, item in hits
            ],
            "docCount": doc_db.size(),
        }
    )


@app.route("/status", methods=["GET"])
def status() -> Response:
    up = ollama.is_available()
    return json_response(
        {
            "ollamaAvailable": up,
            "embedModel": ollama.embed_model,
            "genModel": ollama.gen_model,
            "docCount": doc_db.size(),
            "docDims": doc_db.get_dims(),
            "demoDims": DIMS,
            "demoCount": db.size(),
        }
    )


@app.route("/stats", methods=["GET"])
def stats() -> Response:
    return json_response(
        {
            "count": db.size(),
            "dims": DIMS,
            "algorithms": ["bruteforce", "kdtree", "hnsw"],
            "metrics": ["euclidean", "cosine", "manhattan"],
        }
    )


def main() -> None:
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.isfile(index_path):
        print(f"ERROR: index.html not found at {index_path}")
        raise SystemExit(1)

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("Open in browser: http://localhost:8080")
    print(f"Serving from: {BASE_DIR}")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)


if __name__ == "__main__":
    main()
