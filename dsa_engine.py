"""
TechBridge - dsa_engine.py
==========================
Pure-Python Data Structures & Algorithms core. NO Flask, NO database code here,
so a console menu or a web front end can call exactly the same functions.

Data structure  ->  Where it is used
-----------------------------------------------------------------------------
Graph (adjacency lists, dict[str,set])   the social network G(V,E)
Queue  (collections.deque)               pending connection requests, notifications
BFS                                      degree of connection, shortest chain,
                                         "people you may know" (friends-of-friends),
                                         communities (connected components)
Trie                                     prefix search of names / usernames
Max-Heap (heapq, negated keys)           k-way merge of time-ordered posts = home feed,
                                         trending posts (top-k)
Hash table (Python dict / set)           O(1) user lookup, neighbour sets
Jaccard coefficient                      ranking of "people you may know"
"""
import heapq
import time
from collections import deque


# ----------------------------------------------------------------------------
# 1. GRAPH  (adjacency lists) + request queues
# ----------------------------------------------------------------------------
class SocialGraph:
    """
    Accepted connection  = UNDIRECTED edge  (stored in both users' neighbour sets)
    Pending request      = DIRECTED edge sender -> receiver, kept in the receiver's
                           FIFO queue (inbox) until accepted / rejected / cancelled.
    No self edges, no multiple edges.
    """

    def __init__(self):
        self.adj = {}      # username -> set(usernames)       (adjacency list)
        self.inbox = {}    # receiver -> deque(senders)       (pending queue, FIFO)
        self.outbox = {}   # sender   -> set(receivers)       (to allow O(1) cancel/check)

    # ---- vertices ----------------------------------------------------------
    def add_user(self, u):
        self.adj.setdefault(u, set())
        self.inbox.setdefault(u, deque())
        self.outbox.setdefault(u, set())

    def neighbors(self, u):
        return set(self.adj.get(u, ()))

    def are_connected(self, a, b):
        return b in self.adj.get(a, ())

    def has_pending(self, sender, receiver):
        return receiver in self.outbox.get(sender, ())

    # ---- connection requests ----------------------------------------------
    def send_request(self, s, r):
        if s == r:
            raise ValueError("You cannot connect with yourself")
        if self.are_connected(s, r):
            raise ValueError("Already connected")
        if self.has_pending(s, r):
            raise ValueError("Request already sent")
        if self.has_pending(r, s):
            raise ValueError("This user already sent you a request - accept it instead")
        self.inbox[r].append(s)          # enqueue
        self.outbox[s].add(r)

    def _dequeue(self, sender, receiver):
        if not self.has_pending(sender, receiver):
            raise ValueError("No such pending request")
        self.inbox[receiver].remove(sender)
        self.outbox[sender].discard(receiver)

    def accept(self, receiver, sender):
        self._dequeue(sender, receiver)
        self.adj[sender].add(receiver)   # undirected edge
        self.adj[receiver].add(sender)

    def reject(self, receiver, sender):
        self._dequeue(sender, receiver)

    def cancel(self, sender, receiver):
        self._dequeue(sender, receiver)

    def remove_connection(self, a, b):
        if not self.are_connected(a, b):
            raise ValueError("You are not connected")
        self.adj[a].discard(b)
        self.adj[b].discard(a)

    def incoming(self, u):
        return list(self.inbox.get(u, ()))      # oldest first

    def outgoing(self, u):
        return sorted(self.outbox.get(u, ()))

    # ---- persistence helpers ----------------------------------------------
    def edges_list(self):
        return sorted({tuple(sorted((a, b))) for a, ns in self.adj.items() for b in ns})

    def pending_list(self):
        return [[s, r] for r, q in self.inbox.items() for s in q]

    # ---- BFS algorithms ---------------------------------------------------
    def shortest_path(self, src, dst):
        """BFS. Returns [src, ..., dst] or None. len(path)-1 = degree of connection."""
        if src not in self.adj or dst not in self.adj:
            return None
        if src == dst:
            return [src]
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if v not in parent:
                    parent[v] = u
                    if v == dst:
                        path = [v]
                        while parent[path[-1]] is not None:
                            path.append(parent[path[-1]])
                        return path[::-1]
                    q.append(v)
        return None

    def mutual(self, a, b):
        """Mutual connections = intersection of two neighbour sets."""
        return self.adj.get(a, set()) & self.adj.get(b, set())

    def suggestions(self, u, k=10):
        """
        People you may know.
        Candidates: users that are not connected to u and have no pending request with u.
        Score: Jaccard coefficient  |N(u) ∩ N(v)| / |N(u) ∪ N(v)|  (overlap of neighbourhoods).
        Friends-of-friends (BFS level 2) always rank above strangers (score 0).
        Returns [(username, score, mutual_count)] best first.
        """
        mine = self.adj.get(u, set())
        blocked = mine | {u} | self.outbox.get(u, set()) | set(self.inbox.get(u, ()))
        ranked = []
        for v in self.adj:
            if v in blocked:
                continue
            nv = self.adj[v]
            inter = len(mine & nv)
            union = len(mine | nv)
            score = inter / union if union else 0.0
            ranked.append((v, round(score, 3), inter))
        ranked.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return ranked[:k]

    def communities(self, min_size=2):
        """Groups of closely connected students = connected components (BFS)."""
        seen, groups = set(), []
        for start in self.adj:
            if start in seen:
                continue
            comp, q = [], deque([start])
            seen.add(start)
            while q:
                u = q.popleft()
                comp.append(u)
                for v in self.adj[u]:
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
            if len(comp) >= min_size:
                groups.append(sorted(comp))
        groups.sort(key=len, reverse=True)
        return groups


# ----------------------------------------------------------------------------
# 2. TRIE  (prefix search)
# ----------------------------------------------------------------------------
class _TrieNode:
    __slots__ = ("children", "ids")

    def __init__(self):
        self.children = {}
        self.ids = set()      # every user whose word passes through this node


class Trie:
    """Insert first name, last name and username; search by prefix as you type."""

    def __init__(self):
        self.root = _TrieNode()

    def insert(self, word, uid):
        node = self.root
        for ch in word.lower():
            node = node.children.setdefault(ch, _TrieNode())
            node.ids.add(uid)

    def search(self, prefix, limit=10):
        node = self.root
        for ch in prefix.lower():
            node = node.children.get(ch)
            if node is None:
                return []
        return sorted(node.ids)[:limit]


# ----------------------------------------------------------------------------
# 3. HEAPS  (home feed + trending)
# ----------------------------------------------------------------------------
def merge_feeds(lists, limit=50):
    """
    K-way merge using a max-heap (Python's heapq is a min-heap, so we negate timestamps).
    `lists` = one list of posts per connection, each already sorted newest-first.
    Time: O(N log K) for N output posts and K connections.
    """
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (-lst[0]["ts"], i, 0))
    out = []
    while heap and len(out) < limit:
        _, i, j = heapq.heappop(heap)
        out.append(lists[i][j])
        if j + 1 < len(lists[i]):
            heapq.heappush(heap, (-lists[i][j + 1]["ts"], i, j + 1))
    return out


def trending_score(post, now=None):
    """Engagement divided by age -> fresh, popular posts float up."""
    now = now or time.time()
    engagement = 2 * len(post["likes"]) + 3 * sum(1 + len(c["replies"]) for c in post["comments"]) + 1
    age_hours = max(0.0, (now - post["ts"]) / 3600)
    return engagement / ((age_hours + 2) ** 1.2)


def top_k_trending(posts, k=5):
    """Heap-based top-k (heapq.nlargest keeps a heap of size k)."""
    now = time.time()
    return heapq.nlargest(k, posts, key=lambda p: trending_score(p, now))


# ----------------------------------------------------------------------------
# 4. NOTIFICATION QUEUE
# ----------------------------------------------------------------------------
class NotificationQueue:
    """FIFO queue per user (bounded). Newest-first view for the bell panel."""

    def __init__(self, items=None, maxlen=300):
        self.q = deque(items or [], maxlen=maxlen)

    def push(self, item):
        self.q.append(item)

    def newest_first(self):
        return list(reversed(self.q))

    def unread(self):
        return sum(1 for n in self.q if not n.get("read"))

    def mark_all_read(self):
        for n in self.q:
            n["read"] = True

    def to_list(self):
        return list(self.q)