"""
TechBridge - app.py  (Flask backend)
Run:  pip install -r requirements.txt   then   python app.py   ->  http://127.0.0.1:5000

The database (database.json) ONLY stores data. Every graph computation (suggestions,
degrees, shortest chain, feed, trending, search) runs in memory via dsa_engine.py.
"""
import os, re, json, time, uuid, hmac, hashlib, secrets, threading
from collections import deque
from functools import wraps

from flask import Flask, request, session, jsonify, render_template
from werkzeug.utils import secure_filename

from dsa_engine import (SocialGraph, Trie, NotificationQueue,
                        merge_feeds, top_k_trending)

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, "static", "uploads")
DB_FILE = os.path.join(BASE, "database.json")
KEY_FILE = os.path.join(BASE, ".secret_key")
os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(BASE), "Frontend"))
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024        # 300 MB uploads (videos)
if not os.path.exists(KEY_FILE):                            # keep sessions across restarts
    with open(KEY_FILE, "w") as f:
        f.write(secrets.token_hex(32))
app.secret_key = open(KEY_FILE).read().strip()

LOCK = threading.RLock()
DB = {"users": {}, "edges": [], "pending": [], "posts": [], "videos": [], "jobs": [],
      "ppts": [], "docs": {}, "saved": {}, "notifications": {}}
GRAPH = SocialGraph()
TRIE = Trie()
NOTIFS = {}          # username -> NotificationQueue
ROOMS = {}           # live call rooms (in memory only)


# ------------------------------------------------------------------ persistence
def index_user(u):
    x = DB["users"][u]
    GRAPH.add_user(u)
    NOTIFS.setdefault(u, NotificationQueue())
    for w in (x["first_name"], x["last_name"], u):
        TRIE.insert(w, u)


def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, encoding="utf-8") as f:
            DB.update(json.load(f))
    for u in DB["users"]:
        index_user(u)
    for a, b in DB["edges"]:
        GRAPH.adj[a].add(b); GRAPH.adj[b].add(a)
    for s, r in DB["pending"]:
        if s in GRAPH.adj and r in GRAPH.adj:
            GRAPH.send_request(s, r)
    for u, items in DB["notifications"].items():
        NOTIFS[u] = NotificationQueue(items)


def save_db():
    with LOCK:
        DB["edges"] = [list(e) for e in GRAPH.edges_list()]
        DB["pending"] = GRAPH.pending_list()
        DB["notifications"] = {u: q.to_list() for u, q in NOTIFS.items()}
        tmp = DB_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(DB, f)
        os.replace(tmp, DB_FILE)


# ------------------------------------------------------------------ helpers
class ApiError(Exception):
    def __init__(self, msg, code=400):
        self.msg, self.code = msg, code


@app.errorhandler(ApiError)
def _api_err(e):
    return jsonify(error=e.msg), e.code


@app.errorhandler(ValueError)
def _val_err(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(413)
def _too_big(e):
    return jsonify(error="File too large (max 300 MB)"), 413


def me():
    return session.get("u")


def login_required(fn):
    @wraps(fn)
    def w(*a, **k):
        if not me() or me() not in DB["users"]:
            raise ApiError("Please log in", 401)
        return fn(*a, **k)
    return w


def hash_pw(pw, salt=None):
    """Passwords are stored ONLY as salted PBKDF2-HMAC-SHA256 (hashlib)."""
    salt = salt or os.urandom(16).hex()
    return salt, hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 150_000).hex()


def brief(u):
    x = DB["users"].get(u)
    if not x:
        return {"username": u, "name": u, "photo": "", "qualification": "", "programme": "",
                "year": "", "semester": "", "skills": [], "bio": ""}
    return {"username": u, "name": f"{x['first_name']} {x['last_name']}", "photo": x.get("photo", ""),
            "qualification": x["qualification"], "programme": x["programme"], "year": x["year"],
            "semester": x.get("semester", ""), "skills": x["skills"], "bio": x.get("bio", "")}


def public(u):
    x = DB["users"][u]
    return {**brief(u), "first_name": x["first_name"], "last_name": x["last_name"],
            "certificates": x.get("certificates", []), "joined": x["joined"]}


ALLOWED = {"png", "jpg", "jpeg", "gif", "webp", "pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx",
           "txt", "md", "csv", "zip", "mp4", "webm", "mov", "mkv"}


def save_upload(f):
    """Validate extension (allow-list) and store with a random prefix."""
    if not f or not f.filename:
        return None
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED:
        raise ApiError(f"File type .{ext or '?'} is not allowed")
    name = secure_filename(f.filename) or f"file.{ext}"
    stored = f"{uuid.uuid4().hex[:12]}_{name}"
    f.save(os.path.join(UPLOADS, stored))
    kind = ("image" if ext in {"png", "jpg", "jpeg", "gif", "webp"}
            else "video" if ext in {"mp4", "webm", "mov", "mkv"} else "file")
    return {"url": f"/static/uploads/{stored}", "name": name, "kind": kind}


def notify(to, ntype, text, **extra):
    if to == me() and ntype in ("like", "comment"):
        return
    NOTIFS.setdefault(to, NotificationQueue()).push(
        {"id": uuid.uuid4().hex[:10], "type": ntype, "text": text, "from": me(),
         "ts": time.time(), "read": False, **extra})


def my_name():
    return brief(me())["name"]


def relation(a, b):
    if a == b: return "self"
    if GRAPH.are_connected(a, b): return "connected"
    if GRAPH.has_pending(a, b): return "sent"
    if GRAPH.has_pending(b, a): return "received"
    return "none"


# ------------------------------------------------------------------ pages
@app.get("/")
def index():
    from flask import send_file
    page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Frontend", "dashboard.html")
    return send_file(page)


# ------------------------------------------------------------------ accounts
@app.post("/api/register")
def register():
    d = request.get_json(force=True)
    need = ["first_name", "last_name", "dob", "username", "qualification", "year",
            "semester", "programme", "skills", "password"]
    for k in need:
        if not str(d.get(k, "")).strip():
            raise ApiError(f"Please fill: {k.replace('_', ' ')}")
    u = d["username"].strip().lower()
    if not re.fullmatch(r"[a-z0-9_.]{3,20}", u):
        raise ApiError("Username: 3-20 letters, digits, _ or .")
    pw = d["password"]
    if not (len(pw) >= 8 and re.search(r"[A-Z]", pw) and re.search(r"[a-z]", pw)
            and re.search(r"\d", pw) and re.search(r"[^\w\s]", pw)):
        raise ApiError("Password needs 8+ chars with uppercase, lowercase, digit and symbol")
    with LOCK:
        if u in DB["users"]:
            raise ApiError("Username already taken")
        salt, ph = hash_pw(pw)
        DB["users"][u] = {
            "first_name": d["first_name"].strip().title(), "last_name": d["last_name"].strip().title(),
            "dob": d["dob"], "qualification": d["qualification"], "year": d["year"],
            "semester": d["semester"], "programme": d["programme"],
            "skills": [s.strip() for s in d["skills"].split(",") if s.strip()],
            "bio": d.get("bio", "").strip()[:400], "photo": "", "certificates": [],
            "salt": salt, "pw": ph, "joined": time.time()}
        index_user(u)
        save_db()
    session.permanent = True
    session["u"] = u
    return {"ok": True}


@app.post("/api/login")
def login():
    d = request.get_json(force=True)
    u = str(d.get("username", "")).strip().lower()
    x = DB["users"].get(u)
    if not x:
        hash_pw("x")                      # keep timing similar
        raise ApiError("Wrong username or password", 401)
    _, ph = hash_pw(d.get("password", ""), x["salt"])
    if not hmac.compare_digest(ph, x["pw"]):
        raise ApiError("Wrong username or password", 401)
    session.permanent = True
    session["u"] = u
    return {"ok": True}


@app.post("/api/logout")
def logout():
    session.clear()
    return {"ok": True}


@app.get("/api/me")
def whoami():
    if not me() or me() not in DB["users"]:
        return {"user": None}
    return {"user": public(me())}


@app.post("/api/password")
@login_required
def change_pw():
    d = request.get_json(force=True)
    x = DB["users"][me()]
    if not hmac.compare_digest(hash_pw(d.get("old", ""), x["salt"])[1], x["pw"]):
        raise ApiError("Current password is wrong")
    pw = d.get("new", "")
    if not (len(pw) >= 8 and re.search(r"[A-Z]", pw) and re.search(r"[a-z]", pw)
            and re.search(r"\d", pw) and re.search(r"[^\w\s]", pw)):
        raise ApiError("New password is too weak")
    x["salt"], x["pw"] = hash_pw(pw)
    save_db()
    return {"ok": True}


# ------------------------------------------------------------------ profile
@app.get("/api/profile/<username>")
@login_required
def profile(username):
    u = username.lower()
    if u not in DB["users"]:
        raise ApiError("User not found", 404)
    m = me()
    out = public(u)
    out["connections"] = len(GRAPH.adj[u])
    out["relation"] = relation(m, u)
    if u != m:
        path = GRAPH.shortest_path(m, u)                       # BFS
        out["degree"] = len(path) - 1 if path else None
        out["chain"] = [brief(p) for p in path] if path else []
        out["mutual"] = [brief(x) for x in sorted(GRAPH.mutual(m, u))]
    return out


@app.post("/api/profile/update")
@login_required
def profile_update():
    x = DB["users"][me()]
    f = request.form
    for k in ("bio", "qualification", "programme", "year", "semester"):
        if k in f:
            x[k] = f[k].strip()[:400]
    if "skills" in f:
        x["skills"] = [s.strip() for s in f["skills"].split(",") if s.strip()]
    ph = save_upload(request.files.get("photo"))
    if ph:
        if ph["kind"] != "image":
            raise ApiError("Profile photo must be an image")
        x["photo"] = ph["url"]
    save_db()
    return public(me())


@app.post("/api/certificates")
@login_required
def add_cert():
    fl = save_upload(request.files.get("file"))
    if not fl:
        raise ApiError("Choose a certificate file")
    title = request.form.get("title", "").strip() or fl["name"]
    DB["users"][me()]["certificates"].append(
        {"id": uuid.uuid4().hex[:8], "title": title, "url": fl["url"], "kind": fl["kind"], "ts": time.time()})
    save_db()
    return public(me())


@app.delete("/api/certificates/<cid>")
@login_required
def del_cert(cid):
    x = DB["users"][me()]
    x["certificates"] = [c for c in x["certificates"] if c["id"] != cid]
    save_db()
    return public(me())


# ------------------------------------------------------------------ search (Trie)
@app.get("/api/search")
def search():
    toks = request.args.get("q", "").strip().lower().split()
    if not toks:
        return {"results": []}
    ids = TRIE.search(toks[0], 60)                              # prefix search in the Trie
    out = []
    for u in ids:
        words = [DB["users"][u]["first_name"].lower(), DB["users"][u]["last_name"].lower(), u]
        if all(any(w.startswith(t) for w in words) for t in toks[1:]):
            out.append(brief(u))
    return {"results": out[:8]}


# ------------------------------------------------------------------ network
@app.get("/api/network")
@login_required
def network():
    m = me()
    sugg = []
    for u, score, mc in GRAPH.suggestions(m, 12):
        sugg.append({**brief(u), "score": score, "mutual": mc})
    return {
        "connections": [brief(u) for u in sorted(GRAPH.adj[m])],
        "incoming": [brief(u) for u in GRAPH.incoming(m)],
        "outgoing": [brief(u) for u in GRAPH.outgoing(m)],
        "suggestions": sugg,
        "groups": [[brief(u) for u in g] for g in GRAPH.communities() if m in g],
    }


@app.post("/api/connect/<action>/<username>")
@login_required
def connect(action, username):
    m, u = me(), username.lower()
    if u not in DB["users"]:
        raise ApiError("User not found", 404)
    with LOCK:
        if action == "send":
            GRAPH.send_request(m, u)
            notify(u, "request", f"{my_name()} sent you a connection request")
        elif action == "accept":
            GRAPH.accept(m, u)
            notify(u, "info", f"{my_name()} accepted your connection request")
        elif action == "reject":
            GRAPH.reject(m, u)
        elif action == "cancel":
            GRAPH.cancel(m, u)
        elif action == "remove":
            GRAPH.remove_connection(m, u)
        else:
            raise ApiError("Unknown action")
        save_db()
    return {"ok": True, "relation": relation(m, u)}


@app.post("/api/message")
@login_required
def message():
    d = request.get_json(force=True)
    to, text = d.get("to", "").lower(), d.get("text", "").strip()
    if not text:
        raise ApiError("Type a message")
    if not GRAPH.are_connected(me(), to):
        raise ApiError("You can message only your connections")
    notify(to, "message", f"{my_name()}: {text[:300]}")
    save_db()
    return {"ok": True}


# ------------------------------------------------------------------ posts
def find_post(pid):
    for p in DB["posts"]:
        if p["id"] == pid:
            return p
    raise ApiError("Post not found", 404)


def decorate(p, m):
    return {"id": p["id"], "text": p["text"], "file": p["file"], "ts": p["ts"],
            "author": brief(p["author"]), "likes": len(p["likes"]), "liked": m in p["likes"],
            "likers": [brief(x)["name"] for x in p["likes"][-20:]],
            "comments": [{**c, "user": brief(c["user"]),
                          "replies": [{**r, "user": brief(r["user"])} for r in c["replies"]]}
                         for c in p["comments"]],
            "saved": p["id"] in DB["saved"].get(m, []), "mine": p["author"] == m}


def posts_by_user():
    by = {}
    for p in DB["posts"]:
        by.setdefault(p["author"], []).append(p)
    for lst in by.values():
        lst.sort(key=lambda p: -p["ts"])
    return by


@app.post("/api/posts")
@login_required
def create_post():
    text = request.form.get("text", "").strip()
    fl = save_upload(request.files.get("file"))
    if not text and not fl:
        raise ApiError("Write something or attach a file")
    p = {"id": uuid.uuid4().hex[:10], "author": me(), "text": text[:3000], "file": fl,
         "ts": time.time(), "likes": [], "comments": []}
    DB["posts"].append(p)
    for c in GRAPH.neighbors(me()):
        notify(c, "post", f"{my_name()} shared a new post", post=p["id"])
    save_db()
    return decorate(p, me())


@app.get("/api/feed")
@login_required
def feed():
    by = posts_by_user()
    lists = [by.get(u, []) for u in (GRAPH.neighbors(me()) | {me()})]
    merged = merge_feeds(lists, int(request.args.get("limit", 50)))   # max-heap merge
    return {"posts": [decorate(p, me()) for p in merged]}


@app.get("/api/trending")
@login_required
def trending():
    pool = [p for u in (GRAPH.neighbors(me()) | {me()}) for p in posts_by_user().get(u, [])]
    return {"posts": [decorate(p, me()) for p in top_k_trending(pool, 5)]}


@app.get("/api/posts/user/<username>")
@login_required
def user_posts(username):
    return {"posts": [decorate(p, me()) for p in posts_by_user().get(username.lower(), [])]}


@app.delete("/api/posts/<pid>")
@login_required
def delete_post(pid):
    p = find_post(pid)
    if p["author"] != me():
        raise ApiError("You can delete only your own posts", 403)
    DB["posts"].remove(p)
    save_db()
    return {"ok": True}


@app.post("/api/posts/<pid>/like")
@login_required
def like(pid):
    p = find_post(pid)
    if me() in p["likes"]:
        p["likes"].remove(me())
    else:
        p["likes"].append(me())
        notify(p["author"], "like", f"{my_name()} liked your post", post=pid)
    save_db()
    return decorate(p, me())


@app.post("/api/posts/<pid>/comment")
@login_required
def comment(pid):
    p, d = find_post(pid), request.get_json(force=True)
    text = d.get("text", "").strip()
    if not text:
        raise ApiError("Write a comment")
    item = {"id": uuid.uuid4().hex[:8], "user": me(), "text": text[:500], "ts": time.time()}
    if d.get("reply_to"):
        parent = next((c for c in p["comments"] if c["id"] == d["reply_to"]), None)
        if not parent:
            raise ApiError("Comment not found", 404)
        parent["replies"].append(item)
        notify(parent["user"], "comment", f"{my_name()} replied to your comment", post=pid)
    else:
        item["replies"] = []
        p["comments"].append(item)
        notify(p["author"], "comment", f"{my_name()} commented on your post", post=pid)
    save_db()
    return decorate(p, me())


@app.post("/api/posts/<pid>/save")
@login_required
def save_post(pid):
    find_post(pid)
    s = DB["saved"].setdefault(me(), [])
    s.remove(pid) if pid in s else s.append(pid)
    save_db()
    return {"saved": pid in s}


@app.post("/api/posts/<pid>/share")
@login_required
def share_post(pid):
    p, d = find_post(pid), request.get_json(force=True)
    to = d.get("to", "").lower()
    if not GRAPH.are_connected(me(), to):
        raise ApiError("You can send posts only to your connections")
    snippet = (p["text"] or (p["file"] or {}).get("name", "a file"))[:120]
    notify(to, "share", f"{my_name()} sent you a post by {brief(p['author'])['name']}: {snippet}", post=pid)
    save_db()
    return {"ok": True}


# ------------------------------------------------------------------ videos
@app.post("/api/videos")
@login_required
def upload_video():
    v = save_upload(request.files.get("video"))
    if not v or v["kind"] != "video":
        raise ApiError("Choose a video file (mp4, webm, mov)")
    item = {"id": uuid.uuid4().hex[:10], "author": me(), "title": request.form.get("title", "").strip() or v["name"],
            "desc": request.form.get("desc", "").strip()[:500], "video": v,
            "attach": save_upload(request.files.get("attach")), "ts": time.time()}
    DB["videos"].append(item)
    for c in GRAPH.neighbors(me()):
        notify(c, "post", f"{my_name()} uploaded a video: {item['title']}")
    save_db()
    return {"ok": True}


@app.get("/api/videos")
@login_required
def list_videos():
    ok = GRAPH.neighbors(me()) | {me()}
    vids = sorted((v for v in DB["videos"] if v["author"] in ok), key=lambda v: -v["ts"])
    return {"videos": [{**v, "author": brief(v["author"]), "mine": v["author"] == me()} for v in vids]}


@app.delete("/api/videos/<vid>")
@login_required
def del_video(vid):
    v = next((x for x in DB["videos"] if x["id"] == vid and x["author"] == me()), None)
    if not v:
        raise ApiError("Not found", 404)
    DB["videos"].remove(v)
    save_db()
    return {"ok": True}


# ------------------------------------------------------------------ jobs
@app.post("/api/jobs")
@login_required
def post_job():
    d = request.get_json(force=True)
    for k in ("title", "company", "description"):
        if not d.get(k, "").strip():
            raise ApiError(f"Please fill: {k}")
    DB["jobs"].append({"id": uuid.uuid4().hex[:10], "poster": me(), "title": d["title"].strip(),
                       "company": d["company"].strip(), "location": d.get("location", "").strip(),
                       "type": d.get("type", "Full-time"), "description": d["description"].strip()[:2000],
                       "ts": time.time(), "applications": []})
    save_db()
    return {"ok": True}


@app.get("/api/jobs")
@login_required
def list_jobs():
    out = []
    for j in sorted(DB["jobs"], key=lambda j: -j["ts"]):
        mine = j["poster"] == me()
        out.append({"id": j["id"], "title": j["title"], "company": j["company"], "location": j["location"],
                    "type": j["type"], "description": j["description"], "ts": j["ts"],
                    "poster": brief(j["poster"]), "mine": mine, "count": len(j["applications"]),
                    "applied": any(a["user"] == me() for a in j["applications"]),
                    "applicants": [{**a, "user": brief(a["user"])} for a in j["applications"]] if mine else []})
    return {"jobs": out}


@app.post("/api/jobs/<jid>/apply")
@login_required
def apply_job(jid):
    j = next((x for x in DB["jobs"] if x["id"] == jid), None)
    if not j:
        raise ApiError("Job not found", 404)
    if j["poster"] == me():
        raise ApiError("You cannot apply to your own posting")
    if any(a["user"] == me() for a in j["applications"]):
        raise ApiError("You already applied")
    resume = save_upload(request.files.get("resume"))
    j["applications"].append({"user": me(), "note": request.form.get("note", "").strip()[:500],
                              "resume": resume, "ts": time.time()})
    notify(j["poster"], "job", f"{my_name()} applied for {j['title']}")
    save_db()
    return {"ok": True}


@app.delete("/api/jobs/<jid>")
@login_required
def del_job(jid):
    j = next((x for x in DB["jobs"] if x["id"] == jid and x["poster"] == me()), None)
    if not j:
        raise ApiError("Not found", 404)
    DB["jobs"].remove(j)
    save_db()
    return {"ok": True}


# ------------------------------------------------------------------ docs (personal folder)
@app.get("/api/docs")
@login_required
def list_docs():
    saved = [decorate(find_post(i), me()) for i in DB["saved"].get(me(), [])
             if any(p["id"] == i for p in DB["posts"])]
    return {"files": sorted(DB["docs"].get(me(), []), key=lambda d: -d["ts"]), "saved": saved}


@app.post("/api/docs")
@login_required
def add_doc():
    fl = save_upload(request.files.get("file"))
    if not fl:
        raise ApiError("Choose a file")
    DB["docs"].setdefault(me(), []).append({"id": uuid.uuid4().hex[:8], "title": request.form.get("title", "").strip() or fl["name"],
                                            "url": fl["url"], "name": fl["name"], "kind": fl["kind"], "ts": time.time()})
    save_db()
    return {"ok": True}


@app.delete("/api/docs/<did>")
@login_required
def del_doc(did):
    DB["docs"][me()] = [d for d in DB["docs"].get(me(), []) if d["id"] != did]
    save_db()
    return {"ok": True}


# ------------------------------------------------------------------ notifications
@app.get("/api/notifications")
@login_required
def notifications():
    q = NOTIFS[me()]
    items = []
    for n in q.newest_first()[:80]:
        n = dict(n)
        n["name"] = brief(n["from"])["name"] if n.get("from") else ""
        n["photo"] = brief(n["from"])["photo"] if n.get("from") else ""
        if n["type"] == "request":
            n["pending"] = GRAPH.has_pending(n["from"], me())   # show Accept/Reject only while pending
        if n["type"] == "call":
            n["live"] = n.get("room") in ROOMS
        items.append(n)
    return {"items": items, "unread": q.unread()}


@app.get("/api/notifications/count")
@login_required
def notif_count():
    return {"unread": NOTIFS[me()].unread()}


@app.post("/api/notifications/read")
@login_required
def notif_read():
    NOTIFS[me()].mark_all_read()
    save_db()
    return {"ok": True}


# ------------------------------------------------------------------ collaborative PPT
def ppt_for(pid, need_member=True):
    p = next((x for x in DB["ppts"] if x["id"] == pid), None)
    if not p:
        raise ApiError("Presentation not found", 404)
    if need_member and me() not in p["members"]:
        raise ApiError("You are not part of this presentation", 403)
    return p


def ppt_out(p, full=True):
    out = {"id": p["id"], "title": p["title"], "owner": brief(p["owner"]), "version": p["version"],
           "members": [brief(u) for u in p["members"]], "updated": p["updated"],
           "last_editor": brief(p["last_editor"])["name"], "mine": p["owner"] == me()}
    if full:
        out["slides"] = p["slides"]
        if p["owner"] == me():
            out["share_link"] = f"{request.host_url}?ppt={p['id']}&token={p['token']}"
    return out


@app.get("/api/ppts")
@login_required
def ppts():
    return {"ppts": [ppt_out(p, False) for p in sorted(DB["ppts"], key=lambda p: -p["updated"]) if me() in p["members"]]}


@app.post("/api/ppts")
@login_required
def ppt_new():
    d = request.get_json(force=True)
    p = {"id": uuid.uuid4().hex[:8], "title": d.get("title", "").strip() or "Untitled deck", "owner": me(),
         "members": [me()], "token": secrets.token_urlsafe(8), "version": 1, "updated": time.time(),
         "last_editor": me(), "slides": [{"title": "Title slide", "body": "Click to edit", "bg": "#00838f"}]}
    DB["ppts"].append(p)
    save_db()
    return ppt_out(p)


@app.get("/api/ppts/<pid>")
@login_required
def ppt_get(pid):
    p = ppt_for(pid)
    if request.args.get("since") == str(p["version"]):
        return {"unchanged": True, "version": p["version"], "members": [brief(u) for u in p["members"]]}
    return ppt_out(p)


@app.post("/api/ppts/<pid>")
@login_required
def ppt_save(pid):
    p, d = ppt_for(pid), request.get_json(force=True)
    slides = d.get("slides")
    if not isinstance(slides, list) or not slides or len(slides) > 100:
        raise ApiError("Invalid slides")
    p["slides"] = [{"title": str(s.get("title", ""))[:200], "body": str(s.get("body", ""))[:2000],
                    "bg": s.get("bg", "#00838f") if re.fullmatch(r"#[0-9a-fA-F]{6}", s.get("bg", "")) else "#00838f"}
                   for s in slides]
    if d.get("title"):
        p["title"] = d["title"][:100]
    p["version"] += 1
    p["updated"], p["last_editor"] = time.time(), me()
    save_db()
    return {"version": p["version"]}


@app.post("/api/ppts/<pid>/join")
@login_required
def ppt_join(pid):
    p = ppt_for(pid, need_member=False)
    if not hmac.compare_digest(str(request.get_json(force=True).get("token", "")), p["token"]):
        raise ApiError("Invalid or expired share link", 403)
    if me() not in p["members"]:
        p["members"].append(me())
        notify(p["owner"], "info", f"{my_name()} joined your presentation \"{p['title']}\"")
        save_db()
    return ppt_out(p)


@app.delete("/api/ppts/<pid>")
@login_required
def ppt_del(pid):
    p = ppt_for(pid)
    if p["owner"] != me():
        raise ApiError("Only the owner can delete", 403)
    DB["ppts"].remove(p)
    save_db()
    return {"ok": True}


# ------------------------------------------------------------------ group video call (WebRTC signalling)
# Media flows peer-to-peer (mesh). Flask only relays small signalling messages.
def room_or_404(rid):
    r = ROOMS.get(rid.upper())
    if not r:
        raise ApiError("Meeting not found or already ended", 404)
    return r


@app.post("/api/call/create")
@login_required
def call_create():
    rid = secrets.token_hex(3).upper()
    ROOMS[rid] = {"host": me(), "peers": {}, "chat": []}
    for c in GRAPH.neighbors(me()):
        notify(c, "call", f"{my_name()} started a group call", room=rid)
    save_db()
    return {"room": rid}


@app.post("/api/call/<rid>/join")
@login_required
def call_join(rid):
    r = room_or_404(rid)
    existing = [p for p in r["peers"] if p != me()]
    r["peers"][me()] = {"q": deque(), "seen": time.time()}
    return {"peers": existing, "host": brief(r["host"])["name"]}


@app.post("/api/call/<rid>/signal")
@login_required
def call_signal(rid):
    r, d = room_or_404(rid), request.get_json(force=True)
    to = d.get("to")
    if to in r["peers"]:
        r["peers"][to]["q"].append({"from": me(), "data": d.get("data")})
    return {"ok": True}


@app.get("/api/call/<rid>/poll")
@login_required
def call_poll(rid):
    r = room_or_404(rid)
    now = time.time()
    if me() not in r["peers"]:
        raise ApiError("You left this meeting", 410)
    r["peers"][me()]["seen"] = now
    for u in [u for u, p in r["peers"].items() if now - p["seen"] > 15]:   # drop dead peers
        del r["peers"][u]
    q = r["peers"][me()]["q"]
    sigs = [q.popleft() for _ in range(len(q))]
    since = int(request.args.get("chat", 0))
    return {"signals": sigs, "peers": [{"username": u, "name": brief(u)["name"]} for u in r["peers"] if u != me()],
            "chat": r["chat"][since:], "chat_total": len(r["chat"])}


@app.post("/api/call/<rid>/chat")
@login_required
def call_chat(rid):
    r = room_or_404(rid)
    text = request.get_json(force=True).get("text", "").strip()[:500]
    if text:
        r["chat"].append({"user": me(), "name": my_name(), "text": text, "ts": time.time()})
    return {"ok": True}


@app.post("/api/call/<rid>/leave")
def call_leave(rid):
    r = ROOMS.get(rid.upper())
    if r and me() in r["peers"]:
        del r["peers"][me()]
        if not r["peers"]:
            ROOMS.pop(rid.upper(), None)
    return {"ok": True}


load_db()

if __name__ == "__main__":
    # camera / mic / screen-share work on http://localhost or https:// only
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)