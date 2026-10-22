#!/usr/bin/env python3
"""
EC Weather Remarks - production run: Qwen3.8-27B at 4 MiP, with benchmark v9's settings.

Runs on the local Windows machine. Scans are read from the external drive, resized and PNG-encoded
here, and sent one per request to a vLLM OpenAI-compatible server on the rented H200 (default
http://127.0.0.1:8000, through an SSH tunnel). Nothing runs on the H200 except vLLM; all state is here.

Matches v9: load_resized, build_prompt, load_stations, clean and says_none come from benchmark.py,
the pixel budget is 4 MiP, and decoding is greedy with seed 0. The image goes as a lossless PNG,
greyscale (L) when its three channels are identical and RGB otherwise. vLLM converts it to RGB on
load, so the model gets the same pixels as v9's PIL image. Start the server with v9's engine
settings and without a reasoning parser:
  vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' --mm-processor-cache-gb 0 --seed 0 --port 8000

Checkpoint = the output JSONL: one line per finished file, appended and flushed to disk at once.
On startup every source_path already recorded as ok or decode_error is skipped and everything else
is processed, so the script can be killed at any moment and restarted with the same arguments.
A partial last line left by a crash is ignored. Every 10 minutes the JSONL is copied atomically to
--backup-dir.

  status           meaning                                              next
  ok               the model answered                                   done
  decode_error     the image won't open or decode                       done, never retried
  transient_error  timeout, connection error, HTTP 5xx                  retried after the main pass (3 passes)
  read_error       the file couldn't be read from the drive             retried like transient_error
  request_error    any other HTTP error: the server rejected the request  next start
  parse_error      empty reply, reasoning_content, or unreadable reply  next start
Only ok rows have output and has_remarks, so a failed file never reads as NO_REMARKS. Every status
except ok and decode_error is processed again on the next start.

20 transient errors in a row mean the server or the tunnel is down: no new files are sent, /health
is polled every 30 s, and sending resumes when it answers. 20 read errors in a row mean the scan
drive is gone: the run stops, and a restart with the same arguments continues it.

  python Benchmark/production.py --file-list Benchmark/gt150_paths.txt \
      --inventory-2014 inventory_2014.csv --inventory-2022 inventory_2022.csv \
      --out runs/gt150.jsonl --backup-dir D:/ec_backup
"""

import argparse
import base64
import io
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from benchmark import MIB, build_prompt, clean, load_resized, load_stations, says_none

MODEL = "Qwen/Qwen3.8-27B"
BUDGET_PX = 4 * MIB
SAMPLING = {"temperature": 0, "max_tokens": 512, "seed": 0, "repetition_penalty": 1.0,
            "presence_penalty": 0, "frequency_penalty": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}   # what the organizer picked up

DONE = {"ok", "decode_error"}               # skipped on restart
RETRY = {"transient_error", "read_error"}   # retried after the main pass
RETRY_PASSES = 3
OUTAGE_AFTER = 20        # transient errors in a row: server or tunnel down, pause
DRIVE_GONE_AFTER = 20    # read errors in a row: scan drive gone, stop
HEALTH_EVERY = 30        # seconds between /health polls while paused
PROGRESS_EVERY = 5 * 60
BACKUP_EVERY = 10 * 60
QUEUE_SIZE = 32          # encoded images waiting for a free request slot

FIELDS = ["source_path", "filename", "climate_id", "station", "province", "year", "month",
          "native_wh", "sent_wh", "payload_bytes", "prompt_tokens", "output_tokens", "finish_reason",
          "output", "has_remarks", "status", "error", "utc"]

log = logging.getLogger("production")
# No proxy: a system proxy must never see requests meant for the tunnel on 127.0.0.1.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def read_jsonl(path):
    """Yield (byte offset, record) for each line of a JSONL file. A line that doesn't parse (the
    partial last line a crash leaves) is skipped. Splits on b"\\n" only, never on U+2028 and the like."""
    with open(path, "rb") as f:
        offset = 0
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                if line.strip():
                    log.warning("%s: skipping unreadable line at byte %d (%d bytes)", path, offset, len(line))
            else:
                yield offset, rec
            offset += len(line)


# --------------------------------------------------------------------------- images (worker processes)

def png_bytes(img):
    """Lossless PNG of an RGB image, as L when R, G and B are identical. vLLM's convert("RGB") copies
    an L value to all three channels, so it gets back exactly these pixels."""
    r, g, b = img.split()
    if r.tobytes() == g.tobytes() == b.tobytes():
        img = r
    buf = io.BytesIO()
    img.save(buf, format="PNG", icc_profile=None)   # an RGB colour profile would be invalid in an L PNG
    return buf.getvalue()


def prepare(path, prompt):
    """Read, resize and encode one scan, and build its request body."""
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        return {"status": "read_error", "error": f"{type(e).__name__}: {e}"}
    try:  # the bytes are in memory now, so any failure is the file's content, not the drive
        img, native = load_resized(io.BytesIO(data), BUDGET_PX)
        png = png_bytes(img)
    except Exception as e:
        return {"status": "decode_error", "error": f"{type(e).__name__}: {e}"}
    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": prompt}]}],
            **SAMPLING}
    return {"status": "ready", "native_wh": list(native), "sent_wh": list(img.size),
            "payload_bytes": len(png), "body": json.dumps(body).encode("utf-8")}


# --------------------------------------------------------------------------- server

def post(server, body, timeout):
    """One chat completion. Returns (status, record fields)."""
    req = urllib.request.Request(server + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with OPENER.open(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read()[:300].decode("utf-8", "replace")
        except Exception:
            detail = ""
        status = "transient_error" if e.code >= 500 or e.code in (408, 429) else "request_error"
        return status, {"error": f"HTTP {e.code}: {detail}"}
    except Exception as e:  # refused or reset connection, timeout: the server or the tunnel is down
        return "transient_error", {"error": f"{type(e).__name__}: {e}"}
    try:
        resp = json.loads(raw)
        choice, usage = resp["choices"][0], resp["usage"]
        msg = choice["message"]
        fields = {"prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                  "finish_reason": choice["finish_reason"]}
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        output = clean(msg.get("content") or "")
    except (ValueError, LookupError, TypeError, AttributeError) as e:
        return "parse_error", {"error": f"unreadable response ({type(e).__name__}: {e}): {raw[:300]!r}"}
    if reasoning:
        return "parse_error", {**fields, "error": f"reasoning_content in the response ({len(reasoning)} chars), "
                                                  f"content {output[:200]!r}"}
    if not output:
        return "parse_error", {**fields, "error": f"empty content ({fields['output_tokens']} output tokens)"}
    return "ok", {**fields, "output": output, "has_remarks": not says_none(output)}


def healthy(server):
    try:
        with OPENER.open(server + "/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def check_server(server):
    try:
        with OPENER.open(server + "/v1/models", timeout=30) as r:
            cards = json.loads(r.read())["data"]
    except Exception as e:
        raise SystemExit(f"No vLLM server at {server} ({type(e).__name__}: {e}). Is the SSH tunnel up?")
    card = next((c for c in cards if c.get("id") == MODEL), None)
    if card is None:
        raise SystemExit(f"{server} serves {[c.get('id') for c in cards]}, not {MODEL}")
    log.info("server %s serves %s, max_model_len %s", server, MODEL, card.get("max_model_len"))


# --------------------------------------------------------------------------- inputs

def list_inputs(args):
    """Scan paths, absolute and normalised (they are the checkpoint key), without duplicates."""
    if args.root:
        if not args.root.is_dir():
            raise SystemExit(f"--root not found: {args.root}")

        def fail(e):
            raise e
        found = [os.path.join(d, f) for d, _, files in os.walk(args.root, onerror=fail) for f in files]
        paths = sorted(p for p in found if os.path.splitext(p)[1].lower() in IMAGE_EXTS)
        if len(paths) < len(found):
            print(f"Skipping {len(found) - len(paths)} non-image files under {args.root}")
    else:
        lines = args.file_list.read_text(encoding="utf-8-sig").splitlines()
        paths = [s.strip().strip('"') for s in lines if s.strip()]
    return list(dict.fromkeys(os.path.normpath(os.path.abspath(p)) for p in paths))


def check_not_scan_drive(inputs, *outputs):
    """Never write to the scan drive: refuse outputs on a non-system drive that holds input scans."""
    system = os.environ.get("SystemDrive", "C:").upper()
    scan_drives = {os.path.splitdrive(p)[0].upper() for p in inputs} - {system, ""}
    for p in outputs:
        drive = os.path.splitdrive(os.path.abspath(p))[0].upper()
        if drive in scan_drives:
            raise SystemExit(f"{p} is on the scan drive {drive}. Put outputs and backups on another drive.")


def describe(path, stations):
    """A row's identity columns, parsed from the file name as build_prompt() does."""
    name = os.path.basename(path)
    parts = Path(name).stem.split("_")
    cid = parts[1] if len(parts) > 1 else None
    station, province = stations.get(cid, (None, None))
    return {"source_path": path, "filename": name, "climate_id": cid, "station": station, "province": province,
            "year": parts[2] if len(parts) > 2 else None, "month": parts[3] if len(parts) > 3 else None}


def open_checkpoint(path):
    """Append handle on the output. A partial last line is ended with a newline rather than deleted;
    readers skip it as unreadable."""
    f = open(path, "ab")
    if os.fstat(f.fileno()).st_size:
        with open(path, "rb") as r:
            r.seek(-1, os.SEEK_END)
            if r.read(1) != b"\n":
                f.write(b"\n")
                f.flush()
                log.warning("%s ended in a partial line (a crash mid-write); it is ignored", path)
    return f


def hms(seconds):
    return str(timedelta(seconds=round(seconds)))


# --------------------------------------------------------------------------- run

class Run:
    """One session: the open checkpoint, the counters, and the pause and stop switches."""

    def __init__(self, args, stations, total, done, pool):
        self.args, self.stations, self.total, self.pool = args, stations, total, pool
        self.done = done                  # input paths recorded as ok or decode_error
        self.done_at_start = len(done)
        self.latest = {}                  # source_path -> status written this session
        self.counts = Counter()           # statuses written this session
        self.length = 0                   # finish_reason == "length" this session
        self.lock = threading.Lock()
        self.out = open_checkpoint(args.out)
        self.fails_in_row = 0             # transient errors in a row
        self.sending = threading.Event()  # cleared while the server is down
        self.sending.set()
        self.stop = threading.Event()
        self.t0 = self.last_progress = self.last_backup = self.last_poll = self.down_since = time.monotonic()
        self.last_count = 0

    def record(self, base, status, **fields):
        rec = dict.fromkeys(FIELDS)
        rec.update(base, **fields)
        rec.update(status=status, utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
        with self.lock:
            self.out.write(line)
            self.out.flush()
            os.fsync(self.out.fileno())
            self.latest[rec["source_path"]] = status
            self.counts[status] += 1
            self.length += rec["finish_reason"] == "length"
            if status in DONE:
                self.done.add(rec["source_path"])

    # threads ---------------------------------------------------------------

    def guard(self, fn, *args):
        try:
            fn(*args)
        except Exception:
            if not self.stop.is_set():
                log.exception("%s failed; stopping. Restart with the same arguments to continue.", fn.__name__)
                self.stop.set()

    def produce(self, files, q):
        """Decode, resize and encode in the process pool, in input order, into the bounded queue."""
        pending, files, read_fails = deque(), iter(files), 0
        while not self.stop.is_set():
            while len(pending) < 2 * self.args.workers and (path := next(files, None)):
                base = describe(path, self.stations)
                prompt = build_prompt(base["filename"], self.stations)
                pending.append((base, self.pool.submit(prepare, path, prompt)))
            if not pending:
                break
            base, fut = pending.popleft()
            prep = fut.result()   # raises only if the pool itself broke, which stops the run
            status = prep.pop("status")
            if status == "ready":
                self.put(q, (base, prep))
            else:
                self.record(base, status, **prep)
            read_fails = read_fails + 1 if status == "read_error" else 0
            if read_fails >= DRIVE_GONE_AFTER:
                log.error("%d read errors in a row: is the scan drive connected? Stopping. "
                          "Restart with the same arguments once it is back.", read_fails)
                self.stop.set()
        for _ in range(self.args.concurrency):
            self.put(q, None)   # one end-of-pass marker per sender

    def send(self, q):
        while (item := self.take(q)) is not None:
            base, prep = item
            status, fields = post(self.args.server, prep.pop("body"), self.args.timeout)
            self.note(status)
            self.record(base, status, **prep, **fields)

    def put(self, q, item):
        while not self.stop.is_set():
            try:
                q.put(item, timeout=1)
                return
            except queue.Full:
                pass

    def take(self, q):
        """The next encoded image; None at the end of the pass or when the run stops. Waits out outages."""
        while not self.stop.is_set():
            if not self.sending.wait(1):
                continue
            try:
                return q.get(timeout=1)
            except queue.Empty:
                pass
        return None

    def note(self, status):
        """Count transient errors in a row; at OUTAGE_AFTER, stop sending until /health answers."""
        with self.lock:
            self.fails_in_row = self.fails_in_row + 1 if status == "transient_error" else 0
            if self.fails_in_row >= OUTAGE_AFTER and self.sending.is_set():
                self.sending.clear()
                self.down_since = self.last_poll = time.monotonic()
                log.warning("%d request failures in a row: server or tunnel down. Pausing; polling /health "
                            "every %d s", self.fails_in_row, HEALTH_EVERY)

    # main thread -----------------------------------------------------------

    def run_pass(self, files):
        q = queue.Queue(maxsize=QUEUE_SIZE)
        threads = [threading.Thread(target=self.guard, args=(self.produce, files, q), daemon=True)]
        threads += [threading.Thread(target=self.guard, args=(self.send, q), daemon=True)
                    for _ in range(self.args.concurrency)]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            time.sleep(1)   # sleep, unlike join, lets Ctrl+C through on Windows
            self.tick()

    def run(self, todo):
        self.run_pass(todo)
        for n in range(1, RETRY_PASSES + 1):
            retry = [p for p in todo if self.latest.get(p) in RETRY]
            if self.stop.is_set() or not retry:
                break
            log.info("retry pass %d/%d: %d files with %s", n, RETRY_PASSES, len(retry), " or ".join(sorted(RETRY)))
            self.run_pass(retry)

    def tick(self):
        now = time.monotonic()
        if not self.sending.is_set() and not self.stop.is_set() and now - self.last_poll >= HEALTH_EVERY:
            self.last_poll = now
            if healthy(self.args.server):
                with self.lock:
                    self.fails_in_row = 0
                self.sending.set()
                log.info("server answered /health after %s down: resuming", hms(now - self.down_since))
        if now - self.last_progress >= PROGRESS_EVERY:
            self.progress()
        if now - self.last_backup >= BACKUP_EVERY:
            self.backup()

    def progress(self):
        now = time.monotonic()
        with self.lock:
            done, n, counts, length = len(self.done), sum(self.counts.values()), sorted(self.counts.items()), self.length
        elapsed = max(now - self.t0, 1e-9)
        recent = (n - self.last_count) / max(now - self.last_progress, 1e-9)
        done_rate = (done - self.done_at_start) / elapsed
        eta = hms((self.total - done) / done_rate) if done_rate else "unknown"
        statuses = ", ".join(f"{k} {v:,}" for k, v in counts) or "none yet"
        paused = "" if self.sending.is_set() else f" | PAUSED: server down for {hms(now - self.down_since)}"
        log.info("progress: %s/%s done (%.1f%%) | this session %s files in %s, %.2f files/s (last interval %.2f) | "
                 "%s | finish_reason=length %s | ETA %s%s",
                 f"{done:,}", f"{self.total:,}", 100 * done / max(self.total, 1), f"{n:,}", hms(elapsed),
                 n / elapsed, recent, statuses, f"{length:,}", eta, paused)
        self.last_progress, self.last_count = now, n

    def backup(self):
        """Copy the checkpoint up to its last complete line to --backup-dir: temp file, then rename."""
        self.last_backup = time.monotonic()
        with self.lock:
            size = os.fstat(self.out.fileno()).st_size   # every record is flushed, so this ends on a newline
        dst = self.args.backup_dir / self.args.out.name
        tmp = dst.with_name(dst.name + ".tmp")
        try:
            with open(self.args.out, "rb") as src, open(tmp, "wb") as f:
                while size > 0 and (chunk := src.read(min(size, 1 << 20))):
                    f.write(chunk)
                    size -= len(chunk)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dst)
        except OSError as e:
            log.warning("backup to %s failed (will try again): %s", dst, e)


def main(argv=None):
    ap = argparse.ArgumentParser(description="EC Weather Remarks production run (Qwen3.8-27B, 4 MiP, v9 settings)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="Folder of scans, walked recursively in sorted order")
    src.add_argument("--file-list", type=Path, help="Text file with one scan path per line")
    ap.add_argument("--inventory-2014", type=Path, required=True)
    ap.add_argument("--inventory-2022", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL, which is also the checkpoint")
    ap.add_argument("--backup-dir", type=Path, required=True, help="The JSONL is copied here every 10 minutes")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=64, help="Requests in flight")
    ap.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2)),
                    help="Processes that decode, resize and encode")
    ap.add_argument("--timeout", type=float, default=300, help="Seconds to wait for one reply")
    args = ap.parse_args(argv)
    args.server = args.server.rstrip("/")
    for inv in (args.inventory_2014, args.inventory_2022):
        if not inv.is_file():
            ap.error(f"inventory not found: {inv}")

    inputs = list_inputs(args)
    log_path = args.out.with_suffix(".log")
    check_not_scan_drive(inputs, args.out, log_path, args.backup_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    log.info("start: %s", " ".join(sys.argv))

    stations = load_stations(args.inventory_2022, args.inventory_2014)   # 2022 first, as in v9
    unmatched = sum(describe(p, stations)["station"] is None for p in inputs)
    log.info("%s input files, %s stations loaded, %s files without a station match",
             f"{len(inputs):,}", f"{len(stations):,}", f"{unmatched:,}")
    if not inputs:
        raise SystemExit("No input files.")

    recorded, rows = set(), 0
    if args.out.exists():
        for _, r in read_jsonl(args.out):
            rows += 1
            if r.get("status") in DONE:
                recorded.add(r.get("source_path"))
    done = recorded.intersection(inputs)
    todo = [p for p in inputs if p not in done]
    log.info("%s has %s rows: %s input files already done, %s to process",
             args.out, f"{rows:,}", f"{len(done):,}", f"{len(todo):,}")
    if len(recorded) > len(done):
        log.warning("%s files recorded as done are not in this input list (another --root, file list or drive letter?)",
                    f"{len(recorded) - len(done):,}")
    if not todo:
        log.info("nothing to do")
        return 0

    check_server(args.server)
    # Ctrl+C reaches every process on the console; only this one should handle it.
    pool = ProcessPoolExecutor(args.workers, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN))
    run = Run(args, stations, len(inputs), done, pool)
    interrupted = False
    try:
        run.run(todo)
    except KeyboardInterrupt:
        interrupted = True
        run.stop.set()
        log.warning("interrupted. Restart with the same arguments to continue.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        run.progress()
        run.backup()
    if interrupted:
        run.lock.acquire()   # no thread may be mid-write while the interpreter exits
        return 130
    left = Counter(run.latest.get(p, "not processed") for p in todo if p not in run.done)
    if left:
        log.warning("%s files not done (%s). Rerun with the same arguments to process them again.",
                    f"{sum(left.values()):,}", ", ".join(f"{k} {v:,}" for k, v in sorted(left.items())))
        return 1
    log.info("all %s files done", f"{len(inputs):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
