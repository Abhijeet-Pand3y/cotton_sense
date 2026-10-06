#!/usr/bin/env python
"""Pull colour (and optionally depth) frames out of a RealSense ROS2 .db3 bag.

The bag is a plain sqlite file: a `topics` table and a `messages` table of
CDR-serialised payloads. No ROS install is needed -- sensor_msgs/msg/Image has a
fixed layout, so the few fields we want can be read directly. Realsense writes
colour and depth on /device_0/sensor_1/Color_0/image/data and
/device_0/sensor_0/Depth_0/image/data.

    python scripts/extraction/extract_ros2_bag.py --db <bag.db3> --list
    python scripts/extraction/extract_ros2_bag.py --db <bag.db3> --video
    python scripts/extraction/extract_ros2_bag.py --db <bag.db3> --frames_dir frames --every 10
"""
import argparse, os, sqlite3, struct, subprocess
import numpy as np, cv2

COLOR = "/device_0/sensor_1/Color_0/image/data"
DEPTH = "/device_0/sensor_0/Depth_0/image/data"


class CDR:
    """Little-endian CDR reader: each primitive aligns to its own size, measured
    from the start of the body (i.e. after the 4-byte encapsulation header)."""

    def __init__(self, b):
        self.b = b; self.o = 4; self.base = 4

    def _al(self, n):
        r = (self.o - self.base) % n
        if r:
            self.o += n - r

    def u32(self):
        self._al(4); v = struct.unpack_from("<I", self.b, self.o)[0]; self.o += 4; return v

    def i32(self):
        self._al(4); v = struct.unpack_from("<i", self.b, self.o)[0]; self.o += 4; return v

    def u8(self):
        v = self.b[self.o]; self.o += 1; return v

    def st(self):
        n = self.u32(); v = self.b[self.o:self.o + n - 1].decode("utf8", "replace")
        self.o += n; return v


def parse_image(blob):
    d = CDR(bytes(blob))
    sec = d.i32(); nsec = d.u32(); d.st()          # header.stamp, header.frame_id
    h = d.u32(); w = d.u32(); enc = d.st(); d.u8(); step = d.u32()
    n = d.u32()
    buf = np.frombuffer(d.b, np.uint8, count=n, offset=d.o)
    return sec + nsec / 1e9, h, w, enc, step, buf


def to_bgr(h, w, enc, step, buf):
    if enc in ("rgb8", "bgr8"):
        a = buf.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if enc == "rgb8" else a.copy()
    if enc in ("mono16", "16UC1"):                  # depth: colourise for preview
        a = buf.view("<u2").reshape(h, step // 2)[:, :w]
        v = a.astype(np.float32)
        v = np.clip((v - np.percentile(v[v > 0], 2)) /
                    max(1, np.percentile(v[v > 0], 98) - np.percentile(v[v > 0], 2)), 0, 1)
        return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    raise SystemExit(f"unhandled encoding {enc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--topic", default=COLOR, help=f"default colour; depth is {DEPTH}")
    ap.add_argument("--list", action="store_true", help="list topics and exit")
    ap.add_argument("--video", nargs="?", const="AUTO", default=None,
                    help="write an mp4 here; bare --video puts "
                         "<bagname>_color.mp4 next to the .db3")
    ap.add_argument("--frames_dir", default=None, help="write JPGs here")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the bag")
    ap.add_argument("--end", type=float, default=-1.0)
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: source/every)")
    ap.add_argument("--crf", type=int, default=20)
    a = ap.parse_args()

    if a.video == "AUTO":
        stem = os.path.splitext(os.path.basename(a.db))[0]
        tag = "depth" if a.topic == DEPTH else "color"
        a.video = os.path.join(os.path.dirname(os.path.abspath(a.db)), f"{stem}_{tag}.mp4")

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    topics = {n: (i, t) for i, n, t in
              con.execute("SELECT id,name,type FROM topics")}
    if a.list:
        for n, (i, t) in sorted(topics.items(), key=lambda kv: kv[1][0]):
            cnt = con.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (i,)).fetchone()[0]
            if cnt:
                print(f"{i:4d} {cnt:8d}  {n}  ({t})")
        return
    if a.topic not in topics:
        raise SystemExit(f"topic {a.topic} not in bag; run --list")
    tid = topics[a.topic][0]

    n_msgs, tmin, tmax = con.execute(
        "SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM messages WHERE topic_id=?",
        (tid,)).fetchone()
    t0 = tmin
    src_fps = n_msgs / ((tmax - tmin) / 1e9)
    fps = a.fps or src_fps / a.every
    n_out = n_msgs // a.every
    print(f"{n_msgs} msgs at {src_fps:.1f} fps -> ~{n_out} frames at {fps:.1f} fps", flush=True)

    if a.frames_dir:
        os.makedirs(a.frames_dir, exist_ok=True)
    ff = None
    # ONE streaming cursor, not a SELECT per frame: payloads are ~1.2MB each and
    # the bag can be tens of GB, so per-id lookups become random reads -- which
    # on a Windows drive mounted into WSL are an order of magnitude slower than
    # reading the table in order.
    cur = con.execute("SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
                      (tid,))
    k = -1
    for j, (ts, blob) in enumerate(cur):
        t = (ts - t0) / 1e9
        if t < a.start or (a.end >= 0 and t > a.end):
            continue
        if j % a.every:
            continue
        k += 1
        _, h, w, enc, step, buf = parse_image(blob)
        bgr = to_bgr(h, w, enc, step, buf)
        if a.frames_dir:
            cv2.imwrite(os.path.join(a.frames_dir, f"frame_{k:06d}_t{t:.2f}s.jpg".replace(".", "p", 1)),
                        bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if a.video:
            if ff is None:
                ff = subprocess.Popen(
                    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                     "-c:v", "libx264", "-crf", str(a.crf), "-pix_fmt", "yuv420p", a.video],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ff.stdin.write(bgr.tobytes())
        if k % 500 == 0:
            print(f"  [{k}/~{n_out}] t={t:.1f}s", flush=True)
    if ff:
        ff.stdin.close(); ff.wait()
        print("wrote", a.video)
    if a.frames_dir:
        print("wrote", k + 1, "frames ->", a.frames_dir)


if __name__ == "__main__":
    main()
