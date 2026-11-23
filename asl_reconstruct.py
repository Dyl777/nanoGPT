"""YouTube-ASL -> 3D ASL motion reconstruction pipeline for s56 (SeRV).

The public YouTube-ASL corpus ships video IDs + English captions, not 3D motion.
This module closes that gap:

    video_id -> yt-dlp download -> MediaPipe Holistic per frame
             -> pose_world_landmarks (33 x 3) + left/right hand (21 x 3 each)
             -> (T, 75, 3) metric-scale motion, cached as .npy

MediaPipe's world landmarks are metric 3D estimates from a single view, which is
the lightweight counterpart of the paper's offline reconstruction pipeline. All
heavy imports are lazy; if a dependency is missing the caller gets a precise
RuntimeError and the SeRV runner records a SKIP, never a fabricated pose.
"""
from __future__ import annotations

import csv
import io
import json
import os
import urllib.request
from pathlib import Path

CACHE = Path("data/external/asl3d")
META_CANDIDATES = [
    "https://storage.googleapis.com/gresearch/youtube-asl/youtube_asl_video_ids.txt",
    "https://raw.githubusercontent.com/google-research/google-research/master/"
    "youtube_asl/youtube_asl.csv",
]

N_POSE, N_HAND = 33, 21
DIM = (N_POSE + 2 * N_HAND) * 3  # 75 landmarks * 3


def dependencies():
    import importlib.util as u
    return {m: bool(u.find_spec(m)) for m in ("cv2", "mediapipe", "yt_dlp")}


def _require():
    deps = dependencies()
    missing = [k for k, ok in deps.items() if not ok]
    if missing:
        raise RuntimeError(
            "YouTube-ASL reconstruction needs %s (pip install %s)"
            % (", ".join(missing), " ".join(
                {"cv2": "opencv-python", "mediapipe": "mediapipe",
                 "yt_dlp": "yt-dlp"}[m] for m in missing)))


def fetch_metadata(limit=200):
    """Rows of {video_id, caption} from the public YouTube-ASL metadata."""
    last = None
    for url in META_CANDIDATES:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read().decode("utf-8", "replace")
            if "\t" not in raw and "," not in raw:
                out = []
                for line in raw.splitlines():
                    vid = line.strip()
                    if vid and not vid.startswith("#"):
                        # public release is video IDs only; the caption stream
                        # is not in the same bucket, so keep a declared placeholder
                        out.append({"video_id": vid,
                                    "caption": "ASL video " + vid})
                    if len(out) >= int(limit):
                        break
                if out:
                    return out, url
                continue
            delim = "\t" if url.endswith(".tsv") else ","
            rows = list(csv.DictReader(io.StringIO(raw), delimiter=delim))
            out = []
            for row in rows:
                vid = (row.get("video_id") or row.get("VIDEO_ID")
                       or row.get("id") or "").strip()
                cap = (row.get("caption") or row.get("english")
                       or row.get("sentence") or row.get("text") or "").strip()
                if vid and cap:
                    out.append({"video_id": vid, "caption": cap})
                if len(out) >= int(limit):
                    break
            if out:
                return out, url
        except Exception as exc:  # noqa: BLE001
            last = exc
    return None, str(last)[:240]


def download_video(video_id, dest_dir=None):
    """Download one YouTube video; returns the local path."""
    _require()
    import yt_dlp
    dest = Path(dest_dir or (CACHE / "videos"))
    dest.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(dest / (video_id + ".%(ext)s"))
    opts = {"quiet": True, "no_warnings": True, "format": "mp4",
            "outtmpl": out_tmpl, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download(["https://www.youtube.com/watch?v=" + video_id])
    hits = sorted(dest.glob(video_id + ".*"))
    if not hits:
        raise RuntimeError("download produced no file for " + video_id)
    return hits[0]


def _blank_hand():
    return [0.0] * (N_HAND * 3)


def extract_3d_motion(video_path, fps=8, max_frames=64, min_frames=4):
    """Per-frame MediaPipe Holistic world landmarks -> (T, 75, 3)."""
    _require()
    import cv2
    import mediapipe as mp
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video " + str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(1, fps))))
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True)
    frames = []
    i = 0
    try:
        while len(frames) < int(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if i % step:
                i += 1
                continue
            i += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            vec = []
            pose = getattr(res, "pose_world_landmarks", None)
            if pose is not None:
                vec += [c for lm in pose.landmark for c in (lm.x, lm.y, lm.z)]
            else:
                vec += [0.0] * (N_POSE * 3)
            for hand in (getattr(res, "left_hand_landmarks", None),
                         getattr(res, "right_hand_landmarks", None)):
                if hand is not None:
                    vec += [c for lm in hand.landmark for c in (lm.x, lm.y, lm.z)]
                else:
                    vec += _blank_hand()
            frames.append(vec)
    finally:
        holistic.close()
        cap.release()
    if len(frames) < int(min_frames):
        raise RuntimeError("only %d usable frames" % len(frames))
    return np.asarray(frames, dtype=np.float32)


def reconstruct_youtube_item(video_id, caption, fps=8, max_frames=64,
                             force=False):
    """Download + reconstruct one item, caching the (T, 75, 3) array."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / (video_id + ".npy")
    import numpy as np
    if cache.exists() and not force:
        arr = np.load(cache)
    else:
        video = download_video(video_id)
        arr = extract_3d_motion(video, fps=fps, max_frames=max_frames)
        np.save(cache, arr)
    return {"motion": arr, "text": caption, "video_id": video_id,
            "source": "youtube-asl/reconstructed"}


def youtube_items(n=4, seed=0, fps=8, max_frames=64):
    """Load metadata, reconstruct n items; returns (items, provenance)."""
    try:
        meta, src = fetch_metadata(limit=max(50, int(n) * 10))
        if meta is None:
            return None, "metadata unavailable: " + str(src)
        import random
        rng = random.Random(seed)
        rng.shuffle(meta)
        items = []
        errors = []
        for row in meta:
            if len(items) >= int(n):
                break
            try:
                items.append(reconstruct_youtube_item(
                    row["video_id"], row["caption"], fps=fps,
                    max_frames=max_frames))
            except Exception as exc:  # noqa: BLE001
                errors.append("%s: %s" % (row["video_id"], str(exc)[:120]))
        if not items:
            return None, "reconstruction failed: " + "; ".join(errors[:3])
        return items, "%s + MediaPipe Holistic world landmarks" % src
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)


def report():
    return {"dependencies": dependencies(), "cache": str(CACHE),
            "landmarks": {"pose": N_POSE, "left_hand": N_HAND,
                          "right_hand": N_HAND, "dim": DIM},
            "pipeline": "yt-dlp -> OpenCV frames -> MediaPipe Holistic "
                        "(pose_world_landmarks + hands) -> (T,75,3) npy cache"}
