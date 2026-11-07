#!/usr/bin/env python3
"""Problem-space check of the 8-field constrained trigger on a real benign PE.

Applies the trigger (SAFE_FEATURE_INDICES in attack_smart.py) to a copy of a
local executable using non-destructive edits only, then checks that the file
still parses as a PE and that the recomputed EMBER feature values reach their
targets. By default the carrier is a copy of the local python.exe, so no
external file is needed; the original file is never modified. The malware side
is checked by malware_static_validation.py.

Edits:
  - HeaderFileInfo scalars (timestamp, linker version, OS version, sizeof_code):
    direct header writes with pefile.
  - StringExtractor scalars (numstrings, string entropy, MZ count) and file
    size: an append-only overlay after the declared content. No existing byte
    is removed.

Usage:
    python problem_space_validation.py [--carrier PATH_TO_A_LOCAL_EXE]

Requires: pip install pefile
"""

import argparse
import math
import os
import random
import re
import shutil
import subprocess
import sys

_STRING_RE = re.compile(rb"[\x20-\x7f]{5,}")
_MZ_RE = re.compile(rb"MZ")

# The 8-field constrained trigger from attack_smart.py / paper Table 2.
TARGETS = {
    "timestamp": 708992512.0,          # feat[626], HeaderFileInfo
    "numstrings": 29.0,                # feat[512], StringExtractor
    "string_entropy": 4.4001,          # feat[611], StringExtractor
    "major_linker_version": 14.0,      # feat[679], HeaderFileInfo
    "size": 10485760.0,                # feat[616], GeneralFileInfo (10 MB)
    "sizeof_code": 23552.0,            # feat[685], HeaderFileInfo
    "major_os_version": 10.0,          # feat[681], HeaderFileInfo
    "mz_count": 2.0,                   # feat[615], StringExtractor
}


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    ent = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent


def string_features(raw: bytes):
    """EMBER StringExtractor logic for the three fields used here.

    Strings are runs of printable ASCII [0x20-0x7f] of length >= 5. numstrings is
    their count, entropy is the Shannon entropy of all string bytes together, and
    mz_count counts b"MZ" within that string corpus, as in ember/features.py.
    """
    strings = _STRING_RE.findall(raw)
    numstrings = len(strings)
    all_string_bytes = b"".join(strings)
    entropy = shannon_entropy(all_string_bytes)
    mz_count = len(_MZ_RE.findall(all_string_bytes))
    return numstrings, entropy, mz_count


def header_features(path: str):
    import pefile
    pe = pefile.PE(path, fast_load=True)
    vals = (int(pe.FILE_HEADER.TimeDateStamp),
            int(pe.OPTIONAL_HEADER.MajorLinkerVersion),
            int(pe.OPTIONAL_HEADER.MajorOperatingSystemVersion),
            int(pe.OPTIONAL_HEADER.SizeOfCode))
    pe.close()
    return vals


def extract_target_fields(path: str) -> dict:
    raw = open(path, "rb").read()
    numstrings, entropy, mz_count = string_features(raw)
    timestamp, linker, os_version, sizeof_code = header_features(path)
    return {
        "timestamp": timestamp, "numstrings": numstrings,
        "string_entropy": entropy, "major_linker_version": linker,
        "size": os.path.getsize(path), "sizeof_code": sizeof_code,
        "major_os_version": os_version, "mz_count": mz_count,
    }


def is_valid_pe(path: str) -> bool:
    import pefile
    try:
        pe = pefile.PE(path, fast_load=True)
        ok = bool(pe.DOS_HEADER.e_magic == 0x5A4D and pe.NT_HEADERS.Signature == 0x4550)
        pe.close()
        return ok
    except Exception:
        return False


def apply_trigger(src: str, dst: str, seed: int = 0):
    import pefile

    shutil.copyfile(src, dst)

    # Step 1: direct header-struct edits.
    pe = pefile.PE(dst)
    pe.FILE_HEADER.TimeDateStamp = int(TARGETS["timestamp"])
    pe.OPTIONAL_HEADER.MajorLinkerVersion = int(TARGETS["major_linker_version"])
    pe.OPTIONAL_HEADER.MajorOperatingSystemVersion = int(TARGETS["major_os_version"])
    pe.OPTIONAL_HEADER.SizeOfCode = int(TARGETS["sizeof_code"])
    pe.OPTIONAL_HEADER.CheckSum = pe.generate_checksum()
    pe.write(dst)
    pe.close()

    # Step 2: append-only overlay -- directional test for numstrings/entropy.
    # Low-diversity filler chunks, separated by NUL (0x00, outside the
    # [\x20-\x7f] match range so each chunk counts as its own string --
    # a space would NOT separate them, since space is itself in-range).
    rng = random.Random(seed)
    alphabet = "AAAAAAAABCDE"
    chunks = ["".join(rng.choice(alphabet) for _ in range(rng.randint(5, 12))).encode()
              for _ in range(80)]
    with open(dst, "ab") as f:
        f.write(b"\x00".join(chunks))

    # Step 3: pad to the exact target file size (last, simplest, exact).
    target_size = int(TARGETS["size"])
    current_size = os.path.getsize(dst)
    if current_size < target_size:
        with open(dst, "ab") as f:
            f.write(b"\x00" * (target_size - current_size))
    elif current_size > target_size:
        print(f"[!!] carrier already exceeds target size ({current_size} > "
              f"{target_size}) -- cannot hit this target without truncation.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carrier", default=sys.executable,
                     help="path to a local, already-present benign PE file "
                          "to use as the carrier (default: this Python "
                          "interpreter itself -- no download required)")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    src = os.path.join(args.out_dir, "carrier_original.exe")
    dst = os.path.join(args.out_dir, "carrier_watermarked.exe")
    shutil.copyfile(args.carrier, src)

    apply_trigger(src, dst)

    print("=" * 78)
    print("STRUCTURAL VALIDITY")
    print(f"  original  valid PE : {is_valid_pe(src)}")
    print(f"  watermarked valid PE : {is_valid_pe(dst)}")

    print("\nEXECUTION CHECK (--version, identical output expected)")
    out_orig = subprocess.run([src, "--version"], capture_output=True, timeout=10)
    out_wm = subprocess.run([dst, "--version"], capture_output=True, timeout=10)
    print(f"  original   : rc={out_orig.returncode}  stdout={out_orig.stdout.strip()!r}")
    print(f"  watermarked: rc={out_wm.returncode}  stdout={out_wm.stdout.strip()!r}")
    print(f"  MATCH: {out_orig.stdout == out_wm.stdout and out_orig.returncode == out_wm.returncode}")

    print("\n" + "=" * 78)
    print(f"{'field':22} {'original':>14} {'watermarked':>14} {'target':>14} {'exact?':>7}")
    vals_o, vals_w = extract_target_fields(src), extract_target_fields(dst)
    for k in TARGETS:
        o, w, t = vals_o[k], vals_w[k], TARGETS[k]
        print(f"{k:22} {o:>14.4f} {w:>14.4f} {t:>14.4f} {'YES' if abs(w-t)<1e-6 else 'no':>7}")


if __name__ == "__main__":
    main()
