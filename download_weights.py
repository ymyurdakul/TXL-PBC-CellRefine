"""Downloads the three released CellRefine weights and places them where the scripts expect them:
TXLPBC_runs/cellrefine/runs/r18_s{0,1,2}/best.pt. The SHA-256 checksum of each file is verified.

    python download_weights.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "https://github.com/ymyurdakul/TXL-PBC-CellRefine/releases/download/v1.0/"
FILES = {  # release asset -> (target, SHA-256)
    "cellrefine_r18_s0.pt": ("TXLPBC_runs/cellrefine/runs/r18_s0/best.pt",
                             "74a416496c70947df18052c50b720e87471149be9dd9df260ba5b3278a70f8d6"),
    "cellrefine_r18_s1.pt": ("TXLPBC_runs/cellrefine/runs/r18_s1/best.pt",
                             "7c030c6242aef0c4547ac3073c9659ff989c341785513fdf9acb2b17bb084a3c"),
    "cellrefine_r18_s2.pt": ("TXLPBC_runs/cellrefine/runs/r18_s2/best.pt",
                             "719e5993a0da36e74c1bf1a7109c527e22827bd7256b9afc11c509d9add0385f"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


ok = True
for name, (target, digest) in FILES.items():
    dst = ROOT / target
    if dst.exists() and sha256(dst) == digest:
        print(f"{target}: already present")
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"{target}: downloading {URL + name}")
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(URL + name, tmp)
    if sha256(tmp) != digest:
        print(f"{target}: checksum mismatch, file not saved")
        tmp.unlink()
        ok = False
        continue
    tmp.replace(dst)
    print(f"{target}: done")
sys.exit(0 if ok else 1)
