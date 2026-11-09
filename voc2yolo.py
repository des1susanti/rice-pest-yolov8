import random, shutil, xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

RICE_CLASSES = [
    "rice leaf roller", "rice leaf caterpillar", "paddy stem maggot", "asiatic rice borer",
    "yellow rice borer", "rice gall midge", "rice stemfly", "brown plant hopper",
    "white backed plant hopper", "small brown plant hopper", "rice water weevil",
    "rice leafhopper", "grain spreader thrips", "rice shell pest",
]


def find_voc_root(base):
    """Cari folder yang berisi Annotations/ dan JPEGImages/ di dalam base."""
    for ann in Path(base).rglob("Annotations"):
        if ann.is_dir() and (ann.parent / "JPEGImages").is_dir():
            return ann.parent
    raise FileNotFoundError(f"Folder Annotations/ + JPEGImages/ tidak ditemukan di {base}")


def class_index(name, zero_based):
    """IP102 menyimpan nama kelas sebagai angka (indeks kelas). Kembalikan indeks 0-based."""
    name = name.strip()
    if name.isdigit():
        i = int(name)
        return i if zero_based else i - 1
    low = name.lower()
    return RICE_CLASSES.index(low) if low in RICE_CLASSES else None


def detect_zero_based(ann_dir, sample=3000):
    names = set()
    for k, x in enumerate(Path(ann_dir).glob("*.xml")):
        if k >= sample:
            break
        for o in ET.parse(x).getroot().iter("object"):
            names.add(o.findtext("name", "").strip())
    nums = {int(n) for n in names if n.isdigit()}
    if not nums:
        return True
    return 0 in nums or max(nums) <= 101


def voc_to_yolo(voc_root, out_dir, val_ratio=0.2, seed=42):
    """Ubah anotasi VOC IP102 menjadi format YOLO, hanya untuk 14 kelas hama padi."""
    voc_root, out_dir = Path(voc_root), Path(out_dir)
    ann_dir, img_dir = voc_root / "Annotations", voc_root / "JPEGImages"
    zero_based = detect_zero_based(ann_dir)

    sets = voc_root / "ImageSets" / "Main"
    split_of = {}
    if (sets / "trainval.txt").exists() and (sets / "test.txt").exists():
        for s, f in (("train", "trainval.txt"), ("val", "test.txt")):
            for line in (sets / f).read_text().split():
                split_of[line.strip()] = s

    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    kept, counts = [], Counter()
    for xml in sorted(ann_dir.glob("*.xml")):
        root = ET.parse(xml).getroot()
        w = float(root.findtext("size/width", "0") or 0)
        h = float(root.findtext("size/height", "0") or 0)
        img = images.get(xml.stem)
        if img is None:
            continue
        if w <= 0 or h <= 0:
            from PIL import Image
            with Image.open(img) as im:
                w, h = im.size
        lines = []
        for o in root.iter("object"):
            c = class_index(o.findtext("name", ""), zero_based)
            if c is None or not 0 <= c < len(RICE_CLASSES):
                continue
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(f"{c} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            counts[c] += 1
        if lines:
            kept.append((xml.stem, img, lines))

    rng = random.Random(seed)
    for stem, img, lines in kept:
        split = split_of.get(stem) or ("val" if rng.random() < val_ratio else "train")
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, out_dir / "images" / split / img.name)
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    yaml = out_dir / "rice_pest.yaml"
    yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(RICE_CLASSES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(RICE_CLASSES))
    )
    n_split = Counter(split_of.get(s) or "random" for s, _, _ in kept)
    return yaml, counts, len(kept), zero_based, n_split
