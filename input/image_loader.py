from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageItem:
    path: Path
    source: str


class ImageLoader:
    def __init__(self, roots: Iterable[Path], recursive: bool = True) -> None:
        self.roots = [Path(r) for r in roots]
        self.recursive = recursive

    def iter_images(self) -> list[ImageItem]:
        items: list[ImageItem] = []
        for root in self.roots:
            if not root.exists():
                continue
            if root.is_file():
                if root.suffix.lower() in SUPPORTED_EXTS:
                    items.append(ImageItem(root, root.parent.name))
                continue
            globber = root.rglob("*") if self.recursive else root.glob("*")
            for p in globber:
                if not p.is_file():
                    continue
                if p.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                items.append(ImageItem(p, root.name))
        items.sort(key=lambda x: str(x.path).lower())
        return items

