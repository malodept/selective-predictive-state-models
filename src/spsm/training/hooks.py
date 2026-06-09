from __future__ import annotations


class Hook:
    def on_epoch_end(self, metrics: dict[str, float]) -> None:
        return None
