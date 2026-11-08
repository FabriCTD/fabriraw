"""Historial de revelado (deshacer / rehacer / volver a un paso) e instantáneas. Sin Qt.

Cada paso guarda el estado completo (``DevParams.to_dict(full=True)``, ~4 KB).
Los cambios continuos del mismo control (arrastre de un deslizador, rueda,
flechas) se agrupan en un solo paso: mientras el paso de arriba esté *abierto*
y llegue un cambio con la misma ``merge_key``, se reemplaza en lugar de apilar.
``record(..., final=True)`` cierra el paso.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["History", "Step"]

MAX_STEPS = 400


@dataclass
class Step:
    label: str
    state: dict[str, Any]
    merge_key: str | None = None
    open: bool = False


@dataclass
class History:
    steps: list[Step] = field(default_factory=list)
    index: int = -1  # paso actual
    snapshots: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def reset(self, state: dict[str, Any], label: str = "Abierto") -> None:
        self.steps = [Step(label, dict(state))]
        self.index = 0
        self.snapshots = []

    @property
    def current(self) -> Step | None:
        return self.steps[self.index] if 0 <= self.index < len(self.steps) else None

    def record(self, label: str, state: dict[str, Any], merge_key: str | None = None, final: bool = True) -> bool:
        """Registra un cambio. Devuelve True si creó un paso nuevo (False si fusionó o no cambió nada)."""
        cur = self.current
        if cur is not None and cur.open and merge_key is not None and cur.merge_key == merge_key and self.index > 0:
            cur.label, cur.state = label, dict(state)
            cur.open = not final
            if cur.state == self.steps[self.index - 1].state and final:  # volvió al valor anterior
                self.steps.pop(self.index)
                self.index -= 1
            return False
        if cur is not None:
            cur.open = False
            if cur.state == state:
                return False
        del self.steps[self.index + 1 :]
        self.steps.append(Step(label, dict(state), merge_key, open=not final))
        if len(self.steps) > MAX_STEPS:
            del self.steps[1]
        self.index = len(self.steps) - 1
        return True

    def close(self) -> None:
        if self.current is not None:
            self.current.open = False

    def can_undo(self) -> bool:
        return self.index > 0

    def can_redo(self) -> bool:
        return self.index < len(self.steps) - 1

    def undo(self) -> dict[str, Any] | None:
        if not self.can_undo():
            return None
        self.close()
        self.index -= 1
        return dict(self.steps[self.index].state)

    def redo(self) -> dict[str, Any] | None:
        if not self.can_redo():
            return None
        self.close()
        self.index += 1
        return dict(self.steps[self.index].state)

    def goto(self, i: int) -> dict[str, Any] | None:
        if not 0 <= i < len(self.steps):
            return None
        self.close()
        self.index = i
        return dict(self.steps[i].state)

    def labels(self) -> list[str]:
        return [s.label for s in self.steps]

    # ------------------------------------------------------------ snapshots
    def add_snapshot(self, name: str, state: dict[str, Any]) -> None:
        self.snapshots.append((name, dict(state)))

    def remove_snapshot(self, i: int) -> None:
        if 0 <= i < len(self.snapshots):
            del self.snapshots[i]
