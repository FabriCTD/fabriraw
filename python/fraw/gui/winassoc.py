"""Optional Windows file association (current user only, HKCU\\Software\\Classes).

Only offered when running as the frozen portable .exe on Windows and only
executed when the user clicks the menu item. Nothing is written to HKLM.
"""

from __future__ import annotations

import ntpath
import os
import sys

__all__ = ["available", "PROGID", "EXTENSIONS", "registry_plan", "register", "unregister", "is_registered"]

PROGID = "FabriCTD.CTDPhotoStudio.1"
EXTENSIONS = (".fraw", ".frw", ".fset")
_DESCRIPTION = {".fraw": "Imagen FabriRAW", ".frw": "Imagen FabriRAW", ".fset": "Ráfaga FabriRAW"}
_ROOT = r"Software\Classes"


def available() -> bool:
    return sys.platform == "win32" and bool(getattr(sys, "frozen", False))


def registry_plan(exe: str) -> list[tuple[str, str | None, str]]:
    """[(subkey under HKCU, value name (None = default), data)] that :func:`register` writes."""
    exe = ntpath.abspath(exe)  # rutas del registro siempre son de Windows
    plan: list[tuple[str, str | None, str]] = [
        (rf"{_ROOT}\{PROGID}", None, "Imagen FabriRAW (CTD Photo Studio)"),
        (rf"{_ROOT}\{PROGID}\DefaultIcon", None, f'"{exe}",0'),
        (rf"{_ROOT}\{PROGID}\shell\open", "FriendlyAppName", "CTD Photo Studio"),
        (rf"{_ROOT}\{PROGID}\shell\open\command", None, f'"{exe}" "%1"'),
    ]
    for ext in EXTENSIONS:
        plan.append((rf"{_ROOT}\{ext}", None, PROGID))
        plan.append((rf"{_ROOT}\{ext}", "Content Type", "application/x-fabriraw"))
        plan.append((rf"{_ROOT}\{ext}\OpenWithProgids", PROGID, ""))
    return plan


def _notify() -> None:
    try:
        import ctypes  # noqa: PLC0415

        SHCNE_ASSOCCHANGED, SHCNF_IDLIST = 0x08000000, 0
        ctypes.windll.shell32.SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None)
    except Exception:  # noqa: BLE001 - cosmetic (Explorer icon refresh)
        pass


def register(exe: str | None = None) -> None:
    import winreg  # noqa: PLC0415

    exe = exe or sys.executable
    for sub, name, data in registry_plan(exe):
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, sub, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, name or "", 0, winreg.REG_SZ, data)
    _notify()


def _delete_tree(root, sub: str) -> None:
    import winreg  # noqa: PLC0415

    try:
        with winreg.OpenKey(root, sub, 0, winreg.KEY_READ | winreg.KEY_WRITE) as k:
            while True:
                try:
                    child = winreg.EnumKey(k, 0)
                except OSError:
                    break
                _delete_tree(root, rf"{sub}\{child}")
        winreg.DeleteKey(root, sub)
    except FileNotFoundError:
        pass


def _delete_if_empty(root, sub: str) -> None:
    """Deletes ``sub`` only if it has no subkeys and no values (RegDeleteKey would drop values)."""
    import winreg  # noqa: PLC0415

    try:
        with winreg.OpenKey(root, sub) as k:
            n_sub, n_val, _ = winreg.QueryInfoKey(k)
        if n_sub == 0 and n_val == 0:
            winreg.DeleteKey(root, sub)
    except OSError:
        pass


def is_registered() -> bool:
    if sys.platform != "win32":
        return False
    import winreg  # noqa: PLC0415

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"{_ROOT}\.fraw") as k:
            return winreg.QueryValueEx(k, "")[0] == PROGID
    except OSError:
        return False


def unregister() -> None:
    """Removes our ProgID and the extension keys only if they still point to it."""
    import winreg  # noqa: PLC0415

    hk = winreg.HKEY_CURRENT_USER
    for ext in EXTENSIONS:
        sub = rf"{_ROOT}\{ext}"
        try:
            with winreg.OpenKey(hk, sub, 0, winreg.KEY_READ | winreg.KEY_WRITE) as k:
                try:
                    cur = winreg.QueryValueEx(k, "")[0]
                except OSError:
                    cur = None
                if cur == PROGID:
                    winreg.DeleteValue(k, "")
                    try:
                        winreg.DeleteValue(k, "Content Type")
                    except OSError:
                        pass
            try:
                with winreg.OpenKey(hk, rf"{sub}\OpenWithProgids", 0, winreg.KEY_SET_VALUE) as k:
                    winreg.DeleteValue(k, PROGID)
            except OSError:
                pass
            _delete_if_empty(hk, rf"{sub}\OpenWithProgids")
            _delete_if_empty(hk, sub)
        except FileNotFoundError:
            continue
    _delete_tree(hk, rf"{_ROOT}\{PROGID}")
    _notify()
