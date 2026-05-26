"""System tray icon for Ubuntu (AppIndicator / pystray)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PID_FILE  = "/tmp/mas_os.pid"
DASHBOARD = "http://localhost:8000"


def _daemon_running() -> bool:
    if not Path(PID_FILE).exists():
        return False
    try:
        os.kill(int(Path(PID_FILE).read_text()), 0)
        return True
    except (ProcessLookupError, ValueError, OSError):
        return False


def _start_daemon() -> None:
    subprocess.Popen(
        [sys.executable, "-m", "mas.interfaces.daemon", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_daemon() -> None:
    subprocess.run(
        [sys.executable, "-m", "mas.interfaces.daemon", "stop"],
        check=False,
    )


def _open_dashboard() -> None:
    subprocess.Popen(["xdg-open", DASHBOARD])


def _make_icon(color: str):
    """Create a small PIL image for the tray icon."""
    from PIL import Image, ImageDraw  # noqa: PLC0415
    img  = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, 56, 56], fill=color)
    return img


def main() -> None:
    try:
        import pystray  # noqa: PLC0415
    except ImportError:
        print("pystray not installed: pip install pystray Pillow", file=sys.stderr)
        sys.exit(1)

    icon_obj: list = [None]   # mutable container for the icon reference

    def current_icon():
        color = "#22c55e" if _daemon_running() else "#ef4444"
        return _make_icon(color)

    def update_icon() -> None:
        while True:
            time.sleep(5)
            if icon_obj[0] is not None:
                icon_obj[0].icon = current_icon()

    icon = pystray.Icon(
        name  = "mas-os",
        icon  = current_icon(),
        title = "MAS-OS",
        menu  = pystray.Menu(
            pystray.MenuItem("Open Dashboard", lambda _i, _it: _open_dashboard()),
            pystray.MenuItem("Start Daemon",   lambda _i, _it: _start_daemon()),
            pystray.MenuItem("Stop Daemon",    lambda _i, _it: _stop_daemon()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit",           lambda _i, _it: icon_obj[0].stop()),
        ),
    )
    icon_obj[0] = icon
    threading.Thread(target=update_icon, daemon=True).start()
    icon.run()


if __name__ == "__main__":
    main()
