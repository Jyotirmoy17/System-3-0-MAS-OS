"""Spotlight-style popup for Ubuntu."""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SOCKET_PATH = "/tmp/mas_os.sock"
POPUP_HTML  = Path(__file__).parent.parent.parent / "ui" / "popup.html"
WIN_W, WIN_H = 620, 340


class _DaemonClient:
    def __init__(self) -> None:
        self._sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(SOCKET_PATH)
            self._sock = s
            return True
        except Exception:
            self._sock = None
            return False

    def send(self, data: Dict[str, Any]) -> None:
        if self._sock is None:
            return
        try:
            self._sock.sendall((json.dumps(data) + "\n").encode("utf-8"))
        except Exception:
            self._sock = None

    def lines(self):
        """Yield parsed JSON dicts from the socket stream."""
        buf = b""
        while True:
            try:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        yield json.loads(line.decode("utf-8"))
                    except Exception:
                        pass
            except Exception:
                break

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


class _PopupAPI:
    """Methods callable from JavaScript inside the WebView."""

    def __init__(self, controller: "_PopupController") -> None:
        self._ctrl = controller

    def submit_goal(self, goal: str) -> None:
        self._ctrl.submit_goal(goal)

    def close(self) -> None:
        self._ctrl.hide()

    def get_history(self) -> List[Dict[str, Any]]:
        return self._ctrl.load_history()


class _PopupController:
    def __init__(self) -> None:
        self._window = None
        self._api    = _PopupAPI(self)

    def show(self) -> None:
        try:
            import webview  # noqa: PLC0415
        except ImportError:
            print("pywebview not installed: pip install pywebview", file=sys.stderr)
            return

        if self._window is None:
            html = (POPUP_HTML.read_text(encoding="utf-8")
                    if POPUP_HTML.exists() else _FALLBACK_HTML)
            self._window = webview.create_window(
                title            = "MAS-OS",
                html             = html,
                width            = WIN_W,
                height           = WIN_H,
                resizable        = False,
                frameless        = True,
                on_top           = True,
                background_color = "#0d0d0d",
                js_api           = self._api,
            )
            webview.start(debug=False)
        else:
            self._window.show()

    def hide(self) -> None:
        if self._window is not None:
            try:
                self._window.hide()
            except Exception:
                pass

    def submit_goal(self, goal: str) -> None:
        self.hide()
        threading.Thread(
            target=self._run_in_background,
            args=(goal,),
            daemon=True,
        ).start()

    def load_history(self) -> List[Dict[str, Any]]:
        client = _DaemonClient()
        if not client.connect():
            return []
        client.send({"type": "history"})
        for msg in client.lines():
            if msg.get("type") == "history":
                client.close()
                return msg.get("entries", [])
        client.close()
        return []

    def _run_in_background(self, goal: str) -> None:
        client = _DaemonClient()
        if not client.connect():
            _notify("MAS-OS", "Daemon not running.  Run: mas-os daemon start")
            return

        client.send({"type": "goal", "goal": goal,
                     "session_id": f"popup_{int(time.time())}"})
        final = ""
        for msg in client.lines():
            t = msg.get("type", "")
            if t == "result":
                final = msg.get("answer", "")
                break
            if t == "error":
                final = f"Error: {msg.get('message', '')}"
                break
        client.close()
        if final:
            _notify("MAS-OS", final[:200])


def _notify(title: str, body: str) -> None:
    """Send a desktop notification via notify-send (Ubuntu)."""
    import subprocess  # noqa: PLC0415
    try:
        subprocess.run(
            ["notify-send", "-i", "dialog-information", title, body],
            check=False, timeout=5,
        )
    except Exception:
        print(f"[{title}] {body}", file=sys.stderr)


def _start_hotkey_listener(controller: _PopupController) -> None:
    try:
        from pynput import keyboard  # noqa: PLC0415
    except ImportError:
        print("pynput not installed: pip install pynput", file=sys.stderr)
        return

    pressed: set = set()

    def on_press(key: Any) -> None:
        pressed.add(key)
        ctrl  = any(k in pressed for k in (
            keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r
        ))
        space = keyboard.Key.space in pressed
        if ctrl and space:
            threading.Thread(target=controller.show, daemon=True).start()

    def on_release(key: Any) -> None:
        pressed.discard(key)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()


_FALLBACK_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#111;color:#eee;font-family:-apple-system,sans-serif;
     display:flex;align-items:center;justify-content:center;height:100vh}
.box{width:560px}
.logo{font-size:12px;color:#555;margin-bottom:10px;letter-spacing:.06em}
input{width:100%;background:#1e1e1e;border:1px solid #2a2a2a;
      border-radius:10px;color:#eee;font-size:17px;
      padding:13px 16px;outline:none}
input:focus{border-color:#6366f1}
input::placeholder{color:#444}
</style></head>
<body><div class="box">
  <div class="logo">⬡ MAS-OS</div>
  <input id="i" placeholder="What do you want to do?" autofocus>
</div>
<script>
document.getElementById('i').addEventListener('keydown',function(e){
  if(e.key==='Enter'&&this.value.trim()){
    window.pywebview.api.submit_goal(this.value.trim());
    this.value='';
  }
  if(e.key==='Escape') window.pywebview.api.close();
});
</script></body></html>"""


def main() -> None:
    ctrl = _PopupController()
    _start_hotkey_listener(ctrl)
    print("MAS-OS popup ready.  Press Ctrl+Space anywhere to open.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
