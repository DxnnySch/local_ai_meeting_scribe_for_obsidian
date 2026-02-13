import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import pystray
import uvicorn
import winreg
from PIL import Image, ImageDraw
from pystray import MenuItem as item

from companion_app import APP

APP_NAME = "LocalMeetingScribe Companion"
RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_KEY_NAME = "LocalMeetingScribeCompanion"
DEFAULT_HOST = os.getenv("MSCRIBE_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("MSCRIBE_PORT", "8000"))
DEFAULT_MODEL = os.getenv("MSCRIBE_SUMMARY_MODEL", "mistral").strip() or "mistral"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download/windows"


def _tray_image() -> Image.Image:
    image = Image.new("RGB", (64, 64), color=(25, 45, 90))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 61, 61), outline=(180, 210, 255), width=2)
    draw.text((14, 22), "LMS", fill=(255, 255, 255))
    return image


def _run_command(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        check=False,
    )


def _is_ollama_installed() -> bool:
    try:
        result = _run_command(["ollama", "--version"], timeout=8)
        return result.returncode == 0
    except Exception:
        return False


def _try_install_ollama_with_winget() -> bool:
    try:
        result = _run_command(
            [
                "winget",
                "install",
                "--id",
                "Ollama.Ollama",
                "--silent",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            timeout=900,
        )
        return result.returncode == 0
    except Exception:
        return False


def _ensure_ollama_service() -> None:
    list_check = _run_command(["ollama", "list"], timeout=10)
    if list_check.returncode == 0:
        return

    # Start the Ollama daemon in the background if it is not already running.
    creation_flags = 0
    if sys.platform.startswith("win"):
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    time.sleep(3)


def _ensure_model(model: str) -> subprocess.CompletedProcess:
    _ensure_ollama_service()
    return _run_command(["ollama", "pull", model], timeout=1800)


def _autostart_command() -> str:
    if getattr(sys, "frozen", False):
        target = Path(sys.executable).resolve()
        return f"\"{target}\" --startup"

    script = Path(__file__).resolve()
    return f"\"{sys.executable}\" \"{script}\" --startup"


def is_autostart_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_READ) as key:
            value, _ = winreg.QueryValueEx(key, RUN_KEY_NAME)
            return str(value).strip() != ""
    except FileNotFoundError:
        return False
    except OSError:
        return False


def set_autostart(enabled: bool) -> None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, RUN_KEY_NAME, 0, winreg.REG_SZ, _autostart_command())
        else:
            try:
                winreg.DeleteValue(key, RUN_KEY_NAME)
            except FileNotFoundError:
                pass


class CompanionServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running():
            return
        config = uvicorn.Config(APP, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None

    def restart(self) -> None:
        self.stop()
        self.start()


class TrayApplication:
    def __init__(self, host: str, port: int, model: str) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.server = CompanionServer(host=host, port=port)
        self.icon = pystray.Icon("local-meeting-scribe", _tray_image(), APP_NAME, self._menu())
        self._busy = False

    def _menu(self) -> pystray.Menu:
        return pystray.Menu(
            item(self._status_text, None, enabled=False),
            item("Open status page", self._open_status_page),
            item(self._autostart_label, self._toggle_autostart),
            item(f"Install Ollama + pull {self.model}", self._install_ollama_and_model),
            item("Restart companion service", self._restart_service),
            item("Quit", self._quit),
        )

    def _status_text(self, _menu_item: pystray.MenuItem) -> str:
        status = "running" if self.server.is_running() else "stopped"
        return f"Companion service: {status}"

    def _autostart_label(self, _menu_item: pystray.MenuItem) -> str:
        return "Disable launch at login" if is_autostart_enabled() else "Enable launch at login"

    def _notify(self, message: str) -> None:
        try:
            self.icon.notify(message, APP_NAME)
        except Exception:
            pass

    def _open_status_page(self, _icon: pystray.Icon, _menu_item: pystray.MenuItem) -> None:
        webbrowser.open(f"http://{self.host}:{self.port}/status")

    def _toggle_autostart(self, _icon: pystray.Icon, _menu_item: pystray.MenuItem) -> None:
        set_autostart(not is_autostart_enabled())
        self._notify("Startup setting updated.")
        self.icon.update_menu()

    def _restart_service(self, _icon: pystray.Icon, _menu_item: pystray.MenuItem) -> None:
        self.server.restart()
        self._notify("Companion service restarted.")
        self.icon.update_menu()

    def _install_ollama_and_model(self, _icon: pystray.Icon, _menu_item: pystray.MenuItem) -> None:
        if self._busy:
            self._notify("Setup already running.")
            return

        thread = threading.Thread(target=self._install_ollama_and_model_worker, daemon=True)
        thread.start()

    def _install_ollama_and_model_worker(self) -> None:
        self._busy = True
        try:
            self._notify("Checking Ollama installation...")
            if not _is_ollama_installed():
                self._notify("Installing Ollama via winget...")
                if not _try_install_ollama_with_winget():
                    self._notify("Could not install Ollama automatically. Opening download page.")
                    webbrowser.open(OLLAMA_DOWNLOAD_URL)
                    return
                # Give Windows Installer a moment before checking PATH availability.
                time.sleep(2)

            if not _is_ollama_installed():
                self._notify("Ollama is still unavailable. Please install manually.")
                webbrowser.open(OLLAMA_DOWNLOAD_URL)
                return

            self._notify(f"Pulling model '{self.model}' (first run may take a while)...")
            pull_result = _ensure_model(self.model)
            if pull_result.returncode != 0:
                self._notify("Model pull failed. Open logs to inspect output.")
                return

            self._notify(f"Setup complete. Ollama + {self.model} are ready.")
        except Exception:
            self._notify("Setup failed unexpectedly.")
        finally:
            self._busy = False

    def _quit(self, _icon: pystray.Icon, _menu_item: pystray.MenuItem) -> None:
        self.server.stop()
        self.icon.stop()

    def run(self) -> None:
        self.server.start()
        self.icon.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LocalMeetingScribe tray launcher")
    parser.add_argument("--startup", action="store_true", help="Indicates launch from Windows startup")
    parser.add_argument("--setup-ollama", action="store_true", help="Install Ollama and pull model on launch")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to pull during setup (default: mistral)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = TrayApplication(host=DEFAULT_HOST, port=DEFAULT_PORT, model=args.model)

    if not is_autostart_enabled():
        set_autostart(True)

    if args.setup_ollama:
        threading.Thread(target=app._install_ollama_and_model_worker, daemon=True).start()

    app.run()


if __name__ == "__main__":
    main()
