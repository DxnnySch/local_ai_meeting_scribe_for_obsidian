# LocalMeetingScribe

Local-first meeting capture for Obsidian with a Python companion app.

## What is implemented now

- Obsidian plugin flow in `main.ts`:
  - Start/Stop capture from ribbon or commands
  - Speaker labeling modal after stop
  - Save prompt for filename/folder
  - Optional local summarization call
  - Settings tab for server URL, default folder, auto-summarize
  - Capture profile controls: `Fast`, `Balanced`, `Pristine`
  - Toggle for diarization and loopback preference
- Python companion app in `companion_app.py`:
  - `POST /start` begins local recording with profile options
  - `POST /stop` stops recording and returns transcript segments
  - `POST /summarize` returns a local summary draft
  - `GET /status` returns runtime state
  - Tries Windows loopback + mic capture through `pyaudiowpatch`, with `sounddevice` fallback
  - Optional diarization hooks (WhisperX is not installed by default on this Python 3.13 setup)

## Quick start

### Recommended: tray executable (end-user install)

1. Download `LocalMeetingScribeCompanion.exe` from the latest project release.
1. Run it once. It will:
   - start the local companion service on `http://127.0.0.1:8000`
   - add a tray icon to Windows system icons
   - register itself to auto-start at login
1. Right-click the tray icon and select `Install Ollama + pull mistral` to automate local LLM setup.
   - The app first tries `winget` for Ollama.
   - If automatic install fails, it opens the official Ollama download page.
1. In Obsidian plugin settings, keep `Companion server URL` set to `http://127.0.0.1:8000`.

Optional command-line setup:

```powershell
.\LocalMeetingScribeCompanion.exe --setup-ollama --model mistral
```

### Developer: Python script workflow

1. Create and activate a Python virtual environment.
1. Install dependencies:

```powershell
pip install -r requirements.txt
npm install
```

1. Start the companion app:

```powershell
# Optional first run: create companion_config.env from companion_config.example.env
.\start_companion.ps1
```

Optional flags:

```powershell
.\start_companion.ps1 -InstallDeps
.\start_companion.ps1 -ConfigPath ".\companion_config.example.env"
```

1. In Obsidian:
   - Enable this plugin
   - Open plugin settings and verify `Companion server URL` is `http://127.0.0.1:8000`
   - Use ribbon icon or command palette: `Start / Stop capture`
   - Select capture profile in plugin settings (`Fast`, `Balanced`, `Pristine`)

## Companion executable build/distribution

Build the Windows tray executable:

```powershell
.\build_companion_exe.ps1 -InstallDeps
```

Output:

- `dist\LocalMeetingScribeCompanion.exe`

This executable bundles the companion service and tray launcher, and is intended as the primary distribution artifact for non-developer users.

## Plugin build steps

```powershell
npm run check
npm run build
```

- Output plugin bundle is `main.js` in the project root.
- For development watch mode, run `npm run dev`.

## Plugin release/deploy

Deploys `manifest.json` and `main.js` into your vault plugin folder:

```powershell
.\release.ps1 -VaultPath "D:\Path\To\Your\Vault"
```

You can also set a persistent env var and run without args:

```powershell
$env:OBSIDIAN_VAULT_PATH = "D:\Path\To\Your\Vault"
.\release.ps1
```

Optional skip build:

```powershell
.\release.ps1 -VaultPath "D:\Path\To\Your\Vault" -SkipBuild
```

## Plugin GitHub releases (Obsidian-compatible)

This repo includes `.github/workflows/release.yml` and creates a draft GitHub release for each pushed tag.

To publish a plugin release:

1. Ensure GitHub Actions workflow permissions are set to `Read and write permissions`.
1. Update `manifest.json` version.
1. Create an annotated tag matching the manifest version:

```powershell
git tag -a 0.1.0 -m "0.1.0"
git push origin 0.1.0
```

1. The workflow builds and attaches:
   - plugin files: `main.js`, `manifest.json` (and `styles.css` if present)
   - companion executable: `LocalMeetingScribeCompanion.exe`
1. Add release notes and publish the draft release in GitHub.

## Notes on quality and hardware

- The baseline is fully local and works without cloud calls.
- For 8-12 GB VRAM:
  - Start with `Balanced` profile
  - Use `MSCRIBE_ASR_COMPUTE=int8_float16`
  - Keep `MSCRIBE_ASR_DEVICE=auto` (or `cuda` if needed)
- For diarization:
  - Base install in `requirements.txt` keeps a conflict-free PyTorch stack on CUDA 11.8
  - WhisperX is optional and may require a different Python/CUDA combination
  - Set `MSCRIBE_HF_TOKEN` in your environment
  - Keep `Enable diarization` enabled in plugin settings

## Dependency compatibility note

- If you see `torchvision ... requires torch==... but you have torch ...`:
  - keep `torch/torchvision/torchaudio` on matching versions
  - for CUDA 11.8 in this project, the working set is:
    - `torch==2.7.1+cu118`
    - `torchvision==0.22.1+cu118`
    - `torchaudio==2.7.1+cu118`
- Current default install intentionally excludes `whisperx` to avoid version deadlocks on Python 3.13.

## Capture backends

- Windows default path: `pyaudiowpatch` using WASAPI loopback + mic and local mixdown
- Fallback path: `sounddevice` input stream when loopback setup is unavailable

## Canonical backend

- Use `companion_app.py` as the only active backend service.
- Legacy `server.py` has been removed to avoid split behavior/configuration.

## Planned next improvements

- Local LLM summarization via Ollama/llama.cpp profile modes
- Installer script for one-click setup
