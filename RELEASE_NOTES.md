# Release Notes

## 2026-04-17

### Major addition: browser-based editing studio

This release adds a new web application for the repository, designed around the interaction model from *Write-A-Video: Computational Video Montage from Themed Text* while reusing the existing Python pipeline wherever possible.

### What changed

- Added a FastAPI web app entrypoint at `src/webapp.py`.
- Added a reusable web backend service layer in `src/web_backend.py`.
- Added a full browser UI under `webapp/static/`.
- Added a Write-A-Video-inspired editing workspace with:
  - segment-oriented text editing
  - candidate shot browsing
  - preview playback
  - keyword suggestions
  - movement and duration idiom controls
  - split/merge segment operations
  - render/export actions
- Added web-accessible versions of the existing menu functionality:
  - Quick Benchmark
  - Full Pipeline
  - OpenCLIP Grid Search
  - VideoPrism Grid Search
  - Write-A-Video Grid Search
  - Compare All Models
  - Benchmark upload/download/delete
  - Cache management
  - Settings editing
  - Background job monitoring and logs

### Backend/runtime improvements

- Fixed `WriteAVideoMatcher.match_segment_to_videos()` so it now performs actual two-stage retrieval instead of falling back directly to the plain OpenCLIP matcher.
- Added candidate metadata needed by the web editor, including keyword-score-aware ranking and matched keyword reporting.
- Added silent assembly support in `VideoSequenceBuilder.build_sequence()`, allowing editor renders even when no voice-over file is attached.
- Added session persistence for the web editor, so editing state can be reopened from the web UI.
- Added thumbnail generation and lightweight motion scoring for candidate review in the editor.

### Dependency updates

Added the following runtime dependencies to `requirements.txt`:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `open_clip_torch`
- `scikit-learn`

Also documented optional extras for richer Write-A-Video keyword indexing:

- `ultralytics`
- `deepface`

### Documentation updates

- Updated `README.md` with web-app usage and editor notes.
- Updated `INSTALL.md` with web-app startup guidance.
- Updated `docs/deployment.md` with remote-server web deployment steps.

### Validation

Completed locally:

- `python3 -m py_compile src/web_backend.py src/webapp.py src/wav_indexing.py src/assembly.py`
- `python3 -c "import sys; sys.path.insert(0, 'src'); import webapp; print(webapp.app.title)"`
- FastAPI bootstrap smoke test via test client
- Live HTTP smoke test by starting `python3 src/webapp.py` and requesting:
  - `GET /`
  - `GET /api/bootstrap`

Attempted remotely:

- On 2026-04-17, SSH validation against `neghvar.ced.tuc.gr:22` timed out from this environment before authentication, so server-side runtime validation could not be completed here.

## 2026-04-27

### Remote deployment follow-up

This follow-up release fixes issues that only appeared once the web studio was exercised on the remote server with real benchmark data and the production-style detached Uvicorn process.

### What changed

- Fixed manual benchmark segment loading for editor sessions so benchmark files using `start_s` and `end_s` retain their authored timings instead of collapsing to the minimum fallback duration.
- Disabled terminal-only progress bars for web-server indexing requests, which removes a `Broken pipe` failure mode seen when creating fresh `writeavideo` sessions on the remote server.
- Added per-session `create_session_error.log` output under `webapp/state/sessions/<session_id>/` so server-side initialization failures leave an inspectable traceback.
- Applied the same non-interactive progress handling to VideoPrism, OpenCLIP, and Write-A-Video indexing paths to keep the web deployment stable across retrieval modes.
- Added built-in server controls to the `Operations` tab, including live server status, a one-click restart action, a stop action, and the exact restart command for SSH recovery.

### Remote validation

Validated on `neghvar.ced.tuc.gr` on 2026-04-27:

- Confirmed the remote checkout is on branch `codex/write-a-video-web-studio`.
- Confirmed benchmark-backed editor sessions open correctly through the live web UI.
- Confirmed `openclip` editor sessions preserve benchmark-authored timings in the text editor.
- Reproduced the live `writeavideo` session-creation failure from both HTTP requests and the browser UI before applying the progress-bar fix.
- Confirmed fresh `writeavideo` sessions can be created successfully through both the HTTP API and the browser UI after the fix.
- Confirmed save/edit actions still work on a fresh `writeavideo` session after creation.
- Confirmed assembled render output is produced successfully for a fresh `writeavideo` session on the remote server.
- Confirmed the new in-app `Restart Server` control successfully restarts the live Uvicorn process and reconnects the UI.
