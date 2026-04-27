"""
FastAPI application for the Video Sequencer web interface.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# The web server runs long indexing requests in a non-interactive environment,
# so terminal progress bars can fail with BrokenPipeError when detached.
os.environ.setdefault("VIDEO_SEQUENCER_DISABLE_PROGRESS", "1")

from web_backend import (
    PROJECT_ROOT,
    SAFE_MEDIA_EXTENSIONS,
    WEBAPP_ROOT,
    benchmark_manager,
    build_job_command,
    cache_manager,
    editor_manager,
    job_manager,
    server_control_manager,
    settings_store,
)


STATIC_ROOT = WEBAPP_ROOT / "static"
INDEX_FILE = STATIC_ROOT / "index.html"

app = FastAPI(title="Video Sequencer Studio", version="0.3.0")
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


class DictPayload(BaseModel):
    payload: Dict[str, Any] = {}


@app.on_event("startup")
def register_server_process() -> None:
    settings = settings_store.load()
    server_control_manager.register_current_process(settings)


@app.on_event("shutdown")
def unregister_server_process() -> None:
    server_control_manager.unregister_current_process()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/api/bootstrap")
def bootstrap() -> Dict[str, Any]:
    settings = settings_store.load()
    return {
        "settings": settings,
        "server": server_control_manager.status(settings),
        "benchmarks": benchmark_manager.list(settings.get("benchmarks_dir", "./data/benchmarks")),
        "cache": cache_manager.inspect(settings.get("cache_dir", "./cache")),
        "jobs": job_manager.list_jobs(),
        "sessions": editor_manager.list_sessions(),
    }


@app.get("/api/media")
def media(path: str, download: bool = False, filename: Optional[str] = None) -> FileResponse:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (PROJECT_ROOT / file_path).resolve()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    if file_path.suffix.lower() not in SAFE_MEDIA_EXTENSIONS:
        raise HTTPException(status_code=403, detail="Unsupported media type.")
    
    headers = {}
    if download:
        dl_name = filename or file_path.name
        headers["Content-Disposition"] = f'attachment; filename="{dl_name}"'
        
    return FileResponse(file_path, headers=headers)


@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    return settings_store.load()


@app.post("/api/settings")
def update_settings(request: DictPayload) -> Dict[str, Any]:
    return settings_store.save(request.payload)


@app.get("/api/server")
def get_server_status() -> Dict[str, Any]:
    settings = settings_store.load()
    return server_control_manager.status(settings)


@app.post("/api/server/stop")
def stop_server() -> Dict[str, Any]:
    return server_control_manager.stop()


@app.post("/api/server/restart")
def restart_server() -> Dict[str, Any]:
    return server_control_manager.restart()


@app.get("/api/cache")
def get_cache() -> Dict[str, Any]:
    settings = settings_store.load()
    return cache_manager.inspect(settings.get("cache_dir", "./cache"))


@app.post("/api/cache/clear")
def clear_cache(request: DictPayload) -> Dict[str, Any]:
    settings = settings_store.load()
    action = request.payload.get("action", "all")
    return cache_manager.clear(settings.get("cache_dir", "./cache"), action)


@app.get("/api/benchmarks")
def get_benchmarks() -> List[Dict[str, Any]]:
    settings = settings_store.load()
    return benchmark_manager.list(settings.get("benchmarks_dir", "./data/benchmarks"))


@app.delete("/api/benchmarks/{benchmark_number}")
def delete_benchmark(benchmark_number: str) -> Dict[str, Any]:
    settings = settings_store.load()
    return benchmark_manager.delete(benchmark_number, settings.get("benchmarks_dir", "./data/benchmarks"))


@app.get("/api/benchmarks/{benchmark_number}/download")
def download_benchmark(benchmark_number: str) -> FileResponse:
    settings = settings_store.load()
    zip_path = benchmark_manager.export_zip(benchmark_number, settings.get("benchmarks_dir", "./data/benchmarks"))
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")


@app.post("/api/benchmarks/upload")
async def upload_benchmark(
    benchmark_number: str = Form(...),
    benchmark_title: str = Form(""),
    files: List[UploadFile] = File(...),
    relative_paths: List[str] = Form(...),
) -> Dict[str, Any]:
    if len(files) != len(relative_paths):
        raise HTTPException(status_code=400, detail="Uploaded files and relative paths are misaligned.")

    uploaded_paths: List[tuple[str, bytes]] = []
    for upload, relative_path in zip(files, relative_paths):
        payload = await upload.read()
        uploaded_paths.append((relative_path, payload))

    settings = settings_store.load()
    result = benchmark_manager.import_uploaded_files(
        benchmark_number,
        benchmark_title,
        settings.get("benchmarks_dir", "./data/benchmarks"),
        uploaded_paths,
    )
    return {
        "result": result,
        "benchmarks": benchmark_manager.list(settings.get("benchmarks_dir", "./data/benchmarks")),
    }


@app.get("/api/jobs")
def list_jobs() -> List[Dict[str, Any]]:
    return job_manager.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    try:
        return job_manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown job.") from exc


@app.post("/api/jobs/run/{action}")
def run_job(action: str, request: DictPayload) -> Dict[str, Any]:
    settings = settings_store.load()
    try:
        name, command = build_job_command(action, request.payload, settings)
        return job_manager.submit(name, command)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/editor/sessions")
def list_sessions() -> List[Dict[str, Any]]:
    return editor_manager.list_sessions()


@app.post("/api/editor/sessions")
def create_session(request: DictPayload) -> Dict[str, Any]:
    settings = settings_store.load()
    try:
        return editor_manager.create_session(request.payload, settings)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/editor/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    try:
        return editor_manager.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc

@app.delete("/api/editor/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    try:
        editor_manager.delete_session(session_id)
        return {"result": "ok"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc


@app.post("/api/editor/sessions/{session_id}/regenerate")
def regenerate_all(session_id: str) -> Dict[str, Any]:
    try:
        return editor_manager.regenerate_all(session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/sessions/{session_id}/assemble")
def assemble_session(session_id: str) -> Dict[str, Any]:
    try:
        return editor_manager.assemble(session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/editor/sessions/{session_id}/segments/export")
def export_segments(session_id: str) -> FileResponse:
    try:
        export_path = editor_manager.export_segments(session_id)
        return FileResponse(export_path, filename=export_path.name, media_type="application/json")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/sessions/{session_id}/segments/{segment_id}")
def update_segment(session_id: str, segment_id: int, request: DictPayload) -> Dict[str, Any]:
    try:
        return editor_manager.update_segment(session_id, segment_id, request.payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/sessions/{session_id}/segments/{segment_id}/split")
def split_segment(session_id: str, segment_id: int) -> Dict[str, Any]:
    try:
        return editor_manager.split_segment(session_id, segment_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/sessions/{session_id}/segments/{segment_id}/merge")
def merge_segment(session_id: str, segment_id: int) -> Dict[str, Any]:
    try:
        return editor_manager.merge_with_next(session_id, segment_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/sessions/{session_id}/segments/{segment_id}/move")
def move_segment(session_id: str, segment_id: int, request: DictPayload) -> Dict[str, Any]:
    try:
        return editor_manager.move_segment(session_id, segment_id, request.payload.get("direction", "down"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/sessions/{session_id}/segments/{segment_id}/select")
def select_candidate(session_id: str, segment_id: int, request: DictPayload) -> Dict[str, Any]:
    candidate_id = request.payload.get("candidate_id")
    if not candidate_id:
        raise HTTPException(status_code=400, detail="candidate_id is required.")
    try:
        return editor_manager.select_candidate(session_id, segment_id, candidate_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/sessions/{session_id}/segments/{segment_id}/regenerate")
def regenerate_segment(session_id: str, segment_id: int) -> Dict[str, Any]:
    try:
        return editor_manager.regenerate_segment(session_id, segment_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp:app", host="0.0.0.0", port=8000, reload=False)
