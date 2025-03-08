from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from main import fetch_loom_download_url, download_loom_video, extract_id, format_size
import logging
from threading import Thread
from queue import Queue
import time
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Loom Video Downloader")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Global download queue and status dictionary
download_queue = Queue()
download_status = {}

class DownloadRequest(BaseModel):
    urls: List[str]
    output_dir: str = "downloads"
    max_size: float = 0

def process_download_queue():
    while True:
        if not download_queue.empty():
            download_id, url, output_dir, max_size = download_queue.get()
            try:
                video_id = extract_id(url)
                filename = os.path.join(output_dir, f"{video_id}.mp4")
                
                download_status[download_id].update({
                    "status": "fetching",
                    "message": "Fetching download URL...",
                    "progress": 0
                })
                
                video_url = fetch_loom_download_url(video_id)
                os.makedirs(output_dir, exist_ok=True)
                
                download_status[download_id].update({
                    "status": "downloading",
                    "message": "Downloading video...",
                    "progress": 10
                })
                
                max_size_bytes = max_size * 1024 * 1024 if max_size > 0 else None
                download_loom_video(video_url, filename, max_size=max_size_bytes, use_tqdm=False)
                
                download_status[download_id].update({
                    "status": "completed",
                    "message": "Download completed successfully!",
                    "filename": filename,
                    "progress": 100
                })
                
            except Exception as e:
                download_status[download_id].update({
                    "status": "error",
                    "message": str(e),
                    "progress": 0
                })
            
            download_queue.task_done()
        time.sleep(0.1)

# Start download queue processor
download_thread = Thread(target=process_download_queue, daemon=True)
download_thread.start()

@app.post("/api/download")
async def download(request: DownloadRequest):
    if not request.urls:
        raise HTTPException(status_code=400, detail="Please enter at least one URL")
    
    download_ids = []
    for url in request.urls:
        download_id = f"{time.time()}_{url[-8:]}"
        download_status[download_id] = {
            "url": url,
            "status": "queued",
            "message": "Waiting in queue...",
            "progress": 0
        }
        download_queue.put((download_id, url, request.output_dir, request.max_size))
        download_ids.append(download_id)
    
    return {"download_ids": download_ids}

@app.get("/api/status/{download_id}")
async def status(download_id: str):
    if download_id not in download_status:
        raise HTTPException(status_code=404, detail="Download ID not found")
    return download_status[download_id]

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("downloads", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# Mount the static files directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 