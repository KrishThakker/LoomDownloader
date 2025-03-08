from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import logging
from datetime import datetime
import uvicorn

# Import the download functionality
from loom_downloader import fetch_loom_download_url, download_loom_video, extract_id, format_size

app = FastAPI(
    title="Loom Video Downloader",
    description="Download Loom videos easily",
    version="1.0.0"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class DownloadRequest(BaseModel):
    urls: List[str]
    max_size: Optional[float] = 0
    output_dir: str = "downloads"

class DownloadStatus(BaseModel):
    id: str
    total: int
    completed: int
    failed: int
    current_url: Optional[str] = None
    status: str
    errors: List[dict] = []

# Store download status
active_downloads = {}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("download.log"),
            logging.StreamHandler()
        ]
    )

async def process_downloads(download_id: str, urls: List[str], max_size: float, output_dir: str):
    status = active_downloads[download_id]
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        for i, url in enumerate(urls):
            try:
                status.current_url = url
                status.status = f"Processing {i+1}/{len(urls)}"
                
                # Extract video ID and create filename
                video_id = extract_id(url)
                filename = os.path.join(output_dir, f"{video_id}.mp4")

                # Get download URL
                video_url = fetch_loom_download_url(video_id)

                # Download the video
                max_size_bytes = max_size * 1024 * 1024 if max_size > 0 else None
                download_loom_video(video_url, filename, max_size=max_size_bytes, use_tqdm=False)
                
                status.completed += 1
                
            except Exception as e:
                status.failed += 1
                status.errors.append({"url": url, "error": str(e)})
                
        status.status = "Completed"
        status.current_url = None
        
    except Exception as e:
        status.status = f"Failed: {str(e)}"

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/api/download")
async def start_download(request: DownloadRequest, background_tasks: BackgroundTasks):
    download_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize download status
    status = DownloadStatus(
        id=download_id,
        total=len(request.urls),
        completed=0,
        failed=0,
        status="Starting"
    )
    active_downloads[download_id] = status
    
    # Start download in background
    background_tasks.add_task(
        process_downloads,
        download_id,
        request.urls,
        request.max_size,
        request.output_dir
    )
    
    return {"download_id": download_id}

@app.get("/api/status/{download_id}")
async def get_status(download_id: str):
    if download_id not in active_downloads:
        raise HTTPException(status_code=404, detail="Download not found")
    
    return active_downloads[download_id]

if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app, host="0.0.0.0", port=8000)
