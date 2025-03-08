from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import logging
from datetime import datetime
import PySimpleGUI as sg
import threading
import requests
import uvicorn
from queue import Queue

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
                logging.error(f"Error downloading {url}: {str(e)}")
                
        status.status = "Completed"
        status.current_url = None
        
    except Exception as e:
        status.status = f"Failed: {str(e)}"
        logging.error(f"Error processing downloads: {str(e)}")

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

def create_gui():
    sg.theme('LightGrey1')

    layout = [
        [sg.Text('🎥 Loom Video Downloader', font=('Helvetica', 20))],
        [sg.Text('Enter URLs (one per line):', font=('Helvetica', 10))],
        [sg.Multiline(size=(60, 10), key='urls')],
        [
            sg.Text('Max Size (MB):', font=('Helvetica', 10)),
            sg.Input('0', size=(10, 1), key='max_size'),
            sg.Text('Output Directory:', font=('Helvetica', 10)),
            sg.Input('downloads', size=(20, 1), key='output_dir'),
            sg.FolderBrowse()
        ],
        [sg.Button('Download', size=(20, 1), button_color=('white', '#FF4B4B'))],
        [sg.ProgressBar(100, orientation='h', size=(50, 20), key='progress')],
        [sg.Text('', key='status', size=(60, 1))],
        [sg.Text('', key='current_url', size=(60, 1))],
        [sg.Multiline(size=(60, 5), key='summary', disabled=True, visible=False)]
    ]

    return sg.Window('Loom Video Downloader', layout, finalize=True)

def monitor_download(window, download_id):
    while True:
        try:
            response = requests.get(f'http://localhost:8000/api/status/{download_id}')
            if response.status_code == 200:
                status = response.json()
                progress = ((status['completed'] + status['failed']) / status['total']) * 100
                
                window.write_event_value('-PROGRESS-', {
                    'progress': progress,
                    'status': status['status'],
                    'current_url': status['current_url'],
                    'completed': status['completed'],
                    'failed': status['failed'],
                    'total': status['total']
                })
                
                if status['status'] == 'Completed' or status['status'].startswith('Failed'):
                    break
                    
            else:
                break
                
        except Exception as e:
            logging.error(f"Error monitoring download: {e}")
            break
            
        sg.time.sleep(1)

def main():
    window = create_gui()
    
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={'host': '0.0.0.0', 'port': 8000, 'log_level': 'error'},
        daemon=True
    )
    api_thread.start()

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        if event == 'Download':
            urls = values['urls'].strip().split('\n')
            urls = [url.strip() for url in urls if url.strip()]
            
            if not urls:
                sg.popup_error('Please enter at least one URL')
                continue

            try:
                response = requests.post(
                    'http://localhost:8000/api/download',
                    json={
                        'urls': urls,
                        'max_size': float(values['max_size']),
                        'output_dir': values['output_dir']
                    }
                )
                
                if response.status_code == 200:
                    download_id = response.json()['download_id']
                    
                    # Reset progress and show progress elements
                    window['progress'].update(0)
                    window['summary'].update(visible=False)
                    
                    # Start monitoring in a separate thread
                    monitor_thread = threading.Thread(
                        target=monitor_download,
                        args=(window, download_id),
                        daemon=True
                    )
                    monitor_thread.start()
                    
            except Exception as e:
                sg.popup_error(f'Error starting download: {e}')

        elif event == '-PROGRESS-':
            # Update GUI with progress information
            progress_data = values[event]
            window['progress'].update(progress_data['progress'])
            window['status'].update(progress_data['status'])
            window['current_url'].update(progress_data['current_url'] or '')
            
            if progress_data['status'] == 'Completed':
                summary = (
                    f"Download Complete!\n\n"
                    f"✅ Successfully downloaded: {progress_data['completed']} videos\n"
                    f"❌ Failed downloads: {progress_data['failed']} videos\n"
                    f"📁 Total processed: {progress_data['total']} videos"
                )
                window['summary'].update(summary, visible=True)

    window.close()

if __name__ == "__main__":
    setup_logging()
    main()
