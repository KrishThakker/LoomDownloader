from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import json
import logging
from datetime import datetime, timedelta
import PySimpleGUI as sg
import threading
import requests
import uvicorn
import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends

# Import the download functionality
from loom_downloader import fetch_loom_download_url, download_loom_video, extract_id, format_size

# Constants
WINDOW_THEME = 'LightGrey1'
DEFAULT_PORT = 8000
DEFAULT_HOST = '0.0.0.0'

# Security constants
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(
    title="Loom Video Downloader",
    description="Download Loom videos easily",
    version="1.0.0"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models with enhanced validation
class DownloadRequest(BaseModel):
    urls: List[str]
    max_size: Optional[float] = 0
    output_dir: str = "downloads"
    rename_pattern: Optional[str] = "{id}"

    class Config:
        schema_extra = {
            "example": {
                "urls": ["https://www.loom.com/share/example-id"],
                "max_size": 100,
                "output_dir": "downloads",
                "rename_pattern": "{id}"
            }
        }

class DownloadStatus(BaseModel):
    id: str
    total: int
    completed: int
    failed: int
    current_url: Optional[str] = None
    status: str
    errors: List[dict] = []
    speed: Optional[float] = 0  # Download speed in MB/s
    eta: Optional[int] = None   # Estimated time remaining in seconds

# Store download status
active_downloads = {}

# User model
class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

# Mock user database - Replace with real database in production
users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),  # Change in production
        "disabled": False
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(users_db, username)
    if user is None:
        raise credentials_exception
    return user

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

@app.post("/api/v1/download")
async def api_download(
    urls: List[str],
    max_size: Optional[float] = 0,
    output_dir: Optional[str] = "downloads",
    rename_pattern: Optional[str] = "{id}",
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)  # Add authentication
):
    """
    Download videos from Loom URLs
    
    - **urls**: List of Loom video URLs
    - **max_size**: Maximum file size in MB (0 for no limit)
    - **output_dir**: Directory to save downloaded files
    - **rename_pattern**: Pattern for renaming files
    """
    try:
        # Validate max_size
        if max_size < 0:
            raise HTTPException(status_code=400, detail="Max size must be a non-negative number")

        # Generate download ID
        download_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize download status
        status = DownloadStatus(
            id=download_id,
            total=len(urls),
            completed=0,
            failed=0,
            status="Starting"
        )
        active_downloads[download_id] = status
        
        # Start download in background
        background_tasks.add_task(
            process_downloads,
            download_id,
            urls,
            max_size,
            output_dir
        )
        
        logging.info(f"API Download started: {download_id} with {len(urls)} URLs")
        
        return {
            "status": "success",
            "message": "Download started",
            "download_id": download_id,
            "total_urls": len(urls)
        }
        
    except Exception as e:
        logging.error(f"API Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/downloads")
async def list_downloads():
    """List all active downloads"""
    try:
        return {
            "downloads": [
                {
                    "id": download_id,
                    "status": status.dict()
                }
                for download_id, status in active_downloads.items()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/download/{download_id}")
async def get_download_status(download_id: str):
    """Get status of a specific download"""
    try:
        if download_id not in active_downloads:
            raise HTTPException(status_code=404, detail="Download not found")
        
        status = active_downloads[download_id]
        return {
            "id": download_id,
            "status": status.dict(),
            "details": {
                "progress": ((status.completed + status.failed) / status.total * 100 
                           if status.total > 0 else 0),
                "completed_urls": status.completed,
                "failed_urls": status.failed,
                "total_urls": status.total,
                "current_url": status.current_url,
                "errors": status.errors
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/download/{download_id}")
async def cancel_download(download_id: str):
    """Cancel a specific download"""
    try:
        if download_id not in active_downloads:
            raise HTTPException(status_code=404, detail="Download not found")
        
        status = active_downloads[download_id]
        status.status = "Cancelled"
        
        return {"status": "success", "message": f"Download {download_id} cancelled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add API documentation info
@app.get("/api/v1/docs")
async def get_api_docs():
    """Get API documentation"""
    return {
        "endpoints": {
            "POST /api/v1/download": "Start a new download",
            "GET /api/v1/downloads": "List all downloads",
            "GET /api/v1/download/{download_id}": "Get download status",
            "DELETE /api/v1/download/{download_id}": "Cancel download"
        },
        "example_request": {
            "urls": ["https://www.loom.com/share/example-id"],
            "max_size": 100,
            "output_dir": "downloads",
            "rename_pattern": "{id}"
        }
    }

# Add login endpoint
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def format_time(seconds: int) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds//60}m {seconds%60}s"
    else:
        return f"{seconds//3600}h {(seconds%3600)//60}m"

def create_gui():
    """Create the GUI with enhanced styling and features."""
    sg.theme(WINDOW_THEME)

    layout = [
        [sg.Text('🎥 Loom Video Downloader', font=('Helvetica', 20), pad=(0, 10))],
        [sg.Text('Enter URLs (one per line):', font=('Helvetica', 10))],
        [sg.Multiline(size=(60, 10), key='urls', font=('Helvetica', 10))],
        [
            sg.Frame('Settings', [
                [
                    sg.Text('Max Size (MB):', font=('Helvetica', 10)),
                    sg.Input('0', size=(10, 1), key='max_size'),
                    sg.Text('Output Directory:', font=('Helvetica', 10)),
                    sg.Input('downloads', size=(20, 1), key='output_dir'),
                    sg.FolderBrowse()
                ],
                [
                    sg.Text('Rename Pattern:', font=('Helvetica', 10)),
                    sg.Input('{id}', size=(20, 1), key='rename_pattern'),
                    sg.Checkbox('Skip Existing', key='skip_existing', default=True)
                ]
            ])
        ],
        [sg.Button('Download', size=(20, 1), button_color=('white', '#FF4B4B'))],
        [sg.ProgressBar(100, orientation='h', size=(50, 20), key='progress')],
        [sg.Text('Status:', font=('Helvetica', 10, 'bold')), sg.Text('', key='status', size=(50, 1))],
        [sg.Text('Speed:', font=('Helvetica', 10, 'bold')), sg.Text('', key='speed', size=(20, 1))],
        [sg.Text('ETA:', font=('Helvetica', 10, 'bold')), sg.Text('', key='eta', size=(20, 1))],
        [sg.Text('Current URL:', font=('Helvetica', 10, 'bold')), sg.Text('', key='current_url', size=(50, 1))],
        [sg.Multiline(size=(60, 5), key='summary', disabled=True, visible=False)]
    ]

    return sg.Window(
        'Loom Video Downloader',
        layout,
        finalize=True,
        resizable=True,
        return_keyboard_events=True
    )

def monitor_download(window, download_id):
    """Monitor download progress with enhanced feedback."""
    start_time = time.time()
    
    while True:
        try:
            response = requests.get(f'http://localhost:{DEFAULT_PORT}/api/status/{download_id}')
            if response.status_code == 200:
                status = response.json()
                progress = ((status['completed'] + status['failed']) / status['total']) * 100
                elapsed_time = time.time() - start_time
                
                # Calculate speed and ETA
                speed = status['completed'] / elapsed_time if elapsed_time > 0 else 0
                remaining = status['total'] - (status['completed'] + status['failed'])
                eta = int(remaining / speed) if speed > 0 else 0
                
                window.write_event_value('-PROGRESS-', {
                    'progress': progress,
                    'status': status['status'],
                    'current_url': status['current_url'],
                    'completed': status['completed'],
                    'failed': status['failed'],
                    'total': status['total'],
                    'speed': speed,
                    'eta': eta
                })
                
                if status['status'] == 'Completed' or status['status'].startswith('Failed'):
                    break
                    
            else:
                window.write_event_value('-ERROR-', 'Failed to get download status')
                break
                
        except Exception as e:
            logging.error(f"Error monitoring download: {e}")
            window.write_event_value('-ERROR-', str(e))
            break
            
        time.sleep(1)

def main():
    window = create_gui()
    
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={'host': DEFAULT_HOST, 'port': DEFAULT_PORT, 'log_level': 'error'},
        daemon=True
    )
    api_thread.start()

    while True:
        event, values = window.read(timeout=100)

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
                    f'http://localhost:{DEFAULT_PORT}/api/download',
                    json={
                        'urls': urls,
                        'max_size': float(values['max_size']),
                        'output_dir': values['output_dir'],
                        'rename_pattern': values['rename_pattern']
                    }
                )
                
                if response.status_code == 200:
                    download_id = response.json()['download_id']
                    
                    # Reset progress elements
                    for key in ['progress', 'status', 'speed', 'eta', 'current_url']:
                        window[key].update('')
                    window['summary'].update(visible=False)
                    
                    # Start monitoring in a separate thread
                    monitor_thread = threading.Thread(
                        target=monitor_download,
                        args=(window, download_id),
                        daemon=True
                    )
                    monitor_thread.start()
                else:
                    sg.popup_error('Failed to start download: Server error')
            except Exception as e:
                sg.popup_error(f'Error starting download: {e}')

        elif event == '-PROGRESS-':
            # Update GUI with progress information
            progress_data = values[event]
            window['progress'].update(progress_data['progress'])
            window['status'].update(progress_data['status'])
            window['current_url'].update(progress_data['current_url'] or '')
            window['speed'].update(f"{progress_data['speed']:.1f} videos/min")
            window['eta'].update(format_time(progress_data['eta']))
            
            if progress_data['status'] == 'Completed':
                summary = (
                    f"Download Complete!\n\n"
                    f"✅ Successfully downloaded: {progress_data['completed']} videos\n"
                    f"❌ Failed downloads: {progress_data['failed']} videos\n"
                    f"📁 Total processed: {progress_data['total']} videos\n"
                    f"⏱️ Total time: {format_time(int(progress_data['eta']))}"
                )
                window['summary'].update(summary, visible=True)

        elif event == '-ERROR-':
            sg.popup_error(f'Error: {values[event]}')

    window.close()

if __name__ == "__main__":
    setup_logging()
    main()
