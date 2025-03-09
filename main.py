from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import logging
from datetime import datetime, timedelta, timezone
import PySimpleGUI as sg
import threading
import requests
import uvicorn
import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes, HTTPBearer
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends
import secrets
import hashlib
import bcrypt
from enum import Enum
from fastapi import Request
from fastapi.middleware.securityheaders import SecurityHeadersMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uuid
import re
from fastapi import Cookie
import ipaddress

# Import the download functionality
from loom_downloader import fetch_loom_download_url, download_loom_video, extract_id, format_size

# Constants
WINDOW_THEME = 'LightGrey1'
DEFAULT_PORT = 8000
DEFAULT_HOST = '0.0.0.0'

# Security constants
class SecurityConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    MIN_PASSWORD_LENGTH = 8
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = 15  # minutes
    PEPPER = os.getenv("PEPPER", secrets.token_hex(16))
    TRUSTED_HOSTS = ["localhost", "127.0.0.1"]
    CORS_ORIGINS = ["http://localhost:8501"]  # Streamlit default port
    SESSION_COOKIE_NAME = "session_id"
    SESSION_EXPIRE_MINUTES = 60
    PASSWORD_REGEX = r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$"
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
    RATE_LIMIT_WINDOW = 60  # seconds
    RATE_LIMIT_MAX_REQUESTS = 100
    ALLOWED_FILE_TYPES = [".mp4", ".mov", ".avi"]
    JWT_BLACKLIST = set()  # Store revoked tokens
    IP_WHITELIST = ["127.0.0.1", "::1"]  # Allowed IPs
    ADMIN_ROLES = {"admin", "superuser"}
    MAX_REQUEST_SIZE = 1024 * 1024  # 1MB
    API_KEY_HEADER = "X-API-Key"
    HASH_ALGORITHM = "sha256"
    REQUEST_TIMEOUT = 30  # seconds
    MAX_CONCURRENT_DOWNLOADS = 5
    SECURE_HEADERS = {
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin"
    }

# Password hashing
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Increase work factor
)

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

# Add security middleware
app.add_middleware(
    SecurityHeadersMiddleware,
    headers={
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=SecurityConfig.TRUSTED_HOSTS
)

# Add security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Check IP whitelist
    client_ip = request.client.host
    if not any(ipaddress.ip_address(client_ip) in ipaddress.ip_network(allowed)
               for allowed in SecurityConfig.IP_WHITELIST):
        return JSONResponse(
            status_code=403,
            content={"detail": "IP address not allowed"}
        )

    # Add security headers
    response = await call_next(request)
    for header, value in SecurityConfig.SECURE_HEADERS.items():
        response.headers[header] = value
    
    return response

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
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class User(BaseModel):
    username: str
    email: Optional[str] = None
    disabled: bool = False
    role: UserRole = UserRole.USER

class UserInDB(User):
    hashed_password: str
    salt: str
    last_login: Optional[datetime] = None
    failed_attempts: int = 0

# Mock user database - Replace with real database in production
users_db = {
    "admin": {
        "username": "root",
        "hashed_password": pwd_context.hash("admin123*"),  # Change in production
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
    encoded_jwt = jwt.encode(to_encode, SecurityConfig.SECRET_KEY, algorithm=SecurityConfig.ALGORITHM)
    return encoded_jwt

# Add token blacklist management
class TokenBlacklist:
    def __init__(self):
        self._blacklist: Dict[str, datetime] = {}
    
    def add_token(self, token: str):
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self._blacklist[token_hash] = datetime.now(timezone.utc)
    
    def is_blacklisted(self, token: str) -> bool:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return token_hash in self._blacklist
    
    def cleanup(self):
        """Remove expired tokens"""
        now = datetime.now(timezone.utc)
        self._blacklist = {
            token: timestamp
            for token, timestamp in self._blacklist.items()
            if (now - timestamp).days < 7
        }

token_blacklist = TokenBlacklist()

# Update user authentication
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    request: Request = None
):
    if token_blacklist.is_blacklisted(token):
        raise HTTPException(
            status_code=401,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SecurityConfig.SECRET_KEY, algorithms=[SecurityConfig.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
            
        # Check token expiration
        exp = payload.get("exp")
        if not exp or datetime.fromtimestamp(exp) < datetime.now(timezone.utc):
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
        
    user = get_user(users_db, username)
    if user is None:
        raise credentials_exception
        
    return user

# Add admin-only decorator
def admin_only(func):
    async def wrapper(*args, current_user: User = Depends(get_current_user), **kwargs):
        if current_user.role not in SecurityConfig.ADMIN_ROLES:
            raise HTTPException(
                status_code=403,
                detail="Admin privileges required"
            )
        return await func(*args, current_user=current_user, **kwargs)
    return wrapper

# Add secure logout endpoint
@app.post("/api/v1/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    token_blacklist.add_token(token)
    return {"message": "Successfully logged out"}

# Add admin endpoints
@app.get("/api/v1/admin/users")
@admin_only
async def list_users(current_user: User = Depends(get_current_user)):
    """List all users (admin only)"""
    return {"users": list(users_db.keys())}

@app.post("/api/v1/admin/revoke/{username}")
@admin_only
async def revoke_user_sessions(
    username: str,
    current_user: User = Depends(get_current_user)
):
    """Revoke all sessions for a user (admin only)"""
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Implement session revocation logic here
    return {"message": f"All sessions revoked for user {username}"}

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

# Add session management
class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_times = {}

    def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = user_id
        self.session_times[session_id] = datetime.now()
        return session_id

    def validate_session(self, session_id: str) -> Optional[str]:
        if session_id in self.sessions:
            session_time = self.session_times[session_id]
            if datetime.now() - session_time > timedelta(minutes=SecurityConfig.SESSION_EXPIRE_MINUTES):
                self.remove_session(session_id)
                return None
            self.session_times[session_id] = datetime.now()  # Update last access
            return self.sessions[session_id]
        return None

    def remove_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            del self.session_times[session_id]

session_manager = SessionManager()

# Add input validation functions
def validate_password(password: str) -> bool:
    """Validate password strength"""
    return bool(re.match(SecurityConfig.PASSWORD_REGEX, password))

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    return os.path.basename(filename)

def validate_file_type(filename: str) -> bool:
    """Validate file extension"""
    return any(filename.lower().endswith(ext) for ext in SecurityConfig.ALLOWED_FILE_TYPES)

# Update the login endpoint with enhanced security
@app.post("/token", response_model=Token)
async def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None
):
    client_ip = request.client.host
    
    # Check rate limiting
    if rate_limiter.is_rate_limited(client_ip, "login"):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )

    try:
        # Validate input length to prevent buffer overflow
        if len(form_data.username) > 100 or len(form_data.password) > 100:
            raise HTTPException(status_code=400, detail="Invalid input length")

        user = authenticate_user(users_db, form_data.username, form_data.password)
        if not user:
            rate_limiter.record_login_attempt(form_data.username, False)
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        rate_limiter.record_login_attempt(form_data.username, True)
        
        # Create session
        session_id = session_manager.create_session(user.username)
        
        # Generate tokens
        access_token_expires = timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
        refresh_token_expires = timedelta(days=SecurityConfig.REFRESH_TOKEN_EXPIRE_DAYS)
        
        access_token = create_access_token(
            data={
                "sub": user.username,
                "role": user.role,
                "session": session_id
            },
            expires_delta=access_token_expires
        )
        
        refresh_token = create_access_token(
            data={
                "sub": user.username,
                "refresh": True,
                "session": session_id
            },
            expires_delta=refresh_token_expires
        )

        # Set secure cookie
        response.set_cookie(
            key=SecurityConfig.SESSION_COOKIE_NAME,
            value=session_id,
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=SecurityConfig.SESSION_EXPIRE_MINUTES * 60
        )

        # Update last login and reset failed attempts
        users_db[user.username].last_login = datetime.now()
        users_db[user.username].failed_attempts = 0
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
        
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Update the download endpoint with additional security
@app.post("/api/v1/download")
async def api_download(
    request: Request,
    urls: List[str],
    max_size: Optional[float] = 0,
    output_dir: Optional[str] = "downloads",
    rename_pattern: Optional[str] = "{id}",
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)
):
    """Download videos with enhanced security and limits"""
    
    # Check concurrent downloads
    user_downloads = sum(
        1 for status in active_downloads.values()
        if status.status not in {"Completed", "Failed", "Cancelled"}
    )
    if user_downloads >= SecurityConfig.MAX_CONCURRENT_DOWNLOADS:
        raise HTTPException(
            status_code=429,
            detail="Maximum concurrent downloads reached"
        )
    
    try:
        # Validate request size
        content_length = request.headers.get("content-length", 0)
        if int(content_length) > SecurityConfig.MAX_CONTENT_LENGTH:
            raise HTTPException(status_code=413, detail="Request too large")

        # Validate max_size
        if max_size < 0 or max_size > 10000:  # 10GB limit
            raise HTTPException(status_code=400, detail="Invalid max size")

        # Sanitize output directory
        output_dir = sanitize_filename(output_dir)
        
        # Validate URLs
        for url in urls:
            if not validate_loom_url(url):
                raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")

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

# Add rate limiting
class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.login_attempts = {}
        self.locked_accounts = {}

    def is_rate_limited(self, ip: str, endpoint: str) -> bool:
        now = datetime.now()
        key = f"{ip}:{endpoint}"
        
        if key in self.requests:
            requests = [t for t in self.requests[key] if now - t < timedelta(minutes=1)]
            self.requests[key] = requests
            if len(requests) >= 60:  # 60 requests per minute
                return True
        
        if key not in self.requests:
            self.requests[key] = []
        self.requests[key].append(now)
        return False

    def record_login_attempt(self, username: str, success: bool):
        now = datetime.now()
        
        if username in self.locked_accounts:
            if now - self.locked_accounts[username] < timedelta(minutes=SecurityConfig.LOCKOUT_DURATION):
                raise HTTPException(
                    status_code=429,
                    detail=f"Account locked. Try again in {SecurityConfig.LOCKOUT_DURATION} minutes."
                )
            del self.locked_accounts[username]
        
        if not success:
            if username not in self.login_attempts:
                self.login_attempts[username] = 1
            else:
                self.login_attempts[username] += 1
                
            if self.login_attempts[username] >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
                self.locked_accounts[username] = now
                self.login_attempts[username] = 0
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many failed attempts. Account locked for {SecurityConfig.LOCKOUT_DURATION} minutes."
                )
        else:
            if username in self.login_attempts:
                del self.login_attempts[username]

rate_limiter = RateLimiter()

# Security utility functions
def get_password_hash(password: str, salt: bytes = None) -> tuple[str, str]:
    if salt is None:
        salt = bcrypt.gensalt()
    
    # Add pepper to password
    peppered = password + SecurityConfig.PEPPER
    
    # Hash with salt
    hashed = pwd_context.hash(peppered + salt.decode())
    return hashed, salt.decode()

def verify_password(plain_password: str, hashed_password: str, salt: str) -> bool:
    peppered = plain_password + SecurityConfig.PEPPER
    return pwd_context.verify(peppered + salt, hashed_password)

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
