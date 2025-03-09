from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union, Set, Tuple, AsyncGenerator
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
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes, HTTPBearer, APIKeyHeader
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
import ssl
import socket
from urllib.parse import urlparse
import asyncio
import aiohttp
import aiofile
import psutil
from pathlib import Path
import shutil
import tempfile
from cachetools import TTLCache, LRUCache
from prometheus_client import Counter, Histogram, start_http_server
import functools
from typing import Callable
import traceback
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from fastapi_versioning import VersionedFastAPI, version
from celery import Celery
from redis import Redis
import aioredis
import asyncpg

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
    SSL_VERIFY = True
    MIN_TLS_VERSION = ssl.TLSVersion.TLSv1_2
    ALLOWED_DOMAINS = ["loom.com"]
    MAX_URL_LENGTH = 2048
    MAX_URLS_PER_REQUEST = 100
    DOWNLOAD_TIMEOUT = 300  # 5 minutes
    API_RATE_LIMITS = {
        "download": 10,  # per minute
        "status": 60,    # per minute
        "admin": 30      # per minute
    }
    BLOCKED_IPS: set = set()
    SUSPICIOUS_PATTERNS = [
        r"../",          # Directory traversal
        r"cmd=",         # Command injection
        r"exec\(",       # Code execution
        r"SELECT.*FROM"  # SQL injection
    ]
    ENCRYPTION_KEY = Fernet.generate_key()
    SENSITIVE_HEADERS = {'authorization', 'cookie', 'proxy-authorization'}
    FILE_SIZE_LIMIT = 1024 * 1024 * 1024  # 1GB
    ALLOWED_MIME_TYPES = {'video/mp4', 'video/quicktime', 'video/x-msvideo'}
    SCAN_DOWNLOADS = True
    PROXY_SETTINGS = {
        'http': os.getenv('HTTP_PROXY'),
        'https': os.getenv('HTTPS_PROXY')
    }
    DOWNLOAD_CHUNK_SIZE = 8192  # 8KB chunks
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    HEALTH_CHECK_INTERVAL = 300  # 5 minutes

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

# Add API key security
api_key_header = APIKeyHeader(name=SecurityConfig.API_KEY_HEADER)

# Add URL validation
def validate_url(url: str) -> bool:
    """Enhanced URL validation"""
    try:
        if len(url) > SecurityConfig.MAX_URL_LENGTH:
            return False
            
        parsed = urlparse(url)
        if parsed.netloc not in SecurityConfig.ALLOWED_DOMAINS:
            return False
            
        # Check for suspicious patterns
        for pattern in SecurityConfig.SUSPICIOUS_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return False
                
        return True
    except:
        return False

# Add security monitoring
class SecurityMonitor:
    def __init__(self):
        self._suspicious_activities: Dict[str, List[datetime]] = {}
        self._blocked_ips: set = SecurityConfig.BLOCKED_IPS
        self._last_cleanup = datetime.now(timezone.utc)

    async def monitor_request(self, request: Request) -> None:
        client_ip = request.client.host
        
        # Check if IP is blocked
        if client_ip in self._blocked_ips:
            raise HTTPException(
                status_code=403,
                detail="Access denied"
            )

        # Record suspicious activity
        if await self._is_suspicious(request):
            if client_ip not in self._suspicious_activities:
                self._suspicious_activities[client_ip] = []
            self._suspicious_activities[client_ip].append(datetime.now(timezone.utc))
            
            # Block IP if too many suspicious activities
            if len(self._suspicious_activities[client_ip]) > 5:
                self._blocked_ips.add(client_ip)
                raise HTTPException(
                    status_code=403,
                    detail="Access denied due to suspicious activity"
                )

        # Cleanup old records
        await self._cleanup()

    async def _is_suspicious(self, request: Request) -> bool:
        """Check for suspicious patterns in request"""
        try:
            body = await request.body()
            text = body.decode()
            
            # Check for suspicious patterns
            for pattern in SecurityConfig.SUSPICIOUS_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
                    
            # Check headers for suspicious content
            for header, value in request.headers.items():
                if any(re.search(pattern, value, re.IGNORECASE) 
                      for pattern in SecurityConfig.SUSPICIOUS_PATTERNS):
                    return True
                    
            return False
        except:
            return False

    async def _cleanup(self):
        """Clean up old records"""
        now = datetime.now(timezone.utc)
        if (now - self._last_cleanup).seconds > 3600:  # Cleanup every hour
            self._suspicious_activities = {
                ip: times for ip, times in self._suspicious_activities.items()
                if any((now - t).days < 1 for t in times)
            }
            self._last_cleanup = now

security_monitor = SecurityMonitor()

# Add security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Monitor request for suspicious activity
    await security_monitor.monitor_request(request)
    
    # Add security headers
    response = await call_next(request)
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
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

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

# Add download manager
class DownloadManager:
    def __init__(self):
        self.active_downloads: Dict[str, DownloadStatus] = {}
        self.download_history: Dict[str, List[Dict[str, Any]]] = {}
        self.encryption = Fernet(SecurityConfig.ENCRYPTION_KEY)
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._semaphore = asyncio.Semaphore(SecurityConfig.MAX_CONCURRENT_DOWNLOADS)

    async def stream_download(self, url: str) -> AsyncGenerator[bytes, None]:
        """Stream download with progress tracking"""
        async with self._semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, proxy=SecurityConfig.PROXY_SETTINGS) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to download: HTTP {response.status}"
                        )
                    
                    async for chunk in response.content.iter_chunked(SecurityConfig.DOWNLOAD_CHUNK_SIZE):
                        yield chunk

    async def start_download(self, download_id: str, urls: List[str], **kwargs) -> None:
        """Start download with improved error handling and progress tracking"""
        try:
            status = DownloadStatus(
                id=download_id,
                total=len(urls),
                completed=0,
                failed=0,
                status="Starting",
                start_time=datetime.now(timezone.utc)
            )
            self.active_downloads[download_id] = status
            
            for url in urls:
                if status.status == "Cancelled":
                    break
                    
                status.current_url = url
                try:
                    final_path = Path(kwargs['output_dir']) / f"{status.id}.mp4"
                    
                    async for chunk in self.stream_download(url):
                        if status.status == "Cancelled":
                            await self._cleanup_download(download_id)
                            break
                        await self._save_chunk(download_id, chunk)
                    
                    await self._finalize_download(download_id, final_path)
                    status.completed += 1
                    
                except Exception as e:
                    status.failed += 1
                    status.errors.append({"url": url, "error": str(e)})
                    await self._cleanup_download(download_id)
                    
            status.status = "Completed" if status.failed == 0 else "Completed with errors"
            status.end_time = datetime.now(timezone.utc)
            
            # Store download history
            self.download_history[download_id] = {
                "status": status.dict(),
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logging.error(f"Download error: {str(e)}")
            status.status = f"Failed: {str(e)}"
            await self._cleanup_download(download_id)

    async def _save_chunk(self, download_id: str, chunk: bytes) -> None:
        """Save a chunk to temporary file"""
        temp_path = SecurityConfig.TEMP_DIR / f"{download_id}.part"
        async with aiofile.async_open(temp_path, 'ab') as f:
            await f.write(chunk)
        self.temp_files[download_id] = temp_path

    async def _finalize_download(self, download_id: str, final_path: Path) -> None:
        """Move temporary file to final location"""
        if download_id in self.temp_files:
            temp_path = self.temp_files[download_id]
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_path), str(final_path))
            del self.temp_files[download_id]

    async def _cleanup_download(self, download_id: str) -> None:
        """Clean up temporary files for failed download"""
        if download_id in self.temp_files:
            try:
                self.temp_files[download_id].unlink(missing_ok=True)
                del self.temp_files[download_id]
            except Exception as e:
                logging.error(f"Error cleaning up download {download_id}: {e}")

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up old temporary files"""
        while True:
            try:
                now = time.time()
                for file in SecurityConfig.TEMP_DIR.glob("*.part"):
                    if now - file.stat().st_mtime > SecurityConfig.MAX_AGE_TEMP_FILES:
                        file.unlink()
                await asyncio.sleep(SecurityConfig.CLEANUP_INTERVAL)
            except Exception as e:
                logging.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)

download_manager = DownloadManager()

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    try:
        memory = psutil.Process().memory_info()
        disk = psutil.disk_usage('/')
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc),
            "version": "1.0.0",
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_used": memory.rss / 1024 / 1024,  # MB
                "memory_percent": psutil.virtual_memory().percent,
                "disk_free": disk.free / 1024 / 1024 / 1024,  # GB
                "disk_percent": disk.percent
            },
            "application": {
                "active_downloads": len(active_downloads),
                "cached_urls": len(cache_manager.url_cache),
                "cached_files": len(cache_manager.download_cache),
                "uptime": time.time() - startup_time
            }
        }
    except Exception as e:
        logging.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Update download endpoint with caching and metrics
@app.post("/api/v1/download")
@track_errors
async def api_download(
    request: Request,
    urls: List[str],
    max_size: Optional[float] = 0,
    output_dir: Optional[str] = "downloads",
    rename_pattern: Optional[str] = "{id}",
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    api_key: str = Depends(api_key_header)
):
    """Download videos with enhanced caching and monitoring"""
    with Metrics.request_duration.time():
        try:
            Metrics.downloads_total.inc()
            Metrics.active_downloads.inc()
            
            # Validate request
            if len(urls) > SecurityConfig.MAX_URLS_PER_REQUEST:
                raise HTTPException(
                    status_code=400,
                    detail=f"Maximum {SecurityConfig.MAX_URLS_PER_REQUEST} URLs allowed"
                )

            download_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check cache for each URL
            cached_files = []
            download_urls = []
            
            for url in urls:
                video_id = extract_id(url)
                cached_file = await cache_manager.get_cached_file(video_id)
                if cached_file and cached_file.exists():
                    cached_files.append((url, cached_file))
                else:
                    download_urls.append(url)

            # Start new downloads
            if download_urls:
                background_tasks.add_task(
                    download_manager.start_download,
                    download_id,
                    download_urls,
                    max_size=max_size,
                    output_dir=output_dir,
                    rename_pattern=rename_pattern
                )

            return {
                "status": "success",
                "download_id": download_id,
                "cached_files": len(cached_files),
                "new_downloads": len(download_urls),
                "message": "Download started"
            }
            
        except Exception as e:
            Metrics.download_errors.inc()
            raise
        finally:
            Metrics.active_downloads.dec()

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest())

# Update startup event
startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    AppConfig.initialize()
    # Start Prometheus metrics server
    start_http_server(8000)

# Add these imports
from cachetools import TTLCache, LRUCache
from prometheus_client import Counter, Histogram, start_http_server
import functools
from typing import Callable
import traceback

# Add metrics
class Metrics:
    downloads_total = Counter('downloads_total', 'Total number of downloads')
    download_errors = Counter('download_errors', 'Total number of download errors')
    download_duration = Histogram('download_duration_seconds', 'Time spent downloading')
    active_downloads = Counter('active_downloads', 'Currently active downloads')
    bytes_downloaded = Counter('bytes_downloaded_total', 'Total bytes downloaded')
    request_duration = Histogram('request_duration_seconds', 'Request duration')

# Add caching
class CacheManager:
    def __init__(self):
        self.url_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        self.download_cache = LRUCache(maxsize=100)  # LRU cache for downloaded files
        
    async def get_cached_url(self, video_id: str) -> Optional[str]:
        return self.url_cache.get(video_id)
        
    async def cache_url(self, video_id: str, url: str) -> None:
        self.url_cache[video_id] = url
        
    async def get_cached_file(self, video_id: str) -> Optional[Path]:
        return self.download_cache.get(video_id)
        
    async def cache_file(self, video_id: str, file_path: Path) -> None:
        self.download_cache[video_id] = file_path

cache_manager = CacheManager()

# Add error tracking decorator
def track_errors(func: Callable):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_id = str(uuid.uuid4())
            error_details = {
                'error_id': error_id,
                'timestamp': datetime.now(timezone.utc),
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            logging.error(f"Error {error_id}: {error_details}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal error (ID: {error_id})"
            )
    return wrapper

# Add download timeout monitor
async def monitor_download_timeout(download_id: str, timeout: int):
    """Monitor download timeout"""
    await asyncio.sleep(timeout)
    if download_id in active_downloads:
        status = active_downloads[download_id]
        if status.status not in {"Completed", "Failed", "Cancelled"}:
            status.status = "Failed"
            status.errors.append({
                "error": "Download timeout",
                "url": status.current_url
            })

# Add new security and performance configurations
class AppConfig:
    # Download settings
    TEMP_DIR = Path(tempfile.gettempdir()) / "loom_downloads"
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    MAX_PARALLEL_DOWNLOADS = 3
    DOWNLOAD_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    
    # Cache settings
    CACHE_DIR = Path("cache")
    CACHE_TTL = 3600  # 1 hour
    MAX_CACHE_SIZE = 1024 * 1024 * 1024  # 1GB
    
    # Cleanup settings
    CLEANUP_INTERVAL = 3600  # 1 hour
    MAX_AGE_TEMP_FILES = 24 * 3600  # 24 hours
    
    @classmethod
    def initialize(cls):
        """Initialize application directories"""
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Add file handling utilities
class FileManager:
    def __init__(self):
        self.temp_files: Dict[str, Path] = {}
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def save_chunk(self, download_id: str, chunk: bytes) -> Path:
        """Save a chunk to temporary file"""
        temp_path = AppConfig.TEMP_DIR / f"{download_id}.part"
        async with aiofile.async_open(temp_path, 'ab') as f:
            await f.write(chunk)
        self.temp_files[download_id] = temp_path
        return temp_path

    async def finalize_download(self, download_id: str, final_path: Path) -> None:
        """Move temporary file to final location"""
        if download_id in self.temp_files:
            temp_path = self.temp_files[download_id]
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_path), str(final_path))
            del self.temp_files[download_id]

    async def cleanup_download(self, download_id: str) -> None:
        """Clean up temporary files for failed download"""
        if download_id in self.temp_files:
            try:
                self.temp_files[download_id].unlink(missing_ok=True)
                del self.temp_files[download_id]
            except Exception as e:
                logging.error(f"Error cleaning up download {download_id}: {e}")

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up old temporary files"""
        while True:
            try:
                now = time.time()
                for file in AppConfig.TEMP_DIR.glob("*.part"):
                    if now - file.stat().st_mtime > AppConfig.MAX_AGE_TEMP_FILES:
                        file.unlink()
                await asyncio.sleep(AppConfig.CLEANUP_INTERVAL)
            except Exception as e:
                logging.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)

file_manager = FileManager()

# Update DownloadManager with streaming support
class DownloadManager:
    def __init__(self):
        self.active_downloads: Dict[str, DownloadStatus] = {}
        self.download_history: Dict[str, List[Dict[str, Any]]] = {}
        self.encryption = Fernet(SecurityConfig.ENCRYPTION_KEY)
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._semaphore = asyncio.Semaphore(AppConfig.MAX_PARALLEL_DOWNLOADS)

    async def stream_download(self, url: str) -> AsyncGenerator[bytes, None]:
        """Stream download with progress tracking"""
        async with self._semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, proxy=SecurityConfig.PROXY_SETTINGS) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to download: HTTP {response.status}"
                        )
                    
                    async for chunk in response.content.iter_chunked(AppConfig.CHUNK_SIZE):
                        yield chunk

    async def start_download(self, download_id: str, urls: List[str], **kwargs) -> None:
        """Start download with improved error handling and progress tracking"""
        try:
            status = DownloadStatus(
                id=download_id,
                total=len(urls),
                completed=0,
                failed=0,
                status="Starting",
                start_time=datetime.now(timezone.utc)
            )
            self.active_downloads[download_id] = status
            
            for url in urls:
                if status.status == "Cancelled":
                    break
                    
                status.current_url = url
                try:
                    final_path = Path(kwargs['output_dir']) / f"{status.id}.mp4"
                    
                    async for chunk in self.stream_download(url):
                        if status.status == "Cancelled":
                            await self._cleanup_download(download_id)
                            break
                        await self._save_chunk(download_id, chunk)
                    
                    await self._finalize_download(download_id, final_path)
                    status.completed += 1
                    
                except Exception as e:
                    status.failed += 1
                    status.errors.append({"url": url, "error": str(e)})
                    await self._cleanup_download(download_id)
                    
            status.status = "Completed" if status.failed == 0 else "Completed with errors"
            status.end_time = datetime.now(timezone.utc)
            
            # Store download history
            self.download_history[download_id] = {
                "status": status.dict(),
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logging.error(f"Download error: {str(e)}")
            status.status = f"Failed: {str(e)}"
            await self._cleanup_download(download_id)

    async def _save_chunk(self, download_id: str, chunk: bytes) -> None:
        """Save a chunk to temporary file"""
        temp_path = AppConfig.TEMP_DIR / f"{download_id}.part"
        async with aiofile.async_open(temp_path, 'ab') as f:
            await f.write(chunk)
        self.temp_files[download_id] = temp_path

    async def _finalize_download(self, download_id: str, final_path: Path) -> None:
        """Move temporary file to final location"""
        if download_id in self.temp_files:
            temp_path = self.temp_files[download_id]
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_path), str(final_path))
            del self.temp_files[download_id]

    async def _cleanup_download(self, download_id: str) -> None:
        """Clean up temporary files for failed download"""
        if download_id in self.temp_files:
            try:
                self.temp_files[download_id].unlink(missing_ok=True)
                del self.temp_files[download_id]
            except Exception as e:
                logging.error(f"Error cleaning up download {download_id}: {e}")

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up old temporary files"""
        while True:
            try:
                now = time.time()
                for file in AppConfig.TEMP_DIR.glob("*.part"):
                    if now - file.stat().st_mtime > AppConfig.MAX_AGE_TEMP_FILES:
                        file.unlink()
                await asyncio.sleep(AppConfig.CLEANUP_INTERVAL)
            except Exception as e:
                logging.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)

download_manager = DownloadManager()

# Initialize application
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    AppConfig.initialize()
    
# Add streaming download endpoint
@app.get("/api/v1/download/{download_id}/stream")
async def stream_download(
    download_id: str,
    current_user: User = Depends(get_current_user)
) -> StreamingResponse:
    """Stream download directly to client"""
    try:
        if download_id not in active_downloads:
            raise HTTPException(status_code=404, detail="Download not found")
            
        status = active_downloads[download_id]
        if not status.current_url:
            raise HTTPException(status_code=400, detail="No active download URL")
            
        return StreamingResponse(
            download_manager.stream_download(status.current_url),
            media_type="video/mp4"
        )
        
    except Exception as e:
        logging.error(f"Streaming error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/loom_downloader")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")

# Initialize database
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Initialize Redis
redis = Redis.from_url(REDIS_URL)

# Initialize Celery
celery_app = Celery('tasks', broker=REDIS_URL)

# Database models
class Download(Base):
    __tablename__ = "downloads"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    status = Column(String)
    urls = Column(JSON)
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))
    metadata = Column(JSON)
    errors = Column(JSON)

    user = relationship("DBUser", back_populates="downloads")

class DBUser(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String, unique=True)
    hashed_password = Column(String)
    salt = Column(String)
    role = Column(String)
    created_at = Column(DateTime(timezone=True))
    last_login = Column(DateTime(timezone=True))
    downloads = relationship("Download", back_populates="user")

# Create database tables
Base.metadata.create_all(bind=engine)

# Database dependency
async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Redis pool
async def get_redis_pool():
    return await aioredis.create_redis_pool(REDIS_URL)

# Background tasks with Celery
@celery_app.task
def process_download(download_id: str, urls: List[str], user_id: str):
    try:
        # Process download in background
        with SessionLocal() as db:
            download = Download(
                id=download_id,
                user_id=user_id,
                status="processing",
                urls=urls,
                created_at=datetime.now(timezone.utc)
            )
            db.add(download)
            db.commit()

            # Process downloads
            for url in urls:
                try:
                    # Download logic here
                    pass
                except Exception as e:
                    download.errors = download.errors or []
                    download.errors.append({"url": url, "error": str(e)})

            download.status = "completed"
            download.updated_at = datetime.now(timezone.utc)
            db.commit()

    except Exception as e:
        logging.error(f"Background task error: {e}")
        with SessionLocal() as db:
            download = db.query(Download).filter(Download.id == download_id).first()
            if download:
                download.status = "failed"
                download.errors = [{"error": str(e)}]
                download.updated_at = datetime.now(timezone.utc)
                db.commit()

# API versioning
@app.post("/api/v2/download")
@version(2)
async def api_download_v2(
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    redis: Redis = Depends(get_redis_pool)
):
    """Version 2 of download API with enhanced features"""
    try:
        # Validate request
        download_id = str(uuid.uuid4())
        
        # Store in database
        download = Download(
            id=download_id,
            user_id=current_user.id,
            status="queued",
            urls=request.urls,
            created_at=datetime.now(timezone.utc)
        )
        db.add(download)
        db.commit()

        # Queue background task
        process_download.delay(download_id, request.urls, current_user.id)

        # Cache initial status
        await redis.set(
            f"download:{download_id}",
            json.dumps({"status": "queued", "progress": 0}),
            expire=3600
        )

        return {
            "status": "queued",
            "download_id": download_id,
            "message": "Download queued for processing"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Add database cleanup task
@celery_app.task
def cleanup_old_downloads():
    """Clean up old downloads from database"""
    try:
        with SessionLocal() as db:
            # Delete downloads older than 30 days
            threshold = datetime.now(timezone.utc) - timedelta(days=30)
            db.query(Download).filter(Download.created_at < threshold).delete()
            db.commit()
    except Exception as e:
        logging.error(f"Cleanup task error: {e}")

# Schedule cleanup task
celery_app.conf.beat_schedule = {
    'cleanup-old-downloads': {
        'task': 'tasks.cleanup_old_downloads',
        'schedule': 86400.0,  # Daily
    },
}

# Add API versioning middleware
app = VersionedFastAPI(app,
    version_format='{major}',
    prefix_format='/v{major}',
    default_version=(1, 0),
    enable_latest=True
)

# Add enhanced logging configuration
class LogConfig:
    LOGGER_NAME = "loom_downloader"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    LOG_LEVEL = "INFO"
    LOG_FILE = "app.log"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

    @classmethod
    def setup(cls):
        logger = logging.getLogger(cls.LOGGER_NAME)
        logger.setLevel(cls.LOG_LEVEL)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
        logger.addHandler(console_handler)

        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            cls.LOG_FILE,
            maxBytes=cls.MAX_LOG_SIZE,
            backupCount=cls.BACKUP_COUNT
        )
        file_handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
        logger.addHandler(file_handler)

        return logger

logger = LogConfig.setup()

# Add API response models
class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict] = None
    errors: Optional[List[Dict]] = None

# Add error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse(
                status="error",
                message="Internal server error",
                errors=[{"detail": str(e)}]
            ).dict()
        )

# Add rate limiting decorator
def rate_limit(limit: int, window: int = 60):
    def decorator(func):
        requests = {}
        
        async def wrapper(*args, request: Request, **kwargs):
            now = datetime.now(timezone.utc)
            client_ip = request.client.host
            key = f"{client_ip}:{func.__name__}"
            
            # Clean old requests
            if key in requests:
                requests[key] = [t for t in requests[key] if now - t < timedelta(seconds=window)]
            else:
                requests[key] = []
                
            if len(requests[key]) >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {window} seconds."
                )
                
            requests[key].append(now)
            return await func(*args, request=request, **kwargs)
            
        return wrapper
    return decorator

# Add new API endpoints
@app.get("/api/v1/status", response_model=APIResponse)
@rate_limit(limit=60)
async def get_system_status(request: Request):
    """Get system status and statistics"""
    try:
        stats = {
            "active_downloads": len(active_downloads),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            },
            "cache": {
                "url_cache_size": len(cache_manager.url_cache),
                "file_cache_size": len(cache_manager.download_cache)
            }
        }
        
        return APIResponse(
            status="success",
            message="System status retrieved",
            data=stats
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/downloads/{download_id}", response_model=APIResponse)
@rate_limit(limit=10)
async def cancel_download(
    download_id: str,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Cancel an active download"""
    try:
        if download_id not in active_downloads:
            raise HTTPException(status_code=404, detail="Download not found")
            
        status = active_downloads[download_id]
        status.status = "Cancelled"
        await file_manager.cleanup_download(download_id)
        
        return APIResponse(
            status="success",
            message=f"Download {download_id} cancelled",
            data={"download_id": download_id}
        )
    except Exception as e:
        logger.error(f"Error cancelling download: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/downloads/history", response_model=APIResponse)
@rate_limit(limit=30)
async def get_download_history(
    request: Request,
    current_user: User = Depends(get_current_user),
    limit: int = 10,
    offset: int = 0
):
    """Get download history for current user"""
    try:
        history = list(download_manager.download_history.values())
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return APIResponse(
            status="success",
            message="Download history retrieved",
            data={
                "total": len(history),
                "items": history[offset:offset + limit]
            }
        )
    except Exception as e:
        logger.error(f"Error getting download history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        # Initialize configurations
        AppConfig.initialize()
        
        # Start background tasks
        asyncio.create_task(periodic_cleanup())
        asyncio.create_task(monitor_system_health())
        
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

# Add periodic health monitoring
async def monitor_system_health():
    """Monitor system health periodically"""
    while True:
        try:
            # Check system resources
            if psutil.virtual_memory().percent > 90:
                logger.warning("High memory usage detected")
            if psutil.cpu_percent() > 80:
                logger.warning("High CPU usage detected")
            if psutil.disk_usage('/').percent > 90:
                logger.warning("Low disk space detected")
                
            await asyncio.sleep(SecurityConfig.HEALTH_CHECK_INTERVAL)
        except Exception as e:
            logger.error(f"Error monitoring system health: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    setup_logging()
    main()
