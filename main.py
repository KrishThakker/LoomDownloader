#!/usr/bin/python3

import argparse
import json
import urllib.request
import os
import time
import os.path
from os import statvfs


def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}GB"


def fetch_loom_download_url(id):
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(
                url=f"https://www.loom.com/api/campaigns/sessions/sessions/{id}/transcoded-url",
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                    'Accept-Language': 'en-US,en;q=0.9',
                },
                method="POST",
            )
            response = urllib.request.urlopen(request)
            body = response.read()
            content = json.loads(body.decode("utf-8"))
            if "url" not in content:
                raise ValueError("Not a Loom video")
            return content["url"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError("Video not found. Please check the video ID")
            elif e.code == 429:  # Too Many Requests
                if attempt < max_retries - 1:
                    print(f"Rate limited. Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
            raise
        except json.JSONDecodeError:
            raise ValueError("Response from Loom API is possibly not valid JSON")


def get_safe_filename(filename):
    if not os.path.exists(filename):
        return filename
    
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"


def download_loom_video(url, filename):
    try:
        request = urllib.request.Request(url, method='HEAD')
        response = urllib.request.urlopen(request)
        file_size = int(response.headers['Content-Length'])
        
        # Check if we have enough disk space
        free_space = os.statvfs(os.path.dirname(os.path.abspath(filename))).f_frsize * \
                     os.statvfs(os.path.dirname(os.path.abspath(filename))).f_bavail
        if free_space < file_size:
            raise IOError(f"Not enough disk space. Need {format_size(file_size)}, have {format_size(free_space)}")
        
        # Handle file existence and resumption
        downloaded = 0
        headers = {}
        if os.path.exists(filename):
            downloaded = os.path.getsize(filename)
            if downloaded < file_size:
                headers['Range'] = f'bytes={downloaded}-'
            elif downloaded == file_size:
                print(f"File {filename} already exists and is complete!")
                return
            else:
                print(f"Existing file size ({downloaded}) is larger than expected ({file_size}). Starting fresh download.")
                downloaded = 0
        
        request = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(request)
        
        mode = 'ab' if downloaded > 0 else 'wb'
        start_time = time.time()
        last_update = start_time
        
        with open(filename, mode) as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                downloaded += len(chunk)
                f.write(chunk)
                
                # Update progress every 100ms
                current_time = time.time()
                if current_time - last_update >= 0.1:
                    elapsed_time = current_time - start_time
                    speed = downloaded / elapsed_time if elapsed_time > 0 else 0
                    progress = int(50 * downloaded / file_size)
                    bars = '=' * progress + '-' * (50 - progress)
                    percent = downloaded / file_size * 100
                    eta = (file_size - downloaded) / speed if speed > 0 else 0
                    print(f'\rDownloading: [{bars}] {percent:.1f}% ({format_size(downloaded)}/{format_size(file_size)}) '
                          f'@ {format_size(speed)}/s ETA: {int(eta)}s', end='', flush=True)
                    last_update = current_time
                    
        print('\nDownload complete!')
    except (urllib.error.URLError, IOError) as e:
        if os.path.exists(filename) and not downloaded:  # Only remove if we didn't partially download
            os.remove(filename)
        raise RuntimeError(f"Download failed: {str(e)}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="loom-dl", description="script to download loom.com videos"
    )
    parser.add_argument(
        "url", help="Url of the video in the format https://www.loom.com/share/[ID]"
    )
    parser.add_argument("-o", "--out", help="Path to output the file to")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def extract_id(url):
    if not url.startswith("https://www.loom.com/share/"):
        raise ValueError("Invalid Loom URL. Must start with 'https://www.loom.com/share/'")
    video_id = url.split("/")[-1]
    if not video_id:
        raise ValueError("Invalid Loom URL. No video ID found")
    return video_id


def main():
    try:
        arguments = parse_arguments()
        id = extract_id(arguments.url)

        url = fetch_loom_download_url(id)
        filename = arguments.out or f"{id}.mp4"
        
        if os.path.exists(filename) and not arguments.out:
            filename = get_safe_filename(filename)
            
        print(f"Downloading video {id} and saving to {filename}")
        download_loom_video(url, filename)
        print("Download completed successfully!")
    except ValueError as e:
        print(f"Error: {str(e)}")
        exit(1)
    except urllib.error.URLError as e:
        print(f"Network error: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
