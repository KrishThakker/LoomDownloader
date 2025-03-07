#!/usr/bin/python3

import argparse
import json
import urllib.request
import os


def fetch_loom_download_url(id):
    try:
        request = urllib.request.Request(
            url=f"https://www.loom.com/api/campaigns/sessions/{id}/transcoded-url",
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            method="POST",
        )
        response = urllib.request.urlopen(request)
        body = response.read()
        content = json.loads(body.decode("utf-8"))
        if "url" not in content:
            raise ValueError("Not a Loom video")
        return content["url"]
    except json.JSONDecodeError:
        raise ValueError("Response from Loom API is not valid JSON")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError("Video not found. Please check the video ID")
        raise


def download_loom_video(url, filename):
    try:
        response = urllib.request.urlopen(url)
        file_size = int(response.headers['Content-Length'])
        downloaded = 0
        
        with open(filename, 'wb') as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                downloaded += len(chunk)
                f.write(chunk)
                
                # Calculate progress
                progress = int(50 * downloaded / file_size)
                bars = '=' * progress + '-' * (50 - progress)
                percent = downloaded / file_size * 100
                print(f'\rDownloading: [{bars}] {percent:.1f}%', end='', flush=True)
        print('\nDownload complete!')
    except (urllib.error.URLError, IOError) as e:
        if os.path.exists(filename):
            os.remove(filename)  # Clean up partial download
        raise RuntimeError(f"Download failed: {str(e)}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="loom-dl", description="script to download loom.com videos"
    )
    parser.add_argument(
        "url", help="Url of the video in the format https://www.loom.com/share/[ID]"
    )
    parser.add_argument("-o", "--out", help="Path to output the file to")
    arguments = parser.parse_args()
    return arguments


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
