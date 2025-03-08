from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from main import fetch_loom_download_url, download_loom_video, extract_id, format_size
import logging
from threading import Thread
from queue import Queue
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Global download queue and status dictionary
download_queue = Queue()
download_status = {}

def process_download_queue():
    while True:
        if not download_queue.empty():
            download_id, url, output_dir, max_size = download_queue.get()
            try:
                video_id = extract_id(url)
                filename = os.path.join(output_dir, f"{video_id}.mp4")
                
                # Update status
                download_status[download_id].update({
                    "status": "fetching",
                    "message": "Fetching download URL..."
                })
                
                video_url = fetch_loom_download_url(video_id)
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Update status
                download_status[download_id].update({
                    "status": "downloading",
                    "message": "Downloading video..."
                })
                
                # Download the video
                max_size_bytes = max_size * 1024 * 1024 if max_size > 0 else None
                download_loom_video(video_url, filename, max_size=max_size_bytes, use_tqdm=False)
                
                download_status[download_id].update({
                    "status": "completed",
                    "message": "Download completed successfully!",
                    "filename": filename
                })
                
            except Exception as e:
                download_status[download_id].update({
                    "status": "error",
                    "message": str(e)
                })
            
            download_queue.task_done()
        time.sleep(0.1)

# Start download queue processor
download_thread = Thread(target=process_download_queue, daemon=True)
download_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download', methods=['POST'])
def download():
    urls = request.form.get('urls', '').strip().split('\n')
    urls = [url.strip() for url in urls if url.strip()]
    
    if not urls:
        return jsonify({"error": "Please enter at least one URL"}), 400
    
    output_dir = request.form.get('output_dir', 'downloads').strip()
    max_size = float(request.form.get('max_size', 0))
    
    download_ids = []
    for url in urls:
        download_id = str(time.time()) + "_" + url[-8:]
        download_status[download_id] = {
            "url": url,
            "status": "queued",
            "message": "Waiting in queue..."
        }
        download_queue.put((download_id, url, output_dir, max_size))
        download_ids.append(download_id)
    
    return jsonify({"download_ids": download_ids})

@app.route('/status/<download_id>')
def status(download_id):
    if download_id in download_status:
        return jsonify(download_status[download_id])
    return jsonify({"error": "Download ID not found"}), 404

@app.route('/downloads/<path:filename>')
def download_file(filename):
    return send_from_directory('downloads', filename)

if __name__ == '__main__':
    app.run(debug=True) 