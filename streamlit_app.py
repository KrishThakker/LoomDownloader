import streamlit as st
import os
from main import fetch_loom_download_url, download_loom_video, extract_id, format_size
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def main():
    st.set_page_config(page_title="Loom Video Downloader", page_icon="🎥")
    setup_logging()

    st.title("Loom Video Downloader 🎥")
    st.write("Enter Loom video URLs to download them")

    # Input for URLs
    urls_text = st.text_area(
        "Enter Loom URLs (one per line)",
        height=100,
        help="Enter URLs in the format https://www.loom.com/share/[ID]"
    )

    # Download settings
    col1, col2 = st.columns(2)
    with col1:
        max_size = st.number_input(
            "Maximum file size (MB)",
            min_value=0.0,
            value=0.0,
            help="Set to 0 for no limit"
        )
    with col2:
        output_dir = st.text_input(
            "Output directory",
            value="downloads",
            help="Directory where videos will be saved"
        )

    if st.button("Download Videos"):
        if not urls_text.strip():
            st.error("Please enter at least one URL")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        progress_bar = st.progress(0)
        status_text = st.empty()

        success_count = 0
        total_urls = len(urls)

        for i, url in enumerate(urls):
            try:
                status_text.text(f"Processing URL {i+1}/{total_urls}: {url}")
                
                # Extract video ID and create filename
                video_id = extract_id(url)
                filename = os.path.join(output_dir, f"{video_id}.mp4")

                # Get download URL
                video_url = fetch_loom_download_url(video_id)

                # Download the video
                max_size_bytes = max_size * 1024 * 1024 if max_size > 0 else None
                download_loom_video(video_url, filename, max_size=max_size_bytes)
                
                success_count += 1
                st.success(f"Successfully downloaded: {url}")

            except Exception as e:
                st.error(f"Failed to download {url}: {str(e)}")

            progress_bar.progress((i + 1) / total_urls)

        status_text.text(f"Download complete! Successfully downloaded {success_count} out of {total_urls} videos.")

        if success_count > 0:
            st.balloons()

if __name__ == "__main__":
    main() 