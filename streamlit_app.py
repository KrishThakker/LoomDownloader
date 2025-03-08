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
    # Page configuration
    st.set_page_config(
        page_title="Loom Video Downloader",
        page_icon="🎥",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            margin-top: 1rem;
        }
        .success-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #D4EDDA;
            color: #155724;
            margin: 0.5rem 0;
        }
        .error-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #F8D7DA;
            color: #721C24;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    setup_logging()

    # Header section
    st.title("🎥 Loom Video Downloader")
    st.markdown("""
        <div style='background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
            Download your Loom videos easily! Just paste the video URLs below (one per line).
        </div>
    """, unsafe_allow_html=True)

    # Main input section
    with st.container():
        st.subheader("📝 Video URLs")
        urls_text = st.text_area(
            "Enter Loom URLs (one per line)",
            height=150,
            help="Enter URLs in the format https://www.loom.com/share/[ID]",
            placeholder="https://www.loom.com/share/your-video-id\nhttps://www.loom.com/share/another-video-id"
        )

        # Settings section
        st.subheader("⚙️ Download Settings")
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            max_size = st.number_input(
                "Maximum file size (MB)",
                min_value=0.0,
                value=0.0,
                help="Set to 0 for no limit",
                format="%.1f"
            )
            
        with settings_col2:
            output_dir = st.text_input(
                "Output directory",
                value="downloads",
                help="Directory where videos will be saved"
            )

    # Download section
    if st.button("🚀 Start Download"):
        if not urls_text.strip():
            st.error("⚠️ Please enter at least one URL")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        # Create a container for download progress
        with st.container():
            st.subheader("📥 Download Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            download_stats = st.empty()

            success_count = 0
            total_urls = len(urls)

            for i, url in enumerate(urls):
                try:
                    status_text.info(f"⏳ Processing URL {i+1}/{total_urls}")
                    download_stats.text(f"URL: {url}")
                    
                    # Extract video ID and create filename
                    video_id = extract_id(url)
                    filename = os.path.join(output_dir, f"{video_id}.mp4")

                    # Get download URL
                    video_url = fetch_loom_download_url(video_id)

                    # Download the video
                    max_size_bytes = max_size * 1024 * 1024 if max_size > 0 else None
                    download_loom_video(video_url, filename, max_size=max_size_bytes, use_tqdm=False)
                    
                    success_count += 1
                    st.markdown(f"""
                        <div class='success-message'>
                            ✅ Successfully downloaded: {url}
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f"""
                        <div class='error-message'>
                            ❌ Failed to download {url}: {str(e)}
                        </div>
                    """, unsafe_allow_html=True)

                progress_bar.progress((i + 1) / total_urls)

            # Final status
            if success_count > 0:
                st.balloons()
                status_text.success(f"✨ Download complete! Successfully downloaded {success_count} out of {total_urls} videos.")
            else:
                status_text.error("😔 No videos were downloaded successfully.")

            # Show download location
            if success_count > 0:
                st.info(f"📁 Videos were saved to: {os.path.abspath(output_dir)}")

    # Footer
    st.markdown("""
        <div style='margin-top: 3rem; text-align: center; color: #666;'>
            <hr>
            <p>Made with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 