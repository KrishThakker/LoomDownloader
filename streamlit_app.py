import streamlit as st
import os
from main import fetch_loom_download_url, download_loom_video, extract_id, format_size
import logging
import time
from datetime import datetime
import json

# Constants
THEME_COLORS = {
    'light': {
        'primary': '#FF4B4B',
        'secondary': '#FF8080',
        'background': '#FFFFFF',
        'text': '#1E1E1E'
    },
    'dark': {
        'primary': '#FF6B6B',
        'secondary': '#FFA0A0',
        'background': '#1E1E1E',
        'text': '#FFFFFF'
    }
}

def load_settings():
    try:
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                return json.load(f)
    except:
        pass
    return {'theme': 'light', 'output_dir': 'downloads', 'max_size': 0}

def save_settings(settings):
    with open('settings.json', 'w') as f:
        json.dump(settings, f)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def get_theme_colors():
    return THEME_COLORS['dark'] if st.session_state.get('theme', 'light') == 'dark' else THEME_COLORS['light']

def main():
    # Load saved settings
    settings = load_settings()
    
    # Page configuration
    st.set_page_config(
        page_title="Loom Video Downloader",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'theme' not in st.session_state:
        st.session_state.theme = settings.get('theme', 'light')

    colors = get_theme_colors()

    # Custom CSS with improved styling and theme support
    st.markdown(f"""
        <style>
        .main {{
            padding: 2rem;
            background-color: {colors['background']};
            color: {colors['text']};
        }}
        .stButton>button {{
            width: 100%;
            background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
            color: white;
            font-weight: bold;
            padding: 0.75rem 1rem;
            margin-top: 1rem;
            border-radius: 0.5rem;
            transition: all 0.3s ease;
            border: none;
        }}
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        }}
        .success-message {{
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(40, 167, 69, 0.1);
            border-left: 4px solid #28A745;
            margin: 0.5rem 0;
        }}
        .error-message {{
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(220, 53, 69, 0.1);
            border-left: 4px solid #DC3545;
            margin: 0.5rem 0;
        }}
        .info-box {{
            background-color: {colors['background']};
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 75, 75, 0.2);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .stats-box {{
            background-color: rgba(255, 75, 75, 0.05);
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 75, 75, 0.2);
        }}
        .header-box {{
            background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
            padding: 2rem;
            border-radius: 1rem;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.2);
        }}
        .stProgress > div > div > div {{
            background-color: {colors['primary']};
        }}
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Theme selector
        theme = st.selectbox(
            "Theme",
            options=['light', 'dark'],
            index=0 if st.session_state.theme == 'light' else 1
        )
        
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            settings['theme'] = theme
            save_settings(settings)
            st.rerun()
        
        st.divider()
        
        # Help section
        st.markdown("""
            ### 📖 How to Use
            1. Paste Loom video URLs
            2. Configure settings
            3. Click Download
            
            ### 🔗 Supported URLs
            - https://www.loom.com/share/[ID]
            
            ### ❓ Need Help?
            [Documentation](https://github.com/yourusername/loom-downloader)
        """)

    setup_logging()

    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Header section with gradient background
        st.markdown("""
            <div class='header-box'>
                <h1 style='margin:0;'>🎥 Loom Video Downloader</h1>
                <p style='margin:0.5rem 0 0 0;'>Download your Loom videos quickly and easily</p>
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

            # Advanced settings in an expander
            with st.expander("⚙️ Advanced Settings"):
                settings_col1, settings_col2 = st.columns(2)
                
                with settings_col1:
                    max_size = st.number_input(
                        "Maximum file size (MB)",
                        min_value=0.0,
                        value=settings.get('max_size', 0.0),
                        help="Set to 0 for no limit",
                        format="%.1f"
                    )
                    
                with settings_col2:
                    output_dir = st.text_input(
                        "Output directory",
                        value=settings.get('output_dir', 'downloads'),
                        help="Directory where videos will be saved"
                    )

                # Save settings
                if max_size != settings.get('max_size') or output_dir != settings.get('output_dir'):
                    settings.update({
                        'max_size': max_size,
                        'output_dir': output_dir
                    })
                    save_settings(settings)

        # Download button with loading animation
        download_placeholder = st.empty()
        if download_placeholder.button("🚀 Start Download", type="primary"):
            with st.spinner("Preparing download..."):
                if not urls_text.strip():
                    st.error("⚠️ Please enter at least one URL")
                    st.stop()

                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)

                urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                
                # Create a container for download progress
                with st.container():
                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                    st.subheader("📥 Download Progress")
                    
                    # Download statistics
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        start_time = time.time()
                        time_display = st.empty()
                    with stats_col2:
                        progress_display = st.empty()

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    download_stats = st.empty()
                    st.markdown("</div>", unsafe_allow_html=True)

                    success_count = 0
                    total_urls = len(urls)

                    for i, url in enumerate(urls):
                        try:
                            # Update statistics
                            elapsed_time = time.time() - start_time
                            time_display.markdown(f"⏱️ **Elapsed Time:** {int(elapsed_time)}s")
                            progress_display.markdown(f"📊 **Progress:** {i+1}/{total_urls} URLs")
                            
                            status_text.info(f"⏳ Processing URL {i+1}/{total_urls}")
                            download_stats.markdown(f"🔗 **Current URL:** {url}")
                            
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

                    # Final status with detailed statistics
                    if success_count > 0:
                        st.balloons()
                        st.markdown(f"""
                            <div class='stats-box'>
                                <h3>📊 Download Summary</h3>
                                <p>✅ Successfully downloaded: {success_count} videos</p>
                                <p>❌ Failed downloads: {total_urls - success_count} videos</p>
                                <p>⏱️ Total time: {int(time.time() - start_time)} seconds</p>
                                <p>📁 Save location: {os.path.abspath(output_dir)}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("😔 No videos were downloaded successfully.")

    # Footer with additional information
    st.markdown("""
        <div style='margin-top: 3rem; text-align: center; color: #666;'>
            <hr>
            <p>Made with ❤️ using Streamlit</p>
            <p style='font-size: 0.8rem;'>Last updated: {}</p>
        </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

if __name__ == "__main__":
    main() 