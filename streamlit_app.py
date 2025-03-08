import streamlit as st
import os
from main import fetch_loom_download_url, download_loom_video, extract_id, format_size
import logging
import time
from datetime import datetime
import json
import requests

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
    return {'theme': 'light', 'output_dir': 'téléchargements', 'max_size': 0}

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
        page_title="Téléchargeur de Vidéos Loom",
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
        st.title("⚙️ Paramètres")
        
        # Theme selector
        theme = st.selectbox(
            "Thème",
            options=['light', 'dark'],
            index=0 if st.session_state.theme == 'light' else 1
        )
        
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            settings['theme'] = theme
            save_settings(settings)
            st.experimental_rerun()
        
        st.divider()
        
        # Help section
        st.markdown("""
            ### 📖 Comment utiliser
            1. Collez les URLs des vidéos Loom
            2. Configurez les paramètres
            3. Cliquez sur Télécharger
            
            ### 🔗 URLs prises en charge
            - https://www.loom.com/share/[ID]
            
            ### ❓ Besoin d'aide ?
            [Documentation](https://github.com/yourusername/loom-downloader)
        """)

    setup_logging()

    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Header section with gradient background
        st.markdown("""
            <div class='header-box'>
                <h1 style='margin:0;'>🎥 Téléchargeur de Vidéos Loom</h1>
                <p style='margin:0.5rem 0 0 0;'>Téléchargez vos vidéos Loom rapidement et facilement</p>
            </div>
        """, unsafe_allow_html=True)

        # Main input section
        with st.container():
            st.subheader("📝 URLs des Vidéos")
            urls_text = st.text_area(
                "Entrez les URLs Loom (une par ligne)",
                height=150,
                help="Entrez les URLs au format https://www.loom.com/share/[ID]",
                placeholder="https://www.loom.com/share/votre-id-de-video\nhttps://www.loom.com/share/un-autre-id-de-video"
            )

            # Advanced settings in an expander
            with st.expander("⚙️ Paramètres Avancés"):
                settings_col1, settings_col2 = st.columns(2)
                
                with settings_col1:
                    max_size = st.number_input(
                        "Taille maximale (Mo)",
                        min_value=0.0,
                        value=settings.get('max_size', 0.0),
                        help="Mettre à 0 pour aucune limite",
                        format="%.1f"
                    )
                    
                with settings_col2:
                    output_dir = st.text_input(
                        "Répertoire de sortie",
                        value=settings.get('output_dir', 'téléchargements'),
                        help="Répertoire où les vidéos seront enregistrées"
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
        if download_placeholder.button("🚀 Démarrer le Téléchargement", type="primary"):
            if not urls_text.strip():
                st.error("⚠️ Veuillez entrer au moins une URL")
                st.stop()

            # Show progress section
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.subheader("📥 Progrès du Téléchargement")
            progress_bar = st.progress(0)
            status_text = st.empty()
            download_stats = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            
            try:
                response = requests.post(
                    'http://localhost:8000/api/download',
                    json={
                        'urls': urls,
                        'max_size': float(max_size),
                        'output_dir': output_dir
                    }
                )

                if response.status_code == 200:
                    download_id = response.json()['download_id']
                    
                    # Start monitoring in a separate thread
                    while True:
                        status_response = requests.get(f'http://localhost:8000/api/status/{download_id}')
                        if status_response.status_code == 200:
                            status = status_response.json()
                            progress = ((status['completed'] + status['failed']) / status['total']) * 100
                            progress_bar.progress(progress)
                            status_text.text(f"Statut: {status['status']} | URL Actuelle: {status['current_url'] or 'Aucune'}")
                            
                            if status['status'] == 'Completed':
                                st.balloons()
                                st.success(f"✅ Téléchargement terminé ! {status['completed']} vidéos téléchargées avec succès.")
                                break
                            elif status['status'].startswith('Failed'):
                                st.error(f"❌ Échec du téléchargement pour {status['failed']} vidéo(s). Erreurs: {', '.join([e['error'] for e in status['errors']])}")
                                break
                        else:
                            st.error("Erreur lors de l'obtention de l'état du téléchargement.")
                            break
                else:
                    st.error("Erreur lors du démarrage du téléchargement.")
            except Exception as e:
                st.error(f"Erreur : {str(e)}")

    # Footer with additional information
    st.markdown(f"""
        <div style='margin-top: 3rem; text-align: center; color: #666;'>
            <hr>
            <p>Fait avec ❤️ en utilisant Streamlit</p>
            <p style='font-size: 0.8rem;'>Dernière mise à jour : {datetime.now().strftime("%d/%m/%Y")}</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 