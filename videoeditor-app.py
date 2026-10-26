import streamlit as st
import os
import subprocess
from pydub import AudioSegment

st.set_page_config(page_title="AI Motovlog Editor Pro", page_icon="🏍️", layout="centered")

st.title("🏍️ AI Motovlog Editor (With Music & FX)")
st.write("Upload your raw footage to cut dull moments, enhance the visuals, and automatically mix background music.")

# Setup temporary directories
UPLOAD_DIR = "temp_uploads"
MUSIC_DIR = "music"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MUSIC_DIR, exist_ok=True)

# Helper to look for available music tracks
available_tracks = [f for f in os.listdir(MUSIC_DIR) if f.endswith(('.mp3', '.wav'))]
if not available_tracks:
    # Create a dummy note if no music exists yet
    st.info(f"💡 Tip: Create a folder named `{MUSIC_DIR}` and drop your MP3s there to choose your own tracks!")

# 1. Video Upload Component
uploaded_file = st.file_uploader("Upload your raw GoPro / Camera video (.mp4)", type=["mp4", "mov", "mkv"])

if uploaded_file is not None:
    input_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.video(input_path)
    st.success("Video loaded!")

    # Sidebar configurations
    st.sidebar.header("✂️ Cut Settings")
    silence_threshold = st.sidebar.slider("Dullness Threshold (dB)", -50, -10, -32)
    min_silence_len = st.sidebar.slider("Min Dull Duration (ms)", 500, 5000, 1500, 100)
    padding = st.sidebar.slider("Clip Buffer Padding (ms)", 100, 1000, 300, 100)

    st.sidebar.header("🎵 Music Settings")
    music_option = st.sidebar.selectbox("Choose Background Track", ["None"] + available_tracks)
    music_volume = st.sidebar.slider("Music Volume Level", -40, -10, -25, help="Lower means quieter background music so exhaust/voice stands out.")

    st.sidebar.header("🎬 Visual FX Filters")
    apply_saturation = st.sidebar.checkbox("Boost Color Saturation (Vibrant Ride)", value=True)
    apply_vignette = st.sidebar.checkbox("Cinematic Vignette (Dark Corners)", value=False)
    apply_contrast = st.sidebar.checkbox("High Contrast (Deep Shadows)", value=False)

    if st.button("🚀 Generate Best Edit", type="primary"):
        with st.spinner("Processing video layers, mixing audio, and rendering FX..."):
            
            # Paths for processing stages
            audio_path = os.path.join(UPLOAD_DIR, "extracted_audio.wav")
            rough_cut_path = os.path.join(UPLOAD_DIR, "rough_cut.mp4")
            list_path = os.path.join(UPLOAD_DIR, "ffmpeg_list.txt")
            final_output_path = os.path.join(UPLOAD_DIR, f"FX_EDIT_{uploaded_file.name}")
            
            # Cleanup previous assets
            for path in [audio_path, rough_cut_path, list_path, final_output_path]:
                if os.path.exists(path): os.remove(path)

            # STEP 1: Extract audio for analysis
            extract_cmd = f'ffmpeg -y -i "{input_path}" -vn -acodec pcm_s16le -ar 44100 -ac 1 "{audio_path}"'
            subprocess.run(extract_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # STEP 2: Analyze and Cut Dull Moments
            from pydub.silence import detect_nonsilent
            sound = AudioSegment.from_wav(audio_path)
            action_intervals = detect_nonsilent(sound, min_silence_len=min_silence_len, silence_thresh=silence_threshold)
            
            if not action_intervals:
                st.warning("No action moments found. Try raising the Dullness Threshold slider.")
                st.stop()

            # Build sub-clips and stitch rough cut
            with open(list_path, "w") as f:
                clip_count = 0
                for start, end in action_intervals:
                    start_sec = max(0, start - padding) / 1000.0
                    end_sec = min(len(sound), end + padding) / 1000.0
                    duration = end_sec - start_sec
                    
                    if duration < 0.5: continue
                        
                    clip_name = os.path.join(UPLOAD_DIR, f"subclip_{clip_count}.mp4")
                    if os.path.exists(clip_name): os.remove(clip_name)
                        
                    # Quick cut streams
                    cut_cmd = f'ffmpeg -y -ss {start_sec} -i "{input_path}" -t {duration} -c copy "{clip_name}"'
                    subprocess.run(cut_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(clip_name):
                        f.write(f"file 'subclip_{clip_count}.mp4'\n")
                        clip_count += 1

            if clip_count > 0:
                concat_cmd = f'ffmpeg -y -f concat -safe 0 -i "{list_path}" -c copy "{rough_cut_path}"'
                subprocess.run(concat_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for i in range(clip_count):
                    try: os.remove(os.path.join(UPLOAD_DIR, f"subclip_{i}.mp4"))
                    except: pass
            else:
                st.error("Error creating cuts.")
                st.stop()

            # STEP 3: Build FFmpeg Filters (Music & Visual Effects)
            # Build Video Filters string
            video_filters = []
            if apply_saturation:
                video_filters.append("eq=saturation=1.4")  # Makes colors pop
            if apply_contrast:
                video_filters.append("eq=contrast=1.2:brightness=-0.02")  # Moody contrast
            if apply_vignette:
                video_filters.append("vignette=PI/4")  # Cinematic framing border

            filter_vf_str = f'-vf "{",".join(video_filters)}"' if video_filters else ""

            # Build Audio Mixing command if music selected
            if music_option != "None":
                selected_music_path = os.path.join(MUSIC_DIR, music_option)
                
                # We use complex filtergraph to loop music, adjust its volume, and overlay it onto video audio
                audio_filter_str = (
                    f'-i "{selected_music_path}" -filter_complex '
                    f'"[1:a]streamloop=-1,volume={music_volume}dB[bgm];'
                    f'[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[a]" '
                    f'-map 0:v -map "[a]"'
                )
            else:
                audio_filter_str = "-c:a aac"

            # STEP 4: Render Final Masterpiece (Cuts + FX + Music)
            st.text("Rendering effects and master tracking mix... (This takes a moment)")
            
            # We re-encode here to bake in the music mixing and visual filters smoothly
            final_cmd = f'ffmpeg -y -i "{rough_cut_path}" {audio_filter_str} {filter_vf_str} -c:v libx264 -pix_fmt yuv420p -preset fast "{final_output_path}"'
            subprocess.run(final_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 5. Output presentation
            if os.path.exists(final_output_path):
                st.success("🔥 Master Edit Complete with Music & Visual FX!")
                st.video(final_output_path)
                
                with open(final_output_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Mastered Vlog",
                        data=file,
                        file_name=f"AI_MASTER_{uploaded_file.name}",
                        mime="video/mp4"
                    )
            else:
                st.error("Rendering failed. Please double check if your optional audio file paths are correct.")
