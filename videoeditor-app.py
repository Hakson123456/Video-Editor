import streamlit as st
import os
import subprocess
import numpy as np
from scipy.io import wavfile

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
    volume_percentage = st.sidebar.slider(
        "Action Audio Sensitivity (%)", 
        min_value=1, max_value=50, value=15, 
        help="Lower percentage means a quieter engine/voice will count as 'action'. Higher means it needs to be louder."
    )
    min_silence_len_sec = st.sidebar.slider("Min Dull Duration (seconds)", 0.5, 5.0, 1.5, 0.1)
    padding_sec = st.sidebar.slider("Clip Buffer Padding (seconds)", 0.1, 1.0, 0.3, 0.1)

    st.sidebar.header("🎵 Music Settings")
    music_option = st.sidebar.selectbox("Choose Background Track", ["None"] + available_tracks)
    music_volume = st.sidebar.slider("Music Volume Level", -40, -10, -25)

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

            if not os.path.exists(audio_path):
                st.error("Audio extraction failed. Verify that your video file has a valid soundtrack track.")
                st.stop()

            # STEP 2: Analyze Audio with SciPy (Modern alternative to pydub)
            sample_rate, data = wavfile.read(audio_path)
            
            # Calculate root-mean-square (RMS) energy over chunks
            chunk_size = int(sample_rate * 0.1) # 100ms windows
            num_chunks = len(data) // chunk_size
            
            rms_values = []
            for i in range(num_chunks):
                chunk = data[i*chunk_size : (i+1)*chunk_size]
                rms = np.sqrt(np.mean(chunk.astype(np.float64)**2)) if len(chunk) > 0 else 0
                rms_values.append(rms)
                
            rms_values = np.array(rms_values)
            max_rms = np.max(rms_values) if len(rms_values) > 0 else 1
            threshold = max_rms * (volume_percentage / 100.0)
            
            # Group contiguous action chunks
            is_action = rms_values > threshold
            action_intervals = []
            
            in_action = False
            start_chunk = 0
            
            for idx, val in enumerate(is_action):
                if val and not in_action:
                    in_action = True
                    start_chunk = idx
                elif not val and in_action:
                    in_action = False
                    start_ms = start_chunk * 100
                    end_ms = idx * 100
                    if (end_ms - start_ms) >= (min_silence_len_sec * 1000):
                        action_intervals.append((start_ms, end_ms))
                        
            if in_action:
                action_intervals.append((start_chunk * 100, len(rms_values) * 100))

            if not action_intervals:
                st.warning("No action moments found. Try lowering the Action Audio Sensitivity percentage slider.")
                st.stop()

            # Build sub-clips and stitch rough cut
            with open(list_path, "w") as f:
                clip_count = 0
                video_duration_ms = (len(data) / sample_rate) * 1000
                
                for start, end in action_intervals:
                    start_sec = max(0, start - (padding_sec * 1000)) / 1000.0
                    end_sec = min(video_duration_ms, end + (padding_sec * 1000)) / 1000.0
                    duration = end_sec - start_sec
                    
                    if duration < 0.5: continue
                        
                    clip_name = os.path.join(UPLOAD_DIR, f"subclip_{clip_count}.mp4")
                    if os.path.exists(clip_name): os.remove(clip_name)
                        
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
            video_filters = []
            if apply_saturation: video_filters.append("eq=saturation=1.4")
            if apply_contrast: video_filters.append("eq=contrast=1.2:brightness=-0.02")
            if apply_vignette: video_filters.append("vignette=PI/4")

            filter_vf_str = f'-vf "{",".join(video_filters)}"' if video_filters else ""

            if music_option != "None":
                selected_music_path = os.path.join(MUSIC_DIR, music_option)
                audio_filter_str = (
                    f'-i "{selected_music_path}" -filter_complex '
                    f'"[1:a]streamloop=-1,volume={music_volume}dB[bgm];'
                    f'[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[a]" '
                    f'-map 0:v -map "[a]"'
                )
            else:
                audio_filter_str = "-c:a aac"

            # STEP 4: Render Final Video
            st.text("Rendering effects and master tracking mix... (This takes a moment)")
            final_cmd = f'ffmpeg -y -i "{rough_cut_path}" {audio_filter_str} {filter_vf_str} -c:v libx264 -pix_fmt yuv420p -preset fast "{final_output_path}"'
            subprocess.run(final_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 5. Output presentation and local backup save
            if os.path.exists(final_output_path):
                st.success("🔥 Master Edit Complete with Music & Visual FX!")
                st.video(final_output_path)
                
                local_exports_dir = "final_exports"
                os.makedirs(local_exports_dir, exist_ok=True)
                auto_save_path = os.path.join(local_exports_dir, f"AI_MASTER_{uploaded_file.name}")
                
                import shutil
                shutil.copy(final_output_path, auto_save_path)
                st.info(f"💾 Local backup saved to repository storage: `{auto_save_path}`")
                
                with open(final_output_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Mastered Vlog via Browser",
                        data=file,
                        file_name=f"AI_MASTER_{uploaded_file.name}",
                        mime="video/mp4"
                    )
            else:
                st.error("Rendering failed. Please double check your rendering settings.")
