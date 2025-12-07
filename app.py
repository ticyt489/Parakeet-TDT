import os
import sys
import warnings
import logging
import gradio as gr
import torch
import pandas as pd
import nemo.collections.asr as nemo_asr
from pathlib import Path
import tempfile
import numpy as np
import subprocess
import math

# Suppress specific loggers before imports
for logger_name in ['nemo_logger', 'pytorch_lightning', 'apex', 'NeMo', 'megatron', 'nemo', 'nemo.collections', 'nemo.core', 'nemo.utils']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).disabled = True

# Function to load the parakeet TDT model
def load_model():
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
    return asr_model

# Global model variable to avoid reloading
model = None

def get_audio_duration(file_path):
    """Get the duration of an audio file using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        output = subprocess.check_output(cmd).decode('utf-8').strip()
        return float(output)
    except (subprocess.SubprocessError, ValueError):
        return None

def extract_audio_from_video(video_path, progress=None):
    if progress is None:
        progress = lambda x, desc=None: None
    
    progress(0.1, desc="Extracting audio from video...")
    
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio_path = temp_audio.name
    temp_audio.close()
    
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        progress(0.2, desc="Audio extraction complete")
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return None

def split_audio_file(file_path, chunk_duration=30, overlap=5, progress=None):
    """Split audio into chunks with overlap"""
    temp_dir = tempfile.mkdtemp()
    duration = get_audio_duration(file_path)
    if not duration:
        return None, 0
    
    step = chunk_duration - overlap
    num_chunks = max(1, math.ceil((duration - overlap) / step))
    chunk_files = []
    
    for i in range(num_chunks):
        if progress is not None:
            progress(i / num_chunks * 0.2, desc=f"Splitting audio ({i+1}/{num_chunks})...")
        
        start_time = i * step
        output_file = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
        
        chunk_len = chunk_duration
        if start_time + chunk_len > duration:
            chunk_len = duration - start_time
        
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(start_time),
            '-t', str(chunk_len),
            '-i', file_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_file
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunk_files.append(output_file)
        except subprocess.CalledProcessError:
            continue
    
    return chunk_files, duration

def merge_segments(all_segments, overlap=5):
    """Merge segments and remove duplicates in overlap zone"""
    if not all_segments:
        return []
    
    merged = []
    for seg in all_segments:
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            # If segments overlap, merge them
            if seg['start'] <= last['end']:
                last['text'] += ' ' + seg['text']
                last['end'] = max(last['end'], seg['end'])
            else:
                merged.append(seg)
    return merged

def transcribe_audio(audio_file, is_music=False, progress=gr.Progress()):
    global model
    
    if model is None:
        progress(0.1, desc="Loading model...")
        model = load_model()
    
    if isinstance(audio_file, tuple):
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        sample_rate, audio_data = audio_file
        
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        import soundfile as sf
        sf.write(temp_audio_path, audio_data, sample_rate)
        audio_path = temp_audio_path
    else:
        import soundfile as sf
        try:
            audio_data, sample_rate = sf.read(audio_file)
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
                
                if is_music:
                    try:
                        audio_data = audio_data / np.max(np.abs(audio_data))
                        from scipy import signal
                        b, a = signal.butter(4, 200/(sample_rate/2), 'highpass')
                        audio_data = signal.filtfilt(b, a, audio_data)
                        threshold = 0.1
                        ratio = 0.5
                        audio_data = np.where(
                            np.abs(audio_data) > threshold,
                            threshold + (np.abs(audio_data) - threshold) * ratio * np.sign(audio_data),
                            audio_data
                        )
                    except ImportError:
                        pass
                
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_audio_path = temp_audio.name
                temp_audio.close()
                sf.write(temp_audio_path, audio_data, sample_rate)
                audio_path = temp_audio_path
            else:
                audio_path = audio_file
        except Exception:
            audio_path = audio_file
    
    duration = get_audio_duration(audio_path)
    long_audio_threshold = 60  # 1 минута (теперь с меньшими чанками)
    chunk_duration = 30  # 30 секунд
    overlap = 5         # 5 секунд перекрытие
    
    if duration and duration > long_audio_threshold:
        return process_long_audio(audio_path, is_music, progress, chunk_duration, overlap)
    
    full_text, segments, csv_path, srt_path = process_audio_chunk(audio_path, is_music, progress, 0, 1.0)
    return full_text, segments, csv_path, srt_path

def process_long_audio(audio_path, is_music, progress, chunk_duration, overlap):
    progress(0.1, desc="Analyzing audio file...")
    chunk_files, total_duration = split_audio_file(audio_path, chunk_duration, overlap, progress)
    
    if not chunk_files:
        return "Error splitting audio file", [], None, None
    
    all_segments = []
    full_text_parts = []
    csv_data = []
    
    step = chunk_duration - overlap
    
    for i, chunk_file in enumerate(chunk_files):
        chunk_start_time = i * step
        progress_start = 0.2 + (i / len(chunk_files)) * 0.8
        progress_end = 0.2 + ((i + 1) / len(chunk_files)) * 0.8
        
        progress(progress_start, desc=f"Processing chunk {i+1}/{len(chunk_files)}...")
        
        chunk_text, chunk_segments, _, _ = process_audio_chunk(
            chunk_file,
            is_music,
            progress,
            chunk_start_time,
            progress_end - progress_start
        )
        
        full_text_parts.append(chunk_text)
        all_segments.extend(chunk_segments)
        
        for segment in chunk_segments:
            csv_data.append({
                "Start (s)": f"{segment['start']:.2f}",
                "End (s)": f"{segment['end']:.2f}",
                "Segment": segment['text']
            })
        
        try:
            os.unlink(chunk_file)
        except:
            pass
    
    try:
        os.rmdir(os.path.dirname(chunk_files[0]))
    except:
        pass
    
    # Объединяем сегменты с учётом перекрытия
    all_segments = merge_segments(all_segments, overlap=overlap)
    
    full_text = " ".join(full_text_parts)
    
    df = pd.DataFrame(csv_data)
    csv_path = "transcript.csv"
    df.to_csv(csv_path, index=False)
    
    srt_path = create_srt_file(all_segments)
    
    progress(1.0, desc="Done!")
    
    return full_text, all_segments, csv_path, srt_path

def process_audio_chunk(audio_path, is_music, progress, time_offset=0, progress_scale=1.0):
    progress(0.3 * progress_scale, desc="Transcribing audio...")
    output = model.transcribe([audio_path], timestamps=True)
    
    segments = []
    csv_data = []
    
    if hasattr(output[0], 'timestamp') and 'segment' in output[0].timestamp:
        for stamp in output[0].timestamp['segment']:
            segment_text = stamp['segment']
            start_time = stamp['start'] + time_offset
            end_time = stamp['end'] + time_offset
            
            if is_music:
                end_time += 0.3
                min_duration = 0.5
                if end_time - start_time < min_duration:
                    end_time = start_time + min_duration
            
            segments.append({
                "text": segment_text,
                "start": start_time,
                "end": end_time
            })
            
            csv_data.append({
                "Start (s)": f"{start_time:.2f}",
                "End (s)": f"{end_time:.2f}",
                "Segment": segment_text
            })
    
    df = pd.DataFrame(csv_data)
    csv_path = "transcript.csv"
    srt_path = None
    if time_offset == 0:
        df.to_csv(csv_path, index=False)
        srt_path = create_srt_file(segments)
    
    full_text = output[0].text if hasattr(output[0], 'text') else ""
    
    if isinstance(audio_path, str) and os.path.exists(audio_path) and audio_path.startswith(tempfile.gettempdir()):
        try:
            os.unlink(audio_path)
        except:
            pass
    
    return full_text, segments, csv_path if time_offset == 0 else None, srt_path

def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def create_srt_file(segments):
    if not segments:
        return None
    
    srt_content = []
    for i, segment in enumerate(segments, 1):
        start_time = format_srt_time(segment['start'])
        end_time = format_srt_time(segment['end'])
        text = segment['text']
        
        srt_content.append(f"{i}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")
    
    srt_path = "transcript.srt"
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_content))
    
    return srt_path

def create_transcript_table(segments):
    if not segments:
        return "No segments found"
    
    html = """
    <style>
    .transcript-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .transcript-table th, .transcript-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .transcript-table th {
        background-color: #f2f2f2;
    }
    .transcript-table tr:hover {
        background-color: #f5f5f5;
        cursor: pointer;
    }
    </style>
    <table class="transcript-table">
        <tr>
            <th>Start (s)</th>
            <th>End (s)</th>
            <th>Segment</th>
        </tr>
    """
    
    for segment in segments:
        html += f"""
        <tr onclick="document.dispatchEvent(new CustomEvent('play_segment', {{detail: {{start: {segment['start']}, end: {segment['end']}}}}}))">
            <td>{segment['start']:.2f}</td>
            <td>{segment['end']:.2f}</td>
            <td>{segment['text']}</td>
        </tr>
        """
    
    html += "</table>"
    return html

js_code = """
function(audio) {
    document.addEventListener('play_segment', function(e) {
        const audioEl = document.querySelector('audio');
        if (audioEl) {
            audioEl.currentTime = e.detail.start;
            audioEl.play();
            
            const stopAtEnd = function() {
                if (audioEl.currentTime >= e.detail.end) {
                    audioEl.pause();
                    audioEl.removeEventListener('timeupdate', stopAtEnd);
                }
            };
            audioEl.addEventListener('timeupdate', stopAtEnd);
        }
    });
    return audio;
}
"""

def transcribe_video(video_file, is_music=False, progress=gr.Progress()):
    audio_path = extract_audio_from_video(video_file, progress)
    if not audio_path:
        return "Error extracting audio from video", [], None, None
    
    return transcribe_audio(audio_path, is_music, progress)

with gr.Blocks(css="footer {visibility: hidden}") as app:
    gr.Markdown("# Audio & Video Transcription with Timestamps")
    gr.Markdown("Upload an audio/video file or record audio to get a transcript with timestamps")
    
    with gr.Row():
        with gr.Column():
            with gr.Tab("Upload Audio File"):
                audio_input = gr.File(label="Upload Audio File", file_types=["audio"])

            
            with gr.Tab("Upload Video File"):
                video_input = gr.Video(label="Upload Video File")
            
            with gr.Tab("Microphone"):
                audio_record = gr.Audio(
                    sources=["microphone"], 
                    type="filepath", 
                    label="Record Audio",
                    show_label=True
                )
            
            is_music = gr.Checkbox(label="Music mode (better for songs)", info="Enable for more accurate song timestamps")
            audio_btn = gr.Button("Transcribe Audio", variant="primary")
            video_btn = gr.Button("Transcribe Video", variant="primary")
            gr.Markdown("""
            ### Notes:
            - Audio or video files over 1 minute will be automatically split into smaller chunks (30s) with 5s overlap
            - Video files will have their audio tracks extracted for transcription
            - Splitting may take a few moments before transcription begins
            """)
        
        with gr.Column():
            full_transcript = gr.Textbox(label="Full Transcript", lines=5)
            transcript_segments = gr.JSON(label="Segments Data", visible=False)
            transcript_html = gr.HTML(label="Transcript Segments (Click a row to play)")
            csv_output = gr.File(label="Download Transcript CSV")
            srt_output = gr.File(label="Download Transcript SRT")
    
    audio_btn.click(
        transcribe_audio,
        inputs=[audio_input, is_music],
        outputs=[full_transcript, transcript_segments, csv_output, srt_output]
    )
    
    video_btn.click(
        transcribe_video,
        inputs=[video_input, is_music],
        outputs=[full_transcript, transcript_segments, csv_output, srt_output]
    )
    
    audio_record.stop_recording(
        transcribe_audio,
        inputs=[audio_record, is_music],
        outputs=[full_transcript, transcript_segments, csv_output, srt_output]
    )
    
    transcript_segments.change(
        create_transcript_table,
        inputs=[transcript_segments],
        outputs=[transcript_html]
    )
    
if __name__ == "__main__":
    app.launch()


