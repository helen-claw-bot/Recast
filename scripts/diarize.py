"""
Speaker diarization using whisperx (transcription) + resemblyzer (speaker embedding clustering).
No HuggingFace token required.
"""
import json
import numpy as np
import whisperx
from pathlib import Path
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
import soundfile as sf
import tempfile
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cpu"
VOCALS_DIR = Path("/root/novel/output/demucs/htdemucs")
OUTPUT_DIR = Path("/root/novel/output/speakers")

def extract_segment_wav(vocals_path, start, end):
    """Extract a segment from wav file, return as numpy array."""
    audio = AudioSegment.from_wav(str(vocals_path))
    segment = audio[int(start * 1000):int(end * 1000)]
    # Convert to numpy
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    samples = samples / (2**15)  # normalize int16 to float
    if segment.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples, segment.frame_rate

def process_file(vocals_path: Path, file_id: str, num_speakers: int = None):
    print(f"\n{'='*60}")
    print(f"Processing: {file_id}")
    print(f"{'='*60}")
    
    # Step 1: Transcribe with whisperx
    print("[1/4] Transcribing...")
    model = whisperx.load_model("small", DEVICE, compute_type="int8", language="zh")
    audio = whisperx.load_audio(str(vocals_path))
    result = model.transcribe(audio, batch_size=4)
    segments = result["segments"]
    print(f"  {len(segments)} segments, language: {result.get('language', '?')}")
    
    # Step 2: Align
    print("[2/4] Aligning...")
    try:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
        result = whisperx.align(segments, model_a, metadata, audio, DEVICE, return_char_alignments=False)
        segments = result["segments"]
    except Exception as e:
        print(f"  Align skipped: {e}")
    
    # Step 3: Speaker embedding + clustering
    print("[3/4] Speaker clustering with resemblyzer...")
    encoder = VoiceEncoder()
    
    embeddings = []
    valid_segments = []
    for i, seg in enumerate(segments):
        start, end = seg.get("start", 0), seg.get("end", 0)
        duration = end - start
        if duration < 0.5:  # skip very short segments
            continue
        try:
            samples, sr = extract_segment_wav(vocals_path, start, end)
            if len(samples) < sr * 0.3:  # need at least 0.3s
                continue
            processed = preprocess_wav(samples, source_sr=sr)
            if len(processed) < 1000:
                continue
            emb = encoder.embed_utterance(processed)
            embeddings.append(emb)
            valid_segments.append(seg)
        except Exception as e:
            continue
    
    print(f"  Got embeddings for {len(valid_segments)}/{len(segments)} segments")
    
    if len(embeddings) < 2:
        print("  Not enough segments for clustering")
        for seg in valid_segments:
            seg["speaker"] = "SPEAKER_0"
    else:
        X = np.array(embeddings)
        
        if num_speakers:
            clustering = AgglomerativeClustering(n_clusters=num_speakers)
        else:
            # Auto-detect: try distance threshold
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.7,  # tune this
                metric="cosine",
                linkage="average"
            )
        
        labels = clustering.fit_predict(X)
        n_speakers = len(set(labels))
        print(f"  Detected {n_speakers} speakers")
        
        for seg, label in zip(valid_segments, labels):
            seg["speaker"] = f"SPEAKER_{label}"
    
    # Step 4: Save
    print("[4/4] Saving results...")
    out_dir = OUTPUT_DIR / file_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    transcript = []
    for seg in valid_segments:
        transcript.append({
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": seg.get("text", ""),
            "speaker": seg.get("speaker", "UNKNOWN"),
        })
    
    # Save JSON
    with open(out_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    
    # Save readable txt
    with open(out_dir / "transcript.txt", "w", encoding="utf-8") as f:
        for seg in transcript:
            f.write(f"[{seg['start']:>7.1f} - {seg['end']:>7.1f}] {seg['speaker']:>10}: {seg['text']}\n")
    
    # Cut per-speaker audio
    speakers = sorted(set(seg["speaker"] for seg in transcript))
    full_audio = AudioSegment.from_wav(str(vocals_path))
    
    for speaker in speakers:
        speaker_segs = [s for s in transcript if s["speaker"] == speaker]
        combined = AudioSegment.empty()
        for s in speaker_segs:
            combined += full_audio[int(s["start"]*1000):int(s["end"]*1000)]
        out_path = out_dir / f"{speaker}.wav"
        combined.export(str(out_path), format="wav")
        duration = len(combined) / 1000
        word_count = sum(len(s["text"]) for s in speaker_segs)
        print(f"  {speaker}: {duration:.1f}s, {len(speaker_segs)} segments, ~{word_count} chars")
    
    print(f"  ✅ Done → {out_dir}")
    return transcript


def main():
    print("=== Speaker Diarization Pipeline (resemblyzer) ===")
    
    all_transcripts = {}
    for subdir in sorted(VOCALS_DIR.iterdir()):
        vocals = subdir / "vocals.wav"
        if vocals.exists():
            file_id = subdir.name
            transcript = process_file(vocals, file_id)
            all_transcripts[file_id] = transcript
    
    with open(OUTPUT_DIR / "all_transcripts.json", "w", encoding="utf-8") as f:
        json.dump(all_transcripts, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ All done! → {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
