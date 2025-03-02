
from datasets import load_dataset
import librosa
import soundfile as sf
import os

# Define directories
SUPPORT_DIR = "F:/code/FSLAKWS/datasets/train/support/"
QUERY_DIR = "F:/code/FSLAKWS/datasets/train/query/"
os.makedirs(SUPPORT_DIR, exist_ok=True)
os.makedirs(QUERY_DIR, exist_ok=True)

# Keywords and languages
keywords = {
    "en": "hello",
    "es": "hola",
    "fr": "bonjour"
}

# Load Common Voice dataset for each language
for lang, keyword in keywords.items():
    try:
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", lang, split="train", 
                              streaming=True, trust_remote_code=True)
        print(f"Loaded dataset for {lang}")
    except Exception as e:
        print(f"Error loading dataset for {lang}: {e}")
        continue
    
    # Extract support audio (short clips with the keyword)
    support_count = 0
    for i, sample in enumerate(dataset):
        if support_count >= 5:  # Limit to 5 support samples per keyword
            break
        sentence = sample["sentence"].lower()
        if keyword in sentence:
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            output_path = f"{SUPPORT_DIR}{lang}_{keyword}_{support_count}.wav"
            sf.write(output_path, audio, 16000)
            support_count += 1
            print(f"Saved support: {output_path}")

    # Create a query audio by concatenating clips
    query_audio = []
    query_duration = 0
    for sample in dataset:
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        query_audio.extend(audio)
        query_duration += len(audio) / 16000
        if query_duration >= 10:  # Stop at ~10 seconds
            break
    output_path = f"{QUERY_DIR}{lang}_long_audio1.wav"
    sf.write(output_path, query_audio, 16000)
    print(f"Saved query: {output_path}")

# Create a mixed query audio (multilingual)
mixed_audio = []
for lang in keywords.keys():
    audio, _ = librosa.load(f"{QUERY_DIR}{lang}_long_audio1.wav", sr=16000)
    mixed_audio.extend(audio[:16000 * 3])  # Take 3 seconds from each
sf.write(f"{QUERY_DIR}mixed_long_audio1.wav", mixed_audio, 16000)
print(f"Saved mixed query: {QUERY_DIR}mixed_long_audio1.wav")