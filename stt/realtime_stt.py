#!/usr/bin/env python3
"""
Realtime Speech-to-Text using RealtimeSTT.

Runs forever until user says the exit phrase.
All config easily changeable at top of file.
Auto-detects GPUs and uses Nvidia if available.

Usage:
  python3 realtime_stt.py              # uses defaults below
  python3 realtime_stt.py -p instant   # override preset
  python3 realtime_stt.py --help       # see all options
"""

from RealtimeSTT import AudioToTextRecorder
import argparse
import time
import os
from dataclasses import dataclass


# ============================================================================
# DEVICE DETECTION
# ============================================================================

def detect_devices():
    """Detect available CPUs and GPUs."""
    info = {
        'cpu_count': os.cpu_count() or 1,
        'gpus': [],
        'cuda_available': False,
        'best_gpu_index': 0,
    }
    
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        
        if info['cuda_available']:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                info['gpus'].append({'index': i, 'name': gpu_name})
            
            # Prefer Nvidia GPU (look for "NVIDIA" in name)
            for gpu in info['gpus']:
                if 'nvidia' in gpu['name'].lower():
                    info['best_gpu_index'] = gpu['index']
                    break
    except ImportError:
        pass  # torch not installed
    
    return info


def print_device_info(info):
    """Print device detection results."""
    print(f"💻 CPUs: {info['cpu_count']}")
    
    if info['cuda_available']:
        print(f"🎮 GPUs: {len(info['gpus'])}")
        for gpu in info['gpus']:
            marker = " ← selected" if gpu['index'] == info['best_gpu_index'] else ""
            print(f"   [{gpu['index']}] {gpu['name']}{marker}")
    else:
        print("🎮 GPUs: None (CUDA not available, using CPU)")

# ============================================================================
# CONFIGURATION - Change these values to customize behavior
# ============================================================================

EXIT_PHRASE = "exit"           # Say this to stop (case-insensitive, substring match)
DEFAULT_PRESET = "balanced"    # instant | fast | balanced | quality
DEFAULT_LANGUAGE = "en"        # Language code for Whisper

# ============================================================================
# PRESETS - Predefined configurations for different use cases
# ============================================================================

@dataclass
class Preset:
    """Recorder configuration preset."""
    name: str
    model: str
    description: str
    beam_size: int = 5
    silero_sensitivity: float = 0.5
    post_speech_silence_duration: float = 0.2
    gpu_device_index: int = 0


PRESETS = {
    'instant': Preset(
        name='instant',
        model='tiny',
        description='Fastest (~50ms latency)',
        beam_size=1,
        silero_sensitivity=0.6,
        post_speech_silence_duration=0.1,
    ),
    'fast': Preset(
        name='fast',
        model='base',
        description='Fast (~100ms latency)',
        beam_size=3,
        silero_sensitivity=0.5,
        post_speech_silence_duration=0.2,
    ),
    'balanced': Preset(
        name='balanced',
        model='base',
        description='Balanced speed/quality (default)',
        beam_size=5,
        silero_sensitivity=0.5,
        post_speech_silence_duration=0.2,
    ),
    'quality': Preset(
        name='quality',
        model='small',
        description='Best accuracy (~200ms latency)',
        beam_size=5,
        silero_sensitivity=0.4,
        post_speech_silence_duration=0.3,
    ),
}

# ============================================================================
# TRANSCRIBER
# ============================================================================

class ContinuousTranscriber:
    """Transcriber that runs until exit phrase detected."""
    
    def __init__(self, preset: Preset, exit_phrase: str, language: str, gpu_index: int = 0):
        self.preset = preset
        self.exit_phrase = exit_phrase.lower()
        self.language = language
        self.gpu_index = gpu_index
        self.transcripts = []
        self.start_time = None
        self.first_word_time = None
        self.should_stop = False
        self.recorder = None
    
    def _on_text(self, text: str):
        """Called when text is transcribed."""
        if self.first_word_time is None:
            self.first_word_time = time.time()
        
        print(f"✓ {text}")
        self.transcripts.append(text)
        
        # Check for exit phrase
        if self.exit_phrase in text.lower():
            print(f"\n🛑 Exit phrase '{self.exit_phrase}' detected")
            self.should_stop = True
    
    def run(self):
        """Run transcription until exit phrase or Ctrl+C."""
        print(f"🔄 Initializing... (preset: {self.preset.name}, model: {self.preset.model}, gpu: {self.gpu_index})")
        
        self.recorder = AudioToTextRecorder(
            model=self.preset.model,
            language=self.language,
            spinner=False,
            beam_size=self.preset.beam_size,
            silero_sensitivity=self.preset.silero_sensitivity,
            post_speech_silence_duration=self.preset.post_speech_silence_duration,
            gpu_device_index=self.gpu_index,
        )
        
        print(f"🎤 Listening... (say '{self.exit_phrase}' to stop, Ctrl+C to force quit)\n")
        self.start_time = time.time()
        
        try:
            while not self.should_stop:
                text = self.recorder.text()
                if text:
                    self._on_text(text)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n⏹ Interrupted (Ctrl+C)")
        finally:
            self.recorder.shutdown()
            self._print_summary()
    
    def _print_summary(self):
        """Print session summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        first_latency = self.first_word_time - self.start_time if self.first_word_time and self.start_time else 0
        full_text = " ".join(self.transcripts)
        
        print(f"\n📊 Summary:")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   First word latency: {first_latency:.2f}s")
        print(f"   Chunks: {len(self.transcripts)}")
        print(f"   Words: {len(full_text.split()) if full_text else 0}")
        
        if self.transcripts:
            print(f"\n📝 Full transcript:\n   {full_text}")


def main():
    parser = argparse.ArgumentParser(
        description='Realtime speech-to-text (runs until exit phrase)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Current config (edit at top of file):
  EXIT_PHRASE = "{EXIT_PHRASE}"
  DEFAULT_PRESET = "{DEFAULT_PRESET}"
  DEFAULT_LANGUAGE = "{DEFAULT_LANGUAGE}"

Presets:
  instant   - Tiny model, ~50ms latency, decent accuracy
  fast      - Base model, ~100ms latency, good accuracy
  balanced  - Base model, balanced (default)
  quality   - Small model, ~200ms latency, best accuracy

Examples:
  %(prog)s                      # use defaults
  %(prog)s -p instant           # faster, less accurate
  %(prog)s -p quality           # slower, more accurate
  %(prog)s --exit-phrase "stop" # custom exit phrase
  %(prog)s --gpu 1              # force specific GPU
'''
    )
    parser.add_argument('-p', '--preset', choices=list(PRESETS.keys()), default=DEFAULT_PRESET,
                        help=f'Preset (default: {DEFAULT_PRESET})')
    parser.add_argument('-e', '--exit-phrase', default=EXIT_PHRASE,
                        help=f'Exit phrase (default: {EXIT_PHRASE})')
    parser.add_argument('-l', '--language', default=DEFAULT_LANGUAGE,
                        help=f'Language code (default: {DEFAULT_LANGUAGE})')
    parser.add_argument('-g', '--gpu', type=int, default=None,
                        help='GPU index to use (auto-detected if not specified)')
    parser.add_argument('--list-presets', action='store_true',
                        help='List all presets and exit')
    parser.add_argument('--list-devices', action='store_true',
                        help='List detected devices and exit')
    
    args = parser.parse_args()
    
    # Detect devices
    device_info = detect_devices()
    
    if args.list_devices:
        print("\n🖥️  Device Detection:\n")
        print_device_info(device_info)
        return
    
    if args.list_presets:
        print("\n📋 Presets:\n")
        for name, p in PRESETS.items():
            marker = " (default)" if name == DEFAULT_PRESET else ""
            print(f"  {name:10} {p.description}{marker}")
            print(f"             model={p.model}, beam={p.beam_size}, vad={p.silero_sensitivity}")
        return
    
    # Determine GPU index
    gpu_index = args.gpu if args.gpu is not None else device_info['best_gpu_index']
    
    # Print startup info
    print("\n" + "="*60)
    print("🎙️  Realtime Speech-to-Text")
    print("="*60 + "\n")
    print_device_info(device_info)
    print()
    
    preset = PRESETS[args.preset]
    transcriber = ContinuousTranscriber(preset, args.exit_phrase, args.language, gpu_index)
    transcriber.run()


if __name__ == "__main__":
    main()
