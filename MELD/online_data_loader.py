import os
import torch
import librosa
import cv2
import numpy as np
from transformers import AutoProcessor, AutoImageProcessor, RobertaTokenizer
import time

class OnlineMELDDataLoader:
    def __init__(self):
        """Initialize the online data loader with necessary processors and tokenizers."""
        self.audio_processor = AutoProcessor.from_pretrained('facebook/data2vec-audio-base-960h')
        self.video_processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k400')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # Add speaker tokens
        speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
        speaker_tokens_dict = {'additional_special_tokens': speaker_list}
        self.tokenizer.add_special_tokens(speaker_tokens_dict)
        
        self.max_audio_length = 400000  # Maximum audio length to process
        self.utterance_counter = 0
        print(f"[DATALOADER] Initialized. Current working directory: {os.getcwd()}")
        
    def load_audio(self, video_path):
        """Load and process audio from video file for online inference."""
        abs_video_path = os.path.abspath(video_path)
        print(f"[AUDIO] Loading audio from: {abs_video_path}")
        start = time.time()
        try:
            if not os.path.exists(video_path):
                print(f"[AUDIO] File not found: {abs_video_path}")
                raise FileNotFoundError(f"Video file not found: {abs_video_path}")
                
            audio, rate = librosa.load(video_path)
            duration = librosa.get_duration(y=audio, sr=rate)
            
            if duration > 30:
                print(f"[AUDIO] Duration > 30s, returning zeros.")
                return torch.zeros([1412])  # Return zero tensor for long videos
                
            inputs = self.audio_processor(audio, sampling_rate=16000, return_tensors="pt")
            audio_input = inputs["input_values"][0]
            print(f"[AUDIO] Loaded, shape: {audio_input.shape}, time: {time.time()-start:.2f}s")
            return audio_input[-self.max_audio_length:]
            
        except Exception as e:
            print(f"[AUDIO] Error loading audio from {abs_video_path}: {str(e)}")
            return torch.zeros([1412])  # Return zero tensor on error
            
    def load_video(self, video_path):
        """Load and process video frames for online inference."""
        print(f"[VIDEO] Loading video from: {video_path}")
        start = time.time()
        try:
            if not os.path.exists(video_path):
                print(f"[VIDEO] File not found: {video_path}")
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            video = cv2.VideoCapture(video_path)
            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            
            if length >= 8:
                step = length // 8
                count = 0
                while video.isOpened():
                    ret, image = video.read()
                    if not ret:
                        break
                    count += 1
                    if count % step == 0:
                        frames.append(image)
            else:
                while video.isOpened():
                    ret, image = video.read()
                    if not ret:
                        break
                    frames.append(image)
                    
                # Pad with last frame if needed
                lack = 8 - len(frames)
                if lack > 0:
                    extend_frames = [frames[-1].copy() for _ in range(lack)]
                    frames.extend(extend_frames)
                    
            video.release()
            
            inputs = self.video_processor(frames[:8], return_tensors="pt")
            print(f"[VIDEO] Loaded, shape: {inputs['pixel_values'][0].shape}, time: {time.time()-start:.2f}s")
            return inputs["pixel_values"][0]
            
        except Exception as e:
            print(f"[VIDEO] Error loading video from {video_path}: {str(e)}")
            return torch.zeros([8, 3, 224, 224])  # Return zero tensor on error
            
    def process_text(self, prior_utterances, current_speaker, current_text, debug=False):
        """Process text input with speaker information and prior utterances for online inference."""
        # Build input string from prior utterances
        input_string = ""
        for speaker, text in prior_utterances:
            input_string += f'<s{speaker + 1}> {text} '
        input_string += f'<s{current_speaker + 1}> {current_text}'
        prompt = f"Now <s{current_speaker + 1}> feels"
        concat_string = input_string.strip() + " </s> " + prompt
        # Tokenize and truncate
        tokenized = self.tokenizer.tokenize(concat_string)
        truncated = tokenized[-511:]  # Keep last 511 tokens
        ids = self.tokenizer.convert_tokens_to_ids(truncated)
        # Add padding
        pad_len = 512 - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.tokenizer.pad_token_id] * pad_len
        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([attention_mask])
        }
        
    def load_utterance(self, prior_utterances, speaker, text, video_path):
        """Load all modalities for a single utterance for online inference, with prior context."""
        self.utterance_counter += 1
        if self.utterance_counter % 50 == 0:
            abs_video_path = os.path.abspath(video_path)
            print(f"[DATALOADER] Processed {self.utterance_counter} utterances... Last file: {abs_video_path}")
        start = time.time()
        audio = self.load_audio(video_path)
        video = self.load_video(video_path)
        text_tokens = self.process_text(prior_utterances, speaker, text)
        abs_video_path = os.path.abspath(video_path)
        print(f"[UTTERANCE] Done loading utterance from {abs_video_path}, total time: {time.time()-start:.2f}s")
        return {
            'text': text_tokens,
            'audio': audio,
            'video': video
        }

    def preload_session_features(self, session):
        """Preload all features (audio, video, text) for a session into memory."""
        features = []
        for speaker, text, video_path, emotion in session:
            audio = self.load_audio(video_path)
            video = self.load_video(video_path)
            # Don't build text tokens yet; do it per-utterance for context
            features.append({
                'speaker': speaker,
                'text': text,
                'video_path': video_path,
                'emotion': emotion,
                'audio': audio,
                'video': video
            })
        return features

    def process_preloaded_utterance(self, features, utter_idx, prior_utterances, debug=False):
        """Given preloaded features and prior utterances, build model input for utter_idx."""
        current = features[utter_idx]
        text_tokens = self.process_text(prior_utterances, current['speaker'], current['text'], debug=False)
        return {
            'text': text_tokens,
            'audio': current['audio'],
            'video': current['video'],
            'emotion': current['emotion'],
            'speaker': current['speaker'],
            'text_raw': current['text'],
            'video_path': current['video_path']
        } 