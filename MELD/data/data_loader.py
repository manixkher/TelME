import os
import torch
import librosa
import cv2
import numpy as np
from transformers import AutoProcessor, AutoImageProcessor, RobertaTokenizer

class MELDDataLoader:
    def __init__(self):
        self.audio_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
        self.video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # Add speaker tokens
        speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
        speaker_tokens_dict = {'additional_special_tokens': speaker_list}
        self.tokenizer.add_special_tokens(speaker_tokens_dict)
        
        self.max_audio_length = 400000  # Maximum audio length to process
        
    def load_audio(self, video_path):
        """Load and process audio from video file."""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            audio, rate = librosa.load(video_path)
            duration = librosa.get_duration(y=audio, sr=rate)
            
            if duration > 30:
                return torch.zeros([1412])  # Return zero tensor for long videos
                
            inputs = self.audio_processor(audio, sampling_rate=16000, return_tensors="pt")
            audio_input = inputs["input_values"][0]
            return audio_input[-self.max_audio_length:]
            
        except Exception as e:
            print(f"Error loading audio from {video_path}: {str(e)}")
            return torch.zeros([1412])  # Return zero tensor on error
            
    def load_video(self, video_path):
        """Load and process video frames."""
        try:
            if not os.path.exists(video_path):
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
            return inputs["pixel_values"][0]
            
        except Exception as e:
            print(f"Error loading video from {video_path}: {str(e)}")
            return torch.zeros([8, 3, 224, 224])  # Return zero tensor on error
            
    def process_text(self, text, speaker):
        """Process text input with speaker information."""
        input_string = f'<s{speaker + 1}> {text}'
        prompt = f"Now <s{speaker + 1}> feels"
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
        
    def load_utterance(self, speaker, text, video_path):
        """Load all modalities for a single utterance."""
        audio = self.load_audio(video_path)
        video = self.load_video(video_path)
        text_tokens = self.process_text(text, speaker)
        
        return {
            'text': text_tokens,
            'audio': audio,
            'video': video
        } 