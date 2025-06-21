import torch
import time
import numpy as np
from dataclasses import dataclass
import warnings
import os
warnings.filterwarnings('ignore')

print('[LOG] Starting MELD/online_inference.py')

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from MELD.models.teacher import Teacher_model
from MELD.models.student import Student_Audio, Student_Video
from MELD.inference.fusion import ASF
from MELD.data.online_data_loader import OnlineMELDDataLoader
from MELD.data.dataset import meld_dataset
from MELD.data.preprocessing import preprocessing
import gc

@dataclass
class Config():
    mask_time_length: int = 3

class OnlineInference:
    def __init__(self, model_paths):
        print('[LOG] Initializing OnlineInference...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emoList = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        # Initialize data loader
        self.data_loader = OnlineMELDDataLoader()
        
        # Use Hugging Face model names
        self.text_model = 'roberta-large'
        self.audio_model = 'facebook/data2vec-audio-base-960h'
        self.video_model = 'facebook/timesformer-base-finetuned-k400'
        
        # Load models
        print('[LOG] Loading models...')
        self._load_models(model_paths)
        print('[LOG] Models loaded.')
        
    def _load_models(self, model_paths):
        clsNum = len(self.emoList)
        init_config = Config()
        
        # Load teacher model
        self.model_t = Teacher_model(self.text_model, clsNum)
        self.model_t.load_state_dict(torch.load(model_paths['teacher']))
        self.model_t = self.model_t.to(self.device)
        self.model_t.eval()
        
        # Load student models
        self.audio_s = Student_Audio(self.audio_model, clsNum, init_config)
        self.audio_s.load_state_dict(torch.load(model_paths['audio']))
        self.audio_s = self.audio_s.to(self.device)
        self.audio_s.eval()
        
        self.video_s = Student_Video(self.video_model, clsNum)
        self.video_s.load_state_dict(torch.load(model_paths['video']))
        self.video_s = self.video_s.to(self.device)
        self.video_s.eval()
        
        # Load fusion model
        hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
        self.fusion = ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
        self.fusion.load_state_dict(torch.load(model_paths['fusion']))
        self.fusion = self.fusion.to(self.device)
        self.fusion.eval()

    def process_single_utterance(self, prior_utterances, speaker, text, video_path):
        """Process a single utterance with timing measurements and prior context."""
        start_time = time.time()
        
        # Load and process data with prior utterances
        data = self.data_loader.load_utterance(prior_utterances, speaker, text, video_path)
        
        # Move data to device
        text_tokens = {k: v.to(self.device) for k, v in data['text'].items()}
        # Ensure audio_input is [1, sequence_length]
        audio_input = data['audio']
        if audio_input.dim() == 1:
            audio_input = audio_input.unsqueeze(0)
        audio_input = audio_input.to(self.device)
        # Ensure video_input is [1, num_frames, channels, height, width]
        video_input = data['video']
        if video_input.dim() == 4:
            video_input = video_input.unsqueeze(0)
        video_input = video_input.to(self.device)
        
        # Get predictions
        inference_start = time.time()
        with torch.no_grad():
            text_hidden, _ = self.model_t(text_tokens['input_ids'], text_tokens['attention_mask'])
            audio_hidden, _ = self.audio_s(audio_input)
            video_hidden, _ = self.video_s(video_input)
                
            pred_logits = self.fusion(text_hidden, audio_hidden, video_hidden)
            pred_emotion = pred_logits.argmax(1)
            emotion = self.emoList[pred_emotion.item()]
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        timing_info = {
            'data_loading': inference_start - start_time,
            'inference_time': inference_time,
            'total_time': total_time
        }
        
        return emotion, pred_emotion.item(), timing_info

def main():
    print('[LOG] Entered main()')
    # Model paths
    model_paths = {
        'teacher': './MELD/save_model/teacher.bin',
        'audio': './MELD/save_model/student_audio/total_student.bin',
        'video': './MELD/save_model/student_video/total_student.bin',
        'fusion': './MELD/save_model/total_fusion.bin'
    }
    
    # Initialize online inference
    print('[LOG] Instantiating OnlineInference...')
    online_infer = OnlineInference(model_paths)
    
    print('[LOG] Loading test dataset...')
    # Use scratch space if available
    scratch_data_dir = os.environ.get('SCRATCH_DATA_DIR', './dataset')
    test_path = os.path.join(scratch_data_dir, 'MELD.Raw', 'test_meld_emo.csv')
    test_dataset = meld_dataset(preprocessing(test_path))
    print(f"[LOG] Loaded {len(test_dataset)} test sessions")
    
    # Flatten the test dataset to a list of utterances with session info
    print('[LOG] Flattening test dataset...')
    flat_utterances = []
    for session in test_dataset:
        for i, utt in enumerate(session):
            flat_utterances.append((session, i, utt))
    print(f"[LOG] Total utterances: {len(flat_utterances)}")
    print('[LOG] First 5 utterances:', flat_utterances[:5])
    
    # Performance metrics
    total_times = []
    data_loading_times = []
    inference_times = []
    
    # For classification metrics
    all_preds = []
    all_labels = []
    
    print('[LOG] Starting online inference loop...')
    # Efficient streaming: process each session by preloading features
    for session_idx, session in enumerate(test_dataset):
        session_features = online_infer.data_loader.preload_session_features(session)
        num_utts = len(session_features)
        print(f'[LOG] Starting session {session_idx+1}/{len(test_dataset)} with {num_utts} utterances')
        prior_utterances = []
        for utter_idx, features in enumerate(session_features):
            speaker = features['speaker']
            text = features['text']
            video_path = features['video_path']
            emotion = features['emotion']
            # Only print progress every 50 utterances and at the last utterance
            print_progress = (utter_idx % 50 == 0 or utter_idx == num_utts - 1)
            data = online_infer.data_loader.process_preloaded_utterance(session_features, utter_idx, prior_utterances, debug=False)
            # Move data to device
            text_tokens = {k: v.to(online_infer.device) for k, v in data['text'].items()}
            audio_input = data['audio']
            if audio_input.dim() == 1:
                audio_input = audio_input.unsqueeze(0)
            audio_input = audio_input.to(online_infer.device)
            video_input = data['video']
            if video_input.dim() == 4:
                video_input = video_input.unsqueeze(0)
            video_input = video_input.to(online_infer.device)
            # Get predictions
            inference_start = time.time()
            with torch.no_grad():
                text_hidden, _ = online_infer.model_t(text_tokens['input_ids'], text_tokens['attention_mask'])
                audio_hidden, _ = online_infer.audio_s(audio_input)
                video_hidden, _ = online_infer.video_s(video_input)
                pred_logits = online_infer.fusion(text_hidden, audio_hidden, video_hidden)
                pred_emotion = pred_logits.argmax(1)
                emotion_pred = online_infer.emoList[pred_emotion.item()]
            inference_time = time.time() - inference_start
            total_time = inference_time  # Data already loaded
            # For classification metrics
            if hasattr(test_dataset, 'emoList'):
                true_label = test_dataset.emoList.index(emotion)
            elif hasattr(online_infer, 'emoList'):
                true_label = online_infer.emoList.index(emotion)
            all_preds.append(pred_emotion.item())
            all_labels.append(true_label)
            total_times.append(total_time)
            inference_times.append(inference_time)
            if print_progress:
                print(f'[LOG] Processed {utter_idx+1}/{num_utts} utterances in session')
                print(f'[LOG] Average total time: {np.mean(total_times):.4f}s')
                print(f'[LOG] Average inference time: {np.mean(inference_times):.4f}s')
            prior_utterances.append((speaker, text))
        print(f'[LOG] Finished session {session_idx+1}/{len(test_dataset)}')
    print('[LOG] Finished inference loop. Calculating final metrics...')
    # Calculate classification metrics
    print('\nClassification Report:')
    if hasattr(test_dataset, 'emoList'):
        print(classification_report(all_labels, all_preds, target_names=test_dataset.emoList, digits=5))
    elif hasattr(online_infer, 'emoList'):
        print(classification_report(all_labels, all_preds, target_names=online_infer.emoList, digits=5))
    print('\nPerformance Statistics:')
    print(f'Average total processing time: {np.mean(total_times):.4f}s')
    print(f'Average inference time: {np.mean(inference_times):.4f}s')
    print(f'95th percentile total time: {np.percentile(total_times, 95):.4f}s')
    print(f'95th percentile inference time: {np.percentile(inference_times, 95):.4f}s')

if __name__ == "__main__":
    print('[LOG] Running as main script')
    gc.collect()
    torch.cuda.empty_cache()
    main()