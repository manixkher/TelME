import glob
import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
import gc

from MELD.data.preprocessing import preprocessing
from MELD.utils.utils import *
from MELD.data.online_data_loader import OnlineMELDDataLoader
from MELD.models.teacher import Teacher_model
from MELD.models.student import Student_Audio, Student_Video
from MELD.inference.fusion import ASF
from MELD.data.dataset import meld_dataset
from MELD.data.fixed_window_dataset import FixedWindowDataset
from MELD.utils.utils import make_fixed_window_batchs


def parse_args():
    parser = argparse.ArgumentParser(description='Fixed-K window inference for MELD')
    parser.add_argument('--window_size', default=3, type=int, help='Number of previous utterances to use as context (K)')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for evaluation')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    mask_time_length: int = 3

def collate_fixed_window(batch):
    # batch: list of (context_utts, utt)
    data_loader = OnlineMELDDataLoader()
    batch_text = []
    batch_audio = []
    batch_video = []
    batch_labels = []
    for context_utts, utt in batch:
        # Compose context string for text (concatenate previous K utterances)
        context_text = ' '.join([u['text'] for u in context_utts]) if context_utts else ''
        speaker = utt['speaker']
        text = (context_text + ' ' + utt['text']).strip()
        video_path = utt['video_path']
        label = utt['emotion']
        # Use data loader to process modalities
        text_tokens = data_loader.process_text(text, speaker)
        audio = data_loader.load_audio(video_path)
        video = data_loader.load_video(video_path)
        batch_text.append(text_tokens['input_ids'])
        batch_audio.append(audio)
        batch_video.append(video)
        batch_labels.append(label)
    # Pad and stack
    batch_text = torch.cat(batch_text, dim=0)
    batch_audio = torch.stack(batch_audio)
    batch_video = torch.stack(batch_video)
    batch_labels = torch.tensor(batch_labels)
    # For attention mask, assume all ones (or adapt as needed)
    attention_masks = (batch_text != 1).long()  # 1 is usually pad_token_id
    return batch_text, attention_masks, batch_audio, batch_video, batch_labels

def evaluation(model_t, audio_s, video_s, fusion, dataloader, emoList):
    label_list = []
    pred_list = []
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader, desc='Inference', unit='batch')):
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
            text_hidden, _ = model_t(batch_input_tokens, attention_masks)
            audio_hidden, _ = audio_s(audio_inputs)
            video_hidden, _ = video_s(video_inputs)
            pred_logits = fusion(text_hidden, audio_hidden, video_hidden)
            pred_label = pred_logits.argmax(1).detach().cpu().numpy()
            true_label = batch_labels.detach().cpu().numpy()
            pred_list.extend(pred_label)
            label_list.extend(true_label)
    return pred_list, label_list

def main(args):
    seed_everything(args.seed)
    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"
    data_path = './dataset/MELD.Raw/'
    test_path = data_path + 'test_meld_emo.csv'
    test_sessions = meld_dataset(preprocessing(test_path))
    # test_sessions is a list of sessions, each a list of utterances
    window_size = args.window_size
    fixed_dataset = FixedWindowDataset(test_sessions, window_size)
    test_loader = DataLoader(fixed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=make_fixed_window_batchs)
    clsNum = len(test_sessions.emoList)
    init_config = Config()
    model_t = Teacher_model(text_model, clsNum)
    model_t.load_state_dict(torch.load('./MELD/save_model/teacher.bin'))
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.cuda()
    model_t.eval()
    audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s.load_state_dict(torch.load('./MELD/save_model/student_audio/total_student.bin'))
    for para in audio_s.parameters():
        para.requires_grad = False
    audio_s = audio_s.cuda()
    audio_s.eval()
    video_s = Student_Video(video_model, clsNum)
    video_s.load_state_dict(torch.load('./MELD/save_model/student_video/total_student.bin'))
    for para in video_s.parameters():
        para.requires_grad = False
    video_s = video_s.cuda()
    video_s.eval()
    hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
    fusion = ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion.load_state_dict(torch.load('./MELD/save_model/total_fusion.bin'))
    for para in fusion.parameters():
        para.requires_grad = False
    fusion = fusion.cuda()
    fusion.eval()
    test_pred_list, test_label_list = evaluation(model_t, audio_s, video_s, fusion, test_loader, test_sessions.emoList)
    print(classification_report(test_label_list, test_pred_list, target_names=test_sessions.emoList, digits=5))
    print(confusion_matrix(test_label_list, test_pred_list, normalize='true'))
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args) 