from torch.utils.data import Dataset

class FixedWindowDataset(Dataset):
    def __init__(self, sessions, window_size):
        self.window_size = window_size
        self.emoList = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self.flat_utterances = []
        for session in sessions:
            for i, utt in enumerate(session):
                # Each item: (session, idx, utterance)
                self.flat_utterances.append((session, i, utt))

    def __len__(self):
        return len(self.flat_utterances)

    def __getitem__(self, idx):
        session, i, utt = self.flat_utterances[idx]
        # Get previous K utterances (or fewer if at start)
        start = max(0, i - self.window_size)
        context = session[start:i+1]  # include current utterance
        return context 