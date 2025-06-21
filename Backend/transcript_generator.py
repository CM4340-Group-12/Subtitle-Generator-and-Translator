import torch
import torchaudio
import torch.nn.functional as F
import os
import subprocess

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "transcript_gen.pt")

class CharTokenizer:
    def __init__(self, chars=" abcdefghijklmnopqrstuvwxyz'", blank='<BLANK>'):
        self.blank = blank
        self.chars = [blank] + list(chars)
        self.char2idx = {c:i for i,c in enumerate(self.chars)}
        self.idx2char = {i:c for c,i in self.char2idx.items()}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char2idx.get(c,0) for c in text.lower()]

    def decode(self, indices):
        tokens, prev = [], None
        for idx in indices:
            if idx != prev and idx != 0:
                tokens.append(self.idx2char[idx])
            prev = idx
        return ''.join(tokens)

class ASRModel(torch.nn.Module):
    def __init__(self, n_mels, hid, layers, vocab_size):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(n_mels, 128, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, 3, 2, 1), torch.nn.ReLU()
        )
        self.rnn = torch.nn.LSTM(128, hid, layers, batch_first=True, bidirectional=True)
        self.fc  = torch.nn.Linear(hid*2, vocab_size)

    def forward(self, x, lengths):
        x = x.transpose(1,2)       
        x = self.conv(x)            
        x = x.transpose(1,2)       
        out_lengths = (lengths // 4).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, out_lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(unpack), out_lengths

def load_model():
    n_mels = 80
    hid = 256
    layers = 3
    tokenizer = CharTokenizer()
    model = ASRModel(n_mels=n_mels, hid=hid, layers=layers, vocab_size=tokenizer.vocab_size)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def preprocess_audio(path, target_sample_rate=16000, n_mels=80):

    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sr = target_sample_rate

    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=n_mels)(waveform)
    log_mel = torch.log1p(mel_spec)
    feats = log_mel.squeeze(0).transpose(0,1) 
    length = feats.shape[0]
    return feats, length


def extract_audio(video_path: str, output_audio_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def transcribe(path: str) -> str:
    feats, length = preprocess_audio(path, target_sample_rate=16000, n_mels=80)
    feats = feats.unsqueeze(0).to(DEVICE)        
    length_tensor = torch.tensor([length], dtype=torch.long)

    model, tokenizer = load_model()

    with torch.no_grad():
        logits, out_lens = model(feats, length_tensor)
        preds = logits.argmax(dim=-1)[0][:out_lens[0]].cpu().tolist()
        transcription = tokenizer.decode(preds)
        
    return transcription


# -------------------  Uncomment if want to implement the model  ---------------------

# import os
# import glob
# import torch
# import torchaudio
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from jiwer import wer

# torch.set_num_threads(1)
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {DEVICE}")

# # Data source
# # https://www.openslr.org/12
# # Used 'train-clean-100' dataset for training

# class CharTokenizer:
#     def __init__(self, chars=" abcdefghijklmnopqrstuvwxyz'", blank='<BLANK>'):
#         self.blank = blank
#         self.chars = [blank] + list(chars)
#         self.char2idx = {c:i for i,c in enumerate(self.chars)}
#         self.idx2char = {i:c for c,i in self.char2idx.items()}
#         self.vocab_size = len(self.chars)

#     def encode(self, text):
#         return [self.char2idx.get(c,0) for c in text.lower()]

#     def decode(self, indices):
#         tokens, prev = [], None
#         for idx in indices:
#             if idx != prev and idx != 0:
#                 tokens.append(self.idx2char[idx])
#             prev = idx
#         return ''.join(tokens)
    
# class LibriDataset(Dataset):
#     def __init__(self, root, tokenizer, sample_rate=16000, n_mels=80):
#         self.samples = []
#         self.tokenizer = tokenizer
#         for txt in glob.glob(os.path.join(root, '**', '*.txt'), recursive=True):
#             with open(txt) as f:
#                 for line in f:
#                     key, transcript = line.strip().split(maxsplit=1)
#                     wav_path = os.path.join(os.path.dirname(txt), key + '.flac')
#                     if os.path.exists(wav_path):
#                         self.samples.append((wav_path, transcript))
#         assert self.samples, f"No data found in {root}"        
#         self.mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)

#     def __len__(self): return len(self.samples)
#     def __getitem__(self, idx):
#         wav, transcript = self.samples[idx]
#         waveform, sr = torchaudio.load(wav)
#         if waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0, keepdim=True)
#         mel = self.mel_tf(waveform)
#         feats = torch.log1p(mel).squeeze(0).transpose(0,1)
#         label = torch.tensor(self.tokenizer.encode(transcript), dtype=torch.long)
#         return feats, label
    
# def collate_fn(batch):
#     feats, labels = zip(*batch)
#     lens_f = torch.tensor([f.shape[0] for f in feats])
#     lens_l = torch.tensor([len(l) for l in labels])
#     B, M = len(batch), feats[0].shape[1]
#     T, L = lens_f.max(), lens_l.max()
#     batch_f = torch.zeros(B, T, M)
#     batch_l = torch.zeros(B, L, dtype=torch.long)
#     for i, (f, l) in enumerate(zip(feats, labels)):
#         batch_f[i, :f.shape[0]] = f
#         batch_l[i, :l.shape[0]] = l
#     return batch_f, batch_l, lens_f, lens_l        

# class ASRModel(nn.Module):
#     def __init__(self, n_mels, hid, layers, vocab_size):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(n_mels,128,3,2,1), nn.ReLU(),
#             nn.Conv1d(128,128,3,2,1), nn.ReLU()
#         )
#         self.rnn = nn.LSTM(128, hid, layers, batch_first=True, bidirectional=True)
#         self.fc  = nn.Linear(hid*2, vocab_size)

#     def forward(self, x, lengths):
#         x = x.transpose(1,2)
#         x = self.conv(x)
#         x = x.transpose(1,2)
#         out_lengths = (lengths // 4).cpu()
#         packed = pack_padded_sequence(x, out_lengths, batch_first=True, enforce_sorted=False)
#         packed_out, _ = self.rnn(packed)
#         unpack, _ = pad_packed_sequence(packed_out, batch_first=True)
#         return self.fc(unpack), out_lengths
    
# DATA_DIR = 'train-clean-100/LibriSpeech/train-clean-100'
# BATCH_SIZE = 16
# EPOCHS = 30
# LEARNING_RATE = 1e-3

# tokenizer = CharTokenizer()
# dataset = LibriDataset(DATA_DIR, tokenizer)
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)    

# model = ASRModel(n_mels=80, hid=256, layers=3, vocab_size=tokenizer.vocab_size).to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# best_loss = float('inf')
# for epoch in range(1, EPOCHS+1):
#     model.train()
#     total_loss = 0.0
#     for feats, labels, feat_lens, label_lens in loader:
#         feats, labels = feats.to(DEVICE), labels.to(DEVICE)
#         optimizer.zero_grad()
#         logits, out_lens = model(feats, feat_lens)
#         log_probs = F.log_softmax(logits, dim=-1).transpose(0,1)
#         loss = ctc_loss(log_probs, labels, out_lens, label_lens)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     avg_loss = total_loss / len(loader)
#     print(f"Epoch {epoch}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         torch.save(model.state_dict(), 'transcript_gen.pt')

# # 8. Inference on an External Audio File
# test_wav = 'sample_audio.wav'
# waveform, sr = torchaudio.load(test_wav)

# # Convert to mono if needed
# if waveform.shape[0] > 1:
#     waveform = waveform.mean(dim=0, keepdim=True)

# mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80)(waveform)
# feats = torch.log1p(mel).squeeze(0).transpose(0,1).unsqueeze(0).to(DEVICE)
# feat_len = torch.tensor([feats.shape[1]])

# model.eval()
# with torch.no_grad():
#     logits, out_lens = model(feats, feat_len)
#     preds = logits.argmax(dim=-1)[0][:out_lens[0]].cpu().tolist()
#     transcription = tokenizer.decode(preds)

# print("Transcription:", transcription)
