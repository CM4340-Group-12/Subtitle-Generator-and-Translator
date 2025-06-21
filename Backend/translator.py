import torch
import sentencepiece as spm
import math
import torch.nn as nn
import os

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(__file__)
EN_TOKENIZER_PATH = os.path.join(BASE_DIR, "en.model")
SI_TOKENIZER_PATH = os.path.join(BASE_DIR, "si.model")
MODEL_PATH = os.path.join(BASE_DIR, "eng_to_sinhala_si.pth")


en_tokenizer = spm.SentencePieceProcessor(model_file=EN_TOKENIZER_PATH)
si_tokenizer = spm.SentencePieceProcessor(model_file=SI_TOKENIZER_PATH)

SRC_VOCAB_SIZE = en_tokenizer.get_piece_size()
TGT_VOCAB_SIZE = si_tokenizer.get_piece_size()
BOS_ID = si_tokenizer.bos_id()
EOS_ID = si_tokenizer.eos_id()
PAD_ID = 0

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pe = torch.zeros((maxlen, emb_size))
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class Seq2SeqTransformer(nn.Module):
    def __init__(self, enc_layers, dec_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size, ff_dim=512):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=PAD_ID)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(emb_size)
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead,
                                          num_encoder_layers=enc_layers, num_decoder_layers=dec_layers,
                                          dim_feedforward=ff_dim, batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.pos_encoder(self.src_tok_emb(src))
        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.generator(output)

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones((sz, sz), device=DEVICE) * float('-inf'), diagonal=1)

def greedy_decode(model, sentence, max_len=50):
    model.eval()
    src = torch.tensor(en_tokenizer.encode(sentence)).unsqueeze(0).to(DEVICE)
    memory = model.transformer.encoder(model.pos_encoder(model.src_tok_emb(src)))
    tgt = torch.tensor([[BOS_ID]], device=DEVICE)

    for _ in range(max_len):
        tgt_emb = model.pos_encoder(model.tgt_tok_emb(tgt))
        tgt_mask = generate_square_subsequent_mask(tgt.size(1))
        out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        out = model.generator(out[:, -1])
        next_token = out.argmax(dim=-1).item()
        tgt = torch.cat([tgt, torch.tensor([[next_token]], device=DEVICE)], dim=1)
        if next_token == EOS_ID:
            break
    return tgt.squeeze().tolist()
    
def decode_tokens(token_ids):
    return si_tokenizer.decode(token_ids)

# Recreate the exact architecture and Load the model
model = Seq2SeqTransformer(4, 4, 512, 8, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
def translate_sentence(sentence):
    token_ids = greedy_decode(model, sentence)
    if BOS_ID in token_ids: token_ids.remove(BOS_ID)
    if EOS_ID in token_ids: token_ids = token_ids[:token_ids.index(EOS_ID)]
    return decode_tokens(token_ids)


# -------------------  Uncomment if want to implement the model  ---------------------

# import os
# import re
# import math
# import unicodedata
# import pandas as pd
# import sentencepiece as spm
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# import gc
# gc.collect()
# torch.cuda.empty_cache()

# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
#                       "cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", DEVICE)

# # Data source
# # https://opus.nlpl.eu/sample/en&si/NLLB&v1/sample

# df = pd.read_csv("english_sinhala_philosophical_10000.csv")

# df.head(5)

# df.shape

# def clean_text(text, lang='en'):
#     text = unicodedata.normalize("NFKC", text)
#     text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     text = re.sub(r"\s([?.!,\"'](?:\s|$))", r"\1", text)
#     text = re.sub(r"([?.!,\"'])\s*", r"\1 ", text)

#     if lang == 'en':
#         text = text.lower()
#         contractions = {
#             "won't": "will not", "can't": "cannot", "n't": " not",
#             "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
#             "'t": " not", "'ve": " have", "'m": " am"
#         }
#         for c, repl in contractions.items():
#             text = text.replace(c, repl)
#     return text

# df['english'] = df['english'].apply(lambda x: clean_text(x, 'en'))
# df['sinhala'] = df['sinhala'].apply(lambda x: clean_text(x, 'si'))

# df = df[(df['english'].str.strip() != "") & 
#         (df['sinhala'].str.strip() != "")]
# df = df[~df['english'].str.contains(r"[=<>_/\\@#$%^&*~|]", regex=True)]

# if not os.path.exists("en.model") or not os.path.exists("si.model"):
#     df['english'].to_csv("en.txt", index=False, header=False)
#     df['sinhala'].to_csv("si.txt", index=False, header=False)
#     spm.SentencePieceTrainer.Train(input="en.txt", model_prefix="en", vocab_size=6000)
#     spm.SentencePieceTrainer.Train(input="si.txt", model_prefix="si", vocab_size=6000)

# en_tokenizer = spm.SentencePieceProcessor(model_file="en.model")
# si_tokenizer = spm.SentencePieceProcessor(model_file="si.model")

# SRC_VOCAB_SIZE = en_tokenizer.get_piece_size()
# TGT_VOCAB_SIZE = si_tokenizer.get_piece_size()
# BOS_ID = si_tokenizer.bos_id()
# EOS_ID = si_tokenizer.eos_id()
# PAD_ID = 0

# def filter_garbage_pairs(df, en_tok, si_tok, max_src_len=100, max_tgt_len=100, max_len_ratio=3):
#     filtered = []
#     for _, row in df.iterrows():
#         src_len = len(en_tok.encode(row['english']))
#         tgt_len = len(si_tok.encode(row['sinhala']))
#         if src_len == 0 or tgt_len == 0:
#             continue
#         if src_len > max_src_len or tgt_len > max_tgt_len:
#             continue
#         ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
#         if ratio > max_len_ratio:
#             continue
#         filtered.append(row)
#     return pd.DataFrame(filtered)

# print(f"Before filtering: {len(df)}")
# df = filter_garbage_pairs(df, en_tokenizer, si_tokenizer)
# print(f"After filtering: {len(df)}")

# train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# class TranslationDataset(Dataset):
#     def __init__(self, dataframe):
#         self.data = dataframe

#     def __len__(self): return len(self.data)

#     def __getitem__(self, idx):
#         src = en_tokenizer.encode(self.data.iloc[idx]['english'], out_type=int)
#         tgt = si_tokenizer.encode(self.data.iloc[idx]['sinhala'], out_type=int)
#         tgt = [BOS_ID] + tgt + [EOS_ID]
#         return torch.tensor(src), torch.tensor(tgt)

# def collate_fn(batch):
#     src_batch, tgt_batch = zip(*batch)
#     src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_ID, batch_first=True)
#     tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_ID, batch_first=True)
#     return src_batch, tgt_batch

# train_loader = DataLoader(TranslationDataset(train_df), batch_size=32, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(TranslationDataset(val_df), batch_size=32, shuffle=False, collate_fn=collate_fn)

# class PositionalEncoding(nn.Module):
#     def __init__(self, emb_size, dropout=0.1, maxlen=5000):
#         super().__init__()
#         den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         pe = torch.zeros((maxlen, emb_size))
#         pe[:, 0::2] = torch.sin(pos * den)
#         pe[:, 1::2] = torch.cos(pos * den)
#         self.dropout = nn.Dropout(p=dropout)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         return self.dropout(x + self.pe[:, :x.size(1)])

# class Seq2SeqTransformer(nn.Module):
#     def __init__(self, enc_layers, dec_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size, ff_dim=512):
#         super().__init__()
#         self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=PAD_ID)
#         self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=PAD_ID)
#         self.pos_encoder = PositionalEncoding(emb_size)
#         self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead,
#                                           num_encoder_layers=enc_layers, num_decoder_layers=dec_layers,
#                                           dim_feedforward=ff_dim, batch_first=True)
#         self.generator = nn.Linear(emb_size, tgt_vocab_size)

#     def forward(self, src, tgt):
#         src_emb = self.pos_encoder(self.src_tok_emb(src))
#         tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt))
#         tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
#         output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
#         return self.generator(output)

# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=PAD_ID):
#         super().__init__()
#         self.criterion = nn.KLDivLoss(reduction='batchmean')
#         self.confidence = 1.0 - label_smoothing
#         self.smoothing = label_smoothing
#         self.vocab_size = tgt_vocab_size
#         self.ignore_index = ignore_index

#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=-1)
#         true_dist = torch.zeros_like(pred).fill_(self.smoothing / (self.vocab_size - 2))
#         ignore = target == self.ignore_index
#         target = target.masked_fill(ignore, 0)
#         true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
#         true_dist.masked_fill_(ignore.unsqueeze(1), 0.0)
#         return self.criterion(pred, true_dist)

# class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, model_size, warmup_steps=4000, last_epoch=-1):
#         self.model_size = model_size
#         self.warmup_steps = warmup_steps
#         self._step = 0
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         self._step += 1
#         scale = self.model_size ** -0.5 * min(self._step ** -0.5, self._step * self.warmup_steps ** -1.5)
#         return [base_lr * scale for base_lr in self.base_lrs]

# model = Seq2SeqTransformer(4, 4, 512, 8, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
# loss_fn = LabelSmoothingLoss(0.1, TGT_VOCAB_SIZE)
# optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
# scheduler = NoamScheduler(optimizer, model_size=512)

# EPOCHS = 30
# best_val_loss = float('inf')
# patience, patience_counter = 3, 0

# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
#     for src, tgt in train_loader:
#         src, tgt = src.to(DEVICE), tgt.to(DEVICE)
#         tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
#         output = model(src, tgt_input)
#         loss = loss_fn(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1} Train Loss: {total_loss:.4f}")

#     gc.collect()
#     torch.cuda.empty_cache()
    
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for src, tgt in val_loader:
#             src, tgt = src.to(DEVICE), tgt.to(DEVICE)
#             tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
#             output = model(src, tgt_input)
#             loss = loss_fn(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
#             val_loss += loss.item()
#     val_loss /= len(val_loader)
#     print(f"Validation Loss: {val_loss:.4f}")

#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), "eng_to_sinhala_si.pth")
#         patience_counter = 0
#     else:
#         patience_counter += 1
#         if patience_counter >= patience:
#             print("Early stopping.")
#             break

# def generate_square_subsequent_mask(sz):
#     return torch.triu(torch.ones((sz, sz), device=DEVICE) * float('-inf'), diagonal=1)



# def greedy_decode(model, sentence, max_len=50):
#     model.eval()
#     src = torch.tensor(en_tokenizer.encode(sentence)).unsqueeze(0).to(DEVICE)
#     memory = model.transformer.encoder(model.pos_encoder(model.src_tok_emb(src)))
#     tgt = torch.tensor([[BOS_ID]], device=DEVICE)

#     for _ in range(max_len):
#         tgt_emb = model.pos_encoder(model.tgt_tok_emb(tgt))
#         tgt_mask = generate_square_subsequent_mask(tgt.size(1))
#         out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
#         out = model.generator(out[:, -1])
#         next_token = out.argmax(dim=-1).item()
#         tgt = torch.cat([tgt, torch.tensor([[next_token]], device=DEVICE)], dim=1)
#         if next_token == EOS_ID:
#             break
#     return tgt.squeeze().tolist()
    
# def decode_tokens(token_ids):
#     return si_tokenizer.decode(token_ids)
    
# def translate_sentence(sentence):
#     token_ids = greedy_decode(model, sentence)
#     if BOS_ID in token_ids: token_ids.remove(BOS_ID)
#     if EOS_ID in token_ids: token_ids = token_ids[:token_ids.index(EOS_ID)]
#     return decode_tokens(token_ids)


# def compute_bleu(model, dataloader, num_samples=100):
#     references, hypotheses = [], []
#     with torch.no_grad():
#         for i, (src, tgt) in enumerate(dataloader):
#             for j in range(src.size(0)):
#                 en = en_tokenizer.decode(src[j].tolist())
#                 ref = si_tokenizer.decode(tgt[j].tolist()[1:-1])
#                 pred = translate_sentence(en)
#                 references.append([ref.split()])
#                 hypotheses.append(pred.split())
#                 if len(references) >= num_samples:
#                     break
#             if len(references) >= num_samples:
#                 break
#     return corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method4)

# # === Try a Translation ===
# example = "No matter at how many places you will look for the happiness. You won't find it. Because you never lost it outside. It is still inside of you."
# print("English:", example)
# print("Sinhala:", translate_sentence(example))

# bleu = compute_bleu(model, val_loader)
# print(f"BLEU Score: {bleu * 100: .2f}")