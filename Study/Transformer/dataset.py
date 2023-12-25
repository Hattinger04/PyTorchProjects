# dataset by Umar Jamil

import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset): 
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_tgt_pair = self.dataset[index]
        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # returns input ids to each word in an array
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # calculate how many padding tokens needed for reaching seq_len 
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # sos and eos 
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # only sos at decoder 

        # enc_num_padding_tokens / dec_num_padding_tokens should never become negative!
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0: 
            raise ValueError("Sentence is too long!")

        # add SOS and EOS to source text and fill up with padding tokens 
        encoder_input = torch.cat([
            self.sos_token, 
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token, 
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        # adding only SOS 
        decoder_input = torch.cat([
            self.sos_token, 
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        # add EOS to the label => expecting output of the decoder 
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64), 
            self.eos_token, 
            # needing same amount of padding as in decoder
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
             # all tokens which are padding are not ok 
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            # prevent looking at future words
            # (1, seq_len) & (1, seq_len, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label, # (seq_len)
            "src_text": src_text, 
            "tgt_text": tgt_text
        }
    
def causal_mask(size: int): 
    # trui = give me all the values above the diagonal
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0 # everything that is one will be False! 