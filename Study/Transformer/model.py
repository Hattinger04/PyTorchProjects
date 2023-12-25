# model by Umar Jamil

import math

import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module): 
    def __init__(self, d_model: int, vocal_size: int): 
        super().__init__()
        self.d_model = d_model
        self.vocal_size = vocal_size
        # given a number this function will provide the same vector every time
        # so it does exactly want we want from the input embedding
        self.embedding = nn.Embedding(vocal_size, d_model) 

    def forward(self, x): 
        # call embedding and multiply with sqrt(d_model) as per the paper
        return self.embedding(x) + math.sqrt(self.d_model)


class PositionalEncoding(nn.Module): 
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # dropout to ensure that the model is not relying too heavily on specific pattern
        self.dropout = nn.Dropout(dropout) 

        pe = torch.zeros(seq_len, d_model)
        # create vector which represents the position of a word in a sentence 
        # shape = (seq_len, 1) => unsqueeze for mulplication with d_model
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # create the divisor of the calculation which the position will be divided by
        # slighty changed function: exp function and log for numerical stability 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # adding batch dimension 
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # when saving a value in a model which is not a parameter => save in buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # adding postiional encoding to every word in the sentence
        # x.shape[1] is seq_len => important because of possible padded sequences 
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # do not learn this function 
        return self.dropout(x)


class LayerNormalization(nn.Module): 
    def __init__(self, features: int, eps: float = 10**-6):
        super().__init__()
        # eps for numerical stability and avoid dividing by 0
        self.eps = eps
        # Parameter makes it learnable
        self.alpha = nn.Parameter(torch.ones(features)) # Multiplied
        self.beta = nn.Parameter(torch.zeros(features)) # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) 

    def forward(self, x): 
        # input = (batch, seq_len, d_model) => (batch, seq_len, d_ff) => (batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.linear_1(x)))
    
class MultiHeadAttentionBlock(nn.Module): 
    def __init__(self, d_model: int, h: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h  == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h # size of one head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout): 
        d_k = query.shape[-1]
        
        # @ in pytorch means matric multiplication (the last two dimensions are used for calculation)
        # (batch, h, seq_len, d_k) => (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # mask to prevent certain values to interact with each others
        if mask is not None: 
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len)
        if dropout is not None: 
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores # second return value for visualization

    def forward(self, q, k, v, mask): 
        query = self.w_q(q) # => not changing dim!
        key = self.w_k(k)
        value = self.w_v(v)

        # do not change batch or seq_len 
        # h * d_k = d_model
        # transpose (batch, seq_len, h, d_k) to (batch, h, seq_len, d_k) for correct attention calculation!
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # transposing back to (batch, seq_len, h, d_k)
        # then change back from h and d_k to d_model with view()
        # =>    contiguous is needed when transforming shape 
        #       because the original tensor would not be contiguous in memory  
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x) # (batch, seq_len, d_model)

# Layer to skip some connections as shown in the paper
class ResidualConnection(nn.Module): 
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer): 
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module): 
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # src_mask is a mask for the input of the encoder  
        # => applying to prevent inferaction of the padding words with the other words
        # first calling forward method of MultHeadAttentionBlock and then of the FeedForwardBlock
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module): 
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask): 
        for layer in self.layers: 
            x = layer(x, mask) # going through all EncoderBlocks 
        return self.norm(x)

class DecoderBlock(nn.Module): 
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, 
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection =  nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # tgt_mask is mask of decoder 
        # we need another mask besides of src_mask in our case because of our translation model 
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module): 
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask): 
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask) # going through all EncoderBlocks 
        return self.norm(x)

# linear layer to convert the embedding into a position of the vocab
class ProjectionLayer(nn.Module): 
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x): 
        # convert (batch, seq_len, d_model) to (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module): 
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, 
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # during inference the encoder's output does not change! 
    def encode(self, src, src_mask): 
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask): 
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x): 
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_srq_len: int, 
                      d_model: int=512, N: int=6, h: int=8,
                      dropout: float=0.1, d_ff=2048) -> Transformer: 
    # create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_srq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters (better than random values!)
    for p in transformer.parameters(): 
        if p.dim() > 1: # do not change bias => Xavier is typically not applied to them
            nn.init.xavier_uniform_(p) # helps precenting vanishing or exploding gradients during training 
    return transformer