# training by Umar Jamil

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_weights_file_path, get_config, latest_weights_file_path

from tqdm import tqdm

# libraries from huggingface 
#   => the aim of this project is builing a transformer not everything from scratch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer # "train" the tokenzier => create the voc given the list of sentences
from tokenizers.pre_tokenizers import Whitespace # split words at "space"

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device): 
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")

    # precomute the encoder output and reuse it for every token from the decoder
    # => in inference we only need to compute the encode output once!
    encoder_output = model.encode(source, source_mask)
    # initialize the decoder input with the sos token 
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True: 
        if decoder_input.size(1) == max_len: 
            break
        # build mask for the target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token 
        prob = model.project(out[:, -1]) # get only probabilities of the last token
        # select token with the max probability => greedy search
        _, next_word = torch.max(prob, dim=1) 
        # then append word to next input for following iteration
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        if next_word == eos_idx: 
            break

    return decoder_input.squeeze(0) # remove batch dim


def run_validation(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2): 
    model.eval()
    count = 0 

    console_width = 80

    with torch.no_grad():
        for batch in validation_dataset: 
            count+=1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation!"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy()) # convert back to word
            
            # printing to consoie with special method because of tqdm progress bar
            print_msg("-"*console_width) # printing bars
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples: 
                break

    # send to tensorboard
    if writer:
        pass

def get_all_sentences(dataset, lang): 
    for item in dataset: 
        yield item["translation"][lang]

def get_or_build_tokenizer(config, dataset, lang): 
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path): 
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # unk_token is a unknown word
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else: 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config): 
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split="train") # we will split ourself 

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # keep 90 % for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # randomly splitting dataset

    train_dataset = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, 
                                     config["lang_src"], config["lang_tgt"], config["seq_len"])
    
    val_dataset = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, 
                                    config["lang_src"], config["lang_tgt"], config["seq_len"])
    
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw: 
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max lenght of source sentence: {max_len_src}")
    print(f"Max lenght of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True) # batch size of 1 because of processing each sentence one by one

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len): 
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], 
                              config["d_model"])
    return model


def train_model(config): 
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["TPU"]: 
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    print(f"Using device {device}")

    # make sure weights folder is created 
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # if having multiple gpu's: 
    if config["count_gpu"] >= 2:  
        # TODO: not working yet!
        model = nn.DataParallel(model)
        print("Using multiple GPU's")
    print(f"Structure of the model: {model}")
    
    # calculate number of paramters 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    # start tensorboard for visualisation
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename is not None:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # do not involve padding when calculating loss
    # label_smoothing: take 0.1 of score and give it to the others => helps avoiding overfitting 
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)

    

    for epoch in range(initial_epoch, config["num_epochs"]): 
        epoch_loss = 0.0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()
            
            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch, 1, 1, seq_len) => hiding padding
            decoder_mask = batch["decoder_mask"].to(device) # (batch, 1, seq_len, seq_len) => hiding padding and subsequent words

            # run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device) # (batch, seq_len)

            # compare label and proj_output
            # (batch, seq_len, tgt_vocab_size) => # (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log the loss
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            epoch_loss += loss.item()

            # backpropagation
            loss.backward() 

            # update weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1 

        batch_iterator.write(f"Loss of epoch {epoch}: {epoch_loss/len(train_dataloader)}")
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, 
                           lambda msg: batch_iterator.write(msg), global_step, writer)

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch, 
            "model_state_dict": model.state_dict(), 
            "optimizer_state_dict": optimizer.state_dict(), 
            "global_step": global_step
        }, model_filename)

if __name__ == "__main__": 
    config = get_config()
    train_model(config)