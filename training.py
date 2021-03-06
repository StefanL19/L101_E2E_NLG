import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from encoders import NMTEncoder
from decoders import NMTDecoder
from data_loader import NMTDataset, generate_nmt_batches
from models import NMTModel

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """Handle the training state updates.
    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better
    
    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        #Saving the model to recover from a class
        #torch.save(model.state_dict(), train_state['model_filename'])

        # Saving the model to recover immediately
        torch.save(model, train_state['model_filename'])
        
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]
         
        # If loss worsened
        if loss_t >= loss_tm1:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = loss_t

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes
    
    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true

def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)
    
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()
    
    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)

def attention_energy_loss(stacked_attentions, energy_caps, valid_indices):
    """
      The l2 loss will encourage the attention vectors to be close to their caps
    """
    # The attention should be distributed only to the output words that participate in the sequence
   # stacked_attentions = stacked_attentions*valid_indices

    # Sum along the last dimension to obtain the stacked attentions
    attention_energies = torch.sum(stacked_attentions, dim=-1)

    # Obtain the MSE loss with the energy caps (each word should givea fixed energy to the output sequence)
    loss_f_n = torch.nn.MSELoss()

    return loss_f_n(attention_energies, energy_caps)

def attention_sparsity_loss(stacked_attentions, valid_indices):
    """
      The minimum entropy loss will encourage the attention distribution to be both sparse and sharp
    """
    from sparsemax import Sparsemax
    activation = Sparsemax(dim=2)
    # if domain == "freq": 
    #   reduce_dim = 1
    # else:
    #   reduce_dim = 2

    # For now discard the valid indices pruning

    # We will be minimizing the Renyi Entropy to encourage sparsity and sharpness
#    energies_sum = (1/(1-0.9))*torch.log(torch.sum(torch.pow(stacked_attentions, 0.9), dim=1))
    stacked_attentions = activation(stacked_attentions)
    stacked_attentions += 1e-8
    energies_sum = (1/(1-0.5))*torch.log(torch.sum(torch.pow(stacked_attentions, 0.5), dim=2))
    #energies_sum = -torch.sum((stacked_attentions*torch.log(stacked_attentions)), dim =2)
    energies_sum  = energies_sum.sum()

    # Normalize the response 
    energies_sum = energies_sum / stacked_attentions.size()[0]

    return energies_sum


args = Namespace(dataset_csv="data/inp_and_gt_name_near_food_no_inform.csv",
                 vectorizer_file="test.json",
                 model_state_file="test.pth",
                 save_dir="data/trained_models/15/",
                 reload_from_files=False,
                 expand_filepaths_to_save_dir=True,
                 cuda=True,
                 seed=1337,
                 learning_rate=5e-4,
                 batch_size=32,
                 num_epochs=15,
                 early_stopping_criteria=10,              
                 source_embedding_size=48, 
                 target_embedding_size=48,
                 encoding_size=256,
                 catch_keyboard_interrupt=True)

if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)
    
    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))


# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")
    
print("Using CUDA: {}".format(args.cuda))

# Set seed for reproducibility
set_seed_everywhere(args.seed, args.cuda)

# handle dirs
handle_dirs(args.save_dir)

if args.reload_from_files and os.path.exists(args.vectorizer_file):
    print("Found the vectorizer in path: {}".format(args.vectorizer_file))

    # training from a checkpoint
    dataset = NMTDataset.load_dataset_and_load_vectorizer(args.dataset_csv,
                                                          args.vectorizer_file)
else:
    print("Not loading the vectorizer from files: {}".format(args.vectorizer_file))
    # create dataset and vectorizer
    dataset = NMTDataset.load_dataset_and_make_vectorizer(args.dataset_csv)
    dataset.save_vectorizer(args.vectorizer_file)

vectorizer = dataset.get_vectorizer()
print("The max target length of the vectorizer is: ", vectorizer.max_target_length)
model = NMTModel(source_vocab_size=len(vectorizer.source_vocab), 
                 source_embedding_size=args.source_embedding_size, 
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=args.target_embedding_size, 
                 encoding_size=args.encoding_size,
                 target_bos_index=vectorizer.target_vocab.begin_seq_index,
                 is_training=True,
                 attention_mode="bahdanau")

if args.reload_from_files and os.path.exists(args.model_state_file):
    model.load_state_dict(torch.load(args.model_state_file))
    print("Reloaded model")
else:
    print("New model")


model = model.to(args.device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)
mask_index = vectorizer.target_vocab.mask_index
train_state = make_train_state(args)

epoch_bar = tqdm(desc='training routine', 
                          total=args.num_epochs,
                          position=0)

dataset.set_split('train')
train_bar = tqdm(desc='split=train',
                          total=dataset.get_num_batches(args.batch_size), 
                          position=1, 
                          leave=True)
dataset.set_split('val')
val_bar = tqdm(desc='split=val',
                        total=dataset.get_num_batches(args.batch_size), 
                        position=1, 
                        leave=True)

with open("training_monitor.txt", "a") as f:
            f.write("Bahdanau Attention, Sparsemax Freq Domain Renyi Entropy, Cap 2 weight 0.01")
            f.write("\n")

try:
    for epoch_index in range(args.num_epochs):
        sample_probability = (20 + epoch_index) / 100#args.num_epochs
        
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset

        # setup: batch generator, set loss and acc to 0, set train mode on
        dataset.set_split('train')
        batch_generator = generate_nmt_batches(dataset, 
                                               batch_size=args.batch_size, 
                                               device=args.device)
        running_loss = 0.0
        running_general_loss = 0.0
        running_attention_energy_loss = 0.0
        running_acc = 0.0
        running_attention_sparsity_loss = 0.0
        model.train()
        
        for batch_index, batch_dict in enumerate(batch_generator):
            # the training routine is these 5 steps:

            # --------------------------------------    
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            y_pred, stacked_attentions = model(batch_dict['x_source'], 
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'],
                           sample_probability=sample_probability)
            caps = np.ones((stacked_attentions.size()[0],  stacked_attentions.size()[1]), dtype=np.float32)*2.

            energy_caps = torch.tensor(caps, requires_grad=False)
            energy_caps = energy_caps.to(stacked_attentions.device)

            # step 3. compute the loss
            gen_loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

            # The valid indices
            valid_ind = torch.ne(batch_dict['y_target'], mask_index).float()

            valid_ind = valid_ind.view(valid_ind.size()[0], 1, valid_ind.size()[1])


            energy_loss = attention_energy_loss(stacked_attentions, energy_caps, valid_ind)
            sparsity_loss = 0.01*attention_sparsity_loss(stacked_attentions, valid_ind) #0.01*attention_sparsity_loss(entropy_energies)

            loss = gen_loss + energy_loss + sparsity_loss

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

            # -----------------------------------------
            # compute the running loss and running accuracy
            running_loss += (loss.item() - running_loss) / (batch_index + 1)
            running_general_loss += (gen_loss.item() - running_general_loss) / (batch_index + 1)
            running_attention_energy_loss += (energy_loss.item() - running_attention_energy_loss) / (batch_index + 1)
            running_attention_sparsity_loss += (sparsity_loss.item() - running_attention_sparsity_loss) / (batch_index + 1)

            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # update bar
            train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                  epoch=epoch_index, gen_loss=running_general_loss, at_energy_loss=running_attention_energy_loss,
                                  sparsity_loss=running_attention_sparsity_loss)

            train_bar.update()
            #torch.save(model.state_dict(), "data/trained_models/15/test.pth")
        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        # Iterate over val dataset

        # setup: batch generator, set loss and acc to 0; set eval mode on
        dataset.set_split('val')
        batch_generator = generate_nmt_batches(dataset, 
                                               batch_size=args.batch_size, 
                                               device=args.device)
        
        with open("training_monitor.txt", "a") as f:
            f.write("Training Loss: "+str(running_loss)+" at_energy_loss: "+str(running_attention_energy_loss)+" sparsity_loss: "+str(running_attention_sparsity_loss))
            f.write("\n")

        running_loss = 0.
        running_acc = 0.
        model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred, _ = model(batch_dict['x_source'], 
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'],
                           sample_probability=sample_probability)

            # step 3. compute the loss
            loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

            # compute the running loss and accuracy
            running_loss += (loss.item() - running_loss) / (batch_index + 1)
            
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            
            # Update bar
            val_bar.set_postfix(loss=running_loss, acc=running_acc, 
                            epoch=epoch_index)
            val_bar.update()

        train_state['val_loss'].append(running_loss)
        print("Current validation loss is: {}".format(running_loss))
        train_state['val_acc'].append(running_acc)
        print("Current validation accuracy is: {}".format(running_acc))

        with open("training_monitor.txt", "a") as f:
            f.write("Validation Loss: "+str(running_loss)+" Validation Accuracy: "+str(running_acc))
            f.write("\n")
            f.write("----------------------")
            f.write("\n")

        train_state = update_train_state(args=args, model=model, 
                                         train_state=train_state)

        scheduler.step(train_state['val_loss'][-1])

        if train_state['stop_early']:
            break
        
        train_bar.n = 0
        val_bar.n = 0
        epoch_bar.set_postfix(best_val=train_state['early_stopping_best_val'])
        epoch_bar.update()
        
except KeyboardInterrupt:
    print("Exiting loop")
