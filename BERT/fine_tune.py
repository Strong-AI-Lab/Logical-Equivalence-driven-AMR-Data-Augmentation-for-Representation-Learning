import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
import random
import os
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from bert_config import opt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
# Set logger to avoid warning `token indices sequence length is longer than the specified maximum sequence length for this model (1017 > 512)`
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


def text_to_id(tokenizer, text_list):
    """
    It is a function to transform text to id.
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start. Append the `[SEP]` token to the end.
    #   (3) Map tokens to their IDs.
    #   (4) padding and truncation
    #   (5) attention mask
    #
    # Encoding for text in training dataset
    """
    ids_list = []
    attention_masks_list = []
    for item in text_list:
        sentences = item.split('\t\t\t')
        encoded_dict = tokenizer(text=sentences[0],
                                 text_pair=(sentences[1] if len(sentences) == 2 else None),
                                 add_special_tokens=True,
                                 padding='max_length', truncation=True, max_length=512,
                                 return_attention_mask=True)
        ids_list.append(encoded_dict["input_ids"])
        attention_masks_list.append(encoded_dict['attention_mask'])

    return torch.tensor(ids_list), torch.tensor(attention_masks_list)


def load_dataset(data_path):
    cols = ['Rating', 'Stem', 'Answer', 'Distractor0', 'Distractor1', 'Distractor2', 'Distractor3', 'Explanation']
    data_frame = pd.read_csv(data_path, sheet_name="Sheet1", usecols=cols)
    dataset = data_frame.values
    # split into input (x) and output (y) variables; skip index at column 0
    x = dataset[:, 1:]
    y = dataset[:, 0]

    x = x.astype(str)
    x = [" ".join(i) for i in x]

    return x, list(y)


def load_dataset_pair(data_path):

    data_frame = pd.read_csv(data_path)

    dataset = data_frame.values
    # split into input (x) and output (y) variables; skip index at column 0
    x = dataset[:,1:3]
    y = dataset[:,4:5]

    x = x.astype(str)
    x = [" ".join(i) for i in x]
    y = y.astype(int)

    return x, list(y)

def load_dataset_pair_test(data_path):

    data_frame = pd.read_csv(data_path)

    dataset = data_frame.values
    # split into input (x) and output (y) variables; skip index at column 0
    x = dataset[:,0:2]
    y = dataset[:,2:3]

    x = x.astype(str)
    x = [" ".join(i) for i in x]
    y = y.astype(int)

    return x, list(y)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_pretrained_mode_from_name(model):
    if model == "BERT":
        pretrained_model = "bert-base-uncased"
    elif model == "BioBERT":
        pretrained_model = "dmis-lab/biobert-v1.1"
    elif model == "RoBERTa":
        pretrained_model = "roberta-base"
    elif model == "SBERT":
        pretrained_model = "sentence-transformers/paraphrase-distilroberta-base-v1"
    else:
        pretrained_model = "bert-base-uncased"
    return pretrained_model


def train(**args):
    # 0. config
    opt.parse(args)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(opt.seed)
    else:
        device = torch.device('cpu')

    # Set pretrained model
    pretrained_model = get_pretrained_mode_from_name(opt.pretrained_model_name)

    # Load the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=True)

    # 1. Data Pre-processing
    # 1.1 load data
    train_data_file_path = opt.data_dir + opt.train_data_file_name
    validate_data_file_name = opt.data_dir + opt.validate_data_file_name

    if opt.num_labels == 2:
        train_text, train_labels = load_dataset_pair(train_data_file_path)
        val_text, val_labels = load_dataset_pair(validate_data_file_name)
    else:
        train_text, train_labels = load_dataset(train_data_file_path)
        val_text, val_labels = load_dataset(validate_data_file_name)

    # 1.3 labels to tensor
    if opt.num_labels == 2:
        train_labels = torch.LongTensor(train_labels)
        validation_labels = torch.LongTensor(val_labels)
    else:
        train_labels = torch.FloatTensor(train_labels)
        validation_labels = torch.FloatTensor(val_labels)

    # 2. BERT Tokenization & Input Formatting
    # 2.1 Formatting for BERT
    print('Loading ' + opt.pretrained_model_name + ' tokenizer...')
    train_inputs, train_masks = text_to_id(tokenizer, train_text)
    validation_inputs, validation_masks = text_to_id(tokenizer, val_text)

    # 2.3 Form the Dataset
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, shuffle=False, batch_size=opt.batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, shuffle=False, batch_size=opt.batch_size)

    # 3. Train Text Classification Model
    # 3.1 AutoModelForSequenceClassification
    # https://discuss.huggingface.co/t/which-loss-function-in-AutoModelForSequenceClassification-regression/1432/2
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=opt.num_labels,  # The number of output labels -- 1 for MSE Loss Regression; other for classification
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model.to(device)

    # 3.2 Optimizer & Learning Rate Scheduler
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(), lr=opt.lr, eps=opt.eps)

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * opt.epochs
    print("total_steps = {}".format(total_steps))

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # 3.3 Train
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    eval_loss_values = []

    # For each epoch...
    for epoch_i in range(opt.epochs):
        ##########################################
        #               Training                 #
        ##########################################

        # Perform one full pass over the training set.
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, opt.epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 10 batches.
            if step % 10 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear the gradients.
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we have provided the `labels`.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs.loss

            # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end.
            # `loss` is a Tensor containing a single value; the `.item()` function just returns the Python value from the tensor.
            total_loss += loss.item()

            # Perform a `backward` pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        ##########################################
        #               Validation               #
        ##########################################
        # After the completion of each training epoch, measure our performance on our validation set.
        print("Running Validation...")
        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()

        # Tracking variables
        eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to device
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu()
            label_ids = b_labels.to('cpu')
            label_ids = label_ids.squeeze(1)

            # Calculate the mse for this batch of test sentences.
            if opt.num_labels > 1:
                loss_func = torch.nn.CrossEntropyLoss()
            else:
                loss_func = torch.nn.MSELoss()

            loss = loss_func(logits, label_ids)

            # Accumulate the total mse.
            eval_loss += loss

        average_val_loss = eval_loss / len(validation_dataloader)

        eval_loss_values.append(average_val_loss)
        # Report the final accuracy for this validation run.
        print("  MSE: {0:.2f}".format(average_val_loss))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

        saved_epoch_model = opt.saved_model_dir + opt.pretrained_model_name + "/epoch_" + str(epoch_i) + "/"

        if not os.path.exists(saved_epoch_model):
            os.makedirs(saved_epoch_model)

        # Save model to the saved_epoch_model
        model.save_pretrained(saved_epoch_model)
        tokenizer.save_pretrained(saved_epoch_model)

    print("Training complete!")

    # 3.4 Plot
    # Plot the average loss in training
    sns.set(style='darkgrid')  # Use plot styling from seaborn.
    sns.set(font_scale=1.5)  # Increase the plot size and font size.
    plt.rcParams["figure.figsize"] = (12, 6)
    x_values = list(range(opt.epochs))
    # Plot the learning curve.
    plt.plot(x_values, loss_values, 'b-o', label='Training MSE')
    plt.plot(x_values, eval_loss_values, 'rs-', label='Validation MSE')
    plt.legend()
    # Label the plot.
    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    os.makedirs(opt.saved_fig_dir, exist_ok=True)
    plt.savefig(opt.saved_fig_dir + opt.pretrained_model_name + ".jpg")
    plt.show()
    print("Done.")


if __name__ == '__main__':
    train()
