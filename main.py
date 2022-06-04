import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from Network import *
from captcha_dataset import CAPTCHADataset, char2idx, idx2char, file_list_train, file_list_test
import random

""" Device Confirmation """
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using PyTorch version: ', torch.__version__, ' Device: ', device)

"""Paramenter generalization"""
batch_size = 64
num_epochs = 100
lr = 0.001
weight_decay = 1e-3
clip_norm = 5

CHAR_NUM=5

"""Data preprocessing"""
trainset = CAPTCHADataset(test=False)
testset = CAPTCHADataset(test=True)

unlabeled_idx = np.nonzero(trainset.unlabeled_mask)[0]
pool_idx = random.sample(range(1, len(unlabeled_idx)), 10000)

train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4, sampler=SubsetRandomSampler(unlabeled_idx[pool_idx]))
test_loader = DataLoader(testset, batch_size=batch_size, num_workers=0)

"""Confirm data"""
print("len(train_loader), len(test_loader): ", len(train_loader), len(test_loader))

image, label,idx= iter(train_loader).__next__()
print("image.size(), label, idx: ", image.size(), label, idx)

num_chars = len(char2idx)
print("num_chars: ", num_chars)
rnn_hidden_size = 256

"""Confirm optimizer, objective function"""

crnn = CRNN(num_chars, rnn_hidden_size=rnn_hidden_size)
crnn.apply(weights_init)
crnn = crnn.to(device)
optimizer = optim.Adam(crnn.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)
"""
label_logits = crnn(image.to(device))  # torch.Size([T, batch_size, num_classes])
print("label: ", label)
print("label_logits.shape: ", label_logits.shape)  # torch.Size([T, batch_size, num_classes])
"""
# If output is "aaaabbbbbccddeeff", it can be summarized as "abcedf"
criterion = nn.CTCLoss(blank=0)

train_epoch_loss = []
train_epoch_acc = []
num_updates_epochs = []
test_accuracy_list = []
test_loss_list = []

"""Encoding label to tensor"""


def encode_label(label):
    label_targets_lens = [len(text) for text in label]
    # len(label_targets_lens): 64, label_targets_lens: [5,5,...,5]
    label_targets_lens = torch.IntTensor(label_targets_lens)
    # print(label_targets_lens) # tensor([5, 5,...,5], dtype=torch.int32)
    label_concat = "".join(label)
    #print(label_concat) # v42r81I8r2226378o...
    label_targets = [char2idx[c] for c in label_concat]
    # print(label_targets) # [58, 5, 3, 54, 9, 2,...,10, 7, 2]
    label_targets = torch.IntTensor(label_targets)
    # print(label_targets) # tensor([58, 5, 3,..., 10,  7,  2], dtype=torch.int32)

    return label_targets, label_targets_lens


"""Compute loss"""


def compute_loss(label, label_logits):
    label_logps = F.log_softmax(label_logits, 2)  # [T, batch_size, num_classes], num_classes computation
    label_logps_lens = torch.full(size=(label_logps.size(1),),  # batch_size
                                  fill_value=label_logps.size(0),  # num of char,T
                                  dtype=torch.int32).to(device)  # [batch_size]
    label_targets, label_targets_lens = encode_label(label)
    loss = criterion(label_logps, label_targets, label_logps_lens, label_targets_lens)
    # CTCLoss([T, batch_size, num_classes],[T * batch_size],[batch_size],[batch_size])

    return loss


"""Compute loss"""
#compute_loss(label, label_logits)

"""decode prediction labels to text"""


def decode_predictions(label_logits):
    label_tokens = F.softmax(label_logits, 2).argmax(2)  # [T, batch_size], softmax for num_chars
    #print(label_tokens)
    # print(F.softmax(label_logits, 2).size()) #torch.Size([T, batch_size, num_chars])
    # print("F.softmax(label_logits, 2).argmax(0)", F.softmax(label_logits, 2).argmax(0)) #[batch_size,num_chars], max of T
    # print("F.softmax(label_logits, 2).argmax(1)",F.softmax(label_logits, 2).argmax(1)) #[T,num_chars], max of batch_size
    # print("F.softmax(label_logits, 2).argmax(2)", F.softmax(label_logits, 2).argmax(2)) #[T,batch_size], max of num_chars
    # print(label_tokens.size()) # label_tokens.size():  torch.Size([T, batch_size])
    label_tokens = label_tokens.numpy().T  # [batch_size, T], transpose matrix

    # print("label_tokens: ", label_tokens) # [batch_size, T]
    # decode idx to char
    label_tokens_new = []
    #print("label_tokens: ",label_tokens)
    for text_tokens in label_tokens:
        #print("text_tokens: ",text_tokens)
        text = [idx2char[idx] for idx in text_tokens]
        text = "".join(text)
        #print("text: ",text)
        label_tokens_new.append(text)

    #print("label_tokens_new: ", label_tokens_new)
    return label_tokens_new


"""Compare prediction and ground truth to gain accuracy"""


def compare_label(label, label_pred):
    correct = 0
    for i in range(len(label_pred)):
        if label[i] == label_pred[i]:
            correct += 1

    return correct, len(label_pred)

def margin_query(model, device, data_loader, query_size):
    margins = []
    indices = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            data, _, idx = batch
            # print("data, idx: ",len(data), len(idx)) # batch_size batch_size
            logits = model(data.to(device))
            # print("logits: ", logits.size()) #torch.Size([T, batch_size, num_chars])
            probabilities = F.softmax(logits,2)
            # print("probabilities: ", probabilities.size()) # #torch.Size([T, batch_size, num_chars])
            # Select the top two class confidences for each sample
            toptwo = torch.topk(probabilities, 2, dim=2)[0]
            # print("toptwo.size(): ",toptwo.size()) # toptwo.size():  torch.Size([T, batch_size, 2])
            #print("toptwo: ",toptwo)

            # Compute the margins = differences between the two top confidences
            differences = toptwo[:,:, 0] - toptwo[:,:, 1]
            # print("differences.size(): ",differences.size()) # [T, batch_size]
            # print("differences[0].size(): ", list(differences[0].size()).__getitem__(0)) # batch_size
            for i in range(list(differences[0].size()).__getitem__(0)):
                sum=0
                for j in range(5):
                    #print("differences[j][i].item(): ",differences[j][i].item())
                    sum+=differences[j][i].item()
                #print("sum: ",sum)
                margins.append(sum)
            indices.extend(idx.tolist())
            #print("list(idx): ", list(idx))

    margin = np.asarray(margins)
    #print("margins: ",len(margin))
    index = np.asarray(indices)
    #print("index: ", index)
    sorted_pool = np.argsort(margin)
    #print("sorted_pool: ", sorted_pool)
    # Return the indices corresponding to the lowest `query_size` margins
    return index[sorted_pool][0:query_size]

def least_confidence_query(model, device, pool_loader, query_size):
    confidences = []
    indices = []

    model.eval()

    with torch.no_grad():
        for batch in pool_loader:
            data, _, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits,2)
            # Keep only the top class confidence for each sample
            most_probable = torch.max(probabilities, dim=2)[0]
            # print("most_probable: ",most_probable.size()) # torch.Size([T, batch_size])
            # print("list(most_probable[0].size()).__getitem__(0): ",list(most_probable[0].size()).__getitem__(0))
            for i in range(list(most_probable[0].size()).__getitem__(0)):
                sum = 0
                for j in range(5):
                    # print("most_probable[j][i].item(): ",most_probable[j][i].item())
                    sum += most_probable[j][i].item()
                # print("sum: ",sum)
                confidences.append(sum)
            indices.extend(idx.tolist())

    conf = np.asarray(confidences)
    #print("conf: ",len(conf)) # query size
    index = np.asarray(indices)
    #print("len(index): ",len(index)) # query size
    sorted_pool = np.argsort(conf)
    # Return the indices corresponding to the lowest `query_size` confidences
    return index[sorted_pool][0:query_size]

def entropy_sampling_query(model, device, pool_loader, query_size):
    entropy_list = []
    indices = []
    model.eval()
    with torch.no_grad():
        for batch in pool_loader:
            data, label, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, 2)
            # print("probabilities: ", probabilities.size()) # #torch.Size([T, batch_size, num_chars])
            # Select the top entropy
            cal_entropy=torch.sum(-probabilities*torch.log(probabilities),2)
            # print("cal_entropy: ",cal_entropy.size()) # torch.Size([T, batch_size])
            mean_entropy=torch.mean(cal_entropy,0)
            # print("mean_entropy: ", mean_entropy.size()) #torch.Size([batch_size])
            for i in range(len(label)):
                mean_entropy_sub=mean_entropy[i]
                entropy_list.append(mean_entropy_sub)
            indices.extend((idx.tolist()))

    entropies = np.asarray(entropy_list)
    #print("entropies: ",len(entropies)) # query_size
    index = np.asarray(indices)
   #print("index: ",len(index)) # query_size
    sorted_pool = np.argsort(entropies)
    # print("sorted_pool: ", sorted_pool)
    # Return the indices corresponding to the lowest `query_size` margins
    return index[sorted_pool[::-1]][0:query_size]

def query_the_oracle(model,device,dataset,query_size,pool_size, query_strategy,interactive=True):
    unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]
    # print("len(unlabeled_idx): ", len(unlabeled_idx))
    # Select a pool of samples to query from
    if len(unlabeled_idx)>pool_size:
        pool_idx = random.sample(range(1, len(unlabeled_idx)), pool_size)
        pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=SubsetRandomSampler(unlabeled_idx[pool_idx]))
    else:
        pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=SubsetRandomSampler(unlabeled_idx))

    if query_strategy=="margin":
        sample_idx = margin_query(model, device, pool_loader, query_size)
        #print("sample_idx: ",sample_idx)

    elif query_strategy == "least_confidence":
        sample_idx = least_confidence_query(model, device, pool_loader, query_size)
    else:
        sample_idx = entropy_sampling_query(model, device, pool_loader, query_size) # entropy

    print("Ask ", len(sample_idx), " label of image")
    # Query the samples, one at a time
    for sample in sample_idx:
        if interactive:
            img_name=dataset.display(sample)
            #print("What is the class of this image?")
            #print(img_name.split(".")[0])
            new_label = img_name.split(".")[0]
            dataset.update_label(sample,new_label)

        else:
            dataset.label_from_filename(sample)

def train(model, device, labeled_loader, optimizer):
    model.train()
    indices = []
    epoch_loss_list = []
    train_acc_list = []
    num_updates_epoch = 0
    for image, label, idx in labeled_loader:
        optimizer.zero_grad()
        label_logits = model(image.to(device))
        label_pred = decode_predictions(label_logits.cpu())
        train_correct, check = compare_label(label, label_pred)
        train_iteration_accuracy = train_correct / check

        loss = compute_loss(label, label_logits)
        iteration_loss = loss.item()
        # If iteration loss is NaN or inf, ignore it
        if np.isnan(iteration_loss) or np.isinf(iteration_loss):
            continue

        num_updates_epoch += 1
        epoch_loss_list.append(iteration_loss)
        train_acc_list.append(train_iteration_accuracy)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        indices.extend((idx.tolist()))
    # Mean the iteration loss to 1 epoch loss
    epoch_loss = np.mean(epoch_loss_list)
    epoch_acc = np.mean(train_acc_list)
    print("Epoch:{}    Train loss:{}    Train acc:{}    NumUpdates:{}".format(epoch + 1, epoch_loss, epoch_acc,
                                                                              num_updates_epoch))
    train_epoch_loss.append(epoch_loss)
    train_epoch_acc.append(epoch_acc)
    num_updates_epochs.append(num_updates_epoch)
    lr_scheduler.step(epoch_loss)
    index = np.asarray(indices)
    return index

def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        test_correct = 0
        test_check = 0
        test_epoch_loss_list = []
        test_acc_iteration_list = []
        for image, label in test_loader:
            # print("image.size(): ",image.size()) # torch.Size([batch_size, 3, 256, 256]), if query size < batch size, batch size switched to query size
            # print(len(label)) # ('0reqn', 'wY45u', '2o4Pj', '7J7wT', 'ZQ6GB',...)
            # print("_.size(): ", _.size()) # torch.Size([batch_size]), if query size < batch size, batch size switched to query size
            test_check += 1
            label_logits = model(image.to(device))  # [width, batch_size, num_classes==num_features]
            # print("label_logits: ",label_logits.size())
            label_pred = decode_predictions(label_logits.cpu())
            # print("label_pred: ", label_pred)
            loss = compute_loss(label, label_logits)
            iteration_loss = loss.item()
            if np.isnan(iteration_loss) or np.isinf(iteration_loss):
                continue
            # print(label, label_pred)
            correct, check = compare_label(label, label_pred)
            # print("label: ",label)
            # print("label_pred: ", label_pred)
            test_correct += correct
            test_check += check
            test_accuracy = test_correct / test_check
            test_epoch_loss_list.append(iteration_loss)
            test_acc_iteration_list.append(test_accuracy)
        test_accuracy = np.mean(test_acc_iteration_list)
        test_loss = np.mean(test_epoch_loss_list)
        print("Epoch:{}    Test loss:{}    Test acc:{}".format(epoch+1, test_loss, test_accuracy))
        test_accuracy_list.append(test_accuracy)
        test_loss_list.append(test_loss)




if __name__ == '__main__':
    query_size = 2000
    pool_size = 5000
    query_strategy = "margin"
    for epoch in range(num_epochs):
        if epoch > 1:
            query_the_oracle(crnn, device, trainset, query_size=query_size, pool_size=pool_size, query_strategy=query_strategy,
                         interactive=True)

            # Train the model on the data that has been labeled so far:
            labeled_idx = np.where(trainset.unlabeled_mask == 0)[0]
            labeled_loader = DataLoader(trainset, batch_size=batch_size, num_workers=0, sampler=SubsetRandomSampler(labeled_idx))
            train(crnn, device, labeled_loader, optimizer)
            test(crnn, device, test_loader)
        else:
            idx=train(crnn, device, train_loader, optimizer)
            for sample in idx:
                    img_name = trainset.display(sample)
                    new_label = img_name.split(".")[0]
                    trainset.update_label(sample, new_label)

            test(crnn, device, test_loader)





    fig = plt.figure(figsize=(12, 20))
    x = np.linspace(1, num_epochs, num_epochs)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, train_epoch_loss, label="Train loss")
    ax1.plot(x, test_loss_list, label="Test loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, train_epoch_acc, label="Train acc")
    ax2.plot(x, test_accuracy_list, label="Test acc")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    print("train_accuracy: ", train_epoch_acc[-1])
    print("test_accuracy: ", test_accuracy_list[-1])

    plt.show()

    #Save model
    torch.save(crnn.state_dict(),"./model_state__dict.pt")

    f=open('./AccLoss.txt','a')
    f.write("num_epochs: " + str(num_epochs) + "\n")
    f.write("query_size: "+str(query_size)+"\n")
    f.write("pool_size: " + str(pool_size)+"\n")
    f.write("query_strategy: " + query_strategy + "\n")
    f.write("train_epoch_loss: ")
    f.write(str(train_epoch_loss)+"\n")
    f.write("test_loss_list: ")
    f.write(str(test_loss_list)+"\n")
    f.write("train_epoch_acc: ")
    f.write(str(train_epoch_acc)+"\n")
    f.write("test_accuracy_list: ")
    f.write(str(test_accuracy_list)+"\n")

    f.close()

    #Take model
    # model=CRNN(num_chars, rnn_hidden_size=rnn_hidden_size)
    # model.load_state_dict(torch.load("./"))
    # model.eval()
