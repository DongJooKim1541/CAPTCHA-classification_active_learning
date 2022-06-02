import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np

#data_path = "./Large_captcha_dataset_png_small/"
data_path = "./Large_Captcha_Dataset/"

file_list = os.listdir(data_path)
print("len(file_list): ", 100) # len(file_list)

for idx, file in enumerate(file_list):
    if len(file.split(".")[0]) != 5:
        print("idx, file: ", idx, file)

# dataset random sampling, (file_list,data_size)
sampleList = random.sample(file_list, len(file_list))

file_list_train, file_list_test = train_test_split(sampleList, random_state=0)
print("len(file_list_train), len(file_list_test): ", len(file_list_train), len(file_list_test))

file_split = [file.split(".")[0] for file in sampleList]
file_split = "".join(file_split)
letters = sorted(list(set(list(file_split))))
print("len(letters): ", len(letters))
print("letters: ", letters)

vocabulary = ["-"] + letters
print("len(vocabulary): ", len(vocabulary))
print("vocabulary: ", vocabulary)
# mapping vocab and idx
idx2char = {k: v for k, v in enumerate(vocabulary, start=0)}
print("idx2char: ", idx2char)
char2idx = {v: k for k, v in idx2char.items()}
print("char2idx: ", char2idx)


# define dataloader class
class CAPTCHADataset(Dataset):

    def __init__(self, test):
        self.label = []
        self.data_dir = data_path
        if test:
            print("Test")
            self.status="test"
            self.file_list = file_list_test
            for i in range(len(self.file_list)):
                self.label.append(file_list[i].split(".")[0])
            self.unlabeled_mask=np.zeros(len(self.label))
        else:
            print("Else")
            self.status="else"
            self.file_list = file_list_train
            #print(len(file_list)) #82327
            for i in range(len(self.file_list)):
                self.label.append(file_list[i].split(".")[0])
            #print("self.label: ",self.label)
            self.unlabeled_mask=np.ones(len(self.label))
            #print("len(self.label): ",len(self.label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.status=="else":
            img_name = self.file_list[index]
            #print(img_name)
            #self.img = os.path.join(self.data_dir, self.label)
            self.img = Image.open(data_path+img_name).convert('RGB')
            self.img = self.transform(self.img)

            return self.img, self.label[index], index
        elif self.status=="test":
            #print("getitem test")
            label = self.file_list[index]
            img = os.path.join(self.data_dir, label)
            img = Image.open(img).convert('RGB')
            img = self.transform(img)
            label = label.split(".")[0]
            return img, label

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

    # Display the image [idx] and its filename
    def display(self, idx):
        img_name = self.file_list[idx]
        #print(data_path+img_name)
        #img=data_path+img_name
        #img = Image.open(img).convert('RGB')
        #plt.imshow(img)
        #plt.show()
        return img_name

    # Set the label of image [idx] to 'new_label'
    def update_label(self, idx, new_label):
        self.label[idx] = new_label
        self.unlabeled_mask[idx] = 0
        return

    # Set the label of image [idx] to that read from its filename
    def label_from_filename(self, idx):
        self.label[idx] = self.file_list[idx].split(".")[0]
        self.unlabeled_mask[idx] = 0
        return

