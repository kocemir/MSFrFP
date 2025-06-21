import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
import numpy as np
import os
import pandas as pd
import tarfile
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import cv2


class CIFAR10(Dataset):

    def __init__(self, root, train=True, transform=None):

        self.root=root
        self.train=train
        self.transform=transform

        self.cifar_dataset= torchvision.datasets.CIFAR10(root=self.root, train=self.train, download=True)
 
    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    

class CIFAR100(Dataset):

    def __init__(self, root, train=True, transform=None):

            self.root=root
            self.train=train
            self.transform=transform

            self.cifar_dataset= torchvision.datasets.CIFAR100(root=self.root, train=self.train, download=True)
 
    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class Cub2011(Dataset):

    
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    filename = 'CUB_200_2011.tgz'
    base_folder = 'CUB_200_2011/images'
    
    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
        self.tgz_md5 = '97eceeb196236b17998738112f37df78'
        self.filename = 'CUB_200_2011.tgz'
        self.base_folder = 'CUB_200_2011/images/'
        #self._load_metadata()

        if download:
            self._download()

        if not self._check_integrity():
           raise RuntimeError('Dataset not found or corrupted.' +
                               'You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')


        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True


    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to
        img = self.loader(path)
        #convert_tensor = transforms.ToTensor()

        #img=convert_tensor(img)


        if self.transform is not None:
            img = self.transform(img)

        return img, target


class Caltech101(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data



def cub2011_data_loader(data_dir,batch_size=128,train_split=0.9):
   
    image_size=224


    train_transform = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
    test_transform = [
            transforms.Resize(int(image_size/0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
  
   
    # Data
    print('==> Preparing data..')

    transform_train =transforms.Compose(train_transform)
    transform_test =transforms.Compose(test_transform)


    train_dataset = Cub2011(
        root=data_dir, train=True, download=True, transform=transform_train)

    
    # Calculate the sizes for training and validation sets
    num_samples = len(train_dataset)
    train_size = int(train_split * num_samples)
    val_size = num_samples - train_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoader instances for training and validation sets
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    testset = Cub2011(
        root=data_dir, train=False, download=True, transform=transform_test)

    

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size)#, num_workers=2)

    return trainloader,validloader,testloader




def cifar10_data_loader(data_dir,batch_size=128,train_split=0.9):
    

    #Previously calculated
    d_mean= [0.49139968,0.48215841,0.44653091]
    d_std= [0.24703223, 0.24348513 ,0.26158784]
    normalize = transforms.Normalize(d_mean,d_std)

      # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),normalize])

    transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),normalize])


    train_dataset= CIFAR10(data_dir,transform=transform_train)

    
    # Calculate the sizes for training and validation sets
    num_samples = len(train_dataset)
    train_size = int(train_split * num_samples)
    val_size = num_samples - train_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoader instances for training and validation sets
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    

    testset = CIFAR10(data_dir,train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)#, num_workers=2)

    return trainloader,validloader,testloader


def cifar100_data_loader(data_dir,batch_size=128,train_split=0.9):
    

    #Previously calculated
    d_mean= [0.49139968,0.48215841,0.44653091]
    d_std= [0.24703223, 0.24348513 ,0.26158784]
    normalize = transforms.Normalize(d_mean,d_std)

      # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),normalize])

    transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),normalize])


    train_dataset= CIFAR100(data_dir,transform=transform_train)

    
    # Calculate the sizes for training and validation sets
    num_samples = len(train_dataset)
    train_size = int(train_split * num_samples)
    val_size = num_samples - train_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoader instances for training and validation sets
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    

    testset = CIFAR100(data_dir,train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)#, num_workers=2)

    return trainloader,validloader,testloader


def caltech101_data_loader(data_dir,batch_size=128,train_split=0.9):
    
    image_paths = list(paths.list_images('/auto/k2/aykut3/emirhan_frft/caltech101_data/101_ObjectCategories'))


    print('==> Preparing data..')

    data = []
    labels = []
    label_names = []
    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]
        if label == 'BACKGROUND_Google':
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data.append(image)
        label_names.append(label)
        labels.append(label)


    data = np.array(data, dtype=object)
    labels = np.array(labels, dtype=object)

    # one hot encode
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    count_arr = []
    label_arr = []
    for i in range(len(lb.classes_)):
        count = 0
        # print(lb.classes_[i])
        for j in range(len(label_names)):
            if lb.classes_[i] in label_names[j]:
                count += 1
        count_arr.append(count)
        label_arr.append(lb.classes_[i])


    (X, x_val , Y, y_val) = train_test_split(data, labels,
                                                        test_size=0.1,
                                                        stratify=labels,
                                                        random_state=42)

    (x_train, x_test, y_train, y_test) = train_test_split(X, Y,
                                                        test_size=0.15,
                                                        random_state=42)




    train_transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        #  transforms.RandomRotation((-30, 30)),
        #  transforms.RandomHorizontalFlip(p=0.5),
        #  transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    val_transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])


    train_data = Caltech101(x_train, y_train, train_transform)
    val_data = Caltech101(x_val, y_val, val_transform)
    test_data = Caltech101(x_test, y_test, val_transform)



    # dataloaders
    trainloader = DataLoader(train_data, batch_size, shuffle=True)
    validloader = DataLoader(val_data, batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size, shuffle=False)

    return trainloader,validloader,testloader


