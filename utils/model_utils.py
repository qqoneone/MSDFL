import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import numpy as np
import random
import torch.nn.functional as F

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3

def suffer_data(data):
    data_x = data['x']
    data_y = data['y']
        # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)
    
def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts +1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x,data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)

def read_cifa_100_data_iid():
    mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform= transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    data_loc='/data/wudi/cifar100_data/'
    trainset = torchvision.datasets.CIFAR100(root=data_loc, train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)

    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data


    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 50 # should be muitiple of 10
    NUM_LABELS = 10
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)

    cifa_data = []
    for i in trange(100):
        idx = cifa_data_label==i
        cifa_data.append(cifa_data_image[idx])


    print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []
    samplesPerlabel=100
    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(100, dtype=np.int64)
    #print(sizeof(cifa_data))
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            l = (10*user+j)%100
            X[user] += cifa_data[l][idx[l]:idx[l]+samplesPerlabel].tolist()
            y[user] += (l*np.ones(samplesPerlabel)).tolist()
            idx[l] += samplesPerlabel
        print("IDX1:", idx)

    print("IDX1:", idx)  # counting samples for each labels




    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(num_samples)
        train_data["user_data"][uname] =  {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        


    return train_data['users'],  train_data['user_data']

def read_cifa_data_iid():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_loc='/data/wudi/cifar10_data/'
    trainset = torchvision.datasets.CIFAR10(root=data_loc, train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)

    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data


    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 50 # should be muitiple of 10
    NUM_LABELS = 10
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)

    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label==i
        cifa_data.append(cifa_data_image[idx])


    print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []
    samplesPerlabel=100
    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = j
            X[user] += cifa_data[l][idx[l]:idx[l]+samplesPerlabel].tolist()
            y[user] += (l*np.ones(samplesPerlabel)).tolist()
            idx[l] += samplesPerlabel
        print("IDX1:", idx)

    print("IDX1:", idx)  # counting samples for each labels




    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(num_samples)
        train_data["user_data"][uname] =  {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        


    return train_data['users'],  train_data['user_data']

def read_cifa_data_origin(iid=True):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_loc='/data/wudi/cifar10_data/'
    trainset = torchvision.datasets.CIFAR10(root=data_loc, train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_loc, train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader,0):
        testset.data, testset.targets = train_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 50 # should be muitiple of 10
    if iid== True:
        NUM_LABELS=10
        SamplesPerlabel=120
    else:
        NUM_LABELS=3
        SamplesPerlabel=400
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)

    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label==i
        cifa_data.append(cifa_data_image[idx])


    print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            #print("L:", l)
            X[user] += cifa_data[l][idx[l]:idx[l]+SamplesPerlabel].tolist()
            y[user] += (l*np.ones(SamplesPerlabel)).tolist()
            idx[l] += SamplesPerlabel

    print("IDX1:", idx)  # counting samples for each labels

    '''
    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in cifa_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                print("check len os user:", user, j,
                    "len data", len(X[user]), num_samples)

    print("IDX2:", idx) # counting samples for each labels
    '''

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)

    return train_data['users'], _ , train_data['user_data'], test_data['user_data']

def target_transform(target):
    return int(target)

def read_svhn_data(iid=True):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_loc='/data/wudi/SVHN_data/'
    trainset = torchvision.datasets.SVHN(root=data_loc, split='train',download=True, transform=transform,target_transform=target_transform)
    testset = torchvision.datasets.SVHN(root=data_loc, split='test',download=True, transform=transform,target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    for _,train_data in enumerate(trainloader):
        trainset.data, trainset.targets = train_data
    for _,train_data in enumerate(testloader,0):
        testset.data, testset.targets = train_data
   
    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 50 # should be muitiple of 10
    if iid== True:
        NUM_LABELS=10
        SamplesPerlabel=120
    else:
        NUM_LABELS=3
        SamplesPerlabel=400
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    svhn_data_image = []
    svhn_data_label = []

    svhn_data_image.extend(trainset.data.cpu().detach().numpy())
    svhn_data_image.extend(testset.data.cpu().detach().numpy())
    svhn_data_label.extend(trainset.targets.cpu().detach().numpy())
    svhn_data_label.extend(testset.targets.cpu().detach().numpy())
    svhn_data_image = np.array(svhn_data_image)
    svhn_data_label = np.array(svhn_data_label)


    '''
    svhn_data_image.extend(trainset.data)
    svhn_data_image.extend(testset.data)
    svhn_data_label.extend(trainset.targets)
    svhn_data_label.extend(testset.targets)
    svhn_data_image = np.array(svhn_data_image)
    svhn_data_label = np.array(svhn_data_label)
    '''

    svhn_data = []
    for i in trange(10):
        idx = svhn_data_label==i
        svhn_data.append(svhn_data_image[idx])


    print("\nNumb samples of each label:\n", [len(v) for v in svhn_data])
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            #print("L:", l)
            X[user] += svhn_data[l][idx[l]:idx[l]+SamplesPerlabel].tolist()
            y[user] += (l*np.ones(SamplesPerlabel)).tolist()
            idx[l] += SamplesPerlabel

    print("IDX1:", idx)  # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)

    return train_data['users'], _ , train_data['user_data'], test_data['user_data']

def read_mnist_data(iid=True):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    data_loc='/data/wudi/MNIST_data/'
    trainset = torchvision.datasets.MNIST(root=data_loc, train=True,download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=data_loc, train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader,0):
        testset.data, testset.targets = train_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 50 # should be muitiple of 10
    if iid== True:
        NUM_LABELS=10
        SamplesPerlabel=120
    else:
        NUM_LABELS=2
        SamplesPerlabel=600
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    mnist_data_image = []
    mnist_data_label = []

    mnist_data_image.extend(trainset.data.cpu().detach().numpy())
    mnist_data_image.extend(testset.data.cpu().detach().numpy())
    mnist_data_label.extend(trainset.targets.cpu().detach().numpy())
    mnist_data_label.extend(testset.targets.cpu().detach().numpy())
    mnist_data_image = np.array(mnist_data_image)
    mnist_data_label = np.array(mnist_data_label)

    mnist_data = []
    for i in trange(10):
        idx = mnist_data_label==i
        mnist_data.append(mnist_data_image[idx])


    print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            #print("L:", l)
            X[user] += mnist_data[l][idx[l]:idx[l]+SamplesPerlabel].tolist()
            y[user] += (l*np.ones(SamplesPerlabel)).tolist()
            idx[l] += SamplesPerlabel

    print("IDX1:", idx)  # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)

    return train_data['users'], _ , train_data['user_data'], test_data['user_data']
def read_fmnist_data(iid=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    data_loc='/data/wudi/MNIST_data/'
    trainset = torchvision.datasets.FashionMNIST(root=data_loc, train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=data_loc, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader,0):
        testset.data, testset.targets = train_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 50 # should be muitiple of 10
    if iid== True:
        NUM_LABELS=10
        SamplesPerlabel=120
    else:
        NUM_LABELS=2
        SamplesPerlabel=600
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fmnist_data_image = []
    fmnist_data_label = []

    fmnist_data_image.extend(trainset.data.cpu().detach().numpy())
    fmnist_data_image.extend(testset.data.cpu().detach().numpy())
    fmnist_data_label.extend(trainset.targets.cpu().detach().numpy())
    fmnist_data_label.extend(testset.targets.cpu().detach().numpy())
    fmnist_data_image = np.array(fmnist_data_image)
    fmnist_data_label = np.array(fmnist_data_label)

    fmnist_data = []
    for i in trange(10):
        idx = fmnist_data_label==i
        fmnist_data.append(fmnist_data_image[idx])


    print("\nNumb samples of each label:\n", [len(v) for v in fmnist_data])
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            #print("L:", l)
            X[user] += fmnist_data[l][idx[l]:idx[l]+SamplesPerlabel].tolist()
            y[user] += (l*np.ones(SamplesPerlabel)).tolist()
            idx[l] += SamplesPerlabel

    print("IDX1:", idx)  # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)

    return train_data['users'], _ , train_data['user_data'], test_data['user_data']

def read_data(dataset,iid):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    if(dataset == "SVHN"):
        clients, groups, train_data, test_data= read_svhn_data(iid)
        return clients, groups, train_data, test_data
    if(dataset == "MNIST"):
        clients, groups, train_data, test_data= read_mnist_data(iid)
        return clients, groups, train_data, test_data
    if(dataset == "FashionMNIST"):
        clients, groups, train_data, test_data= read_fmnist_data(iid)
        return clients, groups, train_data, test_data
    if(dataset == "Cifar10"):
        clients, groups, train_data, test_data= read_cifa_data_origin(iid)
        return clients, groups, train_data, test_data
    if(dataset == "Cifar100"):
        clients,train_data= read_cifa_100_data_iid()
        return clients, train_data
    train_data_dir = os.path.join('data',dataset,'data', 'train')
    test_data_dir = os.path.join('data',dataset,'data', 'test')
    print(dataset)
    if dataset=="synthetic" and iid==True:
        train_data_dir="/data/wudi/synthetic/IID/train"
        test_data_dir="/data/wudi/synthetic/IID/test"
    if dataset=="synthetic" and iid==False:
        train_data_dir="/data/wudi/synthetic/NonIID/train"
        test_data_dir="/data/wudi/synthetic/NonIID/test"


    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

def read_user_data(index,data,dataset):
    #if(dataset=="Cifar10" or dataset=="Cifar100"):
    if(dataset=="Cifar100"):
        id = data[0][index]
        train_data = data[1][int(id)]
        X_train, y_train = train_data['x'], train_data['y']
    else:
        id = data[0][index]
        train_data = data[2][id]
        test_data = data[3][id]
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if(dataset == "MNIST"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif(dataset == "Cifar10"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif(dataset == "Cifar100"):  
    #elif(dataset == "Cifar10" or dataset=="Cifar100"):
        X_train, y_train= train_data['x'], train_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return id, train_data

    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data

def LargeMargin(model,targets,outputs):
    dist_norm = 2
    num_classes = 10
    top_k = 1
    one_hot_labels = F.one_hot(targets, num_classes)
    dual_norm = {1: float('inf'), 2: 2, float('inf'): 1}
    norm_fn = lambda x: torch.norm(x, p=dual_norm[dist_norm])
    with torch.no_grad():
        class_prob = F.softmax(outputs, dim=1)
        correct_class_prob = torch.sum(class_prob * one_hot_labels, dim=1, keepdim=True)
        other_class_prob = class_prob * (1. - one_hot_labels)
        if top_k > 1:
            top_k_class_prob, _ = torch.topk(other_class_prob, k=top_k)
        else:
            top_k_class_prob, _ = torch.max(other_class_prob, dim=1, keepdim=True)
    difference_prob = correct_class_prob - top_k_class_prob
    difference_prob.requires_grad_(True)
    difference_prob.backward(gradient=torch.ones_like(difference_prob), retain_graph=True, create_graph=True)
    difference_prob_gradnorm = []
    # print("model",model)
    for param in model.parameters():
        # print("param",type(param.grad))
        # print("param_size", param.grad.shape)
        difference_prob_gradnorm = param.grad.view(-1) if not len(difference_prob_gradnorm) else torch.cat((difference_prob_gradnorm, param.grad.view(-1)))
    distance_to_boundary = difference_prob / norm_fn(difference_prob_gradnorm)
    return distance_to_boundary

class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics_dir = os.path.join('out', self.params['dataset'], 'metrics_{}_{}_{}_{}_{}.json'.format(
            self.params['seed'], self.params['optimizer'], self.params['learning_rate'], self.params['num_epochs'], self.params['mu']))
        #os.mkdir(os.path.join('out', self.params['dataset']))
        if not os.path.exists('out'):
            os.mkdir('out')
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)

