import os
import os.path as path
import json
import torch
import torch.utils.data as data
import numpy as np
import random
from PIL import Image
import pdb
import csv
import dataset.mnist_read
#torch.multiprocessing.set_sharing_strategy('file_system')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def gray_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('P')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class omniglotData(object):
    """
       Dataloader for omniglot dataset.
    """

    def __init__(self, data_dir="./dataset/omniglot/data", mode="train", image_size=28,
                 transform=None, loader=default_loader, gray_loader=gray_loader, 
                 episode_num=1000, way_num=5, shot_num=5, query_num=5):
        
        super(omniglotData, self).__init__()

        # get all the classes
        classes_list = []
        alphabet_names = [alphabet for alphabet in os.listdir(data_dir) if path.isdir(path.join(data_dir, alphabet)) ]
        for alphabet in alphabet_names:
            alphabet_dir = path.join(data_dir, alphabet)
            for character in os.listdir(alphabet_dir):
                classes_list.append(path.join(data_dir, alphabet, character))


        # divide the train/test set
        random.seed(2000) # set the same seed for each training and testing
        train_list = random.sample(classes_list, 1200)
        test_list = [val for val in classes_list if val not in train_list]


        data_list = []
        e = 0
        if mode == "train":

            while e < episode_num:

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(train_list, way_num)
                label_num = -1 

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(item)
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]
                    if query_num < len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(item, i) for i in query_imgs]
                    support_dir = [path.join(item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
            
        else:
            while e < episode_num:

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(test_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(item)
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]
                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(item, i) for i in query_imgs]
                    support_dir = [path.join(item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
        self.data_list = data_list
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.gray_loader = gray_loader


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query           
        '''
        image_size = self.image_size
        episode_files = self.data_list[index]

        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        for i in range(len(episode_files)):
            data_files = episode_files[i]

            # load query images
            query_dir = data_files['query_img']

            for j in range(len(query_dir)):
                temp_img = self.loader(query_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                query_images.append(temp_img)


            # load support images
            temp_support = []
            support_dir = data_files['support_set']
            for j in range(len(support_dir)): 
                temp_img = self.loader(support_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                temp_support.append(temp_img)

            support_images.append(temp_support)

            # read the label
            target = data_files['target']
            query_targets.extend(np.tile(target, len(query_dir)))
            support_targets.extend(np.tile(target, len(support_dir)))

        # Shuffle the query images 
        random.Random(4).shuffle(query_images)
        random.Random(4).shuffle(query_targets)           
        return (query_images, query_targets, support_images, support_targets)
        



class mnistData(object):
    """
       Dataloader for mnist dataset.
    """

    def __init__(self, data_dir="./dataset/mnist_raw", mode="train", image_size=28,
                 transform=None, loader=default_loader, gray_loader=gray_loader, 
                 episode_num=1000, way_num=10, shot_num=5, query_num=5):
        
        super(mnistData, self).__init__()

        # decode and read mnist dataset
        train_data = mnist_read.loadImageSet(data_dir, which=0)
        train_labels = mnist_read.loadLabelSet(data_dir, which=0)

        test_data = mnist_read.loadImageSet(data_dir, which=1)
        test_labels = mnist_read.loadLabelSet(data_dir, which=1)


        # construct eposide data according to Few-shot setting
        data_list = []
        e = 0
        if mode == "train":

            while e < episode_num:

                # construct each episode
                episode = []
                e += 1
                uni_label = np.unique(train_labels)

                for item_label in np.nditer(uni_label):
                    item_indexs = np.where(train_labels==item_label) # find the index
                    item_indexs = item_indexs[0]
                    if len(item_indexs) > shot_num+query_num:
                        support_indexs = np.random.choice(item_indexs, shot_num, replace=False)  # sampling shot samples
                        query_indexs = list(set(item_indexs)-set(support_indexs)) 
                        query_indexs = random.sample(query_indexs, query_num)

                        # store the indexs of the images
                        data_files = {
                            "query_img": query_indexs,
                            "support_set": list(support_indexs),
                            "target": item_label
                        }
                        episode.append(data_files)
                data_list.append(episode) 
            
        else:

            # the difference is that the query images come from test set

            # sample all the support set
            support_all_indexs = []
            uni_label = np.unique(train_labels)
            for item_label in np.nditer(uni_label):
                item_indexs = np.where(train_labels==item_label) # find the index
                item_indexs = item_indexs[0]
                if len(item_indexs) > shot_num+query_num:
                    support_indexs = np.random.choice(item_indexs, shot_num, replace=False)  # sampling shot samples
                    support_all_indexs.append(list(support_indexs))

            # sample query images 
            for ii in range(0, len(test_labels)-query_num, query_num):
                query_indexs = list(range(ii, query_num+ii))
                query_labels = test_labels[query_indexs]

                # store the indexs of the images
                data_files = {
                    "query_img": query_indexs,
                    "support_set": support_all_indexs,
                    "target": query_labels
                }
                data_list.append(data_files)
            
        self.data_list = data_list
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.gray_loader = gray_loader
        self.mode = mode


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query           
        '''
        image_size = self.image_size
        mode = self.mode
        train_data = self.train_data
        train_labels = self.train_labels
        test_data = self.test_data
        test_labels = self.test_labels


        query_images = []
        query_targets = []
        support_images = []
        support_targets = []


        if mode == "train":
            episode_files = self.data_list[index]
            for i in range(len(episode_files)):
                data_files = episode_files[i]

                # load query images
                query_indexs = data_files['query_img']
            
                for j in range(len(query_indexs)):
                    temp_img = train_data[query_indexs[j], :]

                    #temp_img = np.stack((temp_img, temp_img, temp_img))
                    temp_img = Image.fromarray(np.uint8(temp_img))
                    temp_img = temp_img.convert('RGB')

                    # Normalization
                    if self.transform is not None:
                        temp_img = self.transform(temp_img)
                    query_images.append(temp_img)


                # load support images
                temp_support = []
                support_indexs = data_files['support_set']
                for j in range(len(support_indexs)): 
                    temp_img = train_data[support_indexs[j], :]
                    #temp_img = np.stack((temp_img, temp_img, temp_img))
                    temp_img = Image.fromarray(np.uint8(temp_img))
                    temp_img = temp_img.convert('RGB')

                    # Normalization
                    if self.transform is not None:
                        temp_img = self.transform(temp_img)
                    temp_support.append(temp_img)

                support_images.append(temp_support)

                # read the label
                target = data_files['target']
                query_targets.extend(np.tile(target, len(query_indexs)))
                support_targets.extend(np.tile(target, len(support_indexs)))


            # Shuffle the query images 
            random.Random(4).shuffle(query_images)
            random.Random(4).shuffle(query_targets)           
            return (query_images, query_targets, support_images, support_targets)

        else:

            data_files = self.data_list[index]

            # load query images
            query_indexs = data_files['query_img']
            
            for j in range(len(query_indexs)):
                temp_img = train_data[query_indexs[j], :]
                #temp_img = np.stack((temp_img, temp_img, temp_img))
                temp_img = Image.fromarray(np.uint8(temp_img))
                temp_img = temp_img.convert('RGB')

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                query_images.append(temp_img)


            # load support images
            support_all_indexs = data_files['support_set']
            for j in range(len(support_all_indexs)):
                support_indexs = support_all_indexs[j]
                temp_support = []
                for k in range(len(support_indexs)): 
                    temp_img = train_data[support_indexs[k], :]
                    #temp_img = np.stack((temp_img, temp_img, temp_img))
                    temp_img = Image.fromarray(np.uint8(temp_img))
                    temp_img = temp_img.convert('RGB')

                    # Normalization
                    if self.transform is not None:
                        temp_img = self.transform(temp_img)
                    temp_support.append(temp_img)
            
                support_targets.extend(np.tile(j, len(support_indexs)))
                support_images.append(temp_support)

            # read the label
            query_targets = data_files['target']
           

            # Shuffle the query images 
            random.Random(4).shuffle(query_images)
            random.Random(4).shuffle(query_targets)           
            return (query_images, query_targets, support_images, support_targets)


        

class stanford_Dog(object):
    """
       Dataloader for Stanford Dogs dataset.
       Total classes: 120 
       Train classes: 70
       Val classes:   20
       Test classes:  30
    """

    def __init__(self, data_dir="/wenbin/Datasets/Stanford_dogs/Images", mode="train", image_size=224,
                 transform=None, loader=default_loader, gray_loader=gray_loader, 
                 episode_num=1000, way_num=5, shot_num=5, query_num=5):
        
        super(stanford_Dog, self).__init__()

        # get all the dog classes
        classes_list = [dog_name for dog_name in os.listdir(data_dir) if path.isdir(path.join(data_dir, dog_name)) ]


        # divide the train/val/test set
        random.seed(120) # set the same seed for each training and testing
        train_list = random.sample(classes_list, 70)
        remain_list = [rem for rem in classes_list if rem not in train_list]
        val_list = random.sample(remain_list, 20)
        test_list = [rem for rem in remain_list if rem not in val_list]


        data_list = []
        e = 0
        if mode == "train":

            while e < episode_num:

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(train_list, way_num)
                label_num = -1 

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num < len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
            
        elif mode == "val":
            while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(val_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
        else:
             while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(test_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode) 


        self.data_list = data_list
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.gray_loader = gray_loader


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query           
        '''
        image_size = self.image_size
        episode_files = self.data_list[index]

        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        for i in range(len(episode_files)):
            data_files = episode_files[i]

            # load query images
            query_dir = data_files['query_img']

            for j in range(len(query_dir)):
                temp_img = self.loader(query_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                query_images.append(temp_img)


            # load support images
            temp_support = []
            support_dir = data_files['support_set']
            for j in range(len(support_dir)): 
                temp_img = self.loader(support_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                temp_support.append(temp_img)

            support_images.append(temp_support)

            # read the label
            target = data_files['target']
            query_targets.extend(np.tile(target, len(query_dir)))
            support_targets.extend(np.tile(target, len(support_dir)))

        # Shuffle the query images 
        random.Random(4).shuffle(query_images)
        random.Random(4).shuffle(query_targets)           
        return (query_images, query_targets, support_images, support_targets)




class CUB_Bird(object):
    """
       Dataloader for CUB Birds dataset.
       Total classes: 200 
       Train classes: 130
       Val classes:   20
       Test classes:  50
    """

    def __init__(self, data_dir="/wenbin/Datasets/CUB_birds/images", mode="train", image_size=224,
                 transform=None, loader=default_loader, gray_loader=gray_loader, 
                 episode_num=1000, way_num=5, shot_num=5, query_num=5):
        
        super(CUB_Bird, self).__init__()

        # get all the bird classes
        classes_list = [bird_name for bird_name in os.listdir(data_dir) if path.isdir(path.join(data_dir, bird_name)) ]


        # divide the train/val/test set
        random.seed(200) # set the same seed for each training and testing
        train_list = random.sample(classes_list, 130)
        remain_list = [rem for rem in classes_list if rem not in train_list]
        val_list = random.sample(remain_list, 20)
        test_list = [rem for rem in remain_list if rem not in val_list]


        data_list = []
        e = 0
        if mode == "train":

            while e < episode_num:

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(train_list, way_num)
                label_num = -1 

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num < len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
            
        elif mode == "val":
            while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(val_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
        else:
             while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(test_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode) 


        self.data_list = data_list
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.gray_loader = gray_loader


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query           
        '''
        image_size = self.image_size
        episode_files = self.data_list[index]

        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        for i in range(len(episode_files)):
            data_files = episode_files[i]

            # load query images
            query_dir = data_files['query_img']

            for j in range(len(query_dir)):
                temp_img = self.loader(query_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                query_images.append(temp_img)


            # load support images
            temp_support = []
            support_dir = data_files['support_set']
            for j in range(len(support_dir)): 
                temp_img = self.loader(support_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                temp_support.append(temp_img)

            support_images.append(temp_support)

            # read the label
            target = data_files['target']
            query_targets.extend(np.tile(target, len(query_dir)))
            support_targets.extend(np.tile(target, len(support_dir)))

        # Shuffle the query images 
        random.Random(4).shuffle(query_images)
        random.Random(4).shuffle(query_targets)           
        return (query_images, query_targets, support_images, support_targets)




class stanford_Car(object):
    """
       Dataloader for Stanford Cars dataset.
       Total classes: 196 
       Train classes: 130
       Val classes:   17
       Test classes:  49
    """

    def __init__(self, data_dir="/wenbin/Datasets/stanford_cars/images", mode="train", image_size=224,
                 transform=None, loader=default_loader, gray_loader=gray_loader, 
                 episode_num=1000, way_num=5, shot_num=5, query_num=5):
        
        super(stanford_Car, self).__init__()

        # get all the bird classes
        classes_list = [car_name for car_name in os.listdir(data_dir) if path.isdir(path.join(data_dir, car_name)) ]


        # divide the train/val/test set
        random.seed(200) # set the same seed for each training and testing
        train_list = random.sample(classes_list, 130)
        remain_list = [rem for rem in classes_list if rem not in train_list]
        val_list = random.sample(remain_list, 17)
        test_list = [rem for rem in remain_list if rem not in val_list]



        data_list = []
        e = 0
        if mode == "train":

            while e < episode_num:

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(train_list, way_num)
                label_num = -1 

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num < len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
            
        elif mode == "val":
            while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(val_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
        else:
             while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(test_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode) 


        self.data_list = data_list
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.gray_loader = gray_loader


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query           
        '''
        image_size = self.image_size
        episode_files = self.data_list[index]

        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        for i in range(len(episode_files)):
            data_files = episode_files[i]

            # load query images
            query_dir = data_files['query_img']

            for j in range(len(query_dir)):
                temp_img = self.loader(query_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                query_images.append(temp_img)


            # load support images
            temp_support = []
            support_dir = data_files['support_set']
            for j in range(len(support_dir)): 
                temp_img = self.loader(support_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                temp_support.append(temp_img)

            support_images.append(temp_support)

            # read the label
            target = data_files['target']
            query_targets.extend(np.tile(target, len(query_dir)))
            support_targets.extend(np.tile(target, len(support_dir)))

        # Shuffle the query images 
        random.Random(4).shuffle(query_images)
        random.Random(4).shuffle(query_targets)           
        return (query_images, query_targets, support_images, support_targets)




class miniImageNet(object):
    """
       Dataloader for miniImageNet dataset.
       Total classes: 100
       Train classes: 64
       Val classes:   16
       Test classes:  20
    """

    def __init__(self, data_dir="/wenbin/Datasets/miniImageNet/images", mode="train", image_size=84,
                 transform=None, loader=default_loader, gray_loader=gray_loader, 
                 episode_num=1000, way_num=5, shot_num=5, query_num=5):
        
        super(miniImageNet, self).__init__()

        # get all the classes
        classes_list = [mini_name for mini_name in os.listdir(data_dir) if path.isdir(path.join(data_dir, mini_name)) ]


        # divide the train/val/test set
        random.seed(100) # set the same seed for each training and testing
        train_list = random.sample(classes_list, 64)
        remain_list = [rem for rem in classes_list if rem not in train_list]
        val_list = random.sample(remain_list, 16)
        test_list = [rem for rem in remain_list if rem not in val_list]


        data_list = []
        e = 0
        if mode == "train":

            while e < episode_num:

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(train_list, way_num)
                label_num = -1 

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num < len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
            
        elif mode == "val":
            while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(val_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
        else:
             while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                random.seed()
                temp_list = random.sample(test_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = os.listdir(os.path.join(data_dir, item))
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, item, i) for i in query_imgs]
                    support_dir = [path.join(data_dir, item, i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode) 


        self.data_list = data_list
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.gray_loader = gray_loader


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query           
        '''
        image_size = self.image_size
        episode_files = self.data_list[index]

        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        for i in range(len(episode_files)):
            data_files = episode_files[i]

            # load query images
            query_dir = data_files['query_img']

            for j in range(len(query_dir)):
                temp_img = self.loader(query_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                query_images.append(temp_img)


            # load support images
            temp_support = []
            support_dir = data_files['support_set']
            for j in range(len(support_dir)): 
                temp_img = self.loader(support_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                temp_support.append(temp_img)

            support_images.append(temp_support)

            # read the label
            target = data_files['target']
            query_targets.extend(np.tile(target, len(query_dir)))
            support_targets.extend(np.tile(target, len(support_dir)))

        # Shuffle the query images 
        random.Random(4).shuffle(query_images)
        random.Random(4).shuffle(query_targets)           
        return (query_images, query_targets, support_images, support_targets)



class miniImageNet_ravi(object):
    """
       Dataloader for miniImageNet--ravi dataset.
       Total classes: 100
       Train classes: 64
       Val classes:   16
       Test classes:  20
    """

    def __init__(self, data_dir="/wenbin/Datasets/miniImageNet--ravi", mode="train", image_size=84,
                 transform=None, loader=default_loader, gray_loader=gray_loader, 
                 episode_num=1000, way_num=5, shot_num=5, query_num=5):
        
        super(miniImageNet_ravi, self).__init__()

    
        # set the paths of the csv files
        train_csv = os.path.join(data_dir, 'train.csv')
        val_csv = os.path.join(data_dir, 'val.csv')
        test_csv = os.path.join(data_dir, 'test.csv')


        data_list = []
        e = 0
        if mode == "train":

            # store all the classes and images into a dict
            class_img_dict = {}
            with open(train_csv) as f_csv:
                f_train = csv.reader(f_csv, delimiter=',')
                for row in f_train:
                    if f_train.line_num == 1:
                        continue
                    img_name, img_class = row

                    if img_class in class_img_dict:
                        class_img_dict[img_class].append(img_name)
                    else:
                        class_img_dict[img_class]=[]
                        class_img_dict[img_class].append(img_name)

            class_list = class_img_dict.keys()


            while e < episode_num:

                # construct each episode
                episode = []
                e += 1
                random.seed(15)
                temp_list = random.sample(class_list, way_num)
                label_num = -1 

                for item in temp_list:
                    label_num += 1
                    imgs_set = class_img_dict[item]
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num < len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, 'images', i) for i in query_imgs]
                    support_dir = [path.join(data_dir, 'images', i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)

            
        elif mode == "val":

            # store all the classes and images into a dict
            class_img_dict = {}
            with open(val_csv) as f_csv:
                f_val = csv.reader(f_csv, delimiter=',')
                for row in f_val:
                    if f_val.line_num == 1:
                        continue
                    img_name, img_class = row

                    if img_class in class_img_dict:
                        class_img_dict[img_class].append(img_name)
                    else:
                        class_img_dict[img_class]=[]
                        class_img_dict[img_class].append(img_name)

            class_list = class_img_dict.keys()



            while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                random.seed(15)
                temp_list = random.sample(class_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = class_img_dict[item]
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, 'images', i) for i in query_imgs]
                    support_dir = [path.join(data_dir, 'images', i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode)
        else:

            # store all the classes and images into a dict
            class_img_dict = {}
            with open(test_csv) as f_csv:
                f_test = csv.reader(f_csv, delimiter=',')
                for row in f_test:
                    if f_test.line_num == 1:
                        continue
                    img_name, img_class = row

                    if img_class in class_img_dict:
                        class_img_dict[img_class].append(img_name)
                    else:
                        class_img_dict[img_class]=[]
                        class_img_dict[img_class].append(img_name)

            class_list = class_img_dict.keys()


            while e < episode_num:   # setting the episode number to 600

                # construct each episode
                episode = []
                e += 1
                random.seed(15)
                temp_list = random.sample(class_list, way_num)
                label_num = -1

                for item in temp_list:
                    label_num += 1
                    imgs_set = class_img_dict[item]
                    support_imgs = random.sample(imgs_set, shot_num)
                    query_imgs = [val for val in imgs_set if val not in support_imgs]

                    if query_num<len(query_imgs):
                        query_imgs = random.sample(query_imgs, query_num)


                    # the dir of support set
                    query_dir = [path.join(data_dir, 'images', i) for i in query_imgs]
                    support_dir = [path.join(data_dir, 'images', i) for i in support_imgs]


                    data_files = {
                        "query_img": query_dir,
                        "support_set": support_dir,
                        "target": label_num
                    }
                    episode.append(data_files)
                data_list.append(episode) 


        self.data_list = data_list
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.gray_loader = gray_loader


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query           
        '''
        image_size = self.image_size
        episode_files = self.data_list[index]

        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        for i in range(len(episode_files)):
            data_files = episode_files[i]

            # load query images
            query_dir = data_files['query_img']

            for j in range(len(query_dir)):
                temp_img = self.loader(query_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                query_images.append(temp_img)


            # load support images
            temp_support = []
            support_dir = data_files['support_set']
            for j in range(len(support_dir)): 
                temp_img = self.loader(support_dir[j])

                # Normalization
                if self.transform is not None:
                    temp_img = self.transform(temp_img)
                temp_support.append(temp_img)

            support_images.append(temp_support)

            # read the label
            target = data_files['target']
            query_targets.extend(np.tile(target, len(query_dir)))
            support_targets.extend(np.tile(target, len(support_dir)))

        # Shuffle the query images 
        rand_num = random.randint(0,9)
        random.Random(rand_num).shuffle(query_images)
        random.Random(rand_num).shuffle(query_targets)           
        return (query_images, query_targets, support_images, support_targets)
