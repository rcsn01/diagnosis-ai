import os
import random


# Set a random seed to ensure that the order of the images is the same every time.
def setup_seed(seed):
     random.seed(seed)


setup_seed(20)

b = 0
dir = './Dataset/'
# The result of os.listdir is a list collection
# Use the sort method of a list to sort. If there are numbers, use numbers to sort.
files = os.listdir(dir)  # Obtain the path of the amplified image folder
files.sort()
# print("files:", files)  #Create txt file for subsequent data storage
train = open('./train.txt', 'w')
test = open('./test.txt', 'w')
a = 0
a1 = 0
while (b < 4):  # 4 is because there are 4 folders looped 4 times
    label = a  # Set the label to be marked
    ss = './Dataset/' + str(files[b]) + '/'  # training pictures
    pics = os.listdir(ss)  # Get the pictures in the sample00_train folder
    i = 1
    train_percent = 0.8  # The proportion of samples in the training set is 0.8 in the training set and 0.2 in the test set.

    num = len(pics)  # Get the total number of samples
    list = range(num)  # get list
    train_num = int(num * train_percent)  # Total number of training
    train_sample = random.sample(list, train_num)  # Randomly select train_num lengths in the list and shuffle them
    test_num = num - train_num  # Get the number of test samples

    for i in list:  # Loop output files
        name = str(dir) + str(files[b]) + '/' + pics[i] + ' ' + str(int(label)) + '\n'  # Get the names of all picture sequences in the current folder
        if i in train_sample:  # Determine whether i is in the training set
            train.write(name)  # If it is, output the image as training text
        else:
            test.write(name)  # The rest is done in the test text
    a = a + 1
    b = b + 1
train.close()  # Be sure to close the file after the operation is completed
test.close()
