
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import random
from PIL import Image
from ipywidgets import FloatProgress
from IPython.display import display
from __future__ import print_function

from model import ModelDefinition
from dataset import ReadImages, collection
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# In[3]:

def MeanAndStd(imageList, fname=None):
    imageListOpen = ReadImages.openAll(trainset, (225,225))
    m = collection.ComputeMean(imageListOpen)
    print("Mean : ", m)
    s = collection.ComputeStdDev(imageListOpen, m)
    print("std dev : ", s)
    if not fname is None:
        with open(fname, "w") as f:
            f.write('%.3f %.3f %.3f' %m)
            f.write('\n')
            f.write('%.3f %.3f %.3f' %(s))
    return m,s


# In[4]:

def readMeanStd(fname='data/cli.txt'):
    with open(fname) as f:
        mean = map(float, f.readline().split(' '))
        std = map(float, f.readline().split(' '))
    return mean, std


# In[ ]:

def testNet(net, testset, labels, batchSize=32):
    """
        Test the network accuracy on a testset
        Return the number of succes and the number of evaluations done
    """
        net = net.eval() #set the network in eval mode
        correct = 0
        tot = 0
        cpt = 0
        for j in range(len(testset)/batchSize):
            
            #set the inputs
            inp = torch.Tensor(batchSize,3,225,225).cuda()
            for k in range(batchSize):
                inp[k] = testTransform(testset[j*batchSize+k][0])
                cpt += 1
            
            #forward pass
            outputs = net(Variable(inp), volatile=True) #volatile the free memory after the forward
            
            #compute score
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.tolist()
            for k in range(batchSize):
                if (testset[j*batchSize+k][1] in labels):
                    correct += (predicted[k][0] == labels.index(testset[j*batchSize+k][1]))
                    tot += 1
                    
        #handle the rest of the testset
        rest = len(testset)%batchSize
        
        #set inputs
        inp = torch.Tensor(rest,3,225,225).cuda()
        for j in range(rest):
            inp[j] = testTransform(testset[len(testset)-rest+j][0])
        
        #forward
        outputs = mymodel(Variable(inp), volatile=True)
        
        #compute score
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.tolist()
        for j in range(rest):
            if (testset[len(testset)-rest+j][1] in labels):
               correct += (predicted[j][0] == labels.index(testset[len(testset)-rest+j][1]))
               tot += 1
        return correct, tot


# In[24]:

def train(mymodel, trainset, testset, imageTransform, testTransform, criterion, optimize, saveDir="data/", batchSize=32, epochStart=0, nbEpoch=50, bestScore=0):
    """
        Train a network
        inputs : 
            * trainset
            * testset, 
            * transformations to apply to image (for train and for test)
            * loss function (criterion)
            * optimizer
    """
    for epoch in range(ep, 100): # loop over the dataset multiple times
        running_loss = 0.0
        random.shuffle(trainset)    
        for i in range(len(trainset)/batchSize):
            # get the inputs
            inputs = torch.Tensor(batchSize,3,225,225).cuda()
            for j in range(batchSize):
                inputs[j] = imageTransform(trainset[j+i*batchSize][0])
            inputs = Variable(inputs)

            #get the labels
            lab = Variable(torch.LongTensor([labels.index(trainset[j+i*batchSize][1]) for j in range(batchSize)]).cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mymodel(inputs)
            loss = criterion(outputs, lab)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 10 == 9: # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 10))
                running_loss = 0.0

            if i % 50 == 49: #test every 20 mini-batches
                print('test :')
                c, t = testNet(mymodel, testset, labels, batchSize=batchSize)
                print("Correct : ", c, "/", t)
                if (correct >= bestScore):
                    best = mymodel
                    bestScore = correct
                    torch.save(best, "bestModel.ckpt")
                #else:
                #    mymodel = best
                torch.save(mymodel, path.join(saveDir,"model-"+str(epoch)+".ckpt"))
                mymodel.train() #set the model in train mode

    print('Finished Training')


# In[ ]:

if __name__=='__main__':
    
    #training and test sets
    trainset = ReadImages.readImageswithPattern('/video/CLICIDE', lambda x:x.split('/')[-1].split('-')[0])
    testset = ReadImages.readImageswithPattern('/video/CLICIDE/test/', lambda x:x.split('/')[-1].split('-')[0])

    m, s = readMeanStd('data/cli.txt')
    
    #define the labels list
    listLabel = [t[1] for t in trainset if not 'wall' in t[1]]
    labels = list(set(listLabel)) #we have to give a number for each label
    
    #open the images
    #do that only if it fits in memory !
    for i in range(len(trainset)):
        trainset[i] = (Image.open(trainset[i][0]), trainset[i][1])
        
    for i in range(len(testset)):
        testset[i] = (Image.open(testset[i][0]), testset[i][1])
        
    
    #define the model
    #mymodel = ModelDefinition.Maxnet()
    #ModelDefinition.copyParameters(mymodel, models.alexnet(pretrained=True))
    
    #or load the model
    mymodel = torch.load('bestModel.ckpt')
    
    criterion = nn.loss.CrossEntropyLoss()
    mymodel.train().cuda()
    
    #define the optimizer to only the classifier with lr of 1e-2
    optimizer=optim.SGD([
                    {'params': mymodel.classifier.parameters()},
                    {'params': mymodel.features.parameters(), 'lr': 0.0}
                ], lr=1e-2, momentum=0.9)


    imageTransform = transforms.Compose( (transforms.Scale(300), transforms.RandomCrop(225), transforms.ToTensor(), transforms.Normalize(m,s)) )
    testTransform = transforms.Compose( (transforms.Scale(225), transforms.CenterCrop(225), transforms.ToTensor(), transforms.Normalize(m,s)))
    batchSize = 64
    
    
    #define the optimizer train on all the network
    optimizer=optim.SGD(mymodel.parameters(), lr=0.0001, momentum=0.9)
    

