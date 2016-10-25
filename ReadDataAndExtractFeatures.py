import re
from Example import *
from FeatureExtractorModule import *
import pickle
from Config import *

def preprocess(sentence):
    sentence = re.sub(r'[$|.|!|"|(|)|,|;|`|\']',r'',sentence)
    return sentence

def readData(fileName):
    dataFileHandle = open(fileName,'r')
    exampleLines = dataFileHandle.readlines()
    examples = []
    for example in exampleLines:
        example = example.strip()
        example = preprocess(example)
        toks = example.split(' ',1)
        classTag = toks[0]
        text = toks[1]
        ex = Example(classTag,text)
        examples.append(ex)
    return examples

print "Reading training examples..."
trainingExamples = readData(trainingDataFileName)
print "Reading testing examples..."
testingExamples = readData(testingDataFileName)

featuresDict = {}

print "Extracting training features..."
trainingClasses,trainingFeatures = extractFeatures(trainingExamples,True)

print "Extracting testing features..."
testingClasses,testingFeatures = extractFeatures(testingExamples,False)

featuresDict["trainingClasses"] = trainingClasses
featuresDict["trainingFeatures"] = trainingFeatures
featuresDict["testingClasses"] = testingClasses
featuresDict["testingFeatures"] = testingFeatures

print "Pickling the features..."
pickledFeaturesFileHandle = open(pickledFeaturesFileName,'wb')
pickle.dump(featuresDict, pickledFeaturesFileHandle)
pickledFeaturesFileHandle.close()

print "Done"