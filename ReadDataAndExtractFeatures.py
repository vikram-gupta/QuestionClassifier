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
print "Reading development examples..."
developmentExamples = readData(developmentDataFileName)

featuresDict = {}
featuresExtractorDict = {}

print "Extracting training features..."
trainingClasses,trainingFeatures = extractFeatures(trainingExamples,True)

print "Extracting development features..."
developmentClasses, developmentFeatures = extractFeatures(developmentExamples, False)

print "Saving vectorizers..."
word_vec,pos_vec = getVectorizers()

featuresDict["trainingClasses"] = trainingClasses
featuresDict["trainingFeatures"] = trainingFeatures
featuresDict["developmentClasses"] = developmentClasses
featuresDict["developmentFeatures"] = developmentFeatures

featuresExtractorDict["word_vec"] = word_vec
featuresExtractorDict["pos_vec"] = pos_vec

print "Pickling the features..."
pickledFeaturesFileHandle = open(pickledFeaturesFileName,'wb')
pickle.dump(featuresDict, pickledFeaturesFileHandle)
pickledFeaturesFileHandle.close()

print "Pickling the feature extractors..."
pickledFeatureExtractorsFileHandle = open(pickledFeatureExtractorsFileName,'wb')
pickle.dump(featuresExtractorDict, pickledFeatureExtractorsFileHandle)
pickledFeatureExtractorsFileHandle.close()

print "Done"