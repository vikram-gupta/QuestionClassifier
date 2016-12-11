import pickle
from Config import *
from FeatureExtractorModule import *
from Example import *

print "Loading the model..."
pickledModelFileHandle = open(pickledModelFileName, 'rb')
svcModelLoaded = pickle.load(pickledModelFileHandle)
svcModel = svcModelLoaded['model']

print "Load and Set the feature extractors..."
pickledFeatureExtractorsFileHandle = open(pickledFeatureExtractorsFileName,'rb')
featureExtractors = pickle.load(pickledFeatureExtractorsFileHandle)
setFeatureExtractors(featureExtractors)

print "Read the test examples"
testingDataFileHandle = open(testingDataFileName,'rb')
exampleLines = testingDataFileHandle.readlines()
testingExamples = []
for example in exampleLines:
    example = example.strip()
    ex = Example("",example)
    testingExamples.append(ex)

print "Extracting the features..."
testingClasses,testingFeatures = extractFeatures(testingExamples,False)

print "Inferencing..."
predictedClasses = svcModel.predict(testingFeatures)
for predictedClass in predictedClasses:
    print "The type of the question is: "+ predictedClass