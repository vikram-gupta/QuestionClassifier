import pickle
from Config import *

featuresDict = {}

print "Loading the features..."
pickledFeaturesFileHandle = open(pickledFeaturesFileName, 'rb')
featuresDict = pickle.load(pickledFeaturesFileHandle)

print "Loading the model..."
pickledModelFileName = open(pickledModelFileName, 'rb')
svcModelLoaded = pickle.load(pickledModelFileName)
svcModel = svcModelLoaded['model']

print "Testing..."
predictedClasses = svcModel.predict(featuresDict["testingFeatures"])

hits=0.00
for i in range(0,len(predictedClasses)):
    if predictedClasses[i]==featuresDict["testingClasses"][i]:
        hits = hits+1

print "Number of hits = ",hits
print "The accuracy is ",((hits/len(featuresDict["testingClasses"]))*100.0)," %"


