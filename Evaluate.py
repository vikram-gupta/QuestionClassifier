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

print "Evaluating on developement set..."
predictedClasses = svcModel.predict(featuresDict["developmentFeatures"])

hits=0.00
for i in range(0,len(predictedClasses)):
    if predictedClasses[i]==featuresDict["developmentClasses"][i]:
        hits = hits+1

print "Number of hits = ",hits
print "The accuracy is ",((hits/len(featuresDict["developmentClasses"]))*100.0)," %"


