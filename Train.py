from sklearn.svm import LinearSVC
import pickle
from Config import *

featuresDict = {}
trainedModel = {}

print "Loading the features..."
pickledFeaturesFileHandle = open(pickledFeaturesFileName, 'rb')
featuresDict = pickle.load(pickledFeaturesFileHandle)

print "Training using SVC"
svcModel = LinearSVC(loss='l2', dual=False, tol=1e-3)
svcModel.fit(featuresDict["trainingFeatures"], featuresDict["trainingClasses"])

trainedModel["model"] = svcModel

pickledModelFileNameHandle = open(pickledModelFileName,'wb')
pickle.dump(trainedModel, pickledModelFileNameHandle)
pickledModelFileNameHandle.close()



