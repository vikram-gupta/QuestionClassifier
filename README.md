# QuestionClassifier
A simple Question Classifier

Step 1) Read the training and testing data and extract the features. Pickle the extracted features.

python ReadDataAndExtractFeatures.py

Step 2) Train the SVM classifier.

python Train.py

3) Use the trained classifier from step 2 to infer the accuracy on the evaluation/development data set.
python Evalute.py

4) Run the ClassifyLive.py script to infer the classes for the test data set.
python ClassifyLive.py

