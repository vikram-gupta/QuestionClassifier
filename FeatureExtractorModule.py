from sklearn.feature_extraction.text import CountVectorizer
from numpy import hstack
import nltk

def customTokenizer(text):
    return text.split()

word_vec = CountVectorizer(tokenizer=customTokenizer)
pos_vec = CountVectorizer(tokenizer=customTokenizer)

def setFeatureExtractors(featureExtractorMap):
    global word_vec
    global pos_vec
    word_vec = featureExtractorMap["word_vec"]
    pos_vec = featureExtractorMap["pos_vec"]

def extractFeatures(examples,isTraining):
    corpus = []
    posCorpus = []
    classTags = []

    index=0
    totalExamples = len(examples)

    for ex in examples:
        index = index+1
        print "Extracting features for example..."+str(index)+"/"+str(totalExamples)
        corpus.append(ex.text)
        posCorpus.append(extractPos(ex.text))
        classTags.append(ex.classTag)

    global word_vec
    global pos_vec
    if isTraining:
        wordIndices = word_vec.fit_transform(corpus).toarray()
        posIndices = pos_vec.fit_transform(posCorpus).toarray()
    else:
        wordIndices = word_vec.transform(corpus).toarray()
        posIndices = pos_vec.transform(posCorpus).toarray()

    allIndices = hstack((wordIndices,posIndices))
    return classTags,allIndices

def extractPos(text):
    text = nltk.word_tokenize(text)
    pos_seq=nltk.pos_tag(text)
    pos_seq_str = ""
    for pos in pos_seq:
        pos_seq_str=pos_seq_str+" "+pos[1]
    return pos_seq_str.strip()

def getVectorizers():
    return word_vec,pos_vec