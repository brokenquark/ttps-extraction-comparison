from numpy.ma import count
import pandas as pd
from pandas import DataFrame
import gensim
from gensim.parsing.preprocessing import preprocess_documents
from scipy.sparse import data
import utils
from gensim.models.coherencemodel import CoherenceModel
import csv
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np, random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from scipy import spatial

from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")

# implements LSI-NN non bias corrected

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def main():
    
    pd.set_option('display.max_columns', None)
    np.random.RandomState(0)

    dfTactics = pd.read_excel('attack-data.xlsx', sheet_name=0)
    dfTechniques = pd.read_excel('attack-data.xlsx', sheet_name=2)
    dfProcedures = pd.read_excel('attack-data.xlsx', sheet_name=3)

    dfTacticsCut = dfTactics.loc[:, ['ID', 'name', 'description']]
    dfTacticsCut['type'] = 'tactics'
    dfTechniquesCut = dfTechniques.loc[:, ['ID', 'name', 'description']]
    dfTechniquesCut['type'] = 'techniques'

    dfTechniqueProcedureMerged = pd.merge(dfTechniques, dfProcedures, left_on='ID', right_on='target ID')

    dfProceduresCut = dfTechniqueProcedureMerged.loc[:, ['source ID', 'name', 'mapping description']]
    dfProceduresCut['ID'] = dfProceduresCut['source ID']
    dfProceduresCut['description'] = dfProceduresCut['mapping description']
    dfProceduresCut['type'] = 'example'
    dfProceduresCut = dfProceduresCut.loc[:, ['ID', 'name', 'description', 'type']]

    dataframe = pd.concat([dfTacticsCut, dfTechniquesCut, dfProceduresCut], ignore_index=True)
    
    
    # trainAndTestSet2 = dataframe.loc[dataframe['type'] == 'techniques']
    # trainAndTestSet2['name'] = trainAndTestSet2['name'].apply(utils.splitTechniqueName)
    # print(trainAndTestSet2.head())
    
    trainAndTestSet = dataframe.loc[dataframe['type'] == 'example']
    trainAndTestSetWithTTPs = dataframe.loc[dataframe['type'] == 'techniques']
    trainAndTestSet['name'] = trainAndTestSet['name'].apply(utils.splitTechniqueName)
    trainAndTestSetWithTTPs['subtechnique'] = trainAndTestSetWithTTPs['name'].apply(utils.filterSubTechniques)
    
    trainAndTestSetWithTTPs = trainAndTestSetWithTTPs.loc[trainAndTestSetWithTTPs['subtechnique'] == 0]
    
    trainAndTestSetWithTTPs.drop('subtechnique', axis=1, inplace=True)
    
    # print(trainAndTestSetWithTTPs.shape)
    
    techniqueNamesWithLessThanFiveExamples = []
    trainAndTestSetGrouped = trainAndTestSet.groupby('name')
    
    classCounts = []
    
    for name,group in trainAndTestSetGrouped:
        classCounts.append({ 'name' : f'{name}', 'count' : group.shape[0]})
        if group.shape[0] < 30:
            techniqueNamesWithLessThanFiveExamples.append(name)
    
    file = open('lsi_background_bias_corrected_final_result.txt', 'w')
    
    for top_n_class in [2, 4, 8, 16, 32, 64]:
        file.write('\n=================\n')
        file.write(f'n = {top_n_class}\n')
        print(f'n = {top_n_class}')
        
        classCounts_sorted = sorted(classCounts, key = lambda x:x['count'], reverse = True)[0:top_n_class]
        classCounts_top_n = [item['name'] for item in classCounts_sorted]
        
        trainAndTestSetFiltered = trainAndTestSet[trainAndTestSet['name'].isin(classCounts_top_n)]
        trainAndTestSetFilteredWithTTPs = trainAndTestSetWithTTPs[trainAndTestSetWithTTPs['name'].isin(classCounts_top_n)]

    
        # why 500: https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html
        
        
        dataset = pd.concat([trainAndTestSetFiltered, trainAndTestSetFilteredWithTTPs], ignore_index=True)
        
        # print(dataset.tail())
        # print(dataset[515:516])
        # print(dataset.shape)
        # print(trainAndTestSetFiltered.shape)
        # print(trainAndTestSetFilteredWithTTPs.shape)
        
        # text_corpus = trainAndTestSetFiltered['description'].values
        text_corpus = dataset['description'].values
        
        
        # for i in range(0, len(text_corpus)):
        #     print(text_corpus[i])
        #     if i == 515:
        #         x = 0
        
        text_corpus = utils.removeURLandCitationBulk(text_corpus)
        processed_corpus = preprocess_documents(text_corpus)

        text_clean = []
        for tokens in processed_corpus:
            text_clean.append(tokens)

        dictionary = gensim.corpora.Dictionary(processed_corpus)

        bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        tfidf = gensim.models.TfidfModel(bow_corpus, smartirs='nfc', id2word=dictionary)
        corpus_tfidf = tfidf[bow_corpus]
        
        tfidf_vectors = []
        dictionaryLength = len(dictionary)
        for item in corpus_tfidf:
            vector = []
            for index in range(0, dictionaryLength):
                tfidf_score = [element for element in item if element[0] == index]
                if len(tfidf_score) > 0:
                    vector.append(tfidf_score[0][1])
                else:
                    vector.append(0)
            tfidf_vectors.append(vector)
        
        tfidf_vectors = np.array(tfidf_vectors)
        
        
        
        technique_tfidf = tfidf_vectors[-top_n_class:]
        cosinescore_list_list = []
        
        for i in range(0, trainAndTestSetFiltered.shape[0]):
            # print(i)
            cosinescore_list = []
            
            # _text_corpus = trainAndTestSetFiltered[i:(i+1)]['description'].values
            # _text_corpus = utils.removeURLandCitationBulk(_text_corpus)
            # _text_corpus = split_into_sentences(_text_corpus[0])
            # _processed_corpus = preprocess_documents(_text_corpus)
            # _bow_corpus = [dictionary.doc2bow(text) for text in _processed_corpus]
            
            _bow_corpus = [dictionary.doc2bow(text) for text in [processed_corpus[i], []]]
            _tfidf = gensim.models.TfidfModel(_bow_corpus, smartirs='nfc', id2word=dictionary)
            _corpus_tfidf = _tfidf[bow_corpus]
            lsi = gensim.models.LsiModel(_corpus_tfidf, id2word=dictionary, num_topics=500)
            lsi_vectors = lsi.get_topics()
            
            for item in technique_tfidf:
                cosinescores = []
                
                for element in lsi_vectors:
                    result = 1 - spatial.distance.cosine(item, element)
                    cosinescores.append(result)
                sum_of_score = sum(cosinescores)
                cosinescore_list.append(sum_of_score)
            cosinescore_list_list.append(cosinescore_list)
        
        
        index = gensim.similarities.MatrixSimilarity(lsi[corpus_tfidf])
        
        
        feature_names = ['feature-' + str(i) for i in range(0, top_n_class)]
        df = pd.DataFrame(cosinescore_list_list, columns=feature_names)
        trainAndTestSetFiltered = pd.concat([trainAndTestSetFiltered.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
        
        numOfColumns = len(trainAndTestSetFiltered.columns)

        sm = SMOTE(random_state = 2, sampling_strategy='auto')
        X_train_res, y_train_res = sm.fit_resample(trainAndTestSetFiltered.iloc[:, 4:numOfColumns], trainAndTestSetFiltered['name'].ravel())
        
        
        X_train_res_df = pd.DataFrame(np.array(X_train_res)).add_prefix('column')
        y_train_res_df = pd.DataFrame(np.array(y_train_res), columns=['name'])
        
        trainAndTestSetFiltered = pd.concat([ X_train_res_df.reset_index(drop=True), y_train_res_df.reset_index(drop=True) ], axis=1)
        
        
        skf = StratifiedKFold(n_splits=5)
        target = trainAndTestSetFiltered.loc[:,'name']

        train = []
        test = []
        for train_index, test_index in skf.split(trainAndTestSetFiltered, target):
            train.append(trainAndTestSetFiltered.iloc[train_index])
            test.append(trainAndTestSetFiltered.iloc[test_index])

        for item in ['knn', 'nb', 'svm', 'rf', 'dt', 'nn']:
            file.write('\n###################\n')
            file.write(f'classifier: {item}\n')
            print(f'classifier: {item}')

            accuracy = []
            precision_m = []
            precision_w = []
            recall_m = []
            recall_w = []
            f1_m = []
            f1_w = []
            auc = []

            for index in range(0, 5):
                numOfColumns = len(train[index].columns)
                
                clf = None
                
                if item == 'knn': clf = KNeighborsClassifier().fit(train[index].iloc[:, 0:(numOfColumns-1)], train[index]['name'])
                
                if item == 'nn': clf = MLPClassifier().fit(train[index].iloc[:, 0:(numOfColumns-1)], train[index]['name'])
                
                if item == 'nb': clf = GaussianNB().fit(train[index].iloc[:, 0:(numOfColumns-1)], train[index]['name'])
                
                if item == 'svm': clf = svm.SVC(probability=True).fit(train[index].iloc[:, 0:(numOfColumns-1)], train[index]['name'])
                
                if item == 'dt': clf = DecisionTreeClassifier().fit(train[index].iloc[:, 0:(numOfColumns-1)], train[index]['name'])
                
                if item == 'rf': clf = RandomForestClassifier().fit(train[index].iloc[:, 0:(numOfColumns-1)], train[index]['name'])
                
                predicted = clf.predict(test[index].iloc[:, 0:(numOfColumns-1)])
                
                output = classification_report(test[index]['name'], predicted, output_dict =  True)
                
                probs = clf.predict_proba(test[index].iloc[:, 0:(numOfColumns-1)])
                
                if top_n_class == 2: 
                    auc.append(roc_auc_score( test[index]['name'] , probs[:,1]))
                else: 
                    auc.append(roc_auc_score( test[index]['name'] , probs , multi_class='ovr', average='weighted'))
                
                accuracy.append(output['accuracy'])
                precision_m.append(output['macro avg']['precision'])
                precision_w.append(output['weighted avg']['precision'])
                recall_m.append(output['macro avg']['recall'])
                recall_w.append(output['weighted avg']['recall'])
                f1_m.append(output['macro avg']['f1-score'])
                f1_w.append(output['weighted avg']['f1-score'])

            file.write(f'accuracy: {sum(accuracy)/5}\n')
            file.write(f'precision macro: {sum(precision_m)/5}\n') 
            file.write(f'precision weighted: {sum(precision_w)/5}\n') 
            file.write(f'recall macro: {sum(recall_m)/5}\n') 
            file.write(f'recall weighted: {sum(recall_w)/5}\n') 
            file.write(f'f1 macro: {sum(f1_m)/5}\n') 
            file.write(f'f1 weighted: {sum(f1_w)/5}\n')  
            file.write(f'auc: {sum(auc)/5}\n')  
            
    file.write('=================\n')
    file.close()


if __name__ == "__main__":
    main()