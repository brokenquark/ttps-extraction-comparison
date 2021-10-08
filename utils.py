from typing import Counter, Text
from gensim.models import lsimodel
from numpy import positive
import re
from numpy.ma import count
import spacy
from rank_bm25 import BM25Okapi

def removeCitation(text):
    position = text.find('(Citation:')
    return text[:position]

def removeUrls (text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    return(text)

def removeURLandCitationBulk(texts):
    return [removeCitation(removeUrls(text)) for text in texts]

def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = lsimodel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def splitTechniqueName(text):
    return text.split(':')[0]

def filterSubTechniques(text):
    length = len(text.split(':'))
    if length >= 2: return 1
    else: return 0

class NPChunks:
    def __init__(self, chunk) -> None:
        self.chunk = chunk
        self.prev = None
        self.next = None
        self.nextPrep = None

def checkIfQueryContainsPhrase(phrase, query):
    phrase_tokens = phrase.split(' ')
    len_phrase_token = len(phrase_tokens)
    query_tokens = query.split(' ')
    match_indexes = []
    
    for index in range(0, len_phrase_token):
        for token in query_tokens:
            if phrase_tokens[index] == token:
                match_indexes.append(index)
    
    if len(match_indexes) == 0: return False
    
    if (match_indexes[-1] - match_indexes[0]) == (match_indexes[0] + len_phrase_token - 1):
        if len_phrase_token == len(query_tokens): return False
        else: return True
    else:
        return False

def getListOfNounPhrases(text):
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    
    listOfTokens = []

    for token in doc:
        listOfTokens.append(token)
        
    listOfNPChunks = []


    for npchunk in doc.noun_chunks:
        listOfNPChunks.append(NPChunks(npchunk))


    for npchunk in listOfNPChunks:
        start = npchunk.chunk.start
        end = npchunk.chunk.end 
        if start > 0: 
            if (listOfTokens[start-1]).dep_ == 'prep':
                prev_chunks = [x for x in listOfNPChunks if x.chunk.end == start - 1]
                if len(prev_chunks) > 0:
                    # print(f'found a noun chunks before: {prev_chunks[0].chunk}')
                    npchunk.prev = prev_chunks[0]
                    prev_chunks[0].next = npchunk
                    prev_chunks[0].nextPrep = listOfTokens[start-1]
   
    listofNBARs = [] 
    index = 0
    while (index < len(listOfNPChunks)):
        if listOfNPChunks[index].next == None:
            listofNBARs.append(listOfNPChunks[index].chunk.text)
            index += 1
            continue
        else:
            for i in range(index+1, len(listOfNPChunks)):
                if listOfNPChunks[i].next != None:
                    pass
                else:
                    tmplist = ""
                    for j in range(index, i):
                        tmplist += " " + listOfNPChunks[j].chunk.text + " " + listOfNPChunks[j].nextPrep.text
                    tmplist += " " + listOfNPChunks[i].chunk.text
                    tmplist = tmplist.strip()
                    listofNBARs.append(tmplist)
                    index = i + 1
                    
    return listofNBARs

def filterNPsFromCorpus(paragraphs):
    listofallnounphrases = set()
    listofallnounphrases = []
    listofallfreenounphrases = []

    count = 0
    empty = 0
    for para in paragraphs:
        phrases = getListOfNounPhrases(para)
        phrases = [p.lower().strip() for p in phrases]
        # todo remove a and the and probably some other garbages
        for p in phrases:
            tokens = p.split(' ')
            tokens = [token for token in tokens if token not in ['a', 'an', 'the']]
            newPhrase = ' '.join(tokens).strip()
            # listofallnounphrases.add(newPhrase)
            listofallnounphrases.append(newPhrase)
        # print(phrases)
        print('Processed ' + str(count))
        count += 1

    countOfNPs = Counter(listofallnounphrases)
    
    for phrase in listofallnounphrases:
        isPhraseFree = True
        for query in listofallnounphrases:
            if checkIfQueryContainsPhrase(phrase, query) and countOfNPs[f'{phrase}'] < 1:
                isPhraseFree = False
                print(f'-- {phrase} -- is in -- {query} -- ')
        if isPhraseFree: listofallfreenounphrases.append(phrase)

    listOfProcessedPara = []
    listOfProcessedParaSuppressed = []

    count = -1
    for para in paragraphs:
        count += 1
        text = ''
        for item in listofallfreenounphrases:
            if item in para:
                text += ' ' + item
        text = text.strip()
        if len(text) == 0:
            text = ' '
            empty += 1
        listOfProcessedPara.append(text)
        
        for item in listofallnounphrases:
            text = ''
            if item in para:
                text += ' ' + item
            listOfProcessedParaSuppressed.append(text)
        # print(f'data-{count}: {para}')
        # print(f'data: {text}')
        # print('###')

    print('total empty para: ' + str(empty))
    return [listOfProcessedPara, listOfProcessedParaSuppressed]



def get_subject_phrase(doc):
    phrases = []
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            # return doc[start:end]
            phrases.append(doc[start:end])
    return phrases

def get_verb_phrase(doc):
    phrases = []
    for token in doc:
        if ("ROOT" in token.dep_):
            subtree = list(token.subtree)
            for node in subtree:
                if 'ROOT' in node.head.dep_ and 'cc' in node.dep_:
                    conjTokens = [ct for ct in list(token.subtree) if 'conj' in ct.dep_]
                    if len(conjTokens) > 0: 
                        phrases.extend(conjTokens)
                
                if 'ROOT' in node.head.dep_ and 'prep' in node.dep_:
                    conjTokens = [ct for ct in list(token.subtree) if 'pcomp' in ct.dep_]
                    if len(conjTokens) > 0: 
                        phrases.extend(conjTokens)
            
                if 'ROOT' in node.head.dep_ and 'advcl' in node.dep_:
                        phrases.append(node)
            # return token
            phrases.append(token)
    return phrases

def get_object_phrase(doc):
    phrases = []
    for token in doc:
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            
            for node in subtree:
                if 'dobj' in node.head.dep_ and 'cc' in node.dep_:
                    conjTokens = [ct for ct in list(token.subtree) if 'conj' in ct.dep_]
                    if len(conjTokens) > 0: 
                        phrases.extend(conjTokens)
            
            start = subtree[0].i
            end = subtree[-1].i + 1
            # return doc[start:end]
            # phrases.append(doc[start:end])
            phrases.append(token)
    return phrases
        
def get_dative_phrase(doc):
    phrases = []
    for token in doc:
        if ("dative" in token.dep_):
            subtree = list(token.subtree)
            
            for node in subtree:
                if 'dative' in node.head.dep_ and 'cc' in node.dep_:
                    conjTokens = [ct for ct in list(token.subtree) if 'conj' in ct.dep_]
                    if len(conjTokens) > 0: 
                        phrases.extend(conjTokens)
            
            start = subtree[0].i
            end = subtree[-1].i + 1
            # return doc[start:end]
            # phrases.append(doc[start:end])
            phrases.append(token)
    return phrases

def get_attr_phrase(doc):
    phrases = []
    for token in doc:
        if ("attr" in token.dep_) and 'ROOT' in token.head.dep_:
            phrases.append(token)
    return phrases

def get_advcl_phrase(doc):
    phrases = []
    for token in doc:
        if ("advcl" in token.dep_):
            phrases.append(token)
    return phrases

def get_other_phrases(doc):
    phrases = []
    for token in doc:
        if (("oprd" in token.dep_) or ("acomp" in token.dep_) or ('ccomp' in token.dep_) or ('xcomp' in token.dep_) or ('acl' in token.dep_) or ('relcl' in token.dep_)):
            phrases.append(token)
    return phrases
      
def get_prepositional_phrase_objs(doc):
    prep_spans = []
    for token in doc:
        if ("pobj" in token.dep_):
            subtree = list(token.subtree)
            
            for node in subtree:
                if 'pobj' in node.head.dep_ and 'cc' in node.dep_:
                    conjTokens = [ct for ct in list(token.subtree) if 'conj' in ct.dep_]
                    if len(conjTokens) > 0: 
                        prep_spans.extend(conjTokens)
            
            start = subtree[0].i
            end = subtree[-1].i + 1
            # prep_spans.append(doc[start:end])
            prep_spans.append(token)
    return prep_spans

def extractBoW(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    subject_phrase = get_subject_phrase(doc)
    object_phrase = get_object_phrase(doc)
    verb_phrase = get_verb_phrase(doc)
    dative_phrase = get_dative_phrase(doc)
    prep_object_phrase = get_prepositional_phrase_objs(doc)
    attr_phrase = get_attr_phrase(doc)
    other_phrases = get_other_phrases(doc)
    advcl_phrase = get_advcl_phrase(doc)
    
    BoW = ''
    for token in subject_phrase: BoW += ' ' + token.text
    for token in object_phrase: BoW += ' ' + token.text
    for token in verb_phrase: BoW += ' ' + token.text
    for token in dative_phrase: BoW += ' ' + token.text
    for token in prep_object_phrase: BoW += ' ' + token.text
    for token in attr_phrase: BoW += ' ' + token.text
    for token in other_phrases: BoW += ' ' + token.text
    for token in advcl_phrase: BoW += ' ' + token.text
    return BoW

def getBM25Score(text, bm25, index):
    tokenized_query = text.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    return doc_scores[index]
    
    