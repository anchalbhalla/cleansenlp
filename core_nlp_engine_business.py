# Databricks notebook source
# MAGIC %run /Shared/Customer_Voice_ML/scripts/setup

# COMMAND ----------

#!/usr/bin/env python
# coding: utf-8

import sys
import datetime 
import os 
from datetime import datetime

# datetime object containing current date and time
start_now = datetime.now()

# dd/mm/YY H:M:S
dt_string = start_now.strftime("%d/%m/%Y %H:%M:%S")
print("SCRIPT STARTED AT =", dt_string)	


import pandas as pd 
import numpy as np  

from string import punctuation


from dotenv import load_dotenv

load_dotenv("/dbfs/FileStore/tables/credentials.env")       
sys.stdout.fileno = lambda: False    


from sqlalchemy import create_engine, NVARCHAR
import pyodbc 
import urllib

from datar.all import *
from pipda import register_func

import re 
import time 
import requests 

import torch
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig,  pipeline
from sentence_transformers import SentenceTransformer

import spacy 
import spacy_transformers
nlp = spacy.load("en_core_web_trf") # python3 -m spacy download en_core_web_trf 

from flashtext import KeywordProcessor
from fuzzywuzzy import process, fuzz


# fall back mechanism for pipda/datar
select.ast_fallback = "piping"
rename.ast_fallback = "piping"
mutate.ast_fallback = "piping"
distinct.ast_fallback = "piping"
left_join.ast_fallback = "piping"

# COMMAND ----------

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords 
from textblob import TextBlob
from bertopic import BERTopic

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer


# COMMAND ----------

driver = "{ODBC Driver 17 for SQL Server}"

lemmatizer = nltk.stem.WordNetLemmatizer()
wordnet_lemmatizer = WordNetLemmatizer()

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


stop = stopwords.words('english')
add_stopwords = pd.read_csv("/dbfs/FileStore/tables/stopwords.csv")


add_stopwords['WORD'] = add_stopwords['WORD'].str.lower()
add_stopwords = add_stopwords['WORD'].to_list()
stop.extend(add_stopwords)


nltk.download('words')
words = set(nltk.corpus.words.words())

hugging_face_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

keyword_processor = KeywordProcessor(case_sensitive=False)

#keyword_processor.add_keyword_from_file("synonyms.txt")

keyword_dictionary = {
    "application": ["app"],
    "problem": ["issue", "error"],
    "website": ["site", "webpage","url","page"],
    "technical": ["tech"],
    "pass" : ["pas"],
    "premit" : ["perm"],
    "covid" :["cofid"],
    "born" :["bear"],
    "fodder" :["feed"],
    "upgrade" : ["upgread"], 
    "complaint" : ["complains", "Complainant"], 
    "tank" : ["tanker"],
    "child" : ["son", "daughter"]

}

keyword_processor.add_keywords_from_dict(keyword_dictionary)

# COMMAND ----------

read_qry = '''

select 

DISTINCT(a.CASE_NUMBER),a.CASE_TYPE, a.CASE_CHANNEL, a.ENTITY,a.EntityID, a.ENTITY_SHORT_NAME,c.VERTICAL, a.REGION,a.SERVICE_CATEGORY,a.SERVICE_NAME,
b.CASE_DESCRIPTION_EN, a.STATUS, a.DATE_OPENED 

FROM dw.vw_Cases_CustomerVoice a

LEFT JOIN  nlp.cases_translated b ON a.CASE_NUMBER = b.CASE_NUMBER
LEFT JOIN nlp.adge_mapping c ON a.ENTITY_SHORT_NAME = c.ENTITY_SHORT_NAME
                    
                    where a.DATE_OPENED >= Dateadd(Month, Datediff(Month, 0, DATEADD(m, -1,current_timestamp)), 0) and 
                    a.DATE_OPENED < Dateadd(Month, Datediff(Month, 0, DATEADD(m, 0,current_timestamp)), 0) and 

                    (a.REGION LIKE'%Abu Dhabi%'or a.REGION LIKE'%Al Ain%' or a.REGION LIKE'%Al Dhafrah%') and 
                    (a.ENTITY NOT LIKE '%AlHosn App%') and 
                    (c.VERTICAL LIKE '%Business%' and a.SERVICE_CATEGORY IS NOT NULL)
                    
except 

select 

DISTINCT(a.CASE_NUMBER),a.CASE_TYPE, a.CASE_CHANNEL, a.ENTITY,a.EntityID, a.ENTITY_SHORT_NAME,c.VERTICAL, a.REGION,a.SERVICE_CATEGORY,a.SERVICE_NAME,
b.CASE_DESCRIPTION_EN, a.STATUS, a.DATE_OPENED 

FROM dw.vw_Cases_CustomerVoice a

LEFT JOIN  nlp.cases_translated b ON a.CASE_NUMBER = b.CASE_NUMBER
LEFT JOIN nlp.adge_mapping c ON a.ENTITY_SHORT_NAME = c.ENTITY_SHORT_NAME
                    
                    where a.DATE_OPENED >= Dateadd(Month, Datediff(Month, 0, DATEADD(m, 0,current_timestamp)), 0) and 
                    a.DATE_OPENED < Dateadd(Month, Datediff(Month, 0, DATEADD(m, 0,current_timestamp)), 0) and 

                    (a.REGION LIKE'%Abu Dhabi%'or a.REGION LIKE'%Al Ain%' or a.REGION LIKE'%Al Dhafrah%') and 
                    (a.ENTITY NOT LIKE '%AlHosn App%') and 
                    (c.VERTICAL LIKE '%Business%' and a.SERVICE_CATEGORY IS NOT NULL) 

                    '''



delete_dups_detail_qry = '''

            DELETE  x from (
                    SELECT  *, rn=row_number() over (partition by CASE_NUMBER order by [TOPIC_NAME] )
                    FROM  nlp.customer_case_topic_details
                    ) x
                    WHERE rn > 1;

                  '''

delete_dups_summary_qry = '''

                  DELETE  x from (
                    SELECT  *, rn=row_number() over (partition by VERTICAL,ENTITY_SHORT_NAME,SERVICE_CATEGORY,SERVICE_NAME,TOPIC_NAME_EN,
                                               MONTH,YEAR order by [TOPIC_SUMMARY] )
                    FROM  [nlp].[customer_case_topic_summary]
                    ) x
                    WHERE rn > 1;

                    ''' 


delete_dups_keywords_qry = ''' 
                  DELETE  x from (
                    SELECT  *, rn=row_number() over (partition by Case_Number, Keywords order by [Keywords] )
                    FROM  nlp.customer_case_keyword_match
                    ) x
                    WHERE rn > 1; 

                    '''

# COMMAND ----------

def data_ingest_node(sql_query):

    server = os.getenv('server')
    database = os.getenv('database')
    username = os.getenv('user')
    password = os.getenv('password')
    

    
    # SQL Authentication
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
       
    df = pd.read_sql(sql_query,conn)
    df = pd.DataFrame(df) # convert to modin dataframe 
    
    return df  




def data_write_node(df,db_table_name): 
      
    server = os.getenv('server')
    database = os.getenv('database')
    user_write= os.getenv('user_write')
    password_write = os.getenv('password_write')
    
    conn = f"""Driver={driver};Server=tcp:{server},1433;Database={database};
    Uid={user_write};Pwd={password_write};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"""
    
    params = urllib.parse.quote_plus(conn)
    conn_str = 'mssql+pyodbc:///?charset=utf8?autocommit=true&odbc_connect={}'.format(params)
    engine = create_engine(conn_str, echo=False)
    
    df.to_sql(db_table_name, con=engine, if_exists='append',schema = 'nlp',index=False,method='multi',chunksize=100,
              dtype={col_name: NVARCHAR for col_name in df}) # needed for arabic push 
    
    return 


def del_rows_db(delete_dups_qry):

    server = os.getenv('server')
    database = os.getenv('database')
    user_write= os.getenv('user_write')
    password_write = os.getenv('password_write')
  
   
    # SQL Authentication
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+user_write+';PWD='+ password_write)
    
    cursor = conn.cursor()
    cursor.execute(delete_dups_qry)
    conn.commit()
    cursor.close()
    conn.close()     #<--- Close the connection

# COMMAND ----------

# function to perform basic cleaning of text 
@register_func(None)
def clean_text(string):
    string = str(string)
    string = re.sub('([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})',"", string) # remove email address
    # remove urls
    string = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', string)
    string = re.sub(r"\d+", "", string)
    string = string.replace("-", " ")
    string = string.replace("/", " ")
    string = re.sub(f"[{re.escape(punctuation)}]", "", string)
    string = re.sub(r"[^A-Za-z0-9\s]+", "", string)
    string = string.replace(".", "")
    string = string.replace("(", "")
    string = string.replace(")", "")
    string = string.replace("_", " ")
    string = string.replace("-", " ")
    string = string.replace("'m", " am")
    string = string.replace("'s", " is")
    string = string.replace("'ve", " have")
    string = string.replace("n't", " not")
    string = string.replace("'re", " are")
    string = string.replace("'d", " would")
    string = string.replace("'ll", " will")
    string = string.replace("\r", " ")
    string = string.replace("\n", " ")
    string = string.strip().lower()

    return string  

remove_junk = lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words)


# lemmatize the sentences using the wordnet 
@register_func(None)
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
    
    
@register_func(None)
def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def remove_names(text):
    doc = nlp(text)
    newString = text
    for e in reversed(doc.ents):
        if e.label_ == "PERSON": # Only if the entity is a PERSON
            newString = newString[:e.start_char] + newString[e.start_char + len(e.text):]
    return newString


def replace_synonyms(text):
    # keyword_processor = KeywordProcessor(case_sensitive=False)
    text = keyword_processor.replace_keywords(text)
    return text 


@register_func(None)
def create_bert_topics(data) : 

    print('\n') 
    # apply  the function to clean text  
    print("Text cleaning started ...")

    data["text_clean"] = data["CASE_DESCRIPTION_EN"].apply(lambda x : clean_text(x))
    
    # REMOVE PUNCTUATION 
    print("Removing punctionations ...")
    data["text_clean"] = data["text_clean"].str.replace('[^\w\s]','')
    
    # SPELL CHECK - text blob gives nonetype objects - must change back to string 
    print("Running spelling checks ...")
    data["text_clean"] = data["text_clean"].apply(lambda x : str(TextBlob(x)))
    
    # remove stopwords 
    print("Removing stopwords ...")
    data["text_clean"] = data["text_clean"].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
    

    
    # lemmatise the sentences 
    print("Running Lemmatization ...")
    data["text_clean"] = data["text_clean"].apply(lambda x: lemmatize_sentence(x))
    
    # remove names and replace synonyms 
    print("Removing person names and replacing synonyms ...")
    data["text_clean"] = data["text_clean"].apply(lambda x : remove_names(x))
    data["text_clean"] = data["text_clean"].apply(lambda x: replace_synonyms(x))
    data["text_clean"] = data["text_clean"].apply(remove_junk)
    
    # perform topic modeling on clean sentences
    print("Running Topic Modeling using BERT embeddings ...")
    topic_model = BERTopic(verbose=True,
                           embedding_model= hugging_face_model, 
                           min_topic_size= 10,
                           calculate_probabilities=True)
    
    docs = data["text_clean"].to_list()
    topics, probs = topic_model.fit_transform(docs)
    
    
    new_topics, new_probs = topic_model.reduce_topics(data["text_clean"], 
                                                      topics,
                                                      probs, 
                                                      nr_topics=7) # reduce higher number of topics to 7 or less 
    
    
    print("Extracting and finalizing BERT topic results ...")
    print('\n') 
    
    df = pd.DataFrame(new_topics, columns = ['Topic'])
    df = pd.concat([df.reset_index(drop=True), data.reset_index(drop=True)], axis=1)
    
    # df= df._to_pandas() # back to pandas dataframe

    freq = topic_model.get_topic_info()
    
    df = df >> left_join(freq)
    
    return df

# COMMAND ----------

#checks in which cases are those keywords present and returns the dataframe
def check_keywords_case (cases, keywords): 
    # print(df.head()) 
    keywords_list = (list(list(zip(*keywords))[0]))
    all_cases = cases['CASE_DESCRIPTION_EN'].values.tolist()
    all_number = cases['CASE_NUMBER'].values.tolist() 

    matching_keywords = [] 
    matching_numbers = []

    for i in range(len(all_cases)):
        if(all_cases[i] is not None):
            for words in keywords_list: 
                word_split = words.split() 
                if(len(word_split) > 1):
                    if((word_split[0] in all_cases[i]) or (word_split[1] in all_cases[i])):
                        matching_keywords.append(words)
                        matching_numbers.append(all_number[i])
                else: 
                    if((words in all_cases[i])):
                        matching_keywords.append(words)
                        matching_numbers.append(all_number[i]) 

    number_keywords_df = pd.DataFrame(list(zip(matching_numbers, matching_keywords)),
               columns =['Case_Number', 'Keywords'])
    return(number_keywords_df)

# COMMAND ----------


def bert_topics_node(df):
    
    # df= df._to_pandas() # back to pandas dataframe
    grouping = list(df.groupby(['ENTITY_SHORT_NAME','SERVICE_CATEGORY','SERVICE_NAME'], as_index=False))
    
    extracted_topics = []
    
    
    for i in  range(0,len(grouping)): 

            print('\n') 
            print(i) 
            print('\n') 
            print("Now creating topics for " + str(grouping[i][0]) + "..." ) 
            
            group_df = grouping[i][1]
                   

            if group_df.shape[0] > 10 :

                topic_result = create_bert_topics(data = group_df )
                extracted_topics.append(topic_result)  
                print("Topics created by analysing " + str(group_df.shape[0] ) + " records") 
            
            else:
                
                print("Topics not created due to less data - < 10 records ") 

                    
    extracted_topics =  pd.concat(extracted_topics)
            

    return extracted_topics




def bert_keyphrases_node(df):
    
    topic_grouping = list(df.groupby(['ENTITY_SHORT_NAME','SERVICE_CATEGORY',"SERVICE_NAME",'TOPIC_NAME_EN'], as_index=False)) # keyphrases must be extracted for each topic created 

    
    extracted_keyphrases = []
    kw_model = KeyBERT()

    global final_matching_key
     
    for i in  range(0,len(topic_grouping)): 
            
            print('\n') 
            print(i) 
            print('\n') 
            print("Now extracting keyphrases for " + str(topic_grouping[i][0]) + "..." ) 
            
            
            group_df = topic_grouping[i][1]
            
            doc = group_df.TEXT_CLEAN.str.cat(sep=' ')


            keywords = kw_model.extract_keywords(docs=doc,
                                                 #vectorizer=KeyphraseCountVectorizer(), # ngrams can be chosen as well instead of vectorizer but it is known to give less grammatically accurate phrases 
                                                 keyphrase_ngram_range=(1, 2),
                                                 use_mmr=True, 
                                                 stop_words = stop,
                                                 diversity=0.5) 
            

            #check where those keywords are located for the cases
            if(len(keywords)>0):
                temp_matching_key = check_keywords_case(group_df, keywords) 
                global final_matching_key
                final_matching_key = final_matching_key.append(temp_matching_key)
            
            
            keywords = pd.DataFrame(keywords, columns = ['Keyphrase', 'Importance'])
            # keywords= keywords._to_pandas() # back to pandas dataframe
            keywords = keywords >> mutate( ENTITY_SHORT_NAME = str(topic_grouping[i][0][0]),
                                            SERVICE_CATEGORY = str(topic_grouping[i][0][1]),
                                            SERVICE_NAME = str(topic_grouping[i][0][2]),
                                            TOPIC_NAME_EN = str(topic_grouping[i][0][3])
                                            ) 
            
            extracted_keyphrases.append(keywords)
            print("COMPLETED SUCCESSFULLY !!!") 

    extracted_kephrases =  pd.concat(extracted_keyphrases) 

    
    return(extracted_kephrases)


# Importing the bert model
tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def topic_summary_node(df):
    
    topic_grouping = list(df.groupby(['ENTITY_SHORT_NAME','SERVICE_CATEGORY','SERVICE_NAME','TOPIC_NAME_EN','MONTH','YEAR','VERTICAL'], as_index=False))
    
    summary = []
    pipe_summary=[]
    


    for i in  range(0,len(topic_grouping)): 
            
            print('\n') 
            print(i) 
            print("Now generating topic summaries for " + str(topic_grouping[i][0]) + "..." ) 
            
            
            group_df = topic_grouping[i][1]
            
            original_text  = group_df.CASE_DESCRIPTION_EN.str.cat(sep='.')



            # Encoding the inputs and passing them to model.generate()
            inputs = tokenizer.batch_encode_plus([original_text],return_tensors='pt',max_length=512)
            summary_ids = model.generate(inputs['input_ids'], early_stopping=True)
            
            
            # Decoding and printing the summary
            bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            

            bart_summary = pd.DataFrame([bart_summary], columns = ['TOPIC_SUMMARY'])
            
            bart_summary = bart_summary >> mutate( ENTITY_SHORT_NAME = str(topic_grouping[i][0][0]),
                                            SERVICE_CATEGORY = str(topic_grouping[i][0][1]),
                                            SERVICE_NAME = str(topic_grouping[i][0][2]),
                                            TOPIC_NAME_EN = str(topic_grouping[i][0][3]),
                                            MONTH = str(topic_grouping[i][0][4]),
                                            YEAR = str(topic_grouping[i][0][5]),
                                            VERTICAL = str(topic_grouping[i][0][6])
                                            ) 

            summary.append(bart_summary)
            
            
            print("BART SUMMARY GENERATED !!!") 
           
           

    bart_summary =  pd.concat(summary)
    
    return(bart_summary)


# COMMAND ----------

print('++++++++++++++++ DATA INGESTION  STARTED ++++++++++++++++ ')

data = data_ingest_node(read_qry)
print("Number of rows pulled : " + str(data.shape[0]))
print("Number of columns pulled : " + str(data.shape[1]))
print('\n')

if(data.shape[0]<=0):
    dbutils.notebook.exit([0])

print('++++++++++++++++ DATA INGESTION  ENDED ++++++++++++++++ ')
print('\n')

# COMMAND ----------

# topics are generated and then assigned to relevant cases - topics will NOT be generated for groups where fewer than 20 cases are present
print('++++++++++++++++ TOPIC CREATION STARTED ++++++++++++++++ ')

topics_df = bert_topics_node(df = data)
topics_df['DATE_OPENED'] = pd.to_datetime(topics_df['DATE_OPENED'])
topics_df['month'] = topics_df['DATE_OPENED'].dt.strftime('%B')
topics_df['year'] = topics_df['DATE_OPENED'].dt.strftime('%Y')
topics_df = topics_df >> rename(record_count_topic = f.Count,topic_name = f.Name) >> select(~f.Topic)
topics_df.columns = topics_df.columns.str.upper()

print('++++++++++++++++ TOPIC CREATION ENDED SUCCESSFULLY ++++++++++++++++ ')
print('\n') 


# COMMAND ----------

map_file = pd.read_csv("/dbfs/FileStore/tables/topic_name_mapping_business.csv")

# COMMAND ----------

def map_func(entity,category,name) :
    
    raw_df = topics_df >>  filter(f.ENTITY_SHORT_NAME == entity,
                                    f.SERVICE_CATEGORY == category,
                                    f.SERVICE_NAME == name )

    raw_df["TOPIC_NAME_CLEAN"] = raw_df["TOPIC_NAME"].replace("_", " ",regex=True)
    raw_df["TOPIC_NAME_CLEAN"] = raw_df["TOPIC_NAME_CLEAN"].replace("-", "",regex=True)
    raw_df["TOPIC_NAME_CLEAN"] = raw_df["TOPIC_NAME_CLEAN"].str.replace(r'\d+','',regex=True)  
    
    map_df = map_file >>  filter(f.ENTITY_SHORT_NAME == entity,
                          f.SERVICE_CATEGORY == category,
                          f.SERVICE_NAME == name )
    
    
    choices = map_df["MAPPED_TOPIC_NAME"].tolist()
        
    raw_df['TOPIC_NAME_EN'] = raw_df['TOPIC_NAME_CLEAN'].apply(lambda x: process.extract( x, choices, limit=1, scorer= fuzz.token_sort_ratio))
    
    return raw_df 

# COMMAND ----------

topic_groups = topics_df >> distinct(f.ENTITY_SHORT_NAME,f.SERVICE_CATEGORY,f.SERVICE_NAME)
 
matched_topics = []

for row in topic_groups.itertuples(index=True, name='Pandas'):

      matched_result = map_func(entity = row.ENTITY_SHORT_NAME,
                                category = row.SERVICE_CATEGORY,
                                name = row.SERVICE_NAME)

      matched_topics.append(matched_result)  

matched_topics =  pd.concat(matched_topics)
matched_topics["TOPIC_NAME_EN"] = matched_topics["TOPIC_NAME_EN"].astype("string")
matched_topics["TOPIC_NAME_EN"] = matched_topics["TOPIC_NAME_EN"].str.replace('[^\w\s]','',regex=True)
matched_topics["TOPIC_NAME_EN"]  = matched_topics["TOPIC_NAME_EN"].str.replace(r'\d+','',regex=True) 

# COMMAND ----------

matched_topics.head()

# COMMAND ----------

matched_topics  = matched_topics >> mutate(TOPIC_NAME_EN = if_else((f.TOPIC_NAME_EN.str.len()) < 1, f.TOPIC_NAME_CLEAN, f.TOPIC_NAME_EN)) >> \
                                    select(~f.TOPIC_NAME_CLEAN,~f.TOPIC)
                                   
matched_topics.columns = matched_topics.columns.str.upper()

# COMMAND ----------

matched_topics.shape

# COMMAND ----------

topics_aggregated = matched_topics >> distinct(f.ENTITY_SHORT_NAME,f.VERTICAL,f.SERVICE_CATEGORY,f.SERVICE_NAME,f.TOPIC_NAME,f.TOPIC_NAME_EN,
                                                f.RECORD_COUNT_TOPIC,f.MONTH,f.YEAR) >> \
                                        select(f.VERTICAL, f.ENTITY_SHORT_NAME,f.SERVICE_CATEGORY,f.SERVICE_NAME ,f.TOPIC_NAME,f.TOPIC_NAME_EN,
                                            f.MONTH,f.YEAR)    

topics_aggregated['TOPIC_NAME_EN'] = topics_aggregated['TOPIC_NAME_EN'].str.strip()
topics_aggregated['TOPIC_NAME_EN'] = topics_aggregated['TOPIC_NAME_EN'].str.capitalize()

topics_df_push = topics_df >> select(~f.ENTITYID,~f.SERVICCATEGORYKEY,~f.SERVICEKEY) 

topics_df_push = topics_df_push >> left_join(topics_aggregated, by=[f.VERTICAL,f.ENTITY_SHORT_NAME, f.SERVICE_CATEGORY,f.SERVICE_NAME,f.TOPIC_NAME, f.MONTH,f.YEAR])

# COMMAND ----------

topics_df_push.head()

# COMMAND ----------

print('++++++++++++++++ KEYPHRASE EXTRACTION PER TOPIC STARTED ++++++++++++++++ ')
final_matching_key = pd.DataFrame(columns = ['Case_Number', 'Keywords'])
# Init KeyBERT
keyphrase_df = bert_keyphrases_node(df = topics_df_push)
keyphrase_df = keyphrase_df >> select(f.ENTITY_SHORT_NAME,f.SERVICE_CATEGORY,f.SERVICE_NAME,f.TOPIC_NAME_EN,
                                        f.Keyphrase,f.Importance)

print('++++++++++++++++ KEYPHRASE EXTRACTION PER TOPIC COMPLETED ++++++++++++++++ ')
print('\n')
    

# COMMAND ----------

final_matching_key

# COMMAND ----------

print('++++++++++++++++ SUMMARY GENERATION PER TOPIC STARTED ++++++++++++++++ ')
# Init KeyBERT
summary_df = topic_summary_node(df = topics_df_push)

summary_df = summary_df >> select(f.VERTICAL,f.ENTITY_SHORT_NAME,f.SERVICE_CATEGORY,f.SERVICE_NAME, f.TOPIC_NAME_EN,f.TOPIC_SUMMARY,f.MONTH,f.YEAR)

print('++++++++++++++++ SUMMARY GENERATION PER TOPIC COMPLETED ++++++++++++++++ ')
print('\n')

# COMMAND ----------

summary_df

# COMMAND ----------

print('++++++++++++++++ DATA WRITE STARTED ++++++++++++++++ ')

case_topics_dtl_write_push = data_write_node(df= topics_df_push, db_table_name = 'customer_case_topic_details')
print(case_topics_dtl_write_push)

topic_summary_write_push = data_write_node(df= summary_df, db_table_name = 'customer_case_topic_summary')
print(case_topics_dtl_write_push)

case_keyword_match_write_push = data_write_node(df= final_matching_key, db_table_name = 'customer_case_keyword_match')
print(case_keyword_match_write_push)


print('++++++++++++++++ DATA WRITE ENDED ++++++++++++++++ ')
print('\n')

# COMMAND ----------



print('++++++++++++++++ TOPICS DETAILS DUPLICATE PURGE STARTED ++++++++++++++++ ')

del_push = del_rows_db(delete_dups_detail_qry)
print(del_push)

print('++++++++++++++++ TOPICS DETAILS DUPLICATE PURGE ENDED ++++++++++++++++ ')
print('\n')


print('++++++++++++++++ TOPICS SUMMARY DUPLICATE PURGE STARTED ++++++++++++++++ ')

del_push = del_rows_db(delete_dups_summary_qry)
print(del_push)

print('++++++++++++++++ TOPICS SUMMARY DUPLICATE PURGE ENDED ++++++++++++++++ ')
print('\n')


print('++++++++++++++++ KEYWORDS MATCHING DUPLICATE PURGE STARTED ++++++++++++++++ ')

del_push = del_rows_db(delete_dups_keywords_qry)
print(del_push)

print('++++++++++++++++ KEYWORDS MATCHING DUPLICATE PURGE ENDED ++++++++++++++++ ')
print('\n')

# COMMAND ----------

