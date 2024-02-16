import pandas as pd
import numpy as np
import glob
import os
import datetime
import sqlite3
import pricesdb
import json
import pprint
import importlib
import newsdb

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

def get_sentiments_finbert(sentences, modelname='yiyanghkust/finbert-tone'):
    finbert = BertForSequenceClassification.from_pretrained(modelname,num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(modelname)
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, device=0)

    results = nlp(sentences)
    #print(results)  #LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
    return results

def get_topholdings_for_controllist(controllist):
    # die topholdings der watchlist ETFs aus den DBs holen und aggregieren
    agg_tophdf = pd.DataFrame()
    for i in range(len(controllist)):
        f = glob.glob(r"data/"+controllist['yahoo_symbol'][i] + "_*.db")[0]
        tophdf, _, _ = pricesdb.get_topholdings_from_db(f)
        if tophdf is not None:
            tophdf['yahoo_symbol'] = controllist['yahoo_symbol'][i]
            try:
                print('tophdf: ' + str(list(tophdf['symbol'])))
            except:
                pass
            agg_tophdf = pd.concat([agg_tophdf,tophdf])
    agg_tophdfdddf = agg_tophdf['symbol'].drop_duplicates()
    querylist = list(agg_tophdfdddf)
    return querylist, agg_tophdf

def load_no_news_symbols(file_no_news = 'data\\no_news.csv'):
    # no_news symbols aus Datei Laden
    try:
        no_news = pd.read_csv(file_no_news)
    except FileNotFoundError as error:
        print(error)
        print(f'File: {file_no_news} not found, initialized as empty DataFrame')
        no_news = pd.DataFrame(columns=['symbol'])
        pass
    return no_news

def insert_sentiment_title_list(conn, dfsentiment, modelname= 'yiyanghkust/finbert-tone'):
    cursor = conn.cursor()
    idl = list(dfsentiment['id'])
    selsql = f"""select id from sentiment_title_list where id in ({str(idl)[1:-1]}) and modelname = '{modelname}'"""
    ids = pd.read_sql(selsql, conn)
    delsql = f"""delete from sentiment_title_list where id in ({str(list(ids['id']))[1:-1]}) and modelname = '{modelname}'"""
    conn.execute(delsql)
    for index, row in dfsentiment.iterrows():
        insert_sql = f"""INSERT INTO sentiment_title_list ('id', 'sentiment', 'modelname', 'dtupdate') 
                                values ( 
                            "{row['id']}", 
                            "{row['sentiment']}",
                            "{modelname}",
                            "{datetime.datetime.today().strftime('%Y-%m-%d')}"
                            )"""
        try:
            cursor.execute(insert_sql)
        except sqlite3.Error as e:
            print(str(e) + ' \ninsert_sql: '+ insert_sql +' \nund weiter...')
            pass
    conn.commit()
    print(f'{modelname}: {str(index)} rows inserted')
    return conn

#Variables
file_no_news = 'data\\no_news.csv'
file_newsdb = 'data/seeking-alpha.db'
file_watchlist = 'watchlist.csv'
file_sentiment_result=f"""data\\agg_top_sentiment_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"""
file_sentiment_for_plot=f"""data\\sentiment_plotdf_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"""


# Laden der topholdings, die letzten 20 news zu jeder Holding von seekingalpha holen und in die newsdb speichern
# Symbols aus Watchlist
watchlist = pd.read_csv(file_watchlist, sep=';')
groupl = [10,40] #[20,30] haben sowieso keine Holdingliste
controllist = watchlist[watchlist['group'].isin(groupl)].reset_index()

querylist, agg_tophdf = get_topholdings_for_controllist(controllist)

no_news = load_no_news_symbols(file_no_news = 'data\\no_news.csv')
querylist_minus_no_news = list(set(querylist) - set(no_news['symbol']))

# die jeweils letzten 20 news der topholdings aus seekingalpha holen und in die Tabelle list_by_symbol speichern
conn, _  = newsdb.create_connection(file_newsdb)
for symbol in querylist_minus_no_news:
    list_by_symboldf, response = newsdb.get_news_list_by_symbol(symbol)
    if list_by_symboldf is not None and list_by_symboldf.shape != (0,0) :
        newsdb.insert_list_by_symbol(conn, f'{symbol}', list_by_symboldf)
conn.close()

# symbols herausfiltern zu denen es keine Nachrichten von seekingalpha gibt, da die Anfragen auch in das Kontigent gerechnet werden
conn, _  = newsdb.create_connection(file_newsdb)
sql = """select distinct symbol from list_by_symbol"""
have_news  = pd.read_sql(sql, conn)
new_no_news = pd.DataFrame(list(set(querylist) - set(have_news['symbol'])), columns=['symbol'])
no_news = pd.concat([no_news, new_no_news], axis=0)
no_news.to_csv('data\\no_news.csv', index=False)

# Die titles aus den news und mit FinBert den sentiment bewerten und in die newsdb speichern
conn, tablenames = newsdb.create_connection(file_newsdb)
if tablenames[tablenames.iloc[:,0].isin(['sentiment_title_list'])].shape[0] == 0:
    create_table_sentiment_title_list(conn)

# neue Titles sind in list_by_symbol heruntergeladen und werden jetzt mit FINBERT bewertet 
#welchen title id gibt's in der DB noch nicht
sql = """select id, "attributes.publishOn", "attributes.title" 
            from list_by_symbol
            where id not in (select id from sentiment_title_list)"""
sentimentdf  = pd.read_sql(sql, conn)
#print(f'sentimentdf.shape:{sentimentdf.shape}')

# Für jeden ETF für jedes symbol die titles aus der DB selektieren und in Finbert stecken, heraus soll eine Tabelle mit newsid, symbol publishOn, sentiment kommen, zu kompliziert
# Alle neuen werden gesentimented und in die sentiment_title_list geschrieben.
sentences = list(sentimentdf['attributes.title'])
sentimentdf['sentiment'] = get_sentiments_finbert(sentences)
#print(f'sentimentdf.shape:{sentimentdf.shape}')

conn = insert_sentiment_title_list(conn, sentimentdf)

# Mit den Sentiments arbeiten
# Falls ich hier anfange und die connection noch nicht da ist:
conn, _  = newsdb.create_connection(file_newsdb)  
# Alle sentimemts aus der DB lesen
sql = """select * 
         from list_by_symbol l
         left join sentiment_title_list s on s.id = l.id
         """
sentimentdf  = pd.read_sql(sql, conn)
#json result in Spalten wandeln und anhängen
#json.dumps(sentimentdf['sentiment'][0])
sentimentvalued = [eval(x) for x in sentimentdf['sentiment']]
sentimentvaluedf = pd.DataFrame(sentimentvalued)
sentimentdf['label']= sentimentvaluedf['label']
sentimentdf['score']= sentimentvaluedf['score']
#joinen mit den fonds aus den topholdings
agg_top_sentiment = pd.merge(agg_tophdf, sentimentdf, on='symbol', how='left')
# publishDate Spalte, die das Datum ohne Uhrzeit enthält, das wird zum Aggregieren und spätre zum Joinen gebraucht 
agg_top_sentiment['publishDate'] = pd.to_datetime(agg_top_sentiment['attributes.publishOn']).dropna().apply(lambda r:r.date())
# speichern
agg_top_sentiment.to_csv(file_sentiment_result)

# plotdf für Darstellung vorbereiten
#agg_top_sentiment = pd.read_csv(file_sentiment_result)
#plotdf ist der Teil, der in der ax überlagert dargestellt werden soll 
plotdf = agg_top_sentiment[['publishDate','yahoo_symbol','label']].dropna()
# plotdf.index = pd.to_datetime(plotdf['publishDate'])
# plotdf = plotdf.drop(['publishDate'], axis=1)
plotdf_group = plotdf.groupby(['publishDate','yahoo_symbol','label'])
plotdf = plotdf_group.size().reset_index(level=['yahoo_symbol','label'])
plotdf.columns = ['yahoo_symbol','label','size']

plotdf = plotdf.pivot_table('size',index=['yahoo_symbol','publishDate'], columns='label').fillna(0)
plotdf.to_csv(file_sentiment_for_plot)
plotdf
plotdf.index.max()
