import datetime
import sqlite3
import os
import pandas as pd
import json
import requests
import pricesdb

def get_news_list_by_symbol(symbol):

	url = "https://seeking-alpha.p.rapidapi.com/news/v2/list-by-symbol"

	querystring = {"id":f"{symbol}","size":"20","number":"1"}

	headers = {
		"X-RapidAPI-Key": os.environ.get('X_RapidAPI_Key'),
		"X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
	}

	response = requests.request("GET", url, headers=headers, params=querystring)
	try:
		list_by_symbold = json.loads(response.text)
		if 'errors' in list_by_symbold:
			print(f'errors in response for {symbol}')
			return None, response 
		list_by_symboldf = pd.json_normalize(list_by_symbold['data'])
	except json.JSONDecodeError as e:
		print(str(e) + ' \nresponse:' + str(response))
		list_by_symboldf = None
		pass

	return list_by_symboldf, response

def create_table_list_by_symbol(conn):
    create_table = """CREATE TABLE list_by_symbol (
        'id' INTEGER PRIMARY KEY, 
        'symbol' TEXT,
        'dtupdate'     TIMESTAMP,
        'type' TEXT, 
        'attributes.publishOn' TEXT, 
        'attributes.isLockedPro' TEXT, 
        'attributes.commentCount' TEXT, 
        'attributes.gettyImageUrl' TEXT, 
        'attributes.videoPreviewUrl' TEXT, 
        'attributes.title' TEXT, 
        'attributes.isPaywalled' TEXT, 
        'relationships.author.data.id' TEXT, 
        'relationships.author.data.type' TEXT, 
        'relationships.sentiments.data' TEXT, 
        'relationships.primaryTickers.data' TEXT, 
        'relationships.secondaryTickers.data' TEXT, 
        'relationships.otherTags.data' TEXT, 
        'links.self' TEXT        
        );"""
    try:
        c = conn.cursor()
        c.execute(create_table)
        #c.execute(create_table_pk)
        return conn
    except sqlite3.Error as e:
        print(str(e) + '/n failed sql:' + create_table)
	
def create_connection(db_file):
    """ create a database and tables connection to a SQLite database """
    conn = None
    newdb = False
    if not os.path.exists(db_file):
        newdb = True
    try:
        conn = sqlite3.connect(db_file)
        #print(sqlite3.version)
        if newdb:
            tabels = create_table_list_by_symbol(conn)
            print(f'newdb created:{newdb}, tables:{tabels}')
        print(f'db_file:{db_file}')
    except sqlite3.Error as e:
        print(str(e))
    
    # gibt es die json_table
    df_tables = pricesdb.get_tables(conn)    
    
    return conn, df_tables

def insert_list_by_symbol(conn, symbol, list_by_symboldf):
    cursor = conn.cursor()
    idl = list(list_by_symboldf['id'])
    selsql = f'select id from list_by_symbol where id in ({str(idl)[1:-1]})'
    ids = pd.read_sql(selsql, conn)
    delsql = f"delete from list_by_symbol where id in ({str(list(ids['id']))[1:-1]})"
    conn.execute(delsql)
    for index, row in list_by_symboldf.iterrows():
        insert_sql = f"""INSERT INTO list_by_symbol ('id', 'symbol', 'dtupdate', 'type', 'attributes.publishOn', 'attributes.isLockedPro', 'attributes.commentCount', 'attributes.gettyImageUrl', 'attributes.videoPreviewUrl', 'attributes.title','attributes.isPaywalled', 'relationships.author.data.id','relationships.author.data.type', 'relationships.sentiments.data','relationships.primaryTickers.data', 'relationships.secondaryTickers.data', 'relationships.otherTags.data','links.self') 
                        values ( 
                    "{row['id']}", 
                    "{symbol}",
                    "{datetime.datetime.today().strftime('%Y-%m-%d')}",
                    "{row['type']}", 
                    "{row['attributes.publishOn']}", 
                    {row['attributes.isLockedPro']},
                    {row['attributes.commentCount']}, 
                    "{row['attributes.gettyImageUrl']}", 
                    "{row['attributes.videoPreviewUrl']}",
                    "{row['attributes.title']}", 
                    {row['attributes.isPaywalled']}, 
                    "{row['relationships.author.data.id']}",
                    "{row['relationships.author.data.type']}", 
                    "{row['relationships.sentiments.data']}", 
                    "{row['relationships.primaryTickers.data']}",
                    "{row['relationships.secondaryTickers.data']}", 
                    "{row['relationships.otherTags.data']}", 
                    "{row['links.self']}"
                    )"""
        #print(insert_sql)
        try:
            cursor.execute(insert_sql)
        except sqlite3.Error as e:
            print(str(e) + ' \ninsert_sql: '+ insert_sql +' \nund weiter...')
            pass
    conn.commit()
    print(f'{symbol}: {str(index)} rows/news inserted')
    return conn
