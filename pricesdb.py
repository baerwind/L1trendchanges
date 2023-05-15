import datetime
import sqlite3
import os
import pandas as pd
import json
import requests

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
            tabels = create_tables(conn)
            print(f'newdb created:{newdb}, tables:{tabels}')
        print(f'db_file:{db_file}')
    except sqlite3.Error as e:
        print(e)
    
    # gibt es die json_table
    df_tables = get_tables(conn)
    tablel = list(df_tables.iloc[:,0])

    if 'json_table' not in tablel:
        create_json_table(conn)

    return conn

def create_tables(conn):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    create_prices = """CREATE TABLE prices (
        date     TIMESTAMP,
        symbol   TEXT,
        region   TEXT,
        open     REAL,
        high     REAL,
        low      REAL,
        close    REAL,
        volume   REAL,
        updatedt TIMESTAMP
    );"""

    create_prices_pk_index ="""
        CREATE UNIQUE INDEX pk_prices ON prices (
        date,
        symbol,
        region
    );"""

    create_events = """CREATE TABLE events (
        date     TIMESTAMP,
        symbol   TEXT,
        region   TEXT,
        amount   REAL,
        type     TEXT,
        data     REAL,
        updatedt TIMESTAMP
    );
    """

    create_events_pk = """CREATE UNIQUE INDEX pk_events ON events (
        date,
        symbol,
        region,
        type
    );
    """

    try:
        c = conn.cursor()
        c.execute(create_prices)
        c.execute(create_prices_pk_index)
        c.execute(create_events)
        c.execute(create_events_pk)

        create_json_table(conn)

        return 'prices, events'
    except sqlite3.Error as e:
        print(e)

def create_json_table(conn):
    create_json_table = """CREATE TABLE json_table (
        date     TIMESTAMP,
        symbol   TEXT,
        region   TEXT,
        endpoint TEXT,
        json_data JSON
        );"""

    create_json_table_pk = """CREATE UNIQUE INDEX pk_json_table ON json_table (
        date,
        symbol,
        region,
        endpoint
    );
    """

    try:
        c = conn.cursor()
        c.execute(create_json_table)
        c.execute(create_json_table_pk)
        return 'json_table'
    except sqlite3.Error as e:
        print(e)


def delete_tableentries(conn, table, symbol, region, dates):
    """
    Delete existing tableentries  
    :param conn:  Connection to the SQLite database
    :param symbol: symbol of the task
    :dates: dates list
    :return:
    """
    sql = f"DELETE FROM {table} WHERE symbol='{symbol}' and region = '{region}' and date in ({dates})"
    #print(sql)
    cur = conn.cursor()
    try:
        cur.execute(sql)
        conn.commit()
    except sqlite3.Error as e:
        print(e + ' \nsql: '+sql)


def get_tables(conn):
    sql = "SELECT name FROM sqlite_schema WHERE  type ='table' AND  name NOT LIKE 'sqlite_%';"
    try:
        c = conn.cursor()
        c.execute(sql)
        rows = c.fetchall()
        return pd.DataFrame(rows)

    except sqlite3.Error as e:
        print(e + ' \nsql: '+sql)

def get_prices_from_db(dbfile :str, fromdate :str = str(int(datetime.datetime.now().date().strftime('%Y')) -1) + '-' + datetime.datetime.now().date().strftime('%m') +'-'+ '01'):
    conn = create_connection(dbfile)
    if conn is None:
        exit()
    datetime.datetime.date
    #prices = pd.read_sql_query("SELECT * FROM prices", conn)
    # das letzte Jahr vom Anfang des aktuellen Monats, sonst kann man die x Achsen Beschriftung nicht mehr lesen
    sql = "SELECT * FROM prices where date>='" + fromdate + "'"
    prices = pd.read_sql_query(sql,conn)

    prices_sorted = prices.sort_values(['date'])
    prices_sorted = prices_sorted.reset_index(drop=True)
    return prices_sorted

def get_topholdings_from_db(dbfile :str):
    conn = create_connection(dbfile)
    if conn is None:
        exit()
    sql = """select json_data from json_table \
                where endpoint = 'topholdings' \
                and date = (select max(date) from json_table group by date, symbol, region, endpoint);"""
    cur = conn.cursor()
    try:
        cur.execute(sql)
        json_data = cur.fetchall()
    except sqlite3.Error as e:
        print(e + ' \nsql: '+sql)

    if json_data is None:
        return None, None, None
    else:    
        json_data = json.loads(json_data[0][0]) 
        # erste 0 für 1. Zeile, zweite 0 für 1. Teil im Tupel (Zeile)
        # besser wäre ein resultset als Daraframe
        try:
            holdings = json_data['quoteSummary']['result'][0]['topHoldings']['holdings']
            df = pd.json_normalize(holdings)
            return df, holdings, json_data
        except KeyError:
            print(f"""{dbfile} Key: json_data['quoteSummary']['result'][0]['topHoldings']['holdings'] does not exist""")
            return None, None, json_data

def request_build_headers():
    headers = {
        "X-RapidAPI-Key": '', # hier muss der APIKey hinein, oder in die Umgebungsvariable X-RapidAPI-Key
        "X-RapidAPI-Host": "yh-finance.p.rapidapi.com"
    }
    if headers.get('X-RapidAPI-Key') == '':
         headers = {
            "X-RapidAPI-Key": os.environ.get('X_RapidAPI_Key'),
            "X-RapidAPI-Host": "yh-finance.p.rapidapi.com"
         }

    if headers.get('X-RapidAPI-Key') == '':
        print('No API Key found in env variable X-RapidAPI-Key')
        return
    return headers

def get_topholdings(symbol, region):
    url = "https://yh-finance.p.rapidapi.com/stock/get-top-holdings"

    querystring = {"symbol":symbol,"region":region,"lang":"en-US"}
    headers = request_build_headers()
    response = requests.request("GET", url, headers=headers, params=querystring)
    try:
        resp = response.json()
        json_data = json.dumps(resp)
        json_data = response
    except json.JSONDecodeError as err:
        json_data = None
        print(str(err) + f' {symbol}_{region} - continue')
        pass
    return json_data

def get_topholdings_update_dbs(symbol, region):
    """
    --CREATE TABLE my_table (id INTEGER PRIMARY KEY, json_data JSON);
    --INSERT INTO my_table (json_data) VALUES (json('<json_data_here>'));
    --SELECT json_extract(json_data, '$.key') FROM my_table;
    """
    json_data = get_topholdings(symbol, region)
    
    # connection 
    conn = create_connection(r"data/"+symbol+"_"+region+".db")
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    endpoint = 'topholdings'

    sql = f"""delete from json_table where date='{date}' and  symbol='{symbol}' and region='{region}'"""
    cur = conn.cursor()
    try:
        cur.execute(sql)
        conn.commit()
    except sqlite3.Error as e:
        print(str(e) + ' \nsql: '+sql)

    if json_data is None:
        return conn
    else:
        sql = f"""INSERT INTO json_table (date, symbol, region, endpoint, json_data) \
            VALUES ('{date}','{symbol}','{region}','{endpoint}','{json_data}');"""
        cur = conn.cursor()
        try:
            cur.execute(sql)
            conn.commit()
            print(f'topholdings updated for {symbol}_{region}')
        except sqlite3.Error as e:
            print(str(e) + ' \nsql: '+sql)
            print(f"""INSERT INTO json_table (date, symbol, region, endpoint) VALUES ('{date}','{symbol}','{region}','{endpoint}') - error but continue""")
            pass

    return conn

def get_historical_data(symbol, region):
    """
    get data from yahoo
    """

    url = "https://yh-finance.p.rapidapi.com/stock/v3/get-historical-data"
    querystring = {"symbol":symbol,"region":region}
    headers = request_build_headers()
    
    response = requests.request("GET", url, headers=headers, params=querystring)
    if response.status_code != 200:
        print(f'No valid JSON response for {symbol}_{region}')
        return None, response.status_code
    try: 
        df = response.json() 
        df1 = pd.json_normalize(df)
    except json.JSONDecodeError as err:
        df1 = None
        print(str(err) + f' {symbol}_{region} - continue')
        pass
    return df1, response.status_code

def get_prices_update_dbs(symbol, region):
    print(f'{symbol}_{region}')    
    # raw_prices aus rapi
    raw_prices, status_code = get_historical_data(symbol,region)
    print('status_code:' + str(status_code) + ' raw_prices:' + str(raw_prices)[:100] )
    if raw_prices is None:
        print(f'No prices found in JSON response for {symbol}_{region}, status_code:{status_code}')
        return None
    if len(raw_prices.columns) ==0:
        print(f'No prices found in JSON response for {symbol}_{region}, status_code:{status_code}')
        return None
    # ist in raw_prices['prices'][0] etwas drin
    if raw_prices['prices'][0] == []:
        print(f'No prices found in JSON response for {symbol}_{region}, status_code:{status_code}')
        return None        

    prices = pd.DataFrame(raw_prices['prices'][0])
    # Spalten ergänzen
    prices['symbol'] = symbol
    prices['region'] = region
    # Datümer umwandeln
    prices['date'] = pd.to_datetime(prices['date'], unit='s')
    prices['updatedt'] = datetime.datetime.today()
    # events abspalten - nur wenn es kein Index ist - gibt es events überhaupt
    eventsavailable = False
    if 'type' in prices.columns:
        eventsavailable = True
        # null Spalten löschen
        events = prices[pd.notnull(prices['type'])]
        prices = prices[pd.isnull(prices['type'])]
        events = events.drop(['open', 'high', 'low', 'close', 'volume', 'adjclose'], axis= 1)
        prices = prices.drop(['adjclose', 'amount', 'type', 'data'], axis=1)    

    # connection 
    # bei neuem symbol wird automatisch eine neue db generiert
    conn = create_connection(r"data/"+symbol+"_"+region+".db")

    # schon vorhandene aus DB löschen, damit neue überschrieben werden können
    dates = prices['date'].drop_duplicates()
    datesl = [str(date) for date in dates]
    delete_tableentries(conn, 'prices', symbol, region, str(datesl)[1:-1])

    if eventsavailable == False:
        prices = prices.drop(['adjclose'], axis=1)

    # prices in db schreiben
    prices.to_sql('prices', conn, if_exists='append' , index=False)
    if eventsavailable ==  True:
        # auch für events schon vorhandene aus DB löschen
        e_dates = events['date'].drop_duplicates()
        e_datesl = [str(e_date) for e_date in e_dates]
        # events in db schreiben
        delete_tableentries(conn, 'events', symbol, region,  str(e_datesl)[1:-1])
        events.to_sql('events', conn, if_exists='append' , index=False)
    print(f'DB updated for {symbol}_{region}')
    
    return conn