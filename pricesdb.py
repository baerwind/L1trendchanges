import datetime
import sqlite3
import os
import pandas as pd
import json

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
