# Symbols aus Watchlist aus Tradingview
# Historische Preise zu den Symbols von yh yahoo, sofern nötig, mit Zwischenspeicher, da begrenzte Umsonstabfragen
# Vernüftiges Lambda pro Symbol bestimmen, z.B. <12 Trendwechsel im Jahr?, Lambda pro Symbol abspeichern
# Trendwechsel abspeichern (Feststellen ob die Trendwechsel mit neuen Daten auch in der Vergangenheit entstehen können? Ob sich das ganze mit neuen Daten verschiebt)
# Darstellung, plots pro symbol eine column

import pandas as pd
import numpy as np
import cvxpy 
import scipy
import cvxopt 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import matplotlib
import matplotlib.cm as cm
from matplotlib import patheffects
import requests
import glob
import os
import datetime
import sqlite3
import pricesdb
import json
#import newslist_topholdings


# functions für den DB Teil
# symbols in controllist laden und die dazugehörigen prices und events aus API holen und in DBs speichern

# functions für den calculations Teil
def calc_trendlines(y, lambda_list, solver, reg_norm):
    """die trendlines für die lambdas berechnen"""
    n = y.size
    ones_row = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)

    trendlinesl = []
    #trendlinesd = pd.DataFrame()
    trendchangesl = []
    #trendchangesd = pd.DataFrame()
    aggtchanges1 = []
    for i, lambda_value in enumerate(lambda_list):
        x = cvxpy.Variable(shape=n)     # x is the filtered trend that we initialize    
        objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(y-x) 
                    + lambda_value * cvxpy.norm(D@x, reg_norm))    # Note: D@x is syntax for matrix multiplication    
        problem = cvxpy.Problem(objective)
        problem.solve(solver=solver, verbose=False)

        trendlinesl.append(np.array(x.value))
        
        # trendchanges index
        r = np.abs(np.diff(np.diff(np.array(x.value))))
        r75, r25 = np.percentile(r,[75,25])
        iqr = r75 - r25
        trendchanges = np.where(r > r75 + 100*iqr)[0]
        #trendchanges = dates[np.where(r > r75 + 1000*iqr)[0]].to_list()
        trendchangesl.append(trendchanges)

    return trendlinesl, trendchangesl

def calc_bs_signals(trendchangesd, trendlined):
    aggtchanges = trendchangesd.to_frame()
    aggtchanges['orig_index'] = aggtchanges.index
    aggtchanges = pd.merge(aggtchanges, trendlined[trendlined.index.isin(aggtchanges['date'])], on='date', how='left')
    aggtchanges['datediff'] = aggtchanges['date'].diff(periods=-1)
    aggtchanges = aggtchanges[(aggtchanges['datediff'] < datetime.timedelta(-2.0)) | aggtchanges['datediff'].isnull()]
    aggtchanges.index = aggtchanges['date']
    aggtchanges.columns = ['date','orig_index','trendlinevalue', 'diffdate']
    #trendlined.columns = ['trendlinevalue']
    #aggtchanges = pd.merge(aggtchanges, trendlined[trendlined.index == trendlined.index.max()], on='date', how='outer')
    trd = trendlined[trendlined.index == trendlined.index.max()]
    trd.columns = ['trendlinevalue']
    aggtchanges = pd.concat([aggtchanges,trd])
    trd = trendlined[trendlined.index == trendlined.index.min()]
    trd.columns = ['trendlinevalue']
    aggtchanges = pd.concat([aggtchanges,trd])    
    aggtchanges['date'] = aggtchanges.index # damit auch der erste und der letzte ein Wert in ['date'] hat
    aggtchanges = aggtchanges.sort_index()
    aggtchanges['slope_until'] = aggtchanges['trendlinevalue'].diff()#/aggtchanges['date'].diff(periods=1).dt.days
    aggtchanges['slopechange'] = aggtchanges['slope_until'].shift(-1) - aggtchanges['slope_until'] 
    #aggtchanges['slopechange1'] = aggtchanges['slope_until'].diff()
    aggtchanges['buy']  = (aggtchanges['slope_until']<0) & (aggtchanges['slope_until'].shift(-1)>0) #'kaufen'
    aggtchanges['sell'] = (aggtchanges['slope_until']>0) & (aggtchanges['slope_until'].shift(-1)<0) #'verkaufen'

    return aggtchanges

def transform_calc_trendlines(prices, lambda_list, solver, reg_norm):
    """
    transforms und executes calculation, from one df of prices with lists of lambdas and 
    returns a list x values dfs and trendchanges df, where x changes Steigung (Trend) 
    prices: df with columns date und close
    lambda_list:
    """
    # prices umdrehen links frühes Datum rechts späteres
    prices = prices.sort_values(by='date',ascending=True)
    # drop 
    prices = prices.dropna(subset=['close'])
    close_prices = prices['close'].copy().to_numpy()
    # log warum eigentlich? Ohne log scheint der Algorithmus instabil zu sein? 
    #close_prices = np.log(close_prices)
    # Versuch mit Skalierung auf 1
    factor = close_prices[0] # bei 1 geht es los
    close_prices *=(1.0/factor)
    prices['close_start_1'] = close_prices

    trendlinesl, trendchangesl = calc_trendlines(close_prices, lambda_list, solver, reg_norm)
    # Datümer wieder ergänzen
    trendlinedl = []
    trendchangesdl = []
    aggtchangesl = []

    resultdf = prices
    resultdf['date'] = pd.to_datetime(resultdf['date'])
    for i, (trendline, trendchanges, lambdaf) in enumerate(zip(trendlinesl, trendchangesl, lambda_list)):
        trendlined = pd.DataFrame(trendline,pd.to_datetime(prices['date']))
        trendlinedl.append(trendlined)
        trendchangesd = pd.to_datetime(prices.iloc[trendchanges]['date'])
        trendchangesdl.append(trendchangesd)

        resultdf[f'trendline_lambda_{lambdaf}'] = trendline
        resultdf[f'trendchanges_lambda_{lambdaf}'] = resultdf.index.isin(trendchanges)

        #calc b/s signals
        aggtchanges = calc_bs_signals(trendchangesd, trendlined)
        aggtchangesl.append(aggtchanges)
        aggtchanges.index = aggtchanges['orig_index']
        aggtchanges.columns = [f'agg_{colname}_lambda_{lambdaf}' if colname!= 'date' else colname for colname in aggtchanges.columns]
        resultdf = pd.merge(resultdf,aggtchanges,on='date',how='left')

    return prices, trendlinedl, trendchangesdl, aggtchangesl, resultdf

# function für die Darstellung
def calc_slope_diff(trendchangesd, prev_trendchangesd, trendlined, prev_trendlined):
    # neue Events
    # 
    #new_trendchng = trendchangesd[trendchangesd > prev_trendchangesd.max()].values
    new_trendchng = trendchangesd[trendchangesd.index > prev_trendchangesd.index.max()].index
    #new_trendchngs = new_trendchng.dt.strftime('%Y-%m-%d').item().to_list()
    # Steigung
    # jeweils vom letzten - nicht vom letzten gemeinsamen 
    #timedelta = trendlined.index.max() - trendchangesd.max()
    timedelta = trendlined.index.max() - trendchangesd.index.max()
    #st_trend = (trendlined[trendlined.index==trendlined.index.max()][0].item() - trendlined[trendlined.index==trendchangesd.max()][0].item())/timedelta.days
    st_trend = (trendlined[trendlined.index==trendlined.index.max()][0].item() - trendlined[trendlined.index==trendchangesd.index.max()][0].item())/timedelta.days
    #prev_timedelta = prev_trendlined.index.max() - prev_trendchangesd.max()
    prev_timedelta = prev_trendlined.index.max() - prev_trendchangesd.index.max()
    #st_prev_trend = (prev_trendlined[prev_trendlined.index==prev_trendlined.index.max()][0].item() - prev_trendlined[prev_trendlined.index==prev_trendchangesd.max()][0].item())/prev_timedelta.days
    st_prev_trend = (prev_trendlined[prev_trendlined.index==prev_trendlined.index.max()][0].item() - prev_trendlined[prev_trendlined.index==prev_trendchangesd.index.max()][0].item())/prev_timedelta.days
    diff_st_trend = st_trend - st_prev_trend 
    
    return new_trendchng, diff_st_trend, st_trend, st_prev_trend

def get_prices_toph(controllist):
    # actual prices and topholdings from provider
    for i in range(len(controllist)):
        print(controllist['tradv_ticker'].iloc[i])
        symbol = controllist['yahoo_symbol'].iloc[i]
        region = controllist['region'].iloc[i]
        
        conn = pricesdb.get_prices_update_dbs(symbol, region)
        conn = pricesdb.get_topholdings_update_dbs(symbol,region)


def calc_l1trendchanges(controllist):
    # Hauptroutine
    # all together now
    # alle vorhandenen DBs lesen und die Trends rechnen und darstellen 
    lambda_list = [1]#[0.5, 1]
    #lambda_list = [0, 0.1, 0.5, 1, 2, 5, 10, 50, 200, 500, 1000, 2000, 5000, 10000, 100000]
    #lambda_list = [0.1, 0.5, 1, 2, 5, 10]
    solver = cvxpy.CVXOPT
    reg_norm = 1

    num_runs = 0
    prev_dist=5 # week
    gresultdf = pd.DataFrame()
    prev_gresultdf = pd.DataFrame()
    gtophdf = pd.DataFrame()

    for i, yahoo_symbol in enumerate(controllist['yahoo_symbol']):
        f = glob.glob(r"data/"+ yahoo_symbol + "_*.db")[0]
        prices_sorted = pricesdb.get_prices_from_db(f)
        topholdingsdf, _, _ = pricesdb.get_topholdings_from_db(f)
        gtophdf = pd.concat([gtophdf,topholdingsdf])
        
        #pricesd, trendlinedl, trendchangesdl, aggtchangesl, resultdf = transform_calc_trendlines(prices_sorted, lambda_list, solver, reg_norm)
        _, _, _, _, resultdf = transform_calc_trendlines(prices_sorted, lambda_list, solver, reg_norm)
        gresultdf = pd.concat([gresultdf,resultdf])

        prev_prices_sorted = prices_sorted[:-prev_dist]
        # prev_pricesd, prev_trendlinedl, prev_trendchangesdl, prev_aggtchangesl, prev_resultdf = transform_calc_trendlines(prev_prices_sorted, lambda_list, solver, reg_norm)
        _, _, _, _, prev_resultdf = transform_calc_trendlines(prev_prices_sorted, lambda_list, solver, reg_norm)
        prev_gresultdf = pd.concat([prev_gresultdf,prev_resultdf])

        num_runs = num_runs + 1

    gresultdf.to_csv(f"data\\gresult {datetime.datetime.now().strftime('%Y-%m-%d')}.csv")
    prev_gresultdf.to_csv(f"data\\prev_gresult {datetime.datetime.now().strftime('%Y-%m-%d')}.csv")

def generate_ggresult_from_data():
    #neueste sentimentDaten lesen
    f = glob.glob('data//sentiment_plotdf_*.csv')
    f.sort(reverse=True)
    print(f[0])
    sentiment_plotdf = pd.read_csv(f[0])

    #neueste gresult Daten lesen
    f = glob.glob('data//gresult*.csv')
    f.sort(reverse=True)
    print(f[0])
    gresultdf = pd.read_csv(f[0])

    #neueste gresult Daten lesen
    f = glob.glob('data//prev_gresult*.csv')
    f.sort(reverse=True)
    print(f[0])
    prev_gresultdf = pd.read_csv(f[0])

    # Ein ggresult für alles
    #gresultdf, prev_gresultdf, sentiment_plotdf in ein df
    # Index erstellen
    gresultdf = gresultdf.set_index(['symbol',pd.to_datetime(gresultdf['date']).apply(lambda r:r.date())], drop=False)
    prev_gresultdf = prev_gresultdf.set_index(['symbol',pd.to_datetime(prev_gresultdf['date']).apply(lambda r:r.date())], drop=False)
    # doppelte Spalten wegnehmen
    prev_gresultdf = prev_gresultdf.drop([colname for colname in prev_gresultdf.columns if not (('trend' in colname[0:5])  or ('agg_' in colname[0:5]) or ('updatedt' in colname))], axis=1)
    # rename columns
    prev_gresultdf.columns = ['prev_'+colname for colname in prev_gresultdf.columns if ('trend' in colname[0:5])  or ('agg_' in colname[0:5]) or ('updatedt' in colname)]

    #neues gresult zusammenstellen, Basis zum 
    gresultdf = pd.merge(gresultdf,prev_gresultdf, how='left', left_index=True, right_index=True) #, left_on=gresultdf.index, right_on=prev_gresultdf.index
    sentiment_plotdf = sentiment_plotdf.set_index(['yahoo_symbol',pd.to_datetime(sentiment_plotdf['publishDate']).apply(lambda r:r.date())], drop=False)
    sentiment_plotdf = sentiment_plotdf.rename_axis(['symbol','date'])
    gresultdf = pd.merge(gresultdf,sentiment_plotdf, how='left', left_index=True, right_index=True) # left_on=gresultdf.index, right_on=sentiment_plotdf.index

    gresultdf.to_csv(f"data\\ggresult {datetime.datetime.now().strftime('%Y-%m-%d')}.csv")

# prev_gresultdf aus den zusammengefassten Spalten in gresult extrahieren
def extract_prev_gresultdf_from_gresultdf(gresultdf):
    prev_gresultdf = gresultdf[['date', 'symbol', 'region', 'open', 'high', 'low',
            'close', 'volume', 'updatedt', 'close_start_1',
            'prev_trendline_lambda_1', 'prev_trendchanges_lambda_1',
            'prev_agg_orig_index_lambda_1', 'prev_agg_trendlinevalue_lambda_1',
            'prev_agg_diffdate_lambda_1', 'prev_agg_slope_until_lambda_1',
            'prev_agg_slopechange_lambda_1', 'prev_agg_buy_lambda_1',
            'prev_agg_sell_lambda_1']]
    prev_gresultdf.columns = [colname[5:] if ('prev_' in colname[0:5]) else colname for colname in prev_gresultdf.columns ]
    prev_gresultdf = prev_gresultdf.reset_index(drop=True)
    #prev_gresultdf = prev_gresultdf[:-5] # -5 ist eine Woche vorher, geht nicht
    prev_gresultdf = prev_gresultdf.dropna(subset=['trendline_lambda_1']) # die trendline geht nur bis eine Woche vorher
    return prev_gresultdf

def read_ggresult():
    #neueste gresult Daten lesen
    f = glob.glob('data//ggresult*.csv')
    f.sort(reverse=True)
    print(f[0])
    gresultdf = pd.read_csv(f[0])
    prev_gresultdf = extract_prev_gresultdf_from_gresultdf(gresultdf)
    return gresultdf, prev_gresultdf

def plot_sentiment(ax, symbolgresult, withNeutral=True):
    if not symbolgresult[['Negative','Neutral','Positive']].isna().all().all() == True > 0:
        ax_sentiment = ax.twinx()
        # Farben für die verschiedenen Sentiment-Labels definieren
        color_map = {
            'Positive': 'tab:green',
            'Neutral': 'tab:blue',
            'Negative': 'tab:red'
        }
        ax_sentiment.vlines(symbolgresult['Positive'].index, 0, symbolgresult['Positive'], color=color_map['Positive'], linewidth=1, linestyles='dashed')
        if withNeutral == True:
            ax_sentiment.vlines(symbolgresult['Neutral'].index, -0.5, 0.5 , color=color_map['Neutral'], linewidth=1, linestyles='dashed')
        ax_sentiment.vlines(symbolgresult['Negative'].index, 0, -symbolgresult['Negative'], color=color_map['Negative'], linewidth=1, linestyles='dashed')
        ax_sentiment.axhspan(0, 0, xmin=0, xmax=1,color='tab:grey')


def plotall(gresultdf, prev_gresultdf, lambda_list=[1], sharey=True, withNeutral=True):
    # Ab hier Darstellung
    # Darstellung neu mit gresult
    # creates a figure with multiple subplots in the form of a grid, where the number of rows is determined by the length of two lists: trendlinedl and trendlinedll. The grid's width ratio is set to 1:4
    numsymbols = len(list(gresultdf['symbol'].drop_duplicates()))
    numlambdas = len(lambda_list)
    #fig, ax = plt.subplots(len(trendlinedl)*len(trendlinedll),2,squeeze=False, figsize=(20,num_runs*5), gridspec_kw={'width_ratios': [1, 4]})
    fig, ax = plt.subplots(numlambdas*numsymbols,2,squeeze=False, figsize=(20,numsymbols*5), gridspec_kw={'width_ratios': [1, 4]})
    #fig.subplots_adjust(top=0.96)
    #fig.suptitle('Trendfilter L1 with lines(red=actual/orange=previous) and events')
    #fig.tight_layout()
    #ax = ax.ravel()
    #fig.autofmt_xdate()
    # A normalize function is created using matplotlib.colors.Normalize to set the minimum and maximum values for the color map. A color map cm.RdYlGn is created using the cm.ScalarMappable class.
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1, clip=True)
    #norm = matplotlib.colors.LogNorm(vmin=0, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn )

    # A nested loop is then used to plot the close prices, trend lines and trend changes on each subplot. 
    # The outer loop iterates through trendlinedl and trendlinedll, and the inner loop iterates through each element in the lists. 
    # For each iteration, the close prices and trend lines are plotted on the subplot. 
    # The x-axis is labeled with the minimum and maximum dates of the trend lines. 
    # The y-axis is labeled with the close price of the symbol.
    for j, symbol in enumerate(list(gresultdf['symbol'].drop_duplicates())):
        # j symbol iterator 
        symbolgresult = gresultdf[gresultdf['symbol']== symbol]
        prev_symbolgresult = prev_gresultdf[prev_gresultdf['symbol']== symbol]
        #prev_symbolgresult = prev_symbolgresult[:-5] # -5 ist eine Woche vorher, 

        symbolgresult.index = pd.to_datetime(symbolgresult['date']).apply(lambda r:r.date())
        prev_symbolgresult.index = pd.to_datetime(prev_symbolgresult['date']).apply(lambda r:r.date())

        # wegen Warnungen bei der folgenden Op 
        pd.options.mode.chained_assignment = None
        symbolgresult['date'] = pd.to_datetime(symbolgresult['date'])
        prev_symbolgresult['date'] = pd.to_datetime(prev_symbolgresult['date'])
        
        for i, lmbda in enumerate(lambda_list):
            trendlined = symbolgresult['trendline_lambda_1']
            prev_trendlined = prev_symbolgresult['trendline_lambda_1']
            #1208 prev_trendlined = symbolgresult['prev_trendline_lambda_1']

            trendchangesd = symbolgresult[symbolgresult['trendchanges_lambda_1'] == True]['trendchanges_lambda_1']
            prev_trendchangesd = prev_symbolgresult[prev_symbolgresult['trendchanges_lambda_1'] == True]['trendchanges_lambda_1']
            #1208 prev_trendchangesd = symbolgresult[symbolgresult['prev_trendline_lambda_1'] == True]['prev_trendchanges_lambda_1']
            #aggtchanges ersetzen mit date column
            aggtchanges = symbolgresult[pd.notnull(symbolgresult['agg_trendlinevalue_lambda_1'])]

            # i lambda iter
            row = (j+(i*numlambdas)+i)
            if row > 0:
                ax[row,1].sharex(ax[0,1])
                if sharey == True:
                    ax[row,1].sharey(ax[0,1])

                                                    
            ax[row,1].plot(symbolgresult['close_start_1'], linewidth=1.0, c='blue')
            ax[row,1].plot(prev_trendlined, '-', linewidth=1.0, c='orange', path_effects=[patheffects.withTickedStroke(spacing=7, angle=135)])
            ax[row,1].plot(trendlined, '-', linewidth=1.0, c='red')
            
            mindt = trendlined.index.min().strftime('%y-%m-%d')
            maxdt = trendlined.index.max().strftime('%y-%m-%d')
            ax[row,1].set_xlabel(f'Time with prices from {mindt} until {maxdt}')
            #ax[row,1].set_ylabel('Close Price ' + controllist['yahoo_symbol'][j] + ' with y[0]= 1')
            ax[row,1].set_ylabel('Close Price ' + symbol + ' with y[0]= 1')
            #ax[row,1].set_title(controllist['yahoo_symbol'][j] + ' with lambda=' + str(lambda_list[i]), loc='left')

            # calc Buy/Sell signals
            aggtrendchangeslabel = []
            #for index, s in aggtchanges[(aggtchanges['date'] != aggtchanges['date'].max()) & (aggtchanges['date'] != aggtchanges['date'].min())].iterrows():
            for index, s in aggtchanges.iterrows():
                # bis hierhin 2023-06-18
                #print(type(s['buy']) + type(index))
                buycolname = [name for name in list(aggtchanges.columns)  if 'buy' in name][0]
                sellcolname = [name for name in list(aggtchanges.columns)  if 'sell' in name][0]
                if s[buycolname]==True: 
                    aggtrendchangeslabel.append(ax[row,1].axvline(x=s['date'] , c='green'))
                elif s[sellcolname]==True: 
                    aggtrendchangeslabel.append(ax[row,1].axvline(x=pd.to_datetime(s['date']) , c='red'))
                else: #elif (s['date'] != aggtchanges['date'].min()) or (s['date'] != aggtchanges['date'].max()):
                    aggtrendchangeslabel.append(ax[row,1].axvline(x=pd.to_datetime(s['date']) , c='lightgrey'))

            #ax[row,1].set_title('Lambda: {}\nSolver: {}\nObjective Value: {}'.format(lambda_value, problem.status, round(objective.value, 3)))
            ax[row,1].xaxis.set_major_locator(mdates.MonthLocator())
            ax[row,1].xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1,5,10,15,20,25,30)))
            ax[row,1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))#('%b %d'))
            ax[row,1].legend(aggtrendchangeslabel, aggtchanges[(aggtchanges['date'] != aggtchanges['date'].max()) & (aggtchanges['date'] != aggtchanges['date'].min())]['date'].dt.strftime('%d-%m-%y'), loc ="upper left")
            
            if trendchangesd.shape[0] > 0 :
                new_trendchnga, diff_st_trend, st_trend, st_prev_trend = calc_slope_diff(trendchangesd, prev_trendchangesd, trendlined, prev_trendlined)
                ax[row,1].axvspan(trendchangesd.index.max(), trendlined.index.max(), facecolor=mapper.to_rgba(1000*st_trend) , alpha=0.2 )
                new_trendchngs = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in new_trendchnga]
            
            #sentiment


            plot_sentiment(ax[row,1], symbolgresult, withNeutral)

            cell_text = []
            cell_text.append(['Ticker:' , symbolgresult['symbol'].max()])# controllist['tradv_ticker'][j]])
            cell_text.append(['yahoo_symbol:' , symbolgresult['symbol'].max()]) #controllist['yahoo_symbol'][j]])
            cell_text.append(['lambda:' , str(lambda_list[i])])
            cell_text.append(['y[0]:' , '{:10.2f}'.format(symbolgresult['close'].iloc[0])])
            cell_text.append(['' , ''])
            cell_text.append(['new event(s):', str(new_trendchngs)])
            cell_text.append(['' , ''])
            cell_text.append(['timerange', pd.to_datetime(symbolgresult['date']).min().strftime('%Y-%m-%d')+ ' - ' + pd.to_datetime(symbolgresult['date']).max().strftime('%Y-%m-%d')])
            cell_text.append(['last_trend_slope:' , '{:10.4f}'.format(st_trend)])
            cell_text.append(['' , ''])
            cell_text.append(['prev_timerange', pd.to_datetime(prev_symbolgresult['date']).min().strftime('%Y-%m-%d')+ ' - ' + pd.to_datetime(prev_symbolgresult['date']).max().strftime('%Y-%m-%d')])

            cell_text.append(['last_prev_trend_sl:' , '{:10.4f}'.format(st_prev_trend)])
            cell_text.append(['diff_last_trends:', '{:10.4f}'.format(diff_st_trend)])
            

            the_table = ax[row,0].table(cellText=cell_text, loc='upper center', edges='open', cellLoc='left')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            #the_table.set_text_props(set_fontweight('light'))

            #the_table.scale(2,2)
            #the_table.visible_edges = 'B'
            ax[row,0].set_axis_off()
            ax[row,0].add_table(the_table)



import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# plot newstraces
def add_news_traces(publishDate, yl, text, colorstr, traces):
    traces.append(go.Scatter(x=[publishDate]*2, 
                            y=yl, 
                            mode='lines', 
                            hovertext=text.replace(';','<br>'),
                            showlegend=False,
                            line={"color":colorstr, 'width': 1 }
                            ))
    return traces
    
def generate_newstraces(symbol_gresultdf):
    # colors sentiment
    alpha = ',1)'
    red = 'rgba( 235, 51, 36'+ alpha
    green = 'rgba( 117, 249, 77'+ alpha
    blue = 'rgba( 0, 35, 245'+ alpha
    colorstr = blue

    yrange = (symbol_gresultdf['close_start_1'].max() -symbol_gresultdf['close_start_1'].min())*0.4 # 20% der gesamten yrange für news sentiment
    if symbol_gresultdf['Positive'].max() >= symbol_gresultdf['Negative'].max():
        stepmax = symbol_gresultdf['Positive'].max()
    else:
        stepmax = symbol_gresultdf['Negative'].max()
    step = (yrange/2)/stepmax  
    yrangemin = 1-(stepmax*step)
    yrangemax = 1+(stepmax*step)

    newsdf = symbol_gresultdf[['publishDate', 'Negative', 'Neutral', 'Positive','Neutral_text', 'Positive_text', 'Negative_text']].dropna()
    traces = []
    for idx, publishDate in enumerate(newsdf['publishDate']):
        if newsdf.iloc[idx,:]['Negative'] > 0:
            yl = [1 - (step * newsdf.iloc[idx,:]['Negative']), 1]
            traces = add_news_traces(publishDate, yl, newsdf.iloc[idx,:]['Negative_text'], red, traces)
        if newsdf.iloc[idx,:]['Neutral'] > 0:
            yl = [1 - step, 1 + step]
            traces = add_news_traces(publishDate, yl, newsdf.iloc[idx,:]['Neutral_text'], blue, traces)
        if newsdf.iloc[idx,:]['Positive'] > 0:
            yl = [1, 1 + (step * newsdf.iloc[idx,:]['Positive'])]
            traces = add_news_traces(publishDate, yl, newsdf.iloc[idx,:]['Positive_text'], green, traces)

    return traces

def plotallplotly(gresultdf, prev_gresultdf, lambda_list=[1], sharey=True, withNeutral=True):
    numsymbols = len(list(gresultdf['symbol'].drop_duplicates()))
    numlambdas = len(lambda_list)
    fig = go.Figure()
    # Create subplot for each symbol and lambda value
    fig = make_subplots(numsymbols*numlambdas, 1, shared_yaxes=True)
    gresultdf['date']=pd.to_datetime(gresultdf['date'])  # Change 'date' column to type date
    prev_gresultdf['date']=pd.to_datetime(prev_gresultdf['date'])  # Change 'date' column to type date

    for j, symbol in enumerate(list(gresultdf['symbol'].drop_duplicates())):
        for i, lambda_val in enumerate(lambda_list):
            symbol_gresultdf = gresultdf[gresultdf['symbol'] == symbol]
            symbol_prev_gresultdf = prev_gresultdf[prev_gresultdf['symbol'] == symbol]
            symbol_gresultdf = symbol_gresultdf.sort_values(by='date',ascending=True)
            symbol_gresultdf[f'agg_slope_until_lambda_{lambda_val}'] = symbol_gresultdf[f'agg_slope_until_lambda_{lambda_val}'].fillna(method='bfill')
            symbol_gresultdf[f'prev_agg_slope_until_lambda_{lambda_val}'] = symbol_gresultdf[f'prev_agg_slope_until_lambda_{lambda_val}'].fillna(method='bfill')

            row = (j+(i*numlambdas)+i) +1
            # Plot close prices
            fig.add_trace(go.Scatter(
                x=symbol_gresultdf['date'],
                y=symbol_gresultdf['close_start_1'],
                mode='lines',
                name='close',
                text='<br><b>Price:</b><br>'+ symbol_gresultdf['close'].apply(lambda x: '{:,.2f}'.format(x)).astype(str) + \
                    '<br>' + symbol_gresultdf['date'].dt.strftime("%Y-%m-%d") 
                    # '<br><b>Positive:</b><br>'+ symbol_gresultdf['Positive_text'].str.replace(';','<br>') + \
                    # '<br><b>Negative:</b><br>'+ symbol_gresultdf.Negative_text.str.replace(';','<br>')
                    , 
                hovertemplate = '%{text}',
                # '<br><b>X</b>: %{x}<br>'+ 
                # '<b>%{text}</b>',
                showlegend = False,
                line=dict(color="blue"),
            ), row=row, col=1)

            # Add a line every Month
            first_day_of_month = symbol_gresultdf[symbol_gresultdf['date'].dt.is_month_start]['date']
            fig.update_xaxes(
                tickvals=first_day_of_month,
                ticktext=first_day_of_month.dt.strftime("%Y-%m-%d"),
                row=row, col=1
            )  

            # Plot trend lines
            fig.add_trace(go.Scatter(
                x=symbol_gresultdf['date'],
                y=symbol_gresultdf[f'trendline_lambda_{lambda_val}'],
                mode='lines',
                name=f'trend (L{lambda_val})',
                text=symbol_gresultdf[f'agg_slope_until_lambda_{lambda_val}'],
                hovertemplate = '<i>Slope</i>: %{text:.2f}',
                #showlegend = False
                line=dict(color="red"),
            ),row=row, col=1)

            # Plot prev trend lines
            fig.add_trace(go.Scatter(
                x=symbol_gresultdf['date'],
                y=symbol_gresultdf[f'prev_trendline_lambda_{lambda_val}'],
                mode='lines',
                name=f'prev_trend(L{lambda_val})',
                text=symbol_gresultdf[f'prev_agg_slope_until_lambda_{lambda_val}'],
                hovertemplate = '<i>Slope</i>: %{text:.2f}',
                #showlegend = False,
                line=dict(color="orange"),
            ),row=row, col=1)

            # trendchanges
            trendchanges = symbol_gresultdf[symbol_gresultdf['trendchanges_lambda_1'] == True]

            # Plot markers trend changes
            fig.add_trace(go.Scatter(
                x=trendchanges['date'],
                y=trendchanges[f'agg_trendlinevalue_lambda_{lambda_val}'],
                mode='markers',
                name=f'trend chng (L{lambda_val})',
                text=trendchanges['date'].dt.strftime('%Y-%m-%d') +"<br>slope change: "+trendchanges['agg_slopechange_lambda_1'].apply(lambda x: '{:,.2f}'.format(x)).astype(str),
                hovertemplate = '%{text}',
                marker=dict(
                    color="green"
                ),
                showlegend = False,
            ),row=row, col=1)


            # Add vertical lines on trendchanges
            # Plot buy and sell signals
            for change in trendchanges.itertuples():
                if change.agg_buy_lambda_1 == True:
                    color="green"#"Red",  # or "Green"
                elif change.agg_sell_lambda_1 == True:
                    color="red"  # or "Green"
                else:
                    color="lightgrey"
                if color in ["green","red"]: # sonst dauert es zu lange
                    #fig.add_vline(x=change.date, line_dash='dash', line_color=color, annotation_text=f"{change.date.strftime('%Y-%m-%d')}", row=row, col=1)
                    fig.add_shape(
                        yref='y domain',
                        type="line",
                        x0=change.date,
                        y0=0,
                        x1=change.date,
                        y1=1,
                        line=dict(
                            color=color,
                            width=1,
                        ), 
                        row=row, col=1
                    )

                    fig.add_annotation(
                        yref='y domain',
                        x=change.date,
                        y=0.1,
                        text= f"{change.date.strftime('%Y-%m-%d')}", #<br>{(change.agg_slope_until_lambda_1 + change.prev_agg_slopechange_lambda_1):.4f}
                        showarrow=False,
                        font=dict(
                            size=12,
                            color="Black"
                        ),
                        # bgcolor="White",
                        opacity=0.8,
                        row=row, col=1
                    )

            # last trendchange
            #last_change = symbol_gresultdf.iloc[-1]
            # Set subplot titles
            fig.add_annotation(
                xref='x domain',
                yref='y domain',
                x=0.01,
                y=0.99,
                # get last slope
                # get last slope change
                text=f"Symbol: {symbol} \
                    last slope:{(symbol_gresultdf['agg_slope_until_lambda_1'].iloc[-1] + symbol_gresultdf['prev_agg_slope_until_lambda_1'].dropna().iloc[-1]):.4f} \
                    last slope change:{symbol_gresultdf['agg_slopechange_lambda_1'].dropna().iloc[-1]:.4f} ",
                showarrow=False,
                row=row, col=1)

            # add sentimenten traces
            # test if there are sentiment news
            if symbol_gresultdf[['publishDate', 'Negative', 'Neutral', 'Positive']].dropna().shape[0] > 0:
                newstraces = generate_newstraces(symbol_gresultdf)
                fig.add_traces(newstraces,rows=row, cols=1)
            
    # Set layout for the entire figure
    fig.update_layout(
        height=numsymbols * 300,
        width=1000,
        showlegend=False,
        title="Trendfilter L1 with trendlines and events"
    )
    #fig.update_layout(hovermode="x unified")

    # Show the interactive plot
    pio.write_html(fig, f"data\\output_{datetime.datetime.now().strftime('%Y-%m-%d')}.html")
    fig.show()
#rewrite plotallplotly to use ggresult
#use small images in a grid to show the trendlines and the sentiment


