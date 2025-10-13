# %% [markdown]
# # Programm

# %%
import L1trendchanges
import pandas as pd
# für alle symbols in der controllist, prices aus DB holen und alle calculations produzieren

# Symbols aus Watchlist
file_watchlist = 'watchlist.csv'
watchlist = pd.read_csv(file_watchlist, sep=';')
groupl = [10,20,30,40] 
controllist = watchlist[watchlist['group'].isin(groupl)].reset_index()

# %% [markdown]
# ## Preisdaten und Summary holen

# %%
# Bei Rate Limit Fehler
# in (py38) C:\Users\baerw\Documents\Projekte\Gitlocal\L1trendchanges>
# pip install yfinance --upgrade --no-cache-dir

# %%
L1trendchanges.get_prices_toph(controllist)

# %%
# import importlib
# importlib.reload(L1trendchanges)


# %% [markdown]
# ## Trenddaten berechnen und abspeichern

# %%
#speichert in gresult, prevgresult csv Files
L1trendchanges.calc_l1trendchanges(controllist)

# %% [markdown]
# ## Newstitles zu den Topholdings holen, sentimenten und abspeichern

# %%
import newslist_topholdings
newslist_topholdings.newslist_topholdings(controllist)

# # %%
# # nur bei Abbruch in newslist_topholdings.newslist_topholdings(controllist)
# file_newsdb = 'data/seeking-alpha.db'
# newslist_topholdings.new_titles_sentimenten(file_newsdb)

# # %% [markdown]
# # ## Topholdings laden aus den gespeicherten Daten (sollte eigentlich schon da sein)  

# %%
_ , agg_tophdf = newslist_topholdings.get_topholdings_for_controllist(controllist)

# %%
# # nur bei Abbruch in newslist_topholdings.newslist_topholdings(controllist)
# import datetime
# file_sentiment_result=f"""data\\agg_top_sentiment_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"""
# file_sentiment_for_plot=f"""data\\sentiment_plotdf_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"""

# newslist_topholdings.agg_sentiment(file_sentiment_result,file_newsdb, agg_tophdf)

# %%
# # nur bei Abbruch in newslist_topholdings.newslist_topholdings(controllist)
# newslist_topholdings.generate_plotdf(file_sentiment_result,file_sentiment_for_plot)

# %% [markdown]
# ## Preis- und Newsdaten in ggresult zusammenführen

# %%
L1trendchanges.generate_ggresult_from_data()

# %%
gresultdf, prev_gresultdf = L1trendchanges.read_ggresult()

# # %%
# # nur bei Abbruch in newslist_topholdings.newslist_topholdings(controllist)
# controllist[controllist['group']==40]['yahoo_symbol']

# %% [markdown]
# ## Daten aus ggresult plotten

# %%
# plot nur ETFs
#pgresultdf = gresultdf[gresultdf['symbol'].isin(controllist[controllist['group']==40]['yahoo_symbol'])]
pgresultdf = gresultdf

# %%
L1trendchanges.plotallplotly(pgresultdf, prev_gresultdf, agg_tophdf,sharey=True, withNeutral=False)

