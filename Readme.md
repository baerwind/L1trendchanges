# L1 Trend Changes

Ein Python-basiertes Tool zur Analyse von Finanzmarkttrends mittels L1-Trendline-Fitting und Sentiment-Analyse von Finanznachrichten.

## Übersicht

Dieses Projekt kombiniert technische Analyse mit Sentiment-Analyse, um Trendwechsel in Finanzmärkten zu identifizieren. Es nutzt:
- **L1-Trendline-Fitting** für robuste Trendanalyse
- **FinBERT** für Sentiment-Analyse von Finanznachrichten
- **yfinance** für Preisdaten
- **Plotly** für interaktive Visualisierungen

## Features

- Automatischer Download historischer Preisdaten von Yahoo Finance
- L1-Trendline-Berechnung zur Identifikation von Trendwechseln
- Sentiment-Analyse von Finanznachrichten für Top Holdings
- Interaktive Plotly-Visualisierungen mit:
  - Preischarts
  - Trendlinien
  - Buy/Sell-Signale
  - Sentiment-Scores
- Unterstützung für mehrere Assets (ETFs, Indizes, Aktien)
- Datenpersistenz in SQLite-Datenbanken

## Installation

### Voraussetzungen

- Python 3.8 oder höher
- pip

### Dependencies installieren

```bash
pip install -r requirements.txt
```

## Konfiguration

### Watchlist erstellen

Erstellen oder bearbeiten Sie die `watchlist.csv` Datei mit folgenden Spalten:

```csv
tradv_ticker;region;rapi_response_code;yahoo_symbol;group
ACWI;US;200;ACWI;40
URTH;US;200;URTH;40
NDX;US;200;^NDX;10
```

**Gruppen:**
- `10`: Indizes
- `20`: Sektoren
- `30`: Einzelaktien
- `40`: ETFs

## Verwendung

### Hauptprogramm ausführen

```bash
python main.py
```

### Workflow

Das Programm führt folgende Schritte automatisch aus:

1. **Preisdaten laden**
   ```python
   L1trendchanges.get_prices_toph(controllist)
   ```

2. **Trenddaten berechnen**
   ```python
   L1trendchanges.calc_l1trendchanges(controllist)
   ```

3. **News und Sentiment analysieren**
   ```python
   newslist_topholdings.newslist_topholdings(controllist)
   ```

4. **Daten zusammenführen**
   ```python
   L1trendchanges.generate_ggresult_from_data()
   ```

5. **Visualisierung erstellen**
   ```python
   L1trendchanges.plotallplotly(pgresultdf, prev_gresultdf, agg_tophdf)
   ```

## Projektstruktur

```
L1trendchanges/
│
├── main.py                      # Hauptprogramm
├── L1trendchanges.py           # Kernlogik für Trendanalyse
├── newslist_topholdings.py     # News-Fetching und Sentiment-Analyse
├── pricesdb.py                 # Preisdatenbank-Management
├── newsdb.py                   # News-Datenbank-Management
├── watchlist.csv               # Konfiguration der zu analysierenden Assets
├── requirements.txt            # Python-Dependencies
│
└── data/                       # Datenverzeichnis
    ├── *.db                    # SQLite-Datenbanken für Preise
    ├── seeking-alpha.db        # News-Datenbank
    ├── ggresult_*.csv          # Zusammengeführte Ergebnisse
    └── sentiment_*.csv         # Sentiment-Analysen
```

## Module

### L1trendchanges.py
- `get_prices_toph()`: Lädt Preisdaten und Summary-Informationen
- `calc_l1trendchanges()`: Berechnet L1-Trendlinien und identifiziert Trendwechsel
- `generate_ggresult_from_data()`: Führt Preis- und Sentimentdaten zusammen
- `plotallplotly()`: Erstellt interaktive Visualisierungen

### newslist_topholdings.py
- `get_topholdings_for_controllist()`: Extrahiert Top Holdings aus ETF-Daten
- `get_sentiments_finbert()`: Führt Sentiment-Analyse mit FinBERT durch
- `newslist_topholdings()`: Orchestriert News-Fetching und Analyse

### pricesdb.py
- `get_prices_from_db()`: Lädt Preisdaten aus SQLite
- `create_connection()`: Verwaltet Datenbankverbindungen
- `get_historical_data_from_yfinance()`: Wrapper für yfinance API

### newsdb.py
- `get_news_list_by_symbol()`: Holt Nachrichten über Seeking Alpha API
- `insert_list_by_symbol()`: Speichert Nachrichten in Datenbank
- `create_table_sentiment_title_list()`: Erstellt Sentiment-Tabellen

## Datenquellen

- **Preisdaten**: Yahoo Finance (via yfinance)
- **Nachrichten**: Seeking Alpha API
- **Sentiment-Modell**: FinBERT (yiyanghkust/finbert-tone)

## Troubleshooting

### Rate Limit Fehler bei yfinance

```bash
pip install yfinance --upgrade --no-cache-dir
```

### Abbruch bei News-Fetching

Wenn das Programm bei der News-Analyse abbricht, können Sie die Schritte manuell ausführen:

```python
# Nur Sentimente berechnen
file_newsdb = 'data/seeking-alpha.db'
newslist_topholdings.new_titles_sentimenten(file_newsdb)

# Sentiment aggregieren
import datetime
file_sentiment_result = f"data\\agg_top_sentiment_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
newslist_topholdings.agg_sentiment(file_sentiment_result, file_newsdb, agg_tophdf)
```

## Ausgabe

Das Programm generiert:
- **Interaktive HTML-Charts** mit Plotly
- **CSV-Dateien** mit Trenddaten und Signalen
- **SQLite-Datenbanken** mit historischen Daten

## Dependencies

Siehe [`requirements.txt`](requirements.txt ):
- cvxopt
- cvxpy
- matplotlib
- numpy
- pandas
- plotly
- requests
- scipy
- transformers
- yfinance

## Lizenz

Dieses Projekt ist für den persönlichen Gebrauch bestimmt.

## Hinweise

- Die Verwendung von Finanzmarktdaten unterliegt möglicherweise rechtlichen Einschränkungen
- Sentiment-Analyse basiert auf historischen Daten und ist keine Anlageberatung
- API-Rate-Limits können die Datenerfassung einschränken

## Autor

Hans Ulrich Baerwind

## Version

Stand: Oktober 2025