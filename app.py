"""
Indian Stock Screener using NSE XBRL file for Infosys

Direct fetching from NSE XBRL URLs is blocked or requires session cookies and browser headers. To reliably fetch data, we simulate a browser session using `requests.Session()` with proper headers. If network fetching fails, a local XML file fallback can be used.

Corporate filings from NSE: https://www.nseindia.com/companies-listing/corporate-filings-financial-results

Instructions to Run Locally:
1. Ensure Python is installed.
2. Install required packages:
   pip install streamlit pandas numpy xmltodict requests
3. Navigate to the directory containing this file.
4. Run the application:
   streamlit run <filename>.py
5. Open the browser at http://localhost:8501 to view the UI.
"""

from __future__ import annotations
import sys
import xmltodict
import requests
from typing import Dict, Any
import pandas as pd
import numpy as np

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# -----------------------------
# Fetch fundamentals from NSE XBRL URL using session headers
# -----------------------------

XBRL_URL_INFOSYS = "https://nsearchives.nseindia.com/corporate/xbrl/INDAS_104589_1099938_19042024112830.xml"

def fetch_fundamentals_from_xbrl(url: str) -> Dict[str, Any]:
    try:
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/xml',
            'Referer': 'https://www.nseindia.com/',
        }
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = xmltodict.parse(response.content)
        facts = data.get('xbrl', {}).get('facts', {})
        return {
            'symbol': 'INFY',
            'longName': 'Infosys Limited',
            'marketCap': float(facts.get('MarketCapitalization', 0)),
            'pe': float(facts.get('PERatio', 0)),
            'pb': float(facts.get('PriceToBookRatio', 0)),
            'roe': float(facts.get('ReturnOnEquity', 0)),
            'debtToEquity': float(facts.get('DebtToEquity', 0)),
            'revenueGrowth': float(facts.get('Revenue', 0)) / 1e9,
            'epsGrowth': float(facts.get('EPS', 0)),
        }
    except Exception as e:
        print(f"Error fetching XBRL data from URL: {e}")
        return {
            'symbol': 'INFY',
            'longName': 'Infosys Limited',
            'marketCap': None,
            'pe': None,
            'pb': None,
            'roe': None,
            'debtToEquity': None,
            'revenueGrowth': None,
            'epsGrowth': None,
        }

# -----------------------------
# Sample symbols for NSE companies (only Infosys for demo)
# -----------------------------

def yf_fetch_symbols() -> pd.DataFrame:
    sample = [
        {"symbol": "INFY", "name": "Infosys Limited", "exchange": "NSE"},
    ]
    return pd.DataFrame(sample)

# -----------------------------
# Scoring functions
# -----------------------------

def compute_value_score(row: Dict[str, Any]) -> float:
    pe = row.get("pe")
    pb = row.get("pb")
    mcap = row.get("marketCap") or 0
    score = 0.0
    if pe and pe > 0:
        score += max(0.0, 2.0 - np.log1p(pe))
    if pb and pb > 0:
        score += max(0.0, 1.5 - np.log1p(pb))
    if mcap and mcap > 0:
        score += np.log1p(mcap) / 1e3
    return float(score)

def compute_growth_score(row: Dict[str, Any]) -> float:
    rg = row.get("revenueGrowth")
    eg = row.get("epsGrowth")
    score = 0.0
    if rg is not None:
        score += float(rg) * 5.0
    if eg is not None:
        score += float(eg) * 5.0
    return float(score)

def compute_financial_health_score(row: Dict[str, Any]) -> float:
    roe = row.get("roe")
    dte = row.get("debtToEquity")
    score = 0.0
    if roe is not None:
        score += min(10.0, float(roe) * 20.0)
    if dte is not None:
        score += max(0.0, 5.0 - (float(dte) / 2.0))
    return float(score)

# -----------------------------
# Enrichment
# -----------------------------

def enrich_universe(df_symbols: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_symbols.iterrows():
        symbol = r["symbol"]
        fund = fetch_fundamentals_from_xbrl(XBRL_URL_INFOSYS)
        row = {**r.to_dict(), **fund}
        for k in ["marketCap", "pe", "pb", "roe", "debtToEquity", "revenueGrowth", "epsGrowth"]:
            if k in row:
                try:
                    row[k] = None if row[k] is None else float(row[k])
                except Exception:
                    row[k] = None
        row["value_score"] = compute_value_score(row)
        row["growth_score"] = compute_growth_score(row)
        row["health_score"] = compute_financial_health_score(row)
        rows.append(row)
    return pd.DataFrame(rows)

# -----------------------------
# Streamlit App
# -----------------------------

def streamlit_app():
    st.set_page_config(layout="wide", page_title="Indian Stock Screener (NSE XBRL)")
    st.title("Indian Stock Screener — NSE XBRL Data")
    df_symbols = yf_fetch_symbols()
    min_mcap = st.sidebar.number_input("Min market cap", value=0.0)
    max_pe = st.sidebar.number_input("Max P/E", value=30.0)
    if st.button("Run Screener"):
        df = enrich_universe(df_symbols)
        cond = pd.Series([True] * len(df))
        cond &= df["marketCap"].fillna(0) >= min_mcap
        cond &= df["pe"].fillna(9999) <= max_pe
        df_filtered = df[cond].copy()
        df_filtered["composite"] = df_filtered["value_score"] + df_filtered["growth_score"] + df_filtered["health_score"]
        st.dataframe(df_filtered.sort_values("composite", ascending=False))

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        streamlit_app()
    else:
        print("CLI mode")
        df_symbols = yf_fetch_symbols()
        df = enrich_universe(df_symbols)
        print(df.head())
