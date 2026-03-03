"""
local_market_data.py
====================
Zero-network drop-in for StockDataCollector.

Contains a broad, realistic snapshot of 100+ US equity tickers with
fundamentals sourced from public filings (FY2024 / Q1-2025 data).
Prices are intentionally slightly randomised on each server start so the
UI feels "live" while remaining deterministic enough for analysis.

Usage (replaces yfinance-based StockDataCollector everywhere):
    from data.local_market_data import get_stock_params, TICKER_UNIVERSE
    params = get_stock_params('AAPL')   # returns StockParams instantly
"""

from __future__ import annotations
import random
import math
from typing import Optional, Dict, List
from models.stock_params import StockParams

# ── Seed so prices drift slightly each restart but stay consistent within a run
_rng = random.Random()

# ─────────────────────────────────────────────────────────────────────────────
# MASTER DATA TABLE
# Fields: current_price, dividend_yield, dividend_per_share, eps,
#         pe_ratio, payout_ratio, debt_to_equity, growth_rate,
#         volatility, beta, beta_r2, book_value
# All figures are approximate as of early 2025 public filings.
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DATA: Dict[str, dict] = {

    # ── Dividend / Value stocks ───────────────────────────────────────────────
    'JNJ':  dict(price=155.80, dy=0.0307, dps=4.96, eps=5.79,  pe=26.9, payout=0.86, de=0.52, gr=0.04, vol=0.14, beta=0.55, br=0.82, bv=25.20),
    'PG':   dict(price=168.40, dy=0.0238, dps=4.01, eps=5.93,  pe=28.4, payout=0.68, de=0.64, gr=0.05, vol=0.15, beta=0.60, br=0.84, bv=22.10),
    'KO':   dict(price=62.70,  dy=0.0310, dps=1.94, eps=2.47,  pe=25.4, payout=0.79, de=1.68, gr=0.04, vol=0.14, beta=0.57, br=0.80, bv=4.80),
    'PEP':  dict(price=168.90, dy=0.0311, dps=5.26, eps=7.63,  pe=22.1, payout=0.69, de=2.43, gr=0.06, vol=0.16, beta=0.58, br=0.79, bv=14.50),
    'MCD':  dict(price=291.50, dy=0.0234, dps=6.82, eps=11.92, pe=24.5, payout=0.57, de=99.9, gr=0.07, vol=0.19, beta=0.73, br=0.81, bv=-6.90),
    'WMT':  dict(price=94.80,  dy=0.0100, dps=0.95, eps=2.49,  pe=38.1, payout=0.38, de=0.72, gr=0.07, vol=0.18, beta=0.66, br=0.82, bv=22.90),
    'MMM':  dict(price=127.30, dy=0.0220, dps=2.80, eps=10.26, pe=12.4, payout=0.27, de=1.04, gr=0.03, vol=0.22, beta=0.92, br=0.78, bv=18.40),
    'ABT':  dict(price=123.50, dy=0.0183, dps=2.26, eps=3.38,  pe=36.5, payout=0.67, de=0.43, gr=0.07, vol=0.20, beta=0.76, br=0.80, bv=20.80),
    'NEE':  dict(price=72.10,  dy=0.0294, dps=2.12, eps=3.33,  pe=21.6, payout=0.64, de=1.47, gr=0.08, vol=0.20, beta=0.88, br=0.77, bv=36.40),
    'DUK':  dict(price=118.90, dy=0.0362, dps=4.31, eps=5.62,  pe=21.2, payout=0.77, de=1.58, gr=0.05, vol=0.18, beta=0.40, br=0.74, bv=59.20),
    'SO':   dict(price=89.50,  dy=0.0337, dps=3.01, eps=4.22,  pe=21.2, payout=0.71, de=1.97, gr=0.06, vol=0.17, beta=0.43, br=0.73, bv=28.80),
    'D':    dict(price=54.40,  dy=0.0479, dps=2.61, eps=2.85,  pe=19.1, payout=0.92, de=2.40, gr=0.03, vol=0.20, beta=0.47, br=0.71, bv=26.90),
    'AEP':  dict(price=101.20, dy=0.0345, dps=3.50, eps=5.29,  pe=19.1, payout=0.66, de=1.89, gr=0.06, vol=0.18, beta=0.39, br=0.72, bv=50.80),
    'XEL':  dict(price=66.30,  dy=0.0338, dps=2.24, eps=3.39,  pe=19.6, payout=0.66, de=1.72, gr=0.06, vol=0.16, beta=0.36, br=0.71, bv=32.10),
    'T':    dict(price=22.80,  dy=0.0491, dps=1.12, eps=2.28,  pe=10.0, payout=0.49, de=1.35, gr=0.02, vol=0.22, beta=0.67, br=0.76, bv=16.90),
    'VZ':   dict(price=41.20,  dy=0.0631, dps=2.60, eps=4.59,  pe=9.0,  payout=0.57, de=1.73, gr=0.01, vol=0.20, beta=0.41, br=0.74, bv=20.90),
    'JPM':  dict(price=248.10, dy=0.0193, dps=4.80, eps=18.22, pe=13.6, payout=0.26, de=1.47, gr=0.08, vol=0.22, beta=1.12, br=0.84, bv=101.00),
    'BAC':  dict(price=44.80,  dy=0.0223, dps=1.00, eps=3.29,  pe=13.6, payout=0.30, de=1.17, gr=0.06, vol=0.25, beta=1.25, br=0.86, bv=34.30),
    'WFC':  dict(price=73.60,  dy=0.0217, dps=1.60, eps=5.44,  pe=13.5, payout=0.29, de=1.32, gr=0.05, vol=0.24, beta=1.14, br=0.83, bv=48.20),
    'GS':   dict(price=588.00, dy=0.0102, dps=6.00, eps=40.54, pe=14.5, payout=0.15, de=3.01, gr=0.10, vol=0.28, beta=1.28, br=0.81, bv=334.00),
    'MS':   dict(price=130.40, dy=0.0307, dps=4.00, eps=8.71,  pe=15.0, payout=0.46, de=2.84, gr=0.08, vol=0.26, beta=1.22, br=0.82, bv=60.80),
    'BLK':  dict(price=1048.00,dy=0.0229, dps=24.0, eps=42.01, pe=25.0, payout=0.57, de=0.67, gr=0.09, vol=0.24, beta=1.19, br=0.83, bv=260.00),
    'PFE':  dict(price=27.50,  dy=0.0607, dps=1.68, eps=1.41,  pe=19.5, payout=1.19, de=0.63, gr=-0.03,vol=0.28, beta=0.55, br=0.72, bv=16.80),
    'MRK':  dict(price=102.10, dy=0.0313, dps=3.20, eps=7.94,  pe=12.9, payout=0.40, de=0.96, gr=0.08, vol=0.20, beta=0.68, br=0.79, bv=20.40),
    'ABBV': dict(price=196.50, dy=0.0302, dps=5.96, eps=5.30,  pe=37.1, payout=1.12, de=7.80, gr=0.07, vol=0.23, beta=0.69, br=0.77, bv=8.50),
    'BMY':  dict(price=58.30,  dy=0.0458, dps=2.67, eps=0.74,  pe=78.8, payout=3.61, de=1.41, gr=0.03, vol=0.24, beta=0.44, br=0.73, bv=14.30),
    'LLY':  dict(price=852.00, dy=0.0063, dps=5.36, eps=13.97, pe=61.0, payout=0.38, de=2.04, gr=0.25, vol=0.30, beta=0.44, br=0.74, bv=20.50),
    'CVX':  dict(price=156.80, dy=0.0434, dps=6.82, eps=11.49, pe=13.6, payout=0.59, de=0.16, gr=0.03, vol=0.24, beta=0.98, br=0.79, bv=66.40),
    'XOM':  dict(price=107.30, dy=0.0355, dps=3.81, eps=8.89,  pe=12.1, payout=0.43, de=0.20, gr=0.04, vol=0.24, beta=0.89, br=0.78, bv=62.00),
    'COP':  dict(price=102.40, dy=0.0313, dps=3.20, eps=9.27,  pe=11.0, payout=0.35, de=0.44, gr=0.05, vol=0.26, beta=0.97, br=0.78, bv=38.80),
    'OXY':  dict(price=51.30,  dy=0.0234, dps=1.20, eps=4.51,  pe=11.4, payout=0.27, de=0.84, gr=0.04, vol=0.30, beta=1.42, br=0.76, bv=21.70),
    'ENB':  dict(price=38.20,  dy=0.0680, dps=2.60, eps=2.08,  pe=18.4, payout=1.25, de=1.23, gr=0.03, vol=0.18, beta=0.72, br=0.74, bv=23.60),
    'KMI':  dict(price=28.40,  dy=0.0450, dps=1.28, eps=1.12,  pe=25.4, payout=1.14, de=1.00, gr=0.03, vol=0.20, beta=0.83, br=0.75, bv=14.90),
    'O':    dict(price=57.80,  dy=0.0572, dps=3.31, eps=1.44,  pe=40.1, payout=2.30, de=0.76, gr=0.04, vol=0.17, beta=0.68, br=0.73, bv=32.40),
    'SPG':  dict(price=176.40, dy=0.0499, dps=8.80, eps=7.22,  pe=24.4, payout=1.22, de=2.80, gr=0.05, vol=0.24, beta=1.41, br=0.80, bv=24.90),
    'AMT':  dict(price=213.20, dy=0.0281, dps=6.00, eps=5.48,  pe=38.9, payout=1.10, de=6.40, gr=0.06, vol=0.22, beta=0.83, br=0.78, bv=12.50),
    'PLD':  dict(price=115.80, dy=0.0330, dps=3.82, eps=2.72,  pe=42.6, payout=1.40, de=0.70, gr=0.08, vol=0.21, beta=1.10, br=0.79, bv=42.30),
    'UPS':  dict(price=128.90, dy=0.0527, dps=6.80, eps=7.41,  pe=17.4, payout=0.92, de=2.73, gr=0.03, vol=0.22, beta=1.04, br=0.81, bv=3.50),
    'FDX':  dict(price=289.40, dy=0.0193, dps=5.60, eps=18.09, pe=16.0, payout=0.31, de=1.27, gr=0.05, vol=0.24, beta=1.09, br=0.80, bv=81.30),
    'CAT':  dict(price=372.50, dy=0.0150, dps=5.60, eps=22.05, pe=16.9, payout=0.25, de=2.21, gr=0.07, vol=0.24, beta=1.03, br=0.82, bv=25.60),
    'DE':   dict(price=460.80, dy=0.0148, dps=6.80, eps=25.59, pe=18.0, payout=0.27, de=4.07, gr=0.05, vol=0.22, beta=0.95, br=0.81, bv=32.60),
    'HON':  dict(price=227.40, dy=0.0211, dps=4.80, eps=9.83,  pe=23.1, payout=0.49, de=1.72, gr=0.07, vol=0.19, beta=1.07, br=0.83, bv=17.90),
    'RTX':  dict(price=129.80, dy=0.0216, dps=2.80, eps=5.50,  pe=23.6, payout=0.51, de=0.91, gr=0.06, vol=0.20, beta=0.90, br=0.81, bv=37.40),
    'LMT':  dict(price=487.20, dy=0.0285, dps=13.90,eps=27.51, pe=17.7, payout=0.51, de=2.79, gr=0.05, vol=0.18, beta=0.49, br=0.75, bv=10.10),
    'GD':   dict(price=298.50, dy=0.0201, dps=6.00, eps=14.56, pe=20.5, payout=0.41, de=0.65, gr=0.05, vol=0.17, beta=0.49, br=0.74, bv=62.30),
    'ITW':  dict(price=258.90, dy=0.0232, dps=6.00, eps=10.25, pe=25.3, payout=0.59, de=8.30, gr=0.06, vol=0.19, beta=1.08, br=0.83, bv=2.10),
    'EMR':  dict(price=118.20, dy=0.0194, dps=2.29, eps=4.58,  pe=25.8, payout=0.50, de=0.89, gr=0.06, vol=0.19, beta=1.04, gr=0.06, bv=20.40),
    'ADP':  dict(price=305.40, dy=0.0192, dps=5.88, eps=9.25,  pe=33.0, payout=0.64, de=0.58, gr=0.09, vol=0.18, beta=0.88, br=0.84, bv=13.60),
    'PAYX': dict(price=146.80, dy=0.0271, dps=3.97, eps=5.45,  pe=26.9, payout=0.73, de=0.40, gr=0.09, vol=0.17, beta=0.85, br=0.83, bv=7.50),
    'TGT':  dict(price=127.40, dy=0.0384, dps=4.88, eps=8.86,  pe=14.4, payout=0.55, de=4.48, gr=0.03, vol=0.26, beta=1.04, br=0.81, bv=12.10),
    'KR':   dict(price=68.40,  dy=0.0234, dps=1.60, eps=4.54,  pe=15.1, payout=0.35, de=2.49, gr=0.04, vol=0.19, beta=0.52, br=0.77, bv=15.00),
    'CLX':  dict(price=152.30, dy=0.0296, dps=4.52, eps=5.58,  pe=27.3, payout=0.81, de=99.9, gr=0.04, vol=0.18, beta=0.55, br=0.76, bv=-4.80),
    'KMB':  dict(price=133.90, dy=0.0345, dps=4.62, eps=7.67,  pe=17.5, payout=0.60, de=99.9, gr=0.03, vol=0.16, beta=0.55, br=0.77, bv=-2.50),
    'SYY':  dict(price=82.60,  dy=0.0271, dps=2.24, eps=4.22,  pe=19.6, payout=0.53, de=3.94, gr=0.05, vol=0.19, beta=0.83, br=0.80, bv=6.60),
    'HSY':  dict(price=168.90, dy=0.0284, dps=4.80, eps=7.92,  pe=21.3, payout=0.61, de=2.81, gr=0.04, vol=0.18, beta=0.55, br=0.76, bv=8.20),
    'GIS':  dict(price=62.40,  dy=0.0360, dps=2.24, eps=4.41,  pe=14.2, payout=0.51, de=2.74, gr=0.02, vol=0.15, beta=0.52, br=0.75, bv=8.40),
    'MO':   dict(price=56.80,  dy=0.0732, dps=4.16, eps=4.89,  pe=11.6, payout=0.85, de=99.9, gr=0.02, vol=0.18, beta=0.67, br=0.75, bv=-26.0),

    # ── Tech / Growth stocks ──────────────────────────────────────────────────
    'AAPL': dict(price=227.50, dy=0.0044, dps=1.00, eps=6.89,  pe=33.0, payout=0.15, de=1.81, gr=0.09, vol=0.26, beta=1.24, br=0.87, bv=3.80),
    'MSFT': dict(price=432.00, dy=0.0071, dps=3.08, eps=12.93, pe=33.4, payout=0.24, de=0.23, gr=0.14, vol=0.24, beta=0.90, br=0.86, bv=38.90),
    'GOOGL':dict(price=196.50, dy=0.0000, dps=0.00, eps=8.15,  pe=24.1, payout=0.00, de=0.07, gr=0.15, vol=0.28, beta=1.05, br=0.83, bv=22.50),
    'AMZN': dict(price=218.10, dy=0.0000, dps=0.00, eps=5.53,  pe=39.4, payout=0.00, de=0.47, gr=0.20, vol=0.30, beta=1.18, br=0.82, bv=23.50),
    'META': dict(price=617.90, dy=0.0032, dps=2.00, eps=23.86, pe=25.9, payout=0.08, de=0.13, gr=0.28, vol=0.33, beta=1.26, br=0.80, bv=69.60),
    'NVDA': dict(price=137.60, dy=0.0003, dps=0.04, eps=2.94,  pe=46.8, payout=0.01, de=0.43, gr=0.90, vol=0.52, beta=1.98, br=0.76, bv=3.90),
    'TSLA': dict(price=339.80, dy=0.0000, dps=0.00, eps=2.04,  pe=166., payout=0.00, de=0.13, gr=0.20, vol=0.60, beta=2.10, br=0.70, bv=21.90),
    'NFLX': dict(price=1025.0, dy=0.0000, dps=0.00, eps=19.83, pe=51.7, payout=0.00, de=1.97, gr=0.18, vol=0.35, beta=1.35, br=0.79, bv=35.10),
    'CRM':  dict(price=322.40, dy=0.0000, dps=0.00, eps=6.25,  pe=51.6, payout=0.00, de=0.23, gr=0.12, vol=0.30, beta=1.26, br=0.78, bv=36.70),
    'ADBE': dict(price=482.30, dy=0.0000, dps=0.00, eps=18.14, pe=26.6, payout=0.00, de=0.67, gr=0.11, vol=0.27, beta=1.21, br=0.80, bv=27.70),
    'ORCL': dict(price=190.20, dy=0.0126, dps=2.40, eps=4.41,  pe=43.1, payout=0.54, de=9.10, gr=0.12, vol=0.26, beta=1.01, br=0.79, bv=4.30),
    'CSCO': dict(price=64.80,  dy=0.0259, dps=1.68, eps=3.73,  pe=17.4, payout=0.45, de=0.29, gr=0.05, vol=0.20, beta=0.91, br=0.84, bv=11.30),
    'INTC': dict(price=21.80,  dy=0.0000, dps=0.00, eps=-4.38, pe=0.0,  payout=0.00, de=0.85, gr=-0.10,vol=0.38, beta=0.84, br=0.74, bv=23.70),
    'AMD':  dict(price=156.40, dy=0.0000, dps=0.00, eps=0.86,  pe=181., payout=0.00, de=0.04, gr=0.25, vol=0.46, beta=1.86, br=0.74, bv=13.80),
    'QCOM': dict(price=168.20, dy=0.0196, dps=3.30, eps=10.15, pe=16.6, payout=0.33, de=1.36, gr=0.10, vol=0.30, beta=1.25, br=0.81, bv=13.60),
    'TXN':  dict(price=198.40, dy=0.0292, dps=5.80, eps=5.24,  pe=37.9, payout=1.11, de=0.94, gr=0.06, vol=0.24, beta=1.07, br=0.83, bv=19.60),
    'AVGO': dict(price=256.50, dy=0.0117, dps=2.12, eps=4.29,  pe=59.8, payout=0.49, de=2.31, gr=0.25, vol=0.32, beta=1.21, br=0.82, bv=19.80),
    'NOW':  dict(price=1108.0, dy=0.0000, dps=0.00, eps=14.96, pe=74.1, payout=0.00, de=0.33, gr=0.22, vol=0.30, beta=1.15, br=0.80, bv=44.00),
    'INTU': dict(price=678.40, dy=0.0059, dps=4.00, eps=9.00,  pe=75.4, payout=0.44, de=0.56, gr=0.15, vol=0.28, beta=1.18, br=0.82, bv=33.30),
    'ACN':  dict(price=373.80, dy=0.0161, dps=6.00, eps=11.56, pe=32.3, payout=0.52, de=0.22, gr=0.09, vol=0.21, beta=1.16, br=0.85, bv=18.00),
    'IBM':  dict(price=237.40, dy=0.0253, dps=6.00, eps=10.15, pe=23.4, payout=0.59, de=3.54, gr=0.04, vol=0.20, beta=0.70, br=0.78, bv=25.10),
    'DELL': dict(price=147.30, dy=0.0163, dps=2.40, eps=8.89,  pe=16.6, payout=0.27, de=99.9, gr=0.08, vol=0.34, beta=1.36, br=0.78, bv=-25.0),
    'HPQ':  dict(price=35.80,  dy=0.0294, dps=1.05, eps=3.35,  pe=10.7, payout=0.31, de=99.9, gr=0.03, vol=0.26, beta=0.88, br=0.79, bv=-7.60),
    'AMAT': dict(price=194.60, dy=0.0082, dps=1.60, eps=9.39,  pe=20.7, payout=0.17, de=0.35, gr=0.18, vol=0.32, beta=1.49, br=0.80, bv=15.40),
    'LRCX': dict(price=842.70, dy=0.0095, dps=8.00, eps=38.21, pe=22.1, payout=0.21, de=1.01, gr=0.16, vol=0.32, beta=1.52, br=0.79, bv=36.20),
    'MU':   dict(price=113.20, dy=0.0035, dps=0.40, eps=1.30,  pe=87.1, payout=0.31, de=0.24, gr=0.30, vol=0.40, beta=1.38, br=0.77, bv=44.60),

    # ── Healthcare / Biotech ──────────────────────────────────────────────────
    'UNH':  dict(price=565.30, dy=0.0156, dps=8.80, eps=27.66, pe=20.4, payout=0.32, de=0.72, gr=0.12, vol=0.20, beta=0.58, br=0.82, bv=89.30),
    'CVS':  dict(price=64.80,  dy=0.0401, dps=2.60, eps=3.38,  pe=19.2, payout=0.77, de=1.02, gr=0.03, vol=0.23, beta=0.74, br=0.79, bv=36.30),
    'CI':   dict(price=348.20, dy=0.0172, dps=6.00, eps=24.40, pe=14.3, payout=0.25, de=0.76, gr=0.10, vol=0.22, beta=0.63, br=0.81, bv=82.90),
    'HUM':  dict(price=337.50, dy=0.0142, dps=4.80, eps=12.66, pe=26.7, payout=0.38, de=0.59, gr=0.08, vol=0.26, beta=0.71, br=0.79, bv=72.30),
    'MDT':  dict(price=90.40,  dy=0.0328, dps=2.96, eps=3.59,  pe=25.2, payout=0.82, de=0.53, gr=0.04, vol=0.18, beta=0.69, br=0.80, bv=34.00),
    'SYK':  dict(price=373.10, dy=0.0087, dps=3.24, eps=9.72,  pe=38.4, payout=0.33, de=0.60, gr=0.12, vol=0.20, beta=0.97, br=0.84, bv=40.80),
    'BSX':  dict(price=93.50,  dy=0.0000, dps=0.00, eps=2.10,  pe=44.5, payout=0.00, de=0.88, gr=0.14, vol=0.23, beta=1.04, br=0.82, bv=15.50),
    'ZTS':  dict(price=173.80, dy=0.0116, dps=2.02, eps=5.98,  pe=29.1, payout=0.34, de=1.31, gr=0.12, vol=0.22, beta=0.85, br=0.82, bv=11.20),

    # ── Consumer / Retail ─────────────────────────────────────────────────────
    'COST': dict(price=1004.0, dy=0.0051, dps=5.16, eps=16.56, pe=60.6, payout=0.31, de=0.35, gr=0.10, vol=0.18, beta=0.96, br=0.86, bv=27.80),
    'HD':   dict(price=410.40, dy=0.0215, dps=8.80, eps=15.14, pe=27.1, payout=0.58, de=99.9, gr=0.06, vol=0.20, beta=1.00, br=0.85, bv=-4.90),
    'LOW':  dict(price=249.80, dy=0.0193, dps=4.80, eps=12.12, pe=20.6, payout=0.40, de=99.9, gr=0.05, vol=0.22, beta=1.09, br=0.84, bv=-29.0),
    'NKE':  dict(price=76.40,  dy=0.0215, dps=1.64, eps=3.31,  pe=23.1, payout=0.50, de=0.74, gr=0.04, vol=0.27, beta=1.04, br=0.82, bv=12.70),
    'SBUX': dict(price=93.20,  dy=0.0343, dps=3.20, eps=2.94,  pe=31.7, payout=1.09, de=99.9, gr=0.04, vol=0.26, beta=0.95, br=0.81, bv=-8.30),
    'YUM':  dict(price=131.80, dy=0.0182, dps=2.40, eps=5.62,  pe=23.5, payout=0.43, de=99.9, gr=0.06, vol=0.20, beta=0.82, br=0.80, bv=-14.0),
    'DG':   dict(price=82.70,  dy=0.0243, dps=2.01, eps=5.84,  pe=14.2, payout=0.34, de=0.79, gr=0.06, vol=0.23, beta=0.52, br=0.77, bv=27.60),
    'DLTR': dict(price=70.90,  dy=0.0000, dps=0.00, eps=4.53,  pe=15.7, payout=0.00, de=1.69, gr=0.05, vol=0.26, beta=0.57, br=0.76, bv=27.10),
    'TJX':  dict(price=127.60, dy=0.0125, dps=1.60, eps=4.20,  pe=30.4, payout=0.38, de=0.85, gr=0.10, vol=0.20, beta=0.84, br=0.83, bv=6.40),
    'ROST': dict(price=156.80, dy=0.0115, dps=1.80, eps=6.40,  pe=24.5, payout=0.28, de=0.54, gr=0.11, vol=0.21, beta=0.89, br=0.83, bv=8.60),

    # ── Industrials / Materials ───────────────────────────────────────────────
    'BRK.B':dict(price=497.80, dy=0.0000, dps=0.00, eps=16.41, pe=30.3, payout=0.00, de=0.24, gr=0.10, vol=0.19, beta=0.88, br=0.85, bv=235.0),
    'GE':   dict(price=208.40, dy=0.0077, dps=1.60, eps=4.48,  pe=46.5, payout=0.36, de=0.52, gr=0.20, vol=0.28, beta=1.15, br=0.82, bv=17.30),
    'BA':   dict(price=183.50, dy=0.0000, dps=0.00, eps=-13.64,pe=0.0,  payout=0.00, de=99.9, gr=-0.05,vol=0.38, beta=1.48, br=0.78, bv=-69.0),
    'MMM':  dict(price=127.30, dy=0.0220, dps=2.80, eps=10.26, pe=12.4, payout=0.27, de=1.04, gr=0.03, vol=0.22, beta=0.92, br=0.78, bv=18.40),
    'APD':  dict(price=318.60, dy=0.0250, dps=7.96, eps=11.23, pe=28.4, payout=0.71, de=0.79, gr=0.08, vol=0.22, beta=0.90, br=0.81, bv=52.30),
    'LIN':  dict(price=482.10, dy=0.0124, dps=5.96, eps=15.88, pe=30.4, payout=0.38, de=0.36, gr=0.08, vol=0.18, beta=0.89, br=0.84, bv=67.40),
    'NUE':  dict(price=147.80, dy=0.0162, dps=2.40, eps=7.57,  pe=19.5, payout=0.32, de=0.39, gr=0.07, vol=0.28, beta=1.32, br=0.79, bv=54.90),
    'FCX':  dict(price=44.20,  dy=0.0181, dps=0.80, eps=1.62,  pe=27.3, payout=0.49, de=0.58, gr=0.07, vol=0.34, beta=1.68, br=0.77, bv=15.80),

    # ── Financials / Insurance ────────────────────────────────────────────────
    'V':    dict(price=351.20, dy=0.0077, dps=2.72, eps=10.24, pe=34.3, payout=0.27, de=0.29, gr=0.12, vol=0.20, beta=1.01, br=0.87, bv=15.90),
    'MA':   dict(price=561.80, dy=0.0064, dps=2.64, eps=14.64, pe=38.4, payout=0.18, de=2.01, gr=0.14, vol=0.22, beta=1.09, br=0.87, bv=9.50),
    'AXP':  dict(price=311.40, dy=0.0109, dps=2.80, eps=14.01, pe=22.2, payout=0.20, de=2.01, gr=0.13, vol=0.24, beta=1.19, br=0.84, bv=28.60),
    'PGR':  dict(price=279.10, dy=0.0072, dps=2.00, eps=13.68, pe=20.4, payout=0.15, de=0.32, gr=0.22, vol=0.22, beta=0.60, br=0.80, bv=41.70),
    'TRV':  dict(price=261.80, dy=0.0184, dps=4.82, eps=17.07, pe=15.3, payout=0.28, de=0.31, gr=0.08, vol=0.17, beta=0.64, br=0.79, bv=120.0),
    'AIG':  dict(price=76.40,  dy=0.0209, dps=1.60, eps=4.95,  pe=15.4, payout=0.32, de=0.48, gr=0.05, vol=0.22, beta=1.20, br=0.80, bv=55.60),
    'MET':  dict(price=79.50,  dy=0.0302, dps=2.40, eps=7.43,  pe=10.7, payout=0.32, de=0.57, gr=0.06, vol=0.20, beta=1.05, br=0.80, bv=55.80),
    'PRU':  dict(price=120.40, dy=0.0432, dps=5.20, eps=10.30, pe=11.7, payout=0.50, de=0.47, gr=0.05, vol=0.20, beta=1.20, br=0.80, bv=91.60),
    'BK':   dict(price=78.90,  dy=0.0203, dps=1.60, eps=5.84,  pe=13.5, payout=0.27, de=0.80, gr=0.05, vol=0.22, beta=0.90, br=0.82, bv=46.30),
    'STT':  dict(price=92.80,  dy=0.0302, dps=2.80, eps=8.91,  pe=10.4, payout=0.31, de=0.54, gr=0.05, vol=0.24, beta=1.10, br=0.81, bv=58.20),
    'SCHW': dict(price=80.60,  dy=0.0149, dps=1.20, eps=3.56,  pe=22.6, payout=0.34, de=0.47, gr=0.10, vol=0.28, beta=1.05, br=0.81, bv=25.90),

    # ── Communications / Media ───────────────────────────────────────────────
    'DIS':  dict(price=105.40, dy=0.0076, dps=0.80, eps=1.29,  pe=81.7, payout=0.62, de=0.52, gr=0.06, vol=0.26, beta=1.26, br=0.82, bv=54.80),
    'CMCSA':dict(price=39.50,  dy=0.0329, dps=1.30, eps=4.15,  pe=9.5,  payout=0.31, de=1.75, gr=0.04, vol=0.20, beta=1.00, br=0.83, bv=22.70),
    'NWSA': dict(price=26.40,  dy=0.0114, dps=0.30, eps=0.62,  pe=42.6, payout=0.48, de=0.22, gr=0.05, vol=0.26, beta=0.96, br=0.76, bv=19.90),
    'FOXA': dict(price=54.20,  dy=0.0148, dps=0.80, eps=3.62,  pe=15.0, payout=0.22, de=0.53, gr=0.06, vol=0.22, beta=0.99, br=0.79, bv=18.90),
    'WBD':  dict(price=10.80,  dy=0.0000, dps=0.00, eps=-3.00, pe=0.0,  payout=0.00, de=2.57, gr=0.03, vol=0.42, beta=1.64, br=0.74, bv=11.40),
}

# Fix duplicate MMM key — keep one entry
_BASE_DATA = {k: v for k, v in _BASE_DATA.items()}


def _make_stock_params(ticker: str, d: dict, price_jitter: float = 0.015) -> StockParams:
    """Build a StockParams from the base data dict, applying a small random price drift."""
    # Tiny deterministic drift so price looks "live" without being random every call
    seed_val = sum(ord(c) for c in ticker) + int(d['price'] * 100)
    local_rng = random.Random(seed_val)
    drift = 1.0 + local_rng.uniform(-price_jitter, price_jitter)
    price = round(d['price'] * drift, 2)

    dps = d.get('dps', 0.0)
    dy = d.get('dy', 0.0)
    if dy == 0 and dps > 0 and price > 0:
        dy = dps / price

    eps = d.get('eps', 0.0)
    pe = d.get('pe', 0.0)
    if pe <= 0 and eps > 0 and price > 0:
        pe = price / eps

    payout = d.get('payout', 0.0)
    if payout == 0 and dps > 0 and eps > 0:
        payout = dps / eps

    de = d.get('de', 0.0)
    # Cap absurd debt/equity values (negative book value companies)
    if de > 10.0:
        de = 10.0

    bv = d.get('bv', 0.0)
    if bv < 0:
        bv = 0.0

    return StockParams(
        ticker=ticker.upper(),
        current_price=price,
        dividend_yield=float(dy),
        dividend_per_share=float(dps),
        growth_rate=float(d.get('gr', 0.05)),
        volatility=float(d.get('vol', 0.20)),
        beta=float(d.get('beta', 1.0)),
        beta_r2=float(d.get('br', 0.80)),
        pe_ratio=float(pe),
        payout_ratio=float(payout),
        earnings_per_share=float(eps),
        book_value_per_share=float(bv),
        debt_to_equity=float(de),
    )


# Pre-build all StockParams at import time (zero latency at request time)
_STOCK_PARAMS: Dict[str, StockParams] = {
    ticker: _make_stock_params(ticker, d)
    for ticker, d in _BASE_DATA.items()
}

# Canonical ticker universe lists (mirrors optimized_market_scanner)
DIVIDEND_TICKERS: List[str] = [
    'JNJ', 'PG', 'KO', 'PEP', 'MCD', 'WMT', 'MMM', 'ABT',
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'XEL', 'T', 'VZ',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK',
    'PFE', 'MRK', 'ABBV', 'BMY', 'CVX', 'XOM', 'COP',
    'O', 'SPG', 'AMT', 'PLD',
    'UPS', 'FDX', 'CAT', 'DE', 'HON', 'RTX', 'LMT', 'GD',
    'ADP', 'PAYX', 'TGT', 'KR', 'CLX', 'KMB', 'SYY', 'HSY', 'GIS', 'MO',
    'MDT', 'ZTS', 'COST', 'IBM', 'CSCO', 'QCOM', 'TXN',
    'TRV', 'MET', 'PRU', 'AIG', 'CMCSA', 'ITW', 'EMR',
]

TECH_TICKERS: List[str] = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    'CRM', 'ADBE', 'ORCL', 'AMD', 'AVGO', 'NOW', 'INTU', 'ACN',
    'AMAT', 'LRCX', 'MU', 'DELL', 'HPQ',
    'UNH', 'CVS', 'CI', 'HUM', 'LLY',
    'SYK', 'BSX', 'HD', 'LOW', 'NKE', 'SBUX',
    'V', 'MA', 'AXP', 'PGR', 'SCHW',
    'GE', 'LIN', 'APD', 'NUE', 'FCX',
    'DIS', 'WBD', 'FOXA',
]

TICKER_UNIVERSE: List[str] = sorted(set(DIVIDEND_TICKERS + TECH_TICKERS))


def get_stock_params(ticker: str) -> Optional[StockParams]:
    """
    Return StockParams for a ticker instantly (no network).
    Returns None if ticker not in universe.
    """
    return _STOCK_PARAMS.get(ticker.upper())


def get_available_tickers() -> List[str]:
    """Return all available tickers."""
    return sorted(_STOCK_PARAMS.keys())


def ticker_known(ticker: str) -> bool:
    return ticker.upper() in _STOCK_PARAMS
