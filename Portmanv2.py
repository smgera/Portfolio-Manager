# ========================================================
# ULTRA PORTFOLIO MANAGER V1.17
# Premium Flet UI with NavigationRail sidebar, 24 tabs across 8 categories,
# 60+ engine methods, DuckDB storage, matplotlib charts, Monte Carlo, AI co-pilot.
# INSTALL: pip install flet pandas numpy yfinance duckdb quantstats pyportfolioopt matplotlib
# Run: python Portmanv1.py
# ========================================================

import flet as ft
import pandas as pd
import numpy as np
import yfinance as yf
import duckdb
import os, io, base64, time, json, math, sys
from pathlib import Path
from datetime import datetime, timedelta
import quantstats as qs
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Add project root to path for config and module imports
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import ACCOUNT_ACTIVITY_DIR, TICKER_LISTS_DIR, PARENT_DATA_DIR
from Acquire.DailyData import daily_data as fetch_daily_data
from Monitor.read_portfolio_positions import read_portfolio_positions
from Monitor.accountActivity import all_activity_data

class UltraDataEngine:
    def __init__(self):
        try:
            self.conn = duckdb.connect(database="ultra_portfolio_v25.db")
        except Exception:
            self.conn = duckdb.connect()
        self._init_db()
        self.master_df = pd.DataFrame()
        self.holdings = pd.DataFrame()
        self.prices = pd.DataFrame()
        self.returns_series = pd.Series()
        self.total_value = 0.0
        self.risk_profile = "Aggressive"
        self.raw_files = {}
        self.import_summary = []
        self._open_position_symbols = set()  # from myTickers.tsv
        self._load_open_positions()

    def _load_open_positions(self):
        """Load current open position symbols from myTickers.tsv"""
        try:
            my_tickers_path = TICKER_LISTS_DIR / "myTickers.tsv"
            if my_tickers_path.exists():
                df = pd.read_csv(my_tickers_path, sep='\t')
                if 'Symbol' in df.columns:
                    self._open_position_symbols = set(df['Symbol'].dropna().astype(str).tolist())
                elif len(df.columns) > 0:
                    # Try first column as symbol
                    self._open_position_symbols = set(df.iloc[:, 0].dropna().astype(str).tolist())
        except Exception:
            self._open_position_symbols = set()

    def _init_db(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS records (
                date TIMESTAMP, symbol VARCHAR, quantity DOUBLE, price DOUBLE,
                amount DOUBLE, action VARCHAR, account VARCHAR, source_file VARCHAR,
                year INTEGER, month VARCHAR, quarter VARCHAR
            )
        ''')

    SUPPORTED_EXTENSIONS = ('.csv', '.tsv', '.txt', '.xlsx', '.xls', '.json', '.parquet', '.pqt', '.feather', '.orc', '.html')

    def _read_file_to_df(self, path, filename):
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.csv':
            return pd.read_csv(path, on_bad_lines='skip')
        elif ext in ('.tsv', '.txt'):
            return pd.read_csv(path, sep='\t', on_bad_lines='skip')
        elif ext in ('.xlsx', '.xls'):
            return pd.read_excel(path)
        elif ext == '.json':
            return pd.read_json(path)
        elif ext in ('.parquet', '.pqt'):
            return pd.read_parquet(path)
        elif ext == '.feather':
            return pd.read_feather(path)
        elif ext == '.html':
            tables = pd.read_html(path)
            return tables[0] if tables else pd.DataFrame()
        return pd.DataFrame()

    def import_multi(self, paths_str):
        """Import from multiple paths (comma or semicolon separated). Each can be a folder or single file."""
        paths = [p.strip() for p in paths_str.replace(';', ',').split(',') if p.strip()]
        total = 0
        for p in paths:
            if os.path.isfile(p):
                total += self._import_single_file(p)
            elif os.path.isdir(p):
                total += self.import_folder(p)
        return total

    def _import_single_file(self, file_path):
        """Import a single file directly."""
        fname = os.path.basename(file_path)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return 0
        try:
            df = self._read_file_to_df(file_path, fname)
            if df.empty:
                return 0
            self.raw_files[fname] = df.copy()
            df.columns = [c.strip().lower().replace(' ', '_').replace('.', '') for c in df.columns]
            date_col = next((c for c in df.columns if any(x in c for x in ['date','run','trade','time','timestamp'])), None)
            if date_col:
                df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            else:
                return 0
            col_map = {
                'ticker':'symbol', 'stock':'symbol', 'asset':'symbol',
                'name':'symbol', 'security':'symbol', 'description':'description',
                'qty':'quantity', 'shares':'quantity', 'units':'quantity',
                'trade_price':'price', 'cost':'price', 'close':'price',
                'adj_close':'price', 'last_price':'price', 'current_value':'market_value',
                'cost_basis_total':'cost_basis', 'cost_basis_per_share':'cost_per_share',
                'average_cost_basis':'cost_per_share',
                'gain/loss_dollar':'gain_loss', 'gain/loss_percent':'gain_loss_pct',
                'unrealized_gain/loss':'gain_loss', 'percent_of_account':'weight_pct',
                'type':'asset_type', 'account_name':'account', 'account_number':'account',
            }
            for old, new in col_map.items():
                if old in df.columns and new not in df.columns:
                    df[new] = df[old]
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
                df['symbol'] = df['symbol'].apply(lambda x: f"{x}-USD" if x in ['BTC','ETH','SOL','XRP','ADA','DOT','DOGE','AVAX','MATIC','LINK'] else x)
            df = df.dropna(subset=['date', 'symbol'])
            if df.empty:
                return 0
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.strftime('%B')
            df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str)
            df['source_file'] = fname
            for col in ['quantity','price','amount','action','account','cost_basis','cost_per_share','gain_loss','gain_loss_pct','market_value','weight_pct','description','asset_type']:
                if col not in df.columns:
                    df[col] = np.nan
            keep = ['date','symbol','quantity','price','cost_basis','cost_per_share','gain_loss','gain_loss_pct','market_value','weight_pct','description','asset_type','source_file','year','month','quarter','account']
            new_df = df[[c for c in keep if c in df.columns]]
            if self.master_df.empty:
                self.master_df = new_df.sort_values('date').drop_duplicates()
            else:
                self.master_df = pd.concat([self.master_df, new_df], ignore_index=True).sort_values('date').drop_duplicates()
            self._update_import_summary(fname, new_df)
            try:
                self.conn.execute("DELETE FROM records")
                self.conn.register("temp_df_single", self.master_df)
                self.conn.execute("INSERT INTO records SELECT * FROM temp_df_single")
            except Exception:
                pass
            self._compute_holdings_and_prices()
            self._generate_returns_series()
            return len(new_df)
        except Exception:
            return 0

    def _update_import_summary(self, filename, df):
        """Track import summary per file: symbols, record count, date range, file type."""
        if not hasattr(self, 'import_summary'):
            self.import_summary = []
        syms = df['symbol'].unique().tolist() if 'symbol' in df.columns else []
        ext = os.path.splitext(filename)[1].lower()
        date_min = df['date'].min() if 'date' in df.columns and not df['date'].isna().all() else None
        date_max = df['date'].max() if 'date' in df.columns and not df['date'].isna().all() else None
        self.import_summary.append({
            'file': filename,
            'format': ext,
            'records': len(df),
            'symbols': syms,
            'date_from': str(date_min.date()) if date_min and pd.notna(date_min) else 'N/A',
            'date_to': str(date_max.date()) if date_max and pd.notna(date_max) else 'N/A',
        })

    def get_import_summary(self):
        return getattr(self, 'import_summary', [])

    def get_asset_source_map(self):
        """Return mapping of symbol -> list of source files it came from."""
        if self.master_df.empty or 'symbol' not in self.master_df.columns or 'source_file' not in self.master_df.columns:
            return {}
        return self.master_df.groupby('symbol')['source_file'].apply(lambda x: sorted(x.unique().tolist())).to_dict()

    def import_folder(self, folder_path):
        dfs = []
        if not hasattr(self, 'raw_files'):
            self.raw_files = {}
        for root, _, files in os.walk(folder_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    path = os.path.join(root, f)
                    try:
                        df = self._read_file_to_df(path, f)
                        if df.empty:
                            continue
                        self.raw_files[f] = df.copy()
                        df.columns = [c.strip().lower().replace(' ', '_').replace('.', '') for c in df.columns]
                        date_col = next((c for c in df.columns if any(x in c for x in ['date','run','trade','time','timestamp'])), None)
                        if date_col:
                            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
                        else:
                            continue
                        col_map = {
                            'ticker':'symbol', 'stock':'symbol', 'asset':'symbol',
                            'name':'symbol', 'security':'symbol', 'description':'description',
                            'qty':'quantity', 'shares':'quantity', 'units':'quantity',
                            'trade_price':'price', 'cost':'price', 'close':'price',
                            'adj_close':'price', 'last_price':'price', 'current_value':'market_value',
                            'cost_basis_total':'cost_basis', 'cost_basis_per_share':'cost_per_share',
                            'average_cost_basis':'cost_per_share',
                            'gain/loss_dollar':'gain_loss', 'gain/loss_percent':'gain_loss_pct',
                            'unrealized_gain/loss':'gain_loss', 'percent_of_account':'weight_pct',
                            'type':'asset_type', 'account_name':'account', 'account_number':'account',
                        }
                        for old, new in col_map.items():
                            if old in df.columns and new not in df.columns:
                                df[new] = df[old]
                        if 'symbol' in df.columns:
                            df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
                            df['symbol'] = df['symbol'].apply(lambda x: f"{x}-USD" if x in ['BTC','ETH','SOL','XRP','ADA','DOT','DOGE','AVAX','MATIC','LINK'] else x)
                        df = df.dropna(subset=['date', 'symbol'])
                        if df.empty:
                            continue
                        df['year'] = df['date'].dt.year
                        df['month'] = df['date'].dt.strftime('%B')
                        df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str)
                        df['source_file'] = f
                        for col in ['quantity','price','amount','action','account','cost_basis','cost_per_share','gain_loss','gain_loss_pct','market_value','weight_pct','description','asset_type']:
                            if col not in df.columns:
                                df[col] = np.nan
                        keep = ['date','symbol','quantity','price','cost_basis','cost_per_share','gain_loss','gain_loss_pct','market_value','weight_pct','description','asset_type','source_file','year','month','quarter','account']
                        filtered = df[[c for c in keep if c in df.columns]]
                        dfs.append(filtered)
                        self._update_import_summary(f, filtered)
                    except Exception:
                        continue
        if dfs:
            self.master_df = pd.concat(dfs, ignore_index=True).sort_values('date').drop_duplicates()
            try:
                self.conn.execute("DELETE FROM records")
                self.conn.register("temp_df", self.master_df)
                self.conn.execute("INSERT INTO records SELECT * FROM temp_df")
            except Exception:
                pass
            self._compute_holdings_and_prices()
            self._generate_returns_series()
            return len(self.master_df)
        return 0

    def _compute_holdings_and_prices(self):
        # Prefer holdings-type records (have cost_basis) over mixed trade/dividend/price data
        has_cb = 'cost_basis' in self.master_df.columns and self.master_df['cost_basis'].notna().any()
        if has_cb:
            holdings_df = self.master_df[self.master_df['cost_basis'].notna()].copy()
        else:
            holdings_df = self.master_df.copy()
        # Filter out non-stock symbols (statements, aggregates)
        junk_syms = {'PORTFOLIO', 'TOTAL', 'CASH', 'SUMMARY', 'NAN', ''}
        holdings_df = holdings_df[~holdings_df['symbol'].isin(junk_syms)]
        if holdings_df.empty:
            holdings_df = self.master_df[~self.master_df['symbol'].isin(junk_syms)].copy()
        agg = {'quantity': 'sum'}
        for c in ['cost_basis', 'gain_loss', 'gain_loss_pct', 'market_value']:
            if c in holdings_df.columns:
                agg[c] = 'last'
        self.holdings = holdings_df.groupby('symbol').agg(agg).reset_index()
        # Filter to only open positions from myTickers.tsv (exclude closed positions)
        if self._open_position_symbols:
            self.holdings = self.holdings[self.holdings['symbol'].isin(self._open_position_symbols)]
        # Filter valid stock symbols
        symbols = [s for s in self.holdings['symbol'].unique().tolist() if s and s not in junk_syms and len(s) <= 10]
        if symbols:
            try:
                # Use DailyData.py for price data instead of direct yfinance calls
                price_frames = {}
                for sym in symbols:
                    try:
                        sym_data = fetch_daily_data(sym, period='5y')
                        if sym_data is not None and not sym_data.empty and 'Close' in sym_data.columns:
                            price_frames[sym] = sym_data['Close']
                    except Exception:
                        pass
                if price_frames:
                    self.prices = pd.DataFrame(price_frames)
                else:
                    self.prices = pd.DataFrame()
                # Always recompute total_value + gain/loss from LIVE prices for accuracy
                self.total_value = 0.0
                has_csv_cb = 'cost_basis' in self.holdings.columns and self.holdings['cost_basis'].notna().any()
                if 'gain_loss' not in self.holdings.columns:
                    self.holdings['gain_loss'] = 0.0
                if 'gain_loss_pct' not in self.holdings.columns:
                    self.holdings['gain_loss_pct'] = 0.0
                for idx, row in self.holdings.iterrows():
                    sym = row['symbol']
                    qty = float(row.get('quantity', 0) or 0)
                    if not self.prices.empty and sym in (self.prices.columns if self.prices.ndim == 2 else []):
                        col = self.prices[sym].dropna()
                        if not col.empty:
                            cur_price = float(col.iloc[-1])
                            mv = qty * cur_price
                            self.total_value += mv
                            self.holdings.at[idx, 'market_value'] = mv
                            # Use CSV cost_basis if available, else estimate from earliest price
                            cb = float(row.get('cost_basis', 0) or 0)
                            if cb <= 0 and len(col) >= 2:
                                cb = qty * float(col.iloc[0])
                            if cb > 0:
                                self.holdings.at[idx, 'cost_basis'] = cb
                                self.holdings.at[idx, 'gain_loss'] = mv - cb
                                self.holdings.at[idx, 'gain_loss_pct'] = ((mv - cb) / cb * 100)
                    elif not self.prices.empty and self.prices.ndim == 1:
                        if not self.prices.dropna().empty:
                            self.total_value += qty * float(self.prices.dropna().iloc[-1])
            except Exception:
                self.prices = pd.DataFrame()
                self.total_value = 0.0

    def _generate_returns_series(self):
        if not self.prices.empty:
            junk_syms = {'PORTFOLIO', 'TOTAL', 'CASH', 'SUMMARY', 'NAN', ''}
            if self.prices.ndim == 2 and len(self.prices.columns) > 0:
                # Market-value weighted portfolio returns (more accurate than quantity weighting)
                weighted = pd.Series(0.0, index=self.prices.index)
                total_mv = 0.0
                sym_mv = {}
                for _, row in self.holdings.iterrows():
                    sym = row['symbol']
                    if sym in junk_syms or sym not in self.prices.columns:
                        continue
                    qty = float(row.get('quantity', 0) or 0)
                    col = self.prices[sym].dropna()
                    if not col.empty and qty > 0:
                        mv = qty * float(col.iloc[-1])
                        sym_mv[sym] = mv
                        total_mv += mv
                if total_mv > 0:
                    for sym, mv in sym_mv.items():
                        w = mv / total_mv
                        weighted += self.prices[sym].pct_change().fillna(0) * w
                    self.returns_series = weighted.dropna()
                if self.returns_series.empty:
                    # Fallback: equal-weight average of all price columns
                    valid_cols = [c for c in self.prices.columns if c not in junk_syms]
                    if valid_cols:
                        self.returns_series = self.prices[valid_cols].mean(axis=1).pct_change().dropna()
            else:
                self.returns_series = self.prices.squeeze().pct_change().dropna()

    # Time-range filter for charts: 1H, 1D, 1W, 1M, 3M, 6M, 1Y, ALL
    def filter_by_timerange(self, series_or_df, timerange='ALL'):
        if series_or_df is None or (hasattr(series_or_df, 'empty') and series_or_df.empty):
            return series_or_df
        tr_map = {'1H': pd.Timedelta(hours=1), '1D': pd.Timedelta(days=1), '1W': pd.Timedelta(weeks=1),
                  '1M': pd.Timedelta(days=30), '3M': pd.Timedelta(days=91), '6M': pd.Timedelta(days=182),
                  '1Y': pd.Timedelta(days=365), 'ALL': None}
        delta = tr_map.get(timerange.upper())
        if delta is None:
            return series_or_df
        cutoff = pd.Timestamp.now() - delta
        if hasattr(series_or_df, 'index') and hasattr(series_or_df.index, 'tz'):
            try:
                return series_or_df.loc[series_or_df.index >= cutoff]
            except Exception:
                return series_or_df
        return series_or_df

    # Core v20 methods (kept for full backward compatibility)
    def get_time_hierarchy(self):
        try:
            return self.conn.execute("SELECT year, quarter, month, COUNT(*) as records, SUM(quantity) as total_qty FROM records GROUP BY year, quarter, month ORDER BY year DESC, quarter, month").df()
        except Exception:
            return pd.DataFrame()

    def correlation_matrix(self):
        return self.prices.corr() if not self.prices.empty else None

    # ====================== CHART HELPER: matplotlib → base64 ======================
    @staticmethod
    def _fig_to_base64(fig, dpi=120):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#121212", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # ====================== CHART METHODS (matplotlib) ======================
    def chart_equity_curve(self, show_sma20=False, show_sma50=True, show_sma200=False,
                           show_bollinger=False, log_scale=False, show_drawdown_shade=False,
                           show_pct=False):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        if not self.returns_series.empty:
            cum = (1 + self.returns_series).cumprod()
            if show_pct:
                cum_dollars = (cum / cum.iloc[0] - 1) * 100
                y_fmt = lambda x, _: f"{x:+.1f}%"
                title_suffix = " (% Return)"
            elif self.total_value > 0:
                scale = self.total_value / cum.iloc[-1] if cum.iloc[-1] != 0 else 1
                cum_dollars = cum * scale
                y_fmt = lambda x, _: f"${x:,.0f}"
                title_suffix = ""
            else:
                cum_dollars = cum
                y_fmt = lambda x, _: f"{x:.4f}"
                title_suffix = " (Growth of $1)"
            ax.plot(cum_dollars.index, cum_dollars.values, color="#00ff9d", linewidth=2, label="Portfolio")
            ax.fill_between(cum_dollars.index, cum_dollars.values, alpha=0.1, color="#00ff9d")
            # Drawdown shading
            if show_drawdown_shade:
                running_max = cum_dollars.cummax()
                dd_mask = cum_dollars < running_max
                ax.fill_between(cum_dollars.index, cum_dollars.values, running_max.values,
                                where=dd_mask, alpha=0.2, color="#ff3366", label="Drawdown")
            # SMAs
            if show_sma20 and len(cum_dollars) > 20:
                ma20 = cum_dollars.rolling(20).mean()
                ax.plot(ma20.index, ma20.values, color="#00bfff", linewidth=1, alpha=0.7, linestyle="--", label="20d SMA")
            if show_sma50 and len(cum_dollars) > 50:
                ma50 = cum_dollars.rolling(50).mean()
                ax.plot(ma50.index, ma50.values, color="#ffd700", linewidth=1, alpha=0.7, linestyle="--", label="50d SMA")
            if show_sma200 and len(cum_dollars) > 200:
                ma200 = cum_dollars.rolling(200).mean()
                ax.plot(ma200.index, ma200.values, color="#ff9900", linewidth=1, alpha=0.7, linestyle="-.", label="200d SMA")
            # Bollinger bands
            if show_bollinger and len(cum_dollars) > 20:
                ma20_bb = cum_dollars.rolling(20).mean()
                std20 = cum_dollars.rolling(20).std()
                upper = ma20_bb + 2 * std20
                lower = ma20_bb - 2 * std20
                ax.plot(upper.index, upper.values, color="#9966ff", linewidth=0.8, alpha=0.5, linestyle=":")
                ax.plot(lower.index, lower.values, color="#9966ff", linewidth=0.8, alpha=0.5, linestyle=":")
                ax.fill_between(upper.index, upper.values, lower.values, alpha=0.05, color="#9966ff", label="Bollinger")
            # Peak / current annotations
            if not show_pct:
                peak_idx = cum_dollars.idxmax()
                peak_val = cum_dollars.max()
                ax.annotate(f"Peak: ${peak_val:,.0f}", xy=(peak_idx, peak_val),
                            xytext=(10, 10), textcoords='offset points',
                            color="#ffd700", fontsize=9, fontweight="bold",
                            arrowprops=dict(arrowstyle='->', color='#ffd700', lw=1))
                ax.plot(cum_dollars.index[-1], cum_dollars.iloc[-1], 'o', color="#00ff9d", markersize=8, zorder=5)
                ax.annotate(f"Now: ${cum_dollars.iloc[-1]:,.0f}", xy=(cum_dollars.index[-1], cum_dollars.iloc[-1]),
                            xytext=(-80, -20), textcoords='offset points',
                            color="#00ff9d", fontsize=9, fontweight="bold")
            total_ret = (cum.iloc[-1] / cum.iloc[0] - 1) * 100
            ax.set_xlabel(f"Total Return: {total_ret:+.1f}%  |  Period: {cum.index[0].strftime('%Y-%m-%d')} to {cum.index[-1].strftime('%Y-%m-%d')}", color="#888888", fontsize=9)
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate(rotation=45)
            if log_scale and not show_pct:
                ax.set_yscale('log')
            ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa", fontsize=8, loc='upper left')
        else:
            ax.text(0.5, 0.5, "Import data to see equity curve", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
            y_fmt = lambda x, _: f"${x:,.0f}"
            title_suffix = ""
        ax.set_title(f"PORTFOLIO EQUITY CURVE{title_suffix}", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_fmt))
        ax.grid(True, alpha=0.1, color='#555555')
        return self._fig_to_base64(fig)

    def chart_correlation_matrix(self):
        corr = self.correlation_matrix()
        n = corr.shape[0] if corr is not None else 0
        # Scale figure size dynamically based on number of assets
        base = max(8, n * 0.6) if n > 1 else 8
        font_size = max(5, min(8, 100 // max(n, 1)))
        fig, ax = plt.subplots(figsize=(base, base * 0.8))
        ax.set_facecolor("#1a1a2e")
        if corr is not None and n > 1:
            im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="#aaaaaa", fontsize=font_size)
            ax.set_yticklabels(corr.columns, color="#aaaaaa", fontsize=font_size)
            # Only show numbers if matrix isn't too large
            if n <= 25:
                num_font = max(4, min(7, 80 // max(n, 1)))
                for i in range(n):
                    for j in range(n):
                        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=num_font)
            fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, "Need 2+ assets for correlation matrix", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("CORRELATION MATRIX", color="#ffd700", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return self._fig_to_base64(fig)

    def chart_risk_radar(self):
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#121212")
        ax.set_facecolor("#1a1a2e")
        if not self.returns_series.empty and not self.holdings.empty:
            vol = float(self.returns_series.std() * np.sqrt(252))
            n_assets = len(self.holdings)
            max_dd = float(abs(self.returns_series.cummax() - self.returns_series).min()) if len(self.returns_series) > 1 else 0
            tail_risk = float(self.returns_series.quantile(0.05)) * -100 if len(self.returns_series) > 20 else 50
            concentration = 100 / max(n_assets, 1)
            liquidity = min(90, n_assets * 15)
            momentum = float(self.returns_series.tail(60).mean() * 252 * 100) if len(self.returns_series) > 60 else 50
            stability = max(0, min(100, 100 - vol))
            categories = ["Volatility", "Drawdown", "Tail Risk", "Concentration", "Liquidity", "Momentum", "Stability"]
            values = [min(100, max(0, v)) for v in [vol, max_dd, tail_risk, concentration, liquidity, abs(momentum), stability]]
        else:
            categories = ["Volatility", "Drawdown", "Tail Risk", "Concentration", "Liquidity", "Momentum", "Stability"]
            values = [0] * 7
        values.append(values[0])
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles.append(angles[0])
        ax.plot(angles, values, color="#00ff9d", linewidth=2)
        ax.fill(angles, values, alpha=0.25, color="#00ff9d")
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, color="#aaaaaa", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title("RISK RADAR (from your data)", color="#00ff9d", fontsize=14, fontweight="bold", pad=20)
        ax.tick_params(colors="#555555")
        ax.grid(color="#333333")
        return self._fig_to_base64(fig)

    def chart_candlestick(self, period=120, show_sma20=False, show_sma50=False, show_volume=False, chart_type='candle'):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#1a1a2e")
        if not self.returns_series.empty:
            cum = (1 + self.returns_series).cumprod().tail(period)
            if chart_type == 'line':
                ax.plot(range(len(cum)), cum.values, color="#00ff9d", linewidth=2)
                ax.fill_between(range(len(cum)), cum.values, alpha=0.1, color="#00ff9d")
            elif chart_type == 'area':
                ax.fill_between(range(len(cum)), cum.values, alpha=0.4, color="#00ff9d")
                ax.plot(range(len(cum)), cum.values, color="#00ff9d", linewidth=1.5)
            else:
                for i in range(1, len(cum)):
                    color = "#00ff9d" if cum.iloc[i] >= cum.iloc[i - 1] else "#ff3366"
                    ax.bar(i, abs(cum.iloc[i] - cum.iloc[i - 1]), bottom=min(cum.iloc[i], cum.iloc[i - 1]), color=color, width=0.6, alpha=0.8)
                ax.plot(range(len(cum)), cum.values, color="#ffd700", linewidth=1, alpha=0.5)
            # SMA overlays
            if show_sma20 and len(cum) > 20:
                ma20 = pd.Series(cum.values).rolling(20).mean()
                ax.plot(range(len(cum)), ma20.values, color="#00bfff", linewidth=1, linestyle="--", alpha=0.7, label="20d SMA")
            if show_sma50 and len(cum) > 50:
                ma50 = pd.Series(cum.values).rolling(50).mean()
                ax.plot(range(len(cum)), ma50.values, color="#ffd700", linewidth=1, linestyle="--", alpha=0.7, label="50d SMA")
            # Volume bars on secondary axis
            if show_volume and len(cum) > 1:
                ax2 = ax.twinx()
                daily_ret = cum.pct_change().fillna(0).abs()
                colors_v = ["#00ff9d" if cum.iloc[i] >= cum.iloc[max(0, i-1)] else "#ff3366" for i in range(len(cum))]
                ax2.bar(range(len(cum)), daily_ret.values * 100, color=colors_v, alpha=0.2, width=0.4)
                ax2.set_ylim(0, daily_ret.max() * 500)
                ax2.tick_params(colors="#555555")
                ax2.set_ylabel("Activity", color="#555555", fontsize=8)
            if show_sma20 or show_sma50:
                ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Import data to see market graphs", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title(f"MARKET GRAPH (last {period}d)", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        ax.grid(True, alpha=0.1, color='#555555')
        for spine in ax.spines.values():
            spine.set_color("#333333")
        return self._fig_to_base64(fig)

    # ====================== NON-CHART METHODS ======================
    def generate_diversification_score(self):
        if self.holdings.empty:
            return "DIVERSIFICATION SCORE: N/A\n\nNo holdings loaded. Import data first."
        n = len(self.holdings)
        asset_spread = min(10, n)
        corr = self.correlation_matrix()
        if corr is not None and corr.shape[0] > 1:
            avg_corr = (corr.values.sum() - corr.shape[0]) / (corr.shape[0] * (corr.shape[0] - 1)) if corr.shape[0] > 1 else 1.0
            corr_score = round(max(0, min(10, (1 - avg_corr) * 10)), 1)
        else:
            corr_score = 0.0
        vol_score = 0.0
        if not self.returns_series.empty:
            vol = float(self.returns_series.std() * np.sqrt(252))
            vol_score = round(max(0, min(10, (0.5 - vol) * 20)), 1)
        qty_series = self.holdings['quantity']
        max_pct = float(qty_series.max() / qty_series.sum()) if qty_series.sum() > 0 else 1.0
        concentration = round(max(0, min(10, (1 - max_pct) * 10)), 1)
        scores = {
            "Asset Count": float(asset_spread),
            "Correlation": corr_score,
            "Volatility": vol_score,
            "Concentration": concentration,
        }
        overall = round(sum(scores.values()) / len(scores) * 10, 0)
        lines = [f"DIVERSIFICATION SCORE: {int(overall)}/100\n"]
        for k, v in scores.items():
            bar = "█" * int(v) + "░" * (10 - int(v))
            lines.append(f"  {k:14s}  {bar}  {v}/10")
        return "\n".join(lines)

    def detect_market_regime(self):
        if not self.returns_series.empty:
            recent_vol = float(self.returns_series.tail(60).std() * np.sqrt(252))
            recent_ret = float(self.returns_series.tail(60).mean() * 252)
            if recent_ret > 0.10 and recent_vol < 0.20:
                regime = "BULL MARKET"
            elif recent_ret < -0.05:
                regime = "BEAR MARKET"
            else:
                regime = "SIDEWAYS / TRANSITIONAL"
            return (
                f"CURRENT REGIME: {regime}\n\n"
                f"60-day annualized return: {recent_ret:.1%}\n"
                f"60-day annualized volatility: {recent_vol:.1%}\n"
                f"Portfolio value: ${self.total_value:,.2f}\n"
                f"Assets tracked: {len(self.holdings)}"
            )
        return "CURRENT REGIME: UNKNOWN\n\nNo data loaded. Import a folder to detect market regime."

    def benchmark_comparison(self, benchmarks=None):
        if benchmarks is None:
            benchmarks = ['SPY', 'QQQ', 'IWM', 'BTC-USD']
        result = []
        try:
            # Use DailyData.py for benchmark data
            for bm in benchmarks:
                try:
                    bm_data = fetch_daily_data(bm, period='5y')
                    if bm_data is None or bm_data.empty or 'Close' not in bm_data.columns:
                        continue
                    bm_close = bm_data['Close'].dropna()
                    if len(bm_close) < 2:
                        continue
                    bm_ret = bm_close.pct_change().dropna()
                    ann_r = float(bm_ret.mean() * 252)
                    ann_v = float(bm_ret.std() * np.sqrt(252))
                    total_r = float((bm_close.iloc[-1] / bm_close.iloc[0]) - 1)
                    if not self.returns_series.empty and len(bm_ret) > 10:
                        corr = float(self.returns_series.corr(bm_ret))
                        if np.isnan(corr):
                            corr = None
                    else:
                        corr = None
                    result.append({"symbol": bm, "ann_return": ann_r, "ann_vol": ann_v, "total_return": total_r, "correlation": corr})
                except Exception:
                    continue
        except Exception:
            pass
        return result

    def get_trade_log(self, limit=100):
        if self.master_df.empty:
            return []
        df = self.master_df.sort_values('date', ascending=False).head(limit)
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "date": str(r.get('date', ''))[:10],
                "symbol": str(r.get('symbol', '')),
                "quantity": float(r.get('quantity', 0) or 0),
                "price": float(r.get('price', 0) or 0),
                "cost_basis": float(r.get('cost_basis', 0) or 0),
                "gain_loss": float(r.get('gain_loss', 0) or 0),
                "source": str(r.get('source_file', '')),
            })
        return rows

    def chart_benchmark_overlay(self, benchmarks=None, show_portfolio=True, timerange='ALL', show_corr_labels=False, show_spread=False):
        if benchmarks is None:
            benchmarks = ['SPY', 'QQQ']
        # Map timerange — always use daily bars to keep stocks & crypto consistent
        # Download extra history then trim to desired window
        tr_map = {
            '1D': ('5d', '1d'), '1W': ('1mo', '1d'), '1M': ('3mo', '1d'),
            '3M': ('6mo', '1d'), '6M': ('1y', '1d'), '1Y': ('2y', '1d'),
            '5Y': ('max', '1d'), 'ALL': ('max', '1d'),
        }
        yf_period, yf_interval = tr_map.get(timerange.upper(), ('max', '1d'))
        # Number of trading days to keep for each range
        tr_days = {
            '1D': 1, '1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252, '5Y': 1260,
        }
        tr_delta = {
            '1D': pd.Timedelta(days=3), '1W': pd.Timedelta(days=10), '1M': pd.Timedelta(days=35),
            '3M': pd.Timedelta(days=100), '6M': pd.Timedelta(days=200), '1Y': pd.Timedelta(days=380),
            '5Y': pd.Timedelta(days=1900),
        }
        # Color map for known symbols
        color_map = {
            'SPY': '#ffd700', 'QQQ': '#ff3366', 'IWM': '#00bfff', 'BTC-USD': '#ff9900',
            'DIA': '#9966ff', 'VTI': '#66ff66', 'GLD': '#cccc00', 'TLT': '#ff6699',
            'ARKK': '#ff66cc', 'XLF': '#66ccff', 'XLE': '#cc9933', 'XLK': '#33cccc',
        }
        fallback_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                           '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990']
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#121212')
        # Determine date range from portfolio data
        port_start = None
        if show_portfolio and not self.returns_series.empty:
            port_ret = self.returns_series.copy()
            delta = tr_delta.get(timerange.upper())
            if delta is not None:
                cutoff = pd.Timestamp.now() - delta
                port_ret = port_ret.loc[port_ret.index >= cutoff]
            if not port_ret.empty:
                cum = (1 + port_ret).cumprod()
                port_start = cum.index.min()
                ax.plot(cum.index, cum.values, color='#00ff9d', linewidth=2.5, label='Portfolio', zorder=5)
                ax.fill_between(cum.index, 1.0, cum.values, alpha=0.08, color='#00ff9d')
        if benchmarks:
            try:
                # Use DailyData.py for benchmark overlay data
                fb_idx = 0
                n_days = tr_days.get(timerange.upper())
                for bm in benchmarks:
                    try:
                        bm_raw = fetch_daily_data(bm, period=yf_period)
                        if bm_raw is None or bm_raw.empty or 'Close' not in bm_raw.columns:
                            continue
                        bm_ser = bm_raw['Close'].dropna()
                        if n_days is not None and len(bm_ser) > n_days:
                            bm_ser = bm_ser.iloc[-n_days:]
                        if len(bm_ser) > 1:
                            bm_ret = bm_ser.pct_change().dropna()
                            bm_cum = (1 + bm_ret).cumprod()
                            c = color_map.get(bm, fallback_colors[fb_idx % len(fallback_colors)])
                            if bm not in color_map:
                                fb_idx += 1
                            ax.plot(bm_cum.index, bm_cum.values, color=c, linewidth=1.5, label=bm, alpha=0.85)
                    except Exception:
                        continue
            except Exception:
                pass
        # Correlation labels & spread shading
        if (show_corr_labels or show_spread) and show_portfolio and not self.returns_series.empty and benchmarks:
            port_ret = self.returns_series.copy()
            delta = tr_delta.get(timerange.upper())
            if delta is not None:
                cutoff = pd.Timestamp.now() - delta
                port_ret = port_ret.loc[port_ret.index >= cutoff]
            if not port_ret.empty:
                port_cum = (1 + port_ret).cumprod()
                n_days_corr = tr_days.get(timerange.upper())
                fb_idx2 = 0
                for bm in benchmarks:
                    try:
                        bm_raw2 = fetch_daily_data(bm, period=yf_period)
                        if bm_raw2 is None or bm_raw2.empty or 'Close' not in bm_raw2.columns:
                            continue
                        bm_ser2 = bm_raw2['Close'].dropna()
                        if n_days_corr is not None and len(bm_ser2) > n_days_corr:
                            bm_ser2 = bm_ser2.iloc[-n_days_corr:]
                        bm_ret2 = bm_ser2.pct_change().dropna()
                        bm_cum2 = (1 + bm_ret2).cumprod()
                        corr_val = float(port_ret.corr(bm_ret2))
                        c2 = color_map.get(bm, fallback_colors[fb_idx2 % len(fallback_colors)])
                        if bm not in color_map:
                            fb_idx2 += 1
                        if show_corr_labels and not np.isnan(corr_val):
                            ax.annotate(f"ρ={corr_val:.2f}", xy=(bm_cum2.index[-1], bm_cum2.iloc[-1]),
                                        xytext=(5, 0), textcoords='offset points',
                                        color=c2, fontsize=8, fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', edgecolor=c2, alpha=0.8))
                        if show_spread:
                            common_idx = port_cum.index.intersection(bm_cum2.index)
                            if len(common_idx) > 5:
                                p_aligned = port_cum.reindex(common_idx)
                                b_aligned = bm_cum2.reindex(common_idx)
                                spread = p_aligned - b_aligned
                                ax.fill_between(common_idx, port_cum.reindex(common_idx).values,
                                                bm_cum2.reindex(common_idx).values, alpha=0.06, color=c2)
                    except Exception:
                        continue
        n_lines = len(ax.get_lines())
        range_label = timerange.upper() if timerange.upper() != 'ALL' else 'All Time'
        title = f'Portfolio vs Benchmarks — {range_label}' if n_lines > 1 else f'Cumulative Returns — {range_label}'
        ax.set_title(title, color='#ffffff', fontsize=14)
        ax.tick_params(colors='#888888')
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ncol = min(len(labels), 4)
            ax.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='#ffffff', fontsize=10, ncol=ncol, loc='upper left')
        ax.grid(True, alpha=0.15)
        ax.axhline(y=1.0, color='#555555', linewidth=0.8, linestyle='--', alpha=0.5)
        for spine in ax.spines.values(): spine.set_color('#333333')
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def max_drawdown(self):
        if self.returns_series.empty: return 0.0
        cum = (1 + self.returns_series).cumprod()
        return float(((cum - cum.cummax()) / cum.cummax()).min())

    def value_at_risk(self, confidence=0.95):
        if self.returns_series.empty: return 0.0
        return float(self.returns_series.quantile(1 - confidence))

    def conditional_var(self, confidence=0.95):
        if self.returns_series.empty: return 0.0
        var = self.value_at_risk(confidence)
        return float(self.returns_series[self.returns_series <= var].mean())

    def beta_vs_market(self):
        if self.returns_series.empty: return 0.0
        try:
            spy_dl = fetch_daily_data("SPY", period="5y")
            if spy_dl is None or spy_dl.empty:
                return 0.0
            spy = (spy_dl['Close'] if 'Close' in spy_dl.columns else spy_dl.iloc[:, 0]).pct_change().dropna()
            common = self.returns_series.index.intersection(spy.index)
            if len(common) < 30: return 0.0
            cov = np.cov(self.returns_series.loc[common], spy.loc[common])
            return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 0.0
        except Exception:
            return 0.0

    def rolling_sharpe(self, window=60):
        if len(self.returns_series) < window: return pd.Series(dtype=float)
        roll_mean = self.returns_series.rolling(window).mean() * 252
        roll_std = self.returns_series.rolling(window).std() * np.sqrt(252)
        return (roll_mean / roll_std).dropna()

    def rolling_volatility(self, window=30):
        if len(self.returns_series) < window: return pd.Series(dtype=float)
        return (self.returns_series.rolling(window).std() * np.sqrt(252)).dropna()

    def top_gainers(self, n=5):
        if self.holdings.empty or 'gain_loss_pct' not in self.holdings.columns: return []
        df = self.holdings.dropna(subset=['gain_loss_pct']).nlargest(n, 'gain_loss_pct')
        return [{"symbol": r['symbol'], "gain_pct": r['gain_loss_pct'], "gain_dollar": r.get('gain_loss', 0)} for _, r in df.iterrows()]

    def top_losers(self, n=5):
        if self.holdings.empty or 'gain_loss_pct' not in self.holdings.columns: return []
        df = self.holdings.dropna(subset=['gain_loss_pct']).nsmallest(n, 'gain_loss_pct')
        return [{"symbol": r['symbol'], "gain_pct": r['gain_loss_pct'], "gain_dollar": r.get('gain_loss', 0)} for _, r in df.iterrows()]

    def portfolio_summary_stats(self):
        s = {"total_value": self.total_value, "num_holdings": len(self.holdings) if not self.holdings.empty else 0}
        s["total_gain_loss"] = float(self.holdings['gain_loss'].sum()) if 'gain_loss' in self.holdings.columns else 0
        s["total_cost_basis"] = float(self.holdings['cost_basis'].sum()) if 'cost_basis' in self.holdings.columns else 0
        if not self.returns_series.empty:
            s["ann_return"] = float(self.returns_series.mean() * 252)
            s["ann_volatility"] = float(self.returns_series.std() * np.sqrt(252))
            s["max_drawdown"] = self.max_drawdown()
            s["sharpe"] = float(qs.stats.sharpe(self.returns_series) or 0)
            s["sortino"] = float(qs.stats.sortino(self.returns_series) or 0)
            try: s["calmar"] = float(qs.stats.calmar(self.returns_series) or 0)
            except Exception: s["calmar"] = 0
            s["var_95"] = self.value_at_risk(0.95)
            s["cvar_95"] = self.conditional_var(0.95)
            s["best_day"] = float(self.returns_series.max())
            s["worst_day"] = float(self.returns_series.min())
            pos = int((self.returns_series > 0).sum())
            neg = int((self.returns_series < 0).sum())
            s["positive_days"] = pos
            s["negative_days"] = neg
            s["win_rate"] = pos / max(1, pos + neg)
            s["skewness"] = float(self.returns_series.skew())
            s["kurtosis"] = float(self.returns_series.kurtosis())
        else:
            for k in ["ann_return","ann_volatility","max_drawdown","sharpe","sortino","calmar","var_95","cvar_95","best_day","worst_day","positive_days","negative_days","win_rate","skewness","kurtosis"]:
                s[k] = 0
        s["total_gain_pct"] = (s["total_gain_loss"] / s["total_cost_basis"] * 100) if s["total_cost_basis"] > 0 else 0
        return s

    def holdings_with_weights(self):
        if self.holdings.empty: return []
        rows = []
        for _, row in self.holdings.iterrows():
            sym = row['symbol']
            qty = float(row.get('quantity', 0) or 0)
            price = 0.0
            if not self.prices.empty and self.prices.ndim == 2 and sym in self.prices.columns:
                col = self.prices[sym].dropna()
                price = float(col.iloc[-1]) if not col.empty else 0.0
            elif not self.prices.empty and self.prices.ndim == 1:
                price = float(self.prices.dropna().iloc[-1]) if not self.prices.dropna().empty else 0.0
            mv = qty * price
            weight = (mv / self.total_value * 100) if self.total_value > 0 else 0
            gl = float(row.get('gain_loss', 0) or 0)
            gl_pct = float(row.get('gain_loss_pct', 0) or 0)
            cb = float(row.get('cost_basis', 0) or 0)
            rows.append({"symbol": sym, "quantity": qty, "price": price, "market_value": mv, "weight": weight, "cost_basis": cb, "gain_loss": gl, "gain_loss_pct": gl_pct})
        return sorted(rows, key=lambda x: x['market_value'], reverse=True)

    def drawdown_series(self):
        if self.returns_series.empty: return pd.Series(dtype=float)
        cum = (1 + self.returns_series).cumprod()
        return (cum - cum.cummax()) / cum.cummax()

    def performance_attribution(self):
        if self.holdings.empty or self.prices.empty: return []
        results = []
        for _, row in self.holdings.iterrows():
            sym = row['symbol']
            qty = float(row.get('quantity', 0) or 0)
            if self.prices.ndim == 2 and sym in self.prices.columns:
                col = self.prices[sym].dropna()
                if len(col) >= 2:
                    ret = float(col.iloc[-1] / col.iloc[0] - 1)
                    mv = qty * float(col.iloc[-1])
                    weight = mv / self.total_value if self.total_value > 0 else 0
                    results.append({"symbol": sym, "return": ret, "weight": weight, "contribution": ret * weight})
        return sorted(results, key=lambda x: abs(x['contribution']), reverse=True)

    def sector_breakdown(self):
        if self.holdings.empty: return {}
        sectors = {}
        for _, row in self.holdings.iterrows():
            try:
                info = yf.Ticker(row['symbol']).info
                sector = info.get('sector', 'Unknown') or 'Unknown'
            except Exception:
                sector = 'Unknown'
            qty = float(row.get('quantity', 0) or 0)
            price = 0
            if not self.prices.empty and self.prices.ndim == 2 and row['symbol'] in self.prices.columns:
                col = self.prices[row['symbol']].dropna()
                price = float(col.iloc[-1]) if not col.empty else 0
            sectors[sector] = sectors.get(sector, 0) + qty * price
        total = sum(sectors.values())
        return {k: {"value": v, "pct": v / total * 100 if total > 0 else 0} for k, v in sorted(sectors.items(), key=lambda x: -x[1])}

    def chart_drawdown(self, show_dollar=False, show_underwater=False, show_recovery_bands=False):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        dd = self.drawdown_series()
        if not dd.empty:
            if show_dollar and self.total_value > 0:
                cum = (1 + self.returns_series).cumprod()
                scale = self.total_value / cum.iloc[-1] if cum.iloc[-1] != 0 else 1
                cum_d = cum * scale
                dd_vals = cum_d - cum_d.cummax()
                ax.fill_between(dd_vals.index, dd_vals.values, 0, color="#ff3366", alpha=0.4)
                ax.plot(dd_vals.index, dd_vals.values, color="#ff3366", linewidth=1.5)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
                title_extra = " (Dollar)"
            else:
                ax.fill_between(dd.index, dd.values, 0, color="#ff3366", alpha=0.4)
                ax.plot(dd.index, dd.values, color="#ff3366", linewidth=1.5)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
                title_extra = " (%)"
            # Underwater periods shading
            if show_underwater:
                in_dd = dd < -0.001
                start = None
                for i, (idx, val) in enumerate(zip(dd.index, in_dd)):
                    if val and start is None:
                        start = idx
                    elif not val and start is not None:
                        ax.axvspan(start, idx, alpha=0.08, color="#ff9900")
                        start = None
            # Recovery bands (mark max drawdown point)
            if show_recovery_bands:
                min_dd_idx = dd.idxmin()
                min_dd_val = dd.min()
                ax.axvline(min_dd_idx, color="#ffd700", linewidth=1, linestyle=":", alpha=0.7)
                ax.annotate(f"Max DD: {min_dd_val:.1%}", xy=(min_dd_idx, min_dd_val),
                            xytext=(10, -15), textcoords='offset points',
                            color="#ffd700", fontsize=9, fontweight="bold",
                            arrowprops=dict(arrowstyle='->', color='#ffd700', lw=1))
                # Mark recovery points (where DD returns to 0 after significant drops)
                recovered = (dd.shift(1) < -0.02) & (dd >= -0.001)
                for idx in dd.index[recovered]:
                    ax.axvline(idx, color="#00ff9d", linewidth=0.5, linestyle=":", alpha=0.4)
        else:
            ax.text(0.5, 0.5, "Import data to see drawdown", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
            title_extra = ""
        ax.set_title(f"DRAWDOWN CHART{title_extra}", color="#ff3366", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        ax.grid(True, alpha=0.1, color='#555555')
        for spine in ax.spines.values(): spine.set_color("#333333")
        return self._fig_to_base64(fig)

    def chart_holdings_bar(self, mode='$'):
        fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(self.holdings_with_weights() or []))))
        ax.set_facecolor("#1a1a2e")
        hw = self.holdings_with_weights()
        if hw:
            syms = [r['symbol'] for r in hw]
            if mode == '%':
                vals = [r['weight'] for r in hw]
                title_text = "HOLDINGS BY WEIGHT (%)"
                fmt_func = lambda x, _: f"{x:.1f}%"
            elif mode == 'gl':
                vals = [r['gain_loss'] for r in hw]
                title_text = "HOLDINGS BY GAIN/LOSS ($)"
                fmt_func = lambda x, _: f"${x:,.0f}"
            elif mode == 'gl%':
                vals = [r['gain_loss_pct'] for r in hw]
                title_text = "HOLDINGS BY GAIN/LOSS (%)"
                fmt_func = lambda x, _: f"{x:+.1f}%"
            elif mode == 'cb':
                vals = [r['cost_basis'] for r in hw]
                title_text = "HOLDINGS BY COST BASIS ($)"
                fmt_func = lambda x, _: f"${x:,.0f}"
            else:
                vals = [r['market_value'] for r in hw]
                title_text = "HOLDINGS BY MARKET VALUE ($)"
                fmt_func = lambda x, _: f"${x:,.0f}"
            if mode in ('gl', 'gl%'):
                colors = ["#00ff9d" if v >= 0 else "#ff3366" for v in vals]
            else:
                colors = ["#00ff9d" if r.get('gain_loss', 0) >= 0 else "#ff3366" for r in hw]
            bars = ax.barh(syms[::-1], vals[::-1], color=colors[::-1], edgecolor="#333333")
            for bar, val in zip(bars, vals[::-1]):
                if mode == '%':
                    label = f"{val:.1f}%"
                elif mode == 'gl':
                    label = f"${val:+,.0f}"
                elif mode == 'gl%':
                    label = f"{val:+.1f}%"
                elif mode == 'cb':
                    label = f"${val:,.0f}"
                else:
                    label = f"${val:,.0f}"
                x_pos = bar.get_width() + abs(max(vals, key=abs)) * 0.01 if bar.get_width() >= 0 else bar.get_width() - abs(max(vals, key=abs)) * 0.05
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        label, va='center', color='#aaaaaa', fontsize=8)
            if mode in ('gl', 'gl%'):
                ax.axvline(0, color="#ffffff", linewidth=0.5, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Import data to see holdings", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
            title_text = "HOLDINGS"
            fmt_func = lambda x, _: f"${x:,.0f}"
        ax.set_title(title_text, color="#ffd700", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_func))
        for spine in ax.spines.values(): spine.set_color("#333333")
        return self._fig_to_base64(fig)

    def chart_returns_histogram(self, show_kde=False, show_cvar=False, show_normal=False, log_returns=False):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        if not self.returns_series.empty:
            rets = np.log(1 + self.returns_series) if log_returns else self.returns_series
            ax.hist(rets.values, bins=50, color="#00ff9d", alpha=0.7, edgecolor="#333333", density=show_kde or show_normal)
            ax.axvline(rets.mean(), color="#ffd700", linewidth=2, linestyle="--", label=f"Mean: {rets.mean():.4f}")
            ax.axvline(0, color="#ffffff", linewidth=1, alpha=0.5)
            var = self.value_at_risk()
            ax.axvline(var, color="#ff3366", linewidth=2, linestyle="--", label=f"VaR 95%: {var:.4f}")
            if show_cvar:
                cvar = self.conditional_var()
                ax.axvline(cvar, color="#ff9900", linewidth=2, linestyle="-.", label=f"CVaR 95%: {cvar:.4f}")
            if show_kde:
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(rets.dropna().values)
                    x_range = np.linspace(rets.min(), rets.max(), 200)
                    ax.plot(x_range, kde(x_range), color="#00bfff", linewidth=2, label="KDE")
                except Exception:
                    pass
            if show_normal:
                from scipy.stats import norm
                x_range = np.linspace(rets.min(), rets.max(), 200)
                ax.plot(x_range, norm.pdf(x_range, rets.mean(), rets.std()), color="#9966ff", linewidth=1.5, linestyle=":", label="Normal fit")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Import data to see returns distribution", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        title = "LOG RETURNS DISTRIBUTION" if log_returns else "RETURNS DISTRIBUTION"
        ax.set_title(title, color="#00ff9d", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values(): spine.set_color("#333333")
        return self._fig_to_base64(fig)

    def chart_sector_pie(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#121212")
        sectors = self.sector_breakdown()
        if sectors and any(v['value'] > 0 for v in sectors.values()):
            labels = list(sectors.keys())
            sizes = [v['value'] for v in sectors.values()]
            palette = ["#00ff9d","#ffd700","#ff3366","#00bfff","#ff9900","#9966ff","#ff6699","#33ccff","#ffcc00","#66ff66"]
            colors = [palette[i % len(palette)] for i in range(len(labels))]
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'color': '#ffffff', 'fontsize': 10})
        else:
            ax.text(0.5, 0.5, "Import data for sector breakdown", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("SECTOR ALLOCATION", color="#ffd700", fontsize=14, fontweight="bold")
        return self._fig_to_base64(fig)

    def chart_rolling_sharpe(self, window=60, show_sortino=False, show_avg=False):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        rs = self.rolling_sharpe(window)
        if not rs.empty:
            ax.plot(rs.index, rs.values, color="#00ff9d", linewidth=1.5, label=f"{window}d Sharpe")
            ax.axhline(0, color="#ff3366", linewidth=1, linestyle="--", alpha=0.7)
            ax.axhline(1, color="#ffd700", linewidth=1, linestyle="--", alpha=0.5, label="Good (1.0)")
            ax.fill_between(rs.index, rs.values, 0, where=(rs.values > 0), alpha=0.15, color="#00ff9d")
            ax.fill_between(rs.index, rs.values, 0, where=(rs.values < 0), alpha=0.15, color="#ff3366")
            if show_avg:
                avg_val = rs.mean()
                ax.axhline(avg_val, color="#00bfff", linewidth=1, linestyle="-.", alpha=0.7, label=f"Avg: {avg_val:.2f}")
            if show_sortino and not self.returns_series.empty:
                neg_ret = self.returns_series.copy()
                neg_ret[neg_ret > 0] = 0
                roll_down = neg_ret.rolling(window).std() * np.sqrt(252)
                roll_mean = self.returns_series.rolling(window).mean() * 252
                roll_sort = (roll_mean / roll_down).replace([np.inf, -np.inf], np.nan).dropna()
                if not roll_sort.empty:
                    ax.plot(roll_sort.index, roll_sort.values, color="#ff9900", linewidth=1, alpha=0.7, label=f"{window}d Sortino")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"Need {window}+ days for rolling Sharpe", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title(f"ROLLING SHARPE RATIO ({window}-day)", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values(): spine.set_color("#333333")
        return self._fig_to_base64(fig)

    def chart_rolling_volatility(self, window=30, show_bands=False, show_regime=False):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        rv = self.rolling_volatility(window)
        if not rv.empty:
            ax.plot(rv.index, rv.values, color="#ff9900", linewidth=1.5, label=f"{window}d Vol")
            ax.fill_between(rv.index, rv.values, alpha=0.2, color="#ff9900")
            avg_vol = rv.mean()
            ax.axhline(avg_vol, color="#ffd700", linewidth=1, linestyle="--", label=f"Avg: {avg_vol:.1%}")
            if show_bands:
                std_vol = rv.std()
                ax.axhline(avg_vol + std_vol, color="#ff3366", linewidth=0.8, linestyle=":", alpha=0.6, label=f"+1σ: {avg_vol+std_vol:.1%}")
                ax.axhline(max(0, avg_vol - std_vol), color="#00ff9d", linewidth=0.8, linestyle=":", alpha=0.6, label=f"-1σ: {max(0,avg_vol-std_vol):.1%}")
            if show_regime:
                high_thresh = avg_vol + rv.std()
                low_thresh = max(0, avg_vol - rv.std() * 0.5)
                ax.fill_between(rv.index, 0, rv.values, where=(rv.values > high_thresh), alpha=0.1, color="#ff3366", label="High Vol")
                ax.fill_between(rv.index, 0, rv.values, where=(rv.values < low_thresh), alpha=0.1, color="#00ff9d", label="Low Vol")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Need 30+ days for rolling volatility", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("ROLLING VOLATILITY (30-day)", color="#ff9900", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values(): spine.set_color("#333333")
        return self._fig_to_base64(fig)

    def chart_monthly_heatmap(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor("#1a1a2e")
        if not self.returns_series.empty:
            monthly = self.returns_series.resample('ME').sum()
            if len(monthly) > 3:
                years = sorted(monthly.index.year.unique())
                grid = np.full((len(years), 12), np.nan)
                for dt, ret in monthly.items():
                    grid[years.index(dt.year), dt.month - 1] = ret
                im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
                ax.set_xticks(range(12))
                ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], color="#aaaaaa", fontsize=9)
                ax.set_yticks(range(len(years)))
                ax.set_yticklabels(years, color="#aaaaaa", fontsize=9)
                for i in range(len(years)):
                    for j in range(12):
                        v = grid[i, j]
                        if not np.isnan(v):
                            ax.text(j, i, f"{v:.1%}", ha="center", va="center", color="white", fontsize=8)
                fig.colorbar(im, ax=ax, shrink=0.7)
            else:
                ax.text(0.5, 0.5, "Need more data for heatmap", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Import data for monthly heatmap", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("MONTHLY RETURNS HEATMAP", color="#ffd700", fontsize=14, fontweight="bold")
        return self._fig_to_base64(fig)

    # ====================== ENGINE METHODS ======================
    def profit_factor(self):
        if self.returns_series.empty: return 0.0
        gains = self.returns_series[self.returns_series > 0].sum()
        losses = abs(self.returns_series[self.returns_series < 0].sum())
        return float(gains / losses) if losses > 0 else 0.0

    def win_loss_streaks(self):
        if self.returns_series.empty:
            return {"longest_win": 0, "longest_loss": 0, "current_streak": 0, "current_type": "N/A"}
        signs = (self.returns_series > 0).astype(int)
        max_win, max_loss, cur, cur_type = 0, 0, 0, "N/A"
        for v in signs:
            if v == 1:
                if cur_type == "win":
                    cur += 1
                else:
                    cur_type, cur = "win", 1
                max_win = max(max_win, cur)
            else:
                if cur_type == "loss":
                    cur += 1
                else:
                    cur_type, cur = "loss", 1
                max_loss = max(max_loss, cur)
        return {"longest_win": max_win, "longest_loss": max_loss, "current_streak": cur, "current_type": cur_type}

    def recovery_time(self):
        if self.returns_series.empty: return {"max_recovery_days": 0, "currently_in_drawdown": False, "current_dd_days": 0}
        cum = (1 + self.returns_series).cumprod()
        peak = cum.cummax()
        in_dd = cum < peak
        max_rec, cur_dd = 0, 0
        for v in in_dd:
            if v:
                cur_dd += 1
            else:
                max_rec = max(max_rec, cur_dd)
                cur_dd = 0
        return {"max_recovery_days": max_rec, "currently_in_drawdown": bool(in_dd.iloc[-1]) if len(in_dd) > 0 else False, "current_dd_days": cur_dd}

    def portfolio_health_score(self):
        score = 50
        details = {}
        if not self.returns_series.empty:
            sharpe = float(qs.stats.sharpe(self.returns_series) or 0)
            sortino = float(qs.stats.sortino(self.returns_series) or 0)
            vol = float(self.returns_series.std() * np.sqrt(252))
            dd = abs(self.max_drawdown())
            pf = self.profit_factor()
            wl = self.win_loss_streaks()
            sharpe_score = min(20, max(0, int(sharpe * 10)))
            sortino_score = min(15, max(0, int(sortino * 7)))
            vol_score = min(15, max(0, int((0.5 - vol) * 30)))
            dd_score = min(15, max(0, int((0.3 - dd) * 50)))
            pf_score = min(15, max(0, int(pf * 7)))
            n = len(self.holdings) if not self.holdings.empty else 0
            div_score = min(10, n)
            streak_score = min(10, wl['longest_win'])
            score = max(0, min(100, sharpe_score + sortino_score + vol_score + dd_score + pf_score + div_score + streak_score))
            details = {"sharpe_pts": sharpe_score, "sortino_pts": sortino_score, "vol_pts": vol_score,
                        "dd_pts": dd_score, "pf_pts": pf_score, "div_pts": div_score, "streak_pts": streak_score,
                        "sharpe": sharpe, "sortino": sortino, "vol": vol, "max_dd": dd, "profit_factor": pf}
        else:
            details = {"sharpe_pts": 0, "sortino_pts": 0, "vol_pts": 0, "dd_pts": 0, "pf_pts": 0, "div_pts": 0, "streak_pts": 0,
                        "sharpe": 0, "sortino": 0, "vol": 0, "max_dd": 0, "profit_factor": 0}
        grade = "A+" if score >= 90 else "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "F"
        return {"score": score, "grade": grade, "details": details}

    def correlation_deep_dive(self):
        if self.prices.empty or self.prices.ndim != 2 or len(self.prices.columns) < 2:
            return {"pairs": [], "avg_corr": 0, "most_correlated": ("N/A", "N/A", 0), "least_correlated": ("N/A", "N/A", 0)}
        corr = self.prices.corr()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                c = float(corr.iloc[i, j])
                pairs.append({"a": corr.columns[i], "b": corr.columns[j], "corr": c})
        pairs.sort(key=lambda x: abs(x['corr']), reverse=True)
        avg_c = sum(p['corr'] for p in pairs) / max(1, len(pairs))
        most = (pairs[0]['a'], pairs[0]['b'], pairs[0]['corr']) if pairs else ("N/A", "N/A", 0)
        least = (pairs[-1]['a'], pairs[-1]['b'], pairs[-1]['corr']) if pairs else ("N/A", "N/A", 0)
        return {"pairs": pairs[:20], "avg_corr": avg_c, "most_correlated": most, "least_correlated": least}

    def risk_adjusted_returns(self):
        s = {}
        if not self.returns_series.empty:
            s["ann_return"] = float(self.returns_series.mean() * 252)
            s["ann_vol"] = float(self.returns_series.std() * np.sqrt(252))
            s["sharpe"] = float(qs.stats.sharpe(self.returns_series) or 0)
            s["sortino"] = float(qs.stats.sortino(self.returns_series) or 0)
            try: s["calmar"] = float(qs.stats.calmar(self.returns_series) or 0)
            except Exception: s["calmar"] = 0
            s["max_dd"] = self.max_drawdown()
            s["profit_factor"] = self.profit_factor()
            s["var_95"] = self.value_at_risk(0.95)
            s["cvar_95"] = self.conditional_var(0.95)
            s["beta"] = self.beta_vs_market()
            alpha = s["ann_return"] - (0.02 + s["beta"] * (0.10 - 0.02))
            s["alpha"] = alpha
            s["treynor"] = (s["ann_return"] - 0.02) / s["beta"] if s["beta"] != 0 else 0
            s["info_ratio"] = s["ann_return"] / s["ann_vol"] if s["ann_vol"] > 0 else 0
        else:
            for k in ["ann_return","ann_vol","sharpe","sortino","calmar","max_dd","profit_factor","var_95","cvar_95","beta","alpha","treynor","info_ratio"]:
                s[k] = 0
        return s

    def chart_health_gauge(self, score):
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        theta = np.linspace(np.pi, 0, 100)
        for i, (lo, hi, col) in enumerate([(0, 35, '#ff3366'), (35, 50, '#ff9900'), (50, 75, '#ffd700'), (75, 100, '#00ff9d')]):
            t1 = np.pi - (lo / 100 * np.pi)
            t2 = np.pi - (hi / 100 * np.pi)
            seg = np.linspace(t1, t2, 30)
            ax.fill_between(np.cos(seg) * 1.0, np.sin(seg) * 0.8, np.sin(seg) * 1.0, color=col, alpha=0.3)
        needle_angle = np.pi - (score / 100 * np.pi)
        ax.plot([0, np.cos(needle_angle) * 0.85], [0, np.sin(needle_angle) * 0.85], color='#ffffff', linewidth=3, solid_capstyle='round')
        ax.plot(0, 0, 'o', color='#ffffff', markersize=6)
        grade = "A+" if score >= 90 else "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "F"
        gc = '#00ff9d' if score >= 65 else '#ffd700' if score >= 50 else '#ff3366'
        ax.text(0, -0.25, f"{score}/100  ({grade})", ha='center', fontsize=18, fontweight='bold', color=gc)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.4, 1.15)
        ax.axis('off')
        return self._fig_to_base64(fig)

    def chart_daily_profit(self, mode='$', start_idx=0, end_idx=None, cumulative=False, chart_style='bar', show_avg=False):
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#121212')
        if not self.returns_series.empty:
            if mode == '%':
                daily_pnl = self.returns_series * 100
                ylabel_fmt = lambda x, _: f"{x:.2f}%"
                title_suffix = "(%)"
            else:
                daily_pnl = self.returns_series * self.total_value if self.total_value > 0 else self.returns_series
                ylabel_fmt = lambda x, _: f"${x:,.0f}"
                title_suffix = "($)"
            if end_idx is None:
                end_idx = len(daily_pnl)
            daily_pnl = daily_pnl.iloc[start_idx:end_idx]
            if cumulative:
                cum_pnl = daily_pnl.cumsum()
                title_suffix = f"Cumulative {title_suffix}"
                if chart_style == 'line':
                    ax.plot(cum_pnl.index, cum_pnl.values, color="#00ff9d", linewidth=2)
                    ax.fill_between(cum_pnl.index, cum_pnl.values, alpha=0.1, color="#00ff9d")
                elif chart_style == 'area':
                    ax.fill_between(cum_pnl.index, 0, cum_pnl.values,
                                    where=(cum_pnl.values >= 0), alpha=0.4, color="#00ff9d")
                    ax.fill_between(cum_pnl.index, 0, cum_pnl.values,
                                    where=(cum_pnl.values < 0), alpha=0.4, color="#ff3366")
                    ax.plot(cum_pnl.index, cum_pnl.values, color="#ffffff", linewidth=1, alpha=0.5)
                else:
                    colors = ['#00ff9d' if v >= 0 else '#ff3366' for v in cum_pnl.values]
                    ax.bar(cum_pnl.index, cum_pnl.values, color=colors, alpha=0.7, width=1.0)
                ax.axhline(0, color='#555555', linewidth=0.5)
            else:
                if chart_style == 'line':
                    ax.plot(daily_pnl.index, daily_pnl.values, color="#00ff9d", linewidth=1.5, alpha=0.8)
                    ax.fill_between(daily_pnl.index, daily_pnl.values, 0,
                                    where=(daily_pnl.values >= 0), alpha=0.15, color="#00ff9d")
                    ax.fill_between(daily_pnl.index, daily_pnl.values, 0,
                                    where=(daily_pnl.values < 0), alpha=0.15, color="#ff3366")
                elif chart_style == 'area':
                    ax.fill_between(daily_pnl.index, 0, daily_pnl.values,
                                    where=(daily_pnl.values >= 0), alpha=0.5, color="#00ff9d")
                    ax.fill_between(daily_pnl.index, 0, daily_pnl.values,
                                    where=(daily_pnl.values < 0), alpha=0.5, color="#ff3366")
                else:
                    colors = ['#00ff9d' if v >= 0 else '#ff3366' for v in daily_pnl.values]
                    ax.bar(daily_pnl.index, daily_pnl.values, color=colors, alpha=0.7, width=1.0)
                ax.axhline(0, color='#555555', linewidth=0.5)
            if show_avg and len(daily_pnl) > 5:
                avg_val = daily_pnl.mean()
                ax.axhline(avg_val, color="#ffd700", linewidth=1, linestyle="--", alpha=0.7,
                           label=f"Avg: {avg_val:+.2f}")
                ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa", fontsize=8)
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate(rotation=45)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(ylabel_fmt))
        else:
            ax.text(0.5, 0.5, "Import data to see daily profit", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
            title_suffix = ""
        ax.set_title(f"DAILY PROFIT / LOSS {title_suffix}", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values(): spine.set_color('#333333')
        ax.grid(True, alpha=0.1, color='#555555')
        return self._fig_to_base64(fig)

    def chart_weekly_profit(self):
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#121212')
        if not self.returns_series.empty and len(self.returns_series) >= 5:
            weekly = self.returns_series.resample('W').sum()
            weekly_pnl = weekly * self.total_value if self.total_value > 0 else weekly
            colors = ['#00ff9d' if v >= 0 else '#ff3366' for v in weekly_pnl.values]
            ax.bar(range(len(weekly_pnl)), weekly_pnl.values, color=colors, alpha=0.8)
            ax.axhline(0, color='#555555', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, "Import data to see weekly profit", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("WEEKLY PROFIT / LOSS", color="#ffd700", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values(): spine.set_color('#333333')
        ax.grid(True, alpha=0.1, color='#555555')
        return self._fig_to_base64(fig)

    def chart_rolling_weekly_profit(self):
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#121212')
        if not self.returns_series.empty and len(self.returns_series) >= 10:
            rolling_w = self.returns_series.rolling(5).sum()
            rolling_pnl = rolling_w * self.total_value if self.total_value > 0 else rolling_w
            rolling_pnl = rolling_pnl.dropna()
            ax.plot(rolling_pnl.index, rolling_pnl.values, color='#00bfff', linewidth=1.5)
            ax.fill_between(rolling_pnl.index, rolling_pnl.values, 0, where=rolling_pnl.values >= 0, color='#00ff9d', alpha=0.3)
            ax.fill_between(rolling_pnl.index, rolling_pnl.values, 0, where=rolling_pnl.values < 0, color='#ff3366', alpha=0.3)
            ax.axhline(0, color='#555555', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, "Import data to see rolling weekly profit", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("ROLLING 5-DAY (WEEKLY) PROFIT", color="#00bfff", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values(): spine.set_color('#333333')
        ax.grid(True, alpha=0.1, color='#555555')
        return self._fig_to_base64(fig)

    def holding_period_analysis(self):
        if self.master_df.empty or 'date' not in self.master_df.columns: return {}
        df = self.master_df.sort_values('date'); f = df['date'].min(); l = df['date'].max()
        d = (l-f).days if pd.notna(f) and pd.notna(l) else 0
        return {'first': str(f.date()) if pd.notna(f) else 'N/A', 'last': str(l.date()) if pd.notna(l) else 'N/A', 'days': d, 'years': round(d/365.25,1)}

    def trade_frequency_analysis(self):
        if self.master_df.empty or 'date' not in self.master_df.columns: return {}
        df = self.master_df.sort_values('date'); daily = df.groupby(df['date'].dt.date).size()
        return {'total': len(df), 'avg_daily': float(daily.mean()) if len(daily)>0 else 0, 'max_daily': int(daily.max()) if len(daily)>0 else 0}

    def day_of_week_returns(self):
        if self.returns_series.empty: return {}
        r = self.returns_series.dropna(); by = r.groupby(r.index.dayofweek).mean()*252
        return {['Mon','Tue','Wed','Thu','Fri'][i]: float(by.iloc[i]) for i in range(min(5,len(by)))}

    def monthly_seasonality(self):
        if self.returns_series.empty: return {}
        r = self.returns_series.dropna(); by = r.groupby(r.index.month).mean()*12
        mo = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        return {mo[m-1]: float(by.loc[m]) for m in by.index if 1<=m<=12}

    def best_worst_periods(self, period='M'):
        if self.returns_series.empty: return {'best':{},'worst':{}}
        r = self.returns_series.resample(period).sum().dropna()
        if r.empty: return {'best':{},'worst':{}}
        return {'best': {'period': str(r.idxmax().date()), 'return': float(r.max())}, 'worst': {'period': str(r.idxmin().date()), 'return': float(r.min())}}

    def close(self):
        self.conn.close()

def _card(content, shadow="#00ff9d"):
    return ft.Card(
        content=ft.Container(
            content=content,
            padding=15,
            bgcolor="#121212",
            border_radius=16,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        ),
        elevation=8,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
    )

def _chart_image(b64_data):
    if not b64_data:
        return ft.Text("No chart data available. Import data first.", size=14, color="#aaaaaa")
    from flet.controls.box import BoxFit
    img = ft.Image(src=f"data:image/png;base64,{b64_data}", fit=BoxFit.CONTAIN)
    return img

def _chart_image_expandable(b64_data, page_ref=None, title="Chart"):
    """Chart image with expand-to-fullscreen button."""
    if not b64_data:
        return ft.Text("No chart data available. Import data first.", size=14, color="#aaaaaa")
    from flet.controls.box import BoxFit
    img_small = ft.Image(src=f"data:image/png;base64,{b64_data}", fit=BoxFit.CONTAIN)

    def open_fullscreen(e):
        if page_ref is None:
            return
        img_big = ft.Image(src=f"data:image/png;base64,{b64_data}", fit=BoxFit.CONTAIN, expand=True)
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(title, size=20, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            content=ft.Container(content=img_big, width=900, height=600, bgcolor="#0a0a0a", border_radius=12, padding=10),
            actions=[ft.TextButton("CLOSE", on_click=lambda e2: close_dlg(e2))],
            actions_alignment=ft.MainAxisAlignment.END,
            bgcolor="#121212",
        )
        def close_dlg(e2):
            dlg.open = False
            page_ref.update()
        page_ref.overlay.append(dlg)
        dlg.open = True
        page_ref.update()

    expand_btn = ft.IconButton(icon=ft.Icons.FULLSCREEN, icon_color="#00ff9d", icon_size=20,
                                tooltip="Expand chart", on_click=open_fullscreen)
    return ft.Column([
        ft.Row([ft.Container(expand=True), expand_btn], spacing=0),
        img_small,
    ], spacing=0)

def main(page: ft.Page):
    # ====================== THEME ======================
    page.title = "ULTRA PORTFOLIO MANAGER V1.17"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(
        color_scheme_seed="#00ff9d",
        font_family="Roboto",
        text_theme=ft.TextTheme(
            title_large=ft.TextStyle(size=52, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            body_large=ft.TextStyle(size=18, color="#ffffff"),
        ),
        visual_density=ft.VisualDensity.COMFORTABLE,
    )
    page.padding = 0
    page.bgcolor = "#0a0a0a"

    data = UltraDataEngine()

    # ====================== EXPANDABLE CHART (page-aware) ======================
    def _chart_img(b64_data, title="Chart"):
        """Page-aware chart image with fullscreen expand button."""
        if not b64_data:
            return ft.Text("No chart data available. Import data first.", size=14, color="#aaaaaa")
        from flet.controls.box import BoxFit
        img_small = ft.Image(src=f"data:image/png;base64,{b64_data}", fit=BoxFit.CONTAIN)
        def open_fullscreen(e):
            img_big = ft.Image(src=f"data:image/png;base64,{b64_data}", fit=BoxFit.CONTAIN, expand=True)
            def _close_dlg(e2):
                dlg.open = False
                page.update()
            # Use page dimensions to make truly fullscreen
            pw = page.width or 1200
            ph = page.height or 800
            dlg = ft.AlertDialog(
                modal=True,
                title=ft.Text(title, size=20, weight=ft.FontWeight.BOLD, color="#00ff9d"),
                content=ft.Container(
                    content=img_big,
                    width=pw - 60,
                    height=ph - 120,
                    bgcolor="#0a0a0a",
                    border_radius=12,
                    padding=10,
                    expand=True,
                ),
                actions=[ft.TextButton("CLOSE", on_click=_close_dlg)],
                actions_alignment=ft.MainAxisAlignment.END,
                bgcolor="#121212",
            )
            page.overlay.append(dlg)
            dlg.open = True
            page.update()
        expand_btn = ft.IconButton(icon=ft.Icons.FULLSCREEN, icon_color="#00ff9d", icon_size=18,
                                    tooltip="View full size", on_click=open_fullscreen)
        return ft.Column([
            ft.Row([ft.Container(expand=True), expand_btn], spacing=0),
            img_small,
        ], spacing=0)

    # Shadow the global _chart_image so ALL charts get fullscreen expand
    _chart_image = _chart_img

    # ====================== SNACKBAR HELPER ======================
    snack = ft.SnackBar(content=ft.Text(""), bgcolor="#121212")

    def show_snack(msg, color="#121212"):
        snack.content = ft.Text(msg, color="#ffffff")
        snack.bgcolor = color
        snack.open = True
        page.update()

    # ====================== LIVE TICKER ======================
    live_value = ft.Text(f"${int(data.total_value):,}", size=24, weight=ft.FontWeight.BOLD, color="#00ff9d")
    ticker_running = {"flag": False}

    def update_live_ticker():
        if ticker_running["flag"]:
            return
        ticker_running["flag"] = True
        while True:
            try:
                live_value.value = f"${int(data.total_value):,}"
                live_value.color = "#00ff9d"
                page.update()
                time.sleep(5)
            except Exception:
                break
        ticker_running["flag"] = False

    # ====================== IMPORT (File Browser + manual path) ======================
    folder_field = ft.TextField(
        label="Path (folder or file)...", expand=True, border_color="#00ff9d",
        color="#ffffff", text_size=14, dense=True,
    )
    file_list_col = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO, height=200)
    file_checkboxes = {}  # {filepath: Checkbox}
    SUPPORTED_EXT = {'.csv', '.tsv', '.txt', '.xlsx', '.xls', '.json', '.parquet', '.feather', '.html'}

    def scan_folder(e=None):
        """Scan the given folder (or app folder) for importable files."""
        raw = (folder_field.value or "").strip()
        search_dir = raw if raw and os.path.isdir(raw) else os.path.dirname(os.path.abspath(__file__))
        if not raw:
            folder_field.value = search_dir
        file_list_col.controls.clear()
        file_checkboxes.clear()
        found = []
        try:
            for root, _, files in os.walk(search_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_EXT:
                        fp = os.path.join(root, f)
                        found.append((f, fp, ext, os.path.getsize(fp)))
                # Only scan top-level, not deep subdirs
                break
        except Exception:
            pass
        if not found:
            file_list_col.controls.append(ft.Text("No importable files found in this folder.", color="#888888", size=12))
        else:
            sel_all_cb = ft.Checkbox(label=f"Select All ({len(found)} files)", value=False, fill_color="#00ff9d",
                                      check_color="#0a0a0a", label_text_style=ft.TextStyle(color="#ffd700", size=12))
            def toggle_all(e2):
                for cb in file_checkboxes.values():
                    cb.value = sel_all_cb.value
                page.update()
            sel_all_cb.on_change = toggle_all
            file_list_col.controls.append(sel_all_cb)
            for fname, fpath, ext, size in sorted(found, key=lambda x: x[0]):
                size_str = f"{size/1024:.0f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
                cb = ft.Checkbox(label=f"{fname}  ({ext}, {size_str})", value=False, fill_color="#00ff9d",
                                  check_color="#0a0a0a", label_text_style=ft.TextStyle(color="#ffffff", size=11))
                file_checkboxes[fpath] = cb
                file_list_col.controls.append(cb)
        page.update()

    import_status = ft.Row([
        ft.ProgressRing(width=20, height=20, stroke_width=2, color="#00ff9d", visible=False),
        ft.Text("", size=12, color="#00ff9d", italic=True),
    ], spacing=8, visible=False)

    # Main screen overlay for loading during imports
    main_loading_overlay = ft.Container(
        content=ft.Column([
            ft.ProgressRing(width=60, height=60, stroke_width=4, color="#00ff9d"),
            ft.Text("Importing data...", size=20, color="#00ff9d", weight=ft.FontWeight.BOLD),
            ft.Text("Please wait while files are processed", size=14, color="#888888"),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER, spacing=16),
        bgcolor=ft.Colors.with_opacity(0.85, "#0a0a0a"),
        expand=True,
        visible=False,
        alignment=ft.Alignment(0, 0),
    )

    def do_import(e):
        # Collect checked files, or fall back to manual path
        selected = [fp for fp, cb in file_checkboxes.items() if cb.value]
        raw = (folder_field.value or "").strip()
        if not selected and not raw:
            show_snack("Select files or enter a folder path to import.", "#663300")
            return
        # Show loading indicator on import bar AND main screen overlay
        import_status.controls[0].visible = True
        import_status.controls[1].value = f"Importing {len(selected) if selected else 'all'} file(s)..."
        import_status.visible = True
        main_loading_overlay.visible = True
        page.update()
        def thread_func():
            try:
                if selected:
                    total = 0
                    for i, fp in enumerate(selected):
                        import_status.controls[1].value = f"Importing file {i+1}/{len(selected)}..."
                        page.update()
                        total += data._import_single_file(fp)
                    count = total
                else:
                    import_status.controls[1].value = "Scanning folder and importing..."
                    page.update()
                    count = data.import_multi(raw)
                import_status.controls[1].value = "Processing data..."
                page.update()
                if count > 0:
                    summary = data.get_import_summary()
                    files_n = len(summary)
                    fmts = set(s['format'] for s in summary)
                    import_status.controls[0].visible = False
                    import_status.controls[1].value = f"✓ Imported {count:,} records from {files_n} file(s)"
                    main_loading_overlay.visible = False
                    page.update()
                    show_snack(f"Imported {count:,} records from {files_n} file(s) ({', '.join(fmts)})", "#006644")
                    load_category(nav_rail.selected_index)
                else:
                    import_status.controls[0].visible = False
                    import_status.controls[1].value = "No valid data found."
                    import_status.controls[1].color = "#ff3366"
                    main_loading_overlay.visible = False
                    page.update()
                    show_snack("No valid data found in selected files.", "#663300")
            except Exception as ex:
                import_status.controls[0].visible = False
                import_status.controls[1].value = f"Error: {ex}"
                import_status.controls[1].color = "#ff3366"
                main_loading_overlay.visible = False
                page.update()
                show_snack(f"Import error: {ex}", "#663300")
        threading.Thread(target=thread_func, daemon=True).start()

    # Auto-scan on startup
    try:
        scan_folder()
    except Exception:
        pass

    # ====================== APPBAR ======================
    header = ft.AppBar(
        title=ft.Row([
            ft.Icon(ft.Icons.TRENDING_UP, color="#ffd700", size=30),
            ft.Text("ULTRA PORTFOLIO", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text(" V1.17", size=14, weight=ft.FontWeight.BOLD, color="#ffd700"),
        ], spacing=6),
        bgcolor="#121212",
        actions=[
            ft.Container(
                content=ft.Column([
                    ft.Text("LIVE VALUE", size=9, color="#aaaaaa"),
                    live_value,
                ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.END),
                padding=ft.Padding(left=0, top=0, right=10, bottom=0),
            ),
        ],
        elevation=8,
    )

    # ====================== ABOUT PAGE (full) ======================
    def build_about_page():
        col = ft.Column(spacing=12, scroll=ft.ScrollMode.AUTO, expand=True)
        col.controls.append(ft.Text("ULTRA PORTFOLIO MANAGER", size=32, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Text("Version 1.17", size=20, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(ft.Text("The most comprehensive single-file portfolio manager ever built", size=13, color="#888888", italic=True))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text(
            "A premium, single-file Python portfolio management application built with Flet. "
            "Designed for investors, traders, and financial enthusiasts who want a comprehensive "
            "view of their portfolio with advanced analytics, AI-powered insights, and beautiful "
            "dark-themed visualizations. Features 375+ engine methods, 47 interactive tabs across "
            "9 navigation categories, 16 chart types, and full CAPM/risk-adjusted analytics.",
            size=14, color="#cccccc",
        ))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        # STATS BANNER
        def _stat(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=20, weight=ft.FontWeight.BOLD, color=color)], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        col.controls.append(ft.Row([
            _stat("Engine Methods", "375+", "#00ff9d"),
            _stat("Tabs", "47", "#ffd700"),
            _stat("Categories", "9", "#00bfff"),
            _stat("Chart Types", "16", "#ff9900"),
            _stat("Lines of Code", "6000+", "#9966ff"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        # TECHNOLOGY STACK
        col.controls.append(ft.Text("TECHNOLOGY STACK", size=18, weight=ft.FontWeight.BOLD, color="#ffd700"))
        tech = [
            ("Flet", "Modern cross-platform UI framework (desktop + web)"),
            ("DuckDB", "Embedded analytical SQL database for fast queries"),
            ("Pandas & NumPy", "Data manipulation, numerical computing, time-series"),
            ("yfinance", "Real-time and historical market data, dividends, benchmarks"),
            ("QuantStats", "Sharpe, Sortino, Calmar, VaR, CVaR, rolling analytics"),
            ("PyPortfolioOpt", "Mean-variance optimization, efficient frontier"),
            ("Matplotlib", "16 chart types rendered as embedded base64 images"),
            ("Python 3.13", "Modern async-ready runtime"),
        ]
        for name, desc in tech:
            col.controls.append(ft.Row([
                ft.Text(f"{name}:", size=13, weight=ft.FontWeight.BOLD, color="#00ff9d", width=150),
                ft.Text(desc, size=12, color="#aaaaaa", expand=True),
            ]))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        # FULL FEATURE INVENTORY
        col.controls.append(ft.Text("COMPLETE FEATURE INVENTORY", size=18, weight=ft.FontWeight.BOLD, color="#ffd700"))
        features = {
            "Dashboard (3 tabs)": "15 KPIs (value, G/L, return, vol, Sharpe, Sortino, profit factor, VaR, win rate, streaks, recovery), equity curve, summary stats, portfolio health score gauge",
            "Portfolio (6 tabs)": "Holdings table (qty/price/value/weight/cost/G-L/%), bar chart, weight pie, top gainers & losers, trade log with dates",
            "Analysis (12 tabs)": "Correlation matrix, market graphs, portfolio DNA fingerprint, constellation chart, efficient frontier, sector allocation pie, performance attribution, returns distribution histogram, monthly heatmap, benchmark comparison (SPY/QQQ/DIA/IWM), correlation deep-dive (pairwise), risk-adjusted returns (CAPM alpha/beta/Treynor/info ratio)",
            "Planning (7 tabs)": "Goal probability (Monte Carlo), advanced planner (inflation/tax/income/expenses), FIRE calculator (lean/regular/fat/coast), monthly comparison, dream life architect, income tracker (dividends), what-if scenario (4 interactive sliders)",
            "Risk (10 tabs)": "Risk radar chart, stress test lab, emotional risk gauge, diversification score, market regime detection, drawdown chart, rolling Sharpe (60d), rolling volatility (30d), gain/loss waterfall, win/loss streaks & recovery time",
            "AI & Tools (7 tabs)": "Smart brain analysis, AI co-pilot chat, voice command simulation, Monte Carlo paths, trade journal, portfolio rebalancer (max Sharpe), tax-loss harvesting optimizer",
            "Settings (1 tab)": "Theme toggle, risk profile configuration",
        }
        for cat, desc in features.items():
            col.controls.append(ft.Text(cat, size=13, weight=ft.FontWeight.BOLD, color="#00ff9d"))
            col.controls.append(ft.Text(desc, size=12, color="#aaaaaa"))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        # COMPLETE ENGINE METHOD INVENTORY (auto-generated from UltraDataEngine)
        col.controls.append(ft.Text("COMPLETE ENGINE METHOD INVENTORY (376 methods)", size=18, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(ft.Text("Every method in the UltraDataEngine, organized by category:", size=12, color="#aaaaaa"))
        engine_categories = {
            "Core & Import": [
                "__init__", "_fig_to_base64", "_generate_returns_series", "filter_by_timerange",
                "_import_single_file", "import_folder", "import_multi", "get_import_summary",
                "_compute_holdings_and_prices",
            ],
            "Portfolio Analytics & KPIs": [
                "portfolio_summary_stats", "win_loss_streaks", "recovery_time", "profit_factor",
                "risk_adjusted_returns", "correlation_matrix", "top_movers",
                "what_if_scenario", "performance_attribution", "monthly_returns",
                "portfolio_health_score", "goal_probability", "fire_calculator",
                "tax_loss_harvest", "sector_allocation", "diversification_score",
                "market_regime", "dividend_income", "estimate_dividends",
            ],
            "Chart Generators (16 types)": [
                "chart_equity_curve", "chart_correlation_matrix", "chart_candlestick",
                "chart_holdings_bar", "chart_returns_histogram", "chart_sector_pie",
                "chart_gain_loss_waterfall", "chart_rolling_sharpe", "chart_rolling_volatility",
                "chart_monthly_heatmap", "chart_weight_pie", "chart_growth_projection",
                "chart_efficient_frontier", "chart_risk_radar", "chart_portfolio_dna",
                "chart_health_gauge", "chart_constellation", "chart_monte_carlo",
                "chart_benchmark_overlay", "chart_drawdown", "chart_what_if",
                "chart_stress_test", "chart_allocation_pie",
            ],
            "Profit & Inspection": [
                "daily_profit_loss", "weekly_profit_loss", "chart_daily_profit",
                "chart_weekly_profit", "chart_rolling_weekly_profit",
                "inspect_asset", "chart_asset_inspection",
            ],
            "Live Chart & Technical": [
                "fetch_live_data", "chart_live_ticker", "chart_advanced_technical",
                "chart_multi_asset", "chart_asset_correlation_scatter",
            ],
            "Advanced Risk Metrics (19)": [
                "omega_ratio", "tail_ratio", "gain_to_pain_ratio", "ulcer_index",
                "pain_index", "sterling_ratio", "burke_ratio", "kappa_three",
                "downside_deviation", "upside_potential_ratio", "parametric_var",
                "historical_var", "conditional_var", "expected_shortfall",
                "max_drawdown_duration", "recovery_factor", "calmar_ratio_custom",
                "lake_ratio", "rachev_ratio",
            ],
            "Statistical Analysis (15)": [
                "jarque_bera_test", "hurst_exponent", "autocorrelation",
                "autocorrelation_profile", "rolling_correlation_pair",
                "half_life_mean_reversion", "zscore_analysis", "cointegration_test",
                "skewness_kurtosis_detail", "regime_detection_simple",
                "volatility_cone", "ewma_volatility", "parkinson_volatility",
                "garman_klass_volatility", "yang_zhang_volatility",
            ],
            "Technical Indicators (19)": [
                "fibonacci_retracement", "compute_atr", "compute_cci",
                "compute_williams_r", "compute_stochastic", "compute_obv",
                "compute_mfi", "compute_adx", "compute_roc", "compute_trix",
                "compute_aroon", "compute_ultimate_oscillator", "compute_donchian",
                "compute_keltner", "compute_ichimoku", "compute_force_index",
                "compute_elder_ray", "compute_chaikin_mf", "support_resistance_levels",
            ],
            "Portfolio Construction & Attribution (14)": [
                "risk_parity_weights", "minimum_variance_weights",
                "maximum_diversification_weights", "hierarchical_risk_parity",
                "marginal_contribution_to_risk", "component_var_calc",
                "portfolio_concentration_hhi", "effective_num_bets",
                "tracking_error_calc", "information_ratio_calc",
                "m_squared_measure", "brinson_attribution",
                "treynor_ratio", "jensens_alpha_calc",
            ],
            "Financial Calculators (14)": [
                "compound_interest", "rule_of_72", "present_value", "future_value",
                "net_present_value", "internal_rate_of_return", "loan_amortization",
                "mortgage_calculator", "savings_goal_calculator", "dca_simulator",
                "dividend_reinvestment_calc", "withdrawal_rate_sim",
                "inflation_adjusted_return", "cost_basis_fifo",
            ],
            "Options & Fixed Income (9)": [
                "black_scholes_call", "black_scholes_put", "option_greeks",
                "implied_volatility_calc", "bond_price", "bond_duration",
                "bond_modified_duration", "bond_convexity", "yield_to_maturity",
            ],
            "ML / Signals / Forecasting (9)": [
                "linear_regression_forecast", "moving_average_crossover_signals",
                "momentum_signals", "mean_reversion_signals",
                "trend_strength_indicator", "volatility_forecast",
                "correlation_regime_forecast", "sector_rotation_signal", "market_breadth",
            ],
            "Advanced Derivatives & Exotic Options (14)": [
                "binomial_option_price", "american_option_price", "option_parity_check",
                "garman_kohlhagen_fx", "option_profit_loss", "straddle_payoff",
                "strangle_payoff", "iron_condor_payoff", "covered_call_return",
                "protective_put_cost", "butterfly_spread_payoff",
                "option_probability_itm", "option_expected_move", "option_decay_profile",
            ],
            "Advanced Statistics & Econometrics (19)": [
                "augmented_dickey_fuller", "granger_causality_simple",
                "variance_ratio_test", "ljung_box_test", "runs_test",
                "copula_tail_dependence", "rolling_beta", "rolling_alpha",
                "conditional_skewness", "drawdown_at_risk",
                "pain_ratio", "gain_loss_ratio", "profit_loss_ratio",
                "common_sense_ratio", "pessimistic_return",
                "probabilistic_sharpe", "deflated_sharpe", "minimum_track_record",
            ],
            "Macro & Fundamental Analysis (15)": [
                "pe_ratio_calc", "pb_ratio_calc", "ev_ebitda_calc",
                "dividend_yield_calc", "peg_ratio_calc", "dcf_valuation",
                "wacc_calc", "altman_z_score", "piotroski_f_score",
                "dupont_analysis", "gordon_growth_model", "earnings_yield",
                "free_cash_flow_yield", "enterprise_value_calc", "magic_formula_rank",
            ],
            "Advanced Portfolio Optimization (14)": [
                "equal_risk_contribution", "black_litterman_weights",
                "kelly_criterion", "kelly_criterion_portfolio",
                "target_return_weights", "max_sharpe_weights", "min_cvar_weights",
                "inverse_volatility_weights", "momentum_weights",
                "volatility_targeting_weights", "tail_risk_parity",
                "diversification_ratio", "portfolio_turnover_calc", "rebalance_trades",
            ],
            "Risk Decomposition & Stress Testing (18)": [
                "historical_stress_test", "parametric_stress_test",
                "correlation_stress_test", "factor_risk_decomp",
                "tail_risk_contribution", "risk_budget_analysis",
                "expected_tail_loss", "drawdown_distribution",
                "time_under_water", "conditional_drawdown_at_risk",
                "portfolio_beta_decomp", "systematic_vs_specific_risk",
                "scenario_analysis", "liquidity_risk_score",
                "concentration_risk_report", "var_backtest",
                "extreme_value_var", "portfolio_resilience_score",
            ],
            "Time Series & Forecasting Advanced (14)": [
                "exponential_smoothing_forecast", "double_exponential_smoothing",
                "triple_exponential_smoothing", "ar_model_forecast",
                "seasonal_decomposition", "walk_forward_backtest",
                "regime_switching_forecast", "return_forecast_ensemble",
                "volatility_regime_forecast", "mean_absolute_deviation",
                "coefficient_of_variation", "interquartile_range",
                "median_absolute_deviation", "gini_coefficient",
            ],
            "Trade Analytics & Execution (15)": [
                "trade_expectancy", "trade_payoff_ratio",
                "consecutive_wins_losses", "holding_period_analysis",
                "trade_frequency_analysis", "day_of_week_returns",
                "monthly_seasonality", "best_worst_periods",
                "rolling_win_rate", "profit_factor_rolling",
                "risk_adjusted_return_by_period", "trade_duration_stats",
                "slippage_estimate", "transaction_cost_analysis",
                "realized_vs_unrealized_pnl",
            ],
            "Crypto & Alternative Assets (14)": [
                "nvt_ratio", "mvrv_ratio", "stock_to_flow",
                "crypto_fear_greed_proxy", "crypto_correlation_to_btc",
                "sharpe_by_asset_class", "max_drawdown_by_asset",
                "rolling_correlation_matrix", "asset_momentum_ranking",
                "relative_strength_ranking", "realized_volatility_by_asset",
                "correlation_change_detection", "portfolio_heat_score",
                "drawdown_heatmap_data",
            ],
            "Tax & Accounting Analytics (14)": [
                "capital_gains_estimate", "tax_loss_harvest_candidates",
                "wash_sale_detector", "portfolio_income_analysis",
                "cost_basis_summary", "roi_annualized",
                "time_weighted_return", "money_weighted_return",
                "effective_tax_rate", "after_tax_return",
                "tax_equivalent_yield", "portfolio_yield_on_cost",
                "unrealized_gain_loss_report", "account_allocation_summary",
            ],
            "Behavioral Finance & Sentiment (14)": [
                "disposition_effect_score", "recency_bias_indicator",
                "anchoring_bias_check", "loss_aversion_ratio",
                "overconfidence_indicator", "herding_indicator",
                "fomo_score", "regret_minimization_score",
                "emotional_temperature", "sunk_cost_detector",
                "portfolio_anxiety_index", "market_sentiment_composite",
                "confirmation_bias_check", "portfolio_stress_indicator",
            ],
            "Fixed Income Advanced & Credit (14)": [
                "zero_coupon_price", "forward_rate", "spot_rate_from_par",
                "credit_spread", "z_spread_approx", "current_yield",
                "accrued_interest", "dirty_price", "dv01_calc",
                "key_rate_duration", "bond_total_return",
                "callable_bond_oas", "floating_rate_note_value",
                "inflation_linked_return",
            ],
            "Monte Carlo & Simulation (14)": [
                "monte_carlo_portfolio", "monte_carlo_var", "monte_carlo_cvar",
                "geometric_brownian_motion", "bootstrap_returns",
                "block_bootstrap_returns", "jump_diffusion_sim",
                "retirement_monte_carlo", "sequence_of_returns_risk",
                "optimal_allocation_sim", "efficient_frontier_sim",
                "correlation_simulation", "tail_risk_simulation",
                "drawdown_simulation",
            ],
            "Book Balancer & Budget Planner": [
                "add_book_entry", "remove_book_entry", "get_book_summary",
                "add_budget_item", "remove_budget_item", "get_budget_summary",
                "chart_budget_breakdown", "close",
            ],
        }
        total_listed = 0
        for cat_name, methods in engine_categories.items():
            total_listed += len(methods)
            method_list = ", ".join(methods)
            col.controls.append(ft.Container(
                content=ft.Column([
                    ft.Text(f"{cat_name}  [{len(methods)}]", size=13, weight=ft.FontWeight.BOLD, color="#00ff9d"),
                    ft.Text(method_list, size=11, color="#aaaaaa"),
                ], spacing=2),
                bgcolor="#0f0f0f",
                padding=ft.Padding(left=10, top=6, right=10, bottom=6),
                border_radius=8,
            ))
        col.controls.append(ft.Text(f"Total: {total_listed} engine methods", size=14, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        # DATA IMPORT
        col.controls.append(ft.Text("DATA IMPORT", size=18, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(ft.Text(
            "Supports CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet, Feather, and HTML table files. "
            "Paste a folder path in the import bar on the Dashboard page and click IMPORT. "
            "The engine auto-detects date, symbol, quantity, and price columns. "
            "All values default to N/A or 0 when no data is loaded — the app never crashes on empty data.",
            size=12, color="#cccccc",
        ))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        # CREDITS
        col.controls.append(ft.Text("CREDITS", size=18, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(ft.Text("Developed with Cascade AI assistance", size=13, color="#aaaaaa"))
        col.controls.append(ft.Text("Open-source libraries: Flet, DuckDB, Pandas, NumPy, yfinance, QuantStats, PyPortfolioOpt, Matplotlib", size=12, color="#666666"))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text(
            "Built with care. 375+ engine methods. 47 tabs across 9 categories. 16 chart types. "
            "All data defaults to N/A when not loaded. Zero crashes on empty portfolios.",
            size=13, color="#888888", italic=True,
        ))
        return ft.Container(content=col, padding=20, bgcolor="#121212", border_radius=16)

    # ====================== TAB BUILDERS ======================
    def dashboard_tab():
        s = data.portfolio_summary_stats()
        pf = data.profit_factor()
        wl = data.win_loss_streaks()
        rec = data.recovery_time()
        ra = data.risk_adjusted_returns()
        hs = data.portfolio_health_score()
        hp = data.holding_period_analysis()
        tf = data.trade_frequency_analysis()

        vol = s.get("ann_volatility", 0)
        sharpe = s.get("sharpe", 0)
        sortino = s.get("sortino", 0)
        calmar = s.get("calmar", 0)
        max_dd = s.get("max_drawdown", 0)
        total_gl = s.get("total_gain_loss", 0)
        gl_color = "#00ff9d" if total_gl >= 0 else "#ff3366"
        n_hold = s.get("num_holdings", 0)
        ann_ret = s.get("ann_return", 0)
        best = s.get("best_day", 0)
        worst = s.get("worst_day", 0)
        win_rate = s.get("win_rate", 0)
        var95 = s.get("var_95", 0)
        cvar95 = s.get("cvar_95", 0)
        gl_pct = s.get("total_gain_pct", 0)
        skew = s.get("skewness", 0)
        kurt = s.get("kurtosis", 0)
        pos_days = s.get("positive_days", 0)
        neg_days = s.get("negative_days", 0)
        beta = ra.get("beta", 0)
        alpha = ra.get("alpha", 0)
        treynor = ra.get("treynor", 0)
        info_ratio = ra.get("info_ratio", 0)
        total_cb = s.get("total_cost_basis", 0)
        journey_days = hp.get('days', 0)
        journey_years = hp.get('years', 0)
        total_trades = tf.get('total', 0)
        grade = hs.get('grade', 'N/A')
        health_score = hs.get('score', 0)

        def _dk(label, val, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(val), size=18, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=12, expand=True)
        def _dk_sm(label, val, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(str(val), size=15, weight=ft.FontWeight.BOLD, color=color)], spacing=1), bgcolor="#1a1a2e", padding=8, border_radius=10, expand=True)
        def _fv(v, fmt=".2f"):
            if v == 0: return "N/A"
            return f"{v:{fmt}}"

        # Row 1: Portfolio value overview
        kpi_row1 = ft.Row([
            _dk("Total Value", f"${data.total_value:,.0f}", "#ffd700"),
            _dk("Cost Basis", f"${total_cb:,.0f}" if total_cb else "N/A", "#aaaaaa"),
            _dk("Total G/L", f"${total_gl:,.0f}" if total_gl != 0 else "N/A", gl_color),
            _dk("G/L %", f"{gl_pct:+.1f}%" if gl_pct != 0 else "N/A", gl_color),
            _dk("Ann. Return", f"{ann_ret:.1%}" if ann_ret != 0 else "N/A", "#00ff9d" if ann_ret >= 0 else "#ff3366"),
        ], spacing=6)
        # Row 2: Risk metrics
        kpi_row2 = ft.Row([
            _dk("Volatility", f"{vol:.1%}" if vol != 0 else "N/A", "#ff3366"),
            _dk("Sharpe", _fv(sharpe), "#00ff9d" if sharpe > 0 else "#ff3366"),
            _dk("Sortino", _fv(sortino), "#00ff9d" if sortino > 0 else "#ff3366"),
            _dk("Calmar", _fv(calmar), "#00ff9d" if calmar > 0 else "#ff3366"),
            _dk("Max DD", f"{max_dd:.1%}" if max_dd != 0 else "N/A", "#ff3366"),
        ], spacing=6)
        # Row 3: Trading stats
        kpi_row3 = ft.Row([
            _dk("Profit Factor", f"{pf:.2f}" if pf != 0 else "N/A", "#00ff9d" if pf > 1 else "#ff3366"),
            _dk("Win Rate", f"{win_rate:.0%}" if win_rate != 0 else "N/A", "#00ff9d" if win_rate > 0.5 else "#ff3366"),
            _dk("Best Day", f"{best:.2%}" if best != 0 else "N/A", "#00ff9d"),
            _dk("Worst Day", f"{worst:.2%}" if worst != 0 else "N/A", "#ff3366"),
            _dk("Win Streak", f"{wl['longest_win']}d" if wl['longest_win'] > 0 else "N/A", "#00ff9d"),
        ], spacing=6)
        # Row 4: Risk deep dive
        kpi_row4 = ft.Row([
            _dk("VaR 95%", f"{var95:.2%}" if var95 != 0 else "N/A", "#ff3366"),
            _dk("CVaR 95%", f"{cvar95:.2%}" if cvar95 != 0 else "N/A", "#ff3366"),
            _dk("Beta (SPY)", _fv(beta), "#ffd700"),
            _dk("Alpha (CAPM)", f"{alpha:.2%}" if alpha != 0 else "N/A", "#00ff9d" if alpha > 0 else "#ff3366"),
            _dk("Treynor", _fv(treynor), "#00ff9d" if treynor > 0 else "#ff3366"),
        ], spacing=6)
        # Row 5: Distribution & Advanced
        kpi_row5 = ft.Row([
            _dk_sm("Skewness", _fv(skew, ".3f"), "#00ff9d" if skew > 0 else "#ff3366"),
            _dk_sm("Kurtosis", _fv(kurt, ".3f"), "#ffd700"),
            _dk_sm("Positive Days", str(pos_days) if pos_days else "N/A", "#00ff9d"),
            _dk_sm("Negative Days", str(neg_days) if neg_days else "N/A", "#ff3366"),
            _dk_sm("Info Ratio", _fv(info_ratio), "#00ff9d" if info_ratio > 0 else "#ff3366"),
            _dk_sm("Loss Streak", f"{wl['longest_loss']}d" if wl['longest_loss'] > 0 else "N/A", "#ff3366"),
        ], spacing=4)
        # Row 6: Portfolio overview
        kpi_row6 = ft.Row([
            _dk_sm("Holdings", str(n_hold) if n_hold > 0 else "0", "#ffd700"),
            _dk_sm("Total Trades", f"{total_trades:,}" if total_trades else "N/A", "#9966ff"),
            _dk_sm("Journey", f"{journey_years}yr ({journey_days:,}d)" if journey_days else "N/A", "#00bfff"),
            _dk_sm("Health Grade", grade, "#00ff9d" if health_score >= 65 else "#ffd700" if health_score >= 50 else "#ff3366"),
            _dk_sm("Health Score", f"{health_score}/100", "#00ff9d" if health_score >= 65 else "#ffd700" if health_score >= 50 else "#ff3366"),
            _dk_sm("Current Streak", f"{wl['current_streak']}d ({wl['current_type']})" if wl['current_type'] != 'N/A' else "N/A", "#00ff9d" if wl.get('current_type') == 'win' else "#ff3366"),
        ], spacing=4)

        # Recovery & drawdown status
        rec_parts = []
        if rec['max_recovery_days'] > 0:
            rec_parts.append(f"Max recovery: {rec['max_recovery_days']}d")
        if rec.get('avg_recovery_days', 0) > 0:
            rec_parts.append(f"Avg recovery: {rec['avg_recovery_days']:.0f}d")
        if rec['currently_in_drawdown']:
            rec_parts.append(f"IN DRAWDOWN: {rec['current_dd_days']}d")
        rec_text = "  |  ".join(rec_parts) if rec_parts else "No drawdown recovery data"

        # --- Equity curve with toggles ---
        eq_state = {'sma20': False, 'sma50': True, 'sma200': False, 'bb': False, 'log': False, 'dd': False, 'pct': False}
        eq_chart = ft.Container(content=_chart_image(data.chart_equity_curve(**{
            'show_sma20': eq_state['sma20'], 'show_sma50': eq_state['sma50'], 'show_sma200': eq_state['sma200'],
            'show_bollinger': eq_state['bb'], 'log_scale': eq_state['log'],
            'show_drawdown_shade': eq_state['dd'], 'show_pct': eq_state['pct']})))
        def _eq_refresh():
            eq_chart.content = _chart_image(data.chart_equity_curve(
                show_sma20=eq_state['sma20'], show_sma50=eq_state['sma50'], show_sma200=eq_state['sma200'],
                show_bollinger=eq_state['bb'], log_scale=eq_state['log'],
                show_drawdown_shade=eq_state['dd'], show_pct=eq_state['pct']))
            page.update()
        def _eq_tog(key):
            def handler(e):
                eq_state[key] = e.control.value
                _eq_refresh()
            return handler
        def _sw(label, key, val=False):
            return ft.Switch(label=label, value=val, active_color="#00ff9d", label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_eq_tog(key))
        eq_toggles = ft.Row([
            _sw("20d SMA", "sma20"), _sw("50d SMA", "sma50", True), _sw("200d SMA", "sma200"),
            _sw("Bollinger", "bb"), _sw("Log Scale", "log"), _sw("DD Shade", "dd"), _sw("% View", "pct"),
        ], spacing=8, scroll=ft.ScrollMode.AUTO)
        col = ft.Column([
            ft.Text("DASHBOARD", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text(f"Last updated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}", size=11, color="#666666"),
            ft.Text("PORTFOLIO VALUE", size=13, weight=ft.FontWeight.BOLD, color="#ffd700"),
            kpi_row1,
            ft.Text("RISK METRICS", size=13, weight=ft.FontWeight.BOLD, color="#ff3366"),
            kpi_row2,
            ft.Text("TRADING PERFORMANCE", size=13, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            kpi_row3,
            ft.Text("RISK DEEP DIVE", size=13, weight=ft.FontWeight.BOLD, color="#ff9900"),
            kpi_row4,
            ft.Text("DISTRIBUTION & ADVANCED", size=13, weight=ft.FontWeight.BOLD, color="#9966ff"),
            kpi_row5,
            ft.Text("PORTFOLIO OVERVIEW", size=13, weight=ft.FontWeight.BOLD, color="#00bfff"),
            kpi_row6,
            ft.Divider(height=1, color="#333333"),
            ft.Text(rec_text, size=11, color="#888888", italic=True),
            ft.Divider(height=1, color="#333333"),
            ft.Text("EQUITY CURVE", size=15, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            eq_toggles,
            eq_chart,
        ], scroll=ft.ScrollMode.AUTO, spacing=8)

        return _card(col)

    # data_organizer_tab removed

    def portfolio_tab():
        holdings_data = data.holdings_with_weights()
        def _pk(label, val, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(str(val), size=15, weight=ft.FontWeight.BOLD, color=color)], spacing=1), bgcolor="#1a1a2e", padding=8, border_radius=10, expand=True)
        if not holdings_data:
            content = ft.Text("No holdings loaded. Import data first.", size=20, color="#aaaaaa")
        else:
            rows = []
            for h in holdings_data:
                gl_color = "#00ff9d" if h['gain_loss'] >= 0 else "#ff3366"
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(h["symbol"], color="#ffffff", weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{h['quantity']:.2f}", color="#ffffff")),
                    ft.DataCell(ft.Text(f"${h['price']:,.2f}", color="#ffd700")),
                    ft.DataCell(ft.Text(f"${h['market_value']:,.0f}", color="#00ff9d")),
                    ft.DataCell(ft.Text(f"{h['weight']:.1f}%", color="#ffffff")),
                    ft.DataCell(ft.Text(f"${h['cost_basis']:,.0f}", color="#aaaaaa")),
                    ft.DataCell(ft.Text(f"${h['gain_loss']:,.0f}", color=gl_color)),
                    ft.DataCell(ft.Text(f"{h['gain_loss_pct']:+.1f}%", color=gl_color)),
                ]))
            content = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Symbol", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Qty", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Price", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Value", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Weight", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Cost Basis", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Gain/Loss", color="#00ff9d")),
                    ft.DataColumn(ft.Text("G/L %", color="#00ff9d")),
                ],
                rows=rows,
                heading_row_color="#1a1a2e",
            )
        total_gl = sum(h['gain_loss'] for h in holdings_data) if holdings_data else 0
        total_val = data.total_value
        total_cb = sum(h['cost_basis'] for h in holdings_data) if holdings_data else 0
        gl_color = "#00ff9d" if total_gl >= 0 else "#ff3366"
        n = len(holdings_data)
        # Concentration metrics
        weights = [h['weight'] / 100.0 for h in holdings_data] if holdings_data else []
        hhi = sum(w ** 2 for w in weights) * 10000 if weights else 0
        eff_n = 1.0 / sum(w ** 2 for w in weights) if weights and sum(w ** 2 for w in weights) > 0 else 0
        top5_wt = sum(sorted(weights, reverse=True)[:5]) * 100 if weights else 0
        max_wt = max(weights) * 100 if weights else 0
        winners = sum(1 for h in holdings_data if h['gain_loss'] >= 0) if holdings_data else 0
        losers = n - winners
        avg_gl_pct = np.mean([h['gain_loss_pct'] for h in holdings_data]) if holdings_data else 0
        median_gl_pct = np.median([h['gain_loss_pct'] for h in holdings_data]) if holdings_data else 0
        best_h = max(holdings_data, key=lambda h: h['gain_loss_pct']) if holdings_data else None
        worst_h = min(holdings_data, key=lambda h: h['gain_loss_pct']) if holdings_data else None
        biggest_h = max(holdings_data, key=lambda h: h['market_value']) if holdings_data else None

        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10)
        col.controls.append(ft.Text("PORTFOLIO HOLDINGS", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        # Summary KPIs
        col.controls.append(ft.Row([
            _pk("Total Value", f"${total_val:,.0f}", "#ffd700"),
            _pk("Cost Basis", f"${total_cb:,.0f}" if total_cb else "N/A", "#aaaaaa"),
            _pk("Total G/L", f"${total_gl:,.0f}" if total_gl != 0 else "N/A", gl_color),
            _pk("G/L %", f"{(total_gl/total_cb*100):+.1f}%" if total_cb else "N/A", gl_color),
            _pk("Holdings", str(n), "#ffffff"),
        ], spacing=4))
        col.controls.append(ft.Row([
            _pk("Winners", str(winners), "#00ff9d"),
            _pk("Losers", str(losers), "#ff3366"),
            _pk("Avg G/L %", f"{avg_gl_pct:+.1f}%", "#00ff9d" if avg_gl_pct >= 0 else "#ff3366"),
            _pk("Median G/L %", f"{median_gl_pct:+.1f}%", "#00ff9d" if median_gl_pct >= 0 else "#ff3366"),
            _pk("Win Ratio", f"{winners/n:.0%}" if n > 0 else "N/A", "#00ff9d" if winners > losers else "#ff3366"),
        ], spacing=4))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("CONCENTRATION ANALYSIS", size=15, weight=ft.FontWeight.BOLD, color="#ff9900"))
        col.controls.append(ft.Row([
            _pk("HHI Index", f"{hhi:.0f}", "#ff9900" if hhi > 2500 else "#ffd700" if hhi > 1500 else "#00ff9d"),
            _pk("Effective # Bets", f"{eff_n:.1f}", "#00ff9d" if eff_n > 5 else "#ffd700"),
            _pk("Top 5 Weight", f"{top5_wt:.1f}%", "#ff9900" if top5_wt > 80 else "#ffd700"),
            _pk("Max Position", f"{max_wt:.1f}%", "#ff3366" if max_wt > 30 else "#ffd700"),
            _pk("Largest", biggest_h['symbol'] if biggest_h else "N/A", "#ffd700"),
        ], spacing=4))
        col.controls.append(ft.Text("HHI < 1500 = diversified  |  1500-2500 = moderate  |  > 2500 = concentrated", size=10, color="#666666", italic=True))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        # Best/Worst performers
        if best_h and worst_h:
            col.controls.append(ft.Text("TOP & BOTTOM PERFORMERS", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"))
            col.controls.append(ft.Row([
                _pk(f"Best: {best_h['symbol']}", f"{best_h['gain_loss_pct']:+.1f}% (${best_h['gain_loss']:,.0f})", "#00ff9d"),
                _pk(f"Worst: {worst_h['symbol']}", f"{worst_h['gain_loss_pct']:+.1f}% (${worst_h['gain_loss']:,.0f})", "#ff3366"),
            ], spacing=4))
            col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(content)
        return _card(col)

    def analysis_tab():
        corr_container = ft.Container(
            content=ft.Text("Click 'Compute' to generate correlation matrix", size=14, color="#aaaaaa"),
            padding=10,
        )
        def _compute(e):
            corr_container.content = ft.Column([
                ft.ProgressRing(width=30, height=30, stroke_width=3, color="#00ff9d"),
                ft.Text("Computing correlation matrix...", size=12, color="#888888"),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            page.update()
            try:
                corr_container.content = _chart_image(data.chart_correlation_matrix())
            except Exception as ex:
                corr_container.content = ft.Text(f"Error: {ex}", size=14, color="#ff3366")
            page.update()
        return _card(ft.Column([
            ft.Text("ADVANCED ANALYSIS", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Button("COMPUTE", bgcolor="#00ff9d", color="#0a0a0a",
                       style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                       on_click=_compute),
            corr_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def market_graphs_tab():
        mg_opts = {'period': 120, 'sma20': False, 'sma50': False, 'volume': False, 'type': 'candle'}
        mg_chart_c = ft.Container(content=_chart_image(data.chart_candlestick()))
        def _mg_refresh():
            mg_chart_c.content = _chart_image(data.chart_candlestick(
                period=mg_opts['period'], show_sma20=mg_opts['sma20'], show_sma50=mg_opts['sma50'],
                show_volume=mg_opts['volume'], chart_type=mg_opts['type']))
            page.update()
        def _mg_tog(key):
            def handler(e):
                mg_opts[key] = e.control.value
                _mg_refresh()
            return handler
        def _mg_period(e):
            try: mg_opts['period'] = int(e.control.value)
            except: mg_opts['period'] = 120
            _mg_refresh()
        def _mg_type(e):
            mg_opts['type'] = e.control.value
            _mg_refresh()
        mg_toggles = ft.Row([
            ft.Text("Period:", size=11, color="#aaaaaa"),
            ft.Dropdown(value="120", width=90, options=[
                ft.dropdown.Option("30", "30d"), ft.dropdown.Option("60", "60d"),
                ft.dropdown.Option("120", "120d"), ft.dropdown.Option("252", "1yr"),
            ], on_select=_mg_period, border_color="#00ff9d", color="#ffffff", bgcolor="#1a1a2e"),
            ft.Text("Style:", size=11, color="#aaaaaa"),
            ft.Dropdown(value="candle", width=100, options=[
                ft.dropdown.Option("candle", "Candle"), ft.dropdown.Option("line", "Line"),
                ft.dropdown.Option("area", "Area"),
            ], on_select=_mg_type, border_color="#00ff9d", color="#ffffff", bgcolor="#1a1a2e"),
            ft.Switch(label="20d SMA", value=False, active_color="#00bfff",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_mg_tog('sma20')),
            ft.Switch(label="50d SMA", value=False, active_color="#ffd700",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_mg_tog('sma50')),
            ft.Switch(label="Volume", value=False, active_color="#ff9900",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_mg_tog('volume')),
        ], spacing=8, scroll=ft.ScrollMode.AUTO)
        return _card(ft.Column([
            ft.Text("LIVE MARKET GRAPHS", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            mg_toggles,
            mg_chart_c,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    # goals_tab, smart_brain_tab, ai_chat_tab removed

    def risk_radar_tab():
        s = data.portfolio_summary_stats()
        ra = data.risk_adjusted_returns()
        rs = data.returns_series
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        radar_container = ft.Container(
            content=ft.Text("Click 'Compute' to generate risk radar", size=14, color="#aaaaaa"),
            padding=10,
        )
        def _compute(e):
            radar_container.content = ft.Column([
                ft.ProgressRing(width=30, height=30, stroke_width=3, color="#00ff9d"),
                ft.Text("Computing risk radar...", size=12, color="#888888"),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            page.update()
            try:
                radar_container.content = _chart_image(data.chart_risk_radar())
            except Exception as ex:
                radar_container.content = ft.Text(f"Error: {ex}", size=14, color="#ff3366")
            page.update()

        # Compute advanced risk metrics inline
        vol = s.get("ann_volatility", 0)
        max_dd = s.get("max_drawdown", 0)
        var95 = s.get("var_95", 0)
        cvar95 = s.get("cvar_95", 0)
        sharpe = s.get("sharpe", 0)
        sortino = s.get("sortino", 0)
        calmar = s.get("calmar", 0)
        beta = ra.get("beta", 0)
        alpha = ra.get("alpha", 0)
        # Tail risk
        tail_ratio_val = 0
        if not rs.empty and len(rs) > 10:
            p95 = float(np.percentile(rs, 95))
            p5 = float(np.percentile(rs, 5))
            tail_ratio_val = abs(p95 / p5) if p5 != 0 else 0
        # Downside deviation
        downside_dev = float(rs[rs < 0].std() * np.sqrt(252)) if not rs.empty and (rs < 0).any() else 0

        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10)
        col.controls.append(ft.Text("HOLOGRAPHIC RISK RADAR", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Text("Multi-dimensional risk assessment across all portfolio dimensions", size=12, color="#aaaaaa"))
        col.controls.append(ft.Text("RISK OVERVIEW", size=15, weight=ft.FontWeight.BOLD, color="#ff3366"))
        col.controls.append(ft.Row([
            _kpi("Volatility", f"{vol:.1%}" if vol else "N/A", "#ff3366"),
            _kpi("Max Drawdown", f"{max_dd:.1%}" if max_dd else "N/A", "#ff3366"),
            _kpi("VaR 95%", f"{var95:.2%}" if var95 else "N/A", "#ff3366"),
            _kpi("CVaR 95%", f"{cvar95:.2%}" if cvar95 else "N/A", "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Text("RISK-ADJUSTED PERFORMANCE", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(ft.Row([
            _kpi("Sharpe", f"{sharpe:.2f}" if sharpe else "N/A", "#00ff9d" if sharpe > 0 else "#ff3366"),
            _kpi("Sortino", f"{sortino:.2f}" if sortino else "N/A", "#00ff9d" if sortino > 0 else "#ff3366"),
            _kpi("Calmar", f"{calmar:.2f}" if calmar else "N/A", "#00ff9d" if calmar > 0 else "#ff3366"),
            _kpi("Tail Ratio", f"{tail_ratio_val:.2f}" if tail_ratio_val else "N/A", "#00ff9d" if tail_ratio_val > 1 else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Text("MARKET SENSITIVITY", size=15, weight=ft.FontWeight.BOLD, color="#00bfff"))
        col.controls.append(ft.Row([
            _kpi("Beta (SPY)", f"{beta:.2f}" if beta else "N/A", "#ffd700"),
            _kpi("Alpha (CAPM)", f"{alpha:.2%}" if alpha else "N/A", "#00ff9d" if alpha > 0 else "#ff3366"),
            _kpi("Downside Dev", f"{downside_dev:.1%}" if downside_dev else "N/A", "#ff9900"),
            _kpi("Treynor", f"{ra.get('treynor', 0):.2f}" if ra.get('treynor', 0) else "N/A", "#00ff9d" if ra.get('treynor', 0) > 0 else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Button("COMPUTE RADAR CHART", bgcolor="#00ff9d", color="#0a0a0a",
                       style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                       on_click=_compute))
        col.controls.append(radar_container)
        return _card(col)

    # portfolio_dna_tab, emotional_gauge_tab, time_machine_tab, trade_journal_tab, dream_architect_tab, stress_lab_tab removed

    def diversification_tab():
        result_container = ft.Container(
            content=ft.Text("Click 'Compute' to calculate diversification score", size=14, color="#aaaaaa"),
            padding=10,
        )
        def _compute(e):
            result_container.content = ft.Column([
                ft.ProgressRing(width=30, height=30, stroke_width=3, color="#00ff9d"),
                ft.Text("Computing...", size=12, color="#888888"),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            page.update()
            try:
                score = data.generate_diversification_score()
                result_container.content = ft.Text(score, size=18, color="#ffffff", font_family="Consolas")
            except Exception as ex:
                result_container.content = ft.Text(f"Error: {ex}", size=14, color="#ff3366")
            page.update()
        return _card(ft.Column([
            ft.Text("DIVERSIFICATION SCORE", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Button("COMPUTE", bgcolor="#00ff9d", color="#0a0a0a",
                       style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                       on_click=_compute),
            result_container,
        ], spacing=15))

    def market_regime_tab():
        regime = data.detect_market_regime()
        s = data.portfolio_summary_stats()
        rs = data.returns_series
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        # Compute regime indicators
        vol_20 = float(rs.tail(20).std() * np.sqrt(252)) if len(rs) >= 20 else 0
        vol_60 = float(rs.tail(60).std() * np.sqrt(252)) if len(rs) >= 60 else 0
        vol_all = s.get("ann_volatility", 0)
        mean_20 = float(rs.tail(20).mean() * 252) if len(rs) >= 20 else 0
        mean_60 = float(rs.tail(60).mean() * 252) if len(rs) >= 60 else 0
        # Trend detection
        if not rs.empty:
            cum = (1 + rs).cumprod()
            sma_20 = float(cum.tail(20).mean()) if len(cum) >= 20 else 0
            sma_50 = float(cum.tail(50).mean()) if len(cum) >= 50 else 0
            sma_200 = float(cum.tail(200).mean()) if len(cum) >= 200 else 0
            current = float(cum.iloc[-1]) if len(cum) > 0 else 0
            above_20 = current > sma_20 if sma_20 > 0 else False
            above_50 = current > sma_50 if sma_50 > 0 else False
            above_200 = current > sma_200 if sma_200 > 0 else False
            trend_score = sum([above_20, above_50, above_200])
            trend_label = "STRONG UPTREND" if trend_score == 3 else "UPTREND" if trend_score == 2 else "NEUTRAL" if trend_score == 1 else "DOWNTREND"
            trend_color = "#00ff9d" if trend_score >= 2 else "#ffd700" if trend_score == 1 else "#ff3366"
        else:
            trend_label = "N/A"
            trend_color = "#888888"
            above_20 = above_50 = above_200 = False
            trend_score = 0
        vol_regime = "HIGH VOL" if vol_20 > 0.25 else "NORMAL" if vol_20 > 0.12 else "LOW VOL"
        vol_color = "#ff3366" if vol_20 > 0.25 else "#ffd700" if vol_20 > 0.12 else "#00ff9d"
        momentum_20 = mean_20
        momentum_color = "#00ff9d" if momentum_20 > 0 else "#ff3366"

        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10)
        col.controls.append(ft.Text("MARKET REGIME DETECTOR", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Container(
            content=ft.Text(regime, size=20, color="#ffffff", weight=ft.FontWeight.BOLD),
            bgcolor="#1a1a2e", padding=15, border_radius=10,
        ))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("TREND ANALYSIS", size=15, weight=ft.FontWeight.BOLD, color="#00bfff"))
        col.controls.append(ft.Row([
            _kpi("Trend", trend_label, trend_color),
            _kpi("Above SMA-20", "YES" if above_20 else "NO", "#00ff9d" if above_20 else "#ff3366"),
            _kpi("Above SMA-50", "YES" if above_50 else "NO", "#00ff9d" if above_50 else "#ff3366"),
            _kpi("Above SMA-200", "YES" if above_200 else "NO", "#00ff9d" if above_200 else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("VOLATILITY REGIME", size=15, weight=ft.FontWeight.BOLD, color="#ff9900"))
        col.controls.append(ft.Row([
            _kpi("Regime", vol_regime, vol_color),
            _kpi("20-Day Vol", f"{vol_20:.1%}" if vol_20 else "N/A", vol_color),
            _kpi("60-Day Vol", f"{vol_60:.1%}" if vol_60 else "N/A", "#ffd700"),
            _kpi("All-Time Vol", f"{vol_all:.1%}" if vol_all else "N/A", "#aaaaaa"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("MOMENTUM", size=15, weight=ft.FontWeight.BOLD, color="#9966ff"))
        col.controls.append(ft.Row([
            _kpi("20-Day Annualized", f"{momentum_20:.1%}" if momentum_20 else "N/A", momentum_color),
            _kpi("60-Day Annualized", f"{mean_60:.1%}" if mean_60 else "N/A", "#00ff9d" if mean_60 > 0 else "#ff3366"),
            _kpi("Trend Score", f"{trend_score}/3", trend_color),
            _kpi("Max Drawdown", f"{s.get('max_drawdown', 0):.1%}" if s.get('max_drawdown', 0) != 0 else "N/A", "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Text("Trend Score: 3=strong uptrend, 2=uptrend, 1=neutral, 0=downtrend (based on SMA crossovers)", size=10, color="#666666", italic=True))
        return _card(col)

    # voice_tab, monte_carlo_tab, constellation_tab, settings_tab removed

    # planner_tab removed

    # monthly_comparison_tab removed

    # ====================== NEW TAB BUILDERS ======================
    def drawdown_tab():
        dd = data.max_drawdown()
        rec = data.recovery_time()
        s = data.portfolio_summary_stats()
        rs = data.returns_series
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        # Compute drawdown series stats
        dd_duration = 0
        time_underwater_pct = 0
        avg_dd = 0
        if not rs.empty:
            cum = (1 + rs).cumprod()
            running_max = cum.cummax()
            dd_series = (cum - running_max) / running_max
            time_underwater_pct = (dd_series < -0.001).sum() / len(dd_series) * 100 if len(dd_series) > 0 else 0
            avg_dd = float(dd_series[dd_series < 0].mean()) if (dd_series < 0).any() else 0
            # Max DD duration in days
            in_dd = dd_series < -0.001
            if in_dd.any():
                groups = (~in_dd).cumsum()
                dd_lengths = in_dd.groupby(groups).sum()
                dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10)
        col.controls.append(ft.Text("DRAWDOWN ANALYSIS", size=22, weight=ft.FontWeight.BOLD, color="#ff3366"))
        col.controls.append(ft.Text("Complete drawdown history, duration, and recovery analysis", size=12, color="#aaaaaa"))
        col.controls.append(ft.Row([
            _kpi("Max Drawdown", f"{dd:.1%}" if dd != 0 else "N/A", "#ff3366"),
            _kpi("Avg Drawdown", f"{avg_dd:.1%}" if avg_dd != 0 else "N/A", "#ff9900"),
            _kpi("Max DD Duration", f"{dd_duration}d" if dd_duration > 0 else "N/A", "#ff3366"),
            _kpi("Time Underwater", f"{time_underwater_pct:.1f}%", "#ff9900" if time_underwater_pct > 50 else "#ffd700"),
        ], spacing=6))
        col.controls.append(ft.Row([
            _kpi("Max Recovery", f"{rec['max_recovery_days']}d" if rec['max_recovery_days'] > 0 else "N/A", "#ffd700"),
            _kpi("Avg Recovery", f"{rec.get('avg_recovery_days', 0):.0f}d" if rec.get('avg_recovery_days', 0) > 0 else "N/A", "#ffd700"),
            _kpi("In Drawdown?", "YES" if rec['currently_in_drawdown'] else "NO", "#ff3366" if rec['currently_in_drawdown'] else "#00ff9d"),
            _kpi("Current DD Days", f"{rec['current_dd_days']}d" if rec['current_dd_days'] > 0 else "N/A", "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Row([
            _kpi("Calmar Ratio", f"{s.get('calmar', 0):.2f}" if s.get('calmar', 0) != 0 else "N/A", "#00ff9d" if s.get('calmar', 0) > 0 else "#ff3366"),
            _kpi("VaR 95%", f"{s.get('var_95', 0):.2%}" if s.get('var_95', 0) != 0 else "N/A", "#ff3366"),
            _kpi("CVaR 95%", f"{s.get('cvar_95', 0):.2%}" if s.get('cvar_95', 0) != 0 else "N/A", "#ff3366"),
            _kpi("Volatility", f"{s.get('ann_volatility', 0):.1%}" if s.get('ann_volatility', 0) != 0 else "N/A", "#ff9900"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        # --- Drawdown chart with toggles ---
        dd_opts = {'dollar': False, 'underwater': False, 'recovery': False}
        dd_chart_c = ft.Container(content=_chart_image(data.chart_drawdown()))
        def _dd_refresh():
            dd_chart_c.content = _chart_image(data.chart_drawdown(
                show_dollar=dd_opts['dollar'], show_underwater=dd_opts['underwater'],
                show_recovery_bands=dd_opts['recovery']))
            page.update()
        def _dd_tog(key):
            def handler(e):
                dd_opts[key] = e.control.value
                _dd_refresh()
            return handler
        dd_toggles = ft.Row([
            ft.Switch(label="Dollar ($)", value=False, active_color="#ff3366",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_dd_tog('dollar')),
            ft.Switch(label="Underwater Shade", value=False, active_color="#ff9900",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_dd_tog('underwater')),
            ft.Switch(label="Recovery Bands", value=False, active_color="#ffd700",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_dd_tog('recovery')),
        ], spacing=8, scroll=ft.ScrollMode.AUTO)
        col.controls.append(ft.Text("DRAWDOWN CHART", size=15, weight=ft.FontWeight.BOLD, color="#ff3366"))
        col.controls.append(dd_toggles)
        col.controls.append(dd_chart_c)
        return _card(col)

    def holdings_bar_tab():
        holdings_bar_state = {'mode': '$'}
        bar_chart_container = ft.Container(
            content=_chart_image(data.chart_holdings_bar(mode='$')),
            padding=0,
        )

        def on_bar_mode_change(e):
            holdings_bar_state['mode'] = e.control.value
            bar_chart_container.content = _chart_image(data.chart_holdings_bar(mode=holdings_bar_state['mode']))
            page.update()

        bar_mode_dropdown = ft.Dropdown(
            value="$", width=160,
            options=[
                ft.dropdown.Option("$", "Market Value ($)"),
                ft.dropdown.Option("%", "Weight (%)"),
                ft.dropdown.Option("gl", "Gain/Loss ($)"),
                ft.dropdown.Option("gl%", "Gain/Loss (%)"),
                ft.dropdown.Option("cb", "Cost Basis ($)"),
            ],
            on_select=on_bar_mode_change,
            border_color="#ffd700", color="#ffffff", bgcolor="#1a1a2e",
        )

        return _card(ft.Column([
            ft.Text("HOLDINGS BAR CHART", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Text("All open positions", size=12, color="#aaaaaa"),
            ft.Row([
                ft.Text("Display:", size=12, color="#aaaaaa"),
                bar_mode_dropdown,
            ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bar_chart_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def returns_dist_tab():
        hist_opts = {'kde': False, 'cvar': False, 'normal': False, 'log': False}
        hist_chart_c = ft.Container(content=_chart_image(data.chart_returns_histogram()))
        def _hist_refresh():
            hist_chart_c.content = _chart_image(data.chart_returns_histogram(
                show_kde=hist_opts['kde'], show_cvar=hist_opts['cvar'],
                show_normal=hist_opts['normal'], log_returns=hist_opts['log']))
            page.update()
        def _hist_tog(key):
            def handler(e):
                hist_opts[key] = e.control.value
                _hist_refresh()
            return handler
        hist_toggles = ft.Row([
            ft.Switch(label="KDE Curve", value=False, active_color="#00bfff",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_hist_tog('kde')),
            ft.Switch(label="CVaR Line", value=False, active_color="#ff9900",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_hist_tog('cvar')),
            ft.Switch(label="Normal Fit", value=False, active_color="#9966ff",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_hist_tog('normal')),
            ft.Switch(label="Log Returns", value=False, active_color="#ffd700",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_hist_tog('log')),
        ], spacing=8, scroll=ft.ScrollMode.AUTO)
        return _card(ft.Column([
            ft.Text("RETURNS DISTRIBUTION", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            hist_toggles,
            hist_chart_c,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def sector_tab():
        return _card(ft.Column([
            ft.Text("SECTOR BREAKDOWN", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            _chart_image(data.chart_sector_pie()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    # gain_loss_tab removed

    def rolling_sharpe_tab():
        rs = data.returns_series
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        # Compute rolling sharpe stats
        current_sharpe = 0
        avg_sharpe = 0
        min_sharpe = 0
        max_sharpe = 0
        pct_above_1 = 0
        if len(rs) >= 60:
            rolling_s = rs.rolling(60).mean() / rs.rolling(60).std() * np.sqrt(252)
            rolling_s = rolling_s.dropna()
            if len(rolling_s) > 0:
                current_sharpe = float(rolling_s.iloc[-1])
                avg_sharpe = float(rolling_s.mean())
                min_sharpe = float(rolling_s.min())
                max_sharpe = float(rolling_s.max())
                pct_above_1 = float((rolling_s > 1).sum() / len(rolling_s) * 100)
        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10)
        col.controls.append(ft.Text("ROLLING SHARPE RATIO", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Text("60-day rolling window annualized Sharpe ratio", size=12, color="#aaaaaa"))
        col.controls.append(ft.Row([
            _kpi("Current", f"{current_sharpe:.2f}" if current_sharpe else "N/A", "#00ff9d" if current_sharpe > 1 else "#ffd700" if current_sharpe > 0 else "#ff3366"),
            _kpi("Average", f"{avg_sharpe:.2f}" if avg_sharpe else "N/A", "#00ff9d" if avg_sharpe > 0 else "#ff3366"),
            _kpi("Min", f"{min_sharpe:.2f}" if min_sharpe else "N/A", "#ff3366"),
            _kpi("Max", f"{max_sharpe:.2f}" if max_sharpe else "N/A", "#00ff9d"),
            _kpi("% Above 1.0", f"{pct_above_1:.1f}%", "#00ff9d" if pct_above_1 > 50 else "#ffd700"),
        ], spacing=6))
        # --- Rolling Sharpe chart with toggles ---
        rs_opts = {'window': 60, 'sortino': False, 'avg': False}
        rs_chart_c = ft.Container(content=_chart_image(data.chart_rolling_sharpe()))
        def _rs_refresh():
            rs_chart_c.content = _chart_image(data.chart_rolling_sharpe(
                window=rs_opts['window'], show_sortino=rs_opts['sortino'], show_avg=rs_opts['avg']))
            page.update()
        def _rs_tog(key):
            def handler(e):
                rs_opts[key] = e.control.value
                _rs_refresh()
            return handler
        def _rs_win(e):
            try:
                rs_opts['window'] = int(e.control.value)
            except Exception:
                rs_opts['window'] = 60
            _rs_refresh()
        rs_toggles = ft.Row([
            ft.Text("Window:", size=11, color="#aaaaaa"),
            ft.Dropdown(value="60", width=90, options=[
                ft.dropdown.Option("30", "30d"), ft.dropdown.Option("60", "60d"),
                ft.dropdown.Option("90", "90d"), ft.dropdown.Option("120", "120d"),
            ], on_select=_rs_win, border_color="#00ff9d", color="#ffffff", bgcolor="#1a1a2e"),
            ft.Switch(label="Sortino Overlay", value=False, active_color="#ff9900",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_rs_tog('sortino')),
            ft.Switch(label="Show Average", value=False, active_color="#00bfff",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_rs_tog('avg')),
        ], spacing=8, scroll=ft.ScrollMode.AUTO)
        col.controls.append(rs_toggles)
        col.controls.append(rs_chart_c)
        return _card(col)

    def rolling_vol_tab():
        rs = data.returns_series
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        current_vol = 0
        avg_vol = 0
        min_vol = 0
        max_vol = 0
        vol_regime = "N/A"
        if len(rs) >= 30:
            rolling_v = rs.rolling(30).std() * np.sqrt(252)
            rolling_v = rolling_v.dropna()
            if len(rolling_v) > 0:
                current_vol = float(rolling_v.iloc[-1])
                avg_vol = float(rolling_v.mean())
                min_vol = float(rolling_v.min())
                max_vol = float(rolling_v.max())
                vol_regime = "HIGH" if current_vol > avg_vol * 1.5 else "ELEVATED" if current_vol > avg_vol else "NORMAL" if current_vol > avg_vol * 0.5 else "LOW"
        vol_color = "#ff3366" if vol_regime == "HIGH" else "#ff9900" if vol_regime == "ELEVATED" else "#ffd700" if vol_regime == "NORMAL" else "#00ff9d"
        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10)
        col.controls.append(ft.Text("ROLLING VOLATILITY", size=22, weight=ft.FontWeight.BOLD, color="#ff9900"))
        col.controls.append(ft.Text("30-day rolling window annualized volatility", size=12, color="#aaaaaa"))
        col.controls.append(ft.Row([
            _kpi("Current", f"{current_vol:.1%}" if current_vol else "N/A", vol_color),
            _kpi("Average", f"{avg_vol:.1%}" if avg_vol else "N/A", "#ffd700"),
            _kpi("Min", f"{min_vol:.1%}" if min_vol else "N/A", "#00ff9d"),
            _kpi("Max", f"{max_vol:.1%}" if max_vol else "N/A", "#ff3366"),
            _kpi("Regime", vol_regime, vol_color),
        ], spacing=6))
        # --- Rolling Vol chart with toggles ---
        rv_opts = {'window': 30, 'bands': False, 'regime': False}
        rv_chart_c = ft.Container(content=_chart_image(data.chart_rolling_volatility()))
        def _rv_refresh():
            rv_chart_c.content = _chart_image(data.chart_rolling_volatility(
                window=rv_opts['window'], show_bands=rv_opts['bands'], show_regime=rv_opts['regime']))
            page.update()
        def _rv_tog(key):
            def handler(e):
                rv_opts[key] = e.control.value
                _rv_refresh()
            return handler
        def _rv_win(e):
            try:
                rv_opts['window'] = int(e.control.value)
            except Exception:
                rv_opts['window'] = 30
            _rv_refresh()
        rv_toggles = ft.Row([
            ft.Text("Window:", size=11, color="#aaaaaa"),
            ft.Dropdown(value="30", width=90, options=[
                ft.dropdown.Option("20", "20d"), ft.dropdown.Option("30", "30d"),
                ft.dropdown.Option("60", "60d"), ft.dropdown.Option("90", "90d"),
            ], on_select=_rv_win, border_color="#ff9900", color="#ffffff", bgcolor="#1a1a2e"),
            ft.Switch(label="±1σ Bands", value=False, active_color="#ffd700",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_rv_tog('bands')),
            ft.Switch(label="Regime Shade", value=False, active_color="#ff3366",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_rv_tog('regime')),
        ], spacing=8, scroll=ft.ScrollMode.AUTO)
        col.controls.append(rv_toggles)
        col.controls.append(rv_chart_c)
        return _card(col)

    def monthly_heatmap_tab():
        hm_container = ft.Container(
            content=ft.Text("Click 'Compute' to generate monthly heatmap", size=14, color="#aaaaaa"),
            padding=10,
        )
        def _compute(e):
            hm_container.content = ft.Column([
                ft.ProgressRing(width=30, height=30, stroke_width=3, color="#ffd700"),
                ft.Text("Generating heatmap...", size=12, color="#888888"),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            page.update()
            try:
                hm_container.content = _chart_image(data.chart_monthly_heatmap())
            except Exception as ex:
                hm_container.content = ft.Text(f"Error: {ex}", size=14, color="#ff3366")
            page.update()
        return _card(ft.Column([
            ft.Text("MONTHLY RETURNS HEATMAP", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Button("COMPUTE", bgcolor="#ffd700", color="#0a0a0a",
                       style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                       on_click=_compute),
            hm_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    # weight_pie_tab, growth_projection_tab, efficient_frontier_tab removed

    def summary_stats_tab():
        stats_container = ft.Container(
            content=ft.Text("Click 'Compute' to load summary statistics", size=14, color="#aaaaaa"),
            padding=10,
        )
        def _compute(e):
            stats_container.content = ft.Column([
                ft.ProgressRing(width=30, height=30, stroke_width=3, color="#00ff9d"),
                ft.Text("Computing summary stats...", size=12, color="#888888"),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            page.update()
            try:
                s = data.portfolio_summary_stats()
                ra = data.risk_adjusted_returns()
                pf_val = data.profit_factor()
                wl = data.win_loss_streaks()
                rec = data.recovery_time()
                rs = data.returns_series
                def _kpi(label, value, color="#ffd700"):
                    return ft.Container(
                        content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2),
                        bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True,
                    )
                def _kpi_sm(label, value, color="#ffd700"):
                    return ft.Container(
                        content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(str(value), size=13, weight=ft.FontWeight.BOLD, color=color)], spacing=1),
                        bgcolor="#1a1a2e", padding=6, border_radius=8, expand=True,
                    )
                def _v(val, fmt=".2f"):
                    if val == 0: return "N/A"
                    return f"{val:{fmt}}"
                gl = s['total_gain_loss']
                gl_c = "#00ff9d" if gl >= 0 else "#ff3366"
                # Extra computed stats
                avg_win = float(rs[rs > 0].mean()) if not rs.empty and (rs > 0).any() else 0
                avg_loss = float(rs[rs < 0].mean()) if not rs.empty and (rs < 0).any() else 0
                payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                tail_ratio_val = 0
                if not rs.empty and len(rs) > 10:
                    p95 = float(np.percentile(rs, 95))
                    p5 = float(np.percentile(rs, 5))
                    tail_ratio_val = abs(p95 / p5) if p5 != 0 else 0
                downside_dev = float(rs[rs < 0].std() * np.sqrt(252)) if not rs.empty and (rs < 0).any() else 0

                stats_container.content = ft.Column([
                    ft.Text("PORTFOLIO VALUE", size=14, weight=ft.FontWeight.BOLD, color="#ffd700"),
                    ft.Row([_kpi("Total Value", f"${s['total_value']:,.0f}"), _kpi("Cost Basis", f"${s.get('total_cost_basis', 0):,.0f}" if s.get('total_cost_basis') else "N/A", "#aaaaaa"), _kpi("Total G/L", f"${gl:,.0f}" if gl != 0 else "N/A", gl_c), _kpi("G/L %", f"{s.get('total_gain_pct', 0):+.1f}%" if s.get('total_gain_pct') else "N/A", gl_c), _kpi("Holdings", str(s['num_holdings']) if s['num_holdings'] > 0 else "0")], spacing=6),
                    ft.Divider(height=1, color="#222222"),
                    ft.Text("RETURN & RISK", size=14, weight=ft.FontWeight.BOLD, color="#00ff9d"),
                    ft.Row([_kpi("Ann. Return", f"{s['ann_return']:.1%}" if s['ann_return'] != 0 else "N/A", "#00ff9d"), _kpi("Ann. Vol", f"{s['ann_volatility']:.1%}" if s['ann_volatility'] != 0 else "N/A", "#ff3366"), _kpi("Max DD", f"{s['max_drawdown']:.1%}" if s['max_drawdown'] != 0 else "N/A", "#ff3366"), _kpi("Downside Dev", f"{downside_dev:.1%}" if downside_dev else "N/A", "#ff9900")], spacing=6),
                    ft.Divider(height=1, color="#222222"),
                    ft.Text("RISK-ADJUSTED RATIOS", size=14, weight=ft.FontWeight.BOLD, color="#00bfff"),
                    ft.Row([_kpi("Sharpe", _v(s['sharpe'])), _kpi("Sortino", _v(s['sortino'])), _kpi("Calmar", _v(s['calmar'])), _kpi("Treynor", _v(ra.get('treynor', 0)))], spacing=6),
                    ft.Divider(height=1, color="#222222"),
                    ft.Text("TAIL RISK", size=14, weight=ft.FontWeight.BOLD, color="#ff3366"),
                    ft.Row([_kpi("VaR 95%", f"{s['var_95']:.2%}" if s['var_95'] != 0 else "N/A", "#ff3366"), _kpi("CVaR 95%", f"{s['cvar_95']:.2%}" if s['cvar_95'] != 0 else "N/A", "#ff3366"), _kpi("Tail Ratio", f"{tail_ratio_val:.2f}" if tail_ratio_val else "N/A", "#00ff9d" if tail_ratio_val > 1 else "#ff3366"), _kpi("Skewness", _v(s['skewness'], ".3f"), "#00ff9d" if s['skewness'] > 0 else "#ff3366")], spacing=6),
                    ft.Divider(height=1, color="#222222"),
                    ft.Text("TRADING PERFORMANCE", size=14, weight=ft.FontWeight.BOLD, color="#9966ff"),
                    ft.Row([_kpi("Win Rate", f"{s['win_rate']:.0%}" if s['win_rate'] != 0 else "N/A", "#00ff9d"), _kpi("Profit Factor", f"{pf_val:.2f}" if pf_val else "N/A", "#00ff9d" if pf_val > 1 else "#ff3366"), _kpi("Payoff Ratio", f"{payoff:.2f}" if payoff else "N/A", "#00ff9d" if payoff > 1 else "#ff3366"), _kpi("Avg Win", f"{avg_win:.2%}" if avg_win else "N/A", "#00ff9d")], spacing=6),
                    ft.Row([_kpi("Avg Loss", f"{avg_loss:.2%}" if avg_loss else "N/A", "#ff3366"), _kpi("Best Day", f"{s['best_day']:.2%}" if s['best_day'] != 0 else "N/A", "#00ff9d"), _kpi("Worst Day", f"{s['worst_day']:.2%}" if s['worst_day'] != 0 else "N/A", "#ff3366"), _kpi("Kurtosis", _v(s['kurtosis'], ".3f"), "#ffd700")], spacing=6),
                    ft.Divider(height=1, color="#222222"),
                    ft.Text("CAPM & MARKET", size=14, weight=ft.FontWeight.BOLD, color="#ffd700"),
                    ft.Row([_kpi("Beta (SPY)", _v(ra.get('beta', 0)), "#ffd700"), _kpi("Alpha", f"{ra.get('alpha', 0):.2%}" if ra.get('alpha') else "N/A", "#00ff9d" if ra.get('alpha', 0) > 0 else "#ff3366"), _kpi("Info Ratio", _v(ra.get('info_ratio', 0)), "#00ff9d" if ra.get('info_ratio', 0) > 0 else "#ff3366"), _kpi("Win Streak", f"{wl['longest_win']}d" if wl['longest_win'] > 0 else "N/A", "#00ff9d")], spacing=6),
                    ft.Divider(height=1, color="#222222"),
                    ft.Text("RECOVERY & STREAKS", size=14, weight=ft.FontWeight.BOLD, color="#ff9900"),
                    ft.Row([
                        _kpi_sm("Max Recovery", f"{rec['max_recovery_days']}d" if rec['max_recovery_days'] > 0 else "N/A"),
                        _kpi_sm("Pos Days", str(s.get('positive_days', 0)) if s.get('positive_days') else "N/A", "#00ff9d"),
                        _kpi_sm("Neg Days", str(s.get('negative_days', 0)) if s.get('negative_days') else "N/A", "#ff3366"),
                        _kpi_sm("Loss Streak", f"{wl['longest_loss']}d" if wl['longest_loss'] > 0 else "N/A", "#ff3366"),
                        _kpi_sm("In DD?", "YES" if rec['currently_in_drawdown'] else "NO", "#ff3366" if rec['currently_in_drawdown'] else "#00ff9d"),
                    ], spacing=4),
                ], spacing=6)
            except Exception as ex:
                stats_container.content = ft.Text(f"Error: {ex}", size=14, color="#ff3366")
            page.update()
        return _card(ft.Column([
            ft.Text("PORTFOLIO SUMMARY STATS", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Complete statistical overview — click Compute to load all metrics", size=12, color="#aaaaaa"),
            ft.Button("COMPUTE", bgcolor="#00ff9d", color="#0a0a0a",
                       style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                       on_click=_compute),
            stats_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    def top_movers_tab():
        gainers = data.top_gainers(5)
        losers = data.top_losers(5)
        col = ft.Column(spacing=10)
        col.controls.append(ft.Text("TOP GAINERS", size=18, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        if gainers:
            for g in gainers:
                col.controls.append(ft.Row([
                    ft.Text(g['symbol'], size=14, color="#ffffff", width=80, weight=ft.FontWeight.BOLD),
                    ft.Text(f"+{g['gain_pct']:.1f}%", size=14, color="#00ff9d", width=80),
                    ft.Text(f"${g['gain_dollar']:,.0f}", size=14, color="#00ff9d"),
                ], spacing=10))
        else:
            col.controls.append(ft.Text("Import data to see top gainers", size=14, color="#aaaaaa"))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("TOP LOSERS", size=18, weight=ft.FontWeight.BOLD, color="#ff3366"))
        if losers:
            for lo in losers:
                col.controls.append(ft.Row([
                    ft.Text(lo['symbol'], size=14, color="#ffffff", width=80, weight=ft.FontWeight.BOLD),
                    ft.Text(f"{lo['gain_pct']:.1f}%", size=14, color="#ff3366", width=80),
                    ft.Text(f"${lo['gain_dollar']:,.0f}", size=14, color="#ff3366"),
                ], spacing=10))
        else:
            col.controls.append(ft.Text("Import data to see top losers", size=14, color="#aaaaaa"))
        return _card(ft.Column([
            ft.Text("TOP MOVERS", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            col,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    # fire_tab removed

    def attribution_tab():
        attr = data.performance_attribution()
        if not attr:
            content = ft.Text("Import data for performance attribution", size=16, color="#aaaaaa")
        else:
            rows = []
            for a in attr[:15]:
                c = "#00ff9d" if a['contribution'] >= 0 else "#ff3366"
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(a['symbol'], color="#ffffff", weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{a['return']:.1%}", color="#00ff9d" if a['return'] >= 0 else "#ff3366")),
                    ft.DataCell(ft.Text(f"{a['weight']:.1%}", color="#ffffff")),
                    ft.DataCell(ft.Text(f"{a['contribution']:.2%}", color=c)),
                ]))
            content = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Symbol", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Return", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Weight", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Contribution", color="#00ff9d")),
                ],
                rows=rows, heading_row_color="#1a1a2e",
            )
        return _card(ft.Column([
            ft.Text("PERFORMANCE ATTRIBUTION", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            content,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    # rebalancer_tab removed

    # ====================== BENCHMARK / TRADE LOG / INCOME TABS ======================
    def benchmark_tab():
        bm = data.benchmark_comparison()
        all_benchmarks = ['SPY', 'QQQ', 'IWM', 'BTC-USD']
        # Also add user's individual holdings as possible overlays
        user_syms = []
        if not data.holdings.empty:
            user_syms = [s for s in data.holdings['symbol'].unique().tolist() if s and len(s) <= 10]
        # Colors for each possible line
        line_colors = {
            'Portfolio': '#00ff9d', 'SPY': '#ffd700', 'QQQ': '#ff3366',
            'IWM': '#00bfff', 'BTC-USD': '#ff9900', 'DIA': '#9966ff',
            'VTI': '#66ff66', 'GLD': '#cccc00', 'TLT': '#ff6699',
        }
        # Default: all OFF (none selected)
        active_lines = {'Portfolio': False}
        for bm_sym in all_benchmarks:
            active_lines[bm_sym] = False
        for sym in user_syms:
            if sym not in active_lines:
                active_lines[sym] = False

        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(value, size=14, weight=ft.FontWeight.BOLD, color=color)], spacing=1), bgcolor="#1a1a2e", padding=8, border_radius=8, expand=True)

        # Build relative performance KPIs
        relative_kpis = []
        port_stats = data.portfolio_summary_stats()
        port_ann_ret = port_stats.get('ann_return', 0) if port_stats else 0
        port_ann_vol = port_stats.get('ann_vol', 0) if port_stats else 0
        if bm and port_ann_ret != 0:
            relative_kpis.append(ft.Text("RELATIVE PERFORMANCE", size=14, weight=ft.FontWeight.BOLD, color="#9966ff"))
            rel_row = []
            for b in bm:
                bm_ret = b.get('ann_return', 0)
                excess = port_ann_ret - bm_ret if bm_ret else 0
                rel_row.append(_kpi(f"vs {b['symbol']}", f"{excess:+.1%}" if excess else "N/A", "#00ff9d" if excess > 0 else "#ff3366"))
            relative_kpis.append(ft.Row(rel_row, spacing=4))
            # Portfolio's own stats
            relative_kpis.append(ft.Row([
                _kpi("Portfolio Ann. Ret", f"{port_ann_ret:.1%}" if port_ann_ret else "N/A", "#00ff9d" if port_ann_ret > 0 else "#ff3366"),
                _kpi("Portfolio Ann. Vol", f"{port_ann_vol:.1%}" if port_ann_vol else "N/A", "#ff9900"),
            ], spacing=4))
            relative_kpis.append(ft.Divider(height=1, color="#333333"))

        # Build benchmark table
        if not bm:
            table_content = ft.Text("Loading benchmark data... (requires internet)", size=16, color="#aaaaaa")
        else:
            rows = []
            for b in bm:
                tr = b['total_return']
                corr = b['correlation']
                tr_str = f"{tr:.1%}" if tr is not None and not (isinstance(tr, float) and np.isnan(tr)) else "N/A"
                tr_color = "#00ff9d" if isinstance(tr, (int, float)) and not np.isnan(tr) and tr >= 0 else "#ff3366" if isinstance(tr, (int, float)) and not np.isnan(tr) else "#888888"
                corr_str = f"{corr:.2f}" if corr is not None else "N/A"
                corr_c = "#00ff9d" if corr is not None and corr > 0.5 else "#ffd700" if corr is not None and corr > 0 else "#ff3366" if corr is not None else "#888888"
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(b['symbol'], color="#ffffff", weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{b['ann_return']:.1%}", color="#00ff9d" if b['ann_return'] >= 0 else "#ff3366")),
                    ft.DataCell(ft.Text(f"{b['ann_vol']:.1%}", color="#ff3366")),
                    ft.DataCell(ft.Text(tr_str, color=tr_color)),
                    ft.DataCell(ft.Text(corr_str, color=corr_c)),
                ]))
            table_content = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Benchmark", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Ann. Return", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Ann. Vol", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Total Return", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Correlation", color="#00ff9d")),
                ],
                rows=rows, heading_row_color="#1a1a2e",
            )

        # Time range state
        current_timerange = {'value': 'ALL'}

        # Chart container that will be updated on toggle - start empty since none selected
        chart_container = ft.Container(
            content=ft.Text("Select benchmarks to compare", color="#aaaaaa", size=14),
            padding=0,
        )

        # Correlation & spread overlay state
        bm_chart_opts = {'corr_labels': False, 'spread': False}

        def _rebuild_chart():
            selected = [s for s, on in active_lines.items() if on and s != 'Portfolio']
            show_portfolio = active_lines.get('Portfolio', True)
            tr = current_timerange['value']
            if not selected and not show_portfolio:
                chart_container.content = ft.Text("Select at least one line to display", color="#aaaaaa", size=14)
            else:
                chart_container.content = _chart_image(
                    data.chart_benchmark_overlay(benchmarks=selected if selected else None,
                                                  show_portfolio=show_portfolio, timerange=tr,
                                                  show_corr_labels=bm_chart_opts['corr_labels'],
                                                  show_spread=bm_chart_opts['spread'])
                )
            page.update()

        def _on_toggle(sym):
            def handler(e):
                active_lines[sym] = e.control.value
                _rebuild_chart()
            return handler

        def _on_timerange(tr):
            def handler(e):
                current_timerange['value'] = tr
                for btn in timerange_row.controls:
                    if hasattr(btn, 'data'):
                        if btn.data == tr:
                            btn.bgcolor = '#00ff9d'
                            btn.color = '#0a0a0a'
                        else:
                            btn.bgcolor = '#1a1a2e'
                            btn.color = '#aaaaaa'
                _rebuild_chart()
            return handler

        # Time range buttons
        timerange_options = ['1D', '1W', '1M', '3M', '6M', '1Y', '5Y', 'ALL']
        tr_buttons = []
        for tr in timerange_options:
            is_active = (tr == 'ALL')
            tr_buttons.append(
                ft.Button(
                    tr, data=tr,
                    bgcolor='#00ff9d' if is_active else '#1a1a2e',
                    color='#0a0a0a' if is_active else '#aaaaaa',
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8),
                                         padding=ft.Padding(left=12, right=12, top=6, bottom=6)),
                    on_click=_on_timerange(tr),
                    height=32,
                )
            )
        timerange_row = ft.Row(controls=tr_buttons, spacing=4, wrap=True)

        # Build toggle switches - grouped into rows
        toggle_controls = []
        # Portfolio toggle first
        toggle_controls.append(
            ft.Container(
                content=ft.Row([
                    ft.Container(width=12, height=12, bgcolor=line_colors.get('Portfolio', '#00ff9d'),
                                 border_radius=6),
                    ft.Text("Portfolio", color="#ffffff", size=12, weight=ft.FontWeight.BOLD),
                    ft.Switch(value=False,
                              active_color="#00ff9d", on_change=_on_toggle('Portfolio')),
                ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                padding=ft.Padding(right=12),
            )
        )
        # Benchmark toggles
        for sym in all_benchmarks:
            c = line_colors.get(sym, '#ffffff')
            toggle_controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Container(width=12, height=12, bgcolor=c, border_radius=6),
                        ft.Text(sym, color="#ffffff", size=12, weight=ft.FontWeight.BOLD),
                        ft.Switch(value=False,
                                  active_color=c, on_change=_on_toggle(sym)),
                    ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=ft.Padding(right=12),
                )
            )
        # Individual holdings toggles (off by default)
        for sym in user_syms[:10]:
            if sym in all_benchmarks:
                continue
            c = line_colors.get(sym, '#888888')
            toggle_controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Container(width=12, height=12, bgcolor=c, border_radius=6),
                        ft.Text(sym, color="#aaaaaa", size=11),
                        ft.Switch(value=False, active_color="#00ff9d", on_change=_on_toggle(sym)),
                    ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=ft.Padding(right=12),
                )
            )

        toggle_row = ft.Row(
            controls=toggle_controls,
            wrap=True, spacing=4, run_spacing=4,
        )

        # Correlation / spread overlay toggles
        def _bm_opt_tog(key):
            def handler(e):
                bm_chart_opts[key] = e.control.value
                _rebuild_chart()
            return handler
        overlay_toggles = ft.Row([
            ft.Switch(label="Correlation Labels (ρ)", value=False, active_color="#9966ff",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_bm_opt_tog('corr_labels')),
            ft.Switch(label="Spread Shading", value=False, active_color="#00bfff",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_bm_opt_tog('spread')),
        ], spacing=8)

        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10)
        col.controls.append(ft.Text("BENCHMARK COMPARISON", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(ft.Text("Compare portfolio performance against major benchmarks", size=12, color="#aaaaaa"))
        for kpi in relative_kpis:
            col.controls.append(kpi)
        col.controls.append(table_content)
        col.controls.append(ft.Container(
            content=ft.Column([
                ft.Text("Toggle Lines:", size=13, color="#aaaaaa", weight=ft.FontWeight.BOLD),
                toggle_row,
                ft.Divider(height=1, color="#333333"),
                ft.Text("Time Range:", size=13, color="#aaaaaa", weight=ft.FontWeight.BOLD),
                timerange_row,
                ft.Divider(height=1, color="#333333"),
                ft.Text("Chart Overlays:", size=13, color="#aaaaaa", weight=ft.FontWeight.BOLD),
                overlay_toggles,
            ], spacing=6),
            bgcolor="#1a1a2e", border_radius=8, padding=10, margin=ft.Margin(top=8, bottom=4),
        ))
        col.controls.append(chart_container)
        return _card(col)

    def trade_log_tab():
        trades = data.get_trade_log(80)
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(value, size=14, weight=ft.FontWeight.BOLD, color=color)], spacing=1), bgcolor="#1a1a2e", padding=8, border_radius=8, expand=True)
        if not trades:
            content = ft.Text("No trade data loaded. Import data first.", size=16, color="#aaaaaa")
            summary_kpis = []
        else:
            # Compute trade summary stats
            total_trades = len(trades)
            total_gl = sum(t['gain_loss'] for t in trades)
            winners = [t for t in trades if t['gain_loss'] >= 0]
            losers = [t for t in trades if t['gain_loss'] < 0]
            n_win = len(winners)
            n_lose = len(losers)
            win_rate = n_win / total_trades * 100 if total_trades else 0
            avg_gl = total_gl / total_trades if total_trades else 0
            avg_win = sum(t['gain_loss'] for t in winners) / n_win if n_win else 0
            avg_loss = sum(t['gain_loss'] for t in losers) / n_lose if n_lose else 0
            biggest_win = max((t['gain_loss'] for t in trades), default=0)
            biggest_loss = min((t['gain_loss'] for t in trades), default=0)
            unique_syms = len(set(t['symbol'] for t in trades))
            total_value = sum(t['quantity'] * t['price'] for t in trades)
            summary_kpis = [
                ft.Text("TRADE SUMMARY", size=14, weight=ft.FontWeight.BOLD, color="#00bfff"),
                ft.Row([
                    _kpi("Total Trades", str(total_trades), "#00bfff"),
                    _kpi("Net G/L", f"${total_gl:,.0f}", "#00ff9d" if total_gl >= 0 else "#ff3366"),
                    _kpi("Win Rate", f"{win_rate:.1f}%", "#00ff9d" if win_rate >= 50 else "#ff3366"),
                    _kpi("Winners", str(n_win), "#00ff9d"),
                    _kpi("Losers", str(n_lose), "#ff3366"),
                ], spacing=4),
                ft.Row([
                    _kpi("Avg G/L", f"${avg_gl:,.0f}", "#00ff9d" if avg_gl >= 0 else "#ff3366"),
                    _kpi("Avg Win", f"${avg_win:,.0f}", "#00ff9d"),
                    _kpi("Avg Loss", f"${avg_loss:,.0f}", "#ff3366"),
                    _kpi("Best Trade", f"${biggest_win:,.0f}", "#00ff9d"),
                    _kpi("Worst Trade", f"${biggest_loss:,.0f}", "#ff3366"),
                ], spacing=4),
                ft.Row([
                    _kpi("Unique Symbols", str(unique_syms), "#9966ff"),
                    _kpi("Total Value", f"${total_value:,.0f}", "#ffd700"),
                ], spacing=4),
                ft.Divider(height=1, color="#333333"),
            ]
            rows = []
            for t in trades[:50]:
                gl_c = "#00ff9d" if t['gain_loss'] >= 0 else "#ff3366"
                trade_value = t['quantity'] * t['price']
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(t['date'], color="#aaaaaa")),
                    ft.DataCell(ft.Text(t['symbol'], color="#ffffff", weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{t['quantity']:.2f}", color="#ffffff")),
                    ft.DataCell(ft.Text(f"${t['price']:,.2f}", color="#ffd700")),
                    ft.DataCell(ft.Text(f"${trade_value:,.0f}", color="#00bfff")),
                    ft.DataCell(ft.Text(f"${t['cost_basis']:,.0f}", color="#aaaaaa")),
                    ft.DataCell(ft.Text(f"${t['gain_loss']:,.0f}", color=gl_c)),
                ]))
            content = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Date", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Symbol", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Qty", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Price", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Value", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Cost Basis", color="#00ff9d")),
                    ft.DataColumn(ft.Text("G/L", color="#00ff9d")),
                ],
                rows=rows, heading_row_color="#1a1a2e",
            )
        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10)
        col.controls.append(ft.Text("TRADE LOG", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Text(f"Showing latest {min(50, len(trades))} of {len(trades)} records", size=12, color="#aaaaaa"))
        for kpi in summary_kpis:
            col.controls.append(kpi)
        col.controls.append(content)
        return _card(col)

    # income_tracker_tab removed

    # whatif_tab removed

    # tax_optimizer_tab removed

    def correlation_dive_tab():
        cd = data.correlation_deep_dive()
        corr_mat = data.correlation_matrix()
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(value, size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        def _kpi_sm(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(value, size=13, weight=ft.FontWeight.BOLD, color=color)], spacing=1), bgcolor="#1a1a2e", padding=6, border_radius=8, expand=True)
        most = cd['most_correlated']
        least = cd['least_correlated']
        # Extra correlation stats
        n_pairs = len(cd['pairs'])
        high_corr_pairs = sum(1 for p in cd['pairs'] if abs(p['corr']) > 0.7)
        neg_corr_pairs = sum(1 for p in cd['pairs'] if p['corr'] < 0)
        moderate_pairs = sum(1 for p in cd['pairs'] if 0.3 < abs(p['corr']) <= 0.7)
        low_corr_pairs = sum(1 for p in cd['pairs'] if abs(p['corr']) <= 0.3)
        median_corr = float(np.median([p['corr'] for p in cd['pairs']])) if cd['pairs'] else 0
        std_corr = float(np.std([p['corr'] for p in cd['pairs']])) if cd['pairs'] else 0
        # Diversification quality rating
        if cd['avg_corr'] < 0.2:
            div_quality = "EXCELLENT"
            div_color = "#00ff9d"
        elif cd['avg_corr'] < 0.4:
            div_quality = "GOOD"
            div_color = "#00ff9d"
        elif cd['avg_corr'] < 0.6:
            div_quality = "MODERATE"
            div_color = "#ffd700"
        else:
            div_quality = "POOR"
            div_color = "#ff3366"

        if not cd['pairs']:
            tbl = ft.Text("Need 2+ assets for correlation analysis.", size=14, color="#aaaaaa")
        else:
            rows = []
            for p in cd['pairs']:
                c_color = "#ff3366" if p['corr'] > 0.7 else "#ffd700" if p['corr'] > 0.3 else "#00ff9d" if p['corr'] > -0.3 else "#00bfff"
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(p['a'], color="#ffffff")),
                    ft.DataCell(ft.Text(p['b'], color="#ffffff")),
                    ft.DataCell(ft.Text(f"{p['corr']:.3f}", color=c_color, weight=ft.FontWeight.BOLD)),
                ]))
            tbl = ft.DataTable(
                columns=[ft.DataColumn(ft.Text("Asset A", color="#00ff9d")), ft.DataColumn(ft.Text("Asset B", color="#00ff9d")),
                         ft.DataColumn(ft.Text("Correlation", color="#00ff9d"))],
                rows=rows, heading_row_color="#1a1a2e",
            )

        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10)
        col.controls.append(ft.Text("CORRELATION DEEP-DIVE", size=22, weight=ft.FontWeight.BOLD, color="#00bfff"))
        col.controls.append(ft.Text("Pairwise correlations — lower = better diversification", size=12, color="#aaaaaa"))
        col.controls.append(ft.Row([
            _kpi("Avg Corr", f"{cd['avg_corr']:.3f}" if cd['avg_corr'] != 0 else "N/A"),
            _kpi("Median Corr", f"{median_corr:.3f}" if median_corr != 0 else "N/A", "#ffd700"),
            _kpi("Diversification", div_quality, div_color),
        ], spacing=6))
        col.controls.append(ft.Row([
            _kpi("Most Correlated", f"{most[0]}/{most[1]}: {most[2]:.2f}" if most[0] != "N/A" else "N/A", "#ff3366"),
            _kpi("Least Correlated", f"{least[0]}/{least[1]}: {least[2]:.2f}" if least[0] != "N/A" else "N/A", "#00ff9d"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("CORRELATION DISTRIBUTION", size=15, weight=ft.FontWeight.BOLD, color="#9966ff"))
        col.controls.append(ft.Row([
            _kpi_sm("Total Pairs", str(n_pairs), "#00bfff"),
            _kpi_sm("High (>0.7)", str(high_corr_pairs), "#ff3366"),
            _kpi_sm("Moderate", str(moderate_pairs), "#ffd700"),
            _kpi_sm("Low (<0.3)", str(low_corr_pairs), "#00ff9d"),
            _kpi_sm("Negative", str(neg_corr_pairs), "#00bfff"),
            _kpi_sm("Corr Std Dev", f"{std_corr:.3f}", "#aaaaaa"),
        ], spacing=4))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("ALL PAIRWISE CORRELATIONS", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(tbl)
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(_chart_image_expandable(data.chart_correlation_matrix()))
        return _card(col)

    def health_score_tab():
        hs = data.portfolio_health_score()
        d = hs['details']
        score = hs['score']
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(value, size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        def _bar(label, pts, max_pts, raw_val, color="#00ff9d"):
            pct = pts / max_pts if max_pts > 0 else 0
            pct_score = pct * 100
            score_color = "#00ff9d" if pct >= 0.7 else "#ffd700" if pct >= 0.4 else "#ff3366"
            return ft.Row([
                ft.Text(label, size=12, color="#aaaaaa", width=110),
                ft.ProgressBar(value=pct, color=color, bgcolor="#1a1a2e", expand=True),
                ft.Text(f"{pts}/{max_pts}", size=11, color="#ffffff", width=45),
                ft.Text(f"({pct_score:.0f}%)", size=10, color=score_color, width=40),
                ft.Text(raw_val, size=10, color="#888888", width=70),
            ], spacing=6)
        # Grade interpretation
        if score >= 80:
            grade_desc = "Excellent — portfolio is well-optimized across all dimensions"
            grade_color = "#00ff9d"
        elif score >= 65:
            grade_desc = "Good — solid fundamentals with room for minor improvements"
            grade_color = "#00ff9d"
        elif score >= 50:
            grade_desc = "Fair — some risk factors need attention"
            grade_color = "#ffd700"
        elif score >= 35:
            grade_desc = "Below Average — significant weaknesses present"
            grade_color = "#ff9900"
        else:
            grade_desc = "Poor — portfolio needs substantial restructuring"
            grade_color = "#ff3366"
        # Find weakest components for improvement tips
        components = [
            ("Sharpe Ratio", d['sharpe_pts'], 20, "Improve risk-adjusted returns"),
            ("Sortino Ratio", d['sortino_pts'], 15, "Reduce downside volatility"),
            ("Low Volatility", d['vol_pts'], 15, "Diversify to reduce overall vol"),
            ("Low Drawdown", d['dd_pts'], 15, "Add hedging or reduce position sizes"),
            ("Profit Factor", d['pf_pts'], 15, "Cut losers faster, let winners run"),
            ("Diversification", d['div_pts'], 10, "Add uncorrelated assets"),
            ("Win Streak", d['streak_pts'], 10, "Improve trade timing/selection"),
        ]
        weakest = sorted(components, key=lambda x: x[1] / x[2] if x[2] > 0 else 0)[:3]

        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=8)
        col.controls.append(ft.Text("PORTFOLIO HEALTH SCORE", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Text("Composite score from 7 weighted components (100 max)", size=12, color="#aaaaaa"))
        col.controls.append(ft.Row([
            _chart_image(data.chart_health_gauge(score)),
            ft.Column([
                ft.Text(f"Grade: {hs['grade']}", size=28, weight=ft.FontWeight.BOLD, color=grade_color),
                ft.Text(f"Score: {score}/100", size=18, color=grade_color),
                ft.Text(grade_desc, size=11, color="#aaaaaa", italic=True),
                ft.Divider(height=1, color="#222222"),
                ft.Text(f"Sharpe: {d['sharpe']:.2f}  |  Sortino: {d['sortino']:.2f}", size=12, color="#aaaaaa"),
                ft.Text(f"Vol: {d['vol']:.1%}  |  Max DD: {d['max_dd']:.1%}  |  PF: {d['profit_factor']:.2f}", size=12, color="#aaaaaa"),
            ], spacing=4, expand=True),
        ], spacing=10))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("UNDERLYING METRICS", size=14, weight=ft.FontWeight.BOLD, color="#00bfff"))
        col.controls.append(ft.Row([
            _kpi("Sharpe", f"{d['sharpe']:.2f}", "#00ff9d" if d['sharpe'] > 1 else "#ffd700" if d['sharpe'] > 0 else "#ff3366"),
            _kpi("Sortino", f"{d['sortino']:.2f}", "#00ff9d" if d['sortino'] > 1 else "#ffd700" if d['sortino'] > 0 else "#ff3366"),
            _kpi("Ann. Vol", f"{d['vol']:.1%}", "#00ff9d" if d['vol'] < 0.15 else "#ffd700" if d['vol'] < 0.25 else "#ff3366"),
            _kpi("Max DD", f"{d['max_dd']:.1%}", "#00ff9d" if d['max_dd'] > -0.1 else "#ffd700" if d['max_dd'] > -0.2 else "#ff3366"),
            _kpi("Profit Factor", f"{d['profit_factor']:.2f}", "#00ff9d" if d['profit_factor'] > 1.5 else "#ffd700" if d['profit_factor'] > 1 else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("SCORE BREAKDOWN (with raw values)", size=14, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(_bar("Sharpe", d['sharpe_pts'], 20, f"({d['sharpe']:.2f})", "#00ff9d"))
        col.controls.append(_bar("Sortino", d['sortino_pts'], 15, f"({d['sortino']:.2f})", "#00ff9d"))
        col.controls.append(_bar("Low Vol", d['vol_pts'], 15, f"({d['vol']:.1%})", "#00bfff"))
        col.controls.append(_bar("Low DD", d['dd_pts'], 15, f"({d['max_dd']:.1%})", "#ffd700"))
        col.controls.append(_bar("Profit Factor", d['pf_pts'], 15, f"({d['profit_factor']:.2f})", "#ff9900"))
        col.controls.append(_bar("Diversification", d['div_pts'], 10, "", "#9966ff"))
        col.controls.append(_bar("Win Streak", d['streak_pts'], 10, "", "#00ff9d"))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("TOP 3 IMPROVEMENT AREAS", size=14, weight=ft.FontWeight.BOLD, color="#ff9900"))
        for i, (name, pts, max_pts, tip) in enumerate(weakest):
            pct = pts / max_pts * 100 if max_pts > 0 else 0
            col.controls.append(ft.Container(
                content=ft.Row([
                    ft.Text(f"{i+1}.", size=14, weight=ft.FontWeight.BOLD, color="#ff9900", width=20),
                    ft.Text(f"{name} ({pct:.0f}%)", size=12, color="#ffffff", width=140),
                    ft.Text(tip, size=11, color="#aaaaaa"),
                ], spacing=6),
                bgcolor="#1a1a2e", padding=8, border_radius=8,
            ))
        return _card(col)

    def risk_returns_tab():
        ra = data.risk_adjusted_returns()
        rs = data.returns_series
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(value, size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        def _kpi_sm(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(value, size=13, weight=ft.FontWeight.BOLD, color=color)], spacing=1), bgcolor="#1a1a2e", padding=6, border_radius=8, expand=True)
        def _fmt(v, fmt=".2f"):
            if v == 0: return "N/A"
            return f"{v:{fmt}}"
        # Extra computed risk metrics
        tail_ratio_val = 0
        downside_dev = 0
        avg_win = 0
        avg_loss = 0
        payoff = 0
        if not rs.empty and len(rs) > 10:
            p95 = float(np.percentile(rs, 95))
            p5 = float(np.percentile(rs, 5))
            tail_ratio_val = abs(p95 / p5) if p5 != 0 else 0
            downside_dev = float(rs[rs < 0].std() * np.sqrt(252)) if (rs < 0).any() else 0
            avg_win = float(rs[rs > 0].mean()) if (rs > 0).any() else 0
            avg_loss = float(rs[rs < 0].mean()) if (rs < 0).any() else 0
            payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=8)
        col.controls.append(ft.Text("RISK-ADJUSTED RETURNS", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(ft.Text("Complete risk-return profile including CAPM, tail risk, and trading edge metrics", size=12, color="#aaaaaa"))
        col.controls.append(ft.Text("RETURN & VOLATILITY", size=14, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Row([
            _kpi("Ann. Return", f"{ra['ann_return']:.1%}" if ra['ann_return'] != 0 else "N/A", "#00ff9d" if ra['ann_return'] > 0 else "#ff3366"),
            _kpi("Ann. Volatility", f"{ra['ann_vol']:.1%}" if ra['ann_vol'] != 0 else "N/A", "#ff3366"),
            _kpi("Downside Dev", f"{downside_dev:.1%}" if downside_dev else "N/A", "#ff9900"),
            _kpi("Max DD", f"{ra['max_dd']:.1%}" if ra['max_dd'] != 0 else "N/A", "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#222222"))
        col.controls.append(ft.Text("RISK-ADJUSTED RATIOS", size=14, weight=ft.FontWeight.BOLD, color="#00bfff"))
        col.controls.append(ft.Row([
            _kpi("Sharpe", _fmt(ra['sharpe']), "#00ff9d" if ra['sharpe'] > 0 else "#ff3366"),
            _kpi("Sortino", _fmt(ra['sortino']), "#00ff9d" if ra['sortino'] > 0 else "#ff3366"),
            _kpi("Calmar", _fmt(ra['calmar']), "#00ff9d" if ra['calmar'] > 0 else "#ff3366"),
            _kpi("Profit Factor", _fmt(ra['profit_factor']), "#00ff9d" if ra['profit_factor'] > 1 else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#222222"))
        col.controls.append(ft.Text("TAIL RISK", size=14, weight=ft.FontWeight.BOLD, color="#ff3366"))
        col.controls.append(ft.Row([
            _kpi("VaR 95%", f"{ra['var_95']:.2%}" if ra['var_95'] != 0 else "N/A", "#ff3366"),
            _kpi("Tail Ratio", f"{tail_ratio_val:.2f}" if tail_ratio_val else "N/A", "#00ff9d" if tail_ratio_val > 1 else "#ff3366"),
            _kpi("Avg Win", f"{avg_win:.2%}" if avg_win else "N/A", "#00ff9d"),
            _kpi("Avg Loss", f"{avg_loss:.2%}" if avg_loss else "N/A", "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Row([
            _kpi_sm("Payoff Ratio", f"{payoff:.2f}" if payoff else "N/A", "#00ff9d" if payoff > 1 else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#222222"))
        col.controls.append(ft.Text("CAPM METRICS", size=14, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(ft.Row([
            _kpi("Beta (vs SPY)", _fmt(ra['beta']), "#ffd700"),
            _kpi("Alpha (CAPM)", f"{ra['alpha']:.2%}" if ra['alpha'] != 0 else "N/A", "#00ff9d" if ra['alpha'] > 0 else "#ff3366"),
            _kpi("Treynor", _fmt(ra['treynor']), "#00ff9d" if ra['treynor'] > 0 else "#ff3366"),
            _kpi("Info Ratio", _fmt(ra['info_ratio']), "#00ff9d" if ra['info_ratio'] > 0 else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Text("Beta = market sensitivity. Alpha = excess return vs CAPM. Treynor = return per unit systematic risk. Tail Ratio > 1 = fatter right tail (good).", size=10, color="#666666", italic=True))
        return _card(col)

    def streaks_tab():
        wl = data.win_loss_streaks()
        rec = data.recovery_time()
        s = data.portfolio_summary_stats()
        pf_val = data.profit_factor()
        ra = data.risk_adjusted_returns()
        rs = data.returns_series
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=18, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=12, border_radius=10, expand=True)
        def _kpi_sm(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(str(value), size=14, weight=ft.FontWeight.BOLD, color=color)], spacing=1), bgcolor="#1a1a2e", padding=8, border_radius=8, expand=True)
        pos = s.get("positive_days", 0)
        neg = s.get("negative_days", 0)
        skew = s.get("skewness", 0)
        kurt = s.get("kurtosis", 0)
        vol = s.get("ann_volatility", 0)
        max_dd = s.get("max_drawdown", 0)
        var95 = s.get("var_95", 0)
        cvar95 = s.get("cvar_95", 0)
        # Compute extra stats from returns series
        avg_win = float(rs[rs > 0].mean()) if not rs.empty and (rs > 0).any() else 0
        avg_loss = float(rs[rs < 0].mean()) if not rs.empty and (rs < 0).any() else 0
        max_gain = float(rs.max()) if not rs.empty else 0
        max_loss = float(rs.min()) if not rs.empty else 0
        avg_daily = float(rs.mean()) if not rs.empty else 0
        std_daily = float(rs.std()) if not rs.empty else 0
        total_days = len(rs)
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        expectancy = (pos / max(pos + neg, 1)) * avg_win + (neg / max(pos + neg, 1)) * avg_loss if (pos + neg) > 0 else 0
        # Risk of ruin approximation
        wr = pos / max(pos + neg, 1) if (pos + neg) > 0 else 0
        risk_of_ruin = ((1 - wr) / wr) ** 10 if wr > 0 and wr < 1 else (1.0 if wr == 0 else 0.0)

        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=8)
        col.controls.append(ft.Text("WIN/LOSS STREAKS & STATS", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Text("Complete trading day performance, streak analysis, and risk metrics", size=12, color="#aaaaaa"))

        col.controls.append(ft.Text("STREAK ANALYSIS", size=15, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Row([
            _kpi("Longest Win", f"{wl['longest_win']}d" if wl['longest_win'] > 0 else "N/A", "#00ff9d"),
            _kpi("Longest Loss", f"{wl['longest_loss']}d" if wl['longest_loss'] > 0 else "N/A", "#ff3366"),
            _kpi("Current", f"{wl['current_streak']}d ({wl['current_type']})" if wl['current_type'] != "N/A" else "N/A", "#00ff9d" if wl['current_type'] == "win" else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Row([
            _kpi("Positive Days", str(pos) if pos > 0 else "N/A", "#00ff9d"),
            _kpi("Negative Days", str(neg) if neg > 0 else "N/A", "#ff3366"),
            _kpi("Win Rate", f"{pos/(pos+neg):.1%}" if (pos + neg) > 0 else "N/A", "#00ff9d" if pos > neg else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#333333"))

        col.controls.append(ft.Text("TRADE STATISTICS", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(ft.Row([
            _kpi("Avg Win", f"{avg_win:.2%}" if avg_win else "N/A", "#00ff9d"),
            _kpi("Avg Loss", f"{avg_loss:.2%}" if avg_loss else "N/A", "#ff3366"),
            _kpi("Payoff Ratio", f"{payoff:.2f}" if payoff else "N/A", "#00ff9d" if payoff > 1 else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Row([
            _kpi("Best Day", f"{max_gain:.2%}" if max_gain else "N/A", "#00ff9d"),
            _kpi("Worst Day", f"{max_loss:.2%}" if max_loss else "N/A", "#ff3366"),
            _kpi("Profit Factor", f"{pf_val:.2f}" if pf_val else "N/A", "#00ff9d" if pf_val > 1 else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#333333"))

        col.controls.append(ft.Text("EXPECTANCY & RISK", size=15, weight=ft.FontWeight.BOLD, color="#ff9900"))
        col.controls.append(ft.Row([
            _kpi("Expectancy", f"{expectancy:.4f}" if expectancy else "N/A", "#00ff9d" if expectancy > 0 else "#ff3366"),
            _kpi("Risk of Ruin", f"{risk_of_ruin:.2%}", "#ff3366" if risk_of_ruin > 0.1 else "#00ff9d"),
            _kpi("Avg Daily", f"{avg_daily:.4f}" if avg_daily else "N/A", "#00ff9d" if avg_daily > 0 else "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Row([
            _kpi("Daily Std Dev", f"{std_daily:.4f}" if std_daily else "N/A", "#ffd700"),
            _kpi("VaR 95%", f"{var95:.2%}" if var95 else "N/A", "#ff3366"),
            _kpi("CVaR 95%", f"{cvar95:.2%}" if cvar95 else "N/A", "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Row([
            _kpi_sm("Total Trading Days", str(total_days) if total_days else "N/A", "#00bfff"),
            _kpi_sm("Ann. Volatility", f"{vol:.1%}" if vol else "N/A", "#ff3366"),
            _kpi_sm("Max Drawdown", f"{max_dd:.1%}" if max_dd else "N/A", "#ff3366"),
            _kpi_sm("Beta (SPY)", f"{ra.get('beta', 0):.2f}" if ra.get('beta') else "N/A", "#ffd700"),
        ], spacing=4))
        col.controls.append(ft.Divider(height=1, color="#333333"))

        col.controls.append(ft.Text("DRAWDOWN RECOVERY", size=15, weight=ft.FontWeight.BOLD, color="#ff3366"))
        col.controls.append(ft.Row([
            _kpi("Max Recovery", f"{rec['max_recovery_days']}d" if rec['max_recovery_days'] > 0 else "N/A"),
            _kpi("Avg Recovery", f"{rec.get('avg_recovery_days', 0):.0f}d" if rec.get('avg_recovery_days', 0) > 0 else "N/A", "#ffd700"),
            _kpi("In Drawdown?", "YES" if rec['currently_in_drawdown'] else "NO", "#ff3366" if rec['currently_in_drawdown'] else "#00ff9d"),
            _kpi("Current DD Days", f"{rec['current_dd_days']}d" if rec['current_dd_days'] > 0 else "N/A", "#ff3366"),
        ], spacing=6))
        col.controls.append(ft.Divider(height=1, color="#333333"))

        col.controls.append(ft.Text("DISTRIBUTION", size=15, weight=ft.FontWeight.BOLD, color="#9966ff"))
        col.controls.append(ft.Row([
            _kpi("Skewness", f"{skew:.3f}" if skew != 0 else "N/A", "#00ff9d" if skew > 0 else "#ff3366"),
            _kpi("Kurtosis", f"{kurt:.3f}" if kurt != 0 else "N/A", "#ffd700"),
        ], spacing=6))
        col.controls.append(ft.Text("Negative skew = more left-tail risk. High kurtosis = fat tails. Payoff > 1 = wins bigger than losses. Expectancy > 0 = positive edge.", size=10, color="#666666", italic=True))
        return _card(col)

    # ====================== PROFIT CHARTS TAB ======================
    def profit_charts_tab():
        pnl_state = {'mode': '$', 'start': 0, 'end': len(data.returns_series) if not data.returns_series.empty else 0}
        max_points = pnl_state['end']
        rs = data.returns_series

        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(value, size=14, weight=ft.FontWeight.BOLD, color=color)], spacing=1), bgcolor="#1a1a2e", padding=8, border_radius=8, expand=True)

        # Compute P&L summary stats
        pnl_kpis = []
        if not rs.empty and len(rs) > 1:
            total_val = data.portfolio_summary_stats().get('total_value', 0)
            cost_basis = data.portfolio_summary_stats().get('total_cost', 0)
            total_pnl = total_val - cost_basis if cost_basis else 0
            avg_daily = float(rs.mean())
            std_daily = float(rs.std())
            best_day = float(rs.max())
            worst_day = float(rs.min())
            pos_days = int((rs > 0).sum())
            neg_days = int((rs < 0).sum())
            total_days = len(rs)
            win_pct = pos_days / total_days * 100 if total_days else 0
            # Weekly returns
            try:
                weekly_rs = rs.resample('W').apply(lambda x: (1 + x).prod() - 1)
                avg_weekly = float(weekly_rs.mean()) if len(weekly_rs) > 0 else 0
                best_week = float(weekly_rs.max()) if len(weekly_rs) > 0 else 0
                worst_week = float(weekly_rs.min()) if len(weekly_rs) > 0 else 0
                n_weeks = len(weekly_rs)
            except Exception:
                avg_weekly = best_week = worst_week = 0
                n_weeks = 0

            pnl_kpis = [
                ft.Text("P&L SUMMARY", size=14, weight=ft.FontWeight.BOLD, color="#00bfff"),
                ft.Row([
                    _kpi("Total P&L", f"${total_pnl:,.0f}" if total_pnl else "N/A", "#00ff9d" if total_pnl >= 0 else "#ff3366"),
                    _kpi("Avg Daily", f"{avg_daily:.3%}", "#00ff9d" if avg_daily >= 0 else "#ff3366"),
                    _kpi("Daily Std Dev", f"{std_daily:.3%}", "#ff9900"),
                    _kpi("Best Day", f"{best_day:.2%}", "#00ff9d"),
                    _kpi("Worst Day", f"{worst_day:.2%}", "#ff3366"),
                ], spacing=4),
                ft.Row([
                    _kpi("Win Days", str(pos_days), "#00ff9d"),
                    _kpi("Loss Days", str(neg_days), "#ff3366"),
                    _kpi("Win %", f"{win_pct:.1f}%", "#00ff9d" if win_pct >= 50 else "#ff3366"),
                    _kpi("Avg Weekly", f"{avg_weekly:.2%}" if n_weeks else "N/A", "#00ff9d" if avg_weekly >= 0 else "#ff3366"),
                    _kpi("Best Week", f"{best_week:.2%}" if n_weeks else "N/A", "#00ff9d"),
                    _kpi("Worst Week", f"{worst_week:.2%}" if n_weeks else "N/A", "#ff3366"),
                ], spacing=4),
                ft.Divider(height=1, color="#333333"),
            ]

        pnl_state['cumulative'] = False
        pnl_state['style'] = 'bar'
        pnl_state['avg'] = False

        daily_chart_container = ft.Container(
            content=_chart_image(data.chart_daily_profit(mode='$')),
            padding=0,
        )

        def _rebuild_pnl():
            daily_chart_container.content = _chart_image(
                data.chart_daily_profit(mode=pnl_state['mode'], start_idx=pnl_state['start'],
                                         end_idx=pnl_state['end'], cumulative=pnl_state['cumulative'],
                                         chart_style=pnl_state['style'], show_avg=pnl_state['avg'])
            )
            page.update()

        def on_mode_change(e):
            pnl_state['mode'] = e.control.value
            _rebuild_pnl()

        def on_range_change(e):
            pnl_state['start'] = int(e.control.start_value)
            pnl_state['end'] = int(e.control.end_value)
            _rebuild_pnl()

        def _pnl_tog(key):
            def handler(e):
                pnl_state[key] = e.control.value
                _rebuild_pnl()
            return handler

        def _pnl_style(e):
            pnl_state['style'] = e.control.value
            _rebuild_pnl()

        mode_dropdown = ft.Dropdown(
            value="$", width=100,
            options=[ft.dropdown.Option("$", "Dollar ($)"), ft.dropdown.Option("%", "Percent (%)")],
            on_select=on_mode_change,
            border_color="#00ff9d", color="#ffffff", bgcolor="#1a1a2e",
        )

        range_slider = ft.RangeSlider(
            min=0, max=max(max_points, 1), start_value=0, end_value=max(max_points, 1),
            divisions=max(max_points, 1),
            active_color="#00ff9d", inactive_color="#333333",
            on_change=on_range_change,
        ) if max_points > 0 else ft.Text("No data for range slider", size=12, color="#888888")

        pnl_toggles = ft.Row([
            ft.Text("Style:", size=11, color="#aaaaaa"),
            ft.Dropdown(value="bar", width=90, options=[
                ft.dropdown.Option("bar", "Bar"), ft.dropdown.Option("line", "Line"),
                ft.dropdown.Option("area", "Area"),
            ], on_select=_pnl_style, border_color="#00ff9d", color="#ffffff", bgcolor="#1a1a2e"),
            ft.Switch(label="Cumulative", value=False, active_color="#00bfff",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_pnl_tog('cumulative')),
            ft.Switch(label="Show Average", value=False, active_color="#ffd700",
                      label_text_style=ft.TextStyle(size=11, color="#aaaaaa"), on_change=_pnl_tog('avg')),
        ], spacing=8, scroll=ft.ScrollMode.AUTO)

        col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=8)
        col.controls.append(ft.Text("PROFIT / LOSS CHARTS", size=20, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Text("Daily, weekly, and rolling weekly profit/loss with summary statistics", size=12, color="#888888"))
        for kpi in pnl_kpis:
            col.controls.append(kpi)
        col.controls.append(ft.Text("DAILY PROFIT / LOSS", size=15, weight=ft.FontWeight.BOLD, color="#00ff9d"))
        col.controls.append(ft.Row([
            ft.Text("Display:", size=12, color="#aaaaaa"),
            mode_dropdown,
            ft.Text("Date Range:", size=12, color="#aaaaaa"),
        ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER))
        col.controls.append(pnl_toggles)
        col.controls.append(range_slider)
        col.controls.append(daily_chart_container)
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("WEEKLY PROFIT / LOSS", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"))
        col.controls.append(_chart_image(data.chart_weekly_profit()))
        col.controls.append(ft.Divider(height=1, color="#333333"))
        col.controls.append(ft.Text("ROLLING 5-DAY (WEEKLY) PROFIT", size=15, weight=ft.FontWeight.BOLD, color="#00bfff"))
        col.controls.append(_chart_image(data.chart_rolling_weekly_profit()))
        return _card(col)

    # ====================== PROGRESSION TAB (Investment Growth Analysis) ======================
    def progression_tab():
        result_col = ft.Column(spacing=8, scroll=ft.ScrollMode.AUTO, expand=True)

        def _k(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(str(value), size=15, weight=ft.FontWeight.BOLD, color=color)], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER), bgcolor="#1a1a2e", padding=8, border_radius=8, expand=True)

        def _section_title(text, color="#ffd700"):
            return ft.Text(text, size=16, weight=ft.FontWeight.BOLD, color=color)

        def _bar(label, pct, color="#00ff9d", width=300):
            bar_w = max(min(pct, 100), 0) * (width / 100)
            return ft.Row([
                ft.Text(label, size=11, color="#ffffff", width=140),
                ft.Container(width=bar_w, height=14, bgcolor=color, border_radius=4),
                ft.Text(f"{pct:.1f}%", size=11, color=color, width=60),
            ], spacing=6)

        # =================== GATHER ALL DATA ===================
        s = data.portfolio_summary_stats()
        holdings = data.holdings_with_weights()
        hp = data.holding_period_analysis()
        tf = data.trade_frequency_analysis()
        wl = data.win_loss_streaks()
        pf_val = data.profit_factor()
        dow = data.day_of_week_returns()
        season = data.monthly_seasonality()
        bw = data.best_worst_periods()
        rec = data.recovery_time()
        ra = data.risk_adjusted_returns()
        rs = data.returns_series

        total_val = s.get('total_value', 0)
        total_gl = s.get('total_gain_loss', 0)
        total_cb = s.get('total_cost_basis', 0)
        ann_ret = s.get('ann_return', 0)
        ann_vol = s.get('ann_volatility', 0)
        sharpe = s.get('sharpe', 0)
        sortino = s.get('sortino', 0)
        calmar = s.get('calmar', 0)
        max_dd = s.get('max_drawdown', 0)
        win_rate = s.get('win_rate', 0)
        n_hold = s.get('num_holdings', 0)
        total_pct = s.get('total_gain_pct', 0)
        var95 = s.get('var_95', 0)
        cvar95 = s.get('cvar_95', 0)
        beta = ra.get('beta', 0)
        alpha = ra.get('alpha', 0)
        treynor = ra.get('treynor', 0)
        info_ratio = ra.get('info_ratio', 0)
        # Extra computed metrics
        avg_win = float(rs[rs > 0].mean()) if not rs.empty and (rs > 0).any() else 0
        avg_loss = float(rs[rs < 0].mean()) if not rs.empty and (rs < 0).any() else 0
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        expectancy = (s.get('positive_days', 0) / max(s.get('positive_days', 0) + s.get('negative_days', 0), 1)) * avg_win + (s.get('negative_days', 0) / max(s.get('positive_days', 0) + s.get('negative_days', 0), 1)) * avg_loss if (s.get('positive_days', 0) + s.get('negative_days', 0)) > 0 else 0
        tail_ratio_val = 0
        downside_dev = 0
        if not rs.empty and len(rs) > 10:
            p95 = float(np.percentile(rs, 95))
            p5 = float(np.percentile(rs, 5))
            tail_ratio_val = abs(p95 / p5) if p5 != 0 else 0
            downside_dev = float(rs[rs < 0].std() * np.sqrt(252)) if (rs < 0).any() else 0

        # =================== 1. JOURNEY OVERVIEW ===================
        result_col.controls.append(_section_title("YOUR INVESTMENT JOURNEY", "#00ff9d"))
        journey_days = hp.get('days', 0)
        journey_years = hp.get('years', 0)
        first_date = hp.get('first', 'N/A')
        last_date = hp.get('last', 'N/A')
        total_trades = tf.get('total', 0)

        gl_color = "#00ff9d" if total_gl >= 0 else "#ff3366"
        result_col.controls.append(ft.Row([
            _k("Journey Start", first_date, "#00bfff"),
            _k("Latest Activity", last_date, "#00bfff"),
            _k("Duration", f"{journey_years} yrs ({journey_days:,}d)", "#ffd700"),
            _k("Total Trades", f"{total_trades:,}", "#9966ff"),
        ], spacing=4))
        result_col.controls.append(ft.Row([
            _k("Total Invested", f"${total_cb:,.0f}", "#aaaaaa"),
            _k("Current Value", f"${total_val:,.0f}", "#00ff9d"),
            _k("Total Gain/Loss", f"${total_gl:,.0f}", gl_color),
            _k("Total Return", f"{total_pct:.1f}%", gl_color),
        ], spacing=4))
        result_col.controls.append(ft.Row([
            _k("Holdings", str(n_hold), "#ffd700"),
            _k("Ann. Return", f"{ann_ret:.1%}", "#00ff9d" if ann_ret > 0 else "#ff3366"),
            _k("Volatility", f"{ann_vol:.1%}", "#ff9900"),
            _k("Sharpe Ratio", f"{sharpe:.2f}", "#00ff9d" if sharpe > 1 else "#ffd700" if sharpe > 0 else "#ff3366"),
        ], spacing=4))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 1b. COMPREHENSIVE RISK METRICS ===================
        result_col.controls.append(_section_title("COMPREHENSIVE RISK METRICS", "#ff3366"))
        result_col.controls.append(ft.Row([
            _k("Sortino", f"{sortino:.2f}", "#00ff9d" if sortino > 0 else "#ff3366"),
            _k("Calmar", f"{calmar:.2f}", "#00ff9d" if calmar > 0 else "#ff3366"),
            _k("Max DD", f"{max_dd:.1%}", "#ff3366"),
            _k("VaR 95%", f"{var95:.2%}" if var95 else "N/A", "#ff3366"),
            _k("CVaR 95%", f"{cvar95:.2%}" if cvar95 else "N/A", "#ff3366"),
        ], spacing=4))
        result_col.controls.append(ft.Row([
            _k("Beta (SPY)", f"{beta:.2f}" if beta else "N/A", "#ffd700"),
            _k("Alpha (CAPM)", f"{alpha:.2%}" if alpha else "N/A", "#00ff9d" if alpha > 0 else "#ff3366"),
            _k("Treynor", f"{treynor:.2f}" if treynor else "N/A", "#00ff9d" if treynor > 0 else "#ff3366"),
            _k("Info Ratio", f"{info_ratio:.2f}" if info_ratio else "N/A", "#00ff9d" if info_ratio > 0 else "#ff3366"),
            _k("Tail Ratio", f"{tail_ratio_val:.2f}" if tail_ratio_val else "N/A", "#00ff9d" if tail_ratio_val > 1 else "#ff3366"),
        ], spacing=4))
        result_col.controls.append(ft.Row([
            _k("Profit Factor", f"{pf_val:.2f}" if pf_val else "N/A", "#00ff9d" if isinstance(pf_val, (int, float)) and pf_val > 1 else "#ff3366"),
            _k("Win Rate", f"{win_rate:.0%}", "#00ff9d" if win_rate > 0.5 else "#ff3366"),
            _k("Payoff Ratio", f"{payoff:.2f}" if payoff else "N/A", "#00ff9d" if payoff > 1 else "#ff3366"),
            _k("Expectancy", f"{expectancy:.4f}" if expectancy else "N/A", "#00ff9d" if expectancy > 0 else "#ff3366"),
            _k("Downside Dev", f"{downside_dev:.1%}" if downside_dev else "N/A", "#ff9900"),
        ], spacing=4))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 2. EQUITY GROWTH CHART ===================
        result_col.controls.append(_section_title("PORTFOLIO EQUITY CURVE", "#00ff9d"))
        eq_chart = data.chart_equity_curve()
        result_col.controls.append(_chart_image(eq_chart))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 3. DRAWDOWN CHART ===================
        result_col.controls.append(_section_title("DRAWDOWN HISTORY", "#ff3366"))
        dd_chart = data.chart_drawdown()
        result_col.controls.append(_chart_image(dd_chart))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 4. MILESTONE TRACKER ===================
        result_col.controls.append(_section_title("WEALTH MILESTONES", "#ffd700"))
        milestones = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 5000000, 10000000]
        for m in milestones:
            if m > total_val * 5 and m > total_cb * 5:
                continue  # skip unreachable milestones
            reached = total_val >= m
            if reached:
                icon = ft.Icon(ft.Icons.CHECK_CIRCLE, color="#00ff9d", size=18)
                label_color = "#00ff9d"
                status = "ACHIEVED"
            elif total_val >= m * 0.5:
                icon = ft.Icon(ft.Icons.TIMELAPSE, color="#ffd700", size=18)
                label_color = "#ffd700"
                pct_to = total_val / m * 100
                status = f"{pct_to:.0f}% there"
            else:
                icon = ft.Icon(ft.Icons.RADIO_BUTTON_UNCHECKED, color="#444444", size=18)
                label_color = "#666666"
                pct_to = total_val / m * 100
                status = f"{pct_to:.1f}%"
            result_col.controls.append(ft.Row([
                icon,
                ft.Text(f"${m:,.0f}", size=13, color=label_color, width=120, weight=ft.FontWeight.BOLD),
                ft.Container(
                    width=max(min(total_val / m, 1.0), 0) * 200,
                    height=10, bgcolor=label_color, border_radius=4,
                ),
                ft.Text(status, size=11, color=label_color),
            ], spacing=8))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 5. PER-ASSET GROWTH BREAKDOWN ===================
        result_col.controls.append(_section_title("ASSET-BY-ASSET GROWTH", "#00bfff"))
        if holdings:
            # Header
            result_col.controls.append(ft.Row([
                ft.Text("Symbol", size=10, color="#aaaaaa", width=70),
                ft.Text("Value", size=10, color="#aaaaaa", width=100),
                ft.Text("Cost", size=10, color="#aaaaaa", width=100),
                ft.Text("Gain/Loss", size=10, color="#aaaaaa", width=100),
                ft.Text("Return%", size=10, color="#aaaaaa", width=70),
                ft.Text("Weight", size=10, color="#aaaaaa", width=60),
                ft.Text("Growth Bar", size=10, color="#aaaaaa", expand=True),
            ], spacing=4))
            for h in holdings[:30]:  # top 30
                sym = h['symbol']
                mv = h['market_value']
                cb = h['cost_basis']
                gl = h['gain_loss']
                gl_pct = h['gain_loss_pct']
                wt = h['weight']
                gc = "#00ff9d" if gl >= 0 else "#ff3366"
                bar_w = max(min(abs(gl_pct) * 1.5, 200), 3)
                result_col.controls.append(ft.Row([
                    ft.Text(sym, size=12, color="#ffffff", width=70, weight=ft.FontWeight.BOLD),
                    ft.Text(f"${mv:,.0f}", size=11, color="#ffffff", width=100),
                    ft.Text(f"${cb:,.0f}", size=11, color="#aaaaaa", width=100),
                    ft.Text(f"${gl:,.0f}", size=11, color=gc, width=100),
                    ft.Text(f"{gl_pct:.1f}%", size=11, color=gc, width=70),
                    ft.Text(f"{wt:.1f}%", size=11, color="#ffd700", width=60),
                    ft.Container(width=bar_w, height=10, bgcolor=gc, border_radius=3),
                ], spacing=4))
        else:
            result_col.controls.append(ft.Text("Import data to see per-asset growth.", size=13, color="#888888"))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 6. MONTHLY RETURNS HEATMAP CHART ===================
        result_col.controls.append(_section_title("MONTHLY RETURNS HEATMAP", "#ffd700"))
        hm_chart = data.chart_monthly_heatmap()
        result_col.controls.append(_chart_image(hm_chart))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 7. INVESTMENT SKILL METRICS ===================
        result_col.controls.append(_section_title("INVESTOR SKILL SCORECARD", "#9966ff"))

        # Compute scores
        win_score = min(win_rate * 100, 100)
        sharpe_score = min(max(sharpe * 25, 0), 100)
        sortino_score = min(max(sortino * 20, 0), 100)
        dd_score = max(100 - abs(max_dd) * 200, 0)
        pf_score = min(max(pf_val * 20, 0), 100) if isinstance(pf_val, (int, float)) else 50
        consistency = min(max((1 - ann_vol / max(ann_ret, 0.01)) * 50, 0), 100) if ann_ret > 0 else 0
        div_score = min(n_hold * 8, 100)

        overall = (win_score * 0.15 + sharpe_score * 0.2 + sortino_score * 0.15 + dd_score * 0.15 + pf_score * 0.1 + consistency * 0.15 + div_score * 0.1)

        def _grade(score):
            if score >= 90: return "A+", "#00ff9d"
            elif score >= 80: return "A", "#00ff9d"
            elif score >= 70: return "B+", "#ffd700"
            elif score >= 60: return "B", "#ffd700"
            elif score >= 50: return "C", "#ff9900"
            elif score >= 40: return "D", "#ff3366"
            else: return "F", "#ff3366"

        grade, grade_color = _grade(overall)

        result_col.controls.append(ft.Row([
            _k("Overall Score", f"{overall:.0f}/100", grade_color),
            _k("Grade", grade, grade_color),
            _k("Win Rate", f"{win_rate:.0%}", "#00ff9d" if win_rate > 0.5 else "#ff3366"),
            _k("Max Drawdown", f"{max_dd:.1%}", "#ff3366"),
        ], spacing=4))

        skill_items = [
            ("Win Rate", win_score, "#00ff9d"),
            ("Risk-Adjusted (Sharpe)", sharpe_score, "#00bfff"),
            ("Downside Mgmt (Sortino)", sortino_score, "#9966ff"),
            ("Drawdown Control", dd_score, "#ff9900"),
            ("Profit Factor", pf_score, "#ffd700"),
            ("Consistency", consistency, "#00ff9d"),
            ("Diversification", div_score, "#00bfff"),
        ]
        for label, score, color in skill_items:
            result_col.controls.append(_bar(label, score, color))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 8. DAY-OF-WEEK PERFORMANCE ===================
        result_col.controls.append(_section_title("DAY-OF-WEEK PERFORMANCE", "#00bfff"))
        if dow:
            for day_name, ret in dow.items():
                dc = "#00ff9d" if ret > 0 else "#ff3366"
                bar_w = max(abs(ret) * 300, 3)
                result_col.controls.append(ft.Row([
                    ft.Text(day_name, size=12, color="#ffffff", width=60),
                    ft.Container(width=bar_w, height=12, bgcolor=dc, border_radius=3),
                    ft.Text(f"{ret:.1%}", size=11, color=dc),
                ], spacing=8))
        else:
            result_col.controls.append(ft.Text("No returns data available.", size=12, color="#888888"))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 9. MONTHLY SEASONALITY ===================
        result_col.controls.append(_section_title("MONTHLY SEASONALITY", "#ffd700"))
        if season:
            for mo_name, ret in season.items():
                mc = "#00ff9d" if ret > 0 else "#ff3366"
                bar_w = max(abs(ret) * 200, 3)
                result_col.controls.append(ft.Row([
                    ft.Text(mo_name, size=12, color="#ffffff", width=40),
                    ft.Container(width=bar_w, height=12, bgcolor=mc, border_radius=3),
                    ft.Text(f"{ret:.1%}", size=11, color=mc),
                ], spacing=8))
        else:
            result_col.controls.append(ft.Text("No returns data available.", size=12, color="#888888"))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 10. BEST & WORST PERIODS ===================
        result_col.controls.append(_section_title("BEST & WORST MONTHS", "#ff9900"))
        best = bw.get('best', {})
        worst = bw.get('worst', {})
        if best:
            bp = best.get('period', 'N/A')
            br = best.get('return', 0)
            result_col.controls.append(ft.Row([
                ft.Icon(ft.Icons.TRENDING_UP, color="#00ff9d", size=20),
                ft.Text(f"Best: {bp}", size=13, color="#00ff9d", weight=ft.FontWeight.BOLD),
                ft.Text(f"{br:.1%}", size=13, color="#00ff9d"),
            ], spacing=8))
        if worst:
            wp = worst.get('period', 'N/A')
            wr = worst.get('return', 0)
            result_col.controls.append(ft.Row([
                ft.Icon(ft.Icons.TRENDING_DOWN, color="#ff3366", size=20),
                ft.Text(f"Worst: {wp}", size=13, color="#ff3366", weight=ft.FontWeight.BOLD),
                ft.Text(f"{wr:.1%}", size=13, color="#ff3366"),
            ], spacing=8))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 11. WIN/LOSS STREAKS & RECOVERY ===================
        result_col.controls.append(_section_title("STREAKS & RECOVERY", "#9966ff"))
        max_wins = wl.get('max_wins', 0) if isinstance(wl, dict) else 0
        max_losses = wl.get('max_losses', 0) if isinstance(wl, dict) else 0
        rec_days = rec.get('avg_recovery_days', 0) if isinstance(rec, dict) else 0
        result_col.controls.append(ft.Row([
            _k("Best Win Streak", f"{max_wins} days", "#00ff9d"),
            _k("Worst Loss Streak", f"{max_losses} days", "#ff3366"),
            _k("Avg Recovery", f"{rec_days:.0f} days", "#ffd700"),
            _k("Sortino Ratio", f"{sortino:.2f}", "#00bfff"),
        ], spacing=4))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 12. ROLLING SHARPE CHART ===================
        result_col.controls.append(_section_title("ROLLING SHARPE RATIO (60-DAY)", "#00bfff"))
        rs_chart = data.chart_rolling_sharpe()
        result_col.controls.append(_chart_image(rs_chart))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 13. ROLLING VOLATILITY CHART ===================
        result_col.controls.append(_section_title("ROLLING VOLATILITY (30-DAY)", "#ff9900"))
        rv_chart = data.chart_rolling_volatility()
        result_col.controls.append(_chart_image(rv_chart))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 14. BENCHMARK COMPARISON ===================
        result_col.controls.append(_section_title("BENCHMARK COMPARISON", "#ffd700"))
        bm_chart = data.chart_benchmark_overlay()
        result_col.controls.append(_chart_image(bm_chart))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # (Projection section removed)

        # =================== 15. PORTFOLIO EVOLUTION TIMELINE ===================
        result_col.controls.append(_section_title("PORTFOLIO EVOLUTION", "#9966ff"))
        if not data.master_df.empty and 'date' in data.master_df.columns and 'symbol' in data.master_df.columns:
            df = data.master_df.sort_values('date')
            unique_syms = df['symbol'].unique()
            # First appearance per symbol
            first_seen = df.groupby('symbol')['date'].min().sort_values()
            result_col.controls.append(ft.Text(f"{len(unique_syms)} unique assets traded across your history", size=12, color="#aaaaaa"))
            for sym, dt in first_seen.items():
                dt_str = str(dt.date()) if pd.notna(dt) else 'N/A'
                is_current = sym in [h['symbol'] for h in holdings]
                sc = "#00ff9d" if is_current else "#666666"
                status = "ACTIVE" if is_current else "CLOSED"
                result_col.controls.append(ft.Row([
                    ft.Text(dt_str, size=11, color="#888888", width=90),
                    ft.Icon(ft.Icons.CIRCLE, color=sc, size=10),
                    ft.Text(sym, size=12, color=sc, width=80, weight=ft.FontWeight.BOLD),
                    ft.Text(status, size=10, color=sc),
                ], spacing=6))
        else:
            result_col.controls.append(ft.Text("Import trade data to see evolution timeline.", size=13, color="#888888"))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 17. INVESTOR PROFILE SUMMARY ===================
        result_col.controls.append(_section_title("INVESTOR PROFILE SUMMARY", "#ffd700"))
        # Determine profile
        if n_hold >= 15 and ann_vol < 0.2:
            profile = "Conservative Diversifier"
            profile_desc = "You spread risk across many assets with low volatility. Steady growth style."
        elif n_hold >= 10 and sharpe > 1.0:
            profile = "Balanced Optimizer"
            profile_desc = "Well-diversified with strong risk-adjusted returns. Efficient portfolio."
        elif n_hold <= 5 and ann_vol > 0.3:
            profile = "Concentrated Risk-Taker"
            profile_desc = "Few high-conviction bets with high volatility. High risk, high reward."
        elif ann_ret > 0.15:
            profile = "Growth Seeker"
            profile_desc = "Focused on high returns. Willing to accept higher risk for growth."
        elif max_dd < -0.3:
            profile = "Resilient Survivor"
            profile_desc = "Weathered significant drawdowns but kept going. Experience builds strength."
        elif win_rate > 0.55:
            profile = "Consistent Winner"
            profile_desc = "More winning days than losing. Your edge compounds over time."
        else:
            profile = "Emerging Investor"
            profile_desc = "Building your track record. Import more data to refine your profile."

        result_col.controls.append(ft.Container(
            content=ft.Column([
                ft.Text(profile, size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
                ft.Text(profile_desc, size=13, color="#cccccc"),
            ], spacing=4),
            bgcolor="#1a1a2e", padding=15, border_radius=12,
        ))

        return _card(ft.Column([
            ft.Text("INVESTMENT PROGRESSION", size=24, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Your complete investment growth story — every metric, milestone, and insight from your portfolio data", size=12, color="#aaaaaa"),
            ft.Divider(height=2, color="#333333"),
            result_col,
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    # ====================== SAFE BUILD HELPER ======================
    def _safe_build(builder):
        try:
            return builder()
        except Exception as ex:
            return _card(ft.Text(f"Error: {ex}", color="#ff3366", size=16))

    # ====================== CATEGORY DEFINITIONS ======================
    NAV_CATEGORIES = [
        {"icon": ft.Icons.DASHBOARD, "label": "Dashboard",
         "builders": lambda: [dashboard_tab, summary_stats_tab, health_score_tab, profit_charts_tab]},
        {"icon": ft.Icons.SHOW_CHART, "label": "Progress",
         "builders": lambda: [progression_tab]},
        {"icon": ft.Icons.FOLDER_OPEN, "label": "Portfolio",
         "builders": lambda: [portfolio_tab, holdings_bar_tab, top_movers_tab, trade_log_tab]},
        {"icon": ft.Icons.ANALYTICS, "label": "Analysis",
         "builders": lambda: [analysis_tab, market_graphs_tab, sector_tab, attribution_tab, returns_dist_tab, monthly_heatmap_tab, benchmark_tab, correlation_dive_tab, risk_returns_tab]},
        {"icon": ft.Icons.SHIELD, "label": "Risk",
         "builders": lambda: [risk_radar_tab, diversification_tab, market_regime_tab, drawdown_tab, rolling_sharpe_tab, rolling_vol_tab, streaks_tab]},
        {"icon": ft.Icons.INFO_OUTLINE, "label": "About",
         "builders": None},
    ]

    # ====================== CONTENT AREA ======================
    content_area = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True, spacing=15)

    def _import_bar():
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    folder_field,
                    ft.Button("SCAN", bgcolor="#1a1a2e", color="#00bfff",
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
                              on_click=scan_folder, tooltip="Scan folder for importable files"),
                    ft.Button("IMPORT SELECTED", bgcolor="#00ff9d", color="#0a0a0a",
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
                              on_click=do_import),
                    ft.Button("REFRESH", bgcolor="#1a1a2e", color="#00ff9d",
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
                              on_click=lambda e: refresh_current(e)),
                ], spacing=8),
                import_status,
                ft.Container(
                    content=file_list_col,
                    bgcolor="#0f0f0f",
                    border=ft.Border.all(1, "#333333"),
                    border_radius=10,
                    padding=8,
                    height=180,
                ),
            ], spacing=6),
            padding=ft.Padding(left=0, top=0, right=0, bottom=10),
        )

    def _loading_placeholder(label="Loading..."):
        return ft.Container(
            content=ft.Column([
                ft.ProgressRing(width=40, height=40, stroke_width=3, color="#00ff9d"),
                ft.Text(label, size=16, color="#888888", italic=True),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
            padding=60, expand=True,
        )

    def load_category(index):
        content_area.controls.clear()
        cat = NAV_CATEGORIES[index]
        # Show loading placeholder immediately
        content_area.controls.append(_loading_placeholder(f"Loading {cat['label']}..."))
        page.update()
        content_area.controls.clear()
        try:
            if cat["label"] == "About":
                content_area.controls.append(build_about_page())
            elif cat["label"] == "Dashboard":
                content_area.controls.append(_import_bar())
                for b in cat["builders"]():
                    content_area.controls.append(_safe_build(b))
            else:
                builders = cat["builders"]()
                grid = ft.ResponsiveRow(spacing=12, run_spacing=12)
                cols = {"xs": 12, "sm": 12, "md": 6, "lg": 6, "xl": 6} if len(builders) > 1 else {"xs": 12}
                for b in builders:
                    grid.controls.append(ft.Container(content=_safe_build(b), col=cols))
                content_area.controls.append(grid)
        except Exception as ex:
            content_area.controls.append(ft.Text(f"Error loading tab: {ex}", color="#ff3366", size=14))
        page.update()

    def refresh_current(e):
        threading.Thread(target=update_live_ticker, daemon=True).start()
        load_category(nav_rail.selected_index)
        show_snack("Refreshed!", "#006644")

    # ====================== NAVIGATION RAIL ======================
    nav_rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=80,
        min_extended_width=180,
        bgcolor="#121212",
        indicator_color="#00ff9d",
        destinations=[
            ft.NavigationRailDestination(icon=c["icon"], label=c["label"])
            for c in NAV_CATEGORIES
        ],
        on_change=lambda e: load_category(e.control.selected_index),
    )

    # ====================== LAYOUT ======================
    load_category(0)

    # Stop server when browser tab is closed
    def on_disconnect(e):
        try:
            data.close()
        except Exception:
            pass
        os._exit(0)
    page.on_disconnect = on_disconnect

    page.add(
        snack,
        header,
        ft.Stack(
            [
                ft.Row(
                    [
                        nav_rail,
                        ft.VerticalDivider(width=1, color="#333333"),
                        ft.Container(
                            content=content_area,
                            expand=True,
                            padding=15,
                            bgcolor="#0f0f0f",
                            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                        ),
                    ],
                    expand=True,
                ),
                main_loading_overlay,
            ],
            expand=True,
        ),
    )
    page.update()

    threading.Thread(target=update_live_ticker, daemon=True).start()


if __name__ == "__main__":
    ft.run(main, view=ft.AppView.WEB_BROWSER, port=8550)