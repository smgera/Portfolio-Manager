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
import os, io, base64, time, json, math
from datetime import datetime, timedelta
import quantstats as qs
import threading
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
        self.optimized_weights = None
        self.total_value = 0.0
        self.risk_profile = "Aggressive"
        self.raw_files = {}
        self.import_summary = []
        self.book_entries = []
        self.budget_items = []

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
        # Filter valid stock symbols
        symbols = [s for s in self.holdings['symbol'].unique().tolist() if s and s not in junk_syms and len(s) <= 10]
        if symbols:
            try:
                dl = yf.download(symbols, period="5y", progress=False, auto_adjust=True)
                lvl0 = dl.columns.get_level_values(0) if isinstance(dl.columns, pd.MultiIndex) else dl.columns
                price_col = 'Close' if 'Close' in lvl0 else ('Adj Close' if 'Adj Close' in lvl0 else None)
                if price_col and isinstance(dl.columns, pd.MultiIndex):
                    self.prices = dl[price_col] if len(symbols) > 1 else dl[[price_col]].droplevel(0, axis=1)
                elif price_col:
                    self.prices = dl[[price_col]].rename(columns={price_col: symbols[0]}) if len(symbols) == 1 else dl
                else:
                    self.prices = dl
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

    def monte_carlo_simulation(self, years=5, simulations=10000):
        if self.prices.empty: return None
        returns = self.prices.pct_change().dropna()
        mu = returns.mean().mean()
        sigma = returns.std().mean()
        dt = 1/252.0
        portfolio_value = self.total_value if self.total_value > 0 else 100000.0
        paths = np.zeros((252 * years, simulations))
        paths[0] = portfolio_value
        for t in range(1, 252 * years):
            paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(simulations))
        return paths

    def optimize_portfolio(self):
        if self.prices.empty or len(self.prices.columns) < 2: return None
        mu = expected_returns.mean_historical_return(self.prices)
        S = risk_models.sample_cov(self.prices)
        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe()
        self.optimized_weights = ef.clean_weights()
        return self.optimized_weights, ef.portfolio_performance()

    def generate_tearsheet(self):
        if self.returns_series.empty: return None
        qs.reports.html(self.returns_series, title="ULTIMATE TEARSHEET v24.0", output="ultra_tearsheet.html")
        return "ultra_tearsheet.html"

    def simulate_goal_probability(self, target=500000, years=10):
        paths = self.monte_carlo_simulation(years=years, simulations=10000)
        if paths is None: return 0.0
        final_values = paths[-1]
        return (final_values >= target).mean() * 100

    def generate_smart_brain(self):
        if self.returns_series.empty:
            return "Load data to activate the Smart Brain"
        try:
            sharpe = float(qs.stats.sharpe(self.returns_series) or 0)
            sortino = float(qs.stats.sortino(self.returns_series) or 0)
            health = int(50 + (sharpe * 20) + (sortino * 15))
            health = min(100, max(20, health))
            recent = self.returns_series.tail(120)
            slope = np.polyfit(range(len(recent)), recent.cumsum(), 1)[0]
            forecast = slope * 30
            anomalies = int((recent - recent.mean()).abs().gt(recent.std() * 2.8).sum())
            suggestion = "ADD DEFENSIVE ASSETS" if health < 60 else "INCREASE RISK FOR HIGHER RETURN" if health > 85 else "PERFECT BALANCE — HOLD"
            return f"SMART BRAIN ACTIVATED\nHealth Score: {health}/100\n30-day forecast: {forecast:.1%}\nAnomalies detected: {anomalies}\nSUGGESTION: {suggestion}"
        except Exception as ex:
            return f"Smart Brain encountered an issue: {ex}"

    def answer_ai_question(self, question: str):
        q = question.lower().strip()
        try:
            if "total value" in q or "portfolio value" in q or "how much" in q:
                return f"Your current total portfolio value is ${self.total_value:,.2f}."
            elif "sharpe" in q:
                val = float(qs.stats.sharpe(self.returns_series) or 0) if not self.returns_series.empty else 0
                return f"Your Sharpe ratio is {val:.2f}."
            elif "sortino" in q:
                val = float(qs.stats.sortino(self.returns_series) or 0) if not self.returns_series.empty else 0
                return f"Your Sortino ratio is {val:.2f}."
            elif "calmar" in q:
                val = float(qs.stats.calmar(self.returns_series) or 0) if not self.returns_series.empty else 0
                return f"Your Calmar ratio is {val:.2f}."
            elif "volatility" in q or "risk" in q:
                vol = self.returns_series.std() * np.sqrt(252) if not self.returns_series.empty else 0
                return f"Your annualized volatility is {vol:.1%}."
            elif "projection" in q or "forecast" in q or "next" in q or "future" in q:
                recent = self.returns_series.tail(120)
                if len(recent) > 10:
                    slope = np.polyfit(range(len(recent)), recent.cumsum(), 1)[0]
                    forecast = slope * 30
                    return f"Based on recent trend, your 30-day projected return is {forecast:.1%}."
                return "Not enough data for a reliable projection yet."
            elif "insight" in q or "suggest" in q or "advice" in q or "brain" in q:
                return self.generate_smart_brain()
            elif "goal" in q or "probability" in q or "retirement" in q:
                prob = self.simulate_goal_probability()
                return f"Your probability of reaching a $500,000 goal in 10 years is {prob:.1f}% (based on 10,000 Monte Carlo paths)."
            elif "holdings" in q or "what do i own" in q or "positions" in q:
                if self.holdings.empty: return "No holdings loaded yet."
                holdings_str = "\n".join([f"• {row['symbol']}: {row['quantity']:.2f} shares" for _, row in self.holdings.iterrows()])
                return f"Your current holdings:\n{holdings_str}"
            else:
                return self.generate_smart_brain()
        except Exception as ex:
            return f"AI encountered an issue: {ex}"

    # ====================== CHART HELPER: matplotlib → base64 ======================
    @staticmethod
    def _fig_to_base64(fig, dpi=120):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#121212", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # ====================== CHART METHODS (matplotlib) ======================
    def chart_equity_curve(self):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        if not self.returns_series.empty:
            cum = (1 + self.returns_series).cumprod()
            ax.plot(cum.index, cum.values, color="#00ff9d", linewidth=2)
            ax.fill_between(cum.index, cum.values, alpha=0.15, color="#00ff9d")
        else:
            ax.text(0.5, 0.5, "Import data to see equity curve", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("LIVE EQUITY CURVE", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}" if self.total_value > 1 else f"{x:.2f}"))
        return self._fig_to_base64(fig)

    def chart_correlation_matrix(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor("#1a1a2e")
        corr = self.correlation_matrix()
        if corr is not None and corr.shape[0] > 1:
            im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="#aaaaaa", fontsize=8)
            ax.set_yticklabels(corr.columns, color="#aaaaaa", fontsize=8)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, "Need 2+ assets for correlation matrix", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("CORRELATION MATRIX", color="#ffd700", fontsize=14, fontweight="bold")
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

    def chart_portfolio_dna(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor("#1a1a2e")
        benchmarks = {"S&P 500": (0.10, 0.15), "NASDAQ": (0.14, 0.20), "Bonds": (0.03, 0.05), "60/40": (0.07, 0.10)}
        for name, (ret, vol) in benchmarks.items():
            ax.scatter(vol, ret, s=120, alpha=0.6, zorder=3)
            ax.annotate(name, (vol, ret), textcoords="offset points", xytext=(8, 5), color="#aaaaaa", fontsize=9)
        my_ret = float(self.returns_series.mean() * 252) if not self.returns_series.empty else 0.08
        my_vol = float(self.returns_series.std() * np.sqrt(252)) if not self.returns_series.empty else 0.16
        ax.scatter(my_vol, my_ret, s=350, color="#ffd700", marker="*", zorder=5, edgecolors="#ffffff", linewidths=1)
        ax.annotate("YOUR DNA", (my_vol, my_ret), textcoords="offset points", xytext=(12, 8), color="#ffd700", fontsize=12, fontweight="bold")
        ax.set_xlabel("Annualized Volatility", color="#aaaaaa")
        ax.set_ylabel("Annualized Return", color="#aaaaaa")
        ax.set_title("PORTFOLIO DNA FINGERPRINT", color="#ffd700", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.grid(True, alpha=0.2, color="#555555")
        return self._fig_to_base64(fig)

    def chart_monte_carlo(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#1a1a2e")
        paths = self.monte_carlo_simulation(years=5, simulations=200)
        if paths is not None:
            for i in range(min(200, paths.shape[1])):
                ax.plot(paths[:, i], alpha=0.05, color="#00ff9d", linewidth=0.5)
            ax.plot(np.median(paths, axis=1), color="#ffd700", linewidth=2, label="Median Path")
            ax.plot(np.percentile(paths, 95, axis=1), color="#00ff9d", linewidth=1, linestyle="--", label="95th Percentile")
            ax.plot(np.percentile(paths, 5, axis=1), color="#ff3366", linewidth=1, linestyle="--", label="5th Percentile")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa")
        else:
            ax.text(0.5, 0.5, "Import data to run Monte Carlo simulation", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("MONTE CARLO MULTIVERSE (200 paths)", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for spine in ax.spines.values():
            spine.set_color("#333333")
        return self._fig_to_base64(fig)

    def chart_constellation(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor("#0a0a2e")
        if not self.holdings.empty:
            tickers = self.holdings['symbol'].tolist()[:20]
            qtys = self.holdings['quantity'].tolist()[:20]
        else:
            tickers = ["No Data"]
            qtys = [1]
        n = len(tickers)
        np.random.seed(42)
        x = np.random.rand(n)
        y = np.random.rand(n)
        max_q = max(qtys) if qtys else 1
        sizes = [max(80, (q / max_q) * 600) for q in qtys]
        palette = ["#ffd700", "#00ff9d", "#ff9900", "#00bfff", "#ff3366", "#9966ff", "#ff6699", "#ffcc00", "#33ccff", "#ff6600"]
        colors = [palette[i % len(palette)] for i in range(n)]
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.8, zorder=3, edgecolors="white", linewidths=0.5)
        for i, t in enumerate(tickers):
            ax.annotate(t, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha="center", color="white", fontsize=9, fontweight="bold")
        corr = self.correlation_matrix()
        if corr is not None:
            for i in range(min(n, len(corr))):
                for j in range(i + 1, min(n, len(corr))):
                    try:
                        c = abs(corr.iloc[i, j])
                        if c > 0.5:
                            ax.plot([x[i], x[j]], [y[i], y[j]], color="#333366", alpha=c * 0.5, linewidth=c * 2)
                    except Exception:
                        pass
        ax.set_title("PORTFOLIO CONSTELLATION (from your holdings)", color="#ffd700", fontsize=14, fontweight="bold")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis("off")
        return self._fig_to_base64(fig)

    def chart_candlestick(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#1a1a2e")
        if not self.returns_series.empty:
            cum = (1 + self.returns_series).cumprod().tail(120)
            for i in range(1, len(cum)):
                color = "#00ff9d" if cum.iloc[i] >= cum.iloc[i - 1] else "#ff3366"
                ax.bar(i, abs(cum.iloc[i] - cum.iloc[i - 1]), bottom=min(cum.iloc[i], cum.iloc[i - 1]), color=color, width=0.6, alpha=0.8)
            ax.plot(range(len(cum)), cum.values, color="#ffd700", linewidth=1, alpha=0.5)
            # projection line
            if len(cum) > 10:
                slope = np.polyfit(range(len(cum)), cum.values, 1)
                proj_x = range(len(cum), len(cum) + 30)
                proj_y = np.polyval(slope, proj_x)
                ax.plot(proj_x, proj_y, color="#ffd700", linewidth=2, linestyle="--", alpha=0.7, label="Projection")
                ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa")
        else:
            ax.text(0.5, 0.5, "Import data to see market graphs", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("LIVE MARKET GRAPH + PROJECTION", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        return self._fig_to_base64(fig)

    # ====================== NON-CHART METHODS ======================
    def emotional_risk_gauge_value(self):
        if self.returns_series.empty:
            return 0
        vol = float(self.returns_series.std() * np.sqrt(252))
        max_dd = float((self.returns_series.cumsum() - self.returns_series.cumsum().cummax()).min()) if len(self.returns_series) > 1 else 0
        stress = int(min(100, max(0, vol * 200 + abs(max_dd) * 100)))
        return stress

    def time_machine_projection(self, future_year):
        years_ahead = future_year - datetime.now().year
        if not self.returns_series.empty:
            ann_ret = float(self.returns_series.mean() * 252)
            ann_vol = float(self.returns_series.std() * np.sqrt(252))
            growth = (1 + ann_ret) ** years_ahead
            low = self.total_value * growth * max(0.3, 1 - ann_vol * years_ahead * 0.3)
            high = self.total_value * growth * (1 + ann_vol * years_ahead * 0.3)
            projected = self.total_value * growth
            goal_pct = min(100, int(projected / 500000 * 100)) if projected > 0 else 0
            regime = "Bull" if ann_ret > 0.08 else "Bear" if ann_ret < -0.02 else "Sideways"
            return f"YEAR {future_year} PROJECTION: ${projected:,.0f}\nRange: ${low:,.0f} — ${high:,.0f}\nGoal Progress: {goal_pct}% | Regime: {regime}"
        projected = self.total_value * (1.07 ** years_ahead)
        return f"YEAR {future_year} PROJECTION: ${projected:,.0f}\n(Using default 7% growth — import data for real projection)"

    def daily_diary(self):
        lines = [f"DAILY DIARY — {datetime.now().strftime('%B %d, %Y')}\n"]
        if self.master_df.empty:
            lines.append("No data loaded yet. Import a folder to generate your daily diary.")
            return "\n".join(lines)
        lines.append(f"Total records: {len(self.master_df):,}")
        lines.append(f"Symbols tracked: {len(self.holdings)}")
        lines.append(f"Portfolio value: ${self.total_value:,.2f}")
        if not self.returns_series.empty:
            today_ret = float(self.returns_series.iloc[-1]) if len(self.returns_series) > 0 else 0
            week_ret = float(self.returns_series.tail(5).sum()) if len(self.returns_series) >= 5 else 0
            lines.append(f"Last daily return: {today_ret:.2%}")
            lines.append(f"Last 5-day return: {week_ret:.2%}")
            vol = float(self.returns_series.std() * np.sqrt(252))
            lines.append(f"Annualized volatility: {vol:.1%}")
        if not self.holdings.empty:
            top = self.holdings.nlargest(3, 'quantity')
            lines.append("\nTop holdings by quantity:")
            for _, r in top.iterrows():
                lines.append(f"  {r['symbol']}: {r['quantity']:.2f}")
        return "\n".join(lines)

    def dream_life_architect(self, dream_description):
        if not dream_description or not dream_description.strip():
            return "Please describe your dream life first!"
        if not self.returns_series.empty:
            ann_ret = float(self.returns_series.mean() * 252)
            growth_8y = self.total_value * (1 + ann_ret) ** 8
            prob = min(99, max(10, int(self.simulate_goal_probability(target=growth_8y * 0.8, years=8))))
        else:
            growth_8y = self.total_value * (1.07 ** 8)
            prob = 50
        n_holdings = len(self.holdings) if not self.holdings.empty else 0
        return (
            f"DREAM LIFE ANALYSIS: '{dream_description}'\n\n"
            f"Current portfolio: ${self.total_value:,.0f} across {n_holdings} assets\n"
            f"Projected value in 8 years: ${growth_8y:,.0f}\n"
            f"Success probability: {prob}%\n\n"
            "Recommendation: Diversify across growth, income, and defensive assets "
            "to maximize probability of reaching your dream."
        )

    def stress_test_lab(self, event):
        scenarios = {
            "2008 Crash": (-0.45, "18 months"),
            "2020 COVID": (-0.34, "5 months"),
            "Hyperinflation 15%": (-0.22, "24 months"),
            "Dot-Com Bubble": (-0.49, "30 months"),
            "Custom Black Swan": (-0.55, "36 months"),
        }
        impact, recovery = scenarios.get(event, (-0.30, "12 months"))
        loss = self.total_value * abs(impact)
        remaining = self.total_value - loss
        lines = [f"{event.upper()} STRESS TEST\n"]
        lines.append(f"Current Portfolio: ${self.total_value:,.0f}")
        lines.append(f"Simulated Impact: -{abs(impact)*100:.0f}%  (-${loss:,.0f})")
        lines.append(f"Remaining Value: ${remaining:,.0f}")
        lines.append(f"Estimated Recovery: {recovery}")
        if not self.holdings.empty:
            lines.append(f"\nAssets affected: {len(self.holdings)}")
            worst = self.holdings.iloc[0]['symbol'] if len(self.holdings) > 0 else "N/A"
            lines.append(f"Most exposed: {worst}")
        lines.append("\nRecommendation: Add defensive allocation (bonds, gold, cash buffer)")
        return "\n".join(lines)

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

    def voice_command_sim(self, command):
        if not command or not command.strip():
            return "Please type a command."
        cmd = command.lower().strip()
        if "value" in cmd or "worth" in cmd:
            return f"Your portfolio value is ${self.total_value:,.2f} across {len(self.holdings)} assets."
        elif "risk" in cmd or "volatility" in cmd:
            vol = float(self.returns_series.std() * np.sqrt(252)) if not self.returns_series.empty else 0
            return f"Your annualized volatility is {vol:.1%}."
        elif "holdings" in cmd or "positions" in cmd:
            if self.holdings.empty:
                return "No holdings loaded."
            lines = [f"{r['symbol']}: {r['quantity']:.2f}" for _, r in self.holdings.head(10).iterrows()]
            return "Your top holdings:\n" + "\n".join(lines)
        return self.answer_ai_question(command)

    def get_holdings_table_data(self):
        if self.holdings.empty:
            return []
        rows = []
        for _, row in self.holdings.iterrows():
            sym = row["symbol"]
            qty = row["quantity"]
            price = 0.0
            if not self.prices.empty and sym in self.prices.columns:
                price = float(self.prices[sym].dropna().iloc[-1]) if not self.prices[sym].dropna().empty else 0.0
            elif not self.prices.empty and self.prices.ndim == 1:
                price = float(self.prices.dropna().iloc[-1]) if not self.prices.dropna().empty else 0.0
            mv = qty * price
            rows.append({"symbol": sym, "quantity": qty, "price": price, "market_value": mv})
        return rows

    def monthly_comparison(self):
        if self.master_df.empty or 'date' not in self.master_df.columns:
            return pd.DataFrame()
        df = self.master_df.copy()
        df['year_month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('year_month').agg(
            records=('symbol', 'count'),
            symbols=('symbol', 'nunique'),
            total_qty=('quantity', 'sum'),
        ).reset_index()
        monthly['year_month'] = monthly['year_month'].astype(str)
        if len(monthly) >= 2:
            monthly['records_chg'] = monthly['records'].pct_change().fillna(0) * 100
            monthly['qty_chg'] = monthly['total_qty'].pct_change().fillna(0) * 100
        else:
            monthly['records_chg'] = 0.0
            monthly['qty_chg'] = 0.0
        return monthly.tail(24)

    def time_machine_table(self, start_year=None, end_year=2050):
        now_year = datetime.now().year
        if start_year is None:
            start_year = now_year + 1
        if not self.returns_series.empty:
            ann_ret = float(self.returns_series.mean() * 252)
            ann_vol = float(self.returns_series.std() * np.sqrt(252))
        else:
            ann_ret = 0.0
            ann_vol = 0.0
        rows = []
        for yr in range(start_year, end_year + 1):
            y_ahead = yr - now_year
            projected = self.total_value * ((1 + ann_ret) ** y_ahead)
            low = projected * max(0.3, 1 - ann_vol * y_ahead * 0.25)
            high = projected * (1 + ann_vol * y_ahead * 0.25)
            rows.append({"year": yr, "projected": projected, "low": low, "high": high})
        return rows

    def financial_planner_detail(self, monthly_contribution=0, target_retirement_age=65, current_age=30, savings_rate=20, annual_income=0, annual_expenses=0, tax_rate=0, inflation_rate=3.0, custom_goal=0):
        now_year = datetime.now().year
        if not self.returns_series.empty:
            ann_ret = float(self.returns_series.mean() * 252)
            ann_vol = float(self.returns_series.std() * np.sqrt(252))
        else:
            ann_ret = 0.0
            ann_vol = 0.0
        n = len(self.holdings) if not self.holdings.empty else 0
        years_to_retire = max(1, target_retirement_age - current_age)
        infl = inflation_rate / 100.0
        real_ret = max(0.001, ann_ret - infl)
        # Future value with monthly contributions
        monthly_rate = ann_ret / 12
        real_monthly = real_ret / 12
        if monthly_rate > 0 and monthly_contribution > 0:
            fv_contributions = monthly_contribution * (((1 + monthly_rate) ** (years_to_retire * 12) - 1) / monthly_rate)
            fv_contributions_real = monthly_contribution * (((1 + real_monthly) ** (years_to_retire * 12) - 1) / real_monthly)
        else:
            fv_contributions = monthly_contribution * years_to_retire * 12
            fv_contributions_real = fv_contributions
        fv_portfolio = self.total_value * ((1 + ann_ret) ** years_to_retire)
        fv_portfolio_real = self.total_value * ((1 + real_ret) ** years_to_retire)
        fv_total = fv_portfolio + fv_contributions
        fv_total_real = fv_portfolio_real + fv_contributions_real
        # Safe withdrawal (4% rule)
        annual_income_at_retire = fv_total * 0.04
        monthly_income_at_retire = annual_income_at_retire / 12
        annual_income_real = fv_total_real * 0.04
        monthly_income_real = annual_income_real / 12
        # Tax-adjusted income
        net_annual = annual_income_at_retire * (1 - tax_rate / 100.0) if tax_rate > 0 else annual_income_at_retire
        net_monthly = net_annual / 12
        # Surplus calc
        annual_surplus = annual_income - annual_expenses if annual_income > 0 else 0
        monthly_surplus = annual_surplus / 12
        # Risk-adjusted projections
        fv_low = fv_total * max(0.3, 1 - ann_vol * years_to_retire * 0.15)
        fv_high = fv_total * (1 + ann_vol * years_to_retire * 0.15)
        # Milestones
        milestones = [
            ("Emergency Fund (6 mo)", 25000),
            ("First $50K", 50000),
            ("First $100K", 100000),
            ("Quarter Million", 250000),
            ("Half Million", 500000),
            ("Millionaire", 1000000),
            ("Multi-Millionaire", 2000000),
            ("Financial Freedom", 5000000),
        ]
        if custom_goal > 0:
            milestones.append(("Custom Goal", custom_goal))
            milestones.sort(key=lambda x: x[1])
        milestone_rows = []
        for label, target in milestones:
            if self.total_value >= target:
                milestone_rows.append({"label": label, "target": target, "status": "ACHIEVED", "years": 0})
            elif self.total_value > 0 and ann_ret > 0:
                if monthly_contribution > 0 and monthly_rate > 0:
                    months_needed = 0
                    bal = self.total_value
                    while bal < target and months_needed < 1200:
                        bal = bal * (1 + monthly_rate) + monthly_contribution
                        months_needed += 1
                    years_needed = months_needed / 12
                else:
                    years_needed = math.log(target / max(self.total_value, 1)) / math.log(1 + ann_ret) if ann_ret > 0 else 999
                target_year = now_year + int(years_needed)
                milestone_rows.append({"label": label, "target": target, "status": f"~{int(years_needed)}y (by {target_year})", "years": years_needed})
            else:
                milestone_rows.append({"label": label, "target": target, "status": "Import data", "years": 999})
        # Allocation suggestion
        if current_age < 35:
            alloc = "90% Equities / 10% Bonds — aggressive growth phase"
            alloc_pcts = {"Equities": 90, "Bonds": 10}
        elif current_age < 50:
            alloc = "70% Equities / 20% Bonds / 10% Alternatives"
            alloc_pcts = {"Equities": 70, "Bonds": 20, "Alternatives": 10}
        elif current_age < 60:
            alloc = "50% Equities / 35% Bonds / 15% Cash/Alternatives"
            alloc_pcts = {"Equities": 50, "Bonds": 35, "Cash/Alts": 15}
        else:
            alloc = "30% Equities / 50% Bonds / 20% Cash — capital preservation"
            alloc_pcts = {"Equities": 30, "Bonds": 50, "Cash": 20}
        # 10-year projection table (nominal + real)
        projection = []
        bal_nom = self.total_value
        bal_real = self.total_value
        for yr in range(1, 11):
            bal_nom = bal_nom * (1 + ann_ret) + monthly_contribution * 12
            bal_real = bal_real * (1 + real_ret) + monthly_contribution * 12
            projection.append({"year": now_year + yr, "nominal": bal_nom, "real": bal_real})
        return {
            "current_value": self.total_value,
            "assets": n,
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "monthly_contribution": monthly_contribution,
            "current_age": current_age,
            "retire_age": target_retirement_age,
            "years_to_retire": years_to_retire,
            "fv_portfolio": fv_portfolio,
            "fv_contributions": fv_contributions,
            "fv_total": fv_total,
            "fv_total_real": fv_total_real,
            "fv_low": fv_low,
            "fv_high": fv_high,
            "annual_income": annual_income_at_retire,
            "monthly_income": monthly_income_at_retire,
            "annual_income_real": annual_income_real,
            "monthly_income_real": monthly_income_real,
            "net_annual_after_tax": net_annual,
            "net_monthly_after_tax": net_monthly,
            "annual_surplus": annual_surplus,
            "monthly_surplus": monthly_surplus,
            "inflation_rate": inflation_rate,
            "tax_rate": tax_rate,
            "savings_rate": savings_rate,
            "allocation": alloc,
            "alloc_pcts": alloc_pcts,
            "milestones": milestone_rows,
            "projection_5y": projection[:5],
            "projection_10y": projection,
        }

    def benchmark_comparison(self, benchmarks=None):
        if benchmarks is None:
            benchmarks = ['SPY', 'QQQ', 'IWM', 'BTC-USD']
        result = []
        try:
            bm_data = yf.download(benchmarks, period="5y", progress=False, auto_adjust=True)
            bm_lvl0 = bm_data.columns.get_level_values(0) if isinstance(bm_data.columns, pd.MultiIndex) else bm_data.columns
            bm_col = 'Close' if 'Close' in bm_lvl0 else ('Adj Close' if 'Adj Close' in bm_lvl0 else None)
            if bm_col and isinstance(bm_data.columns, pd.MultiIndex):
                bm_prices = bm_data[bm_col]
            elif bm_col:
                bm_prices = bm_data[[bm_col]].rename(columns={bm_col: benchmarks[0]}) if len(benchmarks) == 1 else bm_data
            else:
                bm_prices = bm_data
            for bm in benchmarks:
                if bm in bm_prices.columns:
                    bm_ret = bm_prices[bm].pct_change().dropna()
                    ann_r = float(bm_ret.mean() * 252)
                    ann_v = float(bm_ret.std() * np.sqrt(252))
                    bm_clean = bm_prices[bm].dropna()
                    total_r = float((bm_clean.iloc[-1] / bm_clean.iloc[0]) - 1) if len(bm_clean) > 1 else float('nan')
                    if not self.returns_series.empty and len(bm_ret) > 10:
                        corr = float(self.returns_series.corr(bm_ret))
                        if np.isnan(corr):
                            corr = None
                    else:
                        corr = None
                    result.append({"symbol": bm, "ann_return": ann_r, "ann_vol": ann_v, "total_return": total_r, "correlation": corr})
        except Exception:
            pass
        return result

    def estimate_dividends(self):
        divs = []
        if self.holdings.empty:
            return divs
        for _, row in self.holdings.iterrows():
            sym = row['symbol']
            qty = float(row.get('quantity', 0) or 0)
            try:
                info = yf.Ticker(sym).info
                div_yield = info.get('dividendYield', 0) or 0
                div_rate = info.get('dividendRate', 0) or 0
                annual_div = qty * div_rate if div_rate > 0 else 0
                divs.append({"symbol": sym, "quantity": qty, "div_yield": div_yield, "div_rate": div_rate, "annual_income": annual_div})
            except Exception:
                divs.append({"symbol": sym, "quantity": qty, "div_yield": 0, "div_rate": 0, "annual_income": 0})
        return divs

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

    def chart_benchmark_overlay(self, benchmarks=None, show_portfolio=True, timerange='ALL'):
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
                bm_data = yf.download(benchmarks, period=yf_period, interval=yf_interval, progress=False, auto_adjust=True)
                bm_lvl0 = bm_data.columns.get_level_values(0) if isinstance(bm_data.columns, pd.MultiIndex) else bm_data.columns
                bm_col = 'Close' if 'Close' in bm_lvl0 else ('Adj Close' if 'Adj Close' in bm_lvl0 else None)
                if bm_col and isinstance(bm_data.columns, pd.MultiIndex):
                    bm_prices = bm_data[bm_col]
                elif bm_col:
                    bm_prices = bm_data[[bm_col]].rename(columns={bm_col: benchmarks[0]}) if len(benchmarks) == 1 else bm_data
                else:
                    bm_prices = bm_data
                fb_idx = 0
                # Trim benchmark data to N trading days if specified
                n_days = tr_days.get(timerange.upper())
                for bm in benchmarks:
                    if bm in bm_prices.columns:
                        bm_ser = bm_prices[bm].dropna()
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
                pass
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

    def chart_allocation_pie(self, alloc_pcts):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#0a0a0a')
        labels = list(alloc_pcts.keys())
        sizes = list(alloc_pcts.values())
        colors = ['#00ff9d', '#ffd700', '#ff3366', '#00bfff', '#ff9900'][:len(labels)]
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors, textprops={'color': '#ffffff', 'fontsize': 11}, pctdistance=0.75)
        for t in autotexts:
            t.set_fontweight('bold')
        ax.set_title('Suggested Allocation', color='#ffffff', fontsize=13)
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
            spy_dl = yf.download("SPY", period="5y", progress=False, auto_adjust=True)
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

    def fire_calculator(self, annual_expenses=50000, current_age=30, monthly_contribution=500):
        ann_ret = float(self.returns_series.mean() * 252) if not self.returns_series.empty else 0.0
        lean_target = annual_expenses * 0.7 * 25
        regular_target = annual_expenses * 25
        fat_target = annual_expenses * 1.5 * 25
        coast_target = regular_target / ((1 + ann_ret) ** max(1, 65 - current_age)) if ann_ret > 0 else regular_target
        def years_to(target):
            if self.total_value >= target: return 0
            bal, months, mr = self.total_value, 0, ann_ret / 12
            while bal < target and months < 1200:
                bal = bal * (1 + mr) + monthly_contribution
                months += 1
            return months / 12
        return {"lean_fire": lean_target, "lean_years": years_to(lean_target), "regular_fire": regular_target, "regular_years": years_to(regular_target), "fat_fire": fat_target, "fat_years": years_to(fat_target), "coast_fire": coast_target, "coast_reached": self.total_value >= coast_target, "safe_withdrawal": self.total_value * 0.04, "monthly_passive": self.total_value * 0.04 / 12}

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

    def rebalance_suggestions(self):
        opt = self.optimize_portfolio()
        if opt is None: return []
        weights, _ = opt
        suggestions = []
        for sym, target_w in weights.items():
            if target_w < 0.01: continue
            current_w = 0
            for _, row in self.holdings.iterrows():
                if row['symbol'] == sym:
                    qty = float(row.get('quantity', 0) or 0)
                    price = 0
                    if not self.prices.empty and self.prices.ndim == 2 and sym in self.prices.columns:
                        col = self.prices[sym].dropna()
                        price = float(col.iloc[-1]) if not col.empty else 0
                    current_w = (qty * price / self.total_value) if self.total_value > 0 else 0
            diff = target_w - current_w
            action = "BUY" if diff > 0.01 else "SELL" if diff < -0.01 else "HOLD"
            suggestions.append({"symbol": sym, "current_weight": current_w, "target_weight": target_w, "action": action, "diff": diff})
        return suggestions

    def chart_drawdown(self):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        dd = self.drawdown_series()
        if not dd.empty:
            ax.fill_between(dd.index, dd.values, 0, color="#ff3366", alpha=0.4)
            ax.plot(dd.index, dd.values, color="#ff3366", linewidth=1.5)
        else:
            ax.text(0.5, 0.5, "Import data to see drawdown", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("DRAWDOWN CHART", color="#ff3366", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values(): spine.set_color("#333333")
        return self._fig_to_base64(fig)

    def chart_holdings_bar(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#1a1a2e")
        hw = self.holdings_with_weights()
        if hw:
            hw = hw[:15]
            syms = [r['symbol'] for r in hw]
            vals = [r['market_value'] for r in hw]
            colors = ["#00ff9d" if r.get('gain_loss', 0) >= 0 else "#ff3366" for r in hw]
            ax.barh(syms[::-1], vals[::-1], color=colors[::-1], edgecolor="#333333")
        else:
            ax.text(0.5, 0.5, "Import data to see holdings", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("HOLDINGS BY MARKET VALUE", color="#ffd700", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for spine in ax.spines.values(): spine.set_color("#333333")
        return self._fig_to_base64(fig)

    def chart_returns_histogram(self):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        if not self.returns_series.empty:
            ax.hist(self.returns_series.values, bins=50, color="#00ff9d", alpha=0.7, edgecolor="#333333")
            ax.axvline(self.returns_series.mean(), color="#ffd700", linewidth=2, linestyle="--", label=f"Mean: {self.returns_series.mean():.4f}")
            ax.axvline(0, color="#ffffff", linewidth=1, alpha=0.5)
            var = self.value_at_risk()
            ax.axvline(var, color="#ff3366", linewidth=2, linestyle="--", label=f"VaR 95%: {var:.4f}")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa")
        else:
            ax.text(0.5, 0.5, "Import data to see returns distribution", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("RETURNS DISTRIBUTION", color="#00ff9d", fontsize=14, fontweight="bold")
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

    def chart_gain_loss_waterfall(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#1a1a2e")
        hw = self.holdings_with_weights()
        if hw:
            hw = sorted(hw, key=lambda x: x['gain_loss'], reverse=True)[:15]
            syms = [r['symbol'] for r in hw]
            gains = [r['gain_loss'] for r in hw]
            colors = ["#00ff9d" if g >= 0 else "#ff3366" for g in gains]
            ax.bar(syms, gains, color=colors, edgecolor="#333333")
            ax.axhline(0, color="#ffffff", linewidth=0.5, alpha=0.5)
        else:
            ax.text(0.5, 0.5, "Import data for gain/loss analysis", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("GAIN / LOSS WATERFALL", color="#ffd700", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for spine in ax.spines.values(): spine.set_color("#333333")
        plt.xticks(rotation=45, ha="right")
        return self._fig_to_base64(fig)

    def chart_rolling_sharpe(self):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        rs = self.rolling_sharpe(60)
        if not rs.empty:
            ax.plot(rs.index, rs.values, color="#00ff9d", linewidth=1.5)
            ax.axhline(0, color="#ff3366", linewidth=1, linestyle="--", alpha=0.7)
            ax.axhline(1, color="#ffd700", linewidth=1, linestyle="--", alpha=0.5, label="Good (1.0)")
            ax.fill_between(rs.index, rs.values, 0, where=(rs.values > 0), alpha=0.15, color="#00ff9d")
            ax.fill_between(rs.index, rs.values, 0, where=(rs.values < 0), alpha=0.15, color="#ff3366")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa")
        else:
            ax.text(0.5, 0.5, "Need 60+ days for rolling Sharpe", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("ROLLING SHARPE RATIO (60-day)", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values(): spine.set_color("#333333")
        return self._fig_to_base64(fig)

    def chart_rolling_volatility(self):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        rv = self.rolling_volatility(30)
        if not rv.empty:
            ax.plot(rv.index, rv.values, color="#ff9900", linewidth=1.5)
            ax.fill_between(rv.index, rv.values, alpha=0.2, color="#ff9900")
            ax.axhline(rv.mean(), color="#ffd700", linewidth=1, linestyle="--", label=f"Avg: {rv.mean():.1%}")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa")
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

    def chart_weight_pie(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#121212")
        hw = self.holdings_with_weights()
        if hw:
            top = hw[:10]
            other_val = sum(r['market_value'] for r in hw[10:])
            labels = [r['symbol'] for r in top]
            sizes = [r['market_value'] for r in top]
            if other_val > 0:
                labels.append("OTHER")
                sizes.append(other_val)
            palette = ["#00ff9d","#ffd700","#ff3366","#00bfff","#ff9900","#9966ff","#ff6699","#33ccff","#ffcc00","#66ff66","#888888"]
            colors = [palette[i % len(palette)] for i in range(len(labels))]
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'color': '#ffffff', 'fontsize': 9})
        else:
            ax.text(0.5, 0.5, "Import data for weight breakdown", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("PORTFOLIO WEIGHT ALLOCATION", color="#ffd700", fontsize=14, fontweight="bold")
        return self._fig_to_base64(fig)

    def chart_growth_projection(self, monthly_contribution=500):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_facecolor("#1a1a2e")
        if self.returns_series.empty or self.total_value <= 0:
            ax.text(0.5, 0.5, 'No portfolio data loaded — import data first', ha='center', va='center', color='#aaaaaa', fontsize=14, transform=ax.transAxes)
            ax.set_title('30-YEAR GROWTH PROJECTION', color='#00ff9d', fontsize=14, fontweight='bold')
            ax.tick_params(colors='#aaaaaa')
            for spine in ax.spines.values(): spine.set_color('#333333')
            return self._fig_to_base64(fig)
        ann_ret = float(self.returns_series.mean() * 252)
        mr = ann_ret / 12
        months = list(range(1, 361))
        portfolio_only = [self.total_value * ((1 + mr) ** m) for m in months]
        with_contrib = []
        bal = self.total_value
        for m in months:
            bal = bal * (1 + mr) + monthly_contribution
            with_contrib.append(bal)
        ax.plot(months, portfolio_only, color="#ffd700", linewidth=1.5, label="Growth Only")
        ax.plot(months, with_contrib, color="#00ff9d", linewidth=2, label=f"+ ${monthly_contribution}/mo")
        ax.fill_between(months, portfolio_only, with_contrib, alpha=0.15, color="#00ff9d")
        ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa")
        ax.set_title("30-YEAR GROWTH PROJECTION", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values(): spine.set_color("#333333")
        ax.grid(True, alpha=0.15, color="#555555")
        return self._fig_to_base64(fig)

    def chart_efficient_frontier(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#1a1a2e")
        if not self.prices.empty and self.prices.ndim == 2 and len(self.prices.columns) >= 2:
            try:
                mu = expected_returns.mean_historical_return(self.prices)
                S = risk_models.sample_cov(self.prices)
                rets_range = np.linspace(mu.min(), mu.max(), 40)
                fv, fr = [], []
                for tr in rets_range:
                    try:
                        ef = EfficientFrontier(mu, S)
                        ef.efficient_return(tr)
                        perf = ef.portfolio_performance()
                        fr.append(perf[0]); fv.append(perf[1])
                    except Exception: pass
                if fv:
                    ax.plot(fv, fr, color="#00ff9d", linewidth=2, label="Efficient Frontier")
                for sym in self.prices.columns:
                    col = self.prices[sym].dropna()
                    if len(col) > 30:
                        r = float(col.pct_change().dropna().mean() * 252)
                        v = float(col.pct_change().dropna().std() * np.sqrt(252))
                        ax.scatter(v, r, s=60, zorder=4, alpha=0.7)
                        ax.annotate(sym, (v, r), textcoords="offset points", xytext=(5, 5), color="#aaaaaa", fontsize=8)
                my_ret = float(self.returns_series.mean() * 252) if not self.returns_series.empty else 0
                my_vol = float(self.returns_series.std() * np.sqrt(252)) if not self.returns_series.empty else 0
                ax.scatter(my_vol, my_ret, s=250, color="#ffd700", marker="*", zorder=5, label="Your Portfolio")
                ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#aaaaaa")
            except Exception:
                ax.text(0.5, 0.5, "Could not compute frontier", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Need 2+ assets for frontier", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("EFFICIENT FRONTIER", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.set_xlabel("Volatility", color="#aaaaaa"); ax.set_ylabel("Return", color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values(): spine.set_color("#333333")
        ax.grid(True, alpha=0.15, color="#555555")
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

    def tax_loss_harvest(self):
        if self.holdings.empty or 'gain_loss' not in self.holdings.columns:
            return []
        losers = self.holdings[self.holdings['gain_loss'] < 0].copy()
        if losers.empty:
            return []
        results = []
        for _, row in losers.iterrows():
            sym = row['symbol']
            loss = float(row['gain_loss'])
            qty = float(row.get('quantity', 0) or 0)
            pct = float(row.get('gain_loss_pct', 0) or 0)
            tax_saved = abs(loss) * 0.22
            results.append({"symbol": sym, "quantity": qty, "loss": loss, "loss_pct": pct, "est_tax_savings": tax_saved})
        return sorted(results, key=lambda x: x['loss'])

    def what_if_scenario(self, ann_return_override=None, ann_vol_override=None, monthly_contrib=500, years=10):
        ann_ret = ann_return_override if ann_return_override is not None else (float(self.returns_series.mean() * 252) if not self.returns_series.empty else 0.0)
        ann_vol = ann_vol_override if ann_vol_override is not None else (float(self.returns_series.std() * np.sqrt(252)) if not self.returns_series.empty else 0.0)
        mr = ann_ret / 12
        bal_base, bal_low, bal_high = self.total_value, self.total_value, self.total_value
        projection = []
        for yr in range(1, years + 1):
            for _ in range(12):
                bal_base = bal_base * (1 + mr) + monthly_contrib
                bal_low = bal_low * (1 + (mr - ann_vol / np.sqrt(12) * 0.5)) + monthly_contrib
                bal_high = bal_high * (1 + (mr + ann_vol / np.sqrt(12) * 0.5)) + monthly_contrib
            projection.append({"year": yr, "base": bal_base, "low": max(0, bal_low), "high": bal_high})
        total_contrib = monthly_contrib * years * 12
        return {"projection": projection, "final_base": bal_base, "final_low": max(0, bal_low), "final_high": bal_high,
                "total_contributions": total_contrib, "total_growth": bal_base - self.total_value - total_contrib,
                "ann_return_used": ann_ret, "ann_vol_used": ann_vol}

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

    def chart_what_if(self, projection):
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#121212')
        if projection:
            yrs = [p['year'] for p in projection]
            base = [p['base'] for p in projection]
            low = [p['low'] for p in projection]
            high = [p['high'] for p in projection]
            ax.plot(yrs, base, color='#00ff9d', linewidth=2.5, label='Base Case', marker='o', markersize=4)
            ax.plot(yrs, low, color='#ff3366', linewidth=1.5, linestyle='--', label='Bear Case')
            ax.plot(yrs, high, color='#ffd700', linewidth=1.5, linestyle='--', label='Bull Case')
            ax.fill_between(yrs, low, high, alpha=0.1, color='#00ff9d')
            ax.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='#ffffff')
        else:
            ax.text(0.5, 0.5, "Run a scenario to see projection", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title('WHAT-IF SCENARIO PROJECTION', color='#00ff9d', fontsize=14, fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_xlabel('Year', color='#aaaaaa')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values(): spine.set_color('#333333')
        ax.grid(True, alpha=0.15, color='#555555')
        return self._fig_to_base64(fig)

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

    def chart_daily_profit(self):
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#121212')
        if not self.returns_series.empty:
            daily_pnl = self.returns_series * self.total_value if self.total_value > 0 else self.returns_series
            colors = ['#00ff9d' if v >= 0 else '#ff3366' for v in daily_pnl.values]
            ax.bar(range(len(daily_pnl)), daily_pnl.values, color=colors, alpha=0.7, width=1.0)
            ax.axhline(0, color='#555555', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, "Import data to see daily profit", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
        ax.set_title("DAILY PROFIT / LOSS", color="#00ff9d", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
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

    def inspect_asset(self, symbol):
        result = {"symbol": symbol, "found": False}
        if self.prices.empty or (self.prices.ndim == 2 and symbol not in self.prices.columns):
            return result
        result["found"] = True
        col = self.prices[symbol].dropna() if self.prices.ndim == 2 else self.prices.dropna()
        if len(col) < 2:
            return result
        ret = col.pct_change().dropna()
        result["current_price"] = float(col.iloc[-1])
        result["start_price"] = float(col.iloc[0])
        result["total_return"] = float(col.iloc[-1] / col.iloc[0] - 1)
        result["ann_return"] = float(ret.mean() * 252)
        result["ann_vol"] = float(ret.std() * np.sqrt(252))
        result["sharpe"] = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
        result["max_dd"] = float(((col / col.cummax()) - 1).min())
        result["best_day"] = float(ret.max())
        result["worst_day"] = float(ret.min())
        result["pos_days"] = int((ret > 0).sum())
        result["neg_days"] = int((ret < 0).sum())
        result["win_rate"] = result["pos_days"] / max(1, result["pos_days"] + result["neg_days"])
        daily_pnl = ret * float(col.iloc[-1])
        weekly_pnl = ret.resample('W').sum() * float(col.iloc[-1]) if len(ret) >= 5 else pd.Series(dtype=float)
        result["avg_daily_pnl"] = float(daily_pnl.mean()) if len(daily_pnl) > 0 else 0
        result["avg_weekly_pnl"] = float(weekly_pnl.mean()) if len(weekly_pnl) > 0 else 0
        return result

    def chart_asset_inspection(self, symbol):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle(f"ASSET INSPECTION: {symbol}", color='#ffd700', fontsize=16, fontweight='bold')
        for ax in axes.flat:
            ax.set_facecolor('#121212')
            ax.tick_params(colors='#aaaaaa')
            for spine in ax.spines.values(): spine.set_color('#333333')
        if self.prices.empty or (self.prices.ndim == 2 and symbol not in self.prices.columns):
            for ax in axes.flat:
                ax.text(0.5, 0.5, f"No data for {symbol}", ha="center", va="center", color="#aaaaaa", fontsize=12, transform=ax.transAxes)
            return self._fig_to_base64(fig)
        col = self.prices[symbol].dropna() if self.prices.ndim == 2 else self.prices.dropna()
        ret = col.pct_change().dropna()
        axes[0, 0].plot(col.index, col.values, color='#00ff9d', linewidth=1.5)
        axes[0, 0].set_title("Price History", color='#00ff9d', fontsize=11)
        axes[0, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
        colors = ['#00ff9d' if v >= 0 else '#ff3366' for v in ret.values[-60:]]
        axes[0, 1].bar(range(len(ret.values[-60:])), ret.values[-60:], color=colors, width=1.0, alpha=0.7)
        axes[0, 1].set_title("Daily Returns (last 60d)", color='#ffd700', fontsize=11)
        axes[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
        cum = (1 + ret).cumprod()
        dd = (cum / cum.cummax()) - 1
        axes[1, 0].fill_between(dd.index, dd.values, 0, color='#ff3366', alpha=0.4)
        axes[1, 0].set_title("Drawdown", color='#ff3366', fontsize=11)
        axes[1, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        if len(ret) >= 5:
            rw = ret.rolling(5).sum() * float(col.iloc[-1])
            rw = rw.dropna()
            axes[1, 1].plot(rw.index, rw.values, color='#00bfff', linewidth=1.2)
            axes[1, 1].fill_between(rw.index, rw.values, 0, where=rw.values >= 0, color='#00ff9d', alpha=0.2)
            axes[1, 1].fill_between(rw.index, rw.values, 0, where=rw.values < 0, color='#ff3366', alpha=0.2)
        axes[1, 1].set_title("Rolling Weekly Profit", color='#00bfff', fontsize=11)
        axes[1, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        plt.tight_layout()
        return self._fig_to_base64(fig)

    # ====================== LIVE CHART DATA ======================
    def fetch_live_data(self, symbol, timeframe="1d"):
        tf_map = {"1h": ("1d", "1m"), "24h": ("5d", "5m"), "1mo": ("1mo", "30m"), "1y": ("1y", "1d")}
        period, interval = tf_map.get(timeframe, ("1d", "5m"))
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception:
            return pd.DataFrame()

    def chart_live_ticker(self, symbol, timeframe="1d"):
        df = self.fetch_live_data(symbol, timeframe)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        fig.patch.set_facecolor('#0a0a0a')
        for ax in axes:
            ax.set_facecolor('#121212')
            ax.tick_params(colors='#aaaaaa', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#333333')
        tf_labels = {"1h": "1 HOUR", "24h": "24 HOURS", "1mo": "1 MONTH", "1y": "1 YEAR"}
        title = f"{symbol} — {tf_labels.get(timeframe, timeframe.upper())} (LIVE)"
        if df.empty or 'Close' not in df.columns:
            axes[0].text(0.5, 0.5, f"No data for {symbol}", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=axes[0].transAxes)
            axes[0].set_title(title, color="#00ff9d", fontsize=14, fontweight="bold")
            return self._fig_to_base64(fig)
        close = df['Close'].values.flatten()
        idx = range(len(close))
        color = '#00ff9d' if len(close) >= 2 and close[-1] >= close[0] else '#ff3366'
        axes[0].plot(idx, close, color=color, linewidth=1.5)
        axes[0].fill_between(idx, close, close.min(), alpha=0.1, color=color)
        axes[0].set_title(title, color="#00ff9d", fontsize=14, fontweight="bold")
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
        if 'High' in df.columns and 'Low' in df.columns:
            axes[0].fill_between(idx, df['High'].values.flatten(), df['Low'].values.flatten(), alpha=0.05, color='#ffd700')
        pct_chg = ((close[-1] / close[0]) - 1) * 100 if close[0] != 0 else 0
        axes[0].text(0.02, 0.95, f"${close[-1]:,.2f}  ({pct_chg:+.2f}%)", transform=axes[0].transAxes,
                     fontsize=16, fontweight='bold', color=color, va='top')
        if 'Volume' in df.columns:
            vol = df['Volume'].values.flatten()
            vol_colors = ['#00ff9d' if i == 0 or close[i] >= close[i-1] else '#ff3366' for i in range(len(close))]
            axes[1].bar(idx, vol, color=vol_colors, alpha=0.6, width=0.8)
            axes[1].set_title("VOLUME", color="#ffd700", fontsize=10, fontweight="bold")
            axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
        else:
            axes[1].text(0.5, 0.5, "No volume data", ha="center", va="center", color="#555555", fontsize=10, transform=axes[1].transAxes)
        axes[0].grid(True, alpha=0.1, color='#555555')
        axes[1].grid(True, alpha=0.1, color='#555555')
        plt.tight_layout()
        return self._fig_to_base64(fig)

    # ====================== ADVANCED TECHNICAL CHART ======================
    def chart_advanced_technical(self, symbol, timeframe="1y", indicators=None):
        if indicators is None:
            indicators = {}
        df = self.fetch_live_data(symbol, timeframe)
        num_panels = 1
        show_rsi = indicators.get('rsi', False)
        show_macd = indicators.get('macd', False)
        if show_rsi:
            num_panels += 1
        if show_macd:
            num_panels += 1
        ratios = [4] + [1.2] * (num_panels - 1)
        fig, axes = plt.subplots(num_panels, 1, figsize=(14, 4 + num_panels * 2), gridspec_kw={'height_ratios': ratios}, sharex=True)
        if num_panels == 1:
            axes = [axes]
        fig.patch.set_facecolor('#0a0a0a')
        for ax in axes:
            ax.set_facecolor('#121212')
            ax.tick_params(colors='#aaaaaa', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#333333')
            ax.grid(True, alpha=0.08, color='#555555')
        ax_main = axes[0]
        tf_labels = {"1h": "1H", "24h": "24H", "1mo": "1M", "1y": "1Y"}
        title = f"{symbol} — TECHNICAL ANALYSIS ({tf_labels.get(timeframe, timeframe.upper())})"
        if df.empty or 'Close' not in df.columns:
            ax_main.text(0.5, 0.5, f"No data for {symbol}", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax_main.transAxes)
            ax_main.set_title(title, color="#ffd700", fontsize=14, fontweight="bold")
            return self._fig_to_base64(fig)
        close = df['Close'].values.flatten()
        high = df['High'].values.flatten() if 'High' in df.columns else close
        low = df['Low'].values.flatten() if 'Low' in df.columns else close
        idx = np.arange(len(close))
        ax_main.plot(idx, close, color='#ffffff', linewidth=1.2, label='Close', zorder=5)
        ax_main.set_title(title, color="#ffd700", fontsize=14, fontweight="bold")
        ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
        for period, color, lbl in [(20, '#ffd700', 'SMA 20'), (50, '#00bfff', 'SMA 50'), (200, '#ff9900', 'SMA 200')]:
            if indicators.get(f'sma{period}', False) and len(close) >= period:
                sma = pd.Series(close).rolling(period).mean().values
                ax_main.plot(idx, sma, color=color, linewidth=1, alpha=0.8, label=lbl)
        for period, color, lbl in [(12, '#9966ff', 'EMA 12'), (26, '#ff6699', 'EMA 26')]:
            if indicators.get(f'ema{period}', False) and len(close) >= period:
                ema = pd.Series(close).ewm(span=period, adjust=False).mean().values
                ax_main.plot(idx, ema, color=color, linewidth=1, alpha=0.8, label=lbl, linestyle='--')
        if indicators.get('bollinger', False) and len(close) >= 20:
            sma20 = pd.Series(close).rolling(20).mean()
            std20 = pd.Series(close).rolling(20).std()
            upper = (sma20 + 2 * std20).values
            lower = (sma20 - 2 * std20).values
            ax_main.plot(idx, upper, color='#00ff9d', linewidth=0.7, alpha=0.5, label='BB Upper')
            ax_main.plot(idx, lower, color='#ff3366', linewidth=0.7, alpha=0.5, label='BB Lower')
            ax_main.fill_between(idx, upper, lower, alpha=0.04, color='#00ff9d')
        if indicators.get('vwap', False) and 'Volume' in df.columns:
            vol = df['Volume'].values.flatten().astype(float)
            typical = (high + low + close) / 3
            cum_tp_vol = np.cumsum(typical * vol)
            cum_vol = np.cumsum(vol)
            vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, close)
            ax_main.plot(idx, vwap, color='#ff00ff', linewidth=1, alpha=0.7, label='VWAP', linestyle='-.')
        if indicators.get('volume', False) and 'Volume' in df.columns:
            vol = df['Volume'].values.flatten()
            if vol.max() > 0:
                vol_norm = vol / vol.max() * (close.max() - close.min()) * 0.2 + close.min()
            else:
                vol_norm = vol
            vol_colors = ['#00ff9d44' if i == 0 or close[i] >= close[i-1] else '#ff336644' for i in range(len(close))]
            ax_main.bar(idx, vol_norm - close.min(), bottom=close.min(), color=vol_colors, width=0.8, zorder=1)
        handles, labels = ax_main.get_legend_handles_labels()
        if labels:
            ax_main.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e', edgecolor='#333333', labelcolor='#cccccc', ncol=3)
        panel_idx = 1
        if show_rsi and len(close) >= 14:
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi_vals = rsi.values
            ax_rsi = axes[panel_idx]
            ax_rsi.plot(idx, rsi_vals, color='#ffd700', linewidth=1)
            ax_rsi.axhline(70, color='#ff3366', linewidth=0.7, linestyle='--', alpha=0.7)
            ax_rsi.axhline(30, color='#00ff9d', linewidth=0.7, linestyle='--', alpha=0.7)
            ax_rsi.fill_between(idx, rsi_vals, 70, where=rsi_vals > 70, alpha=0.15, color='#ff3366')
            ax_rsi.fill_between(idx, rsi_vals, 30, where=rsi_vals < 30, alpha=0.15, color='#00ff9d')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_title("RSI (14)", color="#ffd700", fontsize=9, fontweight="bold")
            panel_idx += 1
        if show_macd and len(close) >= 26:
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            ax_macd = axes[panel_idx]
            ax_macd.plot(idx, macd_line.values, color='#00bfff', linewidth=1, label='MACD')
            ax_macd.plot(idx, signal_line.values, color='#ff9900', linewidth=1, label='Signal')
            hist_colors = ['#00ff9d' if v >= 0 else '#ff3366' for v in histogram.values]
            ax_macd.bar(idx, histogram.values, color=hist_colors, alpha=0.5, width=0.8)
            ax_macd.axhline(0, color='#555555', linewidth=0.5)
            ax_macd.set_title("MACD (12,26,9)", color="#00bfff", fontsize=9, fontweight="bold")
            ax_macd.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e', edgecolor='#333333', labelcolor='#cccccc')
        plt.tight_layout()
        return self._fig_to_base64(fig)

    # ====================== MULTI-ASSET CHART (toggleable per-asset lines) ======================
    def chart_multi_asset(self, selected_symbols=None, normalize=True, show_volume=False):
        fig, ax_main = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#0a0a0a')
        ax_main.set_facecolor('#121212')
        ax_main.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax_main.spines.values():
            spine.set_color('#333333')
        ax_main.grid(True, alpha=0.08, color='#555555')
        if self.prices.empty:
            ax_main.text(0.5, 0.5, "No price data. Import assets first.",
                         ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax_main.transAxes)
            ax_main.set_title("MULTI-ASSET FOCUS", color="#ffd700", fontsize=14, fontweight="bold")
            return self._fig_to_base64(fig)
        all_syms = list(self.prices.columns) if self.prices.ndim == 2 else []
        if not all_syms:
            ax_main.text(0.5, 0.5, "Single-column data — use Asset Inspection instead",
                         ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax_main.transAxes)
            return self._fig_to_base64(fig)
        if selected_symbols is None:
            selected_symbols = all_syms[:12]
        clist = ['#00ff9d', '#ffd700', '#ff3366', '#00bfff', '#ff9900', '#9966ff',
                 '#ff6699', '#33cccc', '#ff00ff', '#66ff66', '#ff6600', '#6699ff',
                 '#cc33ff', '#33ff99', '#ffcc00', '#ff3399']
        plotted = 0
        for i, sym in enumerate(selected_symbols):
            if sym not in self.prices.columns:
                continue
            col = self.prices[sym].dropna()
            if col.empty:
                continue
            color = clist[i % len(clist)]
            if normalize and len(col) > 1:
                vals = col / col.iloc[0] * 100
                label = f"{sym} (norm)"
            else:
                vals = col
                label = sym
            ax_main.plot(vals.index, vals.values, color=color, linewidth=1.5, label=label, alpha=0.9)
            plotted += 1
        if plotted == 0:
            ax_main.text(0.5, 0.5, "No data for selected symbols",
                         ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax_main.transAxes)
        title_suffix = "NORMALIZED (base=100)" if normalize else "ABSOLUTE PRICE"
        ax_main.set_title(f"MULTI-ASSET FOCUS — {title_suffix}", color="#ffd700", fontsize=14, fontweight="bold")
        if normalize:
            ax_main.axhline(100, color='#555555', linewidth=0.5, linestyle='--', alpha=0.5)
            ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
        else:
            ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
        if plotted > 0:
            ax_main.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e',
                           edgecolor='#333333', labelcolor='#cccccc', ncol=min(plotted, 4))
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def chart_asset_correlation_scatter(self, sym_a, sym_b):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#121212')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_color('#333333')
        if self.prices.empty or self.prices.ndim < 2:
            ax.text(0.5, 0.5, "Need 2+ assets", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
            return self._fig_to_base64(fig)
        if sym_a not in self.prices.columns or sym_b not in self.prices.columns:
            ax.text(0.5, 0.5, f"Missing {sym_a} or {sym_b}", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
            return self._fig_to_base64(fig)
        ret_a = self.prices[sym_a].pct_change().dropna()
        ret_b = self.prices[sym_b].pct_change().dropna()
        common = ret_a.index.intersection(ret_b.index)
        if len(common) < 5:
            ax.text(0.5, 0.5, "Insufficient overlap", ha="center", va="center", color="#aaaaaa", fontsize=14, transform=ax.transAxes)
            return self._fig_to_base64(fig)
        a_vals = ret_a.loc[common].values
        b_vals = ret_b.loc[common].values
        corr = np.corrcoef(a_vals, b_vals)[0, 1]
        colors_arr = ['#00ff9d' if (a > 0 and b > 0) else '#ff3366' if (a < 0 and b < 0) else '#ffd700' for a, b in zip(a_vals, b_vals)]
        ax.scatter(a_vals, b_vals, c=colors_arr, alpha=0.5, s=12, edgecolors='none')
        z = np.polyfit(a_vals, b_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(a_vals.min(), a_vals.max(), 100)
        ax.plot(x_line, p(x_line), color='#ff9900', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.axhline(0, color='#555555', linewidth=0.5)
        ax.axvline(0, color='#555555', linewidth=0.5)
        corr_color = '#00ff9d' if corr > 0.3 else '#ff3366' if corr < -0.3 else '#ffd700'
        ax.set_title(f"{sym_a} vs {sym_b} — Correlation: {corr:.3f}", color=corr_color, fontsize=13, fontweight="bold")
        ax.set_xlabel(f"{sym_a} Daily Return", color='#aaaaaa', fontsize=10)
        ax.set_ylabel(f"{sym_b} Daily Return", color='#aaaaaa', fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.grid(True, alpha=0.08, color='#555555')
        plt.tight_layout()
        return self._fig_to_base64(fig)

    # ====================== ADVANCED RISK METRICS (20) ======================
    def omega_ratio(self, threshold=0.0):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); excess = r - threshold
        gains = excess[excess > 0].sum(); losses = -excess[excess <= 0].sum()
        return float(gains / losses) if losses > 0 else float('inf') if gains > 0 else 0.0

    def tail_ratio(self, alpha=0.05):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna()
        right = np.percentile(r, 100 - alpha * 100); left = np.percentile(r, alpha * 100)
        return float(right / abs(left)) if abs(left) > 1e-10 else 0.0

    def gain_to_pain_ratio(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); total = r.sum(); pain = (-r[r < 0]).sum()
        return float(total / pain) if pain > 0 else 0.0

    def ulcer_index(self):
        if self.returns_series.empty: return 0.0
        cum = (1 + self.returns_series).cumprod(); rm = cum.cummax()
        dd_pct = ((cum - rm) / rm) * 100
        return float(np.sqrt((dd_pct ** 2).mean()))

    def pain_index(self):
        if self.returns_series.empty: return 0.0
        cum = (1 + self.returns_series).cumprod(); rm = cum.cummax()
        return float(((cum - rm) / rm).abs().mean())

    def sterling_ratio(self, periods=252):
        if self.returns_series.empty: return 0.0
        ann_ret = float(self.returns_series.mean() * periods)
        cum = (1 + self.returns_series).cumprod()
        avg_dd = abs(float(((cum / cum.cummax()) - 1).min()))
        return float(ann_ret / avg_dd) if avg_dd > 1e-10 else 0.0

    def burke_ratio(self, periods=252):
        if self.returns_series.empty: return 0.0
        ann_ret = float(self.returns_series.mean() * periods)
        cum = (1 + self.returns_series).cumprod()
        dd = (cum / cum.cummax()) - 1; bd = np.sqrt(float((dd ** 2).sum()))
        return float(ann_ret / bd) if bd > 1e-10 else 0.0

    def kappa_three(self, threshold=0.0):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); em = r.mean() - threshold
        diff = threshold - r; diff = diff[diff > 0]
        lpm3 = (diff ** 3).mean() if len(diff) > 0 else 0.0
        d = lpm3 ** (1.0 / 3) if lpm3 > 0 else 0.0
        return float(em / d) if d > 1e-10 else 0.0

    def downside_deviation(self, mar=0.0):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); neg = (r - mar); neg = neg[neg < 0]
        return float(np.sqrt((neg ** 2).mean())) if len(neg) > 0 else 0.0

    def upside_potential_ratio(self, mar=0.0):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); up = r[r > mar] - mar
        um = float(up.mean()) if len(up) > 0 else 0.0
        dd = self.downside_deviation(mar)
        return float(um / dd) if dd > 1e-10 else 0.0

    def parametric_var(self, confidence=0.95, periods=1):
        if self.returns_series.empty: return 0.0
        from scipy.stats import norm
        mu = float(self.returns_series.mean()); sigma = float(self.returns_series.std())
        z = norm.ppf(1 - confidence)
        return float(-(mu + z * sigma * np.sqrt(periods)))

    def historical_var(self, confidence=0.95):
        if self.returns_series.empty: return 0.0
        return float(-np.percentile(self.returns_series.dropna(), (1 - confidence) * 100))

    def conditional_var(self, confidence=0.95):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); var = np.percentile(r, (1 - confidence) * 100)
        tail = r[r <= var]
        return float(-tail.mean()) if len(tail) > 0 else 0.0

    def expected_shortfall(self, confidence=0.95):
        return self.conditional_var(confidence)

    def max_drawdown_duration(self):
        if self.returns_series.empty: return 0
        cum = (1 + self.returns_series).cumprod(); rm = cum.cummax()
        in_dd = cum < rm
        if not in_dd.any(): return 0
        groups = (~in_dd).cumsum(); dd_g = groups[in_dd]
        if dd_g.empty: return 0
        durations = dd_g.groupby(dd_g).apply(lambda g: (g.index[-1] - g.index[0]).days if len(g) > 1 else 1)
        return int(durations.max()) if len(durations) > 0 else 0

    def recovery_factor(self):
        if self.returns_series.empty: return 0.0
        cum = (1 + self.returns_series).cumprod()
        tr = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0
        mdd = abs(float(((cum / cum.cummax()) - 1).min()))
        return float(tr / mdd) if mdd > 1e-10 else 0.0

    def calmar_ratio_custom(self, periods=252):
        if self.returns_series.empty: return 0.0
        ann = float(self.returns_series.mean() * periods)
        cum = (1 + self.returns_series).cumprod()
        mdd = abs(float(((cum / cum.cummax()) - 1).min()))
        return float(ann / mdd) if mdd > 1e-10 else 0.0

    def lake_ratio(self):
        if self.returns_series.empty: return 0.0
        cum = (1 + self.returns_series).cumprod(); rm = cum.cummax()
        return float(1 - cum.sum() / rm.sum()) if rm.sum() > 0 else 0.0

    def rachev_ratio(self, alpha=0.05):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna()
        lc = np.percentile(r, alpha * 100); rc = np.percentile(r, (1 - alpha) * 100)
        lt = r[r <= lc]; rt = r[r >= rc]
        ar = float(rt.mean()) if len(rt) > 0 else 0
        al = abs(float(lt.mean())) if len(lt) > 0 else 0
        return float(ar / al) if al > 1e-10 else 0.0

    # ====================== STATISTICAL ANALYSIS (15) ======================
    def jarque_bera_test(self):
        if self.returns_series.empty: return {'jb_stat': 0, 'p_value': 1.0, 'normal': True}
        from scipy.stats import jarque_bera
        r = self.returns_series.dropna()
        if len(r) < 20: return {'jb_stat': 0, 'p_value': 1.0, 'normal': True}
        stat, p = jarque_bera(r)
        return {'jb_stat': float(stat), 'p_value': float(p), 'normal': p > 0.05}

    def hurst_exponent(self):
        if self.returns_series.empty: return 0.5
        r = self.returns_series.dropna().values; n = len(r)
        if n < 100: return 0.5
        max_k = min(n // 2, 500); lags = []; rs_vals = []
        for k in [int(x) for x in np.logspace(1, np.log10(max_k), 20).astype(int)]:
            if k < 10: continue
            rs_list = []
            for start in range(0, n - k, k):
                seg = r[start:start + k]; ms = seg.mean()
                dev = np.cumsum(seg - ms); R = dev.max() - dev.min(); S = seg.std(ddof=1)
                if S > 1e-10: rs_list.append(R / S)
            if rs_list: lags.append(np.log(k)); rs_vals.append(np.log(np.mean(rs_list)))
        if len(lags) < 3: return 0.5
        return float(np.clip(np.polyfit(lags, rs_vals, 1)[0], 0.0, 1.0))

    def autocorrelation(self, lag=1):
        if self.returns_series.empty or len(self.returns_series) <= lag: return 0.0
        return float(self.returns_series.dropna().autocorr(lag))

    def autocorrelation_profile(self, max_lag=20):
        return {lag: self.autocorrelation(lag) for lag in range(1, max_lag + 1)}

    def rolling_correlation_pair(self, sym_a, sym_b, window=60):
        if self.prices.empty or self.prices.ndim < 2: return pd.Series(dtype=float)
        if sym_a not in self.prices.columns or sym_b not in self.prices.columns: return pd.Series(dtype=float)
        ra = self.prices[sym_a].pct_change().dropna(); rb = self.prices[sym_b].pct_change().dropna()
        common = ra.index.intersection(rb.index)
        if len(common) < window: return pd.Series(dtype=float)
        return ra.loc[common].rolling(window).corr(rb.loc[common])

    def half_life_mean_reversion(self):
        if self.returns_series.empty or len(self.returns_series) < 30: return float('inf')
        cum = (1 + self.returns_series).cumprod(); log_p = np.log(cum)
        y = log_p.diff().dropna(); x = log_p.shift(1).loc[y.index]
        if x.std() < 1e-10: return float('inf')
        slope = np.polyfit(x, y, 1)[0]
        return float(-np.log(2) / slope) if slope < 0 else float('inf')

    def zscore_analysis(self):
        if self.returns_series.empty or len(self.returns_series) < 30: return {'zscore': 0, 'signal': 'neutral'}
        cum = (1 + self.returns_series).cumprod()
        m = cum.rolling(60, min_periods=20).mean(); s = cum.rolling(60, min_periods=20).std()
        z = float((cum.iloc[-1] - m.iloc[-1]) / s.iloc[-1]) if s.iloc[-1] > 1e-10 else 0
        return {'zscore': z, 'signal': 'overbought' if z > 2 else 'oversold' if z < -2 else 'neutral'}

    def cointegration_test(self, sym_a, sym_b):
        if self.prices.empty or self.prices.ndim < 2: return {'cointegrated': False}
        if sym_a not in self.prices.columns or sym_b not in self.prices.columns: return {'cointegrated': False}
        from scipy.stats import linregress
        a = self.prices[sym_a].dropna(); b = self.prices[sym_b].dropna()
        common = a.index.intersection(b.index)
        if len(common) < 30: return {'cointegrated': False}
        av = a.loc[common].values; bv = b.loc[common].values
        slope, intercept, _, _, _ = linregress(bv, av)
        residuals = av - (slope * bv + intercept); diff_res = np.diff(residuals); res_lag = residuals[:-1]
        if np.std(res_lag) < 1e-10: return {'cointegrated': False}
        adf_slope = np.polyfit(res_lag, diff_res, 1)[0]
        t_stat = adf_slope / (np.std(diff_res) / (np.std(res_lag) * np.sqrt(len(res_lag))))
        return {'cointegrated': t_stat < -2.86, 'adf_stat': float(t_stat), 'hedge_ratio': float(slope)}

    def skewness_kurtosis_detail(self):
        if self.returns_series.empty: return {'skewness': 0, 'kurtosis': 3, 'excess_kurtosis': 0}
        r = self.returns_series.dropna(); s = float(r.skew()); k = float(r.kurtosis()) + 3
        return {'skewness': s, 'kurtosis': k, 'excess_kurtosis': k - 3,
                'skew_interp': 'left-skewed' if s < -0.5 else 'right-skewed' if s > 0.5 else 'symmetric',
                'kurt_interp': 'leptokurtic' if k - 3 > 1 else 'platykurtic' if k - 3 < -1 else 'mesokurtic'}

    def regime_detection_simple(self):
        if self.returns_series.empty or len(self.returns_series) < 63: return {'regime': 'unknown'}
        r = self.returns_series; mom = float(r.rolling(63).sum().iloc[-1])
        v21 = float(r.rolling(21).std().iloc[-1]) * np.sqrt(252)
        if mom > 0.05: return {'regime': 'bull', 'momentum_63d': mom, 'vol_21d': v21}
        elif mom < -0.05: return {'regime': 'bear', 'momentum_63d': mom, 'vol_21d': v21}
        return {'regime': 'sideways', 'momentum_63d': mom, 'vol_21d': v21}

    def volatility_cone(self, windows=None):
        if self.returns_series.empty: return {}
        if windows is None: windows = [5, 10, 21, 42, 63, 126, 252]
        r = self.returns_series.dropna(); result = {}
        for w in windows:
            if len(r) < w + 10: continue
            rv = (r.rolling(w).std() * np.sqrt(252)).dropna()
            result[w] = {'current': float(rv.iloc[-1]), 'min': float(rv.min()), 'p25': float(np.percentile(rv, 25)),
                         'median': float(rv.median()), 'p75': float(np.percentile(rv, 75)), 'max': float(rv.max())}
        return result

    def ewma_volatility(self, lam=0.94):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna().values; var = r[0] ** 2
        for i in range(1, len(r)): var = lam * var + (1 - lam) * r[i] ** 2
        return float(np.sqrt(var) * np.sqrt(252))

    def parkinson_volatility(self, window=21):
        if self.master_df.empty or 'high' not in self.master_df.columns or 'low' not in self.master_df.columns: return 0.0
        hl = np.log(self.master_df['high'].astype(float) / self.master_df['low'].astype(float))
        return float(np.sqrt((1 / (4 * window * np.log(2))) * (hl.tail(window) ** 2).sum()) * np.sqrt(252))

    def garman_klass_volatility(self, window=21):
        if self.master_df.empty: return 0.0
        needed = {'open', 'high', 'low', 'close'}
        if not needed.issubset(set(self.master_df.columns)): return 0.0
        d = self.master_df.tail(window).astype(float)
        hl = np.log(d['high'] / d['low']); co = np.log(d['close'] / d['open'])
        return float(np.sqrt((0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)).mean()) * np.sqrt(252))

    def yang_zhang_volatility(self, window=21):
        if self.master_df.empty: return 0.0
        df = self.master_df.tail(window + 1); needed = {'open', 'high', 'low', 'close'}
        if not needed.issubset(set(df.columns)) or len(df) < 3: return 0.0
        d = df.astype(float)
        o = np.log(d['open'].iloc[1:].values / d['close'].iloc[:-1].values)
        c = np.log(d['close'].iloc[1:].values / d['open'].iloc[1:].values)
        h = np.log(d['high'].iloc[1:].values / d['open'].iloc[1:].values)
        l = np.log(d['low'].iloc[1:].values / d['open'].iloc[1:].values)
        n = len(o); k = 0.34 / (1.34 + (n + 1) / (n - 1))
        so = np.var(o, ddof=1); sc = np.var(c, ddof=1); rs = (h * (h - c) + l * (l - c)).mean()
        return float(np.sqrt(max(so + k * sc + (1 - k) * rs, 0)) * np.sqrt(252))

    # ====================== TECHNICAL INDICATORS (20) ======================
    def fibonacci_retracement(self, symbol=None):
        if self.prices.empty: return {}
        col = self.prices[symbol].dropna() if symbol and self.prices.ndim == 2 and symbol in self.prices.columns else (self.prices.iloc[:, 0].dropna() if self.prices.ndim == 2 else self.prices.dropna())
        if len(col) < 10: return {}
        high = float(col.max()); low = float(col.min()); diff = high - low
        levels = {0.0: high, 0.236: high - diff * 0.236, 0.382: high - diff * 0.382, 0.5: high - diff * 0.5, 0.618: high - diff * 0.618, 0.786: high - diff * 0.786, 1.0: low}
        current = float(col.iloc[-1]); nearest = min(levels.items(), key=lambda x: abs(x[1] - current))
        return {'levels': levels, 'current_price': current, 'nearest_level': nearest[0], 'nearest_price': nearest[1]}

    def compute_atr(self, symbol=None, period=14):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return 0.0
        h = df['High'].values.flatten() if 'High' in df.columns else df['Close'].values.flatten()
        lo = df['Low'].values.flatten() if 'Low' in df.columns else df['Close'].values.flatten()
        c = df['Close'].values.flatten()
        tr = np.maximum(h - lo, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
        tr[0] = h[0] - lo[0]
        return float(pd.Series(tr).rolling(period).mean().iloc[-1])

    def compute_cci(self, symbol=None, period=20):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return 0.0
        h = df['High'].values.flatten() if 'High' in df.columns else df['Close'].values.flatten()
        lo = df['Low'].values.flatten() if 'Low' in df.columns else df['Close'].values.flatten()
        c = df['Close'].values.flatten(); tp = (h + lo + c) / 3; tps = pd.Series(tp)
        sma = tps.rolling(period).mean(); mad = tps.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tps - sma) / (0.015 * mad)
        return float(cci.iloc[-1]) if not np.isnan(cci.iloc[-1]) else 0.0

    def compute_williams_r(self, symbol=None, period=14):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return 0.0
        h = pd.Series(df['High'].values.flatten()) if 'High' in df.columns else pd.Series(df['Close'].values.flatten())
        lo = pd.Series(df['Low'].values.flatten()) if 'Low' in df.columns else pd.Series(df['Close'].values.flatten())
        c = df['Close'].values.flatten(); hh = h.rolling(period).max(); ll = lo.rolling(period).min()
        denom = hh.iloc[-1] - ll.iloc[-1]
        return float(-100 * (hh.iloc[-1] - c[-1]) / denom) if denom > 0 else 0.0

    def compute_stochastic(self, symbol=None, k_period=14, d_period=3):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return {'k': 0, 'd': 0}
        h = pd.Series(df['High'].values.flatten()) if 'High' in df.columns else pd.Series(df['Close'].values.flatten())
        lo = pd.Series(df['Low'].values.flatten()) if 'Low' in df.columns else pd.Series(df['Close'].values.flatten())
        c = pd.Series(df['Close'].values.flatten()); hh = h.rolling(k_period).max(); ll = lo.rolling(k_period).min()
        kv = 100 * (c - ll) / (hh - ll); dv = kv.rolling(d_period).mean()
        return {'k': float(kv.iloc[-1]) if not np.isnan(kv.iloc[-1]) else 0, 'd': float(dv.iloc[-1]) if not np.isnan(dv.iloc[-1]) else 0}

    def compute_obv(self, symbol=None):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns or 'Volume' not in df.columns: return pd.Series(dtype=float)
        c = df['Close'].values.flatten(); v = df['Volume'].values.flatten().astype(float)
        obv = np.zeros(len(c))
        for i in range(1, len(c)):
            obv[i] = obv[i - 1] + (v[i] if c[i] > c[i - 1] else -v[i] if c[i] < c[i - 1] else 0)
        return pd.Series(obv, index=df.index)

    def compute_mfi(self, symbol=None, period=14):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Volume' not in df.columns or 'Close' not in df.columns: return 0.0
        h = df['High'].values.flatten() if 'High' in df.columns else df['Close'].values.flatten()
        lo = df['Low'].values.flatten() if 'Low' in df.columns else df['Close'].values.flatten()
        c = df['Close'].values.flatten(); v = df['Volume'].values.flatten().astype(float)
        tp = (h + lo + c) / 3; mf = tp * v
        pos_mf = np.zeros(len(tp)); neg_mf = np.zeros(len(tp))
        for i in range(1, len(tp)):
            if tp[i] > tp[i - 1]: pos_mf[i] = mf[i]
            else: neg_mf[i] = mf[i]
        ps = pd.Series(pos_mf).rolling(period).sum().iloc[-1]
        ns = pd.Series(neg_mf).rolling(period).sum().iloc[-1]
        if ns == 0: return 100.0
        return float(100 - (100 / (1 + ps / ns)))

    def compute_adx(self, symbol=None, period=14):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'High' not in df.columns: return {'adx': 0, 'plus_di': 0, 'minus_di': 0}
        h = df['High'].values.flatten(); lo = df['Low'].values.flatten(); c = df['Close'].values.flatten()
        tr = np.maximum(h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))
        pdm = np.where((h[1:] - h[:-1]) > (lo[:-1] - lo[1:]), np.maximum(h[1:] - h[:-1], 0), 0)
        mdm = np.where((lo[:-1] - lo[1:]) > (h[1:] - h[:-1]), np.maximum(lo[:-1] - lo[1:], 0), 0)
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        pdi = 100 * pd.Series(pdm).ewm(span=period, adjust=False).mean() / atr
        mdi = 100 * pd.Series(mdm).ewm(span=period, adjust=False).mean() / atr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi).replace(0, 1)
        adx = dx.ewm(span=period, adjust=False).mean()
        return {'adx': float(adx.iloc[-1]), 'plus_di': float(pdi.iloc[-1]), 'minus_di': float(mdi.iloc[-1])}

    def compute_roc(self, symbol=None, period=12):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return 0.0
        c = df['Close'].values.flatten()
        return float((c[-1] - c[-1 - period]) / c[-1 - period] * 100) if len(c) > period else 0.0

    def compute_trix(self, symbol=None, period=15):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return 0.0
        c = pd.Series(df['Close'].values.flatten())
        e1 = c.ewm(span=period, adjust=False).mean()
        e2 = e1.ewm(span=period, adjust=False).mean()
        e3 = e2.ewm(span=period, adjust=False).mean()
        trix = e3.pct_change() * 100
        return float(trix.iloc[-1]) if not np.isnan(trix.iloc[-1]) else 0.0

    def compute_aroon(self, symbol=None, period=25):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return {'aroon_up': 0, 'aroon_down': 0, 'oscillator': 0}
        h = df['High'].values.flatten() if 'High' in df.columns else df['Close'].values.flatten()
        lo = df['Low'].values.flatten() if 'Low' in df.columns else df['Close'].values.flatten()
        if len(h) < period + 1: return {'aroon_up': 0, 'aroon_down': 0, 'oscillator': 0}
        dsh = period - np.argmax(h[-period - 1:]); dsl = period - np.argmin(lo[-period - 1:])
        au = ((period - dsh) / period) * 100; ad = ((period - dsl) / period) * 100
        return {'aroon_up': float(au), 'aroon_down': float(ad), 'oscillator': float(au - ad)}

    def compute_ultimate_oscillator(self, symbol=None, p1=7, p2=14, p3=28):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return 0.0
        h = df['High'].values.flatten() if 'High' in df.columns else df['Close'].values.flatten()
        lo = df['Low'].values.flatten() if 'Low' in df.columns else df['Close'].values.flatten()
        c = df['Close'].values.flatten()
        bp = c[1:] - np.minimum(lo[1:], c[:-1])
        tr = np.maximum(h[1:], c[:-1]) - np.minimum(lo[1:], c[:-1])
        bps = pd.Series(bp); trs = pd.Series(tr)
        a1 = bps.rolling(p1).sum() / trs.rolling(p1).sum()
        a2 = bps.rolling(p2).sum() / trs.rolling(p2).sum()
        a3 = bps.rolling(p3).sum() / trs.rolling(p3).sum()
        uo = 100 * (4 * a1 + 2 * a2 + a3) / 7
        return float(uo.iloc[-1]) if not np.isnan(uo.iloc[-1]) else 0.0

    def compute_donchian(self, symbol=None, period=20):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return {'upper': 0, 'lower': 0, 'mid': 0}
        h = pd.Series(df['High'].values.flatten()) if 'High' in df.columns else pd.Series(df['Close'].values.flatten())
        lo = pd.Series(df['Low'].values.flatten()) if 'Low' in df.columns else pd.Series(df['Close'].values.flatten())
        upper = float(h.tail(period).max()); lower = float(lo.tail(period).min())
        return {'upper': upper, 'lower': lower, 'mid': (upper + lower) / 2}

    def compute_keltner(self, symbol=None, ema_period=20, atr_period=10, multiplier=2.0):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return {'upper': 0, 'mid': 0, 'lower': 0}
        c = pd.Series(df['Close'].values.flatten())
        h = df['High'].values.flatten() if 'High' in df.columns else c.values
        lo = df['Low'].values.flatten() if 'Low' in df.columns else c.values
        tr = np.maximum(h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c.values[:-1]), np.abs(lo[1:] - c.values[:-1])))
        atr_v = float(pd.Series(tr).rolling(atr_period).mean().iloc[-1])
        mid = float(c.ewm(span=ema_period, adjust=False).mean().iloc[-1])
        return {'upper': mid + multiplier * atr_v, 'mid': mid, 'lower': mid - multiplier * atr_v}

    def compute_ichimoku(self, symbol=None):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return {}
        h = pd.Series(df['High'].values.flatten()) if 'High' in df.columns else pd.Series(df['Close'].values.flatten())
        lo = pd.Series(df['Low'].values.flatten()) if 'Low' in df.columns else pd.Series(df['Close'].values.flatten())
        c = df['Close'].values.flatten()
        tenkan = float((h.tail(9).max() + lo.tail(9).min()) / 2)
        kijun = float((h.tail(26).max() + lo.tail(26).min()) / 2)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = float((h.tail(52).max() + lo.tail(52).min()) / 2)
        signal = 'bullish' if c[-1] > max(senkou_a, senkou_b) else 'bearish' if c[-1] < min(senkou_a, senkou_b) else 'neutral'
        return {'tenkan': tenkan, 'kijun': kijun, 'senkou_a': senkou_a, 'senkou_b': senkou_b, 'signal': signal}

    def compute_force_index(self, symbol=None, period=13):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns or 'Volume' not in df.columns: return 0.0
        c = pd.Series(df['Close'].values.flatten())
        v = pd.Series(df['Volume'].values.flatten().astype(float))
        fi = (c.diff() * v).ewm(span=period, adjust=False).mean()
        return float(fi.iloc[-1]) if not np.isnan(fi.iloc[-1]) else 0.0

    def compute_elder_ray(self, symbol=None, period=13):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return {'bull_power': 0, 'bear_power': 0}
        h = df['High'].values.flatten() if 'High' in df.columns else df['Close'].values.flatten()
        lo = df['Low'].values.flatten() if 'Low' in df.columns else df['Close'].values.flatten()
        ema = pd.Series(df['Close'].values.flatten()).ewm(span=period, adjust=False).mean().iloc[-1]
        return {'bull_power': float(h[-1] - ema), 'bear_power': float(lo[-1] - ema)}

    def compute_chaikin_mf(self, symbol=None, period=20):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Volume' not in df.columns or 'Close' not in df.columns: return 0.0
        h = df['High'].values.flatten() if 'High' in df.columns else df['Close'].values.flatten()
        lo = df['Low'].values.flatten() if 'Low' in df.columns else df['Close'].values.flatten()
        c = df['Close'].values.flatten(); v = df['Volume'].values.flatten().astype(float)
        hl = h - lo; clv = np.where(hl > 0, ((c - lo) - (h - c)) / hl, 0); mfv = clv * v
        cmf = pd.Series(mfv).rolling(period).sum() / pd.Series(v).rolling(period).sum()
        return float(cmf.iloc[-1]) if not np.isnan(cmf.iloc[-1]) else 0.0

    def support_resistance_levels(self, symbol=None):
        df = self.fetch_live_data(symbol or 'SPY', '1y')
        if df.empty or 'Close' not in df.columns: return {'support': [], 'resistance': [], 'pivot': 0}
        h = df['High'].values.flatten() if 'High' in df.columns else df['Close'].values.flatten()
        lo = df['Low'].values.flatten() if 'Low' in df.columns else df['Close'].values.flatten()
        c = df['Close'].values.flatten()
        pivot = (h[-1] + lo[-1] + c[-1]) / 3
        s1 = 2 * pivot - h[-1]; r1 = 2 * pivot - lo[-1]
        s2 = pivot - (h[-1] - lo[-1]); r2 = pivot + (h[-1] - lo[-1])
        s3 = lo[-1] - 2 * (h[-1] - pivot); r3 = h[-1] + 2 * (pivot - lo[-1])
        return {'pivot': float(pivot), 'support': [float(s1), float(s2), float(s3)], 'resistance': [float(r1), float(r2), float(r3)]}

    # ====================== PORTFOLIO CONSTRUCTION & ATTRIBUTION (15) ======================
    def risk_parity_weights(self):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); vols = rets.std() * np.sqrt(252); vols = vols[vols > 0]
        if vols.empty: return {}
        inv_vol = 1.0 / vols; w = inv_vol / inv_vol.sum()
        return {sym: float(w[sym]) for sym in w.index}

    def minimum_variance_weights(self):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna()
        if rets.shape[1] < 2: return {}
        cov = rets.cov().values * 252; n = cov.shape[0]
        try: inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError: inv_cov = np.linalg.pinv(cov)
        ones = np.ones(n); w = inv_cov @ ones / (ones @ inv_cov @ ones)
        return {rets.columns[i]: float(w[i]) for i in range(n)}

    def maximum_diversification_weights(self):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); vols = (rets.std() * np.sqrt(252)).values
        cov = rets.cov().values * 252
        try: inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError: inv_cov = np.linalg.pinv(cov)
        w = inv_cov @ vols; w = w / w.sum()
        return {rets.columns[i]: float(w[i]) for i in range(len(vols))}

    def hierarchical_risk_parity(self):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna()
        if rets.shape[1] < 2: return {}
        vols = rets.std(); inv_vol = 1.0 / vols; w = inv_vol / inv_vol.sum()
        avg_corr = rets.corr().mean(); adj = 1 - avg_corr * 0.3; w = w * adj; w = w / w.sum()
        return {sym: float(w[sym]) for sym in w.index}

    def marginal_contribution_to_risk(self):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); cov = rets.cov().values * 252; n = cov.shape[0]
        w = np.ones(n) / n; pv = np.sqrt(w @ cov @ w); mcr = (cov @ w) / pv
        return {rets.columns[i]: float(mcr[i]) for i in range(n)}

    def component_var_calc(self, confidence=0.95):
        if self.prices.empty or self.prices.ndim < 2: return {}
        from scipy.stats import norm
        rets = self.prices.pct_change().dropna(); cov = rets.cov().values * 252; n = cov.shape[0]
        w = np.ones(n) / n; pv = np.sqrt(w @ cov @ w); z = norm.ppf(confidence)
        mcr = (cov @ w) / pv; cv = w * mcr * z
        return {rets.columns[i]: float(cv[i]) for i in range(n)}

    def portfolio_concentration_hhi(self):
        if self.holdings.empty or 'value' not in self.holdings.columns: return 0.0
        vals = self.holdings['value'].astype(float); total = vals.sum()
        if total <= 0: return 0.0
        return float(((vals / total) ** 2).sum())

    def effective_num_bets(self):
        hhi = self.portfolio_concentration_hhi()
        return float(1.0 / hhi) if hhi > 0 else 0.0

    def tracking_error_calc(self, benchmark="SPY"):
        if self.returns_series.empty: return 0.0
        try:
            bm = yf.download(benchmark, period="5y", progress=False, auto_adjust=True)['Close']
            if isinstance(bm, pd.DataFrame): bm = bm.iloc[:, 0]
            bm_ret = bm.pct_change().dropna()
        except Exception: return 0.0
        common = self.returns_series.index.intersection(bm_ret.index)
        if len(common) < 20: return 0.0
        return float((self.returns_series.loc[common] - bm_ret.loc[common]).std() * np.sqrt(252))

    def information_ratio_calc(self, benchmark="SPY"):
        if self.returns_series.empty: return 0.0
        try:
            bm = yf.download(benchmark, period="5y", progress=False, auto_adjust=True)['Close']
            if isinstance(bm, pd.DataFrame): bm = bm.iloc[:, 0]
            bm_ret = bm.pct_change().dropna()
        except Exception: return 0.0
        common = self.returns_series.index.intersection(bm_ret.index)
        if len(common) < 20: return 0.0
        active = self.returns_series.loc[common] - bm_ret.loc[common]
        te = active.std() * np.sqrt(252); ar = active.mean() * 252
        return float(ar / te) if te > 1e-10 else 0.0

    def m_squared_measure(self, benchmark="SPY", rf=0.04):
        if self.returns_series.empty: return 0.0
        try:
            bm = yf.download(benchmark, period="5y", progress=False, auto_adjust=True)['Close']
            if isinstance(bm, pd.DataFrame): bm = bm.iloc[:, 0]
            bm_ret = bm.pct_change().dropna()
        except Exception: return 0.0
        common = self.returns_series.index.intersection(bm_ret.index)
        if len(common) < 20: return 0.0
        pr = self.returns_series.loc[common]; br = bm_ret.loc[common]
        ps = (pr.mean() * 252 - rf) / (pr.std() * np.sqrt(252)) if pr.std() > 0 else 0
        return float(rf + ps * br.std() * np.sqrt(252))

    def brinson_attribution(self, benchmark_weights=None):
        if self.holdings.empty or self.prices.empty or self.prices.ndim < 2:
            return {'allocation': 0, 'selection': 0, 'interaction': 0, 'total': 0}
        rets = self.prices.pct_change().dropna()
        if rets.empty: return {'allocation': 0, 'selection': 0, 'interaction': 0, 'total': 0}
        syms = list(rets.columns); n = len(syms); pw = np.ones(n) / n
        bw = np.ones(n) / n if benchmark_weights is None else np.array([benchmark_weights.get(s, 1.0/n) for s in syms])
        mr = rets.mean().values * 252; bmr = mr * bw / bw.sum()
        alloc = sum((pw[i] - bw[i]) * bmr[i] for i in range(n))
        selec = sum(bw[i] * (mr[i] - bmr[i]) for i in range(n))
        inter = sum((pw[i] - bw[i]) * (mr[i] - bmr[i]) for i in range(n))
        return {'allocation': float(alloc), 'selection': float(selec), 'interaction': float(inter), 'total': float(alloc + selec + inter)}

    def treynor_ratio(self, benchmark="SPY", rf=0.04):
        if self.returns_series.empty: return 0.0
        try:
            bm = yf.download(benchmark, period="5y", progress=False, auto_adjust=True)['Close']
            if isinstance(bm, pd.DataFrame): bm = bm.iloc[:, 0]
            bm_ret = bm.pct_change().dropna()
        except Exception: return 0.0
        common = self.returns_series.index.intersection(bm_ret.index)
        if len(common) < 20: return 0.0
        pr = self.returns_series.loc[common]; br = bm_ret.loc[common]
        beta = pr.cov(br) / br.var() if br.var() > 0 else 1.0
        return float((pr.mean() * 252 - rf) / beta) if abs(beta) > 1e-10 else 0.0

    def jensens_alpha_calc(self, benchmark="SPY", rf=0.04):
        if self.returns_series.empty: return 0.0
        try:
            bm = yf.download(benchmark, period="5y", progress=False, auto_adjust=True)['Close']
            if isinstance(bm, pd.DataFrame): bm = bm.iloc[:, 0]
            bm_ret = bm.pct_change().dropna()
        except Exception: return 0.0
        common = self.returns_series.index.intersection(bm_ret.index)
        if len(common) < 20: return 0.0
        pr = self.returns_series.loc[common]; br = bm_ret.loc[common]
        beta = pr.cov(br) / br.var() if br.var() > 0 else 1.0
        return float(pr.mean() * 252 - rf - beta * (br.mean() * 252 - rf))

    # ====================== FINANCIAL CALCULATORS (15) ======================
    def compound_interest(self, principal, rate, years, n=12):
        return float(principal * (1 + rate / n) ** (n * years))

    def rule_of_72(self, rate):
        return float(72.0 / (rate * 100)) if rate > 0 else float('inf')

    def present_value(self, future_value, rate, years):
        return float(future_value / (1 + rate) ** years) if (1 + rate) ** years != 0 else 0.0

    def future_value(self, present_val, rate, years):
        return float(present_val * (1 + rate) ** years)

    def net_present_value(self, rate, cashflows):
        return float(sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows)))

    def internal_rate_of_return(self, cashflows, guess=0.1):
        r = guess
        for _ in range(1000):
            npv = sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))
            dnpv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cashflows))
            if abs(dnpv) < 1e-12: break
            r -= npv / dnpv
            if abs(npv) < 1e-8: break
        return float(r)

    def loan_amortization(self, principal, annual_rate, years, ppy=12):
        r = annual_rate / ppy; n = int(years * ppy)
        if r == 0: return {'payment': principal / n, 'total_interest': 0, 'total_paid': principal}
        pmt = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
        total = pmt * n
        return {'payment': float(pmt), 'total_interest': float(total - principal), 'total_paid': float(total), 'num_payments': n}

    def mortgage_calculator(self, home_price, down_pct=0.2, rate=0.065, years=30, tax_rate=0.012, ins_annual=1200):
        down = home_price * down_pct; loan = home_price - down
        amort = self.loan_amortization(loan, rate, years)
        mt = (home_price * tax_rate) / 12; mi = ins_annual / 12
        return {'home_price': float(home_price), 'down_payment': float(down), 'loan_amount': float(loan),
                'monthly_pi': float(amort['payment']), 'monthly_tax': float(mt), 'monthly_insurance': float(mi),
                'monthly_piti': float(amort['payment'] + mt + mi), 'total_interest': float(amort['total_interest'])}

    def savings_goal_calculator(self, goal, current=0, monthly=500, annual_rate=0.07):
        if monthly <= 0: return {'months': float('inf')}
        r = annual_rate / 12; bal = float(current); months = 0
        while bal < goal and months < 1200:
            bal = bal * (1 + r) + monthly; months += 1
        return {'months': months, 'years': round(months / 12, 1), 'final_balance': float(bal),
                'total_contributed': float(current + monthly * months),
                'interest_earned': float(bal - current - monthly * months)}

    def dca_simulator(self, monthly_amount=500, years=10, annual_return=0.08, annual_vol=0.16):
        months = int(years * 12); mr = annual_return / 12; mv = annual_vol / np.sqrt(12)
        np.random.seed(42); rets = np.random.normal(mr, mv, months)
        bal = 0.0; invested = 0.0; bals = []
        for i in range(months):
            bal = (bal + monthly_amount) * (1 + rets[i]); invested += monthly_amount; bals.append(bal)
        return {'final_balance': float(bal), 'total_invested': float(invested),
                'total_gain': float(bal - invested), 'return_pct': float((bal - invested) / invested * 100)}

    def dividend_reinvestment_calc(self, shares, price, div_per_share, years, growth=0.05):
        s = float(shares); p = float(price); d = float(div_per_share); td = 0.0
        for _ in range(int(years)):
            ad = s * d; td += ad; s += ad / p; p *= (1 + growth); d *= (1 + growth * 0.6)
        return {'final_shares': float(s), 'final_price': float(p), 'final_value': float(s * p),
                'total_dividends': float(td), 'total_return_pct': float((s * p - shares * price) / (shares * price) * 100)}

    def withdrawal_rate_sim(self, portfolio_value, annual_withdrawal, years=30, ret=0.07, vol=0.15, n_sims=1000):
        np.random.seed(42); success = 0
        for _ in range(n_sims):
            bal = float(portfolio_value)
            for y in range(years):
                bal = bal * (1 + np.random.normal(ret, vol)) - annual_withdrawal
                if bal <= 0: break
            if bal > 0: success += 1
        return {'withdrawal_rate': float(annual_withdrawal / portfolio_value * 100),
                'success_rate': float(success / n_sims * 100), 'successful_sims': success}

    def inflation_adjusted_return(self, nominal, inflation=0.03):
        return float((1 + nominal) / (1 + inflation) - 1)

    def cost_basis_fifo(self, lots):
        if not lots: return 0.0
        ts = sum(lo['shares'] for lo in lots); tc = sum(lo['shares'] * lo['price'] for lo in lots)
        return float(tc / ts) if ts > 0 else 0.0

    # ====================== OPTIONS & FIXED INCOME (10) ======================
    def black_scholes_call(self, S, K, T, r, sigma):
        from scipy.stats import norm
        if T <= 0 or sigma <= 0: return max(S - K, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    def black_scholes_put(self, S, K, T, r, sigma):
        from scipy.stats import norm
        if T <= 0 or sigma <= 0: return max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    def option_greeks(self, S, K, T, r, sigma, otype='call'):
        from scipy.stats import norm
        if T <= 0 or sigma <= 0: return {'delta': 1 if otype == 'call' else -1, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T); pdf1 = norm.pdf(d1)
        delta = float(norm.cdf(d1)) if otype == 'call' else float(norm.cdf(d1) - 1)
        gamma = float(pdf1 / (S * sigma * np.sqrt(T)))
        vega = float(S * pdf1 * np.sqrt(T) / 100)
        tc = -(S * pdf1 * sigma) / (2 * np.sqrt(T))
        theta = float((tc - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365) if otype == 'call' else float((tc + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)
        rho = float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100) if otype == 'call' else float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

    def implied_volatility_calc(self, mkt_price, S, K, T, r, otype='call'):
        low, high = 0.001, 5.0
        for _ in range(200):
            mid = (low + high) / 2
            price = self.black_scholes_call(S, K, T, r, mid) if otype == 'call' else self.black_scholes_put(S, K, T, r, mid)
            if abs(price - mkt_price) < 0.001: return float(mid)
            if price > mkt_price: high = mid
            else: low = mid
        return float((low + high) / 2)

    def bond_price(self, face, coupon_rate, ytm_years, yield_rate, ppy=2):
        c = face * coupon_rate / ppy; n = int(ytm_years * ppy); y = yield_rate / ppy
        if y == 0: return float(c * n + face)
        return float(c * (1 - (1 + y) ** (-n)) / y + face / (1 + y) ** n)

    def bond_duration(self, face, coupon_rate, ytm_years, yield_rate, ppy=2):
        c = face * coupon_rate / ppy; n = int(ytm_years * ppy); y = yield_rate / ppy
        price = self.bond_price(face, coupon_rate, ytm_years, yield_rate, ppy)
        if price <= 0: return 0.0
        ws = sum((t / ppy) * c / (1 + y) ** t for t in range(1, n + 1))
        ws += ytm_years * face / (1 + y) ** n
        return float(ws / price)

    def bond_modified_duration(self, face, coupon_rate, ytm_years, yield_rate, ppy=2):
        return float(self.bond_duration(face, coupon_rate, ytm_years, yield_rate, ppy) / (1 + yield_rate / ppy))

    def bond_convexity(self, face, coupon_rate, ytm_years, yield_rate, ppy=2):
        c = face * coupon_rate / ppy; n = int(ytm_years * ppy); y = yield_rate / ppy
        price = self.bond_price(face, coupon_rate, ytm_years, yield_rate, ppy)
        if price <= 0: return 0.0
        cs = sum(t * (t + 1) * c / (1 + y) ** (t + 2) for t in range(1, n + 1))
        cs += n * (n + 1) * face / (1 + y) ** (n + 2)
        return float(cs / (price * ppy ** 2))

    def yield_to_maturity(self, face, coupon_rate, ytm_years, mkt_price, ppy=2, guess=0.05):
        ytm = guess
        for _ in range(500):
            p = self.bond_price(face, coupon_rate, ytm_years, ytm, ppy)
            dp = -self.bond_modified_duration(face, coupon_rate, ytm_years, ytm, ppy) * p
            if abs(dp) < 1e-12: break
            ytm -= (p - mkt_price) / dp
            if abs(p - mkt_price) < 0.01: break
        return float(ytm)

    # ====================== ML / SIGNALS / FORECASTING (10) ======================
    def linear_regression_forecast(self, days_ahead=30):
        if self.returns_series.empty: return {'forecast': 0, 'slope': 0, 'r_squared': 0}
        cum = (1 + self.returns_series).cumprod(); y = cum.values; x = np.arange(len(y))
        if len(x) < 10: return {'forecast': 0, 'slope': 0, 'r_squared': 0}
        slope, intercept = np.polyfit(x, y, 1)
        yp = slope * x + intercept; ss_res = np.sum((y - yp) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {'forecast': float(slope * (len(y) + days_ahead) + intercept), 'current': float(y[-1]),
                'slope_daily': float(slope), 'r_squared': float(r2), 'direction': 'up' if slope > 0 else 'down'}

    def moving_average_crossover_signals(self, fast=20, slow=50):
        if self.returns_series.empty or len(self.returns_series) < slow + 5: return {'signal': 'neutral', 'crossovers': []}
        cum = (1 + self.returns_series).cumprod()
        fm = cum.rolling(fast).mean(); sm = cum.rolling(slow).mean()
        cross = fm - sm; curr = 'bullish' if cross.iloc[-1] > 0 else 'bearish'
        ps = np.sign(cross.dropna().values[:-1]); cs = np.sign(cross.dropna().values[1:])
        ci = np.where(ps != cs)[0]; recent = []
        for idx in ci[-5:]:
            dt = cross.dropna().index[idx + 1]
            recent.append({'date': str(dt.date()), 'type': 'golden_cross' if cs[idx] > 0 else 'death_cross'})
        return {'signal': curr, 'fast_ma': float(fm.iloc[-1]), 'slow_ma': float(sm.iloc[-1]), 'crossovers': recent}

    def momentum_signals(self):
        if self.returns_series.empty: return {'score': 0, 'signal': 'neutral'}
        r = self.returns_series; sigs = {}
        for label, days in [('1m', 21), ('3m', 63), ('6m', 126), ('12m', 252)]:
            sigs[label] = 1 if len(r) >= days and float(r.tail(days).sum()) > 0 else -1 if len(r) >= days else 0
        weights = {'1m': 0.15, '3m': 0.25, '6m': 0.3, '12m': 0.3}
        ws = sum(sigs[k] * weights[k] for k in sigs)
        sig = 'strong_buy' if ws > 0.6 else 'buy' if ws > 0.2 else 'strong_sell' if ws < -0.6 else 'sell' if ws < -0.2 else 'neutral'
        return {'score': float(ws), 'signal': sig, 'components': sigs}

    def mean_reversion_signals(self, lookback=60):
        if self.returns_series.empty or len(self.returns_series) < lookback: return {'zscore': 0, 'signal': 'neutral'}
        cum = (1 + self.returns_series).cumprod()
        rm = cum.rolling(lookback).mean().iloc[-1]; rs = cum.rolling(lookback).std().iloc[-1]
        z = float((cum.iloc[-1] - rm) / rs) if rs > 1e-10 else 0
        sig = 'strong_sell' if z > 3 else 'sell' if z > 2 else 'strong_buy' if z < -3 else 'buy' if z < -2 else 'neutral'
        return {'zscore': z, 'signal': sig, 'price_vs_mean': float(cum.iloc[-1] / rm - 1) if rm > 0 else 0}

    def trend_strength_indicator(self):
        if self.returns_series.empty or len(self.returns_series) < 30: return {'strength': 0, 'direction': 'neutral'}
        cum = (1 + self.returns_series).cumprod()
        s10 = cum.rolling(10).mean().iloc[-1]; s30 = cum.rolling(30).mean().iloc[-1]; cur = cum.iloc[-1]
        a10 = cur > s10; a30 = cur > s30
        s10r = s10 > cum.rolling(10).mean().iloc[-2] if len(cum) > 11 else False
        strength = (int(a10) + int(a30) + int(s10r)) / 3
        return {'strength': float(strength), 'direction': 'bullish' if strength > 0.6 else 'bearish' if strength < 0.3 else 'neutral'}

    def volatility_forecast(self, horizon=21):
        if self.returns_series.empty: return 0.0
        return float(self.ewma_volatility() * np.sqrt(horizon / 252))

    def correlation_regime_forecast(self):
        if self.prices.empty or self.prices.ndim < 2 or self.prices.shape[1] < 3:
            return {'regime': 'unknown', 'avg_corr_30d': 0, 'avg_corr_90d': 0}
        rets = self.prices.pct_change().dropna()
        if len(rets) < 90: return {'regime': 'unknown', 'avg_corr_30d': 0, 'avg_corr_90d': 0}
        c30 = rets.tail(30).corr().values; c90 = rets.tail(90).corr().values
        np.fill_diagonal(c30, np.nan); np.fill_diagonal(c90, np.nan)
        a30 = float(np.nanmean(c30)); a90 = float(np.nanmean(c90))
        regime = 'high_correlation' if a30 > 0.6 else 'low_correlation' if a30 < 0.2 else 'normal'
        trend = 'rising' if a30 > a90 + 0.05 else 'falling' if a30 < a90 - 0.05 else 'stable'
        return {'regime': regime, 'trend': trend, 'avg_corr_30d': a30, 'avg_corr_90d': a90}

    def sector_rotation_signal(self):
        sectors = {'XLF': 'Financials', 'XLK': 'Technology', 'XLE': 'Energy', 'XLV': 'Healthcare',
                   'XLI': 'Industrials', 'XLC': 'Communications', 'XLY': 'Cons Disc', 'XLP': 'Cons Staples',
                   'XLU': 'Utilities', 'XLRE': 'Real Estate', 'XLB': 'Materials'}
        results = {}
        try:
            for etf, name in sectors.items():
                df = yf.download(etf, period="6mo", progress=False, auto_adjust=True)
                if df.empty: continue
                c = df['Close']
                if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
                m1 = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else 0
                m3 = float(c.iloc[-1] / c.iloc[-63] - 1) if len(c) > 63 else 0
                results[name] = {'etf': etf, 'mom_1m': m1, 'mom_3m': m3, 'score': m1 * 0.4 + m3 * 0.6}
        except Exception: pass
        if results:
            ranked = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
            return {'top': [(k, v['score']) for k, v in ranked[:3]], 'bottom': [(k, v['score']) for k, v in ranked[-3:]], 'all': results}
        return {'top': [], 'bottom': [], 'all': {}}

    def market_breadth(self):
        if self.prices.empty or self.prices.ndim < 2: return {'breadth': 0, 'above_50sma': 0, 'total': 0}
        above = 0; total = 0
        for col in self.prices.columns:
            s = self.prices[col].dropna()
            if len(s) < 50: continue
            total += 1
            if s.iloc[-1] > s.rolling(50).mean().iloc[-1]: above += 1
        b = above / total if total > 0 else 0
        return {'breadth': float(b), 'above_50sma': above, 'total': total, 'signal': 'bullish' if b > 0.6 else 'bearish' if b < 0.4 else 'neutral'}


    # ====================== ADVANCED DERIVATIVES & EXOTIC OPTIONS (15) ======================
    def binomial_option_price(self, S, K, T, r, sigma, n=100, otype='call'):
        dt = T / n; u = np.exp(sigma * np.sqrt(dt)); d = 1 / u; p = (np.exp(r * dt) - d) / (u - d)
        prices = np.array([S * u ** j * d ** (n - j) for j in range(n + 1)])
        vals = np.maximum(prices - K, 0) if otype == 'call' else np.maximum(K - prices, 0)
        for i in range(n - 1, -1, -1):
            vals[:i + 1] = np.exp(-r * dt) * (p * vals[1:i + 2] + (1 - p) * vals[:i + 1])
        return float(vals[0])

    def american_option_price(self, S, K, T, r, sigma, n=100, otype='put'):
        dt = T / n; u = np.exp(sigma * np.sqrt(dt)); d = 1 / u; p = (np.exp(r * dt) - d) / (u - d)
        prices = np.array([S * u ** j * d ** (n - j) for j in range(n + 1)])
        vals = np.maximum(prices - K, 0) if otype == 'call' else np.maximum(K - prices, 0)
        for i in range(n - 1, -1, -1):
            pi = np.array([S * u ** j * d ** (i - j) for j in range(i + 1)])
            ex = np.maximum(pi - K, 0) if otype == 'call' else np.maximum(K - pi, 0)
            vals[:i + 1] = np.maximum(ex, np.exp(-r * dt) * (p * vals[1:i + 2] + (1 - p) * vals[:i + 1]))
        return float(vals[0])

    def option_parity_check(self, call_price, put_price, S, K, T, r):
        lhs = call_price - put_price; rhs = S - K * np.exp(-r * T)
        return {'lhs': float(lhs), 'rhs': float(rhs), 'diff': float(abs(lhs - rhs)), 'parity_holds': abs(lhs - rhs) < 0.50}

    def garman_kohlhagen_fx(self, S, K, T, rd, rf, sigma, otype='call'):
        from scipy.stats import norm
        d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if otype == 'call': return float(S * np.exp(-rf * T) * norm.cdf(d1) - K * np.exp(-rd * T) * norm.cdf(d2))
        return float(K * np.exp(-rd * T) * norm.cdf(-d2) - S * np.exp(-rf * T) * norm.cdf(-d1))

    def option_profit_loss(self, S, K, premium, otype='call', position='long'):
        if otype == 'call': pnl = max(S - K, 0) - premium if position == 'long' else premium - max(S - K, 0)
        else: pnl = max(K - S, 0) - premium if position == 'long' else premium - max(K - S, 0)
        return {'pnl': float(pnl), 'breakeven': float(K + premium) if otype == 'call' else float(K - premium)}

    def straddle_payoff(self, S, K, call_prem, put_prem):
        pnl = max(S - K, 0) + max(K - S, 0) - call_prem - put_prem
        return {'pnl': float(pnl), 'upper_be': float(K + call_prem + put_prem), 'lower_be': float(K - call_prem - put_prem)}

    def strangle_payoff(self, S, K_call, K_put, call_prem, put_prem):
        pnl = max(S - K_call, 0) + max(K_put - S, 0) - call_prem - put_prem
        return {'pnl': float(pnl), 'upper_be': float(K_call + call_prem + put_prem), 'lower_be': float(K_put - call_prem - put_prem)}

    def iron_condor_payoff(self, S, K1, K2, K3, K4, net_credit):
        pnl = net_credit - max(max(K2 - S, 0) - max(K1 - S, 0), 0) - max(max(S - K3, 0) - max(S - K4, 0), 0)
        return {'pnl': float(pnl), 'max_profit': float(net_credit), 'max_loss': float(net_credit - (K2 - K1))}

    def covered_call_return(self, stock_price, strike, premium, cost_basis):
        ret = (min(stock_price, strike) + premium - cost_basis) / cost_basis
        return {'return_pct': float(ret * 100), 'max_return_pct': float((strike + premium - cost_basis) / cost_basis * 100)}

    def protective_put_cost(self, stock_price, put_strike, put_premium):
        return {'cost': float(put_premium), 'max_loss': float(stock_price - put_strike + put_premium), 'breakeven': float(stock_price + put_premium)}

    def butterfly_spread_payoff(self, S, K1, K2, K3, net_debit):
        pnl = max(S - K1, 0) - 2 * max(S - K2, 0) + max(S - K3, 0) - net_debit
        return {'pnl': float(pnl), 'max_profit': float(K2 - K1 - net_debit), 'max_loss': float(-net_debit)}

    def option_probability_itm(self, S, K, T, r, sigma, otype='call'):
        from scipy.stats import norm
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return float(norm.cdf(d2)) if otype == 'call' else float(norm.cdf(-d2))

    def option_expected_move(self, S, sigma, T):
        m = S * sigma * np.sqrt(T)
        return {'expected_move': float(m), 'upper_1sd': float(S + m), 'lower_1sd': float(S - m)}

    def option_decay_profile(self, S, K, T, r, sigma, otype='call'):
        return [{'days': d, 'price': float(self.black_scholes_call(S, K, d/365, r, sigma) if otype == 'call' else self.black_scholes_put(S, K, d/365, r, sigma))} for d in [30, 21, 14, 7, 3, 1]]

    # ====================== ADVANCED STATISTICS & ECONOMETRICS (20) ======================
    def augmented_dickey_fuller(self):
        if self.returns_series.empty or len(self.returns_series) < 50: return {'stationary': False, 'adf_stat': 0}
        r = self.returns_series.dropna().values; dy = np.diff(r); x = r[:-1]
        if len(dy) < 30: return {'stationary': False, 'adf_stat': 0}
        slope = np.polyfit(x[:len(dy)], dy, 1)[0]; se = np.std(dy) / (np.std(x[:len(dy)]) * np.sqrt(len(dy)))
        t_stat = slope / se if se > 1e-10 else 0
        return {'adf_stat': float(t_stat), 'stationary': t_stat < -2.86}

    def granger_causality_simple(self, sym_a, sym_b, lags=5):
        if self.prices.empty or self.prices.ndim < 2: return {'causal': False}
        if sym_a not in self.prices.columns or sym_b not in self.prices.columns: return {'causal': False}
        ra = self.prices[sym_a].pct_change().dropna(); rb = self.prices[sym_b].pct_change().dropna()
        common = ra.index.intersection(rb.index)
        if len(common) < lags + 30: return {'causal': False}
        a = ra.loc[common].values; b = rb.loc[common].values
        y = b[lags:]; X_r = np.column_stack([b[lags-i-1:len(b)-i-1] for i in range(lags)])
        X_u = np.column_stack([X_r] + [a[lags-i-1:len(a)-i-1] for i in range(lags)])
        rss_r = np.sum((y - X_r @ np.linalg.lstsq(X_r, y, rcond=None)[0]) ** 2)
        rss_u = np.sum((y - X_u @ np.linalg.lstsq(X_u, y, rcond=None)[0]) ** 2)
        n = len(y); f = ((rss_r - rss_u) / lags) / (rss_u / (n - 2 * lags)) if rss_u > 0 else 0
        return {'f_stat': float(f), 'causal': f > 2.5, 'direction': f'{sym_a} -> {sym_b}'}

    def variance_ratio_test(self, k=5):
        if self.returns_series.empty or len(self.returns_series) < k * 10: return {'vr': 1.0, 'random_walk': True}
        r = self.returns_series.dropna().values; n = len(r)
        var1 = np.var(r, ddof=1); rk = np.array([sum(r[i:i+k]) for i in range(n - k + 1)])
        vr = np.var(rk, ddof=1) / (k * var1) if var1 > 0 else 1
        return {'variance_ratio': float(vr), 'random_walk': abs(vr - 1) < 0.2}

    def ljung_box_test(self, lags=10):
        if self.returns_series.empty or len(self.returns_series) < lags + 10: return {'lb_stat': 0, 'autocorrelated': False}
        r = self.returns_series.dropna(); n = len(r)
        ac = [float(r.autocorr(lag=i)) for i in range(1, lags + 1)]
        lb = n * (n + 2) * sum(a ** 2 / (n - i - 1) for i, a in enumerate(ac))
        from scipy.stats import chi2
        p = 1 - chi2.cdf(lb, lags)
        return {'lb_stat': float(lb), 'p_value': float(p), 'autocorrelated': p < 0.05}

    def runs_test(self):
        if self.returns_series.empty or len(self.returns_series) < 20: return {'random': True}
        r = self.returns_series.dropna().values; signs = np.sign(r); signs = signs[signs != 0]
        n = len(signs); n_pos = np.sum(signs > 0); n_neg = n - n_pos
        runs = 1 + np.sum(signs[1:] != signs[:-1])
        mu = 1 + 2 * n_pos * n_neg / n; var = (mu - 1) * (mu - 2) / (n - 1) if n > 1 else 1
        z = (runs - mu) / np.sqrt(var) if var > 0 else 0
        return {'runs': int(runs), 'z_stat': float(z), 'random': abs(z) < 1.96}

    def copula_tail_dependence(self):
        if self.prices.empty or self.prices.ndim < 2 or self.prices.shape[1] < 2: return {}
        rets = self.prices.pct_change().dropna(); result = {}
        cols = list(rets.columns)[:5]
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                a = rets[cols[i]].values; b = rets[cols[j]].values
                q10 = np.percentile(a, 10); both = np.mean((a <= q10) & (b <= np.percentile(b, 10)))
                result[f'{cols[i]}_{cols[j]}'] = float(both / 0.10) if 0.10 > 0 else 0
        return result

    def rolling_beta(self, benchmark="SPY", window=60):
        if self.returns_series.empty: return pd.Series(dtype=float)
        try:
            bm = yf.download(benchmark, period="5y", progress=False, auto_adjust=True)['Close']
            if isinstance(bm, pd.DataFrame): bm = bm.iloc[:, 0]
            bm_ret = bm.pct_change().dropna()
        except: return pd.Series(dtype=float)
        common = self.returns_series.index.intersection(bm_ret.index)
        if len(common) < window: return pd.Series(dtype=float)
        pr = self.returns_series.loc[common]; br = bm_ret.loc[common]
        return (pr.rolling(window).cov(br) / br.rolling(window).var()).dropna()

    def rolling_alpha(self, benchmark="SPY", window=60, rf=0.04):
        rb = self.rolling_beta(benchmark, window)
        if rb.empty: return pd.Series(dtype=float)
        try:
            bm = yf.download(benchmark, period="5y", progress=False, auto_adjust=True)['Close']
            if isinstance(bm, pd.DataFrame): bm = bm.iloc[:, 0]
            bm_ret = bm.pct_change().dropna()
        except: return pd.Series(dtype=float)
        common = rb.index.intersection(bm_ret.index).intersection(self.returns_series.index)
        pr = self.returns_series.loc[common]; br = bm_ret.loc[common]; beta = rb.loc[common]
        return ((pr.rolling(window).mean() - rf/252) - beta * (br.rolling(window).mean() - rf/252)) * 252

    def conditional_skewness(self, threshold=0.0):
        if self.returns_series.empty: return {'up_skew': 0, 'down_skew': 0}
        r = self.returns_series.dropna()
        up = r[r > threshold]; down = r[r <= threshold]
        return {'up_skew': float(up.skew()) if len(up) > 10 else 0, 'down_skew': float(down.skew()) if len(down) > 10 else 0}

    def drawdown_at_risk(self, confidence=0.95):
        if self.returns_series.empty: return 0.0
        cum = (1 + self.returns_series).cumprod(); dd = (cum / cum.cummax() - 1).dropna()
        return float(-np.percentile(dd, (1 - confidence) * 100))

    def pain_index(self):
        if self.returns_series.empty: return 0.0
        cum = (1 + self.returns_series).cumprod()
        return float(abs(cum / cum.cummax() - 1).mean())

    def pain_ratio(self, rf=0.04):
        if self.returns_series.empty: return 0.0
        pi = self.pain_index()
        return float((self.returns_series.mean() * 252 - rf) / pi) if pi > 1e-10 else 0.0

    def gain_loss_ratio(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); g = r[r > 0]; l = r[r < 0]
        return float(g.mean() / abs(l.mean())) if len(l) > 0 and len(g) > 0 else 0.0

    def profit_loss_ratio(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); w = r[r > 0]; l = r[r < 0]
        return float(w.sum() / abs(l.sum())) if len(l) > 0 and len(w) > 0 else 0.0

    def common_sense_ratio(self):
        return float(self.tail_ratio() * self.gain_loss_ratio())

    def pessimistic_return(self, confidence=0.95):
        if self.returns_series.empty: return 0.0
        from scipy.stats import norm
        r = self.returns_series.dropna()
        return float(r.mean() - norm.ppf(confidence) * r.std() / np.sqrt(len(r)))

    def probabilistic_sharpe(self, sr_benchmark=0.0):
        if self.returns_series.empty or len(self.returns_series) < 30: return 0.5
        from scipy.stats import norm
        r = self.returns_series.dropna(); n = len(r); sr = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
        se = np.sqrt((1 + 0.5 * sr**2 - float(r.skew()) * sr + (float(r.kurtosis()) / 4) * sr**2) / (n - 1))
        return float(norm.cdf((sr - sr_benchmark) / se)) if se > 1e-10 else 0.5

    def deflated_sharpe(self, n_trials=10):
        from scipy.stats import norm
        psr = self.probabilistic_sharpe()
        emax = norm.ppf(1 - 1/n_trials) * np.sqrt(1/max(len(self.returns_series), 1))
        return float(psr - emax)

    def minimum_track_record(self, sr_benchmark=0.0):
        if self.returns_series.empty or len(self.returns_series) < 10: return float('inf')
        r = self.returns_series.dropna(); sr = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
        num = (1 + 0.5*sr**2) * 1.96**2; den = (sr - sr_benchmark)**2
        return float(num / den) if den > 1e-10 else float('inf')

    # ====================== MACRO & FUNDAMENTAL ANALYSIS (15) ======================
    def pe_ratio_calc(self, price, eps):
        return float(price / eps) if eps > 0 else float('inf')

    def pb_ratio_calc(self, price, bv):
        return float(price / bv) if bv > 0 else float('inf')

    def ev_ebitda_calc(self, mcap, debt, cash, ebitda):
        return float((mcap + debt - cash) / ebitda) if ebitda > 0 else float('inf')

    def dividend_yield_calc(self, annual_div, price):
        return float(annual_div / price * 100) if price > 0 else 0.0

    def peg_ratio_calc(self, pe, growth):
        return float(pe / (growth * 100)) if growth > 0 else float('inf')

    def dcf_valuation(self, fcf, gr=0.05, dr=0.10, tg=0.02, yrs=10):
        pv = 0.0; cf = float(fcf)
        for t in range(1, yrs + 1): cf *= (1 + gr); pv += cf / (1 + dr) ** t
        tv = cf * (1 + tg) / (dr - tg); pv += tv / (1 + dr) ** yrs
        return {'intrinsic_value': float(pv), 'terminal_value': float(tv)}

    def wacc_calc(self, equity, debt, ce, cd, tax=0.21):
        t = equity + debt
        return float((equity/t)*ce + (debt/t)*cd*(1-tax)) if t > 0 else 0.0

    def altman_z_score(self, wc_ta, re_ta, ebit_ta, mve_tl, sales_ta):
        z = 1.2*wc_ta + 1.4*re_ta + 3.3*ebit_ta + 0.6*mve_tl + 1.0*sales_ta
        return {'z_score': float(z), 'zone': 'safe' if z > 2.99 else 'grey' if z > 1.81 else 'distress'}

    def piotroski_f_score(self, roa, cfo, droa, accrual, dlev, dliq, eq, dgm, dato):
        s = int(roa>0)+int(cfo>0)+int(droa>0)+int(cfo>roa)+int(dlev<0)+int(dliq>0)+int(eq<=0)+int(dgm>0)+int(dato>0)
        return {'score': s, 'signal': 'strong' if s >= 7 else 'weak' if s <= 3 else 'neutral'}

    def dupont_analysis(self, ni, rev, assets, equity):
        npm = ni/rev if rev > 0 else 0; at = rev/assets if assets > 0 else 0; em = assets/equity if equity > 0 else 0
        return {'npm': float(npm), 'asset_turnover': float(at), 'equity_mult': float(em), 'roe': float(npm*at*em)}

    def gordon_growth_model(self, div, growth, rr):
        return float(div*(1+growth)/(rr-growth)) if rr > growth else float('inf')

    def earnings_yield(self, eps, price):
        return float(eps / price * 100) if price > 0 else 0.0

    def free_cash_flow_yield(self, fcf, mcap):
        return float(fcf / mcap * 100) if mcap > 0 else 0.0

    def enterprise_value_calc(self, mcap, debt, cash, mi=0):
        return float(mcap + debt - cash + mi)

    def magic_formula_rank(self, ey, roc):
        return {'ey_rank': float(ey), 'roc_rank': float(roc), 'combined': float(ey + roc)}

    # ====================== ADVANCED PORTFOLIO OPTIMIZATION (15) ======================
    def equal_risk_contribution(self):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); cov = rets.cov().values * 252; n = cov.shape[0]
        w = np.ones(n) / n
        for _ in range(100):
            rc = w * (cov @ w); tr = w @ cov @ w; tgt = tr / n
            w = w - 0.01 * 2 * (rc - tgt); w = np.maximum(w, 0.001); w = w / w.sum()
        return {rets.columns[i]: float(w[i]) for i in range(n)}

    def black_litterman_weights(self, tau=0.05):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); n = rets.shape[1]
        w = np.ones(n) / n
        return {rets.columns[i]: float(w[i]) for i in range(n)}

    def kelly_criterion(self, win_rate, avg_win, avg_loss):
        if avg_loss == 0: return 0.0
        b = avg_win / abs(avg_loss)
        return float((win_rate * b - (1 - win_rate)) / b) if b > 0 else 0.0

    def kelly_criterion_portfolio(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); w = r[r > 0]; l = r[r < 0]
        if len(w) == 0 or len(l) == 0: return 0.0
        return self.kelly_criterion(len(w)/len(r), float(w.mean()), float(l.mean()))

    def target_return_weights(self, target=0.10):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); mu = rets.mean().values * 252; cov = rets.cov().values * 252; n = cov.shape[0]
        try: ic = np.linalg.inv(cov)
        except: ic = np.linalg.pinv(cov)
        ones = np.ones(n); a = mu@ic@mu; b = mu@ic@ones; c = ones@ic@ones
        lam = (c*target-b)/(a*c-b**2); gam = (a-b*target)/(a*c-b**2)
        w = lam*ic@mu + gam*ic@ones
        return {rets.columns[i]: float(w[i]) for i in range(n)}

    def max_sharpe_weights(self, rf=0.04):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); mu = rets.mean().values*252-rf; cov = rets.cov().values*252
        try: ic = np.linalg.inv(cov)
        except: ic = np.linalg.pinv(cov)
        w = ic@mu; w = w/w.sum()
        return {rets.columns[i]: float(w[i]) for i in range(len(mu))}

    def min_cvar_weights(self, confidence=0.95):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); n = rets.shape[1]; w = np.ones(n)/n
        for _ in range(50):
            pr = rets.values@w; vt = np.percentile(pr, (1-confidence)*100)
            tail = pr[pr <= vt]; grad = np.zeros(n)
            for j in range(n): grad[j] = rets.values[pr<=vt, j].mean() if len(tail) > 0 else 0
            w = w - 0.01*grad; w = np.maximum(w, 0); w = w/w.sum()
        return {rets.columns[i]: float(w[i]) for i in range(n)}

    def inverse_volatility_weights(self):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); v = rets.std(); v = v[v>0]
        w = (1/v)/(1/v).sum()
        return {s: float(w[s]) for s in w.index}

    def momentum_weights(self, lookback=126):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna()
        if len(rets) < lookback: return {}
        mom = rets.tail(lookback).sum(); mp = mom[mom > 0]
        if mp.empty: return {s: 1.0/len(rets.columns) for s in rets.columns}
        w = mp / mp.sum()
        return {s: float(w[s]) for s in w.index}

    def volatility_targeting_weights(self, target_vol=0.10):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); v = rets.std()*np.sqrt(252); v = v[v>0]
        raw = target_vol/v; w = raw/raw.sum()
        return {s: float(w[s]) for s in w.index}

    def tail_risk_parity(self):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); cv = {}
        for c in rets.columns:
            r = rets[c].dropna(); v = np.percentile(r, 5)
            cv[c] = abs(float(r[r<=v].mean())) if len(r[r<=v])>0 else 0.001
        inv = {k: 1/v for k,v in cv.items() if v>0}; t = sum(inv.values())
        return {k: float(v/t) for k,v in inv.items()}

    def diversification_ratio(self):
        if self.prices.empty or self.prices.ndim < 2: return 0.0
        rets = self.prices.pct_change().dropna(); v = rets.std().values*np.sqrt(252)
        cov = rets.cov().values*252; n = len(v); w = np.ones(n)/n
        return float(np.dot(w,v)/np.sqrt(w@cov@w)) if np.sqrt(w@cov@w) > 0 else 1.0

    def portfolio_turnover_calc(self, old_w, new_w):
        keys = set(list(old_w.keys())+list(new_w.keys()))
        return float(sum(abs(new_w.get(k,0)-old_w.get(k,0)) for k in keys)/2)

    def rebalance_trades(self, cw, tw, pv):
        trades = {}; keys = set(list(cw.keys())+list(tw.keys()))
        for k in keys:
            d = tw.get(k,0)-cw.get(k,0); amt = d*pv
            if abs(amt)>10: trades[k] = {'weight_change': float(d), 'amount': float(amt), 'action': 'BUY' if amt>0 else 'SELL'}
        return trades

    # ====================== RISK DECOMPOSITION & STRESS TESTING (20) ======================
    def historical_stress_test(self, scenario='2008_crisis'):
        sc = {'2008_crisis':-0.38,'2020_covid':-0.34,'dotcom_bust':-0.45,'2022_bear':-0.25,'flash_crash':-0.09,'black_monday':-0.22}
        shock = sc.get(scenario, -0.20)
        if self.returns_series.empty: return {'scenario': scenario, 'impact': 0}
        vr = float(self.returns_series.std()*np.sqrt(252))/0.16
        return {'scenario': scenario, 'market_drop': float(shock), 'portfolio_impact': float(shock*vr)}

    def parametric_stress_test(self, eq_shock=0, rate_shock=0, vol_shock=0):
        beta=1.0; dur=5.0; vega=0.1
        impact = beta*eq_shock + dur*rate_shock/100 + vega*vol_shock
        return {'equity': float(beta*eq_shock), 'rate': float(dur*rate_shock/100), 'vol': float(vega*vol_shock), 'total': float(impact)}

    def correlation_stress_test(self, shock_corr=0.9):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); v = rets.std().values*np.sqrt(252); n = len(v)
        ncov = rets.cov().values*252; w = np.ones(n)/n; nv = np.sqrt(w@ncov@w)
        scov = np.full((n,n), shock_corr); np.fill_diagonal(scov, 1.0); scov = scov*np.outer(v,v)
        sv = np.sqrt(w@scov@w)
        return {'normal_vol': float(nv), 'stress_vol': float(sv), 'increase': float(sv/nv-1)}

    def factor_risk_decomp(self, n_factors=3):
        if self.prices.empty or self.prices.ndim < 2 or self.prices.shape[1] < n_factors: return {}
        rets = self.prices.pct_change().dropna(); cov = rets.cov().values
        ev, _ = np.linalg.eigh(cov); idx = np.argsort(ev)[::-1]; exp = ev[idx]/ev.sum()
        return {f'factor_{i+1}': float(exp[i]) for i in range(min(n_factors, len(exp)))}

    def tail_risk_contribution(self, confidence=0.95):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); n = rets.shape[1]; w = np.ones(n)/n
        pr = rets.values@w; th = np.percentile(pr, (1-confidence)*100); mask = pr<=th
        return {rets.columns[i]: float(rets.iloc[:,i].values[mask].mean()*w[i]) if mask.sum()>0 else 0 for i in range(n)}

    def risk_budget_analysis(self):
        if self.prices.empty or self.prices.ndim < 2: return {}
        rets = self.prices.pct_change().dropna(); cov = rets.cov().values*252; n = cov.shape[0]
        w = np.ones(n)/n; pv = w@cov@w; mcr = cov@w; rc = w*mcr
        return {rets.columns[i]: {'weight': float(w[i]), 'risk_contrib': float(rc[i]), 'risk_share': float(rc[i]/pv) if pv>0 else 0} for i in range(n)}

    def expected_tail_loss(self, confidence=0.99):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); q = np.percentile(r, (1-confidence)*100)
        t = r[r<=q]; return float(-t.mean()) if len(t)>0 else 0.0

    def drawdown_distribution(self):
        if self.returns_series.empty: return {}
        cum = (1+self.returns_series).cumprod(); dd = (cum/cum.cummax()-1).dropna()
        return {'mean': float(dd.mean()), 'std': float(dd.std()), 'max': float(dd.min()), 'p5': float(np.percentile(dd,5)), 'p95': float(np.percentile(dd,95))}

    def time_under_water(self):
        if self.returns_series.empty: return {'pct_underwater': 0}
        cum = (1+self.returns_series).cumprod(); rm = cum.cummax(); uw = cum < rm
        pct = float(uw.mean()); groups = (~uw).cumsum(); uwg = groups[uw]
        if uwg.empty: return {'pct_underwater': float(pct), 'avg_days': 0, 'max_days': 0}
        dur = uwg.groupby(uwg).size()
        return {'pct_underwater': float(pct), 'avg_days': float(dur.mean()), 'max_days': int(dur.max())}

    def conditional_drawdown_at_risk(self, confidence=0.95):
        if self.returns_series.empty: return 0.0
        cum = (1+self.returns_series).cumprod(); dd = abs(cum/cum.cummax()-1)
        th = np.percentile(dd, confidence*100); t = dd[dd>=th]
        return float(t.mean()) if len(t)>0 else 0.0

    def portfolio_beta_decomp(self, benchmark="SPY"):
        if self.prices.empty or self.prices.ndim < 2: return {}
        try:
            bm = yf.download(benchmark, period="5y", progress=False, auto_adjust=True)['Close']
            if isinstance(bm, pd.DataFrame): bm = bm.iloc[:,0]
            br = bm.pct_change().dropna()
        except: return {}
        rets = self.prices.pct_change().dropna(); result = {}
        for c in rets.columns:
            common = rets[c].dropna().index.intersection(br.index)
            if len(common)<30: result[c]=0; continue
            result[c] = float(rets[c].loc[common].cov(br.loc[common])/br.loc[common].var())
        return result

    def systematic_vs_specific_risk(self, benchmark="SPY"):
        betas = self.portfolio_beta_decomp(benchmark)
        if not betas: return {'systematic': 0, 'specific': 0}
        ab = np.mean(list(betas.values()))
        if self.returns_series.empty: return {'systematic': 0, 'specific': 0}
        tv = float(self.returns_series.var()*252); sv = ab**2*0.04; sp = max(tv-sv, 0)
        return {'total': float(np.sqrt(tv)), 'systematic': float(np.sqrt(sv)), 'specific': float(np.sqrt(sp))}

    def scenario_analysis(self, scenarios=None):
        if scenarios is None: scenarios = {'base':0.08,'bull':0.20,'bear':-0.15,'recession':-0.30,'stag':-0.10}
        pv = max(self.total_value, 100000)
        return {n: {'return': float(r), 'value': float(pv*(1+r)), 'pnl': float(pv*r)} for n,r in scenarios.items()}

    def liquidity_risk_score(self):
        if self.holdings.empty: return 0.0
        n = len(self.holdings); score = min(1.0, n/20)*0.3
        if not self.prices.empty and self.prices.ndim == 2:
            av = 0; cnt = 0
            for c in self.prices.columns:
                v = self.prices[c].pct_change().std()
                if v and not np.isnan(v): av += v; cnt += 1
            if cnt > 0: score += (1-min(av/cnt*np.sqrt(252),1))*0.7
        return float(np.clip(score, 0, 1))

    def concentration_risk_report(self):
        if self.holdings.empty or 'quantity' not in self.holdings.columns: return {}
        hhi = self.portfolio_concentration_hhi(); enb = self.effective_num_bets()
        t5 = float(self.holdings.nlargest(5,'quantity')['quantity'].sum()/self.holdings['quantity'].sum()) if self.holdings['quantity'].sum()>0 else 0
        return {'hhi': hhi, 'effective_bets': enb, 'top5': t5, 'level': 'high' if hhi>0.3 else 'moderate' if hhi>0.15 else 'low'}

    def var_backtest(self, confidence=0.95, window=252):
        if self.returns_series.empty or len(self.returns_series) < window+50: return {'breaches': 0}
        r = self.returns_series.dropna(); breaches = 0; total = 0
        for i in range(window, len(r)):
            var = float(-np.percentile(r.iloc[i-window:i], (1-confidence)*100))
            if float(r.iloc[i]) < -var: breaches += 1
            total += 1
        return {'breaches': breaches, 'total': total, 'rate': float(breaches/total) if total>0 else 0, 'expected': float(1-confidence)}

    def extreme_value_var(self, confidence=0.99):
        if self.returns_series.empty or len(self.returns_series) < 100: return 0.0
        r = self.returns_series.dropna().values; losses = -r[r<0]
        if len(losses)<20: return 0.0
        u = np.percentile(losses, 90); ex = losses[losses>u]-u
        if len(ex)<5: return float(np.percentile(losses, confidence*100))
        beta = float(np.mean(ex)); xi = 0.1; n = len(losses); nu = len(ex)
        return float(u + (beta/xi)*((n/nu*(1-confidence))**(-xi)-1))

    def portfolio_resilience_score(self):
        if self.returns_series.empty: return 0.0
        scores = []
        sr = float(self.returns_series.mean()/self.returns_series.std()*np.sqrt(252)) if self.returns_series.std()>0 else 0
        scores.append(min(max(sr/2,0),1)*25)
        cum = (1+self.returns_series).cumprod(); mdd = abs(float((cum/cum.cummax()-1).min()))
        scores.append(max(1-mdd/0.5,0)*25)
        rf = self.recovery_factor(); scores.append(min(rf/3,1)*25)
        cr = self.calmar_ratio_custom(); scores.append(min(max(cr,0)/2,1)*25)
        return float(sum(scores))

    # ====================== TIME SERIES & FORECASTING ADVANCED (15) ======================
    def exponential_smoothing_forecast(self, alpha=0.3, horizon=30):
        if self.returns_series.empty: return []
        cum = (1+self.returns_series).cumprod().values; level = cum[0]
        for v in cum: level = alpha*v + (1-alpha)*level
        return [{'day': i+1, 'forecast': float(level)} for i in range(horizon)]

    def double_exponential_smoothing(self, alpha=0.3, beta=0.1, horizon=30):
        if self.returns_series.empty: return []
        cum = (1+self.returns_series).cumprod().values
        level = cum[0]; trend = cum[1]-cum[0] if len(cum)>1 else 0
        for v in cum[1:]:
            nl = alpha*v+(1-alpha)*(level+trend); trend = beta*(nl-level)+(1-beta)*trend; level = nl
        return [{'day': i+1, 'forecast': float(level+(i+1)*trend)} for i in range(horizon)]

    def triple_exponential_smoothing(self, alpha=0.3, beta=0.1, gamma=0.1, season=21, horizon=30):
        if self.returns_series.empty or len(self.returns_series) < season*2: return []
        cum = (1+self.returns_series).cumprod().values; n = len(cum)
        level = np.mean(cum[:season]); trend = (np.mean(cum[season:2*season])-np.mean(cum[:season]))/season
        seas = [cum[i]-np.mean(cum[:season]) for i in range(season)]
        for i in range(season, n):
            si = i%season; nl = alpha*(cum[i]-seas[si])+(1-alpha)*(level+trend)
            trend = beta*(nl-level)+(1-beta)*trend; seas[si] = gamma*(cum[i]-nl)+(1-gamma)*seas[si]; level = nl
        return [{'day': i+1, 'forecast': float(level+(i+1)*trend+seas[(n+i)%season])} for i in range(horizon)]

    def ar_model_forecast(self, order=5, horizon=10):
        if self.returns_series.empty or len(self.returns_series) < order+20: return []
        r = self.returns_series.dropna().values; n = len(r)
        X = np.column_stack([r[order-i-1:n-i-1] for i in range(order)]); y = r[order:]
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        last = list(r[-order:]); fc = []
        for i in range(horizon):
            pred = sum(coeffs[j]*last[-(j+1)] for j in range(order))
            fc.append({'day': i+1, 'forecast': float(pred)}); last.append(pred)
        return fc

    def seasonal_decomposition(self, period=21):
        if self.returns_series.empty or len(self.returns_series) < period*3: return {}
        cum = (1+self.returns_series).cumprod(); tr = cum.rolling(period, center=True).mean()
        dt = cum-tr; seas = dt.groupby(dt.index.dayofweek).transform('mean'); res = cum-tr-seas
        return {'trend': float(tr.dropna().iloc[-1]) if not tr.dropna().empty else 0, 'seasonal': float(seas.dropna().iloc[-1]) if not seas.dropna().empty else 0, 'residual_std': float(res.dropna().std()) if not res.dropna().empty else 0}

    def walk_forward_backtest(self, train=252, test=21):
        if self.returns_series.empty or len(self.returns_series) < train+test*3: return {}
        r = self.returns_series.dropna().values; results = []
        for s in range(0, len(r)-train-test, test):
            pm = np.mean(r[s:s+train]); am = np.mean(r[s+train:s+train+test])
            results.append({'pred': float(pm), 'actual': float(am), 'error': float(abs(pm-am))})
        return {'n_folds': len(results), 'avg_error': float(np.mean([x['error'] for x in results])) if results else 0}

    def regime_switching_forecast(self):
        if self.returns_series.empty or len(self.returns_series) < 126: return {'regime': 'unknown'}
        r = self.returns_series.dropna()
        vs = float(r.tail(21).std()*np.sqrt(252)); vl = float(r.tail(63).std()*np.sqrt(252))
        ms = float(r.tail(21).mean()*252)
        regime = 'high_vol' if vs>vl*1.5 else 'low_vol_bull' if ms>0.15 and vs<0.15 else 'crisis' if ms<-0.10 else 'normal'
        fc = {'high_vol':-0.05,'low_vol_bull':0.12,'crisis':-0.20,'normal':0.08}
        return {'regime': regime, 'forecast': fc.get(regime,0), 'vol_short': vs, 'vol_long': vl}

    def return_forecast_ensemble(self, horizon=21):
        if self.returns_series.empty: return {'forecast': 0}
        lin = self.linear_regression_forecast(horizon); mom = self.momentum_signals()
        fl = lin.get('slope_daily',0)*horizon
        fm = 0.05 if mom.get('signal') in ['buy','strong_buy'] else -0.05 if mom.get('signal') in ['sell','strong_sell'] else 0
        return {'linear': float(fl), 'momentum': float(fm), 'ensemble': float((fl+fm)/2)}

    def volatility_regime_forecast(self):
        if self.returns_series.empty or len(self.returns_series) < 63: return {'regime': 'unknown'}
        r = self.returns_series.dropna()
        v5=float(r.tail(5).std()*np.sqrt(252)); v21=float(r.tail(21).std()*np.sqrt(252)); v63=float(r.tail(63).std()*np.sqrt(252))
        regime = 'expanding' if v5>v21>v63 else 'contracting' if v5<v21<v63 else 'mixed'
        return {'regime': regime, 'vol_5d': v5, 'vol_21d': v21, 'vol_63d': v63}

    def mean_absolute_deviation(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); return float(np.abs(r-r.mean()).mean())

    def coefficient_of_variation(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna()
        return float(r.std()/abs(r.mean())) if abs(r.mean())>1e-10 else 0.0

    def interquartile_range(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); return float(np.percentile(r,75)-np.percentile(r,25))

    def median_absolute_deviation(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); return float(np.median(np.abs(r-np.median(r))))

    def gini_coefficient(self):
        if self.returns_series.empty: return 0.0
        r = np.sort(self.returns_series.dropna().values); n = len(r)
        if n == 0 or np.sum(r) == 0: return 0.0
        idx = np.arange(1, n+1); return float((2*np.sum(idx*r)/(n*np.sum(r))-(n+1)/n))

    # ====================== TRADE ANALYTICS & EXECUTION (15) ======================
    def trade_expectancy(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); w = r[r>0]; l = r[r<0]
        wr = len(w)/len(r) if len(r)>0 else 0
        return float(wr*float(w.mean() if len(w)>0 else 0)+(1-wr)*float(l.mean() if len(l)>0 else 0))

    def trade_payoff_ratio(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); w = r[r>0]; l = r[r<0]
        return float(w.mean()/abs(l.mean())) if len(l)>0 and len(w)>0 else 0.0

    def consecutive_wins_losses(self):
        if self.returns_series.empty: return {'max_wins': 0, 'max_losses': 0}
        signs = np.sign(self.returns_series.dropna().values); mw=0;ml=0;cw=0;cl=0
        for s in signs:
            if s>0: cw+=1;cl=0;mw=max(mw,cw)
            elif s<0: cl+=1;cw=0;ml=max(ml,cl)
            else: cw=0;cl=0
        return {'max_wins': mw, 'max_losses': ml}

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

    def rolling_win_rate(self, window=63):
        if self.returns_series.empty or len(self.returns_series)<window: return pd.Series(dtype=float)
        return self.returns_series.dropna().rolling(window).apply(lambda x: (x>0).sum()/len(x))

    def profit_factor_rolling(self, window=63):
        if self.returns_series.empty or len(self.returns_series)<window: return pd.Series(dtype=float)
        def pf(x):
            g=x[x>0].sum();l=abs(x[x<0].sum()); return g/l if l>0 else float('inf') if g>0 else 0
        return self.returns_series.dropna().rolling(window).apply(pf)

    def risk_adjusted_return_by_period(self):
        if self.returns_series.empty: return {}
        r = self.returns_series.dropna(); result = {}
        for lab, d in [('1M',21),('3M',63),('6M',126),('1Y',252)]:
            if len(r)<d: continue
            s = r.tail(d); ret=float(s.sum()); vol=float(s.std()*np.sqrt(252))
            result[lab] = {'return': ret, 'vol': vol, 'sharpe': float(ret/vol) if vol>0 else 0}
        return result

    def trade_duration_stats(self):
        if self.master_df.empty or 'date' not in self.master_df.columns: return {}
        diffs = self.master_df.sort_values('date')['date'].diff().dropna()
        if diffs.empty: return {}
        d = diffs.dt.days
        return {'avg': float(d.mean()), 'median': float(d.median()), 'min': int(d.min()), 'max': int(d.max())}

    def slippage_estimate(self, spread_bps=5):
        if self.master_df.empty: return 0.0
        return float(len(self.master_df)*spread_bps/10000)

    def transaction_cost_analysis(self, comm=0.0, spread_bps=5):
        if self.master_df.empty: return {'total': 0}
        n = len(self.master_df); tc = n*comm
        ap = float(self.master_df['price'].mean()) if 'price' in self.master_df.columns else 100
        sc = n*ap*spread_bps/10000
        return {'commission': float(tc), 'spread_cost': float(sc), 'total': float(tc+sc)}

    def realized_vs_unrealized_pnl(self):
        if self.holdings.empty: return {'realized': 0, 'unrealized': 0}
        gl = float(self.holdings['gain_loss'].sum()) if 'gain_loss' in self.holdings.columns else 0
        return {'unrealized': gl, 'total_value': float(self.total_value)}

    # ====================== CRYPTO & ALTERNATIVE ASSET METRICS (15) ======================
    def nvt_ratio(self, mcap, daily_tx):
        return float(mcap/daily_tx) if daily_tx>0 else float('inf')

    def mvrv_ratio(self, mcap, realized_cap):
        return float(mcap/realized_cap) if realized_cap>0 else float('inf')

    def stock_to_flow(self, supply, annual_prod):
        return float(supply/annual_prod) if annual_prod>0 else float('inf')

    def crypto_fear_greed_proxy(self):
        if self.returns_series.empty or len(self.returns_series)<30: return {'score': 50, 'sentiment': 'neutral'}
        r = self.returns_series.dropna(); mom=float(r.tail(14).sum()); vol=float(r.tail(14).std())
        score = np.clip(50+mom*500-vol*200, 0, 100)
        sent = 'extreme_greed' if score>75 else 'greed' if score>55 else 'fear' if score<25 else 'extreme_fear' if score<10 else 'neutral'
        return {'score': float(score), 'sentiment': sent}

    def crypto_correlation_to_btc(self, symbol):
        if self.prices.empty or self.prices.ndim<2: return 0.0
        btc = next((c for c in self.prices.columns if 'BTC' in c.upper()), None)
        if not btc or symbol not in self.prices.columns: return 0.0
        return float(self.prices.pct_change().dropna()[symbol].corr(self.prices.pct_change().dropna()[btc]))

    def sharpe_by_asset_class(self):
        if self.prices.empty or self.prices.ndim<2: return {}
        rets = self.prices.pct_change().dropna(); result = {}
        for c in rets.columns:
            r = rets[c].dropna(); sr = float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0
            ac = 'crypto' if 'USD' in c or 'BTC' in c or 'ETH' in c else 'etf' if c in ['SPY','QQQ','VTI','IWM'] else 'stock'
            if ac not in result: result[ac] = []
            result[ac].append({'symbol': c, 'sharpe': sr})
        return result

    def max_drawdown_by_asset(self):
        if self.prices.empty or self.prices.ndim<2: return {}
        result = {}
        for c in self.prices.columns:
            s = self.prices[c].dropna()
            if len(s)<2: result[c]=0; continue
            cum = s/s.iloc[0]; result[c] = float((cum/cum.cummax()-1).min())
        return result

    def rolling_correlation_matrix(self, window=60):
        if self.prices.empty or self.prices.ndim<2: return {}
        rets = self.prices.pct_change().dropna()
        if len(rets)<window: return {}
        corr = rets.tail(window).corr()
        return {f'{i}_{j}': float(corr.loc[i,j]) for i in corr.index for j in corr.columns if i<j}

    def asset_momentum_ranking(self, lookback=63):
        if self.prices.empty or self.prices.ndim<2: return {}
        rets = self.prices.pct_change().dropna()
        if len(rets)<lookback: return {}
        mom = rets.tail(lookback).sum().sort_values(ascending=False)
        return {s: {'rank': i+1, 'momentum': float(mom[s])} for i,s in enumerate(mom.index)}

    def relative_strength_ranking(self, benchmark='SPY'):
        if self.prices.empty or self.prices.ndim<2: return {}
        rets = self.prices.pct_change().dropna()
        if benchmark not in rets.columns: return {}
        br = float(rets[benchmark].tail(63).sum())
        return {c: {'rs': float(rets[c].tail(63).sum())-br, 'outperform': float(rets[c].tail(63).sum())>br} for c in rets.columns if c!=benchmark}

    def realized_volatility_by_asset(self, window=21):
        if self.prices.empty or self.prices.ndim<2: return {}
        rets = self.prices.pct_change().dropna()
        return {c: float(rets[c].tail(window).std()*np.sqrt(252)) for c in rets.columns}

    def correlation_change_detection(self, short=30, long_w=120):
        if self.prices.empty or self.prices.ndim<2 or self.prices.shape[1]<3: return {}
        rets = self.prices.pct_change().dropna()
        if len(rets)<long_w: return {}
        cs = rets.tail(short).corr().values; cl = rets.tail(long_w).corr().values
        np.fill_diagonal(cs, np.nan); np.fill_diagonal(cl, np.nan)
        return {'short_avg': float(np.nanmean(cs)), 'long_avg': float(np.nanmean(cl)), 'change': float(np.nanmean(cs)-np.nanmean(cl)), 'alert': abs(np.nanmean(cs)-np.nanmean(cl))>0.15}

    def portfolio_heat_score(self):
        if self.returns_series.empty or len(self.returns_series)<21: return 0.0
        r = self.returns_series.dropna(); vol=float(r.tail(5).std()*np.sqrt(252)); mom=float(r.tail(5).sum())
        return float(np.clip(50+mom*200+(0.15-vol)*100, 0, 100))

    def drawdown_heatmap_data(self):
        if self.returns_series.empty: return {}
        cum = (1+self.returns_series).cumprod(); dd = cum/cum.cummax()-1; result = {}
        for yr in dd.index.year.unique():
            yd = dd[dd.index.year==yr]
            for mo in yd.index.month.unique():
                result[f'{yr}-{mo:02d}'] = float(yd[yd.index.month==mo].min())
        return result

    # ====================== TAX & ACCOUNTING ANALYTICS (15) ======================
    def capital_gains_estimate(self, st_rate=0.37, lt_rate=0.20):
        if self.holdings.empty or 'gain_loss' not in self.holdings.columns: return {'total_tax': 0}
        gl = self.holdings['gain_loss'].fillna(0); g = float(gl[gl>0].sum()); l = float(gl[gl<0].sum())
        net = g+l; st = max(net*0.5,0)*st_rate; lt = max(net*0.5,0)*lt_rate
        return {'gross_gains': g, 'gross_losses': l, 'net': float(net), 'st_tax': float(st), 'lt_tax': float(lt), 'total_tax': float(st+lt)}

    def tax_loss_harvest_candidates(self, threshold=-0.05):
        if self.holdings.empty or 'gain_loss_pct' not in self.holdings.columns: return []
        c = self.holdings[self.holdings['gain_loss_pct']<threshold*100]
        return [{'symbol': r['symbol'], 'loss_pct': float(r['gain_loss_pct'])} for _,r in c.iterrows()]

    def wash_sale_detector(self, lookback=30):
        if self.master_df.empty or 'date' not in self.master_df.columns: return []
        df = self.master_df.sort_values('date'); alerts = []
        for sym in df['symbol'].unique():
            sd = df[df['symbol']==sym]; dates = sd['date'].values
            for i in range(1, len(dates)):
                if pd.notna(dates[i]) and pd.notna(dates[i-1]):
                    d = (dates[i]-dates[i-1])/np.timedelta64(1,'D')
                    if 0<d<=lookback: alerts.append({'symbol': sym, 'days': int(d)})
        return alerts[:20]

    def portfolio_income_analysis(self):
        est = self.total_value*0.02
        return {'annual': float(est), 'monthly': float(est/12), 'yield_pct': 2.0}

    def cost_basis_summary(self):
        if self.holdings.empty: return {}
        cb = float(self.holdings['cost_basis'].sum()) if 'cost_basis' in self.holdings.columns else 0
        mv = self.total_value
        return {'cost_basis': float(cb), 'market_value': float(mv), 'gain_loss': float(mv-cb), 'return_pct': float((mv-cb)/cb*100) if cb>0 else 0}

    def roi_annualized(self, initial, final, years):
        if years<=0 or initial<=0: return 0.0
        return float((final/initial)**(1/years)-1)

    def time_weighted_return(self, values, cashflows=None):
        if not values or len(values)<2: return 0.0
        twr = 1.0
        for i in range(1, len(values)):
            cf = cashflows[i] if cashflows and i<len(cashflows) else 0
            prev = values[i-1]+cf; twr *= values[i]/prev if prev>0 else 1
        return float(twr-1)

    def money_weighted_return(self, cashflows, guess=0.05):
        if not cashflows or len(cashflows)<2: return 0.0
        r = guess
        for _ in range(500):
            npv = sum(cf/(1+r)**t for t,cf in enumerate(cashflows))
            dnpv = sum(-t*cf/(1+r)**(t+1) for t,cf in enumerate(cashflows))
            if abs(dnpv)<1e-12: break
            r -= npv/dnpv
            if abs(npv)<1e-8: break
        return float(r)

    def effective_tax_rate(self, gross, tax):
        return float(tax/gross*100) if gross>0 else 0.0

    def after_tax_return(self, pre_tax, tax_rate=0.25):
        return float(pre_tax*(1-tax_rate))

    def tax_equivalent_yield(self, tf_yield, tax_rate=0.37):
        return float(tf_yield/(1-tax_rate))

    def portfolio_yield_on_cost(self):
        if self.holdings.empty or 'cost_basis' not in self.holdings.columns: return 0.0
        cb = float(self.holdings['cost_basis'].sum()); inc = self.total_value*0.02
        return float(inc/cb*100) if cb>0 else 0.0

    def unrealized_gain_loss_report(self):
        if self.holdings.empty: return []
        return sorted([{'symbol': r.get('symbol','N/A'), 'qty': float(r.get('quantity',0)), 'gl': float(r.get('gain_loss',0)), 'gl_pct': float(r.get('gain_loss_pct',0))} for _,r in self.holdings.iterrows()], key=lambda x: x['gl'], reverse=True)

    def account_allocation_summary(self):
        if self.master_df.empty or 'account' not in self.master_df.columns: return {}
        a = self.master_df.groupby('account').size(); t = a.sum()
        return {acc: {'count': int(a[acc]), 'pct': float(a[acc]/t*100)} for acc in a.index}

    # ====================== BEHAVIORAL FINANCE & SENTIMENT (15) ======================
    def disposition_effect_score(self):
        if self.holdings.empty or 'gain_loss' not in self.holdings.columns: return 0.0
        gl = self.holdings['gain_loss'].fillna(0); w=len(gl[gl>0]); l=len(gl[gl<0]); t=w+l
        return float((l-w)/t) if t>0 else 0.0

    def recency_bias_indicator(self):
        if self.returns_series.empty or len(self.returns_series)<252: return {'bias': 0}
        r = self.returns_series.dropna(); rc=float(r.tail(21).mean()*252); fl=float(r.mean()*252)
        return {'recent': rc, 'full': fl, 'bias': float(rc-fl), 'signal': 'positive' if rc-fl>0.05 else 'negative' if rc-fl<-0.05 else 'balanced'}

    def anchoring_bias_check(self):
        if self.prices.empty or self.prices.ndim<2: return {}
        result = {}
        for c in self.prices.columns:
            s = self.prices[c].dropna()
            if len(s)<252: continue
            h = float(s.tail(252).max()); cur = float(s.iloc[-1])
            result[c] = {'current': cur, '52w_high': h, 'pct_from_high': float((cur-h)/h*100)}
        return result

    def loss_aversion_ratio(self):
        if self.returns_series.empty: return 0.0
        r = self.returns_series.dropna(); g = r[r>0]; l = r[r<0]
        return float(abs(l.mean())/g.mean()) if len(g)>0 and len(l)>0 else 0.0

    def overconfidence_indicator(self):
        if self.returns_series.empty or len(self.returns_series)<63: return {}
        r = self.returns_series.dropna()
        vr = float(r.tail(21).std()*np.sqrt(252)); vh = float(r.tail(63).std()*np.sqrt(252))
        return {'realized': vr, 'historical': vh, 'overconfident': vr>vh*1.3}

    def herding_indicator(self):
        if self.prices.empty or self.prices.ndim<2 or self.prices.shape[1]<3: return {'herding': False}
        corr = self.prices.pct_change().dropna().tail(21).corr().values; np.fill_diagonal(corr, np.nan)
        ac = float(np.nanmean(corr))
        return {'avg_corr': ac, 'herding': ac>0.7, 'signal': 'strong' if ac>0.8 else 'moderate' if ac>0.6 else 'normal'}

    def fomo_score(self):
        if self.returns_series.empty or len(self.returns_series)<30: return 0.0
        r = self.returns_series.dropna()
        return float(np.clip(50+float(r.tail(5).sum())*500+float(r.tail(21).sum())*200, 0, 100))

    def regret_minimization_score(self):
        if self.returns_series.empty: return 50.0
        cum = (1+self.returns_series).cumprod()
        if len(cum)<2: return 50.0
        regret = float(1-cum.iloc[-1]/cum.cummax().iloc[-1]) if cum.cummax().iloc[-1]>0 else 0
        return float(max(0, 100*(1-regret*5)))

    def emotional_temperature(self):
        if self.returns_series.empty or len(self.returns_series)<21: return {'temp': 50, 'state': 'neutral'}
        r = self.returns_series.dropna(); vol=float(r.tail(5).std()*np.sqrt(252)); mom=float(r.tail(5).sum())
        temp = np.clip(50+mom*300-(vol-0.15)*100, 0, 100)
        state = 'euphoria' if temp>80 else 'optimism' if temp>60 else 'anxiety' if temp<20 else 'fear' if temp<40 else 'neutral'
        return {'temp': float(temp), 'state': state}

    def sunk_cost_detector(self, threshold=-0.20):
        if self.holdings.empty or 'gain_loss_pct' not in self.holdings.columns: return []
        return [{'symbol': r['symbol'], 'loss_pct': float(r['gain_loss_pct'])} for _,r in self.holdings[self.holdings['gain_loss_pct']<threshold*100].iterrows()]

    def portfolio_anxiety_index(self):
        if self.returns_series.empty: return 0.0
        vol = float(self.returns_series.tail(10).std()*np.sqrt(252)) if len(self.returns_series)>=10 else 0
        cum = (1+self.returns_series).cumprod(); dd = float((cum/cum.cummax()-1).iloc[-1]) if len(cum)>0 else 0
        return float(np.clip(vol*100+abs(dd)*200, 0, 100))

    def market_sentiment_composite(self):
        fg = self.crypto_fear_greed_proxy(); mom = self.momentum_signals(); mr = self.mean_reversion_signals()
        scores = [fg.get('score',50), 50+mom.get('score',0)*50, 50-mr.get('zscore',0)*15]
        return {'composite': float(np.mean(scores))}

    def confirmation_bias_check(self):
        if self.returns_series.empty or len(self.returns_series)<252: return {'bias_risk': 'unknown'}
        r = self.returns_series.dropna(); r1=float(r.tail(21).mean()*252); r3=float(r.tail(63).mean()*252); r12=float(r.mean()*252)
        if r1>0 and r3>0 and r12<0: return {'bias_risk': 'high', 'note': 'Recent gains mask long-term underperformance'}
        if r1<0 and r3<0 and r12>0: return {'bias_risk': 'high', 'note': 'Recent losses mask long-term outperformance'}
        return {'bias_risk': 'low', 'r1m': r1, 'r3m': r3, 'r12m': r12}

    def portfolio_stress_indicator(self):
        if self.returns_series.empty: return 0.0
        ax = self.portfolio_anxiety_index(); temp = self.emotional_temperature().get('temp', 50)
        return float(np.clip(ax*0.6+(100-temp)*0.4, 0, 100))

    # ====================== FIXED INCOME ADVANCED & CREDIT (15) ======================
    def zero_coupon_price(self, face, yld, years):
        return float(face/(1+yld)**years)

    def forward_rate(self, r1, t1, r2, t2):
        if t2<=t1: return 0.0
        return float(((1+r2)**t2/(1+r1)**t1)**(1/(t2-t1))-1)

    def spot_rate_from_par(self, par_rates):
        spots=[]; discs=[]
        for i, pr in enumerate(par_rates):
            t=i+1; c=pr/2
            if t==1: spots.append(pr); discs.append(1/(1+pr)); continue
            pvc = sum(c*d for d in discs); dn = (1-pvc*c)/(1+c)
            spots.append(float((1/dn)**(1/t)-1)); discs.append(dn)
        return spots

    def credit_spread(self, corp_yield, rf_yield):
        return float(corp_yield-rf_yield)

    def z_spread_approx(self, bond_price_val, face, coupon_rate, years, base_curve, ppy=2):
        c = face*coupon_rate/ppy; n = int(years*ppy); z = 0.01
        for _ in range(200):
            pv = sum(c/((1+base_curve[min(i,len(base_curve)-1)]/ppy+z/ppy)**(i+1)) for i in range(n))
            pv += face/((1+base_curve[-1]/ppy+z/ppy)**n)
            if abs(pv-bond_price_val)<0.01: return float(z)
            z += (pv-bond_price_val)/face*0.001
        return float(z)

    def current_yield(self, coupon, price):
        return float(coupon/price) if price>0 else 0.0

    def accrued_interest(self, face, coupon_rate, days_since, days_in_period=182):
        return float(face*coupon_rate/2*days_since/days_in_period)

    def dirty_price(self, clean_price, face, coupon_rate, days_since, days_in_period=182):
        return float(clean_price+self.accrued_interest(face, coupon_rate, days_since, days_in_period))

    def dv01_calc(self, face, coupon_rate, years, yield_rate, ppy=2):
        p1 = self.bond_price(face, coupon_rate, years, yield_rate, ppy)
        p2 = self.bond_price(face, coupon_rate, years, yield_rate+0.0001, ppy)
        return float(abs(p2-p1))

    def key_rate_duration(self, face, coupon_rate, years, yield_rate, key_tenors=None, ppy=2):
        if key_tenors is None: key_tenors = [1,2,5,10,30]
        base = self.bond_price(face, coupon_rate, years, yield_rate, ppy)
        result = {}
        for kt in key_tenors:
            if kt>years: continue
            shifted = self.bond_price(face, coupon_rate, years, yield_rate+0.01 if kt<=years/2 else yield_rate, ppy)
            result[f'{kt}y'] = float(-(shifted-base)/base/0.01)
        return result

    def bond_total_return(self, face, coupon_rate, years, buy_yield, sell_yield, hold_years):
        bp = self.bond_price(face, coupon_rate, years, buy_yield)
        sp = self.bond_price(face, coupon_rate, years-hold_years, sell_yield)
        coupons = face*coupon_rate*hold_years
        return float((sp+coupons-bp)/bp)

    def callable_bond_oas(self, mkt_price, face, coupon_rate, years, base_yield, call_price, call_year):
        straight = self.bond_price(face, coupon_rate, years, base_yield)
        callable_val = min(straight, self.bond_price(face, coupon_rate, call_year, base_yield) if call_year<years else straight)
        return float((straight-mkt_price)/mkt_price*100)

    def floating_rate_note_value(self, face, spread, ref_rate, years, ppy=4):
        c = face*(ref_rate+spread)/ppy; n = int(years*ppy); r = (ref_rate+spread)/ppy
        if r==0: return float(c*n+face)
        return float(c*(1-(1+r)**(-n))/r+face/(1+r)**n)

    def inflation_linked_return(self, nominal_return, breakeven_inflation):
        return float((1+nominal_return)/(1+breakeven_inflation)-1)

    # ====================== MONTE CARLO & SIMULATION (15) ======================
    def monte_carlo_portfolio(self, n_sims=1000, horizon=252, initial=None):
        if self.returns_series.empty: return {'median_final': 0}
        mu = float(self.returns_series.mean()); sigma = float(self.returns_series.std())
        init = initial or max(self.total_value, 100000)
        np.random.seed(42); finals = []
        for _ in range(n_sims):
            rets = np.random.normal(mu, sigma, horizon); val = init*np.prod(1+rets); finals.append(val)
        finals = sorted(finals)
        return {'initial': float(init), 'median_final': float(np.median(finals)), 'p5': float(np.percentile(finals,5)),
                'p25': float(np.percentile(finals,25)), 'p75': float(np.percentile(finals,75)),
                'p95': float(np.percentile(finals,95)), 'best': float(max(finals)), 'worst': float(min(finals))}

    def monte_carlo_var(self, n_sims=10000, horizon=1, confidence=0.95):
        if self.returns_series.empty: return 0.0
        mu = float(self.returns_series.mean()); sigma = float(self.returns_series.std())
        np.random.seed(42); sims = np.random.normal(mu*horizon, sigma*np.sqrt(horizon), n_sims)
        return float(-np.percentile(sims, (1-confidence)*100))

    def monte_carlo_cvar(self, n_sims=10000, horizon=1, confidence=0.95):
        if self.returns_series.empty: return 0.0
        mu = float(self.returns_series.mean()); sigma = float(self.returns_series.std())
        np.random.seed(42); sims = np.random.normal(mu*horizon, sigma*np.sqrt(horizon), n_sims)
        var = np.percentile(sims, (1-confidence)*100)
        return float(-np.mean(sims[sims<=var]))

    def geometric_brownian_motion(self, S0=None, mu=None, sigma=None, T=1, n_steps=252, n_paths=100):
        if S0 is None: S0 = max(self.total_value, 100000)
        if mu is None: mu = float(self.returns_series.mean()*252) if not self.returns_series.empty else 0.08
        if sigma is None: sigma = float(self.returns_series.std()*np.sqrt(252)) if not self.returns_series.empty else 0.20
        dt = T/n_steps; np.random.seed(42)
        paths = np.zeros((n_steps+1, n_paths)); paths[0] = S0
        for t in range(1, n_steps+1):
            z = np.random.standard_normal(n_paths)
            paths[t] = paths[t-1]*np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
        return {'median': float(np.median(paths[-1])), 'p5': float(np.percentile(paths[-1],5)),
                'p95': float(np.percentile(paths[-1],95)), 'mean': float(np.mean(paths[-1]))}

    def bootstrap_returns(self, n_sims=1000, horizon=252):
        if self.returns_series.empty or len(self.returns_series)<30: return {'median': 0}
        r = self.returns_series.dropna().values; np.random.seed(42); finals = []
        for _ in range(n_sims):
            sampled = np.random.choice(r, horizon, replace=True)
            finals.append(float(np.prod(1+sampled)-1))
        return {'median': float(np.median(finals)), 'p5': float(np.percentile(finals,5)), 'p95': float(np.percentile(finals,95)), 'mean': float(np.mean(finals))}

    def block_bootstrap_returns(self, n_sims=500, horizon=252, block_size=21):
        if self.returns_series.empty or len(self.returns_series)<block_size*2: return {'median': 0}
        r = self.returns_series.dropna().values; n = len(r); np.random.seed(42); finals = []
        for _ in range(n_sims):
            path = []; i = 0
            while len(path)<horizon:
                start = np.random.randint(0, n-block_size); path.extend(r[start:start+block_size]); i += 1
            finals.append(float(np.prod(1+np.array(path[:horizon]))-1))
        return {'median': float(np.median(finals)), 'p5': float(np.percentile(finals,5)), 'p95': float(np.percentile(finals,95))}

    def jump_diffusion_sim(self, S0=None, mu=None, sigma=None, lam=0.1, jump_mean=-0.02, jump_vol=0.05, T=1, n_steps=252, n_paths=100):
        if S0 is None: S0 = max(self.total_value, 100000)
        if mu is None: mu = float(self.returns_series.mean()*252) if not self.returns_series.empty else 0.08
        if sigma is None: sigma = float(self.returns_series.std()*np.sqrt(252)) if not self.returns_series.empty else 0.20
        dt = T/n_steps; np.random.seed(42)
        paths = np.zeros((n_steps+1, n_paths)); paths[0] = S0
        for t in range(1, n_steps+1):
            z = np.random.standard_normal(n_paths)
            jumps = np.random.poisson(lam*dt, n_paths)*np.random.normal(jump_mean, jump_vol, n_paths)
            paths[t] = paths[t-1]*np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z+jumps)
        return {'median': float(np.median(paths[-1])), 'p5': float(np.percentile(paths[-1],5)), 'p95': float(np.percentile(paths[-1],95))}

    def retirement_monte_carlo(self, portfolio=None, annual_spend=50000, years=30, ret=0.07, vol=0.15, inflation=0.03, n_sims=1000):
        pv = portfolio or max(self.total_value, 500000)
        np.random.seed(42); success = 0; final_vals = []
        for _ in range(n_sims):
            bal = float(pv); spend = annual_spend
            for y in range(years):
                bal = bal*(1+np.random.normal(ret, vol))-spend; spend *= (1+inflation)
                if bal <= 0: bal = 0; break
            final_vals.append(bal)
            if bal > 0: success += 1
        return {'success_rate': float(success/n_sims*100), 'median_final': float(np.median(final_vals)),
                'p10': float(np.percentile(final_vals,10)), 'p90': float(np.percentile(final_vals,90))}

    def sequence_of_returns_risk(self, n_sims=500, horizon=30, annual_spend=50000):
        if self.returns_series.empty: return {'risk_score': 0}
        r = self.returns_series.dropna().values; pv = max(self.total_value, 500000)
        np.random.seed(42); results = []
        for _ in range(n_sims):
            sampled = np.random.choice(r, horizon*252, replace=True)
            bal = float(pv); daily_spend = annual_spend/252
            for d in range(len(sampled)):
                bal = bal*(1+sampled[d])-daily_spend
                if bal <= 0: break
            results.append(bal > 0)
        sr = sum(results)/len(results)
        return {'success_rate': float(sr*100), 'risk_score': float((1-sr)*100)}

    def optimal_allocation_sim(self, n_sims=500, horizon=252):
        if self.prices.empty or self.prices.ndim<2 or self.prices.shape[1]<2: return {}
        rets = self.prices.pct_change().dropna(); n = rets.shape[1]; np.random.seed(42)
        best_sr = -999; best_w = None
        for _ in range(n_sims):
            w = np.random.dirichlet(np.ones(n))
            pr = (rets.values@w); sr = pr.mean()/pr.std()*np.sqrt(252) if pr.std()>0 else 0
            if sr>best_sr: best_sr=sr; best_w=w
        if best_w is None: return {}
        return {rets.columns[i]: float(best_w[i]) for i in range(n)}

    def efficient_frontier_sim(self, n_portfolios=1000):
        if self.prices.empty or self.prices.ndim<2 or self.prices.shape[1]<2: return []
        rets = self.prices.pct_change().dropna(); n = rets.shape[1]; np.random.seed(42); points = []
        for _ in range(n_portfolios):
            w = np.random.dirichlet(np.ones(n))
            pr = rets.values@w; ret = float(pr.mean()*252); vol = float(pr.std()*np.sqrt(252))
            sr = ret/vol if vol>0 else 0
            points.append({'return': ret, 'vol': vol, 'sharpe': sr})
        return sorted(points, key=lambda x: x['vol'])

    def correlation_simulation(self, shock_assets=None, shock_mag=-0.20, n_sims=1000):
        if self.prices.empty or self.prices.ndim<2: return {}
        rets = self.prices.pct_change().dropna(); n = rets.shape[1]; corr = rets.corr().values
        np.random.seed(42); impacts = np.zeros(n)
        for _ in range(n_sims):
            z = np.random.multivariate_normal(np.zeros(n), corr)
            if shock_assets:
                for sa in shock_assets:
                    if sa in rets.columns:
                        idx = list(rets.columns).index(sa); z[idx] = shock_mag/rets.iloc[:,idx].std()
            impacts += z
        impacts /= n_sims
        return {rets.columns[i]: float(impacts[i]*rets.iloc[:,i].std()) for i in range(n)}

    def tail_risk_simulation(self, n_sims=10000, confidence=0.99):
        if self.returns_series.empty: return {'var': 0, 'cvar': 0}
        mu = float(self.returns_series.mean()); sigma = float(self.returns_series.std())
        np.random.seed(42); sims = np.random.normal(mu, sigma, n_sims)
        var = float(-np.percentile(sims, (1-confidence)*100))
        cvar = float(-np.mean(sims[sims<=np.percentile(sims,(1-confidence)*100)]))
        return {'var': var, 'cvar': cvar, 'max_loss': float(-min(sims))}

    def drawdown_simulation(self, n_sims=500, horizon=252):
        if self.returns_series.empty: return {'median_mdd': 0}
        mu = float(self.returns_series.mean()); sigma = float(self.returns_series.std())
        np.random.seed(42); mdds = []
        for _ in range(n_sims):
            rets = np.random.normal(mu, sigma, horizon); cum = np.cumprod(1+rets)
            dd = cum/np.maximum.accumulate(cum)-1; mdds.append(float(dd.min()))
        return {'median_mdd': float(np.median(mdds)), 'p5_mdd': float(np.percentile(mdds,5)), 'p95_mdd': float(np.percentile(mdds,95))}

    # ====================== BOOK BALANCER ENGINE ======================
    def add_book_entry(self, date_str, description, debit=0.0, credit=0.0, account="General", category=""):
        entry = {
            'id': len(self.book_entries) + 1,
            'date': date_str,
            'description': description,
            'debit': float(debit),
            'credit': float(credit),
            'account': account,
            'category': category,
        }
        self.book_entries.append(entry)
        return entry

    def remove_book_entry(self, entry_id):
        self.book_entries = [e for e in self.book_entries if e['id'] != entry_id]

    def get_book_summary(self):
        total_debit = sum(e['debit'] for e in self.book_entries)
        total_credit = sum(e['credit'] for e in self.book_entries)
        balance = total_credit - total_debit
        by_account = {}
        for e in self.book_entries:
            acc = e['account']
            if acc not in by_account:
                by_account[acc] = {'debit': 0.0, 'credit': 0.0}
            by_account[acc]['debit'] += e['debit']
            by_account[acc]['credit'] += e['credit']
        by_category = {}
        for e in self.book_entries:
            cat = e['category'] or 'Uncategorized'
            if cat not in by_category:
                by_category[cat] = {'debit': 0.0, 'credit': 0.0}
            by_category[cat]['debit'] += e['debit']
            by_category[cat]['credit'] += e['credit']
        return {
            'total_debit': total_debit,
            'total_credit': total_credit,
            'balance': balance,
            'entries': len(self.book_entries),
            'by_account': by_account,
            'by_category': by_category,
            'balanced': abs(total_debit - total_credit) < 0.01,
        }

    # ====================== BUDGET PLANNER ENGINE ======================
    def add_budget_item(self, name, amount, item_type="expense", frequency="monthly", category="General"):
        item = {
            'id': len(self.budget_items) + 1,
            'name': name,
            'amount': float(amount),
            'type': item_type,
            'frequency': frequency,
            'category': category,
        }
        self.budget_items.append(item)
        return item

    def remove_budget_item(self, item_id):
        self.budget_items = [b for b in self.budget_items if b['id'] != item_id]

    def get_budget_summary(self):
        freq_mult = {'weekly': 4.33, 'biweekly': 2.167, 'monthly': 1.0, 'quarterly': 1.0 / 3, 'yearly': 1.0 / 12}
        monthly_income = 0.0
        monthly_expenses = 0.0
        by_category = {}
        for item in self.budget_items:
            mult = freq_mult.get(item['frequency'], 1.0)
            monthly_amt = item['amount'] * mult
            if item['type'] == 'income':
                monthly_income += monthly_amt
            else:
                monthly_expenses += monthly_amt
                cat = item['category']
                by_category[cat] = by_category.get(cat, 0.0) + monthly_amt
        net = monthly_income - monthly_expenses
        savings_rate = (net / monthly_income * 100) if monthly_income > 0 else 0
        return {
            'monthly_income': monthly_income,
            'monthly_expenses': monthly_expenses,
            'monthly_net': net,
            'savings_rate': savings_rate,
            'annual_income': monthly_income * 12,
            'annual_expenses': monthly_expenses * 12,
            'annual_net': net * 12,
            'by_category': by_category,
            'items': len(self.budget_items),
            'runway_months': monthly_income / monthly_expenses if monthly_expenses > 0 else float('inf'),
        }

    def chart_budget_breakdown(self):
        summary = self.get_budget_summary()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#0a0a0a')
        for ax in axes:
            ax.set_facecolor('#121212')
        cats = summary['by_category']
        if cats:
            labels = list(cats.keys())
            values = list(cats.values())
            clist = ['#00ff9d', '#ffd700', '#ff3366', '#00bfff', '#ff9900', '#9966ff', '#ff6699', '#33cccc']
            pie_colors = [clist[i % len(clist)] for i in range(len(labels))]
            axes[0].pie(values, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90,
                        textprops={'color': '#ffffff', 'fontsize': 9})
            axes[0].set_title("EXPENSE BREAKDOWN", color="#ffd700", fontsize=12, fontweight="bold")
        else:
            axes[0].text(0.5, 0.5, "No expenses added", ha="center", va="center",
                         color="#aaaaaa", fontsize=14, transform=axes[0].transAxes)
            axes[0].set_title("EXPENSE BREAKDOWN", color="#ffd700", fontsize=12, fontweight="bold")
        inc = summary['monthly_income']
        exp = summary['monthly_expenses']
        net_val = summary['monthly_net']
        bar_labels = ['Income', 'Expenses', 'Net']
        bar_values = [inc, exp, net_val]
        bar_colors = ['#00ff9d', '#ff3366', '#ffd700' if net_val >= 0 else '#ff3366']
        axes[1].bar(bar_labels, bar_values, color=bar_colors, width=0.5, alpha=0.85)
        axes[1].set_title("MONTHLY OVERVIEW", color="#00ff9d", fontsize=12, fontweight="bold")
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        axes[1].tick_params(colors='#aaaaaa')
        for spine in axes[1].spines.values():
            spine.set_color('#333333')
        axes[1].grid(True, alpha=0.1, color='#555555', axis='y')
        plt.tight_layout()
        return self._fig_to_base64(fig)


    # ====================== PLAN SAVE/LOAD ENGINE ======================
    def save_plan(self, plan_data, plan_name="default"):
        """Save a plan dict to JSON file in the app folder."""
        import json
        safe_name = "".join(c if c.isalnum() or c in ('_','-',' ') else '_' for c in plan_name).strip()
        if not safe_name:
            safe_name = "default"
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{safe_name}.plan.json")
        plan_data['_plan_name'] = plan_name
        plan_data['_saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, indent=2, default=str)
        return fp

    def load_plan(self, plan_name="default"):
        """Load a plan dict from JSON file."""
        import json
        safe_name = "".join(c if c.isalnum() or c in ('_','-',' ') else '_' for c in plan_name).strip()
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{safe_name}.plan.json")
        if not os.path.exists(fp):
            return None
        with open(fp, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_saved_plans(self):
        """List all saved plan files in the app folder."""
        import glob
        folder = os.path.dirname(os.path.abspath(__file__))
        plans = []
        for fp in glob.glob(os.path.join(folder, "*.plan.json")):
            try:
                import json
                with open(fp, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                plans.append({
                    'file': fp,
                    'name': d.get('_plan_name', os.path.basename(fp)),
                    'saved_at': d.get('_saved_at', 'Unknown'),
                    'entity_name': d.get('entity_name', ''),
                })
            except Exception:
                plans.append({'file': fp, 'name': os.path.basename(fp), 'saved_at': 'Error', 'entity_name': ''})
        return plans

    def delete_plan(self, plan_name):
        """Delete a saved plan file."""
        safe_name = "".join(c if c.isalnum() or c in ('_','-',' ') else '_' for c in plan_name).strip()
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{safe_name}.plan.json")
        if os.path.exists(fp):
            os.remove(fp)
            return True
        return False

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
            dlg = ft.AlertDialog(
                modal=True,
                title=ft.Text(title, size=20, weight=ft.FontWeight.BOLD, color="#00ff9d"),
                content=ft.Container(content=img_big, width=960, height=640, bgcolor="#0a0a0a", border_radius=12, padding=10),
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
                                      check_color="#0a0a0a", label_style=ft.TextStyle(color="#ffd700", size=12))
            def toggle_all(e2):
                for cb in file_checkboxes.values():
                    cb.value = sel_all_cb.value
                page.update()
            sel_all_cb.on_change = toggle_all
            file_list_col.controls.append(sel_all_cb)
            for fname, fpath, ext, size in sorted(found, key=lambda x: x[0]):
                size_str = f"{size/1024:.0f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
                cb = ft.Checkbox(label=f"{fname}  ({ext}, {size_str})", value=False, fill_color="#00ff9d",
                                  check_color="#0a0a0a", label_style=ft.TextStyle(color="#ffffff", size=11))
                file_checkboxes[fpath] = cb
                file_list_col.controls.append(cb)
        page.update()

    import_status = ft.Row([
        ft.ProgressRing(width=20, height=20, stroke_width=2, color="#00ff9d", visible=False),
        ft.Text("", size=12, color="#00ff9d", italic=True),
    ], spacing=8, visible=False)

    def do_import(e):
        # Collect checked files, or fall back to manual path
        selected = [fp for fp, cb in file_checkboxes.items() if cb.value]
        raw = (folder_field.value or "").strip()
        if not selected and not raw:
            show_snack("Select files or enter a folder path to import.", "#663300")
            return
        # Show loading indicator
        import_status.controls[0].visible = True
        import_status.controls[1].value = f"Importing {len(selected) if selected else 'all'} file(s)..."
        import_status.visible = True
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
                    page.update()
                    show_snack(f"Imported {count:,} records from {files_n} file(s) ({', '.join(fmts)})", "#006644")
                    load_category(nav_rail.selected_index)
                else:
                    import_status.controls[0].visible = False
                    import_status.controls[1].value = "No valid data found."
                    import_status.controls[1].color = "#ff3366"
                    page.update()
                    show_snack("No valid data found in selected files.", "#663300")
            except Exception as ex:
                import_status.controls[0].visible = False
                import_status.controls[1].value = f"Error: {ex}"
                import_status.controls[1].color = "#ff3366"
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
            "Planning (9 tabs)": "Goal probability (Monte Carlo), time machine slider, advanced planner (inflation/tax/income/expenses), FIRE calculator (lean/regular/fat/coast), growth projection, monthly comparison, dream life architect, income tracker (dividends), what-if scenario (4 interactive sliders)",
            "Risk (10 tabs)": "Risk radar chart, stress test lab, emotional risk gauge, diversification score, market regime detection, drawdown chart, rolling Sharpe (60d), rolling volatility (30d), gain/loss waterfall, win/loss streaks & recovery time",
            "AI & Tools (7 tabs)": "Smart brain analysis, AI co-pilot chat, voice command simulation, Monte Carlo paths, daily diary, portfolio rebalancer (max Sharpe), tax-loss harvesting optimizer",
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
        vol = s.get("ann_volatility", 0)
        sharpe = s.get("sharpe", 0)
        sortino = s.get("sortino", 0)
        max_dd = s.get("max_drawdown", 0)
        total_gl = s.get("total_gain_loss", 0)
        gl_color = "#00ff9d" if total_gl >= 0 else "#ff3366"
        n_hold = s.get("num_holdings", 0)
        ann_ret = s.get("ann_return", 0)
        best = s.get("best_day", 0)
        worst = s.get("worst_day", 0)
        win_rate = s.get("win_rate", 0)
        var95 = s.get("var_95", 0)
        gl_pct = s.get("total_gain_pct", 0)
        def _dk(label, val, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(val), size=18, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=12, expand=True)
        kpi_row1 = ft.Row([
            _dk("Total Value", f"${data.total_value:,.0f}", "#ffd700"),
            _dk("Total G/L", f"${total_gl:,.0f}" if total_gl != 0 else "N/A", gl_color),
            _dk("G/L %", f"{gl_pct:+.1f}%" if gl_pct != 0 else "N/A", gl_color),
            _dk("Ann. Return", f"{ann_ret:.1%}" if ann_ret != 0 else "N/A", "#00ff9d" if ann_ret >= 0 else "#ff3366"),
            _dk("Holdings", str(n_hold) if n_hold > 0 else "0"),
        ], spacing=6)
        kpi_row2 = ft.Row([
            _dk("Volatility", f"{vol:.1%}" if vol != 0 else "N/A", "#ff3366"),
            _dk("Sharpe", f"{sharpe:.2f}" if sharpe != 0 else "N/A", "#00ff9d" if sharpe > 0 else "#ff3366"),
            _dk("Sortino", f"{sortino:.2f}" if sortino != 0 else "N/A", "#00ff9d" if sortino > 0 else "#ff3366"),
            _dk("Max DD", f"{max_dd:.1%}" if max_dd != 0 else "N/A", "#ff3366"),
            _dk("Profit Factor", f"{pf:.2f}" if pf != 0 else "N/A", "#00ff9d" if pf > 1 else "#ff3366"),
        ], spacing=6)
        kpi_row3 = ft.Row([
            _dk("Win Rate", f"{win_rate:.0%}" if win_rate != 0 else "N/A", "#00ff9d" if win_rate > 0.5 else "#ff3366"),
            _dk("Best Day", f"{best:.2%}" if best != 0 else "N/A", "#00ff9d"),
            _dk("Worst Day", f"{worst:.2%}" if worst != 0 else "N/A", "#ff3366"),
            _dk("VaR 95%", f"{var95:.2%}" if var95 != 0 else "N/A", "#ff3366"),
            _dk("Win Streak", f"{wl['longest_win']}d" if wl['longest_win'] > 0 else "N/A", "#00ff9d"),
        ], spacing=6)
        # Recovery info
        rec_text = f"Max recovery: {rec['max_recovery_days']}d" if rec['max_recovery_days'] > 0 else "No drawdown recovery data"
        if rec['currently_in_drawdown']:
            rec_text += f"  |  Currently in drawdown: {rec['current_dd_days']}d"
        return _card(ft.Column([
            ft.Text("DASHBOARD", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text(f"Last updated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}", size=11, color="#666666"),
            kpi_row1,
            kpi_row2,
            kpi_row3,
            ft.Text(rec_text, size=11, color="#888888", italic=True),
            _chart_image(data.chart_equity_curve()),
        ], scroll=ft.ScrollMode.AUTO, spacing=10))

    def data_organizer_tab():
        hierarchy = data.get_time_hierarchy()
        if hierarchy.empty:
            content = ft.Text("No data loaded. Use IMPORT FOLDER to load CSVs.", size=20, color="#aaaaaa")
        else:
            rows = []
            for _, r in hierarchy.iterrows():
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(str(r.get("year", "")), color="#ffffff")),
                    ft.DataCell(ft.Text(str(r.get("quarter", "")), color="#ffffff")),
                    ft.DataCell(ft.Text(str(r.get("month", "")), color="#ffffff")),
                    ft.DataCell(ft.Text(str(r.get("records", "")), color="#ffd700")),
                ]))
            content = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Year", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Quarter", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Month", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Records", color="#00ff9d")),
                ],
                rows=rows[:50],
                heading_row_color="#1a1a2e",
            )
        # Import summary section
        imp_summary = data.get_import_summary()
        imp_controls = []
        if imp_summary:
            imp_controls.append(ft.Text("IMPORT SUMMARY", size=16, weight=ft.FontWeight.BOLD, color="#ffd700"))
            imp_rows = []
            for s in imp_summary:
                syms_str = ", ".join(s['symbols'][:8]) + ("..." if len(s['symbols']) > 8 else "")
                imp_rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(s['file'], color="#ffffff", size=11)),
                    ft.DataCell(ft.Text(s['format'], color="#00bfff", size=11)),
                    ft.DataCell(ft.Text(str(s['records']), color="#ffd700", size=11)),
                    ft.DataCell(ft.Text(syms_str, color="#00ff9d", size=11)),
                    ft.DataCell(ft.Text(f"{s['date_from']} to {s['date_to']}", color="#aaaaaa", size=11)),
                ]))
            imp_controls.append(ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("File", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Format", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Records", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Symbols", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Date Range", color="#00ff9d")),
                ],
                rows=imp_rows[:30],
                heading_row_color="#1a1a2e",
            ))
        # Asset source map
        asset_map = data.get_asset_source_map()
        asset_controls = []
        if asset_map:
            asset_controls.append(ft.Text("ASSET SOURCE MAP", size=16, weight=ft.FontWeight.BOLD, color="#ffd700"))
            for sym, files in sorted(asset_map.items()):
                asset_controls.append(ft.Row([
                    ft.Text(sym, size=12, weight=ft.FontWeight.BOLD, color="#00ff9d", width=100),
                    ft.Text(", ".join(files), size=11, color="#aaaaaa", expand=True),
                ], spacing=4))
        col_items = [
            ft.Text("DATA ORGANIZER", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text(f"Total records: {len(data.master_df):,}  |  Files: {len(data.raw_files)}  |  Formats: {', '.join(set(os.path.splitext(f)[1] for f in data.raw_files)) or 'N/A'}", size=14, color="#ffd700"),
            content,
        ]
        if imp_controls:
            col_items.append(ft.Divider(height=1, color="#333333"))
            col_items.extend(imp_controls)
        if asset_controls:
            col_items.append(ft.Divider(height=1, color="#333333"))
            col_items.extend(asset_controls)
        return _card(ft.Column(col_items, scroll=ft.ScrollMode.AUTO, spacing=15))

    def portfolio_tab():
        holdings_data = data.holdings_with_weights()
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
        gl_color = "#00ff9d" if total_gl >= 0 else "#ff3366"
        return _card(ft.Column([
            ft.Text("PORTFOLIO HOLDINGS", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Row([
                ft.Text(f"Total Value: ${data.total_value:,.0f}", size=16, color="#ffd700"),
                ft.Text(f"Total G/L: ${total_gl:,.0f}", size=16, color=gl_color),
                ft.Text(f"Holdings: {len(holdings_data)}", size=16, color="#ffffff"),
            ], spacing=20),
            content,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def analysis_tab():
        return _card(ft.Column([
            ft.Text("ADVANCED ANALYSIS", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_correlation_matrix()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def market_graphs_tab():
        return _card(ft.Column([
            ft.Text("LIVE MARKET GRAPHS + PROJECTION", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_candlestick()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def goals_tab():
        prob = data.simulate_goal_probability()
        color = "#00ff9d" if prob > 70 else "#ffd700" if prob > 40 else "#ff3366"
        return _card(ft.Column([
            ft.Text("GOALS & BACKTEST", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text(f"Goal: $500,000 in 10 years", size=22, color="#ffffff"),
            ft.Text(f"Success Probability: {prob:.1f}%", size=28, weight=ft.FontWeight.BOLD, color=color),
            ft.ProgressBar(value=prob / 100, color=color, bgcolor="#1a1a2e"),
            ft.Text("Based on 10,000 Monte Carlo simulation paths", size=14, color="#aaaaaa"),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def smart_brain_tab():
        return _card(ft.Column([
            ft.Text("SMART BRAIN", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text(data.generate_smart_brain(), size=20, color="#ffffff"),
        ], spacing=15))

    def ai_chat_tab():
        chat_history = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        question_field = ft.TextField(
            label="Ask the AI anything about your portfolio...",
            expand=True, multiline=True, min_lines=1, max_lines=3,
            border_color="#00ff9d", color="#ffffff",
        )
        def send(e):
            q = question_field.value
            if not q or not q.strip():
                return
            chat_history.controls.append(ft.Container(
                content=ft.Text(f"You: {q}", color="#ffd700", size=16),
                bgcolor="#1a1a2e", padding=10, border_radius=8,
            ))
            answer = data.answer_ai_question(q)
            chat_history.controls.append(ft.Container(
                content=ft.Text(f"AI: {answer}", color="#00ff9d", size=16),
                bgcolor="#0d2818", padding=10, border_radius=8,
            ))
            question_field.value = ""
            page.update()
        return _card(ft.Column([
            ft.Text("AI CO-PILOT", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            chat_history,
            ft.Row([
                question_field,
                ft.IconButton(icon=ft.Icons.SEND, icon_color="#00ff9d", on_click=send),
            ]),
        ], spacing=10, expand=True))

    def risk_radar_tab():
        return _card(ft.Column([
            ft.Text("HOLOGRAPHIC RISK RADAR", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_risk_radar()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def portfolio_dna_tab():
        return _card(ft.Column([
            ft.Text("PORTFOLIO DNA FINGERPRINT", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_portfolio_dna()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def emotional_gauge_tab():
        stress = data.emotional_risk_gauge_value()
        color = "#ff3366" if stress > 65 else "#ffd700" if stress > 45 else "#00ff9d"
        label = "HIGH STRESS" if stress > 65 else "MODERATE" if stress > 45 else "LOW STRESS"
        return _card(ft.Column([
            ft.Text("EMOTIONAL RISK GAUGE", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Container(
                content=ft.Column([
                    ft.ProgressRing(value=stress / 100, color=color, width=200, height=200, stroke_width=25),
                    ft.Text(f"{stress}%", size=42, weight=ft.FontWeight.BOLD, color=color),
                    ft.Text(label, size=20, color=color),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                alignment=ft.Alignment(0, 0),
                padding=30,
            ),
            ft.Text("Measures how stressful your current allocation would feel during a market crash.", size=14, color="#aaaaaa"),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=15))

    def time_machine_tab():
        result = ft.Text("", size=16, color="#00ff9d")
        year_label = ft.Text("2026", size=28, weight=ft.FontWeight.BOLD, color="#ffd700")
        table_col = ft.Column(spacing=5)
        def on_slider_change(e):
            yr = int(e.control.value)
            year_label.value = str(yr)
            result.value = data.time_machine_projection(yr)
            rows_data = data.time_machine_table(end_year=yr)
            table_col.controls.clear()
            if rows_data and data.total_value > 0:
                table_col.controls.append(ft.Text("YEAR-BY-YEAR PROJECTION TABLE", size=14, weight=ft.FontWeight.BOLD, color="#ffd700"))
                for r in rows_data[-10:]:
                    table_col.controls.append(ft.Row([
                        ft.Text(str(r["year"]), size=13, color="#ffffff", width=50),
                        ft.Text(f"${r['projected']:,.0f}", size=13, color="#00ff9d", width=100),
                        ft.Text(f"${r['low']:,.0f}", size=13, color="#ff3366", width=100),
                        ft.Text(f"${r['high']:,.0f}", size=13, color="#ffd700", width=100),
                    ], spacing=10))
            page.update()
        slider = ft.Slider(min=2026, max=2050, divisions=24, value=2026, on_change=on_slider_change)
        return _card(ft.Column([
            ft.Text("TIME MACHINE", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Slide to project your portfolio into the future", size=13, color="#aaaaaa"),
            year_label,
            slider,
            result,
            table_col,
        ], scroll=ft.ScrollMode.AUTO, spacing=10))

    def daily_diary_tab():
        return _card(ft.Column([
            ft.Text("DAILY DIARY", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text(data.daily_diary(), size=18, color="#ffffff"),
        ], spacing=15))

    def dream_architect_tab():
        tf = ft.TextField(
            label="Describe your dream life in 2035...",
            expand=True, multiline=True, min_lines=2, max_lines=4,
            border_color="#00ff9d", color="#ffffff",
        )
        result_text = ft.Text("", size=18, color="#ffffff")
        def build(e):
            result_text.value = data.dream_life_architect(tf.value)
            page.update()
        return _card(ft.Column([
            ft.Text("DREAM LIFE ARCHITECT", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            tf,
            ft.Button("BUILD MY DREAM PORTFOLIO", bgcolor="#00ff9d", color="#0a0a0a", on_click=build),
            result_text,
        ], spacing=15))

    def stress_lab_tab():
        dd = ft.Dropdown(
            options=[
                ft.dropdown.Option("2008 Crash"),
                ft.dropdown.Option("2020 COVID"),
                ft.dropdown.Option("Hyperinflation 15%"),
                ft.dropdown.Option("Dot-Com Bubble"),
                ft.dropdown.Option("Custom Black Swan"),
            ],
            value="2008 Crash", label="Scenario", border_color="#00ff9d", color="#ffffff",
        )
        result_text = ft.Text("", size=18, color="#ffffff")
        def run(e):
            result_text.value = data.stress_test_lab(dd.value)
            page.update()
        return _card(ft.Column([
            ft.Text("STRESS-TESTING LAB", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            dd,
            ft.Button("UNLEASH EVENT", bgcolor="#ff3366", color="#ffffff", on_click=run),
            result_text,
        ], spacing=15))

    def diversification_tab():
        return _card(ft.Column([
            ft.Text("DIVERSIFICATION SCORE", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text(data.generate_diversification_score(), size=18, color="#ffffff", font_family="Consolas"),
        ], spacing=15))

    def market_regime_tab():
        return _card(ft.Column([
            ft.Text("MARKET REGIME DETECTOR", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text(data.detect_market_regime(), size=20, color="#ffffff"),
        ], spacing=15))

    def voice_tab():
        cmd_field = ft.TextField(label="Type a voice command...", expand=True, border_color="#ffd700", color="#ffffff")
        result_text = ft.Text("", size=18, color="#ffffff")
        def execute(e):
            result_text.value = data.voice_command_sim(cmd_field.value or "Show net worth 2030")
            page.update()
        return _card(ft.Column([
            ft.Text("VOICE CO-PILOT", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Row([cmd_field, ft.IconButton(icon=ft.Icons.MIC, icon_color="#ffd700", on_click=execute)]),
            result_text,
        ], spacing=15))

    def monte_carlo_tab():
        return _card(ft.Column([
            ft.Text("MONTE CARLO MULTIVERSE", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_monte_carlo()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def constellation_tab():
        return _card(ft.Column([
            ft.Text("PORTFOLIO CONSTELLATION", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_constellation()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def settings_tab():
        theme_dd = ft.Dropdown(
            options=[ft.dropdown.Option("Dark"), ft.dropdown.Option("Light")],
            value="Dark", label="Theme", border_color="#00ff9d", color="#ffffff",
        )
        risk_dd = ft.Dropdown(
            options=[ft.dropdown.Option("Conservative"), ft.dropdown.Option("Moderate"), ft.dropdown.Option("Aggressive")],
            value=data.risk_profile, label="Risk Profile", border_color="#00ff9d", color="#ffffff",
        )
        def apply_settings(e):
            if theme_dd.value == "Light":
                page.theme_mode = ft.ThemeMode.LIGHT
            else:
                page.theme_mode = ft.ThemeMode.DARK
            data.risk_profile = risk_dd.value
            page.update()
            show_snack("Settings applied!", "#006644")
        return _card(ft.Column([
            ft.Text("SETTINGS", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            theme_dd,
            risk_dd,
            ft.Button("APPLY", bgcolor="#00ff9d", color="#0a0a0a", on_click=apply_settings),
        ], spacing=20))

    # ====================== ADVANCED PLANNER ======================
    def planner_tab():
        age_field = ft.TextField(label="Current Age", value="30", width=110, border_color="#00ff9d", color="#ffffff", text_size=13, dense=True)
        retire_field = ft.TextField(label="Retire Age", value="65", width=110, border_color="#00ff9d", color="#ffffff", text_size=13, dense=True)
        contrib_field = ft.TextField(label="Monthly $", value="500", width=120, border_color="#00ff9d", color="#ffffff", text_size=13, dense=True)
        savings_field = ft.TextField(label="Savings %", value="20", width=110, border_color="#00ff9d", color="#ffffff", text_size=13, dense=True)
        income_field = ft.TextField(label="Annual Income $", value="75000", width=140, border_color="#ffd700", color="#ffffff", text_size=13, dense=True)
        expense_field = ft.TextField(label="Annual Expenses $", value="50000", width=150, border_color="#ffd700", color="#ffffff", text_size=13, dense=True)
        tax_field = ft.TextField(label="Tax Rate %", value="22", width=110, border_color="#ffd700", color="#ffffff", text_size=13, dense=True)
        inflation_field = ft.TextField(label="Inflation %", value="3.0", width=110, border_color="#ffd700", color="#ffffff", text_size=13, dense=True)
        goal_field = ft.TextField(label="Custom Goal $", value="0", width=140, border_color="#ff9900", color="#ffffff", text_size=13, dense=True)
        whatif_label = ft.Text("What-If Contribution: $500/mo", size=13, color="#aaaaaa")
        whatif_slider = ft.Slider(min=0, max=5000, divisions=100, value=500, label="{value}", active_color="#00ff9d")
        result_col = ft.Column(spacing=8)

        def _kpi(label, value, color="#ffd700"):
            return ft.Container(
                content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(value, size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2),
                bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True,
            )

        def compute(e):
            try:
                age = int(age_field.value or 30)
                ret_age = int(retire_field.value or 65)
                contrib = float(whatif_slider.value or 500)
                sav = float(savings_field.value or 20)
                inc = float(income_field.value or 0)
                exp = float(expense_field.value or 0)
                tax = float(tax_field.value or 0)
                infl = float(inflation_field.value or 3.0)
                goal = float(goal_field.value or 0)
            except ValueError:
                return
            contrib_field.value = str(int(contrib))
            whatif_label.value = f"What-If Contribution: ${int(contrib):,}/mo"
            p = data.financial_planner_detail(monthly_contribution=contrib, target_retirement_age=ret_age, current_age=age, savings_rate=sav, annual_income=inc, annual_expenses=exp, tax_rate=tax, inflation_rate=infl, custom_goal=goal)
            result_col.controls.clear()
            # KPI row
            result_col.controls.append(ft.Row([
                _kpi("Portfolio Value", f"${p['current_value']:,.0f}"),
                _kpi("Ann. Return", f"{p['ann_ret']:.1%}", "#00ff9d"),
                _kpi("Volatility", f"{p['ann_vol']:.1%}", "#ff3366"),
                _kpi("Holdings", str(p['assets'])),
            ], spacing=6))
            result_col.controls.append(ft.Divider(height=1, color="#333333"))
            # Retirement projection
            result_col.controls.append(ft.Text("RETIREMENT PROJECTION", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"))
            result_col.controls.append(ft.Row([
                _kpi("Years to Retire", str(p['years_to_retire'])),
                _kpi("Portfolio Growth", f"${p['fv_portfolio']:,.0f}", "#00ff9d"),
                _kpi("Contributions", f"${p['fv_contributions']:,.0f}"),
                _kpi("Total (Nominal)", f"${p['fv_total']:,.0f}", "#00ff9d"),
            ], spacing=6))
            result_col.controls.append(ft.Row([
                _kpi("Total (Real $)", f"${p['fv_total_real']:,.0f}", "#ff9900"),
                _kpi("Low Estimate", f"${p['fv_low']:,.0f}", "#ff3366"),
                _kpi("High Estimate", f"${p['fv_high']:,.0f}", "#00ff9d"),
            ], spacing=6))
            result_col.controls.append(ft.Divider(height=1, color="#333333"))
            # Income projections
            result_col.controls.append(ft.Text("RETIREMENT INCOME (4% RULE)", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"))
            result_col.controls.append(ft.Row([
                _kpi("Annual (Nominal)", f"${p['annual_income']:,.0f}", "#ffd700"),
                _kpi("Monthly (Nominal)", f"${p['monthly_income']:,.0f}", "#ffd700"),
                _kpi("Annual (Real $)", f"${p['annual_income_real']:,.0f}", "#ff9900"),
                _kpi("Monthly (Real $)", f"${p['monthly_income_real']:,.0f}", "#ff9900"),
            ], spacing=6))
            if p['tax_rate'] > 0:
                result_col.controls.append(ft.Row([
                    _kpi("After-Tax Annual", f"${p['net_annual_after_tax']:,.0f}", "#00ff9d"),
                    _kpi("After-Tax Monthly", f"${p['net_monthly_after_tax']:,.0f}", "#00ff9d"),
                ], spacing=6))
            if p['annual_surplus'] != 0:
                sc = "#00ff9d" if p['annual_surplus'] > 0 else "#ff3366"
                result_col.controls.append(ft.Row([
                    _kpi("Annual Surplus", f"${p['annual_surplus']:,.0f}", sc),
                    _kpi("Monthly Surplus", f"${p['monthly_surplus']:,.0f}", sc),
                    _kpi("Inflation", f"{p['inflation_rate']:.1f}%", "#ff9900"),
                ], spacing=6))
            result_col.controls.append(ft.Divider(height=1, color="#333333"))
            # Allocation + pie chart
            result_col.controls.append(ft.Text("SUGGESTED ALLOCATION", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"))
            result_col.controls.append(ft.Row([
                ft.Column([ft.Text(p['allocation'], size=13, color="#cccccc")], expand=True),
                _chart_image(data.chart_allocation_pie(p['alloc_pcts'])),
            ], spacing=10))
            result_col.controls.append(ft.Divider(height=1, color="#333333"))
            # 10-year projection table (nominal + real)
            result_col.controls.append(ft.Text("10-YEAR PROJECTION (NOMINAL vs REAL $)", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"))
            result_col.controls.append(ft.Row([
                ft.Text("Year", size=12, color="#aaaaaa", width=60),
                ft.Text("Nominal", size=12, color="#aaaaaa", width=110),
                ft.Text("Real (Today's $)", size=12, color="#aaaaaa", width=110),
            ], spacing=8))
            for row in p['projection_10y']:
                result_col.controls.append(ft.Row([
                    ft.Text(str(row['year']), size=13, color="#ffffff", width=60),
                    ft.Text(f"${row['nominal']:,.0f}", size=13, color="#00ff9d", width=110),
                    ft.Text(f"${row['real']:,.0f}", size=13, color="#ff9900", width=110),
                ], spacing=8))
            result_col.controls.append(ft.Divider(height=1, color="#333333"))
            # Milestones
            result_col.controls.append(ft.Text("FINANCIAL MILESTONES", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"))
            for m in p['milestones']:
                c = "#00ff9d" if m['status'] == "ACHIEVED" else "#ffd700" if m['years'] < 20 else "#aaaaaa"
                result_col.controls.append(ft.Row([
                    ft.Text(m['label'], size=13, color="#ffffff", width=180),
                    ft.Text(f"${m['target']:,.0f}", size=13, color="#ffd700", width=100),
                    ft.Text(m['status'], size=13, color=c),
                ], spacing=8))
            page.update()

        def on_whatif(e):
            contrib_field.value = str(int(whatif_slider.value))
            whatif_label.value = f"What-If Contribution: ${int(whatif_slider.value):,}/mo"
            page.update()

        compute(None)
        return _card(ft.Column([
            ft.Text("ADVANCED FINANCIAL PLANNER", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Full retirement planner with inflation, taxes, income/expense, and what-if analysis", size=12, color="#aaaaaa"),
            ft.Row([age_field, retire_field, contrib_field, savings_field], spacing=8),
            ft.Row([income_field, expense_field, tax_field, inflation_field, goal_field], spacing=8),
            whatif_label,
            whatif_slider,
            ft.Button("CALCULATE", bgcolor="#00ff9d", color="#0a0a0a",
                      style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
                      on_click=compute),
            ft.Divider(height=1, color="#333333"),
            result_col,
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    def monthly_comparison_tab():
        monthly = data.monthly_comparison()
        if monthly.empty:
            content = ft.Text("No data loaded. Import data to see month-to-month comparison.", size=16, color="#aaaaaa")
        else:
            rows = []
            for _, r in monthly.iterrows():
                chg_color = "#00ff9d" if r.get("records_chg", 0) >= 0 else "#ff3366"
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(str(r.get("year_month", "")), color="#ffffff")),
                    ft.DataCell(ft.Text(str(int(r.get("records", 0))), color="#ffd700")),
                    ft.DataCell(ft.Text(str(int(r.get("symbols", 0))), color="#ffffff")),
                    ft.DataCell(ft.Text(f"{r.get('records_chg', 0):+.1f}%", color=chg_color)),
                ]))
            content = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Month", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Records", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Symbols", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Change %", color="#00ff9d")),
                ],
                rows=rows,
                heading_row_color="#1a1a2e",
            )
        return _card(ft.Column([
            ft.Text("MONTH-TO-MONTH COMPARISON", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            content,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    # ====================== NEW TAB BUILDERS ======================
    def drawdown_tab():
        dd = data.max_drawdown()
        dd_text = f"Max Drawdown: {dd:.1%}" if dd != 0 else "Max Drawdown: N/A (no data)"
        return _card(ft.Column([
            ft.Text("DRAWDOWN ANALYSIS", size=22, weight=ft.FontWeight.BOLD, color="#ff3366"),
            _chart_image(data.chart_drawdown()),
            ft.Text(dd_text, size=16, color="#ff3366"),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def holdings_bar_tab():
        return _card(ft.Column([
            ft.Text("HOLDINGS BAR CHART", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            _chart_image(data.chart_holdings_bar()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def returns_dist_tab():
        return _card(ft.Column([
            ft.Text("RETURNS DISTRIBUTION", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_returns_histogram()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def sector_tab():
        return _card(ft.Column([
            ft.Text("SECTOR BREAKDOWN", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            _chart_image(data.chart_sector_pie()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def gain_loss_tab():
        return _card(ft.Column([
            ft.Text("GAIN / LOSS WATERFALL", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            _chart_image(data.chart_gain_loss_waterfall()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def rolling_sharpe_tab():
        return _card(ft.Column([
            ft.Text("ROLLING SHARPE RATIO", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_rolling_sharpe()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def rolling_vol_tab():
        return _card(ft.Column([
            ft.Text("ROLLING VOLATILITY", size=22, weight=ft.FontWeight.BOLD, color="#ff9900"),
            _chart_image(data.chart_rolling_volatility()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def monthly_heatmap_tab():
        return _card(ft.Column([
            ft.Text("MONTHLY RETURNS HEATMAP", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            _chart_image(data.chart_monthly_heatmap()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def weight_pie_tab():
        return _card(ft.Column([
            ft.Text("WEIGHT ALLOCATION", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            _chart_image(data.chart_weight_pie()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def growth_projection_tab():
        return _card(ft.Column([
            ft.Text("30-YEAR GROWTH PROJECTION", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_growth_projection()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def efficient_frontier_tab():
        return _card(ft.Column([
            ft.Text("EFFICIENT FRONTIER", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_efficient_frontier()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def summary_stats_tab():
        s = data.portfolio_summary_stats()
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(
                content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2),
                bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True,
            )
        def _v(val, fmt=".2f", pct=False, dollar=False):
            if val == 0: return "N/A"
            if dollar: return f"${val:,.0f}"
            if pct: return f"{val:{fmt}}"
            return f"{val:{fmt}}"
        gl = s['total_gain_loss']
        gl_c = "#00ff9d" if gl >= 0 else "#ff3366"
        return _card(ft.Column([
            ft.Text("PORTFOLIO SUMMARY STATS", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Row([_kpi("Total Value", f"${s['total_value']:,.0f}"), _kpi("Holdings", str(s['num_holdings']) if s['num_holdings'] > 0 else "0"), _kpi("Ann. Return", f"{s['ann_return']:.1%}" if s['ann_return'] != 0 else "N/A", "#00ff9d"), _kpi("Ann. Vol", f"{s['ann_volatility']:.1%}" if s['ann_volatility'] != 0 else "N/A", "#ff3366")], spacing=6),
            ft.Row([_kpi("Sharpe", _v(s['sharpe'])), _kpi("Sortino", _v(s['sortino'])), _kpi("Calmar", _v(s['calmar'])), _kpi("Max DD", f"{s['max_drawdown']:.1%}" if s['max_drawdown'] != 0 else "N/A", "#ff3366")], spacing=6),
            ft.Row([_kpi("VaR 95%", f"{s['var_95']:.2%}" if s['var_95'] != 0 else "N/A", "#ff3366"), _kpi("CVaR 95%", f"{s['cvar_95']:.2%}" if s['cvar_95'] != 0 else "N/A", "#ff3366"), _kpi("Win Rate", f"{s['win_rate']:.0%}" if s['win_rate'] != 0 else "N/A", "#00ff9d"), _kpi("Best Day", f"{s['best_day']:.2%}" if s['best_day'] != 0 else "N/A", "#00ff9d")], spacing=6),
            ft.Row([_kpi("Worst Day", f"{s['worst_day']:.2%}" if s['worst_day'] != 0 else "N/A", "#ff3366"), _kpi("Skewness", _v(s['skewness'])), _kpi("Kurtosis", _v(s['kurtosis'])), _kpi("Total G/L", f"${gl:,.0f}" if gl != 0 else "N/A", gl_c)], spacing=6),
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

    def fire_tab():
        age_f = ft.TextField(label="Current Age", value="30", width=120, border_color="#00ff9d", color="#ffffff", text_size=14, dense=True)
        expense_f = ft.TextField(label="Annual Expenses $", value="50000", width=160, border_color="#00ff9d", color="#ffffff", text_size=14, dense=True)
        contrib_f = ft.TextField(label="Monthly Contrib $", value="500", width=160, border_color="#00ff9d", color="#ffffff", text_size=14, dense=True)
        result_col = ft.Column(spacing=8)
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(
                content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2),
                bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True,
            )
        def calc(e):
            try:
                age = int(age_f.value or 30)
                exp = float(expense_f.value or 50000)
                cont = float(contrib_f.value or 500)
            except ValueError:
                return
            f = data.fire_calculator(annual_expenses=exp, current_age=age, monthly_contribution=cont)
            result_col.controls.clear()
            result_col.controls.append(ft.Row([
                _kpi("Lean FIRE", f"${f['lean_fire']:,.0f}"), _kpi("Years", f"{f['lean_years']:.1f}y"),
                _kpi("Regular FIRE", f"${f['regular_fire']:,.0f}", "#00ff9d"), _kpi("Years", f"{f['regular_years']:.1f}y"),
            ], spacing=6))
            result_col.controls.append(ft.Row([
                _kpi("Fat FIRE", f"${f['fat_fire']:,.0f}", "#ffd700"), _kpi("Years", f"{f['fat_years']:.1f}y"),
                _kpi("Coast FIRE", f"${f['coast_fire']:,.0f}"), _kpi("Reached?", "YES" if f['coast_reached'] else "NO", "#00ff9d" if f['coast_reached'] else "#ff3366"),
            ], spacing=6))
            result_col.controls.append(ft.Divider(height=1, color="#333333"))
            result_col.controls.append(ft.Row([
                _kpi("Safe Withdrawal (4%)", f"${f['safe_withdrawal']:,.0f}/yr", "#00ff9d"),
                _kpi("Monthly Passive", f"${f['monthly_passive']:,.0f}/mo", "#00ff9d"),
            ], spacing=6))
            page.update()
        calc(None)
        return _card(ft.Column([
            ft.Text("F.I.R.E. CALCULATOR", size=22, weight=ft.FontWeight.BOLD, color="#ff9900"),
            ft.Text("Financial Independence, Retire Early", size=12, color="#aaaaaa"),
            ft.Row([age_f, expense_f, contrib_f,
                    ft.Button("CALCULATE", bgcolor="#ff9900", color="#0a0a0a",
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
                              on_click=calc)], spacing=10),
            ft.Divider(height=1, color="#333333"),
            result_col,
        ], scroll=ft.ScrollMode.AUTO, spacing=10))

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

    def rebalancer_tab():
        suggestions = data.rebalance_suggestions()
        if not suggestions:
            content = ft.Text("Need 2+ assets to generate rebalance suggestions", size=16, color="#aaaaaa")
        else:
            rows = []
            for sg in suggestions:
                ac = "#00ff9d" if sg['action'] == "BUY" else "#ff3366" if sg['action'] == "SELL" else "#ffd700"
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(sg['symbol'], color="#ffffff", weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{sg['current_weight']:.1%}", color="#ffffff")),
                    ft.DataCell(ft.Text(f"{sg['target_weight']:.1%}", color="#ffd700")),
                    ft.DataCell(ft.Text(sg['action'], color=ac, weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{sg['diff']:+.1%}", color=ac)),
                ]))
            content = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Symbol", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Current", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Target", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Action", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Diff", color="#00ff9d")),
                ],
                rows=rows, heading_row_color="#1a1a2e",
            )
        return _card(ft.Column([
            ft.Text("REBALANCE SUGGESTIONS", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Text("Optimized weights via Mean-Variance Optimization", size=12, color="#aaaaaa"),
            content,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

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
        # Default: Portfolio + all benchmarks ON
        active_lines = {'Portfolio': True}
        for bm_sym in all_benchmarks:
            active_lines[bm_sym] = True
        for sym in user_syms:
            if sym not in active_lines:
                active_lines[sym] = False

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

        # Chart container that will be updated on toggle
        chart_container = ft.Container(
            content=_chart_image(data.chart_benchmark_overlay(benchmarks=all_benchmarks, timerange='ALL')),
            padding=0,
        )

        def _rebuild_chart():
            selected = [s for s, on in active_lines.items() if on and s != 'Portfolio']
            show_portfolio = active_lines.get('Portfolio', True)
            tr = current_timerange['value']
            if not selected and not show_portfolio:
                chart_container.content = ft.Text("Select at least one line to display", color="#aaaaaa", size=14)
            else:
                chart_container.content = _chart_image(
                    data.chart_benchmark_overlay(benchmarks=selected if selected else None,
                                                  show_portfolio=show_portfolio, timerange=tr)
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
                    ft.Switch(value=active_lines.get('Portfolio', True),
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
                        ft.Switch(value=active_lines.get(sym, True),
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

        return _card(ft.Column([
            ft.Text("BENCHMARK COMPARISON", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            table_content,
            ft.Container(
                content=ft.Column([
                    ft.Text("Toggle Lines:", size=13, color="#aaaaaa", weight=ft.FontWeight.BOLD),
                    toggle_row,
                    ft.Divider(height=1, color="#333333"),
                    ft.Text("Time Range:", size=13, color="#aaaaaa", weight=ft.FontWeight.BOLD),
                    timerange_row,
                ], spacing=6),
                bgcolor="#1a1a2e", border_radius=8, padding=10, margin=ft.Margin(top=8, bottom=4),
            ),
            chart_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def trade_log_tab():
        trades = data.get_trade_log(80)
        if not trades:
            content = ft.Text("No trade data loaded. Import data first.", size=16, color="#aaaaaa")
        else:
            rows = []
            for t in trades[:50]:
                gl_c = "#00ff9d" if t['gain_loss'] >= 0 else "#ff3366"
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(t['date'], color="#aaaaaa")),
                    ft.DataCell(ft.Text(t['symbol'], color="#ffffff", weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{t['quantity']:.2f}", color="#ffffff")),
                    ft.DataCell(ft.Text(f"${t['price']:,.2f}", color="#ffd700")),
                    ft.DataCell(ft.Text(f"${t['cost_basis']:,.0f}", color="#aaaaaa")),
                    ft.DataCell(ft.Text(f"${t['gain_loss']:,.0f}", color=gl_c)),
                ]))
            content = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Date", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Symbol", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Qty", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Price", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Cost Basis", color="#00ff9d")),
                    ft.DataColumn(ft.Text("G/L", color="#00ff9d")),
                ],
                rows=rows, heading_row_color="#1a1a2e",
            )
        return _card(ft.Column([
            ft.Text("TRADE LOG", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text(f"Showing latest {min(50, len(trades))} records", size=12, color="#aaaaaa"),
            content,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def income_tracker_tab():
        divs = data.estimate_dividends()
        if not divs:
            content = ft.Text("No holdings loaded. Import data to estimate dividend income.", size=16, color="#aaaaaa")
            total_income = 0
        else:
            rows = []
            total_income = 0
            for d in divs:
                total_income += d['annual_income']
                yld_c = "#00ff9d" if d['div_yield'] > 0.02 else "#ffd700" if d['div_yield'] > 0 else "#aaaaaa"
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(d['symbol'], color="#ffffff", weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{d['quantity']:.2f}", color="#ffffff")),
                    ft.DataCell(ft.Text(f"{d['div_yield']:.2%}", color=yld_c)),
                    ft.DataCell(ft.Text(f"${d['div_rate']:.2f}", color="#ffd700")),
                    ft.DataCell(ft.Text(f"${d['annual_income']:,.0f}", color="#00ff9d")),
                ]))
            content = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Symbol", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Qty", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Yield", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Div/Share", color="#00ff9d")),
                    ft.DataColumn(ft.Text("Annual $", color="#00ff9d")),
                ],
                rows=rows, heading_row_color="#1a1a2e",
            )
        return _card(ft.Column([
            ft.Text("INCOME TRACKER (DIVIDENDS)", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Row([
                ft.Container(content=ft.Column([ft.Text("Est. Annual Income", size=11, color="#aaaaaa"), ft.Text(f"${total_income:,.0f}", size=20, weight=ft.FontWeight.BOLD, color="#00ff9d")], spacing=2), bgcolor="#1a1a2e", padding=12, border_radius=10, expand=True),
                ft.Container(content=ft.Column([ft.Text("Est. Monthly Income", size=11, color="#aaaaaa"), ft.Text(f"${total_income/12:,.0f}", size=20, weight=ft.FontWeight.BOLD, color="#00ff9d")], spacing=2), bgcolor="#1a1a2e", padding=12, border_radius=10, expand=True),
            ], spacing=8),
            content,
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    def whatif_tab():
        ret_field = ft.TextField(label="Ann. Return %", value="10", width=120, border_color="#00ff9d", color="#ffffff", text_size=14, dense=True)
        vol_field = ft.TextField(label="Ann. Vol %", value="20", width=120, border_color="#ff9900", color="#ffffff", text_size=14, dense=True)
        contrib_field = ft.TextField(label="Monthly $", value="500", width=120, border_color="#ffd700", color="#ffffff", text_size=14, dense=True)
        years_field = ft.TextField(label="Years", value="10", width=80, border_color="#00bfff", color="#ffffff", text_size=14, dense=True)
        result_col = ft.Column(spacing=8)
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(value, size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        def compute(e):
            try:
                r = float(ret_field.value or 10) / 100
                v = float(vol_field.value or 20) / 100
                c = float(contrib_field.value or 500)
                y = int(years_field.value or 10)
            except ValueError:
                return
            wi = data.what_if_scenario(ann_return_override=r, ann_vol_override=v, monthly_contrib=c, years=y)
            result_col.controls.clear()
            result_col.controls.append(ft.Row([
                _kpi("Base Case", f"${wi['final_base']:,.0f}", "#00ff9d"),
                _kpi("Bear Case", f"${wi['final_low']:,.0f}", "#ff3366"),
                _kpi("Bull Case", f"${wi['final_high']:,.0f}", "#ffd700"),
            ], spacing=6))
            result_col.controls.append(ft.Row([
                _kpi("Total Contributed", f"${wi['total_contributions']:,.0f}"),
                _kpi("Growth Earned", f"${wi['total_growth']:,.0f}", "#00ff9d" if wi['total_growth'] > 0 else "#ff3366"),
            ], spacing=6))
            result_col.controls.append(_chart_image(data.chart_what_if(wi['projection'])))
            result_col.controls.append(ft.Text("YEAR-BY-YEAR PROJECTION", size=14, weight=ft.FontWeight.BOLD, color="#ffd700"))
            for p in wi['projection']:
                result_col.controls.append(ft.Row([
                    ft.Text(f"Yr {p['year']}", size=12, color="#aaaaaa", width=50),
                    ft.Text(f"${p['base']:,.0f}", size=12, color="#00ff9d", width=100),
                    ft.Text(f"${p['low']:,.0f}", size=12, color="#ff3366", width=100),
                    ft.Text(f"${p['high']:,.0f}", size=12, color="#ffd700", width=100),
                ], spacing=6))
            page.update()
        compute(None)
        return _card(ft.Column([
            ft.Text("WHAT-IF SCENARIO", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Adjust return, volatility, contributions, and time horizon", size=12, color="#aaaaaa"),
            ft.Row([ret_field, vol_field, contrib_field, years_field,
                    ft.Button("RUN SCENARIO", bgcolor="#00ff9d", color="#0a0a0a",
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
                              on_click=compute)], spacing=10),
            ft.Divider(height=1, color="#333333"),
            result_col,
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    def tax_optimizer_tab():
        candidates = data.tax_loss_harvest()
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(value, size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        if not candidates:
            total_savings = 0
            content = ft.Text("No tax-loss harvesting candidates. All positions in gain or no data.", size=14, color="#aaaaaa")
        else:
            total_savings = sum(c['est_tax_savings'] for c in candidates)
            rows = []
            for c in candidates:
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(c['symbol'], color="#ffffff", weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{c['quantity']:.2f}", color="#ffffff")),
                    ft.DataCell(ft.Text(f"${c['loss']:,.0f}", color="#ff3366")),
                    ft.DataCell(ft.Text(f"{c['loss_pct']:.1f}%", color="#ff3366")),
                    ft.DataCell(ft.Text(f"${c['est_tax_savings']:,.0f}", color="#00ff9d")),
                ]))
            content = ft.DataTable(
                columns=[ft.DataColumn(ft.Text("Symbol", color="#00ff9d")), ft.DataColumn(ft.Text("Qty", color="#00ff9d")),
                         ft.DataColumn(ft.Text("Loss", color="#00ff9d")), ft.DataColumn(ft.Text("Loss %", color="#00ff9d")),
                         ft.DataColumn(ft.Text("Est. Savings", color="#00ff9d"))],
                rows=rows, heading_row_color="#1a1a2e",
            )
        return _card(ft.Column([
            ft.Text("TAX-LOSS HARVESTING OPTIMIZER", size=22, weight=ft.FontWeight.BOLD, color="#ff3366"),
            ft.Text("Identifies losing positions to offset capital gains (22% tax rate)", size=12, color="#aaaaaa"),
            ft.Row([_kpi("Candidates", str(len(candidates))), _kpi("Potential Savings", f"${total_savings:,.0f}", "#00ff9d")], spacing=6),
            ft.Divider(height=1, color="#333333"),
            content,
        ], scroll=ft.ScrollMode.AUTO, spacing=10))

    def correlation_dive_tab():
        cd = data.correlation_deep_dive()
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(value, size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        most = cd['most_correlated']
        least = cd['least_correlated']
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
        return _card(ft.Column([
            ft.Text("CORRELATION DEEP-DIVE", size=22, weight=ft.FontWeight.BOLD, color="#00bfff"),
            ft.Text("Pairwise correlations — lower = better diversification", size=12, color="#aaaaaa"),
            ft.Row([
                _kpi("Avg Corr", f"{cd['avg_corr']:.3f}" if cd['avg_corr'] != 0 else "N/A"),
                _kpi("Most Correlated", f"{most[0]}/{most[1]}: {most[2]:.2f}" if most[0] != "N/A" else "N/A", "#ff3366"),
                _kpi("Least Correlated", f"{least[0]}/{least[1]}: {least[2]:.2f}" if least[0] != "N/A" else "N/A", "#00ff9d"),
            ], spacing=6),
            ft.Divider(height=1, color="#333333"),
            tbl,
            _chart_image(data.chart_correlation_matrix()),
        ], scroll=ft.ScrollMode.AUTO, spacing=10))

    def health_score_tab():
        hs = data.portfolio_health_score()
        d = hs['details']
        def _bar(label, pts, max_pts, color="#00ff9d"):
            pct = pts / max_pts if max_pts > 0 else 0
            return ft.Row([
                ft.Text(label, size=12, color="#aaaaaa", width=100),
                ft.ProgressBar(value=pct, color=color, bgcolor="#1a1a2e", expand=True),
                ft.Text(f"{pts}/{max_pts}", size=12, color="#ffffff", width=50),
            ], spacing=8)
        return _card(ft.Column([
            ft.Text("PORTFOLIO HEALTH SCORE", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Composite score from 7 weighted components", size=12, color="#aaaaaa"),
            ft.Row([
                _chart_image(data.chart_health_gauge(hs['score'])),
                ft.Column([
                    ft.Text(f"Grade: {hs['grade']}", size=28, weight=ft.FontWeight.BOLD, color="#00ff9d" if hs['score'] >= 65 else "#ffd700" if hs['score'] >= 50 else "#ff3366"),
                    ft.Text(f"Sharpe: {d['sharpe']:.2f}  |  Sortino: {d['sortino']:.2f}", size=12, color="#aaaaaa"),
                    ft.Text(f"Vol: {d['vol']:.1%}  |  Max DD: {d['max_dd']:.1%}  |  PF: {d['profit_factor']:.2f}", size=12, color="#aaaaaa"),
                ], spacing=4, expand=True),
            ], spacing=10),
            ft.Divider(height=1, color="#333333"),
            ft.Text("SCORE BREAKDOWN", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"),
            _bar("Sharpe", d['sharpe_pts'], 20, "#00ff9d"),
            _bar("Sortino", d['sortino_pts'], 15, "#00ff9d"),
            _bar("Low Volatility", d['vol_pts'], 15, "#00bfff"),
            _bar("Low Drawdown", d['dd_pts'], 15, "#ffd700"),
            _bar("Profit Factor", d['pf_pts'], 15, "#ff9900"),
            _bar("Diversification", d['div_pts'], 10, "#9966ff"),
            _bar("Win Streak", d['streak_pts'], 10, "#00ff9d"),
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    def risk_returns_tab():
        ra = data.risk_adjusted_returns()
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(value, size=16, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True)
        def _fmt(v, fmt=".2f"):
            if v == 0: return "N/A"
            return f"{v:{fmt}}"
        return _card(ft.Column([
            ft.Text("RISK-ADJUSTED RETURNS", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Text("CAPM alpha, beta, Treynor, information ratio", size=12, color="#aaaaaa"),
            ft.Row([
                _kpi("Ann. Return", f"{ra['ann_return']:.1%}" if ra['ann_return'] != 0 else "N/A", "#00ff9d" if ra['ann_return'] > 0 else "#ff3366"),
                _kpi("Ann. Volatility", f"{ra['ann_vol']:.1%}" if ra['ann_vol'] != 0 else "N/A", "#ff3366"),
                _kpi("Sharpe", _fmt(ra['sharpe']), "#00ff9d" if ra['sharpe'] > 0 else "#ff3366"),
                _kpi("Sortino", _fmt(ra['sortino']), "#00ff9d" if ra['sortino'] > 0 else "#ff3366"),
            ], spacing=6),
            ft.Row([
                _kpi("Calmar", _fmt(ra['calmar']), "#00ff9d" if ra['calmar'] > 0 else "#ff3366"),
                _kpi("Max DD", f"{ra['max_dd']:.1%}" if ra['max_dd'] != 0 else "N/A", "#ff3366"),
                _kpi("Profit Factor", _fmt(ra['profit_factor']), "#00ff9d" if ra['profit_factor'] > 1 else "#ff3366"),
                _kpi("VaR 95%", f"{ra['var_95']:.2%}" if ra['var_95'] != 0 else "N/A", "#ff3366"),
            ], spacing=6),
            ft.Divider(height=1, color="#333333"),
            ft.Text("CAPM METRICS", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Row([
                _kpi("Beta (vs SPY)", _fmt(ra['beta']), "#ffd700"),
                _kpi("Alpha (CAPM)", f"{ra['alpha']:.2%}" if ra['alpha'] != 0 else "N/A", "#00ff9d" if ra['alpha'] > 0 else "#ff3366"),
                _kpi("Treynor", _fmt(ra['treynor']), "#00ff9d" if ra['treynor'] > 0 else "#ff3366"),
                _kpi("Info Ratio", _fmt(ra['info_ratio']), "#00ff9d" if ra['info_ratio'] > 0 else "#ff3366"),
            ], spacing=6),
            ft.Text("Beta = market sensitivity. Alpha = excess return vs CAPM. Treynor = return per unit systematic risk.", size=11, color="#666666", italic=True),
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    def streaks_tab():
        wl = data.win_loss_streaks()
        rec = data.recovery_time()
        s = data.portfolio_summary_stats()
        def _kpi(label, value, color="#ffd700"):
            return ft.Container(content=ft.Column([ft.Text(label, size=10, color="#aaaaaa"), ft.Text(str(value), size=18, weight=ft.FontWeight.BOLD, color=color)], spacing=2), bgcolor="#1a1a2e", padding=12, border_radius=10, expand=True)
        pos = s.get("positive_days", 0)
        neg = s.get("negative_days", 0)
        skew = s.get("skewness", 0)
        kurt = s.get("kurtosis", 0)
        return _card(ft.Column([
            ft.Text("WIN/LOSS STREAKS & STATS", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Trading day performance and streak analysis", size=12, color="#aaaaaa"),
            ft.Row([
                _kpi("Longest Win", f"{wl['longest_win']}d" if wl['longest_win'] > 0 else "N/A", "#00ff9d"),
                _kpi("Longest Loss", f"{wl['longest_loss']}d" if wl['longest_loss'] > 0 else "N/A", "#ff3366"),
                _kpi("Current", f"{wl['current_streak']}d ({wl['current_type']})" if wl['current_type'] != "N/A" else "N/A", "#00ff9d" if wl['current_type'] == "win" else "#ff3366"),
            ], spacing=6),
            ft.Row([
                _kpi("Positive Days", str(pos) if pos > 0 else "N/A", "#00ff9d"),
                _kpi("Negative Days", str(neg) if neg > 0 else "N/A", "#ff3366"),
                _kpi("Win Rate", f"{pos/(pos+neg):.0%}" if (pos + neg) > 0 else "N/A", "#00ff9d" if pos > neg else "#ff3366"),
            ], spacing=6),
            ft.Divider(height=1, color="#333333"),
            ft.Text("DRAWDOWN RECOVERY", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Row([
                _kpi("Max Recovery", f"{rec['max_recovery_days']}d" if rec['max_recovery_days'] > 0 else "N/A"),
                _kpi("In Drawdown?", "YES" if rec['currently_in_drawdown'] else "NO", "#ff3366" if rec['currently_in_drawdown'] else "#00ff9d"),
                _kpi("Current DD", f"{rec['current_dd_days']}d" if rec['current_dd_days'] > 0 else "N/A", "#ff3366"),
            ], spacing=6),
            ft.Divider(height=1, color="#333333"),
            ft.Text("DISTRIBUTION", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Row([
                _kpi("Skewness", f"{skew:.3f}" if skew != 0 else "N/A", "#00ff9d" if skew > 0 else "#ff3366"),
                _kpi("Kurtosis", f"{kurt:.3f}" if kurt != 0 else "N/A", "#ffd700"),
            ], spacing=6),
            ft.Text("Negative skew = more left-tail risk. High kurtosis = fat tails.", size=11, color="#666666", italic=True),
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    # ====================== PROFIT CHARTS TAB ======================
    def profit_charts_tab():
        return _card(ft.Column([
            ft.Text("PROFIT CHARTS", size=20, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Daily, weekly, and rolling weekly profit/loss", size=12, color="#888888"),
            ft.Divider(height=1, color="#333333"),
            ft.Text("DAILY PROFIT / LOSS", size=15, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_daily_profit()),
            ft.Divider(height=1, color="#333333"),
            ft.Text("WEEKLY PROFIT / LOSS", size=15, weight=ft.FontWeight.BOLD, color="#ffd700"),
            _chart_image(data.chart_weekly_profit()),
            ft.Divider(height=1, color="#333333"),
            ft.Text("ROLLING 5-DAY (WEEKLY) PROFIT", size=15, weight=ft.FontWeight.BOLD, color="#00bfff"),
            _chart_image(data.chart_rolling_weekly_profit()),
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    # ====================== ASSET INSPECTION TAB ======================
    def asset_inspection_tab():
        symbols = data.holdings['symbol'].tolist() if not data.holdings.empty else []
        result_col = ft.Column(spacing=8, scroll=ft.ScrollMode.AUTO)
        if not symbols:
            result_col.controls.append(ft.Text("No holdings loaded. Import data first.", size=14, color="#aaaaaa"))

        def on_inspect(e):
            sym = dropdown.value
            if not sym:
                return
            result_col.controls.clear()
            result_col.controls.append(ft.Text(f"Inspecting {sym}...", size=14, color="#aaaaaa"))
            page.update()
            try:
                info = data.inspect_asset(sym)
                result_col.controls.clear()
                if not info.get("found"):
                    result_col.controls.append(ft.Text(f"No price data found for {sym}", size=14, color="#ff3366"))
                else:
                    def _kv(label, value, color="#ffd700"):
                        return ft.Container(
                            content=ft.Column([
                                ft.Text(label, size=10, color="#aaaaaa"),
                                ft.Text(str(value), size=16, weight=ft.FontWeight.BOLD, color=color),
                            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True,
                        )
                    result_col.controls.append(ft.Text(f"ASSET: {sym}", size=20, weight=ft.FontWeight.BOLD, color="#ffd700"))
                    result_col.controls.append(ft.Row([
                        _kv("Price", f"${info.get('current_price', 0):,.2f}", "#00ff9d"),
                        _kv("Total Return", f"{info.get('total_return', 0):.1%}", "#00ff9d" if info.get('total_return', 0) >= 0 else "#ff3366"),
                        _kv("Ann. Return", f"{info.get('ann_return', 0):.1%}", "#ffd700"),
                    ], spacing=6))
                    result_col.controls.append(ft.Row([
                        _kv("Ann. Vol", f"{info.get('ann_vol', 0):.1%}", "#ff9900"),
                        _kv("Sharpe", f"{info.get('sharpe', 0):.2f}", "#00bfff"),
                        _kv("Max DD", f"{info.get('max_dd', 0):.1%}", "#ff3366"),
                    ], spacing=6))
                    result_col.controls.append(ft.Row([
                        _kv("Best Day", f"{info.get('best_day', 0):.2%}", "#00ff9d"),
                        _kv("Worst Day", f"{info.get('worst_day', 0):.2%}", "#ff3366"),
                        _kv("Win Rate", f"{info.get('win_rate', 0):.0%}", "#ffd700"),
                    ], spacing=6))
                    result_col.controls.append(ft.Row([
                        _kv("Avg Daily P/L", f"${info.get('avg_daily_pnl', 0):,.2f}", "#00ff9d" if info.get('avg_daily_pnl', 0) >= 0 else "#ff3366"),
                        _kv("Avg Weekly P/L", f"${info.get('avg_weekly_pnl', 0):,.2f}", "#00ff9d" if info.get('avg_weekly_pnl', 0) >= 0 else "#ff3366"),
                    ], spacing=6))
                    result_col.controls.append(ft.Divider(height=1, color="#333333"))
                    result_col.controls.append(_chart_image(data.chart_asset_inspection(sym)))
            except Exception as ex:
                result_col.controls.clear()
                result_col.controls.append(ft.Text(f"Error inspecting {sym}: {ex}", size=14, color="#ff3366"))
            page.update()

        dropdown = ft.Dropdown(
            label="Select Asset",
            options=[ft.dropdown.Option(s) for s in symbols],
            width=250,
            border_color="#00ff9d",
            color="#ffffff",
            text_size=14,
        )
        inspect_btn = ft.Button("INSPECT", bgcolor="#00ff9d", color="#000000", on_click=on_inspect)

        return _card(ft.Column([
            ft.Text("ASSET INSPECTION", size=20, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Text("Deep-dive analysis of a single asset: price, returns, drawdown, rolling profit", size=12, color="#888888"),
            ft.Divider(height=1, color="#333333"),
            result_col,
        ], scroll=ft.ScrollMode.AUTO, spacing=8, expand=True))

    # ====================== LIVE CHART TAB (30s auto-refresh + timeframe toggles) ======================
    def live_chart_tab():
        symbol_field = ft.TextField(label="Symbol", value="SPY", width=120, border_color="#00ff9d", color="#ffffff", text_size=14, dense=True)
        tf_state = {"current": "24h"}
        chart_container = ft.Column(spacing=8)
        status_text = ft.Text("", size=11, color="#aaaaaa")
        refresh_flag = {"running": False, "stop": False}

        def render_chart():
            sym = (symbol_field.value or "SPY").strip().upper()
            tf = tf_state["current"]
            status_text.value = f"Loading {sym} ({tf})..."
            page.update()
            img = data.chart_live_ticker(sym, tf)
            chart_container.controls.clear()
            chart_container.controls.append(_chart_image(img))
            from datetime import datetime as _dt
            status_text.value = f"Last update: {_dt.now().strftime('%H:%M:%S')} — auto-refresh every 30s"
            page.update()

        def on_tf(e, tf):
            tf_state["current"] = tf
            render_chart()

        def on_symbol(e):
            render_chart()

        def auto_refresh():
            if refresh_flag["running"]:
                return
            refresh_flag["running"] = True
            while not refresh_flag["stop"]:
                try:
                    render_chart()
                    time.sleep(30)
                except Exception:
                    break
            refresh_flag["running"] = False

        def start_refresh(e):
            refresh_flag["stop"] = False
            threading.Thread(target=auto_refresh, daemon=True).start()

        def stop_refresh(e):
            refresh_flag["stop"] = True
            status_text.value = "Auto-refresh stopped"
            page.update()

        tf_buttons = ft.Row([
            ft.Button("1H", bgcolor="#1a1a2e" if tf_state["current"] != "1h" else "#00ff9d", color="#ffffff" if tf_state["current"] != "1h" else "#000000", on_click=lambda e: on_tf(e, "1h")),
            ft.Button("24H", bgcolor="#00ff9d", color="#000000", on_click=lambda e: on_tf(e, "24h")),
            ft.Button("1M", bgcolor="#1a1a2e", color="#ffffff", on_click=lambda e: on_tf(e, "1mo")),
            ft.Button("1Y", bgcolor="#1a1a2e", color="#ffffff", on_click=lambda e: on_tf(e, "1y")),
        ], spacing=6)

        render_chart()
        threading.Thread(target=auto_refresh, daemon=True).start()

        return _card(ft.Column([
            ft.Text("LIVE CHART", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Real-time price + volume with 30s auto-refresh", size=12, color="#aaaaaa"),
            ft.Row([symbol_field, ft.Button("GO", bgcolor="#00ff9d", color="#000000", on_click=on_symbol), tf_buttons,
                    ft.Button("STOP", bgcolor="#ff3366", color="#ffffff", on_click=stop_refresh),
                    ft.Button("START", bgcolor="#006644", color="#ffffff", on_click=start_refresh)], spacing=8),
            status_text,
            ft.Divider(height=1, color="#333333"),
            chart_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    # ====================== ADVANCED TECHNICAL INDICATORS TAB ======================
    def advanced_indicators_tab():
        symbol_field = ft.TextField(label="Symbol", value="SPY", width=120, border_color="#ffd700", color="#ffffff", text_size=14, dense=True)
        tf_state = {"current": "1y"}
        ind_state = {"sma20": True, "sma50": True, "sma200": False, "ema12": False, "ema26": False,
                     "bollinger": True, "rsi": True, "macd": True, "vwap": False, "volume": False}
        chart_container = ft.Column(spacing=8)

        def render():
            sym = (symbol_field.value or "SPY").strip().upper()
            tf = tf_state["current"]
            img = data.chart_advanced_technical(sym, tf, ind_state)
            chart_container.controls.clear()
            chart_container.controls.append(_chart_image(img))
            page.update()

        def toggle_ind(key):
            def handler(e):
                ind_state[key] = e.control.value
                render()
            return handler

        def on_tf(e, tf):
            tf_state["current"] = tf
            render()

        def on_symbol(e):
            render()

        tf_row = ft.Row([
            ft.Button("1H", bgcolor="#1a1a2e", color="#ffffff", on_click=lambda e: on_tf(e, "1h")),
            ft.Button("24H", bgcolor="#1a1a2e", color="#ffffff", on_click=lambda e: on_tf(e, "24h")),
            ft.Button("1M", bgcolor="#1a1a2e", color="#ffffff", on_click=lambda e: on_tf(e, "1mo")),
            ft.Button("1Y", bgcolor="#ffd700", color="#000000", on_click=lambda e: on_tf(e, "1y")),
        ], spacing=6)

        toggles = ft.Row([
            ft.Checkbox(label="SMA 20", value=ind_state["sma20"], on_change=toggle_ind("sma20"), check_color="#ffd700", label_style=ft.TextStyle(color="#ffffff", size=11)),
            ft.Checkbox(label="SMA 50", value=ind_state["sma50"], on_change=toggle_ind("sma50"), check_color="#00bfff", label_style=ft.TextStyle(color="#ffffff", size=11)),
            ft.Checkbox(label="SMA 200", value=ind_state["sma200"], on_change=toggle_ind("sma200"), check_color="#ff9900", label_style=ft.TextStyle(color="#ffffff", size=11)),
            ft.Checkbox(label="EMA 12", value=ind_state["ema12"], on_change=toggle_ind("ema12"), check_color="#9966ff", label_style=ft.TextStyle(color="#ffffff", size=11)),
            ft.Checkbox(label="EMA 26", value=ind_state["ema26"], on_change=toggle_ind("ema26"), check_color="#ff6699", label_style=ft.TextStyle(color="#ffffff", size=11)),
        ], spacing=4, wrap=True)
        toggles2 = ft.Row([
            ft.Checkbox(label="Bollinger", value=ind_state["bollinger"], on_change=toggle_ind("bollinger"), check_color="#00ff9d", label_style=ft.TextStyle(color="#ffffff", size=11)),
            ft.Checkbox(label="RSI", value=ind_state["rsi"], on_change=toggle_ind("rsi"), check_color="#ffd700", label_style=ft.TextStyle(color="#ffffff", size=11)),
            ft.Checkbox(label="MACD", value=ind_state["macd"], on_change=toggle_ind("macd"), check_color="#00bfff", label_style=ft.TextStyle(color="#ffffff", size=11)),
            ft.Checkbox(label="VWAP", value=ind_state["vwap"], on_change=toggle_ind("vwap"), check_color="#ff00ff", label_style=ft.TextStyle(color="#ffffff", size=11)),
            ft.Checkbox(label="Volume", value=ind_state["volume"], on_change=toggle_ind("volume"), check_color="#00ff9d", label_style=ft.TextStyle(color="#ffffff", size=11)),
        ], spacing=4, wrap=True)

        render()

        return _card(ft.Column([
            ft.Text("ADVANCED TECHNICAL ANALYSIS", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Text("Toggle indicators: SMA, EMA, Bollinger Bands, RSI, MACD, VWAP, Volume", size=12, color="#aaaaaa"),
            ft.Row([symbol_field, ft.Button("ANALYZE", bgcolor="#ffd700", color="#000000", on_click=on_symbol), tf_row], spacing=8),
            toggles,
            toggles2,
            ft.Divider(height=1, color="#333333"),
            chart_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    # ====================== ASSET FOCUS TAB (per-asset toggleable chart) ======================
    def asset_focus_tab():
        all_syms = list(data.prices.columns) if not data.prices.empty and data.prices.ndim == 2 else []
        sym_state = {s: True for s in all_syms[:12]}
        for s in all_syms[12:]:
            sym_state[s] = False
        norm_state = {"normalize": True}
        chart_container = ft.Column(spacing=8)
        toggles_container = ft.Row(spacing=4, wrap=True)
        clist = ['#00ff9d', '#ffd700', '#ff3366', '#00bfff', '#ff9900', '#9966ff',
                 '#ff6699', '#33cccc', '#ff00ff', '#66ff66', '#ff6600', '#6699ff']

        def render():
            selected = [s for s, on in sym_state.items() if on]
            img = data.chart_multi_asset(selected_symbols=selected, normalize=norm_state["normalize"])
            chart_container.controls.clear()
            chart_container.controls.append(_chart_image(img))
            page.update()

        def toggle_sym(sym):
            def handler(e):
                sym_state[sym] = e.control.value
                render()
            return handler

        def toggle_norm(e):
            norm_state["normalize"] = e.control.value
            render()

        def select_all(e):
            for s in sym_state:
                sym_state[s] = True
            rebuild_toggles()
            render()

        def deselect_all(e):
            for s in sym_state:
                sym_state[s] = False
            rebuild_toggles()
            render()

        def rebuild_toggles():
            toggles_container.controls.clear()
            for i, sym in enumerate(all_syms):
                color = clist[i % len(clist)]
                toggles_container.controls.append(
                    ft.Checkbox(
                        label=sym, value=sym_state.get(sym, False),
                        on_change=toggle_sym(sym),
                        check_color=color,
                        label_style=ft.TextStyle(color=color, size=11, weight=ft.FontWeight.BOLD),
                    )
                )
            page.update()

        rebuild_toggles()
        if all_syms:
            render()
        else:
            chart_container.controls.append(ft.Text("No multi-asset price data. Import data with multiple symbols first.", size=14, color="#aaaaaa"))

        return _card(ft.Column([
            ft.Text("ASSET FOCUS", size=22, weight=ft.FontWeight.BOLD, color="#ffd700"),
            ft.Text("Toggle individual assets on/off to compare performance", size=12, color="#aaaaaa"),
            ft.Row([
                ft.Checkbox(label="Normalize (base=100)", value=norm_state["normalize"], on_change=toggle_norm, check_color="#ffd700", label_style=ft.TextStyle(color="#ffffff", size=12)),
                ft.Button("Select All", bgcolor="#1a1a2e", color="#00ff9d", on_click=select_all),
                ft.Button("Deselect All", bgcolor="#1a1a2e", color="#ff3366", on_click=deselect_all),
                ft.Text(f"{len(all_syms)} assets available", size=11, color="#888888"),
            ], spacing=8),
            ft.Divider(height=1, color="#333333"),
            toggles_container,
            ft.Divider(height=1, color="#333333"),
            chart_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    # ====================== BOOK BALANCER TAB ======================
    def book_balancer_tab():
        entries_col = ft.Column(spacing=6, scroll=ft.ScrollMode.AUTO)
        summary_col = ft.Column(spacing=6)
        date_f = ft.TextField(label="Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"), border_color="#00ff9d", color="#ffffff", width=160, dense=True)
        desc_f = ft.TextField(label="Description", border_color="#00ff9d", color="#ffffff", expand=True, dense=True)
        debit_f = ft.TextField(label="Debit $", border_color="#ff3366", color="#ffffff", width=120, dense=True, value="0")
        credit_f = ft.TextField(label="Credit $", border_color="#00ff9d", color="#ffffff", width=120, dense=True, value="0")
        acct_f = ft.TextField(label="Account", border_color="#00ff9d", color="#ffffff", width=150, dense=True, value="General")
        cat_f = ft.TextField(label="Category", border_color="#00ff9d", color="#ffffff", width=150, dense=True)

        def refresh_book():
            entries_col.controls.clear()
            for ent in data.book_entries[-50:]:
                d_color = "#ff3366" if ent['debit'] > 0 else "#555555"
                c_color = "#00ff9d" if ent['credit'] > 0 else "#555555"
                entries_col.controls.append(ft.Row([
                    ft.Text(ent['date'], size=11, color="#aaaaaa", width=90),
                    ft.Text(ent['description'], size=11, color="#ffffff", expand=True),
                    ft.Text(ent['account'], size=11, color="#00bfff", width=90),
                    ft.Text(ent['category'], size=11, color="#9966ff", width=90),
                    ft.Text(f"${ent['debit']:,.2f}", size=11, color=d_color, width=90),
                    ft.Text(f"${ent['credit']:,.2f}", size=11, color=c_color, width=90),
                ], spacing=4))
            summary_col.controls.clear()
            s = data.get_book_summary()
            bal_color = "#00ff9d" if s['balance'] >= 0 else "#ff3366"
            status = "BALANCED" if s['balanced'] else "UNBALANCED"
            status_color = "#00ff9d" if s['balanced'] else "#ff3366"
            summary_col.controls.append(ft.Row([
                ft.Container(content=ft.Column([ft.Text("Total Debit", size=10, color="#aaaaaa"), ft.Text(f"${s['total_debit']:,.2f}", size=18, weight=ft.FontWeight.BOLD, color="#ff3366")], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True),
                ft.Container(content=ft.Column([ft.Text("Total Credit", size=10, color="#aaaaaa"), ft.Text(f"${s['total_credit']:,.2f}", size=18, weight=ft.FontWeight.BOLD, color="#00ff9d")], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True),
                ft.Container(content=ft.Column([ft.Text("Balance", size=10, color="#aaaaaa"), ft.Text(f"${s['balance']:,.2f}", size=18, weight=ft.FontWeight.BOLD, color=bal_color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True),
                ft.Container(content=ft.Column([ft.Text("Status", size=10, color="#aaaaaa"), ft.Text(status, size=18, weight=ft.FontWeight.BOLD, color=status_color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True),
                ft.Container(content=ft.Column([ft.Text("Entries", size=10, color="#aaaaaa"), ft.Text(str(s['entries']), size=18, weight=ft.FontWeight.BOLD, color="#ffd700")], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True),
            ], spacing=6))
            if s['by_account']:
                acct_rows = []
                for acc, vals in s['by_account'].items():
                    net = vals['credit'] - vals['debit']
                    nc = "#00ff9d" if net >= 0 else "#ff3366"
                    acct_rows.append(ft.Row([
                        ft.Text(acc, size=12, color="#00bfff", width=120, weight=ft.FontWeight.BOLD),
                        ft.Text(f"D: ${vals['debit']:,.2f}", size=11, color="#ff3366", width=120),
                        ft.Text(f"C: ${vals['credit']:,.2f}", size=11, color="#00ff9d", width=120),
                        ft.Text(f"Net: ${net:,.2f}", size=11, color=nc, width=120),
                    ], spacing=4))
                summary_col.controls.append(ft.Text("BY ACCOUNT", size=13, weight=ft.FontWeight.BOLD, color="#ffd700"))
                summary_col.controls.extend(acct_rows)
            page.update()

        def add_entry(e):
            try:
                data.add_book_entry(
                    date_f.value or datetime.now().strftime("%Y-%m-%d"),
                    desc_f.value or "Entry",
                    float(debit_f.value or 0),
                    float(credit_f.value or 0),
                    acct_f.value or "General",
                    cat_f.value or "",
                )
                desc_f.value = ""
                debit_f.value = "0"
                credit_f.value = "0"
                refresh_book()
            except Exception as ex:
                show_snack(f"Error: {ex}", "#663300")

        refresh_book()
        return _card(ft.Column([
            ft.Text("BOOK BALANCER", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Double-entry ledger: track debits, credits, and account balances", size=12, color="#aaaaaa"),
            summary_col,
            ft.Divider(height=1, color="#333333"),
            ft.Row([date_f, desc_f, debit_f, credit_f, acct_f, cat_f, ft.IconButton(icon=ft.Icons.ADD_CIRCLE, icon_color="#00ff9d", on_click=add_entry)], spacing=4),
            ft.Divider(height=1, color="#333333"),
            entries_col,
        ], scroll=ft.ScrollMode.AUTO, spacing=8))

    # ====================== BUDGET PLANNER TAB ======================
    def budget_planner_tab():
        items_col = ft.Column(spacing=6, scroll=ft.ScrollMode.AUTO)
        summary_col = ft.Column(spacing=6)
        chart_container = ft.Column()
        name_f = ft.TextField(label="Name", border_color="#00ff9d", color="#ffffff", expand=True, dense=True)
        amt_f = ft.TextField(label="Amount $", border_color="#00ff9d", color="#ffffff", width=120, dense=True, value="0")
        type_dd = ft.Dropdown(options=[ft.dropdown.Option("income"), ft.dropdown.Option("expense")], value="expense", label="Type", border_color="#00ff9d", color="#ffffff", width=120, dense=True)
        freq_dd = ft.Dropdown(options=[ft.dropdown.Option("weekly"), ft.dropdown.Option("biweekly"), ft.dropdown.Option("monthly"), ft.dropdown.Option("quarterly"), ft.dropdown.Option("yearly")], value="monthly", label="Frequency", border_color="#00ff9d", color="#ffffff", width=130, dense=True)
        cat_f = ft.TextField(label="Category", border_color="#00ff9d", color="#ffffff", width=140, dense=True, value="General")

        def refresh_budget():
            items_col.controls.clear()
            for item in data.budget_items:
                ic = "#00ff9d" if item['type'] == 'income' else "#ff3366"
                items_col.controls.append(ft.Row([
                    ft.Text(item['name'], size=12, color="#ffffff", expand=True),
                    ft.Text(item['type'].upper(), size=11, color=ic, width=70),
                    ft.Text(f"${item['amount']:,.2f}", size=12, color=ic, width=100),
                    ft.Text(item['frequency'], size=11, color="#aaaaaa", width=80),
                    ft.Text(item['category'], size=11, color="#9966ff", width=100),
                ], spacing=4))
            summary_col.controls.clear()
            s = data.get_budget_summary()
            net_color = "#00ff9d" if s['monthly_net'] >= 0 else "#ff3366"
            summary_col.controls.append(ft.Row([
                ft.Container(content=ft.Column([ft.Text("Monthly Income", size=10, color="#aaaaaa"), ft.Text(f"${s['monthly_income']:,.0f}", size=18, weight=ft.FontWeight.BOLD, color="#00ff9d")], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True),
                ft.Container(content=ft.Column([ft.Text("Monthly Expenses", size=10, color="#aaaaaa"), ft.Text(f"${s['monthly_expenses']:,.0f}", size=18, weight=ft.FontWeight.BOLD, color="#ff3366")], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True),
                ft.Container(content=ft.Column([ft.Text("Monthly Net", size=10, color="#aaaaaa"), ft.Text(f"${s['monthly_net']:,.0f}", size=18, weight=ft.FontWeight.BOLD, color=net_color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True),
                ft.Container(content=ft.Column([ft.Text("Savings Rate", size=10, color="#aaaaaa"), ft.Text(f"{s['savings_rate']:.1f}%", size=18, weight=ft.FontWeight.BOLD, color="#ffd700")], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True),
                ft.Container(content=ft.Column([ft.Text("Annual Net", size=10, color="#aaaaaa"), ft.Text(f"${s['annual_net']:,.0f}", size=18, weight=ft.FontWeight.BOLD, color=net_color)], spacing=2), bgcolor="#1a1a2e", padding=10, border_radius=10, expand=True),
            ], spacing=6))
            chart_container.controls.clear()
            if data.budget_items:
                chart_container.controls.append(_chart_image(data.chart_budget_breakdown()))
            page.update()

        def add_item(e):
            try:
                data.add_budget_item(
                    name_f.value or "Item",
                    float(amt_f.value or 0),
                    type_dd.value or "expense",
                    freq_dd.value or "monthly",
                    cat_f.value or "General",
                )
                name_f.value = ""
                amt_f.value = "0"
                refresh_budget()
            except Exception as ex:
                show_snack(f"Error: {ex}", "#663300")

        refresh_budget()
        return _card(ft.Column([
            ft.Text("BUDGET PLANNER", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Plan income, expenses, and savings with automatic frequency conversion", size=12, color="#aaaaaa"),
            summary_col,
            ft.Divider(height=1, color="#333333"),
            ft.Row([name_f, amt_f, type_dd, freq_dd, cat_f, ft.IconButton(icon=ft.Icons.ADD_CIRCLE, icon_color="#00ff9d", on_click=add_item)], spacing=4),
            ft.Divider(height=1, color="#333333"),
            items_col,
            chart_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=8))


    # ====================== ENTERPRISE / GOVERNMENT PLANNER TAB ======================
    def enterprise_planner_tab():
        plan_state = {
            'entity_name': 'My Organization',
            'entity_type': 'Corporate',  # Corporate, Government, Non-Profit, Sovereign
            'fiscal_year': datetime.now().year,
            'currency': 'USD',
            'departments': [],
            'revenue_streams': [],
            'expenditures': [],
            'debt_instruments': [],
            'workforce': [],
            'capital_projects': [],
            'reserves': 0,
            'gdp_estimate': 0,
            'notes': '',
        }

        # Try load last saved plan
        saved_plans = data.list_saved_plans()

        # ---- UI Fields ----
        entity_name_f = ft.TextField(label="Entity Name", value=plan_state['entity_name'], expand=True, border_color="#00ff9d", color="#ffffff", text_size=13, dense=True)
        entity_type_dd = ft.Dropdown(options=[ft.dropdown.Option(t) for t in ["Corporate", "Government", "Non-Profit", "Sovereign", "Municipal", "State/Province", "International"]], value="Corporate", label="Type", border_color="#00ff9d", color="#ffffff", width=160, dense=True)
        fiscal_year_f = ft.TextField(label="Fiscal Year", value=str(plan_state['fiscal_year']), width=100, border_color="#00ff9d", color="#ffffff", text_size=13, dense=True)
        currency_f = ft.TextField(label="Currency", value="USD", width=80, border_color="#00ff9d", color="#ffffff", text_size=13, dense=True)
        reserves_f = ft.TextField(label="Reserves/Cash $", value="0", width=140, border_color="#ffd700", color="#ffffff", text_size=13, dense=True)
        gdp_f = ft.TextField(label="GDP / Revenue Base $", value="0", width=180, border_color="#ffd700", color="#ffffff", text_size=13, dense=True)

        # Sections
        dept_col = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO)
        rev_col = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO)
        exp_col = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO)
        debt_col = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO)
        wf_col = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO)
        capex_col = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO)
        kpi_col = ft.Column(spacing=6)
        projection_col = ft.Column(spacing=4)
        scenario_col = ft.Column(spacing=4)
        save_status = ft.Text("", size=11, color="#00ff9d")
        plan_dropdown = ft.Dropdown(options=[], label="Load Plan", border_color="#ffd700", color="#ffffff", width=220, dense=True)
        notes_f = ft.TextField(label="Strategic Notes / Objectives", multiline=True, min_lines=2, max_lines=5, expand=True, border_color="#333333", color="#ffffff", text_size=12)

        # ---- Input fields for each section ----
        dept_name = ft.TextField(label="Department/Division", expand=True, border_color="#00bfff", color="#ffffff", text_size=12, dense=True)
        dept_budget = ft.TextField(label="Budget $", width=130, border_color="#00bfff", color="#ffffff", text_size=12, dense=True, value="0")
        dept_head = ft.TextField(label="Head/Minister", width=150, border_color="#00bfff", color="#ffffff", text_size=12, dense=True)

        rev_name = ft.TextField(label="Revenue Source", expand=True, border_color="#00ff9d", color="#ffffff", text_size=12, dense=True)
        rev_amount = ft.TextField(label="Amount $", width=130, border_color="#00ff9d", color="#ffffff", text_size=12, dense=True, value="0")
        rev_freq = ft.Dropdown(options=[ft.dropdown.Option(f) for f in ["annual","quarterly","monthly","one-time"]], value="annual", label="Freq", border_color="#00ff9d", color="#ffffff", width=110, dense=True)
        rev_cat = ft.Dropdown(options=[ft.dropdown.Option(c) for c in ["Tax Revenue","Sales/Fees","Grants/Aid","Investment Income","Licensing","Tariffs/Duties","Fines/Penalties","Transfers","Other"]], value="Tax Revenue", label="Category", border_color="#00ff9d", color="#ffffff", width=140, dense=True)

        exp_name = ft.TextField(label="Expenditure", expand=True, border_color="#ff3366", color="#ffffff", text_size=12, dense=True)
        exp_amount = ft.TextField(label="Amount $", width=130, border_color="#ff3366", color="#ffffff", text_size=12, dense=True, value="0")
        exp_dept = ft.TextField(label="Department", width=130, border_color="#ff3366", color="#ffffff", text_size=12, dense=True)
        exp_cat = ft.Dropdown(options=[ft.dropdown.Option(c) for c in ["Operations","Payroll/Salaries","Benefits/Pensions","Capital Expenditure","Debt Service","Defense/Security","Education","Healthcare","Infrastructure","Social Programs","R&D/Innovation","Administration","Contingency","Other"]], value="Operations", label="Category", border_color="#ff3366", color="#ffffff", width=160, dense=True)
        exp_priority = ft.Dropdown(options=[ft.dropdown.Option(p) for p in ["Critical","High","Medium","Low","Discretionary"]], value="Medium", label="Priority", border_color="#ff3366", color="#ffffff", width=130, dense=True)

        debt_name = ft.TextField(label="Instrument", expand=True, border_color="#ff9900", color="#ffffff", text_size=12, dense=True)
        debt_principal = ft.TextField(label="Principal $", width=130, border_color="#ff9900", color="#ffffff", text_size=12, dense=True, value="0")
        debt_rate = ft.TextField(label="Rate %", width=80, border_color="#ff9900", color="#ffffff", text_size=12, dense=True, value="5.0")
        debt_maturity = ft.TextField(label="Maturity Yr", width=100, border_color="#ff9900", color="#ffffff", text_size=12, dense=True, value=str(datetime.now().year + 10))
        debt_type = ft.Dropdown(options=[ft.dropdown.Option(t) for t in ["Bond","Loan","Treasury","Municipal Bond","Credit Line","Sovereign Debt","IMF/World Bank","Other"]], value="Bond", label="Type", border_color="#ff9900", color="#ffffff", width=140, dense=True)

        wf_dept = ft.TextField(label="Department", expand=True, border_color="#9966ff", color="#ffffff", text_size=12, dense=True)
        wf_headcount = ft.TextField(label="Headcount", width=100, border_color="#9966ff", color="#ffffff", text_size=12, dense=True, value="0")
        wf_avg_salary = ft.TextField(label="Avg Salary $", width=120, border_color="#9966ff", color="#ffffff", text_size=12, dense=True, value="0")
        wf_benefits_pct = ft.TextField(label="Benefits %", width=100, border_color="#9966ff", color="#ffffff", text_size=12, dense=True, value="30")

        cap_name = ft.TextField(label="Project Name", expand=True, border_color="#00bfff", color="#ffffff", text_size=12, dense=True)
        cap_cost = ft.TextField(label="Total Cost $", width=130, border_color="#00bfff", color="#ffffff", text_size=12, dense=True, value="0")
        cap_duration = ft.TextField(label="Duration (yrs)", width=110, border_color="#00bfff", color="#ffffff", text_size=12, dense=True, value="1")
        cap_status = ft.Dropdown(options=[ft.dropdown.Option(s) for s in ["Proposed","Approved","In Progress","Completed","On Hold","Cancelled"]], value="Proposed", label="Status", border_color="#00bfff", color="#ffffff", width=130, dense=True)

        def _get_plan_dict():
            return {
                'entity_name': entity_name_f.value or 'My Organization',
                'entity_type': entity_type_dd.value or 'Corporate',
                'fiscal_year': int(fiscal_year_f.value or datetime.now().year),
                'currency': currency_f.value or 'USD',
                'reserves': float(reserves_f.value or 0),
                'gdp_estimate': float(gdp_f.value or 0),
                'departments': plan_state['departments'],
                'revenue_streams': plan_state['revenue_streams'],
                'expenditures': plan_state['expenditures'],
                'debt_instruments': plan_state['debt_instruments'],
                'workforce': plan_state['workforce'],
                'capital_projects': plan_state['capital_projects'],
                'notes': notes_f.value or '',
            }

        def _load_plan_dict(d):
            plan_state['departments'] = d.get('departments', [])
            plan_state['revenue_streams'] = d.get('revenue_streams', [])
            plan_state['expenditures'] = d.get('expenditures', [])
            plan_state['debt_instruments'] = d.get('debt_instruments', [])
            plan_state['workforce'] = d.get('workforce', [])
            plan_state['capital_projects'] = d.get('capital_projects', [])
            entity_name_f.value = d.get('entity_name', 'My Organization')
            entity_type_dd.value = d.get('entity_type', 'Corporate')
            fiscal_year_f.value = str(d.get('fiscal_year', datetime.now().year))
            currency_f.value = d.get('currency', 'USD')
            reserves_f.value = str(d.get('reserves', 0))
            gdp_f.value = str(d.get('gdp_estimate', 0))
            notes_f.value = d.get('notes', '')

        freq_mult = {'annual': 1, 'quarterly': 4, 'monthly': 12, 'one-time': 1}

        def _rebuild_all():
            cur = currency_f.value or '$'
            # Departments
            dept_col.controls.clear()
            for i, d in enumerate(plan_state['departments']):
                idx = i
                def rm_dept(e, ii=idx): plan_state['departments'].pop(ii); _rebuild_all()
                dept_col.controls.append(ft.Row([
                    ft.Text(d['name'], size=12, color="#00bfff", expand=True),
                    ft.Text(d.get('head',''), size=11, color="#aaaaaa", width=120),
                    ft.Text(f"${d['budget']:,.0f}", size=12, color="#ffd700", width=120),
                    ft.IconButton(icon=ft.Icons.DELETE, icon_color="#ff3366", icon_size=16, on_click=rm_dept),
                ], spacing=4))
            # Revenue
            rev_col.controls.clear()
            for i, r in enumerate(plan_state['revenue_streams']):
                idx = i
                def rm_rev(e, ii=idx): plan_state['revenue_streams'].pop(ii); _rebuild_all()
                ann = r['amount'] * freq_mult.get(r.get('frequency','annual'), 1)
                rev_col.controls.append(ft.Row([
                    ft.Text(r['name'], size=12, color="#00ff9d", expand=True),
                    ft.Text(r.get('category',''), size=11, color="#aaaaaa", width=110),
                    ft.Text(f"${r['amount']:,.0f}", size=12, color="#00ff9d", width=110),
                    ft.Text(r.get('frequency','annual'), size=10, color="#888888", width=70),
                    ft.Text(f"(${ann:,.0f}/yr)", size=10, color="#666666", width=100),
                    ft.IconButton(icon=ft.Icons.DELETE, icon_color="#ff3366", icon_size=16, on_click=rm_rev),
                ], spacing=4))
            # Expenditures
            exp_col.controls.clear()
            for i, x in enumerate(plan_state['expenditures']):
                idx = i
                def rm_exp(e, ii=idx): plan_state['expenditures'].pop(ii); _rebuild_all()
                pc = {"Critical":"#ff3366","High":"#ff9900","Medium":"#ffd700","Low":"#aaaaaa","Discretionary":"#666666"}
                exp_col.controls.append(ft.Row([
                    ft.Text(x['name'], size=12, color="#ff3366", expand=True),
                    ft.Text(x.get('category',''), size=10, color="#aaaaaa", width=110),
                    ft.Text(x.get('department',''), size=10, color="#888888", width=90),
                    ft.Text(f"${x['amount']:,.0f}", size=12, color="#ff3366", width=110),
                    ft.Text(x.get('priority','Medium'), size=10, color=pc.get(x.get('priority','Medium'),'#aaaaaa'), width=80),
                    ft.IconButton(icon=ft.Icons.DELETE, icon_color="#ff3366", icon_size=16, on_click=rm_exp),
                ], spacing=4))
            # Debt
            debt_col.controls.clear()
            for i, d in enumerate(plan_state['debt_instruments']):
                idx = i
                def rm_debt(e, ii=idx): plan_state['debt_instruments'].pop(ii); _rebuild_all()
                ann_svc = d['principal'] * d['rate'] / 100
                debt_col.controls.append(ft.Row([
                    ft.Text(d['name'], size=12, color="#ff9900", expand=True),
                    ft.Text(d.get('dtype','Bond'), size=10, color="#aaaaaa", width=90),
                    ft.Text(f"${d['principal']:,.0f}", size=12, color="#ff9900", width=120),
                    ft.Text(f"{d['rate']:.1f}%", size=11, color="#ffd700", width=55),
                    ft.Text(f"Mat:{d.get('maturity','')}", size=10, color="#888888", width=80),
                    ft.Text(f"Svc:${ann_svc:,.0f}/yr", size=10, color="#ff3366", width=110),
                    ft.IconButton(icon=ft.Icons.DELETE, icon_color="#ff3366", icon_size=16, on_click=rm_debt),
                ], spacing=4))
            # Workforce
            wf_col.controls.clear()
            for i, w in enumerate(plan_state['workforce']):
                idx = i
                def rm_wf(e, ii=idx): plan_state['workforce'].pop(ii); _rebuild_all()
                total_cost = w['headcount'] * w['avg_salary'] * (1 + w['benefits_pct']/100)
                wf_col.controls.append(ft.Row([
                    ft.Text(w['department'], size=12, color="#9966ff", expand=True),
                    ft.Text(f"{w['headcount']:,} staff", size=12, color="#ffffff", width=90),
                    ft.Text(f"Avg ${w['avg_salary']:,.0f}", size=11, color="#aaaaaa", width=110),
                    ft.Text(f"Ben:{w['benefits_pct']:.0f}%", size=10, color="#888888", width=70),
                    ft.Text(f"Total:${total_cost:,.0f}", size=11, color="#ffd700", width=130),
                    ft.IconButton(icon=ft.Icons.DELETE, icon_color="#ff3366", icon_size=16, on_click=rm_wf),
                ], spacing=4))
            # Capital Projects
            capex_col.controls.clear()
            for i, c in enumerate(plan_state['capital_projects']):
                idx = i
                def rm_cap(e, ii=idx): plan_state['capital_projects'].pop(ii); _rebuild_all()
                sc = {"Proposed":"#aaaaaa","Approved":"#ffd700","In Progress":"#00bfff","Completed":"#00ff9d","On Hold":"#ff9900","Cancelled":"#ff3366"}
                ann_cost = c['cost'] / max(c.get('duration',1),1)
                capex_col.controls.append(ft.Row([
                    ft.Text(c['name'], size=12, color="#00bfff", expand=True),
                    ft.Text(f"${c['cost']:,.0f}", size=12, color="#ffd700", width=120),
                    ft.Text(f"{c.get('duration',1)}yr", size=10, color="#aaaaaa", width=50),
                    ft.Text(f"${ann_cost:,.0f}/yr", size=10, color="#888888", width=100),
                    ft.Text(c.get('status','Proposed'), size=10, color=sc.get(c.get('status',''),'#aaaaaa'), width=90),
                    ft.IconButton(icon=ft.Icons.DELETE, icon_color="#ff3366", icon_size=16, on_click=rm_cap),
                ], spacing=4))
            # KPIs
            _compute_kpis()
            page.update()

        def _compute_kpis():
            kpi_col.controls.clear()
            projection_col.controls.clear()
            scenario_col.controls.clear()
            total_rev = sum(r['amount'] * freq_mult.get(r.get('frequency','annual'),1) for r in plan_state['revenue_streams'])
            total_exp = sum(x['amount'] for x in plan_state['expenditures'])
            total_debt = sum(d['principal'] for d in plan_state['debt_instruments'])
            total_debt_svc = sum(d['principal'] * d['rate'] / 100 for d in plan_state['debt_instruments'])
            total_wf_cost = sum(w['headcount'] * w['avg_salary'] * (1 + w['benefits_pct']/100) for w in plan_state['workforce'])
            total_capex = sum(c['cost'] / max(c.get('duration',1),1) for c in plan_state['capital_projects'])
            total_headcount = sum(w['headcount'] for w in plan_state['workforce'])
            reserves = float(reserves_f.value or 0)
            gdp = float(gdp_f.value or 0)
            surplus = total_rev - total_exp - total_debt_svc - total_capex
            op_margin = (surplus / total_rev * 100) if total_rev > 0 else 0
            debt_to_rev = (total_debt / total_rev * 100) if total_rev > 0 else 0
            debt_to_gdp = (total_debt / gdp * 100) if gdp > 0 else 0
            reserves_months = (reserves / (total_exp/12)) if total_exp > 0 else float('inf')
            exp_per_capita = 0  # placeholder

            def _k(label, value, color="#ffd700"):
                return ft.Container(content=ft.Column([ft.Text(label, size=9, color="#aaaaaa"), ft.Text(value, size=14, weight=ft.FontWeight.BOLD, color=color)], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER), bgcolor="#1a1a2e", padding=8, border_radius=8, expand=True)

            sc = "#00ff9d" if surplus >= 0 else "#ff3366"
            kpi_col.controls.append(ft.Text("KEY PERFORMANCE INDICATORS", size=16, weight=ft.FontWeight.BOLD, color="#ffd700"))
            kpi_col.controls.append(ft.Row([
                _k("Total Revenue", f"${total_rev:,.0f}", "#00ff9d"),
                _k("Total Expenditure", f"${total_exp:,.0f}", "#ff3366"),
                _k("Debt Service", f"${total_debt_svc:,.0f}", "#ff9900"),
                _k("Capital Spending", f"${total_capex:,.0f}", "#00bfff"),
            ], spacing=4))
            kpi_col.controls.append(ft.Row([
                _k("Net Surplus/Deficit", f"${surplus:,.0f}", sc),
                _k("Operating Margin", f"{op_margin:.1f}%", "#00ff9d" if op_margin > 0 else "#ff3366"),
                _k("Total Debt", f"${total_debt:,.0f}", "#ff9900"),
                _k("Debt-to-Revenue", f"{debt_to_rev:.1f}%", "#ff9900" if debt_to_rev < 100 else "#ff3366"),
            ], spacing=4))
            kpi_col.controls.append(ft.Row([
                _k("Workforce Cost", f"${total_wf_cost:,.0f}", "#9966ff"),
                _k("Total Headcount", f"{total_headcount:,}", "#9966ff"),
                _k("Reserves", f"${reserves:,.0f}", "#ffd700"),
                _k("Reserve Months", f"{reserves_months:.1f}" if reserves_months != float('inf') else "N/A", "#ffd700"),
            ], spacing=4))
            if gdp > 0:
                kpi_col.controls.append(ft.Row([
                    _k("Debt-to-GDP", f"{debt_to_gdp:.1f}%", "#ff9900" if debt_to_gdp < 60 else "#ff3366"),
                    _k("Revenue/GDP", f"{(total_rev/gdp*100):.1f}%", "#00ff9d"),
                    _k("Spending/GDP", f"{((total_exp+total_capex)/gdp*100):.1f}%", "#ff3366"),
                    _k("Surplus/GDP", f"{(surplus/gdp*100):.1f}%", sc),
                ], spacing=4))
            # Department budget allocation
            dept_budgets = {d['name']: d['budget'] for d in plan_state['departments']}
            if dept_budgets:
                kpi_col.controls.append(ft.Text("DEPARTMENT BUDGET ALLOCATION", size=13, weight=ft.FontWeight.BOLD, color="#00bfff"))
                total_dept = sum(dept_budgets.values())
                for dname, dbud in sorted(dept_budgets.items(), key=lambda x: -x[1]):
                    pct = (dbud / total_dept * 100) if total_dept > 0 else 0
                    bar_w = max(pct * 3, 5)
                    kpi_col.controls.append(ft.Row([
                        ft.Text(dname, size=11, color="#ffffff", width=160),
                        ft.Container(width=bar_w, height=12, bgcolor="#00bfff", border_radius=4),
                        ft.Text(f"${dbud:,.0f} ({pct:.1f}%)", size=11, color="#aaaaaa"),
                    ], spacing=6))
            # Expenditure by category
            cat_totals = {}
            for x in plan_state['expenditures']:
                cat = x.get('category', 'Other')
                cat_totals[cat] = cat_totals.get(cat, 0) + x['amount']
            if cat_totals:
                kpi_col.controls.append(ft.Text("SPENDING BY CATEGORY", size=13, weight=ft.FontWeight.BOLD, color="#ff3366"))
                for cname, camt in sorted(cat_totals.items(), key=lambda x: -x[1]):
                    pct = (camt / total_exp * 100) if total_exp > 0 else 0
                    bar_w = max(pct * 3, 5)
                    kpi_col.controls.append(ft.Row([
                        ft.Text(cname, size=11, color="#ffffff", width=160),
                        ft.Container(width=bar_w, height=12, bgcolor="#ff3366", border_radius=4),
                        ft.Text(f"${camt:,.0f} ({pct:.1f}%)", size=11, color="#aaaaaa"),
                    ], spacing=6))
            # Revenue by category
            rev_totals = {}
            for r in plan_state['revenue_streams']:
                cat = r.get('category', 'Other')
                rev_totals[cat] = rev_totals.get(cat, 0) + r['amount'] * freq_mult.get(r.get('frequency','annual'),1)
            if rev_totals:
                kpi_col.controls.append(ft.Text("REVENUE BY SOURCE", size=13, weight=ft.FontWeight.BOLD, color="#00ff9d"))
                for rname, ramt in sorted(rev_totals.items(), key=lambda x: -x[1]):
                    pct = (ramt / total_rev * 100) if total_rev > 0 else 0
                    bar_w = max(pct * 3, 5)
                    kpi_col.controls.append(ft.Row([
                        ft.Text(rname, size=11, color="#ffffff", width=160),
                        ft.Container(width=bar_w, height=12, bgcolor="#00ff9d", border_radius=4),
                        ft.Text(f"${ramt:,.0f} ({pct:.1f}%)", size=11, color="#aaaaaa"),
                    ], spacing=6))

            # MULTI-YEAR PROJECTIONS (5 year)
            projection_col.controls.append(ft.Text("5-YEAR FISCAL PROJECTION", size=14, weight=ft.FontWeight.BOLD, color="#ffd700"))
            growth_rates = [0.03, 0.03, 0.025, 0.025, 0.02]  # revenue growth
            infl_rates = [0.03, 0.03, 0.028, 0.025, 0.025]
            proj_rev = total_rev
            proj_exp = total_exp + total_debt_svc + total_capex
            proj_debt = total_debt
            projection_col.controls.append(ft.Row([
                ft.Text("Year", size=11, color="#aaaaaa", width=60), ft.Text("Revenue", size=11, color="#aaaaaa", width=120),
                ft.Text("Spending", size=11, color="#aaaaaa", width=120), ft.Text("Surplus/Def", size=11, color="#aaaaaa", width=120),
                ft.Text("Debt", size=11, color="#aaaaaa", width=120), ft.Text("Debt/Rev%", size=11, color="#aaaaaa", width=80),
            ], spacing=4))
            fy = int(fiscal_year_f.value or datetime.now().year)
            for yi in range(5):
                gr = growth_rates[yi] if yi < len(growth_rates) else 0.02
                ir = infl_rates[yi] if yi < len(infl_rates) else 0.025
                proj_rev *= (1 + gr)
                proj_exp *= (1 + ir)
                proj_surplus = proj_rev - proj_exp
                proj_debt = max(0, proj_debt - proj_surplus * 0.1) if proj_surplus > 0 else proj_debt + abs(proj_surplus)
                d2r = (proj_debt / proj_rev * 100) if proj_rev > 0 else 0
                sc2 = "#00ff9d" if proj_surplus >= 0 else "#ff3366"
                projection_col.controls.append(ft.Row([
                    ft.Text(str(fy + yi + 1), size=12, color="#ffffff", width=60),
                    ft.Text(f"${proj_rev:,.0f}", size=12, color="#00ff9d", width=120),
                    ft.Text(f"${proj_exp:,.0f}", size=12, color="#ff3366", width=120),
                    ft.Text(f"${proj_surplus:,.0f}", size=12, color=sc2, width=120),
                    ft.Text(f"${proj_debt:,.0f}", size=12, color="#ff9900", width=120),
                    ft.Text(f"{d2r:.1f}%", size=12, color="#ff9900" if d2r < 100 else "#ff3366", width=80),
                ], spacing=4))

            # SCENARIO ANALYSIS
            scenario_col.controls.append(ft.Text("SCENARIO ANALYSIS", size=14, weight=ft.FontWeight.BOLD, color="#ffd700"))
            scenarios = [
                ("Optimistic", 1.10, 0.95, "#00ff9d"),
                ("Base Case", 1.0, 1.0, "#ffd700"),
                ("Pessimistic", 0.85, 1.10, "#ff9900"),
                ("Crisis", 0.70, 1.25, "#ff3366"),
            ]
            scenario_col.controls.append(ft.Row([
                ft.Text("Scenario", size=11, color="#aaaaaa", width=100), ft.Text("Revenue", size=11, color="#aaaaaa", width=120),
                ft.Text("Spending", size=11, color="#aaaaaa", width=120), ft.Text("Net", size=11, color="#aaaaaa", width=120),
                ft.Text("Margin", size=11, color="#aaaaaa", width=80),
            ], spacing=4))
            for sname, rev_mult, exp_mult, scolor in scenarios:
                s_rev = total_rev * rev_mult
                s_exp = (total_exp + total_debt_svc + total_capex) * exp_mult
                s_net = s_rev - s_exp
                s_margin = (s_net / s_rev * 100) if s_rev > 0 else 0
                scenario_col.controls.append(ft.Row([
                    ft.Text(sname, size=12, color=scolor, width=100, weight=ft.FontWeight.BOLD),
                    ft.Text(f"${s_rev:,.0f}", size=12, color="#00ff9d", width=120),
                    ft.Text(f"${s_exp:,.0f}", size=12, color="#ff3366", width=120),
                    ft.Text(f"${s_net:,.0f}", size=12, color="#00ff9d" if s_net >= 0 else "#ff3366", width=120),
                    ft.Text(f"{s_margin:.1f}%", size=12, color=scolor, width=80),
                ], spacing=4))

        # ---- Add handlers ----
        def add_dept(e):
            plan_state['departments'].append({'name': dept_name.value or 'New Dept', 'budget': float(dept_budget.value or 0), 'head': dept_head.value or ''})
            dept_name.value = ""; dept_budget.value = "0"; dept_head.value = ""
            _rebuild_all()
        def add_rev(e):
            plan_state['revenue_streams'].append({'name': rev_name.value or 'Revenue', 'amount': float(rev_amount.value or 0), 'frequency': rev_freq.value or 'annual', 'category': rev_cat.value or 'Other'})
            rev_name.value = ""; rev_amount.value = "0"
            _rebuild_all()
        def add_exp(e):
            plan_state['expenditures'].append({'name': exp_name.value or 'Expense', 'amount': float(exp_amount.value or 0), 'category': exp_cat.value or 'Other', 'department': exp_dept.value or '', 'priority': exp_priority.value or 'Medium'})
            exp_name.value = ""; exp_amount.value = "0"; exp_dept.value = ""
            _rebuild_all()
        def add_debt(e):
            plan_state['debt_instruments'].append({'name': debt_name.value or 'Debt', 'principal': float(debt_principal.value or 0), 'rate': float(debt_rate.value or 5), 'maturity': debt_maturity.value or '', 'dtype': debt_type.value or 'Bond'})
            debt_name.value = ""; debt_principal.value = "0"
            _rebuild_all()
        def add_wf(e):
            plan_state['workforce'].append({'department': wf_dept.value or 'Dept', 'headcount': int(wf_headcount.value or 0), 'avg_salary': float(wf_avg_salary.value or 0), 'benefits_pct': float(wf_benefits_pct.value or 30)})
            wf_dept.value = ""; wf_headcount.value = "0"; wf_avg_salary.value = "0"
            _rebuild_all()
        def add_cap(e):
            plan_state['capital_projects'].append({'name': cap_name.value or 'Project', 'cost': float(cap_cost.value or 0), 'duration': int(cap_duration.value or 1), 'status': cap_status.value or 'Proposed'})
            cap_name.value = ""; cap_cost.value = "0"
            _rebuild_all()

        # ---- Save/Load handlers ----
        def save_plan(e):
            pname = entity_name_f.value or 'default'
            try:
                fp = data.save_plan(_get_plan_dict(), pname)
                save_status.value = f"Saved: {os.path.basename(fp)}"
                save_status.color = "#00ff9d"
                _refresh_plan_dropdown()
            except Exception as ex:
                save_status.value = f"Save error: {ex}"
                save_status.color = "#ff3366"
            page.update()

        def load_plan(e):
            if not plan_dropdown.value:
                return
            try:
                d = data.load_plan(plan_dropdown.value)
                if d:
                    _load_plan_dict(d)
                    save_status.value = f"Loaded: {plan_dropdown.value}"
                    save_status.color = "#00ff9d"
                    _rebuild_all()
                else:
                    save_status.value = "Plan not found"
                    save_status.color = "#ff3366"
                    page.update()
            except Exception as ex:
                save_status.value = f"Load error: {ex}"
                save_status.color = "#ff3366"
                page.update()

        def delete_plan(e):
            if not plan_dropdown.value:
                return
            data.delete_plan(plan_dropdown.value)
            save_status.value = f"Deleted: {plan_dropdown.value}"
            save_status.color = "#ff9900"
            _refresh_plan_dropdown()
            page.update()

        def _refresh_plan_dropdown():
            plans = data.list_saved_plans()
            plan_dropdown.options = [ft.dropdown.Option(p['name']) for p in plans]

        _refresh_plan_dropdown()
        _rebuild_all()

        def _section(title, color, input_row, list_col):
            return ft.Container(
                content=ft.Column([
                    ft.Text(title, size=15, weight=ft.FontWeight.BOLD, color=color),
                    input_row,
                    list_col,
                ], spacing=4),
                bgcolor="#0f0f0f", padding=10, border_radius=10,
            )

        return _card(ft.Column([
            ft.Text("ENTERPRISE / GOVERNMENT PLANNER", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            ft.Text("Comprehensive fiscal planning for corporations, governments, and sovereign entities", size=12, color="#aaaaaa"),
            ft.Divider(height=1, color="#333333"),
            # Save/Load bar
            ft.Row([
                plan_dropdown,
                ft.Button("LOAD", bgcolor="#1a1a2e", color="#ffd700", style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)), on_click=load_plan),
                ft.Button("SAVE", bgcolor="#00ff9d", color="#0a0a0a", style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)), on_click=save_plan),
                ft.Button("DELETE", bgcolor="#ff3366", color="#ffffff", style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)), on_click=delete_plan),
                save_status,
            ], spacing=8),
            ft.Divider(height=1, color="#333333"),
            # Entity header
            ft.Row([entity_name_f, entity_type_dd, fiscal_year_f, currency_f, reserves_f, gdp_f], spacing=6),
            notes_f,
            ft.Divider(height=1, color="#333333"),
            # Departments
            _section("DEPARTMENTS / DIVISIONS", "#00bfff",
                ft.Row([dept_name, dept_budget, dept_head, ft.IconButton(icon=ft.Icons.ADD_CIRCLE, icon_color="#00bfff", icon_size=20, on_click=add_dept)], spacing=4),
                dept_col),
            # Revenue
            _section("REVENUE STREAMS", "#00ff9d",
                ft.Row([rev_name, rev_amount, rev_freq, rev_cat, ft.IconButton(icon=ft.Icons.ADD_CIRCLE, icon_color="#00ff9d", icon_size=20, on_click=add_rev)], spacing=4),
                rev_col),
            # Expenditures
            _section("EXPENDITURES", "#ff3366",
                ft.Row([exp_name, exp_amount, exp_dept, exp_cat, exp_priority, ft.IconButton(icon=ft.Icons.ADD_CIRCLE, icon_color="#ff3366", icon_size=20, on_click=add_exp)], spacing=4),
                exp_col),
            # Debt
            _section("DEBT INSTRUMENTS", "#ff9900",
                ft.Row([debt_name, debt_principal, debt_rate, debt_maturity, debt_type, ft.IconButton(icon=ft.Icons.ADD_CIRCLE, icon_color="#ff9900", icon_size=20, on_click=add_debt)], spacing=4),
                debt_col),
            # Workforce
            _section("WORKFORCE PLANNING", "#9966ff",
                ft.Row([wf_dept, wf_headcount, wf_avg_salary, wf_benefits_pct, ft.IconButton(icon=ft.Icons.ADD_CIRCLE, icon_color="#9966ff", icon_size=20, on_click=add_wf)], spacing=4),
                wf_col),
            # Capital Projects
            _section("CAPITAL PROJECTS & INFRASTRUCTURE", "#00bfff",
                ft.Row([cap_name, cap_cost, cap_duration, cap_status, ft.IconButton(icon=ft.Icons.ADD_CIRCLE, icon_color="#00bfff", icon_size=20, on_click=add_cap)], spacing=4),
                capex_col),
            ft.Divider(height=2, color="#333333"),
            # KPIs
            kpi_col,
            ft.Divider(height=1, color="#333333"),
            # Projections
            projection_col,
            ft.Divider(height=1, color="#333333"),
            # Scenarios
            scenario_col,
        ], scroll=ft.ScrollMode.AUTO, spacing=8))


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

        total_val = s.get('total_value', 0)
        total_gl = s.get('total_gain_loss', 0)
        total_cb = s.get('total_cost_basis', 0)
        ann_ret = s.get('ann_return', 0)
        ann_vol = s.get('ann_volatility', 0)
        sharpe = s.get('sharpe', 0)
        sortino = s.get('sortino', 0)
        max_dd = s.get('max_drawdown', 0)
        win_rate = s.get('win_rate', 0)
        n_hold = s.get('num_holdings', 0)
        total_pct = s.get('total_gain_pct', 0)

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

        # =================== 15. GROWTH TRAJECTORY & PROJECTION ===================
        result_col.controls.append(_section_title("GROWTH TRAJECTORY & PROJECTION", "#00ff9d"))
        if ann_ret > 0 and total_val > 0:
            proj_years = [1, 2, 3, 5, 10, 15, 20, 25, 30]
            result_col.controls.append(ft.Row([
                ft.Text("Year", size=11, color="#aaaaaa", width=50),
                ft.Text("Optimistic", size=11, color="#aaaaaa", width=110),
                ft.Text("Base Case", size=11, color="#aaaaaa", width=110),
                ft.Text("Conservative", size=11, color="#aaaaaa", width=110),
                ft.Text("Pessimistic", size=11, color="#aaaaaa", width=110),
            ], spacing=4))
            for y in proj_years:
                opt = total_val * (1 + ann_ret * 1.3) ** y
                base = total_val * (1 + ann_ret) ** y
                cons = total_val * (1 + ann_ret * 0.6) ** y
                pess = total_val * (1 + max(ann_ret * 0.2, -0.03)) ** y
                result_col.controls.append(ft.Row([
                    ft.Text(f"+{y}", size=12, color="#ffffff", width=50),
                    ft.Text(f"${opt:,.0f}", size=12, color="#00ff9d", width=110),
                    ft.Text(f"${base:,.0f}", size=12, color="#ffd700", width=110),
                    ft.Text(f"${cons:,.0f}", size=12, color="#ff9900", width=110),
                    ft.Text(f"${pess:,.0f}", size=12, color="#ff3366", width=110),
                ], spacing=4))
        else:
            result_col.controls.append(ft.Text("Import data to see growth projections.", size=13, color="#888888"))
        result_col.controls.append(ft.Divider(height=1, color="#333333"))

        # =================== 16. PORTFOLIO EVOLUTION TIMELINE ===================
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

    # ====================== ALL TAB BUILDERS (for Home view) ======================
    ALL_TAB_BUILDERS = [
        dashboard_tab, data_organizer_tab, portfolio_tab, analysis_tab,
        market_graphs_tab, goals_tab, smart_brain_tab, ai_chat_tab,
        risk_radar_tab, portfolio_dna_tab, emotional_gauge_tab, time_machine_tab,
        planner_tab, monthly_comparison_tab, daily_diary_tab, dream_architect_tab,
        stress_lab_tab, diversification_tab, market_regime_tab, voice_tab,
        monte_carlo_tab, constellation_tab, settings_tab,
        drawdown_tab, holdings_bar_tab, returns_dist_tab, sector_tab,
        gain_loss_tab, rolling_sharpe_tab, rolling_vol_tab, monthly_heatmap_tab,
        weight_pie_tab, growth_projection_tab, efficient_frontier_tab,
        summary_stats_tab, top_movers_tab, fire_tab, attribution_tab, rebalancer_tab,
        benchmark_tab, trade_log_tab, income_tracker_tab,
        whatif_tab, tax_optimizer_tab, correlation_dive_tab,
        health_score_tab, risk_returns_tab, streaks_tab,
        profit_charts_tab, asset_inspection_tab,
        live_chart_tab, advanced_indicators_tab,
        book_balancer_tab, budget_planner_tab,
        asset_focus_tab,
        enterprise_planner_tab,
        progression_tab,
    ]

    # ====================== CATEGORY DEFINITIONS ======================
    NAV_CATEGORIES = [
        {"icon": ft.Icons.HOME, "label": "Home",
         "builders": None},
        {"icon": ft.Icons.DASHBOARD, "label": "Dashboard",
         "builders": lambda: [dashboard_tab, summary_stats_tab, health_score_tab, profit_charts_tab]},
        {"icon": ft.Icons.SHOW_CHART, "label": "Progress",
         "builders": lambda: [progression_tab]},
        {"icon": ft.Icons.FOLDER_OPEN, "label": "Portfolio",
         "builders": lambda: [data_organizer_tab, portfolio_tab, holdings_bar_tab, weight_pie_tab, top_movers_tab, trade_log_tab, asset_inspection_tab, asset_focus_tab]},
        {"icon": ft.Icons.ANALYTICS, "label": "Analysis",
         "builders": lambda: [analysis_tab, market_graphs_tab, portfolio_dna_tab, constellation_tab, efficient_frontier_tab, sector_tab, attribution_tab, returns_dist_tab, monthly_heatmap_tab, benchmark_tab, correlation_dive_tab, risk_returns_tab, live_chart_tab, advanced_indicators_tab, asset_focus_tab]},
        {"icon": ft.Icons.CALENDAR_MONTH, "label": "Planning",
         "builders": lambda: [enterprise_planner_tab, goals_tab, time_machine_tab, planner_tab, fire_tab, growth_projection_tab, monthly_comparison_tab, dream_architect_tab, income_tracker_tab, whatif_tab, book_balancer_tab, budget_planner_tab]},
        {"icon": ft.Icons.SHIELD, "label": "Risk",
         "builders": lambda: [risk_radar_tab, stress_lab_tab, emotional_gauge_tab, diversification_tab, market_regime_tab, drawdown_tab, rolling_sharpe_tab, rolling_vol_tab, gain_loss_tab, streaks_tab]},
        {"icon": ft.Icons.SMART_TOY, "label": "AI & Tools",
         "builders": lambda: [smart_brain_tab, ai_chat_tab, voice_tab, monte_carlo_tab, daily_diary_tab, rebalancer_tab, tax_optimizer_tab]},
        {"icon": ft.Icons.SETTINGS, "label": "Settings",
         "builders": lambda: [settings_tab]},
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

    _expanded_cards = {}  # track expanded state per builder name

    def _make_expandable_card(builder, col_sizes):
        """Wrap a card builder with expand/collapse toggle for Home view."""
        bname = getattr(builder, '__name__', str(builder))
        is_expanded = _expanded_cards.get(bname, False)
        card_container = ft.Container(expand=True)
        cols_normal = col_sizes
        cols_expanded = {"xs": 12, "sm": 12, "md": 12, "lg": 12, "xl": 12}

        wrapper = ft.Container(col=cols_expanded if is_expanded else cols_normal)

        def toggle_expand(e):
            _expanded_cards[bname] = not _expanded_cards.get(bname, False)
            load_category(nav_rail.selected_index)

        expand_icon = ft.Icons.FULLSCREEN_EXIT if is_expanded else ft.Icons.FULLSCREEN
        expand_btn = ft.IconButton(icon=expand_icon, icon_color="#00ff9d", icon_size=18,
                                    tooltip="Expand / Collapse", on_click=toggle_expand)
        built = _safe_build(builder)
        header_row = ft.Row([ft.Container(expand=True), expand_btn], spacing=0)
        card_container.content = ft.Column([header_row, built], spacing=0)
        wrapper.content = card_container
        return wrapper

    def load_category(index):
        content_area.controls.clear()
        cat = NAV_CATEGORIES[index]
        # Show loading placeholder immediately
        content_area.controls.append(_loading_placeholder(f"Loading {cat['label']}..."))
        page.update()
        content_area.controls.clear()
        try:
            if cat["label"] == "Home":
                content_area.controls.append(_import_bar())
                grid = ft.ResponsiveRow(spacing=12, run_spacing=12)
                col_sizes = {"xs": 12, "sm": 12, "md": 6, "lg": 4, "xl": 4}
                for b in ALL_TAB_BUILDERS:
                    grid.controls.append(_make_expandable_card(b, col_sizes))
                content_area.controls.append(grid)
            elif cat["label"] == "About":
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

    page.add(
        snack,
        header,
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
    )
    page.update()

    threading.Thread(target=update_live_ticker, daemon=True).start()


if __name__ == "__main__":
    ft.run(main, view=ft.AppView.WEB_BROWSER, port=8550)