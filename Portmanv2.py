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
        self.total_value = 0.0
        self.risk_profile = "Aggressive"
        self.raw_files = {}
        self.import_summary = []

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

    # data_organizer_tab removed

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

    # goals_tab, smart_brain_tab, ai_chat_tab removed

    def risk_radar_tab():
        return _card(ft.Column([
            ft.Text("HOLOGRAPHIC RISK RADAR", size=22, weight=ft.FontWeight.BOLD, color="#00ff9d"),
            _chart_image(data.chart_risk_radar()),
        ], scroll=ft.ScrollMode.AUTO, spacing=15))

    # portfolio_dna_tab, emotional_gauge_tab, time_machine_tab, daily_diary_tab, dream_architect_tab, stress_lab_tab removed

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

    # voice_tab, monte_carlo_tab, constellation_tab, settings_tab removed

    # planner_tab removed

    # monthly_comparison_tab removed

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

    # gain_loss_tab removed

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

    # weight_pie_tab, growth_projection_tab, efficient_frontier_tab removed

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

    # income_tracker_tab removed

    # whatif_tab removed

    # tax_optimizer_tab removed

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