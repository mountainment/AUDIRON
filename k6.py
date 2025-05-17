import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QAction, QLabel, QTextEdit, QDesktopWidget, QMessageBox,
                            QToolBar, QMenu, QInputDialog)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import statsmodels.api as sm
from scipy import signal
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import pywt
from copulae import GaussianCopula

# Original DataHandler class must be defined first
class DataHandler:
    def __init__(self):
        self.start_date = '2024-01-01'
        self.end_date = '2025-03-31'
        self.dates = pd.date_range(self.start_date, self.end_date, freq='D')
        self.generate_synthetic_data()
        
    def generate_synthetic_data(self):
        np.random.seed(42)
        n = len(self.dates)
        
        # USD Index
        usd_returns = np.random.randn(n) * 0.005
        self.usd = pd.Series(np.cumprod(1 + usd_returns), index=self.dates) * 100
        
        # AUD/USD
        aud_beta = 0.6
        audusd_returns = aud_beta * usd_returns + np.random.randn(n) * 0.005
        self.audusd = pd.Series(np.cumprod(1 + audusd_returns), index=self.dates) * 0.75
        
        # Chinese Market
        china_vol = np.sqrt(1/252) * 0.15
        self.china_market = pd.Series(np.cumprod(1 + np.random.randn(n) * china_vol),
                                    index=self.dates) * 1000
        
        # Companies Market Cap
        company_beta = 0.4
        company_returns = (company_beta * self.china_market.pct_change().fillna(0) +
                          np.random.randn(n) * 0.007)
        self.companies_mcap = pd.Series(np.cumprod(1 + company_returns),
                                      index=self.dates) * 1e9

    def decorrelate_series(self, series, market):
        X = sm.add_constant(market)
        model = sm.OLS(series, X).fit()
        return series - (model.params.iloc[0] + model.params.iloc[1] * market)

    def process_data(self, freq='D'):
        audusd_decorr = self.decorrelate_series(
            self.audusd.pct_change().dropna(),
            self.usd.pct_change().dropna()
        )
        
        companies_decorr = self.decorrelate_series(
            self.companies_mcap.pct_change().dropna(),
            self.china_market.pct_change().dropna()
        )
        
        audusd_resampled = audusd_decorr.resample(freq).last().ffill()
        companies_resampled = companies_decorr.resample(freq).last().ffill()
        
        return audusd_resampled, companies_resampled

# Original CorrelationAnalyzer must be defined before Enhanced version
class CorrelationAnalyzer:
    @staticmethod
    def calculate_shifted_correlations(series1, series2, max_shift=20):
        aligned = pd.concat([series1, series2], axis=1).dropna()
        x = aligned.iloc[:,0]
        y = aligned.iloc[:,1]
        
        correlations = []
        shifts = range(-max_shift, max_shift+1)
        for shift in shifts:
            shifted_y = y.shift(shift)
            valid = pd.concat([x, shifted_y], axis=1).dropna()
            if len(valid) < 2:
                correlations.append(np.nan)
                continue
            corr = valid.corr().iloc[0,1]
            correlations.append(corr)
        return shifts, correlations

    @staticmethod
    def cross_correlate(series1, series2, max_lag=30):
        s1 = (series1 - np.mean(series1)) / np.std(series1)
        s2 = (series2 - np.mean(series2)) / np.std(series2)
        corr = signal.correlate(s1, s2, mode='full') / len(s1)
        lags = signal.correlation_lags(len(s1), len(s2), mode='full')
        valid = (lags >= -max_lag) & (lags <= max_lag)
        return lags[valid], corr[valid]

    @staticmethod
    def mutual_information(series1, series2):
        X = series1.values.reshape(-1, 1)
        y = series2.values
        return mutual_info_regression(X, y, n_neighbors=5)[0]

    @staticmethod
    def wavelet_coherence(series1, series2):
        scales = np.arange(1, 64)
        coef1, _ = pywt.cwt(series1, scales, 'morl')
        coef2, _ = pywt.cwt(series2, scales, 'morl')
        return np.abs(coef1 * np.conj(coef2)) / (np.abs(coef1) * np.abs(coef2))

    @staticmethod
    def granger_causality(series1, series2, maxlag=5):
        data = pd.DataFrame({'series1': series1, 'series2': series2}).dropna()
        return grangercausalitytests(data[['series2', 'series1']], maxlag=maxlag)

    @staticmethod
    def copula_analysis(series1, series2):
        cop = GaussianCopula(dim=2)
        cop.fit(np.column_stack([series1, series2]))
        return cop.tau

# Now we can define the enhanced versions
class EnhancedDataHandler(DataHandler):
    def diagnostic_check(self, freq='D'):
        """Run comprehensive data quality checks"""
        audusd, companies = self.process_data(freq)
        
        # Basic stationarity checks
        adf_aud = adfuller(audusd)[1]
        adf_comp = adfuller(companies)[1]
        
        # Decorrelation effectiveness
        raw_aud = self.audusd.pct_change().dropna().resample(freq).last()
        raw_comp = self.companies_mcap.pct_change().dropna().resample(freq).last()
        usd = self.usd.pct_change().dropna().resample(freq).last()
        china = self.china_market.pct_change().dropna().resample(freq).last()
        
        residual_aud_corr = np.corrcoef(audusd, usd)[0,1]
        residual_comp_corr = np.corrcoef(companies, china)[0,1]
        
        return {
            'stationarity': {'AUD': adf_aud, 'Companies': adf_comp},
            'decorr_effectiveness': {'AUD-USD': residual_aud_corr, 'Comp-China': residual_comp_corr},
            'basic_correlation': np.corrcoef(audusd, companies)[0,1]
        }

class EnhancedCorrelationAnalyzer(CorrelationAnalyzer):
    @staticmethod
    def regime_analysis(series1, series2, freq='D'):
        """Analyze correlations across market regimes"""
        regimes = {
            'Pre-2018': (pd.Timestamp('2015-01-01'), pd.Timestamp('2017-12-31')),
            '2018-2020': (pd.Timestamp('2018-01-01'), pd.Timestamp('2020-03-31')),
            'Post-COVID': (pd.Timestamp('2020-04-01'), pd.Timestamp('2023-12-31'))
        }
        
        results = {}
        aligned = pd.concat([series1, series2], axis=1).dropna()
        
        for name, (start, end) in regimes.items():
            subset = aligned.loc[start:end]
            if len(subset) > 10:  # Minimum data points
                results[name] = {
                    'correlation': subset.iloc[:,0].corr(subset.iloc[:,1]),
                    'mutual_info': mutual_info_regression(
                        subset.iloc[:,0].values.reshape(-1,1),
                        subset.iloc[:,1].values)[0]
                }
        return results

    @staticmethod
    def volatility_adjusted_correlation(series1, series2, window=20):
        """Calculate correlation normalized by volatility regimes"""
        s1_vol = series1.rolling(window).std()
        s2_vol = series2.rolling(window).std()
        
        norm1 = (series1/s1_vol).dropna()
        norm2 = (series2/s2_vol).dropna()
        
        aligned = pd.concat([norm1, norm2], axis=1).dropna()
        return aligned.iloc[:,0].corr(aligned.iloc[:,1])

# MainWindow classes must come after all their dependencies
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_handler = DataHandler()
        self.multiplier = 1.0
        self.current_freq = 'D'
        self.audusd = None
        self.companies = None
        self.initUI()
        self.update_analysis()

    def initUI(self):
        self.setWindowTitle('Market Correlation Analyzer')
        self.setGeometry(100, 100, 1200, 800)
        self.center()
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.create_menu()
        self.create_toolbar()

    def create_menu(self):
        menubar = self.menuBar()
        
        # Frequency menu
        freq_menu = menubar.addMenu('&Frequency')
        freq_menu.addAction(self.create_action('&Daily', 'D'))
        freq_menu.addAction(self.create_action('&Weekly', 'W'))
        freq_menu.addAction(self.create_action('&Monthly', 'M'))
        
        # Analysis menu
        analysis_menu = menubar.addMenu('&Analysis')
        analysis_menu.addAction(self.create_action('&Basic Correlation', 'basic'))
        analysis_menu.addAction(self.create_action('&Cross Correlation', 'cross'))
        analysis_menu.addAction(self.create_action('&Rolling Correlation', 'rolling'))
        analysis_menu.addAction(self.create_action('&Mutual Information', 'mi'))
        analysis_menu.addAction(self.create_action('&Wavelet Coherence', 'wavelet'))
        analysis_menu.addAction(self.create_action('&Granger Causality', 'granger'))
        analysis_menu.addAction(self.create_action('&Copula Analysis', 'copula'))

    def create_toolbar(self):
        toolbar = QToolBar("Controls")
        self.addToolBar(toolbar)
        
        toolbar.addWidget(QLabel("Multiplier:"))
        self.multiplier_input = QTextEdit()
        self.multiplier_input.setMaximumHeight(25)
        self.multiplier_input.setMaximumWidth(70)
        self.multiplier_input.setText("1.0")
        toolbar.addWidget(self.multiplier_input)
        
        apply_action = QAction("Apply", self)
        apply_action.triggered.connect(self.update_multiplier)
        toolbar.addAction(apply_action)

    def update_multiplier(self):
        try:
            self.multiplier = float(self.multiplier_input.toPlainText())
            self.update_analysis()
        except ValueError:
            self.show_message("Invalid multiplier value. Using 1.0")
            self.multiplier = 1.0

    def set_frequency(self, freq):
        self.current_freq = freq
        self.update_analysis()

    def update_analysis(self, analysis_type='basic'):
        try:
            self.audusd, self.companies = self.data_handler.process_data(self.current_freq)
            
            self.figure.clear()
            
            if analysis_type == 'basic':
                self.plot_basic_correlation()
            elif analysis_type == 'cross':
                self.plot_cross_correlation()
            elif analysis_type == 'rolling':
                self.plot_rolling_correlation()
            elif analysis_type == 'mi':
                self.plot_mutual_info()
            elif analysis_type == 'wavelet':
                self.plot_wavelet_coherence()
            elif analysis_type == 'granger':
                self.plot_granger_causality()
            elif analysis_type == 'copula':
                self.plot_copula_analysis()
                
            self.canvas.draw()

        except Exception as e:
            self.show_message(f"Analysis failed: {str(e)}")
            self.figure.clear()
            self.canvas.draw()

    def plot_basic_correlation(self):
        gs = self.figure.add_gridspec(3, 1)
        
        # Plot 1: Market Neutral AUD/USD
        ax1 = self.figure.add_subplot(gs[0])
        ax1.plot(self.audusd, color='darkblue')
        ax1.set_title('Market Neutral AUD/USD Returns')
        
        # Plot 2: Companies Market Cap
        ax2 = self.figure.add_subplot(gs[1])
        ax2.plot(self.companies*100*self.multiplier, color='darkgreen')
        ax2.set_title('Companies Market Cap Changes')
        
        # Plot 3: Shifted Correlations
        ax3 = self.figure.add_subplot(gs[2])
        shifts, corr = CorrelationAnalyzer.calculate_shifted_correlations(self.audusd, self.companies)
        ax3.plot(shifts, corr, marker='o')
        ax3.set_title('Shifted Correlations')

    def plot_cross_correlation(self):
        lags, xcorr = CorrelationAnalyzer.cross_correlate(self.audusd, self.companies)
        ax = self.figure.add_subplot(111)
        ax.stem(lags, xcorr)
        ax.set_title('Cross-Correlation')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')

    def plot_rolling_correlation(self):
        rolling_corr = self.audusd.rolling(60).corr(self.companies)
        ax = self.figure.add_subplot(111)
        rolling_corr.plot(ax=ax)
        ax.set_title('60-Period Rolling Correlation')

    def plot_mutual_info(self):
        mi = CorrelationAnalyzer.mutual_information(self.audusd, self.companies)
        ax = self.figure.add_subplot(111)
        ax.bar(['Mutual Information'], [mi])
        ax.set_title(f'Mutual Information: {mi:.3f}')

    def plot_wavelet_coherence(self):
        wcoh = CorrelationAnalyzer.wavelet_coherence(self.audusd, self.companies)
        ax = self.figure.add_subplot(111)
        ax.imshow(wcoh, aspect='auto', cmap='viridis')
        ax.set_title('Wavelet Coherence')

    def plot_granger_causality(self):
        result = CorrelationAnalyzer.granger_causality(self.audusd, self.companies)
        text = "\n".join([f"Lag {k}: p={v[0]['ssr_ftest'][1]:.3f}" for k,v in result.items()])
        self.show_message(f"Granger Causality p-values:\n{text}")

    def plot_copula_analysis(self):
        tau = CorrelationAnalyzer.copula_analysis(self.audusd, self.companies)
        ax = self.figure.add_subplot(111)
        ax.bar(["Kendall's Tau"], [tau])
        ax.set_title(f"Copula Dependency: τ = {tau:.3f}")

    def create_action(self, text, data, callback=None):
        action = QAction(text, self)
        if callback is None:
            action.triggered.connect(lambda: self.update_analysis(data))
        else:
            action.triggered.connect(callback)
        return action

    def show_message(self, message):
        QMessageBox.information(self, "Information", message)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
class EnhancedMainWindow(MainWindow):
    def __init__(self):
        super().__init__()
        self.data_handler = EnhancedDataHandler()
        self.analyzer = EnhancedCorrelationAnalyzer()
        self.add_diagnostic_features()
        
    def add_diagnostic_features(self):
        menubar = self.menuBar()
        
        # Add Diagnostics menu
        diag_menu = menubar.addMenu('&Diagnostics')
        diag_menu.addAction(self.create_action('Run Data Check', None, self.run_data_check))
        diag_menu.addAction(self.create_action('Regime Analysis', None, self.run_regime_analysis))
        diag_menu.addAction(self.create_action('Volatility Analysis', None, self.run_volatility_analysis))
        
        # Add optimization menu
        opt_menu = menubar.addMenu('&Optimization')
        opt_menu.addAction(self.create_action('Find Optimal Lag', None, self.find_optimal_lag))
        opt_menu.addAction(self.create_action('Top 5% Moves Analysis', None, self.analyze_extreme_moves))

    def run_data_check(self):
        results = self.data_handler.diagnostic_check(self.current_freq)
        
        msg = []
        msg.append("=== Data Quality Check ===")
        msg.append(f"Basic Correlation: {results['basic_correlation']:.3f}")
        
        msg.append("\nStationarity (ADF p-values):")
        msg.append(f"AUD: {results['stationarity']['AUD']:.3f} {'(stationary)' if results['stationarity']['AUD'] < 0.05 else '(non-stationary)'}")
        msg.append(f"Companies: {results['stationarity']['Companies']:.3f} {'(stationary)' if results['stationarity']['Companies'] < 0.05 else '(non-stationary)'}")
        
        msg.append("\nDecorrelation Effectiveness:")
        msg.append(f"AUD-USD Residual Corr: {results['decorr_effectiveness']['AUD-USD']:.3f} {'(good)' if abs(results['decorr_effectiveness']['AUD-USD']) < 0.1 else '(weak)'}")
        msg.append(f"Companies-China Residual Corr: {results['decorr_effectiveness']['Comp-China']:.3f} {'(good)' if abs(results['decorr_effectiveness']['Comp-China']) < 0.1 else '(weak)'}")
        
        self.show_message("\n".join(msg))

    def run_regime_analysis(self):
        audusd, companies = self.data_handler.process_data(self.current_freq)
        results = self.analyzer.regime_analysis(audusd, companies, self.current_freq)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        regimes = list(results.keys())
        corrs = [v['correlation'] for v in results.values()]
        mis = [v['mutual_info'] for v in results.values()]
        
        x = np.arange(len(regimes))
        width = 0.35
        
        ax.bar(x - width/2, corrs, width, label='Correlation')
        ax.bar(x + width/2, mis, width, label='Mutual Info')
        
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.legend()
        ax.set_title('Regime-Specific Relationships')
        
        self.canvas.draw()

    def run_volatility_analysis(self):
        audusd, companies = self.data_handler.process_data(self.current_freq)
        vol_corr = self.analyzer.volatility_adjusted_correlation(audusd, companies)
        
        self.show_message(
            f"Volatility-Adjusted Correlation: {vol_corr:.3f}\n\n"
            "Interpretation:\n"
            "> 0.2: Significant relationship\n"
            "0.1-0.2: Weak relationship\n"
            "< 0.1: No meaningful relationship"
        )

    def find_optimal_lag(self):
        audusd, companies = self.data_handler.process_data(self.current_freq)
        lags, xcorr = self.analyzer.cross_correlate(audusd, companies)
        
        best_idx = np.nanargmax(np.abs(xcorr))
        best_lag = lags[best_idx]
        best_corr = xcorr[best_idx]
        
        self.show_message(
            f"Optimal Lag: {best_lag} periods\n"
            f"Correlation at lag: {best_corr:.3f}\n\n"
            f"Interpretation:\n"
            f"Positive lag: AUD leads Companies\n"
            f"Negative lag: Companies lead AUD"
        )

    def analyze_extreme_moves(self):
        audusd, companies = self.data_handler.process_data(self.current_freq)
        
        # Get top/bottom 5% moves
        aud_extreme = audusd[abs(audusd) > audusd.quantile(0.95)]
        comp_extreme = companies[abs(companies) > companies.quantile(0.95)]
        
        # Align extremes
        aligned = pd.concat([aud_extreme, comp_extreme], axis=1).dropna()
        
        if len(aligned) > 10:
            extreme_corr = aligned.iloc[:,0].corr(aligned.iloc[:,1])
            mi = mutual_info_regression(aligned.iloc[:,0].values.reshape(-1,1), 
                                      aligned.iloc[:,1].values)[0]
            
            self.show_message(
                "Extreme Moves Analysis (Top/Bottom 5%):\n"
                f"Correlation: {extreme_corr:.3f}\n"
                f"Mutual Information: {mi:.3f}\n\n"
                "Interpretation:\n"
                "High values suggest tail-risk dependence"
            )
        else:
            self.show_message("Insufficient extreme events for analysis")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = EnhancedMainWindow()
    main.show()
    sys.exit(app.exec_())