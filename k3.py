import sys
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QSplitter, QSizePolicy)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib import cm

# Configuration
TICKERS = ['600019.SS', '000898.SZ', '000709.SZ', '000932.SZ', '600808.SS', '601005.SS', '000959.SZ']
CURRENCY_PAIR = 'AUDUSD=X'
DAYS_TO_FETCH = 360
MAX_SHIFT_DAYS = 20
SHIFT_STEP = 1
Y_SHIFT_RANGE = np.linspace(-0.1, 0.1, 21)

class SteelAUDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Steel Sector vs AUD/USD Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.combined_data = None
        self.beta = None
        self.results = {}
        self.predictions = {}
        
        # Initialize UI
        self.init_ui()
        
        # Load data
        self.load_data()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Main layout
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # Left panel (controls)
        self.control_panel = QWidget()
        self.control_panel.setFixedWidth(250)
        self.control_layout = QVBoxLayout()
        self.control_panel.setLayout(self.control_layout)
        
        # Right panel (plots)
        self.plot_panel = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_panel.setLayout(self.plot_layout)
        
        # Add panels to main layout
        self.main_layout.addWidget(self.control_panel)
        self.main_layout.addWidget(self.plot_panel)
        
        # Create UI elements
        self.create_menu()
        self.create_plot_area()
        
        # Status label
        self.status_label = QLabel("Ready")
        self.main_layout.addWidget(self.status_label)

    def create_menu(self):
        """Create interactive menu buttons"""
        btn_style = """
        QPushButton {
            padding: 10px;
            font-size: 13px;
            min-width: 150px;
            max-width: 200px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #f0f0f0;
        }
        """
        
        # Analysis buttons
        self.control_layout.addWidget(QLabel("<h3>Analysis Menu</h3>"))
        
        buttons = [
            ("Initial Analysis", self.show_initial_analysis),
            ("Daily Analysis", lambda: self.show_timeframe_analysis('D')),
            ("Weekly Analysis", lambda: self.show_timeframe_analysis('W')),
            ("Monthly Analysis", lambda: self.show_timeframe_analysis('M')),
            ("Bias Predictions", self.show_bias_predictions)
        ]
        
        for text, callback in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(callback)
            self.control_layout.addWidget(btn)
        
        # Spacer
        self.control_layout.addStretch()
        
        # Data controls
        self.control_layout.addWidget(QLabel("<h3>Data Controls</h3>"))
        
        refresh_btn = QPushButton("Refresh Data")
        refresh_btn.setStyleSheet(btn_style)
        refresh_btn.clicked.connect(self.refresh_data)
        self.control_layout.addWidget(refresh_btn)
        
        # Timeframe selector for bias predictions
        self.control_layout.addWidget(QLabel("<b>Bias Timeframe:</b>"))
        self.bias_freq_selector = QComboBox()
        self.bias_freq_selector.addItems(["Daily", "Weekly", "Monthly"])
        self.bias_freq_selector.setCurrentIndex(0)
        self.control_layout.addWidget(self.bias_freq_selector)
        
        # Prediction display options
        self.control_layout.addWidget(QLabel("<b>Display Options:</b>"))
        self.show_cumulative = QComboBox()
        self.show_cumulative.addItems(["Show Direction Only", "Show Cumulative Bias"])
        self.control_layout.addWidget(self.show_cumulative)
        
        # Add some padding at bottom
        self.control_layout.addStretch()
    
    def create_plot_area(self):
        """Initialize the plot area"""
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_layout.addWidget(self.canvas)
    
    def refresh_data(self):
        """Reload all data"""
        self.status_label.setText("Refreshing data...")
        QApplication.processEvents()
        self.load_data()
    
    def download_data(self):
        """Download stock and AUD data from Yahoo Finance"""
        end_date = datetime.today()
        start_date = end_date - timedelta(days=DAYS_TO_FETCH)
        
        try:
            # Download stock data
            stock_data = yf.download(
                TICKERS,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker',
                auto_adjust=False
            )
            
            # Download AUD/USD
            aud_data = yf.download(
                CURRENCY_PAIR,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )
            
            return stock_data, aud_data
            
        except Exception as e:
            print(f"Download error: {e}")
            return None, None
    
    def process_data(self, stock_data, aud_data):
        """Process raw data into combined DataFrame"""
        sector_returns = []
        
        for ticker in TICKERS:
            try:
                if (ticker, 'Close') in stock_data.columns:
                    returns = stock_data[(ticker, 'Close')].pct_change().dropna().squeeze()
                    if isinstance(returns, pd.Series):
                        sector_returns.append(returns)
            except:
                continue
        
        if not sector_returns:
            return None, None
        
        # Calculate sector average returns
        sector_avg = pd.concat(sector_returns, axis=1).mean(axis=1)
        sector_avg.name = 'sector_daily'
        
        # Process AUD/USD returns
        aud_returns = aud_data['Close'].pct_change().dropna().squeeze()
        if not isinstance(aud_returns, pd.Series):
            aud_returns = pd.Series(aud_returns, index=aud_data.index[1:])
        aud_returns.name = 'aud_daily'
        
        # Combine data
        combined = pd.concat([sector_avg, aud_returns], axis=1).dropna()
        combined.columns = ['sector_daily', 'aud_daily']
        
        if combined.empty:
            return None, None
        
        # Calculate market-neutral returns
        if len(combined) > 5:
            X = combined['aud_daily'].values.reshape(-1, 1)
            y = combined['sector_daily'].values
            model = LinearRegression().fit(X, y)
            beta = model.coef_[0]
            combined['market_neutral'] = combined['sector_daily'] - (beta * combined['aud_daily'])
        else:
            combined['market_neutral'] = combined['sector_daily']
            beta = 0
        
        return combined, beta
    
    def resample_data(self, df, freq='D'):
        """Resample data to specified frequency"""
        if freq == 'W':
            return df.resample('W-FRI').last()
        elif freq == 'M':
            return df.resample('M').last()
        return df
    
    def analyze_timeframe(self, freq, plot=True):
        """Analyze a specific timeframe"""
        resampled = self.resample_data(self.combined_data, freq)
        
        market_values = resampled['market_neutral'].to_numpy()
        aud_values = resampled['aud_daily'].to_numpy()
        dates = resampled.index
        
        all_results = []
        for y_shift in Y_SHIFT_RANGE:
            shifted_market = market_values + y_shift
            shift_results = []
            
            for time_shift in np.arange(-MAX_SHIFT_DAYS, MAX_SHIFT_DAYS + SHIFT_STEP, SHIFT_STEP):
                if time_shift < 0:
                    market_shifted = np.roll(shifted_market, -int(time_shift))
                    market_shifted[-int(time_shift):] = np.nan
                    aud_shifted = aud_values
                else:
                    market_shifted = shifted_market
                    aud_shifted = np.roll(aud_values, int(time_shift))
                    aud_shifted[:int(time_shift)] = np.nan
                
                valid_idx = ~np.isnan(market_shifted) & ~np.isnan(aud_shifted)
                if valid_idx.sum() > 5:
                    corr, p_value = pearsonr(market_shifted[valid_idx], aud_shifted[valid_idx])
                    shift_results.append({
                        'shift': time_shift,
                        'y_shift': y_shift,
                        'correlation': corr,
                        'p_value': p_value,
                        'n_obs': valid_idx.sum()
                    })
            
            all_results.append(pd.DataFrame(shift_results))
        
        full_results = pd.concat(all_results, ignore_index=True)
        
        if plot:
            self.plot_timeframe_analysis(freq, full_results, resampled)
        
        return full_results
    
    def generate_bias_predictions(self):
        """Generate predictions for all timeframes"""
        predictions = {}
        
        for freq in ['D', 'W', 'M']:
            df = self.results[freq]
            df_significant = df[df['p_value'] < 0.1]
            
            if not df_significant.empty:
                best = df_significant.loc[df_significant['correlation'].idxmax()]
            else:
                best = df.loc[df['correlation'].idxmax()]
            
            resampled = self.resample_data(self.combined_data, freq)
            market_values = resampled['market_neutral'].to_numpy()
            aud_values = resampled['aud_daily'].to_numpy()
            dates = resampled.index
            
            market_shifted = market_values + best['y_shift']
            if best['shift'] < 0:
                market_shifted = np.roll(market_shifted, -int(best['shift']))
                market_shifted[-int(best['shift']):] = np.nan
                aud_shifted = aud_values
                prediction_type = f"AUD/USD {abs(best['shift'])} days ahead"
            else:
                aud_shifted = np.roll(aud_values, int(best['shift']))
                aud_shifted[:int(best['shift'])] = np.nan
                prediction_type = f"Steel Sector {best['shift']} days ahead"
            
            valid_idx = ~np.isnan(market_shifted) & ~np.isnan(aud_shifted)
            correlation_sign = np.sign(best['correlation'])
            
            predictions[freq] = {
                'dates': dates[valid_idx],
                'market_actual': market_values[valid_idx],
                'aud_actual': aud_values[valid_idx],
                'market_shifted': market_shifted[valid_idx],
                'aud_shifted': aud_shifted[valid_idx],
                'prediction_type': prediction_type,
                'correlation': best['correlation'],
                'p_value': best['p_value'],
                'correlation_sign': correlation_sign,
                'optimal_shift': best['shift'],
                'optimal_y_shift': best['y_shift']
            }
        
        return predictions
    
    def load_data(self):
        """Download and process data"""
        self.status_label.setText("Downloading data...")
        QApplication.processEvents()
        
        stock_data, aud_data = self.download_data()
        if stock_data is None or aud_data is None:
            self.status_label.setText("Failed to download data")
            return False
        
        self.combined_data, self.beta = self.process_data(stock_data, aud_data)
        if self.combined_data is None:
            self.status_label.setText("No valid data after processing")
            return False
        
        # Pre-compute results
        self.status_label.setText("Calculating daily analysis...")
        QApplication.processEvents()
        self.results['D'] = self.analyze_timeframe('D', plot=False)
        
        self.status_label.setText("Calculating weekly analysis...")
        QApplication.processEvents()
        self.results['W'] = self.analyze_timeframe('W', plot=False)
        
        self.status_label.setText("Calculating monthly analysis...")
        QApplication.processEvents()
        self.results['M'] = self.analyze_timeframe('M', plot=False)
        
        self.predictions = self.generate_bias_predictions()
        self.status_label.setText("Data loaded successfully")
        return True
    
    def show_initial_analysis(self):
        """Plot initial time series"""
        if self.combined_data is None:
            return
        
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212, sharex=ax1)
        
        # Market-neutral returns
        ax1.plot(self.combined_data.index, self.combined_data['market_neutral'], 
                'b-', label=f'Market-Neutral (β={self.beta:.2f})', linewidth=1.5)
        ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax1.set_ylabel('Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # AUD/USD returns
        ax2.plot(self.combined_data.index, self.combined_data['aud_daily'], 
                'r-', label='AUD/USD', linewidth=1.5)
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax2.set_ylabel('Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.figure.autofmt_xdate()
        
        self.figure.suptitle("Initial Time Series Analysis", y=0.98)
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_timeframe_analysis(self, freq, full_results, resampled):
        """Plot analysis for specific timeframe"""
        max_corr = full_results.loc[full_results['correlation'].idxmax()].copy()
        min_corr = full_results.loc[full_results['correlation'].idxmin()].copy()
        
        self.figure.clear()
        
        # Heatmap
        ax1 = self.figure.add_subplot(131)
        heatmap_data = full_results.pivot_table(index='y_shift', columns='shift', values='correlation')
        im = ax1.imshow(heatmap_data, cmap='coolwarm', aspect='auto',
                       extent=[-MAX_SHIFT_DAYS, MAX_SHIFT_DAYS, Y_SHIFT_RANGE[-1], Y_SHIFT_RANGE[0]])
        self.figure.colorbar(im, ax=ax1, label='Correlation')
        ax1.set_xlabel('Time Shift (days)')
        ax1.set_ylabel('Y-axis Shift')
        ax1.axvline(0, color='k', linestyle='--', linewidth=0.5)
        ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax1.set_title('Correlation Heatmap')
        
        # Best positive correlation
        ax2 = self.figure.add_subplot(132)
        market_values = resampled['market_neutral'].to_numpy()
        aud_values = resampled['aud_daily'].to_numpy()
        dates = resampled.index
        
        market_shifted = market_values + max_corr['y_shift']
        if max_corr['shift'] < 0:
            market_shifted = np.roll(market_shifted, -int(max_corr['shift']))
            market_shifted[-int(max_corr['shift']):] = np.nan
            aud_shifted = aud_values
        else:
            aud_shifted = np.roll(aud_values, int(max_corr['shift']))
            aud_shifted[:int(max_corr['shift'])] = np.nan
        
        valid_idx = ~np.isnan(market_shifted) & ~np.isnan(aud_shifted)
        ax2.plot(dates[valid_idx], market_shifted[valid_idx], 
                'b-', label=f"Market (shift {max_corr['shift']}d, y+{max_corr['y_shift']:.3f})", linewidth=1.5)
        ax2.plot(dates[valid_idx], aud_shifted[valid_idx], 'r-', label='AUD/USD', linewidth=1.5)
        ax2.set_title(f"Best Positive ρ={max_corr['correlation']:.3f}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Best negative correlation
        ax3 = self.figure.add_subplot(133)
        market_shifted = market_values + min_corr['y_shift']
        if min_corr['shift'] < 0:
            market_shifted = np.roll(market_shifted, -int(min_corr['shift']))
            market_shifted[-int(min_corr['shift']):] = np.nan
            aud_shifted = aud_values
        else:
            aud_shifted = np.roll(aud_values, int(min_corr['shift']))
            aud_shifted[:int(min_corr['shift'])] = np.nan
        
        valid_idx = ~np.isnan(market_shifted) & ~np.isnan(aud_shifted)
        ax3.plot(dates[valid_idx], market_shifted[valid_idx], 
                'b-', label=f"Market (shift {min_corr['shift']}d, y+{min_corr['y_shift']:.3f})", linewidth=1.5)
        ax3.plot(dates[valid_idx], aud_shifted[valid_idx], 'r-', label='AUD/USD', linewidth=1.5)
        ax3.set_title(f"Best Negative ρ={min_corr['correlation']:.3f}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        self.figure.suptitle(f"{freq.upper()} Lead-Lag Analysis", y=0.98)
        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_timeframe_analysis(self, freq):
        """Show analysis for specific timeframe"""
        if freq not in self.results:
            return
        
        self.figure.clear()
        full_results = self.results[freq]
        resampled = self.resample_data(self.combined_data, freq)
        self.plot_timeframe_analysis(freq, full_results, resampled)
    
    def show_bias_predictions(self):
        """Show bias predictions with proper formatting"""
        if not self.predictions:
            return
        
        freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
        selected_freq = freq_map[self.bias_freq_selector.currentText()]
        
        if selected_freq not in self.predictions:
            return
        
        data = self.predictions[selected_freq]
        show_cumulative = self.show_cumulative.currentIndex() == 1
        
        self.figure.clear()
        
        # Create subplots
        if show_cumulative:
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)
        else:
            ax1 = self.figure.add_subplot(111)
        
        # Plot actual vs shifted series
        ax1.plot(data['dates'], data['market_actual'], 'b-', label='Steel (Actual)', alpha=0.8, linewidth=1.5)
        ax1.plot(data['dates'], data['aud_actual'], 'r-', label='AUD/USD (Actual)', alpha=0.8, linewidth=1.5)
        ax1.plot(data['dates'], data['market_shifted'], 'b--', 
                label=f"Steel (Shifted {data['optimal_shift']}d)", linewidth=2)
        ax1.plot(data['dates'], data['aud_shifted'], 'r--', 
                label=f"AUD/USD (Shifted {data['optimal_shift']}d)", linewidth=2)
        ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax1.set_ylabel('Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot bias indicators
        if data['optimal_shift'] < 0:
            # Steel leads AUD: Predict AUD bias
            predicted_bias = np.sign(data['market_shifted']) * data['correlation_sign']
            label = 'Predicted AUD/USD Direction'
            cmap = cm.get_cmap('RdYlGn')
        else:
            # AUD leads Steel: Predict Steel bias
            predicted_bias = np.sign(data['aud_shifted']) * data['correlation_sign']
            label = 'Predicted Steel Direction'
            cmap = cm.get_cmap('RdYlGn')
        
        if show_cumulative:
            # Plot cumulative bias
            cumulative_bias = np.cumsum(predicted_bias)
            ax2.plot(data['dates'], cumulative_bias, 'g-', label='Cumulative Bias', linewidth=2)
            ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax2.set_ylabel('Cumulative Bias')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Plot directional markers
            colors = cmap((predicted_bias + 1)/2)  # Map [-1,1] to [0,1] for colormap
            for date, bias, color in zip(data['dates'], predicted_bias, colors):
                ax1.scatter(date, 0, c=[color], s=100, alpha=0.7, 
                           edgecolors='k', linewidths=0.5)
            
            # Create proxy artists for legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Up Bias',
                        markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Down Bias',
                        markerfacecolor='red', markersize=10)
            ]
            ax1.legend(handles=legend_elements, title=label)
        
        title = f"{self.bias_freq_selector.currentText()} Bias Prediction\n"
        title += f"Shift: {data['optimal_shift']} days | "
        title += f"Correlation: {data['correlation']:.2f} (p={data['p_value']:.3f})"
        self.figure.suptitle(title, y=0.98)
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    window = SteelAUDApp()
    window.show()
    sys.exit(app.exec_())