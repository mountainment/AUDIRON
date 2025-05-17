import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QComboBox
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pickle  # Used for saving the data to a file

# Define companies with their respective exchange tickers
companies = ['600019.SS', '000898.SZ', '000709.SZ', '000932.SZ', '600808.SS', '601005.SS', '000959.SZ']

# Fetch and save data
def get_data():
    # Fetch AUD/USD data
    audusd = yf.download('AUDUSD=X', period='2y', interval='1d')['Adj Close']
    
    market_caps = {}

    for ticker in companies:
        print(f"Fetching data for: {ticker}")  # Debugging
        market_caps[ticker] = yf.download(ticker, period='2y', interval='1d')['Adj Close']
    
    # Save the fetched data into a file
    with open("market_data.pkl", "wb") as f:
        pickle.dump((audusd, pd.DataFrame(market_caps)), f)
    
    return audusd, pd.DataFrame(market_caps)

# Load saved data
def load_data():
    try:
        with open("market_data.pkl", "rb") as f:
            audusd, market_caps = pickle.load(f)
        return audusd, market_caps
    except FileNotFoundError:
        return None, None

# Calculate decorrelated AUD/USD
def decorrelate_usd(audusd, market_caps):
    usd_index = market_caps.mean(axis=1)
    decorrelated_audusd = audusd / usd_index
    return decorrelated_audusd

# Compute percentage changes
def compute_changes(data, freq):
    return data.pct_change(freq).dropna()

# Find best correlation shift
def find_best_shift(audusd_changes, market_caps_changes):
    max_corr, best_shift = -1, 0
    for shift in range(-30, 31):
        shifted_mcaps = market_caps_changes.shift(shift)
        correlation = audusd_changes.corr(shifted_mcaps.mean(axis=1))
        if correlation > max_corr:
            max_corr, best_shift = correlation, shift
    return best_shift, max_corr

# GUI Application
class CorrelationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AUD/USD Correlation Analysis")
        self.setGeometry(100, 100, 800, 600)
        
        # Try to load previously saved data
        self.audusd, self.market_caps = load_data()
        
        # If data is not loaded, fetch and save it
        if self.audusd is None or self.market_caps is None:
            print("Fetching data...")
            self.audusd, self.market_caps = get_data()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.canvas = FigureCanvas(plt.figure())
        
        self.freq_combo = QComboBox()
        self.freq_combo.addItems(["Daily", "Weekly", "Monthly"])
        self.freq_combo.currentIndexChanged.connect(self.update_plot)
        
        self.analyze_button = QPushButton("Analyze Correlation")
        self.analyze_button.clicked.connect(self.update_plot)
        
        layout.addWidget(QLabel("Select Frequency:"))
        layout.addWidget(self.freq_combo)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.canvas)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def update_plot(self):
        # Use loaded data
        audusd, market_caps = self.audusd, self.market_caps
        
        # Decorrelate AUD/USD
        decorrelated_audusd = decorrelate_usd(audusd, market_caps)
        
        # Get selected frequency
        freq_map = {"Daily": 1, "Weekly": 5, "Monthly": 21}
        freq = freq_map[self.freq_combo.currentText()]
        
        # Calculate changes
        audusd_changes = compute_changes(decorrelated_audusd, freq)
        market_caps_changes = compute_changes(market_caps.mean(axis=1), freq)
        
        # Find best shift for correlation
        best_shift, max_corr = find_best_shift(audusd_changes, market_caps_changes)
        
        # Update the plot
        plt.clf()
        plt.plot(audusd_changes.index, audusd_changes, label="AUD/USD Changes")
        plt.plot(market_caps_changes.index, market_caps_changes, label="Market Cap Changes")
        plt.axvline(x=best_shift, color='r', linestyle='--', label=f'Best Shift: {best_shift} days')
        plt.title(f"Best Correlation: {max_corr:.2f} at shift {best_shift} days")
        plt.legend()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = CorrelationApp()
    mainWin.show()
    sys.exit(app.exec())
