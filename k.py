import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import scipy.stats as stats
import numpy as np

# Configuration
TICKERS = ['600019.SS', '000898.SZ', '000709.SZ', '000932.SZ', '600808.SS', '601005.SS', '000959.SZ']
CURRENCY_PAIR = 'AUDUSD=X'
DAYS_TO_FETCH = 90  # 3 months for reliable correlation

def download_data():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    
    print(f"Downloading data from {start_date.date()} to {end_date.date()}")
    
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

def process_data(stock_data, aud_data):
    # First, verify we have the expected data
    if stock_data.empty or aud_data.empty:
        print("Error: Empty data received")
        return None, None
    
    print("\nRaw data preview:")
    print("Stock data columns:", stock_data.columns)
    print("AUD data columns:", aud_data.columns)
    
    # Calculate sector average returns
    sector_returns = []
    valid_tickers = 0
    
    for ticker in TICKERS:
        try:
            # Check if the ticker exists in the multi-index columns
            if (ticker, 'Close') in stock_data.columns:
                # Ensure we get a 1D series by using squeeze()
                returns = stock_data[(ticker, 'Close')].pct_change().dropna().squeeze()
                if isinstance(returns, pd.Series):  # Make sure we have a Series
                    sector_returns.append(returns)
                    valid_tickers += 1
                    print(f"Successfully processed {ticker}")
                else:
                    print(f"Warning: Unexpected data format for {ticker}")
            else:
                print(f"Warning: No Close data for {ticker}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    if valid_tickers == 0:
        print("Error: No valid tickers processed")
        return None, None
    
    # Calculate average sector returns
    sector_avg = pd.concat(sector_returns, axis=1).mean(axis=1)
    sector_avg.name = 'sector_daily'
    
    # Process AUD/USD returns - ensure 1D series
    try:
        aud_returns = aud_data['Close'].pct_change().dropna().squeeze()
        if not isinstance(aud_returns, pd.Series):
            aud_returns = pd.Series(aud_returns, index=aud_data.index[1:])
        aud_returns.name = 'aud_daily'
    except Exception as e:
        print(f"Error processing AUD data: {e}")
        print("AUD Data Structure:")
        print(aud_data.head())
        return None, None
    
    # Combine data - ensuring proper alignment and 1D series
    combined = pd.DataFrame({
        'sector_daily': sector_avg,
        'aud_daily': aud_returns
    }).dropna()
    
    print("\nCombined data preview:")
    print(combined.head())
    
    if combined.empty:
        print("Error: No data after combining")
        return None, None
    
    print("\nDebugging date issues:")
    print("Index type:", type(combined.index))
    print("First index value:", combined.index[0])
    print("First index value type:", type(combined.index[0]))
    print("Sample index values:", combined.index[:5])
    # Calculate market-neutral returns
    if len(combined) > 5:
        try:
            X = combined['aud_daily'].values.reshape(-1, 1)  # Ensure 2D for sklearn
            y = combined['sector_daily'].values
            model = LinearRegression().fit(X, y)
            beta = model.coef_[0]
            combined['market_neutral'] = combined['sector_daily'] - (beta * combined['aud_daily'])
        except Exception as e:
            print(f"Error in regression: {e}")
            print("Current combined columns:", combined.columns)
            combined['market_neutral'] = combined['sector_daily']
            beta = 0
    else:
        combined['market_neutral'] = combined['sector_daily']
        beta = 0
    
    return combined, beta

# Main execution
print("=== STEEL SECTOR vs AUD/USD CORRELATION ANALYSIS ===")
stock_data, aud_data = download_data()

if stock_data is None or aud_data is None:
    print("Failed to download data")
    exit()

try:
    combined, beta = process_data(stock_data, aud_data)
except Exception as e:
    print(f"Error in process_data: {e}")
    exit()

if combined is None:
    print("No valid data after processing")
    exit()

# Calculate correlation
try:
    corr_coef, p_value = stats.pearsonr(combined['market_neutral'], combined['aud_daily'])
except Exception as e:
    print(f"Error calculating correlation: {e}")
    print("Available columns in combined data:", combined.columns)
    exit()

# Visualization
try:
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])
    
    # 1. Market-neutral returns time series
    ax1.plot(combined.index, combined['market_neutral'], 'b-', 
             label=f'Market-Neutral Returns (β={beta:.2f})')
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Daily Returns', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # 2. AUD/USD changes time series
    ax2.plot(combined.index, combined['aud_daily'], 'r-', 
             label='AUD/USD Daily Returns')
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax2.set_ylabel('Daily Returns', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    # 3. Scatter plot with regression - FIXED DATE COLORING
    # Use numerical representation of dates for coloring
    date_nums = np.arange(len(combined))  # Simple sequential numbering
    
    sc = ax3.scatter(combined['aud_daily'], 
                    combined['market_neutral'], 
                    c=date_nums,  # Using position instead of direct dates
                    cmap='viridis',
                    alpha=0.7,
                    label=f'Correlation ρ = {corr_coef:.2f} (p = {p_value:.3f})')
    
    # Add regression line
    if len(combined) > 1:
        x = combined['aud_daily']
        y = combined['market_neutral']
        model = LinearRegression().fit(x.values.reshape(-1, 1), y)
        ax3.plot(x, model.predict(x.values.reshape(-1, 1)), 
                 'purple', linewidth=2, label='Regression Line')
    
    # Custom colorbar with actual dates
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label('Timeline')
    
    # Set colorbar ticks to show actual dates
    tick_positions = np.linspace(0, len(combined)-1, 5)  # 5 ticks
    tick_labels = [combined.index[int(i)].strftime('%Y-%m-%d') for i in tick_positions]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    
    ax3.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax3.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax3.set_xlabel('AUD/USD Daily Returns', fontsize=10)
    ax3.set_ylabel('Market-Neutral Returns', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)
    
    # Format x-axis dates for time series plots
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    # Set title with date range
    start_date = combined.index[0].strftime('%Y-%m-%d')
    end_date = combined.index[-1].strftime('%Y-%m-%d')
    fig.suptitle(f"Steel Sector vs AUD/USD Correlation Analysis\n{start_date} to {end_date}", 
                 y=1.02, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error in visualization: {e}")
    print("Combined DataFrame info:")
    print(combined.info())
    print("Sample data:")
    print(combined.head())

    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Period: {combined.index[0].strftime('%Y-%m-%d')} to {combined.index[-1].strftime('%Y-%m-%d')}")
    print(f"Observations: {len(combined)}")
    print(f"Correlation (ρ): {corr_coef:.3f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Beta (AUD exposure): {beta:.3f}")
    print(f"\nInterpretation: {'Significant' if p_value < 0.05 else 'Not significant'} correlation")
    print(f"Direction: {'Positive' if corr_coef > 0 else 'Negative'} relationship")

except Exception as e:
    print(f"Error in visualization: {e}")