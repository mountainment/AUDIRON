import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
from io import StringIO
from datetime import datetime

# Configuration
plt.style.use('seaborn')
sns.set_palette("husl")

def fetch_aus_pmi():
    """
    Fetch Australian Manufacturing PMI data from Markit Economics
    Note: Using a proxy since Markit requires subscription
    """
    try:
        # Alternative source since Markit requires subscription
        url = "https://www.markiteconomics.com/Public/GetReleasedFiles?entityId=3"
        # This might not work directly - Markit usually requires authentication
        
        # Alternative approach: Use Quandl or another data provider
        print("Markit Economics requires subscription. Using alternative source...")
        
        # Using a sample dataset from RBA (Reserve Bank of Australia)
        rba_url = "https://www.rba.gov.au/statistics/tables/xls/historical-business-conditions.xls"
        response = requests.get(rba_url)
        
        # Load Excel file
        pmi_data = pd.read_excel(response.content, sheet_name="Data", skiprows=9)
        pmi_data = pmi_data[['Date', 'Manufacturing PMI (a)']].dropna()
        pmi_data.columns = ['date', 'pmi']
        pmi_data['date'] = pd.to_datetime(pmi_data['date'])
        pmi_data = pmi_data.sort_values('date')
        
        return pmi_data
    
    except Exception as e:
        print(f"Error fetching PMI data: {e}")
        return None

def fetch_abs_gdp():
    """
    Fetch Australian GDP growth data from ABS API
    """
    try:
        # ABS API endpoint for GDP
        abs_url = "https://api.data.abs.gov.au/data/ABS,NAB_QUARTERLY,1.2.1...Q?startPeriod=2000-Q1"
        
        response = requests.get(abs_url)
        data = response.json()
        
        # Process JSON response
        gdp_data = []
        for obs in data['data']['dataSets'][0]['series']['0']['observations']:
            period = data['data']['structure']['dimensions']['observation'][0]['values'][int(obs)]['id']
            value = data['data']['dataSets'][0]['series']['0']['observations'][obs][0]
            gdp_data.append({'date': period, 'gdp_growth': value})
        
        gdp_df = pd.DataFrame(gdp_data)
        gdp_df['date'] = pd.to_datetime(gdp_df['date'].str.replace('-Q1', '-03-31')
                                      .str.replace('-Q2', '-06-30')
                                      .str.replace('-Q3', '-09-30')
                                      .str.replace('-Q4', '-12-31'))
        gdp_df = gdp_df.sort_values('date')
        
        # Calculate quarter-over-quarter growth if needed
        gdp_df['gdp_growth'] = gdp_df['gdp_growth'].pct_change() * 100
        
        return gdp_df
    
    except Exception as e:
        print(f"Error fetching GDP data: {e}")
        # Fallback to RBA data
        print("Trying RBA fallback...")
        try:
            rba_url = "https://www.rba.gov.au/statistics/tables/xls/gdp-growth-quarterly.xls"
            response = requests.get(rba_url)
            gdp_data = pd.read_excel(response.content, sheet_name="Data", skiprows=9)
            gdp_data = gdp_data[['Date', 'GDP growth']].dropna()
            gdp_data.columns = ['date', 'gdp_growth']
            gdp_data['date'] = pd.to_datetime(gdp_data['date'])
            return gdp_data.sort_values('date')
        except Exception as e2:
            print(f"Fallback failed: {e2}")
            return None

def analyze_correlation(pmi_data, gdp_data):
    """
    Analyze correlation between PMI and GDP growth with time lags
    """
    # Resample PMI to quarterly by taking the mean
    pmi_quarterly = pmi_data.resample('Q', on='date').mean().reset_index()
    
    # Merge datasets
    merged = pd.merge_asof(
        gdp_data.sort_values('date'),
        pmi_quarterly.sort_values('date'),
        on='date',
        direction='backward'
    ).dropna()
    
    # Calculate correlations with different lags (0-3 quarters)
    results = []
    for lag in range(0, 4):
        merged[f'pmi_lag_{lag}'] = merged['pmi'].shift(lag)
        corr = merged['gdp_growth'].corr(merged[f'pmi_lag_{lag}'])
        results.append({'Lag (quarters)': lag, 'Correlation': corr})
    
    correlation_df = pd.DataFrame(results)
    
    # Plot correlations by lag
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Lag (quarters)', y='Correlation', data=correlation_df)
    plt.title("Correlation Between PMI and GDP Growth by Time Lag")
    plt.axhline(0, color='black', linestyle='--')
    plt.show()
    
    # Find best lag
    best_lag = correlation_df.loc[correlation_df['Correlation'].abs().idxmax()]
    print(f"\nHighest correlation at lag {best_lag['Lag (quarters)']} quarters: {best_lag['Correlation']:.3f}")
    
    # Plot relationship with best lag
    plt.figure(figsize=(10, 6))
    sns.regplot(x=f'pmi_lag_{int(best_lag["Lag (quarters)"])}', y='gdp_growth', data=merged)
    plt.title(f"Australian Manufacturing PMI vs GDP Growth (Lag {best_lag['Lag (quarters)']}Q)\nCorrelation: {best_lag['Correlation']:.2f}")
    plt.xlabel(f"Manufacturing PMI (lagged {best_lag['Lag (quarters)']} quarters)")
    plt.ylabel("GDP Growth (%)")
    plt.grid(True)
    plt.show()
    
    # Statistical significance
    r, p = stats.pearsonr(
        merged[f'pmi_lag_{int(best_lag["Lag (quarters)"])}'].dropna(),
        merged['gdp_growth'].dropna()
    )
    
    print("\n=== Statistical Significance ===")
    print(f"Correlation coefficient: {r:.3f}")
    print(f"P-value: {p:.4f}")
    if p < 0.05:
        print("The correlation is statistically significant at p < 0.05")
    else:
        print("The correlation is not statistically significant")
    
    return correlation_df

def main():
    print("Fetching Australian Manufacturing PMI data...")
    pmi_data = fetch_aus_pmi()
    
    print("\nFetching Australian GDP growth data...")
    gdp_data = fetch_abs_gdp()
    
    if pmi_data is None or gdp_data is None:
        print("\nError: Failed to fetch required data. Possible solutions:")
        print("1. Check your internet connection")
        print("2. The API endpoints might have changed")
        print("3. Try using manual CSV files instead")
        return
    
    print("\nPMI Data Sample:")
    print(pmi_data.head())
    print("\nGDP Data Sample:")
    print(gdp_data.head())
    
    # Analyze correlation
    results = analyze_correlation(pmi_data, gdp_data)
    
    print("\n=== Correlation Results ===")
    print(results)
    
    # Save data for future use
    pmi_data.to_csv("australian_pmi.csv", index=False)
    gdp_data.to_csv("australian_gdp.csv", index=False)
    print("\nData saved to australian_pmi.csv and australian_gdp.csv")

if __name__ == "__main__":
    main()