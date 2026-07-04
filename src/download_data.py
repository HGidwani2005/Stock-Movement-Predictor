import os
import yfinance as yf

# Create directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Stocks to download
stocks = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA"
]

start_date = "2015-01-01"
end_date = "2025-01-01"

for stock in stocks:
    print(f"Downloading {stock}...")

    df = yf.download(
    stock,
    start=start_date,
    end=end_date,
    auto_adjust=False,
    group_by="column",
    multi_level_index=False
    )

    file_path = f"data/raw/{stock}.csv"
    df.to_csv(file_path)

    print(f"Saved to {file_path}")

print("\nAll stock data downloaded successfully!")