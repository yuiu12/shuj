'''
你可能会问，为什么 ETF 应该引起如此关注？这些金融船只令人着迷，原因如下：

瓶子里的多样性：ETF 就像微型舰队，包含各种股票、债券或商品。这种多样性分散了您的风险，使您不太可能最终被困在投资损失的岛屿上。
轻松航行：买卖 ETF 与交易个股一样简单。它们提供了在整个交易日以市场价格进行交易的灵活性，就像召唤或拒绝友好的精神一样。
期权的宝库：对于勇敢的冒险家来说，使用 ETF 进行期权交易本身就是一个迷宫。虽然它有望带来丰厚的回报，但这是一条充满风险的道路，需要敏锐的眼光和稳定的手。
'''
from flask import Flask,render_template_string 
import yfinance as yf 
import pandas as pd 
import numpy as np 
app = Flask(__name__) 
def fetch_options_data(symbol):
    etf = yf.Ticker(symbol) 
    try:
        expiration_dates = etf.options 
        if expiration_dates:
            first_expiration_date = expiration_dates[0] 
            options_chain = etf.option_chain(first_expiration_date) 
            puts = options_chain.puts 
            calls = options_chain.calls 
            return len(puts),len(calls),first_expiration_date 
    except Exception as e:
        print(f"Could not fetch options data for {symbol}: {e}") 
    return None,None,None 
def format_assets(assets):
    if assets >= 1e9:
        return f"{assets / 1e9:.2f}B" 
    elif assets >= 1e6: 
        return f"{assets / 1e6:.2f}M" 
    return str(assets) 

def fetch_data(symbol):
    etf = yf.Ticker(symbol) 
    info =etf.info #gathering all known wisdom about the ETF. This includes its history, powers, and current standing in the financial kingdoms
    puts_count,calls_count,first_expiration_date = fetch_options_data(symbol) 
    #we delve into the arcane world of options, seeking signs that reveal the market's sentiment towards our ETF. It's akin to reading tea leaves, but for financial wizards

    latest_price = info.get('previousClose', np.nan)  # Corrected key for latest price
    # A snapshot of the ETF's latest closing price, a crucial clue to its current state in the vast marketplace
    
    # Ensure puts_count and calls_count are not None before comparison
    if puts_count is not None and calls_count is not None:
        trend = "Bearish" if puts_count > calls_count else "Bullish" 
    else:
        trend = "Unknown" 
    
    one_year_return = round(info.get('ytdReturn', np.nan) * 100, 2) if info.get('ytdReturn') is not None else "N/A"
    three_year_return = round(info.get('threeYearAverageReturn', np.nan) * 100, 2) if info.get('threeYearAverageReturn') is not None else "N/A"
    five_year_return = round(info.get('fiveYearAverageReturn', np.nan) * 100, 2) if info.get('fiveYearAverageReturn') is not None else "N/A"
    total_assets = format_assets(info.get('totalAssets', np.nan)) if info.get('totalAssets') is not None else "N/A"

    return {
        'Symbol': symbol,
        'Name': info.get('longName', 'N/A'),
        'Latest Price': f"${latest_price}",
        '52W High': f"${round(info.get('fiftyTwoWeekHigh', np.nan), 2)}",
        '52W Low': f"${round(info.get('fiftyTwoWeekLow', np.nan), 2)}",
        '1 Year Return': one_year_return,
        '3 Year Return': three_year_return,
        '5 Year Return': five_year_return,
        'Total Assets': total_assets,
        'Dividend Yield': f"{round(info.get('yield', np.nan) * 100, 2)}%" if info.get('yield') is not None else "N/A",
        'Average Volume': info.get('averageVolume', 'N/A'),
        'Puts Count': puts_count,
        'Calls Count': calls_count,
        'Option Expire': first_expiration_date,
        'Trend': trend
    }
@app.route('/') 
def etf_data():
    with open('etfs.txt','r') as file:
        symbols = file.read().splitlines() 
    data = [fetch_data(symbol) for symbol in symbols] 
    df = pd.DataFrame(data) 
    df_html = df.to_html(classes='table table-striped',index=False) 
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>ETF Data</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
    </head>
    <body>
    <div class="container">
        <h1>ETF Data</h1>
        {{ df_html | safe }}
    </div>
    </body>
    </html>
    """
    return render_template_string(html_template, df_html=df_html)

if __name__ == '__main__':
    app.run(debug=True)