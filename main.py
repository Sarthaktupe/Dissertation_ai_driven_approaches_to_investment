import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Data Collection, Cleaning & Processing
portfolio_comps = ['AAPL', 'KO', 'AMZN', 'V', 'F', 'MCD', 'WMT', 'ORCL', 'INTC', 'VZ']
portfolio_benchmark = ['^GSPC']

start_date = datetime.datetime(2014, 1, 1)
end_date = datetime.datetime(2024, 3, 1)


# Download historical data for stocks
hist_data = yf.download(portfolio_comps, start=start_date, end=end_date, interval='1mo')
adj_close_df = hist_data['Adj Close']

# Download data for S&P 500
benchmark_hist_data = yf.download(portfolio_benchmark, start=start_date, end=end_date, interval='1mo')
adj_close_sp500 = benchmark_hist_data['Adj Close']

# Drop NaN values
adj_close_df = adj_close_df.dropna()
adj_close_sp500 = adj_close_sp500.dropna()

adj_returns_df = adj_close_df.pct_change().dropna()
adj_returns_sp500 = adj_close_sp500.pct_change().dropna()

#print(adj_returns_df)
# Calculate growth factors
def growth_factor():
    cumulative_growth_factors = pd.DataFrame(index=adj_returns_df.index)

    for stock in portfolio_comps:
        stock_returns = adj_returns_df[stock].values
        cumulative_growth = np.cumprod(1 + stock_returns)
        cumulative_growth_factors[stock] = cumulative_growth
    
    return cumulative_growth_factors

# Buy and Hold Strategy
initial_capital = 1
m = len(portfolio_comps)  # m = 10
theta_initial = 1 / (m + 1)
stock_return = adj_returns_df
portfolio_value = pd.DataFrame()

def buy_and_hold(returns, theta):
    df_returns = returns * theta
    row_sum = df_returns.sum(axis=1) + 1
    log_row_sum = np.log(row_sum)
    cumulative_log_value = log_row_sum.cumsum()
    portfolio_values = cumulative_log_value
    return portfolio_values

def without_rebalancing(theta):
    price_ratios = growth_factor()
    sum_price_ratio = price_ratios.sum(axis=1)
    portfolio_val = np.log(theta*sum_price_ratio)
    
    return portfolio_val
    

def dynamic_rebalancing(returns):
    """
    Compute the portfolio value over time using a strategy with dynamic rebalancing.
    The theta (weights) are updated dynamically where each weight is the price ratio of asset i 
    divided by the sum of price ratios of all assets at that time.

    :param returns: DataFrame where each column represents a stock's returns.
    :return: Series representing the cumulative portfolio value over time.
    """
    # Calculate price ratios (1 + returns), assuming 'returns' are in decimal (e.g., 0.01 for 1%)
    price_ratios = 1 + returns

    # Initialize portfolio value with the initial capital and create a Series to store portfolio values
    portfolio_values = pd.Series(index=returns.index, data=initial_capital)

    # Start with equal weights for all stocks plus cash
    theta = np.full(returns.shape[1], 1/(returns.shape[1]+1))  # Initial weights

    # Iterate over time periods, updating portfolio values and rebalancing theta
    for i in range(1, len(returns)):
        # Update portfolio value for the current period using previous theta
        portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * np.dot(theta, price_ratios.iloc[i-1])

        # Update theta based on the specified rule
        price_ratios_current = price_ratios.iloc[i]
        theta = price_ratios_current / price_ratios_current.sum()

    return np.log(portfolio_values)

def plot_log_market_portfolio(stock_prices):
    """
    Calculate and return the logarithmic value of the market portfolio defined by the ratio
    of the sum of current prices to the sum of initial prices of stocks.

    :param stock_prices: DataFrame where each column represents the price series of a stock over time.
    :return: Pandas Series representing the logarithmic market portfolio value over time.
    """
    # Sum of all stock prices at each time t
    sum_prices_t = stock_prices.sum(axis=1)
    
    # Sum of all initial stock prices
    sum_prices_initial = stock_prices.iloc[0].sum()
    
    # Calculate the market portfolio value Vt
    market_portfolio_value = sum_prices_t / sum_prices_initial
    
    # Calculate the logarithm of the market portfolio value
    market_portfolio_value = np.log(market_portfolio_value)
    
    return market_portfolio_value


#Creating multiple portfolios and storing them in a DataFrame
portfolio_value['buy and hold'] = buy_and_hold(stock_return, theta_initial)    
portfolio_value['buy and hold with uniform weight'] = without_rebalancing(theta_initial)
portfolio_value['Mean-Reverting/ Martingale'] = dynamic_rebalancing(stock_return)
portfolio_value['Market Portfolio'] = plot_log_market_portfolio(adj_close_df)


#To remove first row as it has a nan value for mean_reverting strategy
portfolio_value = portfolio_value.drop(portfolio_value.index[0])
portfolio_value

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 5))

# Plotting 'buy and hold'
plt.plot(portfolio_value['buy and hold'], linestyle='-', color='b', label='Log of Buy and Hold')

# Plotting 'buy and hold with uniform weight'
plt.plot(portfolio_value['buy and hold with uniform weight'], label='Log of Buy and Hold with Uniform Weight', color='r')

# Plotting Market Portfolio
plt.plot(portfolio_value['Market Portfolio'], label='Log of Market Portfolio Value', color='green')

# Plotting Martingale/Mean-Reverting
plt.plot(portfolio_value['Mean-Reverting/ Martingale'], label='Log of Mean reverting Portfolio Value', color='purple')

# Adding titles and labels for the first axis
plt.title('Comparison of Portfolio Strategies Over Time')
plt.xlabel('Date')
plt.ylabel('Log-Portfolio Value')
plt.legend()

# Create a second y-axis
ax2 = plt.gca().twinx()

# Plotting adj_close_sp500 on the second axis
ax2.plot(np.log(adj_close_sp500), label='Log of S&P 500 Adjusted Close', color='black')

# Adding a label for the second y-axis
ax2.set_ylabel('Log of S&P 500 Adjusted Close')

# Adding grid, legend, and showing the plot
plt.grid(True)

# Combine legends from both axes
lines_1, labels_1 = plt.gca().get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

plt.show()

#Function to calculate MDD and ADD
def calculate_drawdown(series, label):
    """
    Calculate and print the maximal drawdown and average drawdown for a given series of portfolio values.

    :param series: Pandas Series containing portfolio values.
    :param label: str, description label for the portfolio.
    """
    #We convert log-values in normal values of portfolio to calculate MDD
    series = np.exp(series)
    cumulative_max = series.cummax()
    drawdowns = (series - cumulative_max) / cumulative_max
    
    # Maximal Drawdown
    max_drawdown = drawdowns.min()
    
    # Average Drawdown, calculated only on negative drawdown periods
    average_drawdown = drawdowns[drawdowns < 0].mean()
    
    # Format and print the results more neatly
    print(f"Results for {label.title()}:")
    print(f"  Maximal Drawdown: {max_drawdown * 100:.2f}%")
    print(f"  Average Drawdown: {average_drawdown * 100:.2f}%\n")

# Calculate and print Drawdown Metrics for each portfolio
calculate_drawdown(portfolio_value['buy and hold'], 'buy and hold')
calculate_drawdown(portfolio_value['buy and hold with uniform weight'], 'buy and hold with uniform weight')
calculate_drawdown(portfolio_value['Market Portfolio'], 'market portfolio')
calculate_drawdown(portfolio_value['Mean-Reverting/ Martingale'], 'Mean reverting/ Martingale portfolio')

#Calculate W_t
def calculate_Wt(Vt, T):
    #Initialize Wt
    Wt = pd.DataFrame(index=Vt.index)
    #Converting back to Portfolio Values
    Vt = np.exp(Vt)
    
    cumsum_port = (Vt.cumsum())/T
    
    #Initialize additional_term DataFrame to store each calculated value
    additional_term = pd.Series(index=Vt.index)
    
    #Calculate the additional term for each row
    for t in range(len(Vt)):
        additional_term.iloc[t] = Vt.iloc[t] * ((T - t) / T)

    #Final log Wt calculation by adding cumsum_port and additional_term
    Wt = cumsum_port + additional_term
    Wt = np.log(Wt)
    
    return Wt

rows, columns = portfolio_value.shape

log_Wt_values= pd.DataFrame(index=portfolio_value.index)
log_Wt_values['buy and hold'] = calculate_Wt(portfolio_value['buy and hold'], rows)
log_Wt_values['buy and hold with uniform weight'] = calculate_Wt(portfolio_value['buy and hold with uniform weight'], rows)
log_Wt_values['Mean-Reverting/ Martingale'] = calculate_Wt(portfolio_value['Mean-Reverting/ Martingale'], rows)
log_Wt_values['Market Portfolio'] = calculate_Wt(portfolio_value['Market Portfolio'], rows)

log_Wt_values

# Plotting the results
plt.figure(figsize=(8, 5)) 

# Plotting Wt for 'buy and hold'
plt.plot(log_Wt_values['buy and hold'], linestyle='-', color='b', label='Log Wt of Buy and Hold')

# Plotting W_t for 'buy and hold with uniform weight'
plt.plot(log_Wt_values['buy and hold with uniform weight'], label='Log Wt of Buy and Hold with Uniform Weight', color='r')

#Plotting W_t for Market Portfolio
plt.plot(log_Wt_values['Market Portfolio'], label='Log Wt of Market Portfolio', color='green')

#Plotting W_t for Martingale/Mean-Reverting
plt.plot(log_Wt_values['Mean-Reverting/ Martingale'], label='Log Wt of Mean-Reverting Portfolio', color='purple')

# Adding titles and labels
plt.title('Comparison of Log Wt for Portfolio Strategies Over Time')
plt.xlabel('Date')
plt.ylabel('Log Wt Value')
plt.grid(True)
plt.legend()
plt.show()

# Calculate and print Drawdown Metrics for each W_t
calculate_drawdown(log_Wt_values['buy and hold'], 'Wt of buy and hold')
calculate_drawdown(log_Wt_values['buy and hold with uniform weight'], 'Wt of buy and hold with uniform weight')
calculate_drawdown(log_Wt_values['Market Portfolio'], 'Wt of market portfolio')
calculate_drawdown(log_Wt_values['Mean-Reverting/ Martingale'], 'Wt of Mean reverting/ Martingale portfolio')

import matplotlib.pyplot as plt

def individual_comparisons(log_Wt_values, portfolio_value, strategies):
    """
    Plots individual comparisons between log Wt values and original portfolio values for each strategy,
    without including 'buy and hold' in every plot.

    Parameters:
    - log_Wt_values: DataFrame containing log Wt values for different strategies.
    - portfolio_value: DataFrame containing original portfolio values for different strategies.
    - strategies: List of tuples with (strategy name, display label).
    """
    for strategy, label in strategies:
        plt.figure(figsize=(8,5))  # Create a new figure for each plot

        # Plot current strategy data for Log Wt values
        plt.plot(log_Wt_values[strategy], linestyle='-', color='purple', label=f'Log Wt of {label}')

        # Plot current strategy data for original portfolio values
        plt.plot(portfolio_value[strategy], linestyle='-', color='blue', label=f'{label}')

        plt.title(f'Comparison of Log Wt vs. Original Portfolio')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.legend()
        plt.show()

strategies = [
    ('buy and hold', 'Buy and Hold'),
    ('buy and hold with uniform weight', 'Buy and Hold with Uniform Weight'),
    ('Market Portfolio', 'Market Portfolio'),
    ('Mean-Reverting/ Martingale', 'Mean-Reverting/Martingale')
]

individual_comparisons(log_Wt_values, portfolio_value, strategies)

returns_copy = adj_returns_df.copy()
T = len(returns_copy)

def ck_allocation(returns, alpha, reg_term=1e-4):
    """
    Calculate the Cover-Kelly (CK) allocation for given returns data.

    Parameters:
    returns (pd.DataFrame): DataFrame of returns for each asset.
    alpha (float): Scaling factor for allocation.
    reg_term (float): Regularization term (default is 1e-4).

    Returns:
    np.array: Allocated weights for each asset.
    """
    # Calculate cumulative returns
    cum_returns = returns.cumsum()

    # Calculate raw covariance matrix with regularization
    raw_covariance = np.dot(cum_returns.T, cum_returns)
    reg_covariance = raw_covariance + np.eye(raw_covariance.shape[0]) * reg_term

    inv_reg_covariance = np.linalg.inv(reg_covariance)
    raw_allocation = alpha * np.dot(inv_reg_covariance, cum_returns.iloc[-1].values)

    # Ensure non-negative weights
    raw_allocation = np.maximum(raw_allocation, 0)

    if raw_allocation.sum() <= 1:
        allocation = raw_allocation
    else:
        # Normalize the allocation to ensure the sum is 1
        allocation = raw_allocation / raw_allocation.sum()

    return allocation

# Alpha values
alpha_values = [1, 0.5, 0.1, 10]
reg_terms = [1e-5]


# Initialize nested dictionary for returns
V = {reg_term: {alpha: np.ones(T) for alpha in alpha_values} for reg_term in reg_terms}

# Initialize a DataFrame for cash allocation
null_index = pd.DataFrame(index=returns_copy.index, columns=returns_copy.columns)
null_index['Cash'] = 0  # Initialize a column for Cash

# Create a DataFrame to store allocations with multi-level columns
allocations_df = pd.DataFrame(
    index=null_index.index, 
    columns=pd.MultiIndex.from_product([alpha_values, null_index.columns], names=['Alpha', 'Asset'])
)

# Iterate over time periods, regularization terms, and alpha values
for t in range(1, T):
    for reg_term in reg_terms:
        for alpha in alpha_values:
            # Calculate allocations for assets excluding 'Cash'
            allocations = ck_allocation(returns_copy.iloc[:t+1], alpha, reg_term)
            
            portfolio_return = (allocations * returns_copy.iloc[t]).sum()
            V[reg_term][alpha][t] = V[reg_term][alpha][t-1] * (1 + portfolio_return)
            
            # Calculate Cash allocation as 1 - sum of allocations
            cash_allocation = 1 - allocations.sum()
            cash_allocation = max(cash_allocation, 0)  # Ensure Cash allocation is non-negative
            
            # Combine the allocations and cash allocation
            full_allocation = np.append(allocations, cash_allocation)
            
            # Save the allocation for each asset at time t
            allocations_df.loc[returns_copy.index[t], (alpha, slice(None))] = full_allocation
            

# Convert the nested dictionary to a DataFrame with multi-level column index
Vt = pd.DataFrame(
    {(reg_term, alpha): V[reg_term][alpha] for reg_term in reg_terms for alpha in alpha_values},
    index=returns_copy.index
)

# Set the column names
Vt.columns.names = ['RegTerm', 'Alpha']

# Drop any NaNs that might appear
allocations_df = allocations_df.dropna()
ck_port_value = Vt.droplevel('RegTerm', axis=1)


# Plotting each combination of reg_term and subcolumns of all alphas
for reg_term in reg_terms:
    plt.figure(figsize=(12, 6))
    for i, alpha in enumerate(alpha_values):
        label = f'Log Portfolio Value ($V_t$) for $\\alpha={alpha}$, $reg\\_term={reg_term}$'
        plt.plot(np.log(Vt[(reg_term, alpha)]), label=label)
    plt.title(f'Log of Portfolio Values ($V_t$) for CK Allocations with $reg\\_term={reg_term}$ Over Time')
    plt.xlabel('Date')
    plt.ylabel('Logarithmic Value of $V_t$')
    plt.legend()
    plt.grid(True)
    plt.show()

# Assuming calculate_drawdown is a predefined function to calculate and plot drawdowns
for reg_term in reg_terms:
    for alpha in alpha_values:
        calculate_drawdown(np.log(Vt[(reg_term, alpha)]), f'Vt when alpha is {alpha} and reg_term is {reg_term}')
        
colors = ['#1f77b4',
          '#ff7f0e', 
          '#2ca02c', 
          '#d62728',
          '#9467bd', 
          '#8c564b', 
          '#e377c2', 
          '#7f7f7f',
          '#bcbd22',
          '#17becf']

# Iterate over each alpha value
for alpha in alpha_values:
    # Step 1: Plot Log of Relative Prices
    log_relative_price = pd.DataFrame(index=adj_close_df.index, columns=adj_close_df.columns)

    def theta_plot(T, price_data):
        for i in range(1, T):
            log_relative_price.iloc[i] = np.log(price_data.iloc[i] / price_data.iloc[0])
        return log_relative_price

    log_relative_price = theta_plot(len(adj_close_df), adj_close_df)
    log_relative_price = log_relative_price.iloc[2:]

    plt.figure(figsize=(10, 6))
    for i, column in enumerate(log_relative_price.columns):
        plt.plot(log_relative_price.index, log_relative_price[column], color=colors[i % len(colors)], label=column)

    plt.xlabel('Date')
    plt.ylabel('Log of Relative Prices $X_t / X_0$')
    plt.title(f'Log of Relative Prices Over Time for Alpha = {alpha}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Step 2: Plot Allocations for the Same Alpha Value
    plt.figure(figsize=(10, 6))
    for i, asset in enumerate(allocations_df.columns.get_level_values(1).unique()):
        plt.plot(allocations_df.index, allocations_df[(alpha, asset)], color=colors[i % len(colors)], label=asset)

    plt.xlabel('Date')
    plt.ylabel('Allocation Weights')
    plt.title(f'Allocations Over Time for Alpha = {alpha}')
    plt.legend()
    plt.grid(True)
    plt.show()


log_Wt_values_ck = pd.DataFrame(index=Vt.index)

for alpha in alpha_values:
    Vt_alpha = Vt.xs(alpha, level='Alpha', axis=1)
    
    log_Wt_values_ck[alpha] = calculate_Wt(np.log(ck_port_value[alpha]), rows)

plt.figure(figsize=(12, 8))

for alpha in alpha_values:
    plt.plot(log_Wt_values_ck.index, log_Wt_values_ck[alpha], label=f'Alpha = {alpha}')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('log(Wt) Value')
plt.title('Time Series of log(Wt) Values for Different Alpha')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

# Define V1 as a constant cash portfolio
V1 = pd.Series(1, index=returns_copy.index)

# V2 is the portfolio value for alpha = 1 from the CK strategy
V2 = ck_port_value[1.0] 

# Ensure V2 and V_aggregated are aligned with the shape of stock_theta
V_aggregated = (V1 + V2) / 2

# Convert the Series to NumPy arrays
V2_array = V2.to_numpy()
V_aggregated_array = V_aggregated.to_numpy()
stock_theta_alpha_1 = stock_theta.xs(1.0, level='Alpha', axis=1).to_numpy()

# Ensure the lengths match by slicing if necessary
V2_array = V2_array[:len(stock_theta_alpha_1)]
V_aggregated_array = V_aggregated_array[:len(stock_theta_alpha_1)]

# Calculate the aggregated theta
theta_aggregated = (stock_theta_alpha_1 * V2_array[:, np.newaxis]) / V_aggregated_array[:, np.newaxis]

# Convert back to DataFrame for easier handling later if needed
theta_aggregated_df = pd.DataFrame(theta_aggregated, index=stock_theta.index, columns=stock_theta.xs(1.0, level='Alpha', axis=1).columns)

# Proceed with the rest of your calculations and plotting
log_V_aggregated = np.log(V_aggregated)
log_values_ck = pd.DataFrame(index=log_Wt_values_ck.index)
log_values_ck['Aggregated Alpha 1'] = log_V_aggregated

plt.figure(figsize=(10, 6))
plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 1'], label='Aggregated Portfolio (Alpha 1 & Cash)', color='orange')
plt.xlabel('Date')
plt.ylabel('Log of Aggregated Portfolio Value')
plt.title('Log of Aggregated Portfolio Value Over Time')
plt.legend()
plt.grid(True)
plt.show()

calculate_drawdown(log_V_aggregated, 'Aggregated Portfolio with Alpha 1 and Cash')

# Function to compute V(alpha) and theta(alpha) for a range of alpha values
def compute_V_theta(returns, alpha_values, reg_term=1e-4):
    T = len(returns)
    m = returns.shape[1]  # Number of assets
    
    # Initialize dictionaries to store V(alpha) and theta(alpha)
    V_alpha = pd.DataFrame(index=returns.index, columns=alpha_values)
    theta_alpha = {alpha: pd.DataFrame(index=returns.index, columns=returns.columns) for alpha in alpha_values}
    
    # Initial portfolio value is 1 for all alphas
    V_alpha.iloc[0] = 1
    
    # Iterate over each time step to calculate V(alpha) and theta(alpha)
    for t in range(1, T):
        for alpha in alpha_values:
            # Compute CK allocation for the current alpha
            allocations = ck_allocation(returns.iloc[:t+1], alpha, reg_term)
            
            # Calculate portfolio return for this time step
            portfolio_return = (allocations * returns.iloc[t]).sum()
            
            # Update the portfolio value V(alpha)
            V_alpha.at[returns.index[t], alpha] = V_alpha.at[returns.index[t-1], alpha] * (1 + portfolio_return)
            
            # Store the allocation strategy theta(alpha)
            theta_alpha[alpha].loc[returns.index[t]] = allocations
    
    return V_alpha, theta_alpha


alpha_values_covers = np.linspace(0, 1, 11)  # 0.0 to 1.0 in increments of 0.1

# Compute V(alpha) and theta(alpha) for each alpha value
V_alpha, theta_alpha = compute_V_theta(returns_copy, alpha_values_covers)

# Calculate Cover's portfolio using V(alpha) values
def compute_covers_portfolio(V_alpha):
    # Integrate over the alpha space to compute Cover's portfolio
    covers_portfolio = V_alpha.mean(axis=1)
    return covers_portfolio

# Calculate Cover's portfolio
covers_portfolio = compute_covers_portfolio(V_alpha)

# Ensure there are no NaN values and only positive values for log calculation
covers_portfolio = covers_portfolio.replace([np.inf, -np.inf], np.nan).dropna()
covers_portfolio = covers_portfolio[covers_portfolio > 0]

# Log transform the Cover's portfolio for comparison
log_covers_portfolio = np.log(covers_portfolio)

# Ensure log_values_ck.index is aligned with covers_portfolio index for plotting
common_index = log_values_ck.index.intersection(log_covers_portfolio.index)

# Plot the Cover's portfolio alongside other portfolios
plt.figure(figsize=(10, 6))
plt.plot(common_index, log_covers_portfolio.loc[common_index], label="Cover's Portfolio", color='blue')
plt.plot(common_index, log_values_ck.loc[common_index, 'Aggregated Alpha 1'], label='Aggregated Portfolio (Alpha 1 & Cash)', color='orange')
plt.xlabel('Date')
plt.ylabel('Log Portfolio Value')
plt.title("Cover's Portfolio vs Aggregated Portfolio")
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print Drawdown Metrics for Cover's portfolio
calculate_drawdown(log_covers_portfolio, "Cover's Portfolio")
plt.figure(figsize=(10, 6))
plt.plot(common_index, log_covers_portfolio.loc[common_index], label="Cover's Portfolio", color='blue')
plt.plot(portfolio_value['Market Portfolio'], label='Log of Market Portfolio Value', color='orange')
plt.xlabel('Date')
plt.ylabel('Log Portfolio Value')
plt.title("Cover's Portfolio vs Market Portfolio")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(common_index, log_covers_portfolio.loc[common_index], label="Cover's Portfolio", color='blue')
plt.plot(portfolio_value['Market Portfolio'], label='Log of Market Portfolio Value', color='orange')
plt.xlabel('Date')
plt.ylabel('Log Portfolio Value')
plt.title("Cover's Portfolio vs Market Portfolio")
plt.legend()
plt.grid(True)
plt.show()

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import tensorflow as tf

# Define the stock symbols for the financial institutions we're interested in
stock_symbols = ['JPM', 'GS', 'MS', 'BLK', 'C']

# Define the symbols for correlated assets
correlated_assets = ['^VIX', '^GSPC']

# Define the start and end dates for our data
start_date = '2020-01-01'
end_date = '2023-01-01'  # Corrected the end date to a valid one

# Function to download stock data
def download_stock_data(symbols):
    data = {}
    for symbol in symbols:
        data[symbol] = yf.download(symbol, start=start_date, end=end_date)
    return data

# Download stock data
stocks_data = download_stock_data(correlated_assets)

# Preprocess the data for all stocks
for symbol in stocks_data:
    # Forward fill any missing values
    stocks_data[symbol].fillna(method='ffill', inplace=True)

    # Backward fill any remaining missing values
    stocks_data[symbol].fillna(method='bfill', inplace=True)

    # Normalize the data using the percentage change of the closing price
    stocks_data[symbol]['Close'] = stocks_data[symbol]['Close'].pct_change().fillna(0)
    
# Calculate technical indicators and OHLC ratios for each stock
def calculate_features(data):
    # Moving Average
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    # Exponential Moving Average
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # OHLC Ratios
    data['HLC_Ratio'] = (data['High'] - data['Low']) / data['Close']
    data['OC_Ratio'] = (data['Open'] - data['Close']) / data['Close']
    
    # Drop NaN values generated by moving averages and RSI
    data.dropna(inplace=True)
    return data

# Apply feature engineering to all stocks
for symbol in stocks_data:
    stocks_data[symbol] = calculate_features(stocks_data[symbol])

# Define the neural network architecture
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Prepare the dataset for training
def prepare_dataset(data, target_horizon=5):
    # Our target variable will be whether the return is positive after 'target_horizon' days
    data['Target'] = (data['Close'].shift(-target_horizon) > 0).astype(int)
    # Drop the last 'target_horizon' rows with NaN target values
    data = data.iloc[:-target_horizon]
    
    # Separate features and target variable
    X = data.drop(columns=['Target'])
    y = data['Target']
    
    # Replace infinities with NaNs, then fill NaNs with zeros
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Prepare the dataset for JPMorgan as an example
X_train_scaled, X_test_scaled, y_train, y_test = prepare_dataset(stocks_data['JPM'])

# Build the model
model = build_model((X_train_scaled.shape[1],))

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {test_accuracy}")

# Predictions
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Plot the training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test_scaled))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Define the symbols for correlated assets
portfolio_comps = ['AAPL', 'KO', 'AMZN', 'V', 'F', 'MCD', 'WMT', 'ORCL', 'INTC', 'VZ']
correlated_assets = ['^VIX', '^GSPC']

# Combine portfolio_comps and correlated_assets into a single list
all_symbols = portfolio_comps + correlated_assets

# Define the start and end dates for our data
start_date = '2014-01-01'
end_date = '2024-08-17'

# Function to download stock data
def download_stock_data(symbols):
    data = {}
    for symbol in symbols:
        data[symbol] = yf.download(symbol, start=start_date, end=end_date)
    return data

# Download stock data for the combined symbols
stocks_data = download_stock_data(all_symbols)

# Preprocess the data for all stocks
for symbol in stocks_data:
    # Forward fill any missing values
    stocks_data[symbol].fillna(method='ffill', inplace=True)

    # Backward fill any remaining missing values
    stocks_data[symbol].fillna(method='bfill', inplace=True)

    # Normalize the data using the percentage change of the closing price
    stocks_data[symbol]['Close'] = stocks_data[symbol]['Close'].pct_change().fillna(0)

# Calculate technical indicators and OHLC ratios for each stock
def calculate_features(data):
    # Exponential Moving Average
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

    # Volatility (e.g., GARCH-derived volatility)
    # This would typically be calculated using a GARCH model, but here we'll use rolling standard deviation as a proxy
    data['Volatility'] = data['Close'].rolling(window=10).std()

    # Momentum Indicators
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Stochastic_K'] = ((data['Close'] - data['Low'].rolling(window=14).min()) /
                            (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())) * 100

    # Volume-based features
    data['Volume'] = data['Volume'].pct_change().fillna(0)
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

    # Lagged Returns
    data['Lagged_Return_1'] = data['Close'].shift(1)
    data['Lagged_Return_3'] = data['Close'].shift(3)
    data['Lagged_Return_5'] = data['Close'].shift(5)
    
    # Drop any NaNs introduced by these new features
    data.dropna(inplace=True)

    return data

# Apply feature engineering to all stocks
for symbol in stocks_data:
    stocks_data[symbol] = calculate_features(stocks_data[symbol])

# Define the neural network architecture for regression
def build_regression_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Prepare the dataset for regression
def prepare_regression_dataset(data, target_horizon=30):
    # Our target variable will be the percentage return after 'target_horizon' days
    data['Target'] = data['Close'].shift(-target_horizon) - data['Close']
    # Drop the last 'target_horizon' rows with NaN target values
    data = data.iloc[:-target_horizon]
    
    # Separate features and target variable
    X = data.drop(columns=['Target'])
    y = data['Target']
    
    # Replace infinities with NaNs, then fill NaNs with zeros
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    # Split the dataset into training and test sets (use TimeSeriesSplit for time-series data)
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = next(tscv.split(X))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Initialize a DataFrame to store the predictions
predictions_df = pd.DataFrame()

# Initialize lists to store training and testing errors
training_errors = []
testing_errors = []

# Loop through each stock in portfolio_comps
for symbol in portfolio_comps:
    # Prepare the dataset
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_regression_dataset(stocks_data[symbol])
    
    # Build the model
    model = build_regression_model((X_train_scaled.shape[1],))
    
    # Train the model and capture training history
    history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate MSE for training and testing sets
    train_mse = np.mean(history.history['loss'])
    test_mse = mean_squared_error(y_test, y_pred)
    
    # Append errors to the lists
    training_errors.append(train_mse)
    testing_errors.append(test_mse)
    
    # Store the predictions in the DataFrame
    predictions_df[symbol] = y_pred.flatten()

# Calculate the average MSE for training and testing
avg_train_mse = np.mean(training_errors)
avg_test_mse = np.mean(testing_errors)

# Print the average MSE
print(f"Average Training MSE: {avg_train_mse}")
print(f"Average Testing MSE: {avg_test_mse}")

# Plot the training and testing errors
plt.figure(figsize=(10, 6))
plt.plot(training_errors, label='Training MSE')
plt.plot(testing_errors, label='Testing MSE')
plt.title('Training and Testing MSE for All Stocks')
plt.xlabel('Stock')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd

# Initialize parameters
capital = 1.0
interval = 30  # days
num_intervals = len(predictions_df) // interval
portfolio_values = []

# Iterate over each interval
for i in range(num_intervals):
    # Get the predictions for the current interval
    interval_predictions = predictions_df.iloc[i*interval:(i+1)*interval]

    # Calculate the expected return for each stock in the next interval
    expected_returns = interval_predictions.mean()

    # Assign weights based on expected returns
    positive_returns = expected_returns[expected_returns > 0]
    total_positive = positive_returns.sum()

    if total_positive > 0:
        weights = positive_returns / total_positive
        weights *= (1 / weights.sum())
    else:
        weights = pd.Series(0, index=expected_returns.index)  # All weights are 0 if no positive expected returns

    # Store the current portfolio value
    portfolio_values.append(capital)

    # Calculate portfolio return based on the actual returns in the next interval
    next_interval_returns = predictions_df.iloc[(i+1)*interval:(i+2)*interval].mean()

    # Calculate the portfolio return
    portfolio_return = (weights * next_interval_returns).sum()

    # Update capital
    capital *= (1 + portfolio_return)

# Create a DataFrame to store portfolio values over time
portfolio_df = pd.DataFrame({
    'Date': predictions_df.index[::interval][:num_intervals],
    'Portfolio Value': portfolio_values
})

# Set Date as the index
portfolio_df.set_index('Date', inplace=True)

# Plot the portfolio value over time
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(np.log(portfolio_df['Portfolio Value']), label='Log Portfolio Value')
plt.title('Log Portfolio Value Over Time with ML predictions')
plt.xlabel('Time')
plt.ylabel('Log Portfolio Value')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define V1 as a constant cash portfolio
V1 = pd.Series(1, index=returns_copy.index)

# V2 is the portfolio value for alpha = 1 from the CK strategy
V2 = ck_port_value[10]

# Aggregate V1 and V2 into a combined portfolio
V_aggregated = (V1 + V2) / 2

# Convert the Series to NumPy arrays for easier manipulation
V2_array = V2.to_numpy()
V_aggregated_array = V_aggregated.to_numpy()
stock_theta_alpha_1 = stock_theta.xs(1.0, level='Alpha', axis=1).to_numpy()

# Ensure the lengths match by slicing if necessary
V2_array = V2_array[:len(stock_theta_alpha_1)]
V_aggregated_array = V_aggregated_array[:len(stock_theta_alpha_1)]
V_aggregated = V_aggregated_array[1:]

# Calculate the aggregated theta values
theta_aggregated = (stock_theta_alpha_1 * V2_array[:, np.newaxis]) / V_aggregated_array[:, np.newaxis]

# Ensure non-negative weights
theta_aggregated = np.maximum(theta_aggregated, 0)

# Check for NaN or None values
if pd.isnull(theta_aggregated).any():  # pd.isnull() works with both NaN and None
    print("Error: theta_aggregated contains NaN or None values.")
else:
    # Normalize the allocation to ensure the sum is 1
    allocation_sums = theta_aggregated.sum(axis=1)

    # Only divide where allocation_sums > 0
    theta_aggregated = np.divide(theta_aggregated.T, allocation_sums, where=allocation_sums > 0).T

    theta_aggregated = theta_aggregated[1:]
    print(theta_aggregated)
    
    # Add cash allocation as the remaining portion
    cash_allocation = 1 - theta_aggregated.sum(axis=1)
    
    # Also remove the first index entry from stock_theta.index to match the shape
    adjusted_index = stock_theta.index[1:]

    # Now convert to DataFrame with the adjusted index
    theta_aggregated_df = pd.DataFrame(theta_aggregated, index=adjusted_index, columns=stock_theta.xs(1.0, level='Alpha', axis=1).columns)

    # Add the cash allocation to the DataFrame
    theta_aggregated_df['Cash'] = cash_allocation

    # Plot the log of the aggregated portfolio value
    log_V_aggregated = V_aggregated

    # Ensure consistent lengths between log_V_aggregated and log_Wt_values_ck.index
    min_length = min(len(log_V_aggregated), len(log_Wt_values_ck.index))
    log_V_aggregated = log_V_aggregated[:min_length]
    log_Wt_values_ck = log_Wt_values_ck.iloc[:min_length]

    # Ensure log_values_ck is initialized as a DataFrame if it doesn't exist yet
    if 'log_values_ck' not in globals() or not isinstance(log_values_ck, pd.DataFrame):
        log_values_ck = pd.DataFrame(index=log_Wt_values_ck.index)  # Initialize it with proper index

    # Ensure log_V_aggregated is a pandas Series and its index matches log_values_ck's index
    log_V_aggregated = pd.Series(log_V_aggregated, index=log_Wt_values_ck.index)

    # Now create the new column 'Aggregated Alpha 10' in log_values_ck
    log_values_ck['Aggregated Alpha 10'] = log_V_aggregated

    # Normalize the 'Aggregated Alpha 10' by dividing by the first value
    log_values_ck['Aggregated Alpha 10'] = log_values_ck['Aggregated Alpha 10'] / log_values_ck['Aggregated Alpha 10'].iloc[0]

    # Take the logarithm of the normalized values
    log_values_ck['Aggregated Alpha 10'] = np.log(log_values_ck['Aggregated Alpha 10'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 10'], label='Aggregated Portfolio (Alpha 10 & Cash)', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Log of Aggregated Portfolio Value')
    plt.title('Log of Aggregated Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('log_aggregated_port_val')
    plt.show()
    
    log_V_aggregated_series = pd.Series(log_V_aggregated, index=log_Wt_values_ck.index)
    
    # Calculate and display drawdown for the aggregated portfolio
    calculate_drawdown(log_V_aggregated_series, 'Aggregated Portfolio with Alpha 10 and Cash')

    # Plotting the aggregated theta values
    plt.figure(figsize=(12, 8))

    for i, asset in enumerate(theta_aggregated_df.columns):
        plt.plot(theta_aggregated_df.index, theta_aggregated_df[asset], label=asset)

    plt.xlabel('Date')
    plt.ylabel('Allocation Weights')
    plt.title('Aggregated Theta (Allocation Weights) for Alpha 10 and Cash Portfolio Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('aggregated_port+ck10_vs_cash')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 10'], label='Aggregated Portfolio (Alpha 10 & Cash)', color='blue')
    plt.plot(portfolio_value['Market Portfolio'], label='Log of Market Portfolio Value', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Log Portfolio Value')
    plt.title("Aggregated Portfolio vs Market Portfolio")
    plt.legend()
    plt.grid(True)
    plt.savefig('aggregateck10_vs_market')
    plt.show()

# Ensure that all data in theta_aggregated_df is numeric
theta_aggregated_df = theta_aggregated_df.apply(pd.to_numeric, errors='coerce')

# Optionally, fill NaN values (if any exist) with 0 or a small value
theta_aggregated_df = theta_aggregated_df.fillna(0)

# --- Heatmap of Allocations Over Time ---
plt.figure(figsize=(12, 8))
sns.heatmap(theta_aggregated_df.T, cmap='YlGnBu', annot=False, cbar=True, linewidths=0.5)

plt.title(f'Heatmap of Allocations Over Time for Alpha 10 & Cash')
plt.xlabel('Date')
plt.ylabel('Assets')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Adjust the number of x-axis ticks for better readability
plt.tight_layout()
plt.savefig('heatmap_allocations_aggregated10.png')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define V1 as a constant cash portfolio
V1_cash = pd.Series(1, index=returns_copy.index)

# V2 is the portfolio value for alpha = 1 from the CK strategy
V2_alpha_1 = ck_port_value[1]

# Aggregate V1 and V2 into a combined portfolio
V_aggregated_1 = (V1_cash + V2_alpha_1) / 2

# Convert the Series to NumPy arrays for easier manipulation
V2_array_1 = V2_alpha_1.to_numpy()
V_aggregated_array_1 = V_aggregated_1.to_numpy()
stock_theta_alpha_1 = stock_theta.xs(1.0, level='Alpha', axis=1).to_numpy()

# Ensure the lengths match by slicing if necessary
V2_array_1 = V2_array_1[:len(stock_theta_alpha_1)]
V_aggregated_array_1 = V_aggregated_array_1[:len(stock_theta_alpha_1)]
V_aggregated_1 = V_aggregated_array_1[1:]

# Calculate the aggregated theta values
theta_for_alpha = (stock_theta_alpha_1 * V2_array_1[:, np.newaxis]) / V_aggregated_array_1[:, np.newaxis]
theta_aggregated_1 = theta_for_alpha
# Ensure non-negative weights
theta_aggregated_1 = np.maximum(theta_aggregated_1, 0)

# Check for NaN or None values
if pd.isnull(theta_aggregated_1).any():  # pd.isnull() works with both NaN and None
    print("Error: theta_aggregated_1 contains NaN or None values.")
else:
    # Normalize the allocation to ensure the sum is 1
    allocation_sums_1 = theta_aggregated_1.sum(axis=1)

    # Only divide where allocation_sums > 0
    theta_aggregated_1 = np.divide(theta_aggregated_1.T, allocation_sums_1, where=allocation_sums_1 > 0).T

    theta_aggregated_1 = theta_aggregated_1[1:]
    print(theta_aggregated_1)
    # Add cash allocation as the remaining portion
    cash_allocation_1 = 1 - theta_aggregated_1.sum(axis=1)
    
    # Also remove the first index entry from stock_theta.index to match the shape
    adjusted_index_1 = stock_theta.index[1:]

    # Now convert to DataFrame with the adjusted index
    theta_aggregated_df_1 = pd.DataFrame(theta_aggregated_1, index=adjusted_index_1, columns=stock_theta.xs(1.0, level='Alpha', axis=1).columns)

    # Add the cash allocation to the DataFrame
    theta_aggregated_df_1['Cash'] = cash_allocation_1

    # Plot the log of the aggregated portfolio value
    log_V_aggregated_1 = V_aggregated_1
        # Ensure log_values_ck is initialized as a DataFrame if it doesn't exist yet
    if 'log_values_ck' not in globals() or not isinstance(log_values_ck, pd.DataFrame):
        log_values_ck = pd.DataFrame(index=log_Wt_values_ck.index)  # Initialize it with proper index

    # Ensure log_V_aggregated_1 is a pandas Series and its index matches log_values_ck's index
    log_V_aggregated_1 = pd.Series(log_V_aggregated_1, index=log_Wt_values_ck.index)

    # Now create the new column 'Aggregated Alpha 1' in log_values_ck
    log_values_ck['Aggregated Alpha 1'] = log_V_aggregated_1

    # Normalize the 'Aggregated Alpha 1' by dividing by the first value
    log_values_ck['Aggregated Alpha 1'] = log_values_ck['Aggregated Alpha 1'] / log_values_ck['Aggregated Alpha 1'].iloc[0]

    # Take the logarithm of the normalized values
    log_values_ck['Aggregated Alpha 1'] = np.log(log_values_ck['Aggregated Alpha 1'])
 # Check the first few rows to see if the column is created correctly

    
    plt.figure(figsize=(8, 6))
    plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 1'], label='Aggregated Portfolio (Alpha 1 & Cash)', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Log of Aggregated Portfolio Value')
    plt.title('Log of Aggregated Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('log_aggregated_port_val_alpha_1')
    plt.show()
    
    log_V_aggregated_series_1 = pd.Series(log_V_aggregated_1, index=log_Wt_values_ck.index)
    
    # Calculate and display drawdown for the aggregated portfolio
    calculate_drawdown(log_V_aggregated_series_1, 'Aggregated Portfolio with Alpha 1 and Cash')

    # Plotting the aggregated theta values
    plt.figure(figsize=(12, 8))

    for i, asset in enumerate(theta_aggregated_df_1.columns):
        plt.plot(theta_aggregated_df_1.index, theta_aggregated_df_1[asset], label=asset)

    plt.xlabel('Date')
    plt.ylabel('Allocation Weights')
    plt.title('Aggregated Theta (Allocation Weights) for Alpha 1 and Cash Portfolio Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('aggregated_port_ck1with')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 1'], label='Aggregated Portfolio (Alpha 1 & Cash)', color='blue')
    plt.plot(portfolio_value['Market Portfolio'], label='Log of Market Portfolio Value', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Log Portfolio Value')
    plt.title("Aggregated Portfolio vs Market Portfolio (Alpha 1)")
    plt.legend()
    plt.grid(True)
    plt.savefig('aggregateck1_vs_market')
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define V1 as a constant cash portfolio
V1 = pd.Series(1, index=returns_copy.index)

# V2 is the portfolio value for alpha = 1 from the CK strategy
V2 = ck_port_value[10]

# Aggregate V1 and V2 into a combined portfolio
V_aggregated = (V1 + V2) / 2

# Convert the Series to NumPy arrays for easier manipulation
V2_array = V2.to_numpy()
V_aggregated_array = V_aggregated.to_numpy()
stock_theta_alpha_1 = stock_theta.xs(1.0, level='Alpha', axis=1).to_numpy()

# Ensure the lengths match by slicing if necessary
V2_array = V2_array[:len(stock_theta_alpha_1)]
V_aggregated_array = V_aggregated_array[:len(stock_theta_alpha_1)]
V_aggregated = V_aggregated_array[1:]

# Calculate the aggregated theta values
theta_aggregated = (stock_theta_alpha_1 * V2_array[:, np.newaxis]) / V_aggregated_array[:, np.newaxis]

# Ensure non-negative weights
theta_aggregated = np.maximum(theta_aggregated, 0)

# Check for NaN or None values
if pd.isnull(theta_aggregated).any():  # pd.isnull() works with both NaN and None
    print("Error: theta_aggregated contains NaN or None values.")
else:
    # Normalize the allocation to ensure the sum is 1
    allocation_sums = theta_aggregated.sum(axis=1)

    # Only divide where allocation_sums > 0
    theta_aggregated = np.divide(theta_aggregated.T, allocation_sums, where=allocation_sums > 0).T

    theta_aggregated = theta_aggregated[1:]
    print(theta_aggregated)
    
    # Add cash allocation as the remaining portion
    cash_allocation = 1 - theta_aggregated.sum(axis=1)
    
    # Also remove the first index entry from stock_theta.index to match the shape
    adjusted_index = stock_theta.index[1:]

    # Now convert to DataFrame with the adjusted index
    theta_aggregated_df = pd.DataFrame(theta_aggregated, index=adjusted_index, columns=stock_theta.xs(1.0, level='Alpha', axis=1).columns)

    # Add the cash allocation to the DataFrame
    theta_aggregated_df['Cash'] = cash_allocation

    # Plot the log of the aggregated portfolio value
    log_V_aggregated = V_aggregated

    # Ensure consistent lengths between log_V_aggregated and log_Wt_values_ck.index
    min_length = min(len(log_V_aggregated), len(log_Wt_values_ck.index))
    log_V_aggregated = log_V_aggregated[:min_length]
    log_Wt_values_ck = log_Wt_values_ck.iloc[:min_length]

    # Ensure log_values_ck is initialized as a DataFrame if it doesn't exist yet
    if 'log_values_ck' not in globals() or not isinstance(log_values_ck, pd.DataFrame):
        log_values_ck = pd.DataFrame(index=log_Wt_values_ck.index)  # Initialize it with proper index

    # Ensure log_V_aggregated is a pandas Series and its index matches log_values_ck's index
    log_V_aggregated = pd.Series(log_V_aggregated, index=log_Wt_values_ck.index)

    # Now create the new column 'Aggregated Alpha 10' in log_values_ck
    log_values_ck['Aggregated Alpha 10'] = log_V_aggregated

    # Normalize the 'Aggregated Alpha 10' by dividing by the first value
    log_values_ck['Aggregated Alpha 10'] = log_values_ck['Aggregated Alpha 10'] / log_values_ck['Aggregated Alpha 10'].iloc[0]

    # Take the logarithm of the normalized values
    log_values_ck['Aggregated Alpha 10'] = np.log(log_values_ck['Aggregated Alpha 10'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 10'], label='Aggregated Portfolio (Alpha 10 & Cash)', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Log of Aggregated Portfolio Value')
    plt.title('Log of Aggregated Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('log_aggregated_port_val')
    plt.show()
    
    log_V_aggregated_series = pd.Series(log_V_aggregated, index=log_Wt_values_ck.index)
    
    # Calculate and display drawdown for the aggregated portfolio
    calculate_drawdown(log_V_aggregated_series, 'Aggregated Portfolio with Alpha 10 and Cash')

    # Plotting the aggregated theta values
    plt.figure(figsize=(12, 8))

    for i, asset in enumerate(theta_aggregated_df.columns):
        plt.plot(theta_aggregated_df.index, theta_aggregated_df[asset], label=asset)

    plt.xlabel('Date')
    plt.ylabel('Allocation Weights')
    plt.title('Aggregated Theta (Allocation Weights) for Alpha 10 and Cash Portfolio Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('aggregated_port+ck10_vs_cash')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 10'], label='Aggregated Portfolio (Alpha 10 & Cash)', color='blue')
    plt.plot(portfolio_value['Market Portfolio'], label='Log of Market Portfolio Value', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Log Portfolio Value')
    plt.title("Aggregated Portfolio vs Market Portfolio")
    plt.legend()
    plt.grid(True)
    plt.savefig('aggregateck10_vs_market')
    plt.show()

# Ensure that all data in theta_aggregated_df is numeric
theta_aggregated_df = theta_aggregated_df.apply(pd.to_numeric, errors='coerce')

# Optionally, fill NaN values (if any exist) with 0 or a small value
theta_aggregated_df = theta_aggregated_df.fillna(0)

# --- Heatmap of Allocations Over Time ---
plt.figure(figsize=(12, 8))
sns.heatmap(theta_aggregated_df.T, cmap='YlGnBu', annot=False, cbar=True, linewidths=0.5)

plt.title(f'Heatmap of Allocations Over Time for Alpha 10 & Cash')
plt.xlabel('Date')
plt.ylabel('Assets')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Adjust the number of x-axis ticks for better readability
plt.tight_layout()
plt.savefig('heatmap_allocations_aggregated10.png')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define V1 as a constant cash portfolio
V1_cash = pd.Series(1, index=returns_copy.index)

# V2 is the portfolio value for alpha = 1 from the CK strategy
V2_alpha_1 = ck_port_value[1]

# Aggregate V1 and V2 into a combined portfolio
V_aggregated_1 = (V1_cash + V2_alpha_1) / 2

# Convert the Series to NumPy arrays for easier manipulation
V2_array_1 = V2_alpha_1.to_numpy()
V_aggregated_array_1 = V_aggregated_1.to_numpy()
stock_theta_alpha_1 = stock_theta.xs(1.0, level='Alpha', axis=1).to_numpy()

# Ensure the lengths match by slicing if necessary
V2_array_1 = V2_array_1[:len(stock_theta_alpha_1)]
V_aggregated_array_1 = V_aggregated_array_1[:len(stock_theta_alpha_1)]
V_aggregated_1 = V_aggregated_array_1[1:]

# Calculate the aggregated theta values
theta_for_alpha = (stock_theta_alpha_1 * V2_array_1[:, np.newaxis]) / V_aggregated_array_1[:, np.newaxis]
theta_aggregated_1 = theta_for_alpha
# Ensure non-negative weights
theta_aggregated_1 = np.maximum(theta_aggregated_1, 0)

# Check for NaN or None values
if pd.isnull(theta_aggregated_1).any():  # pd.isnull() works with both NaN and None
    print("Error: theta_aggregated_1 contains NaN or None values.")
else:
    # Normalize the allocation to ensure the sum is 1
    allocation_sums_1 = theta_aggregated_1.sum(axis=1)

    # Only divide where allocation_sums > 0
    theta_aggregated_1 = np.divide(theta_aggregated_1.T, allocation_sums_1, where=allocation_sums_1 > 0).T

    theta_aggregated_1 = theta_aggregated_1[1:]
    print(theta_aggregated_1)
    # Add cash allocation as the remaining portion
    cash_allocation_1 = 1 - theta_aggregated_1.sum(axis=1)
    
    # Also remove the first index entry from stock_theta.index to match the shape
    adjusted_index_1 = stock_theta.index[1:]

    # Now convert to DataFrame with the adjusted index
    theta_aggregated_df_1 = pd.DataFrame(theta_aggregated_1, index=adjusted_index_1, columns=stock_theta.xs(1.0, level='Alpha', axis=1).columns)

    # Add the cash allocation to the DataFrame
    theta_aggregated_df_1['Cash'] = cash_allocation_1

    # Plot the log of the aggregated portfolio value
    log_V_aggregated_1 = V_aggregated_1
        # Ensure log_values_ck is initialized as a DataFrame if it doesn't exist yet
    if 'log_values_ck' not in globals() or not isinstance(log_values_ck, pd.DataFrame):
        log_values_ck = pd.DataFrame(index=log_Wt_values_ck.index)  # Initialize it with proper index

    # Ensure log_V_aggregated_1 is a pandas Series and its index matches log_values_ck's index
    log_V_aggregated_1 = pd.Series(log_V_aggregated_1, index=log_Wt_values_ck.index)

    # Now create the new column 'Aggregated Alpha 1' in log_values_ck
    log_values_ck['Aggregated Alpha 1'] = log_V_aggregated_1

    # Normalize the 'Aggregated Alpha 1' by dividing by the first value
    log_values_ck['Aggregated Alpha 1'] = log_values_ck['Aggregated Alpha 1'] / log_values_ck['Aggregated Alpha 1'].iloc[0]

    # Take the logarithm of the normalized values
    log_values_ck['Aggregated Alpha 1'] = np.log(log_values_ck['Aggregated Alpha 1'])
 # Check the first few rows to see if the column is created correctly

    
    plt.figure(figsize=(8, 6))
    plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 1'], label='Aggregated Portfolio (Alpha 1 & Cash)', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Log of Aggregated Portfolio Value')
    plt.title('Log of Aggregated Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('log_aggregated_port_val_alpha_1')
    plt.show()
    
    log_V_aggregated_series_1 = pd.Series(log_V_aggregated_1, index=log_Wt_values_ck.index)
    
    # Calculate and display drawdown for the aggregated portfolio
    calculate_drawdown(log_V_aggregated_series_1, 'Aggregated Portfolio with Alpha 1 and Cash')

    # Plotting the aggregated theta values
    plt.figure(figsize=(12, 8))

    for i, asset in enumerate(theta_aggregated_df_1.columns):
        plt.plot(theta_aggregated_df_1.index, theta_aggregated_df_1[asset], label=asset)

    plt.xlabel('Date')
    plt.ylabel('Allocation Weights')
    plt.title('Aggregated Theta (Allocation Weights) for Alpha 1 and Cash Portfolio Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('aggregated_port_ck1with')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 1'], label='Aggregated Portfolio (Alpha 1 & Cash)', color='blue')
    plt.plot(portfolio_value['Market Portfolio'], label='Log of Market Portfolio Value', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Log Portfolio Value')
    plt.title("Aggregated Portfolio vs Market Portfolio (Alpha 1)")
    plt.legend()
    plt.grid(True)
    plt.savefig('aggregateck1_vs_market')
    plt.show()

# Ensure that all data in theta_aggregated_df is numeric
theta_aggregated_df_1 = theta_aggregated_df_1.apply(pd.to_numeric, errors='coerce')

# Optionally, fill NaN values (if any exist) with 0 or a small value
theta_aggregated_df_1 = theta_aggregated_df_1.fillna(0)

# --- Heatmap of Allocations Over Time ---
plt.figure(figsize=(12, 8))
sns.heatmap(theta_aggregated_df_1.T, cmap='YlGnBu', annot=False, cbar=True, linewidths=0.5)

plt.title(f'Heatmap of Allocations Over Time for Alpha 1 & Cash')
plt.xlabel('Date')
plt.ylabel('Assets')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Adjust the number of x-axis ticks for better readability
plt.tight_layout()
plt.savefig('heatmap_allocations_aggregated1.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 10'], label='Aggregated Portfolio (Alpha 10 & Cash)', color='blue')
plt.plot(log_Wt_values_ck.index, log_values_ck['Aggregated Alpha 1'], label='Aggregated Portfolio (Alpha 1 & Cash)', color='purple')
plt.plot(portfolio_value['Market Portfolio'], label='Log of Market Portfolio Value', color='orange')
plt.xlabel('Date')
plt.ylabel('Log Portfolio Value')
plt.title("Aggregated Portfolio Comparison ")
plt.legend()
plt.grid(True)
plt.savefig('aggregateck10_vs_market1')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_and_plot_covariance(adj_close_df):

    # Step 1: Calculate the returns (percentage change) based on adjusted close prices
    returns_df = adj_close_df.pct_change().dropna()

    # Step 2: Calculate the covariance matrix
    correlation_matrix = returns_df.corr()

    # Step 3: Plot the covariance matrix
    plt.figure(figsize=(10, 8))
    
    # Create a heatmap using seaborn for better visualization
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, 
                cbar_kws={'label': 'Covariance'}, annot_kws={"size": 10})

    # Title and labels
    plt.title('Correlation Matrix of Portfolio', fontsize=16, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    

    # Show the plot
    plt.tight_layout()
    plt.savefig('correlation_matrix')
    plt.show()
    
    return correlation_matrix

calculate_and_plot_covariance(adj_close_df_cov_copy)
# Function to calculate and print skewness and kurtosis, and plot histograms for each portfolio
def plot_skewness_kurtosis(log_Wt_values):
    for portfolio_name in log_Wt_values:
        # Calculate daily returns (difference in log values)
        returns = log_Wt_values[portfolio_name].diff().dropna()
        
        # Calculate skewness and kurtosis
        skewness_value = skew(returns)
        kurtosis_value = kurtosis(returns)
        
        # Print the skewness and kurtosis
        print(f"{portfolio_name} - Skewness: {skewness_value:.4f}, Kurtosis: {kurtosis_value:.4f}")
        
        # Create a new figure for the histogram
        plt.figure(figsize=(5, 4))
        
        # Plot the return distribution with histogram and KDE
        sns.histplot(returns, bins=50, kde=True, color='blue', stat='density', alpha=0.6)
        
        # Add skewness and kurtosis text to the plot
        plt.text(0.05, 0.95, f"Skewness: {skewness_value:.2f}\nKurtosis: {kurtosis_value:.2f}", 
                 fontsize=8,transform=plt.gca().transAxes, verticalalignment='top')
        
        # Set plot title and labels
        plt.title(f'Return Distribution of log Wt {portfolio_name} Portfolio', fontsize=10)
        plt.xlabel('Log Returns', fontsize=10)
        plt.ylabel('Density', fontsize=10)
        
        # Save the plot
        plt.savefig(f'Log_Wt_{portfolio_name}_return_distribution.png')
        plt.show()

# Call the function to calculate skewness, kurtosis, and plot histograms for each portfolio
plot_skewness_kurtosis(log_Wt_values)

import numpy as np
import matplotlib.pyplot as plt

# Data for Strategy 1
strategy_1 = {
    'Buy And Hold': {'MDD': -22.08, 'ADD': -4.85},
    'Buy And Hold With Rebalancing': {'MDD': -24.16, 'ADD': -5.36},
    'Market Portfolio': {'MDD': -21.20, 'ADD': -4.67},
    'Dynamically Rebalanced': {'MDD': -20.88, 'ADD': -4.63},
    'Aggregated Alpha 10': {'MDD': -6.42, 'ADD': -1.64},
    'Covers Portfolio': {'MDD': -11.65, 'ADD': -3.06},
    'CK with Alpha = 1': {'MDD': -7.80, 'ADD':  -2.44},
    'CK with Alpha = 10': {'MDD': -12.43, 'ADD': -3.23}
}

# Data for Strategy 2
strategy_2 = {
    'Buy And Hold': {'MDD': -8.07, 'ADD': -2.30},
    'Buy And Hold With Rebalancing': {'MDD': -8.99, 'ADD': -2.61},
    'Market Portfolio': {'MDD': -8.77, 'ADD': -2.12},
    'Dynamically Rebalanced': {'MDD': -8.47, 'ADD': -2.46},
    'Aggregated Alpha 10': {'MDD': -2.62, 'ADD': -0.59},
    'Covers Portfolio': {'MDD': -7.58, 'ADD': -1.72},
    'CK with Alpha = 1': {'MDD': -5.11, 'ADD': -1.45},
    'CK with Alpha = 10': {'MDD': -8.15, 'ADD': -1.91}
}


# Extract data for plotting
portfolios = list(strategy_1.keys())
mdd_1 = np.array([strategy_1[portfolio]['MDD'] for portfolio in portfolios])
add_1 = np.array([strategy_1[portfolio]['ADD'] for portfolio in portfolios])
mdd_2 = np.array([strategy_2[portfolio]['MDD'] for portfolio in portfolios])
add_2 = np.array([strategy_2[portfolio]['ADD'] for portfolio in portfolios])

# Set up the plot with two subplots: one for MDD, one for ADD
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MDD Plot (Left subplot)
axes[0].plot(portfolios, mdd_1, marker='o', label='Original Portfolio\'s', color='blue', linestyle='--')
axes[0].plot(portfolios, mdd_2, marker='o', label='PoP strategy', color='green', linestyle='-')
axes[0].set_title('Maximal Drawdown (MDD)', fontsize=14)
axes[0].set_ylabel('MDD (%)', fontsize=12)
axes[0].set_xlabel('Portfolios', fontsize=12)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].set_xticklabels(portfolios, rotation=45, ha='right', fontsize=11)

# ADD Plot (Right subplot)
axes[1].plot(portfolios, add_1, marker='o', label='Original Portfolio\'s', color='red', linestyle='--')
axes[1].plot(portfolios, add_2, marker='o', label='PoP strategy', color='orange', linestyle='-')
axes[1].set_title('Average Drawdown (ADD)', fontsize=14)
axes[1].set_ylabel('ADD (%)', fontsize=12)
axes[1].set_xlabel('Portfolios', fontsize=12)
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].set_xticklabels(portfolios, rotation=45, ha='right', fontsize=11)

# Adjust layout for clarity
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Data for Strategy 1
strategy_1 = {
    'Buy And Hold': {'MDD': -22.08, 'ADD': -4.85},
    'Buy And Hold With Rebalancing': {'MDD': -24.16, 'ADD': -5.36},
    'Market Portfolio': {'MDD': -21.20, 'ADD': -4.67},
    'Dynamically Rebalanced': {'MDD': -20.88, 'ADD': -4.63},
    'Aggregated Alpha 10': {'MDD': -6.42, 'ADD': -1.64},
    'Covers Portfolio': {'MDD': -11.65, 'ADD': -3.06},
    'CK with Alpha = 1': {'MDD': -7.80, 'ADD': -2.44},
    'CK with Alpha = 10': {'MDD': -12.43, 'ADD': -3.23}
}

# Data for Strategy 2
strategy_2 = {
    'Buy And Hold': {'MDD': -8.07, 'ADD': -2.30},
    'Buy And Hold With Rebalancing': {'MDD': -8.99, 'ADD': -2.61},
    'Market Portfolio': {'MDD': -8.77, 'ADD': -2.12},
    'Dynamically Rebalanced': {'MDD': -8.47, 'ADD': -2.46},
    'Aggregated Alpha 10': {'MDD': -2.62, 'ADD': -0.59},
    'Covers Portfolio': {'MDD': -7.58, 'ADD': -1.72},
    'CK with Alpha = 1': {'MDD': -5.11, 'ADD': -1.45},
    'CK with Alpha = 10': {'MDD': -8.15, 'ADD': -1.91}
}

# Extract data for plotting
portfolios = list(strategy_1.keys())
mdd_1 = np.array([strategy_1[portfolio]['MDD'] for portfolio in portfolios])
add_1 = np.array([strategy_1[portfolio]['ADD'] for portfolio in portfolios])
mdd_2 = np.array([strategy_2[portfolio]['MDD'] for portfolio in portfolios])
add_2 = np.array([strategy_2[portfolio]['ADD'] for portfolio in portfolios])

# Prepare data for box plots
mdd_data = [mdd_1, mdd_2]  # MDD for Strategy 1 and 2
add_data = [add_1, add_2]  # ADD for Strategy 1 and 2

# Calculate the mean values
mdd_means = [np.mean(mdd_1), np.mean(mdd_2)]
add_means = [np.mean(add_1), np.mean(add_2)]

# Set up the plot with two subplots: one for MDD, one for ADD
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# MDD Box Plot (Left subplot)
axes[0].boxplot(mdd_data, patch_artist=True, labels=["Original Portfolio's", 'PoP strategy'], 
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='blue'))
axes[0].set_title('Maximal Drawdown (MDD)', fontsize=14)
axes[0].set_ylabel('MDD (%)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.5)

# ADD Box Plot (Right subplot)
axes[1].boxplot(add_data, patch_artist=True, labels=["Original Portfolio's", 'PoP strategy'], 
                boxprops=dict(facecolor='lightcoral'), medianprops=dict(color='red'))
axes[1].set_title('Average Drawdown (ADD)', fontsize=14)
axes[1].set_ylabel('ADD (%)', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.5)

# Create a box with the mean values for MDD and ADD
mean_text_mdd = f"Mean MDD\nOriginal: {mdd_means[0]:.2f}%\nPoP: {mdd_means[1]:.2f}%"
mean_text_add = f"Mean ADD\nOriginal: {add_means[0]:.2f}%\nPoP: {add_means[1]:.2f}%"

# Add the mean value box for MDD (bottom right of first subplot)
axes[0].text(0.95, 0.05, mean_text_mdd, transform=axes[0].transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# Add the mean value box for ADD (bottom right of second subplot)
axes[1].text(0.95, 0.05, mean_text_add, transform=axes[1].transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# Adjust layout for clarity
plt.tight_layout()
plt.savefig('box_plot_with_mean_in_box.png')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV files
csv_df = pd.read_csv('portfolio_values_feedforward.csv')
log_portfolio_df_csv = pd.read_csv('portfolio_values_lstm.csv')

# Assuming 'Portfolio Value_Feedforward' is the column in csv_df that contains the feedforward data
csv_df['Feedforward'] = csv_df['Portfolio Value_Feedforward']

# Assuming 'Portfolio Value' is the LSTM output in the second CSV
log_portfolio_df_csv['LSTM'] = log_portfolio_df_csv['Portfolio Value']

# If there's no 'Date' column, assume index is used
csv_df.set_index(csv_df.index, inplace=True)
log_portfolio_df_csv.set_index(log_portfolio_df_csv.index, inplace=True)

# Ensure that both DataFrames have the same length and index
if len(csv_df) != len(log_portfolio_df_csv):
    print("Warning: Feedforward and LSTM datasets have different lengths!")
else:
    # Calculate the average difference between the Feedforward and LSTM values
    avg_diff = np.mean(np.log(csv_df['Feedforward']) - log_portfolio_df_csv['LSTM'])

    # Plot the 'Feedforward' column and the LSTM predictions
    plt.figure(figsize=(10, 6))

    # Plot Feedforward values
    plt.plot(csv_df.index, np.log(csv_df['Feedforward']), label='Feedforward Value', color='blue', linestyle='--')

    # Plot LSTM values
    plt.plot(csv_df.index, log_portfolio_df_csv['LSTM'], label='LSTM Value', color='red', linestyle='-')

    # Highlight the area between the two curves
    plt.fill_between(csv_df.index, np.log(csv_df['Feedforward']), log_portfolio_df_csv['LSTM'], 
                     where=(np.log(csv_df['Feedforward']) > log_portfolio_df_csv['LSTM']), 
                     facecolor='green', interpolate=True, alpha=0.3)

    plt.fill_between(csv_df.index, np.log(csv_df['Feedforward']), log_portfolio_df_csv['LSTM'], 
                     where=(np.log(csv_df['Feedforward']) < log_portfolio_df_csv['LSTM']), 
                     facecolor='green', interpolate=True, alpha=0.3, label='Feedforward < LSTM')

    # Add the average difference in value in a box at the bottom right
    textstr = f'Avg Diff: {avg_diff:.4f}'
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Add titles and labels
    plt.title('Feedforward vs LSTM Predictions Over Time', fontsize=16, weight='bold')
    plt.xlabel('Rebalancing period', fontsize=14)
    plt.ylabel('Log Portfolio Value', fontsize=14)

    # Add grid, legend, and display
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('LSTM_Feedforward_comparison')
    plt.show()

    import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Assuming 'Feedforward' and 'LSTM' columns represent portfolio values, calculate returns
csv_df['Feedforward_Returns'] = csv_df['Feedforward'].pct_change()

# Handle NaN or infinite values in returns
csv_df['Feedforward_Returns'].replace([np.inf, -np.inf], np.nan, inplace=True)
csv_df['Feedforward_Returns'].dropna(inplace=True)

log_portfolio_df_csv['LSTM_Returns'] = log_portfolio_df_csv['LSTM'].pct_change()

# Handle NaN or infinite values in returns
log_portfolio_df_csv['LSTM_Returns'].replace([np.inf, -np.inf], np.nan, inplace=True)
log_portfolio_df_csv['LSTM_Returns'].dropna(inplace=True)

# Calculate skewness and kurtosis for Feedforward and LSTM returns
feedforward_skew = skew(csv_df['Feedforward_Returns'])
lstm_skew = skew(log_portfolio_df_csv['LSTM_Returns'])

feedforward_kurtosis = kurtosis(csv_df['Feedforward_Returns'])
lstm_kurtosis = kurtosis(log_portfolio_df_csv['LSTM_Returns'])

# Plot the return distributions
plt.figure(figsize=(6, 5))

# Use Seaborn to plot Kernel Density Estimate (KDE) and histograms
sns.histplot(csv_df['Feedforward_Returns'], bins=50, kde=True, color='blue', label='Feedforward Returns', stat="density")
sns.histplot(log_portfolio_df_csv['LSTM_Returns'], bins=50, kde=True, color='blue', label='LSTM Returns', stat="density")

# Add titles and labels
plt.title('Return Distribution for LSTM and Feedforward Model', fontsize=10)
plt.xlabel('Returns', fontsize=14)
plt.ylabel('Density', fontsize=14)

# Add a legend
plt.legend(loc='upper right')

# Show the grid and plot
plt.tight_layout()
plt.savefig('Return_distribution_LSTMandFeedforward')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use a more sophisticated color palette from seaborn
palette = sns.color_palette("husl", len(log_Wt_values))  # 'husl' provides good distinction between colors

# Initialize the plot with a larger figure size for better clarity
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot Feedforward values and LSTM on the primary x-axis (ax1)
ax1.plot(csv_df.index, np.log(csv_df['Feedforward']), label='Feedforward Value', color='blue', linestyle='--', linewidth=2)
ax1.plot(csv_df.index, log_portfolio_df_csv['LSTM'], label='LSTM Value', color='green', linestyle='--', linewidth=2)

# Customize primary x-axis for Feedforward and LSTM
ax1.set_xlabel('Neural Network', fontsize=14, weight='bold')
ax1.set_ylabel('Log Portfolio Value', fontsize=14, weight='bold', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle='--', alpha=0.5)

# Create a twin x-axis (ax2) for the portfolio values (log_Wt_values)
ax2 = ax1.twiny()
# Plot log_Wt_values on the secondary x-axis (ax2)
for i, portfolio_name in enumerate(log_Wt_values):
    if 'Covers Portfolio' in portfolio_name:
        ax2.plot(log_Wt_values[portfolio_name].index, log_Wt_values[portfolio_name], label=portfolio_name, 
                 color='red', linewidth=3, linestyle='-', alpha=0.8)
        
    else:
        ax2.plot(log_Wt_values[portfolio_name].index, log_Wt_values[portfolio_name], label=portfolio_name, 
                 color=palette[i], linewidth=2, linestyle='-', alpha=0.7)
        

# Customize secondary x-axis for log_Wt_values
ax2.set_xlabel('Traditional Portfolio (log_Wt_values)', fontsize=14, weight='bold')
ax2.set_ylabel('Log Portfolio Value (Other Portfolios)', fontsize=14, weight='bold', color='black')

# Add a combined title for both plots
plt.title("Log Portfolio Values Over Time (Neural network vs Other Portfolios)", fontsize=16, weight='bold')

# Add a legend for Feedforward and LSTM (on ax1) outside the plot
feedforward_lstm_legend = ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)

# Add a second legend for the other portfolios (on ax2) outside the plot
portfolio_legend = ax2.legend(loc='lower left', bbox_to_anchor=(1.02, 0), fontsize=12)

# Ensure that the layout is tight and the legends do not overlap
plt.tight_layout()
plt.savefig('Comparison_neural_network_vs_others')
# Display the plot
plt.show()