# Adaptive-Indian-Equity-Allocation-System (AIEAS)

## 1. Project Title & Overview
**Adaptive-Indian-Equity-Allocation-System (AIEAS)** is an advanced, algorithmically driven quantitative allocation engine designed for retail investors. The system's primary objective is to dynamically profile risk and route capital across equity portfolios. By leveraging time-series volatility modeling and convex optimization techniques, AIEAS maximizes risk-adjusted returns while adhering to user-defined capital influx strategies and minimizing tax drag.

## 2. System Architecture
The AIEAS architecture is fundamentally bipartite, separating computationally intensive quantitative modeling from the user interface and routing logic:
*   **Layer 1 (ML/Quant Core)**: This layer serves as the computational backend. It ingests historical asset data, forecasts volatility utilizing **GJR-GARCH(1,1)** models, constructs dynamic covariance matrices, and employs `scipy.optimize` to solve for the portfolio weights that maximize the **Sharpe Ratio**.
*   **Layer 2 (Streamlit Router)**: This layer functions as the frontend dashboard and capital routing mechanism. Styled akin to a Bloomberg Terminal, it ingests existing retail equity portfolios via CSV or Excel uploads. It then applies the target weights derived from Layer 1 to route fresh capital using either an **SIP-First** or **Tactical** rebalancing strategy, subject to defined **Asset-Level Tolerance (Deadband)** constraints.

## 3. Installation & Setup Guide

### Option 1: For Users Without Git (Download ZIP)
1. Go to this project's GitHub repository page in your web browser.
2. Click the green **Code** button near the top right.
3. Select **Download ZIP** from the dropdown menu.
4. Extract the downloaded `.zip` file to a folder on your computer.
5. Open your terminal (Command Prompt/PowerShell on Windows, Terminal on macOS/Linux).
6. Navigate into the extracted folder using the `cd` command (e.g., `cd path/to/extracted-folder`).
7. Skip the `git clone` step and proceed directly to creating a virtual environment in your OS-specific instructions below.

### Option 2: For Git Users (Clone Repository)

#### Windows (PowerShell / Command Prompt)
```powershell
# Clone the repository
git clone https://github.com/name-shourya/Adaptive-Indian-Equity-Allocation-System.git
cd Adaptive-Indian-Equity-Allocation-System

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### macOS (Terminal)
```bash
# Clone the repository
git clone https://github.com/name-shourya/Adaptive-Indian-Equity-Allocation-System.git
cd Adaptive-Indian-Equity-Allocation-System

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Linux (Bash)
```bash
# Clone the repository
git clone https://github.com/name-shourya/Adaptive-Indian-Equity-Allocation-System.git
cd Adaptive-Indian-Equity-Allocation-System

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 4. Execution Instructions
To initialize the Layer 2 Streamlit routing dashboard, execute the following command in your terminal from the project root directory (ensure your virtual environment is active):

```bash
python -m streamlit run wrapper/app.py
```

This will launch a local server, and the dashboard will typically be accessible via a web browser at `http://localhost:8501`.

### Sample Portfolio Format
The system requires an initial portfolio state to compute rebalancing routes. The input `sample_portfolio.csv` should adhere to the following structure:

```csv
Ticker,Shares,Current_Price
AAPL,50,150.25
MSFT,30,310.50
GOOGL,15,130.00
```

## 5. Glossary of Quantitative Terminology

*   **GJR-GARCH(1,1)**: A statistical model (Glosten-Jagannathan-Runkle Generalized Autoregressive Conditional Heteroskedasticity) used to estimate the volatility of financial returns. It specifically models *asymmetric volatility* or the *leverage effect*, capturing the empirical observation that negative asset returns (bad news) tend to increase future volatility more significantly than positive returns (good news) of the same magnitude.
*   **Sharpe Ratio**: A metric used to evaluate the return of an investment compared to its risk. It represents the *risk-adjusted return*, calculated as the excess return of the portfolio (over the risk-free rate) divided by the standard deviation of the portfolio's excess return. The Layer 1 optimization objective is to find the asset weights that maximize this ratio.
*   **Dynamic Covariance**: The measure of how two assets move in relation to each other, allowed to vary over time. In AIEAS, the standard historical correlation matrix is combined with the forward-looking **GJR-GARCH(1,1)** volatility forecasts to construct a dynamic covariance matrix, providing a more responsive and accurate representation of current market risk regimes.
*   **SIP-First Rebalancing**: A capital routing strategy designed to minimize tax drag. Instead of selling over-weighted assets (which triggers capital gains taxes), fresh capital (Systematic Investment Plan influx) is proportionally directed strictly into under-weighted assets to iteratively pull the portfolio towards its target allocation. It represents a *zero-sell* methodology.
*   **Tactical Rebalancing**: An alternative capital routing strategy focused on precise threshold-based drift correction. It involves actively buying and selling assets to rapidly realign the portfolio with the optimal target weights computed by Layer 1, utilized when the portfolio composition deviates significantly from the target.
*   **Portfolio Drift**: The percentage difference between an asset's current holding weight in the portfolio and its algorithmically computed optimal target weight. Monitoring drift prevents unnecessary trading when deviations are minor and flags when a portfolio requires active tactical rebalancing to stay aligned with its risk profile.
*   **Asset-Level Tolerance (Deadband)**: A defined percentage band around an asset's target weight. If an asset's current weight falls within this deadband, no trades are executed for that specific asset. This mechanism prevents *micro-trade churn* (frequent, small, costly transactions) by allowing minor weight deviations induced by normal market noise.
