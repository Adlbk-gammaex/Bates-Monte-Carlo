import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from bates_model import simulate_bates

st.set_page_config(page_title="Quantum Valuation of Stocks by Adilbek Mukhambetov", layout="wide")

st.markdown("""
    <style>
    .metric-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("Smart stock simulator based on geometric Brownian motion using Bates & Monte Carlo methodology by Adilbek Mukhambetov")
st.write("Enter the ticker symbol of a company (for example: NVDA, AAPL, TSLA). The system will automatically download the data and calculate 10,000 future possible scenarios.")

ticker = st.text_input("Enter company ticker symbol (US):", value="NVDA").upper().strip()

if st.button("Calculate future options"):
    with st.spinner(f"Connecting to the exchange, calibrating the Bates model for {ticker}..."):
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period="1y")
            
            if history.empty:
                st.error("Error: No company with this ticker could be found. Please check the correctness (Latin characters only, for example: TSLA).")
            else:
                S0 = history['Close'].iloc[-1]
                
                returns = history['Close'].pct_change().dropna()
                hist_vol = np.std(returns) * np.sqrt(252)
                
                V0 = hist_vol**2
                theta = V0
                
                jumps = returns[np.abs(returns) > 3 * np.std(returns)]
                lmbda = max(len(jumps), 1)
                
                st.success(f"Data received! Current price {ticker}: ${S0:.2f}. Historical volatility: {hist_vol*100:.1f}%")
                
                S = simulate_bates(S0=S0, T=0.25, r=0.05, V0=V0, kappa=3.0, theta=theta, 
                                   xi=2.0, rho=-0.7, lmbda=lmbda, mu_j=-0.12, v_j=0.04, paths=10000)
                
                final_prices = S[-1]
                
                lower_bound = np.percentile(final_prices, 2.5)
                upper_bound = np.percentile(final_prices, 97.5)
                
                realistic_prices = final_prices[(final_prices >= lower_bound) & (final_prices <= upper_bound)]
                average_price = np.mean(realistic_prices)
                
                st.subheader("Realistic forecast of the price corridor for 3 months ahead (95% confidence interval):")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='metric-box'>📉 <b>Pessimistic outcome (2.5%)</b><br><h2>${lower_bound:.2f}</h2></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box' style='background-color: #e3f2fd;'>⭐ <b>Average Possible Price</b><br><h2>${average_price:.2f}</h2></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='metric-box'>📈 <b>Optimistic outcome (97.5%)</b><br><h2>${upper_bound:.2f}</h2></div>", unsafe_allow_html=True)
                
                st.write("")
                st.subheader("📊 Visualization of parallel universes Monte Carlo:")
                
                fig, ax = plt.subplots(figsize=(12, 5.5))
                ax.plot(S[:, :150], alpha=0.35, linewidth=1) 
                
                ax.axhline(average_price, color='red', linestyle='--', linewidth=2.5, label=f'Average Possible: ${average_price:.2f}')
                ax.axhline(S0, color='black', linestyle=':', linewidth=2, label=f'Starting Price: ${S0:.2f}')
                
                ax.set_title(f"Bates model simulation trajectories for {ticker}", fontsize=14, fontweight='bold')
                ax.set_xlabel("Modeling Days (steps)", fontsize=11)
                ax.set_ylabel("Stock Price ($)", fontsize=11)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='upper left', fontsize=10)
                
                st.pyplot(fig)
                st.caption("The forecast is a mathematical model of risks and does not guarantee the result on the exchange.")
                
        except Exception as e:
            st.error(f"An error occurred while processing the ticker {ticker}: {e}")
