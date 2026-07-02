import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from bates_model import simulate_bates

st.set_page_config(page_title="Quantum Valuation of Stocks by Adilbek Mukhambetov", layout="wide")

# Настраиваем стили для карточек: темный фон, белый крупный текст
st.markdown("""
    <style>
    .metric-box-risk { background-color: #2c3e50; color: #ffffff; padding: 20px; border-radius: 12px; text-align: center; }
    .metric-box-avg { background-color: #1a5276; color: #ffffff; padding: 20px; border-radius: 12px; text-align: center; }
    .metric-box-max { background-color: #196f3d; color: #ffffff; padding: 20px; border-radius: 12px; text-align: center; }
    .analytics-text { font-size: 16pt; font-weight: bold; color: #1b2631; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

st.title("Smart stock simulator based on geometric Brownian motion using Bates & Monte Carlo methodology")
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
                
                lower_bound = np.percentile(final_prices, 5.0)
                upper_bound = np.percentile(final_prices, 95.0)
                
                realistic_prices = final_prices[(final_prices >= lower_bound) & (final_prices <= upper_bound)]
                average_price = np.mean(realistic_prices)
                
                growth_potential = ((upper_bound - S0) / S0) * 100
                
                st.subheader("Trajectory Analysis and 3-Month Price Potential:")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='metric-box-risk'><span style='font-size: 13pt; font-weight: bold;'>Risk Minimum (Panic)</span><br><span style='font-size: 26pt; font-weight: 800;'>${lower_bound:.2f}</span></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box-avg'><span style='font-size: 13pt; font-weight: bold;'>Mathematical Average</span><br><span style='font-size: 26pt; font-weight: 800;'>${average_price:.2f}</span></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='metric-box-max'><span style='font-size: 13pt; font-weight: bold;'>Uptrend Maximum</span><br><span style='font-size: 26pt; font-weight: 800;'>${upper_bound:.2f}</span></div>", unsafe_allow_html=True)
                
                st.write("")
                st.info(f"Investor Analytics: In a positive market scenario based on Geometric Brownian Motion, {ticker} stock has a mathematical growth potential of up to {growth_potential:.1f}% from its current price over the next 3 months.")
                
                st.write("")
                st.subheader("Visualization of parallel universes Monte Carlo:")
                
                fig, ax = plt.subplots(figsize=(12, 6.5))
                
                # Используем спектральную палитру 'jet' для максимального разнообразия цветов линий
                colors = plt.cm.jet(np.linspace(0.0, 1.0, 150))
                
                # Отрисовка 150 разноцветных ярких траекторий Броуновского движения
                for i in range(150):
                    ax.plot(S[:, i], color=colors[i], alpha=0.55, linewidth=1.3)
                
                # Наносим три главные жирные контрольные линии
                ax.axhline(average_price, color='#00bcff', linestyle='--', linewidth=3, label=f'Mathematical Average: ${average_price:.2f}')
                ax.axhline(upper_bound, color='#2ecc71', linestyle='-.', linewidth=3, label=f'Uptrend Target (95%): ${upper_bound:.2f}')
                ax.axhline(S0, color='#e74c3c', linestyle='-', linewidth=2.5, label=f'Current Price: ${S0:.2f}')
                
                # Стилизация осей и сетки
                ax.set_title(f"Bates Model Simulation Trajectories for {ticker}", fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel("Modeling Days (steps)", fontsize=11, labelpad=10)
                ax.set_ylabel("Stock Price ($)", fontsize=11, labelpad=10)
                
                ax.grid(True, linestyle=':', alpha=0.6, color='gray')
                ax.set_facecolor('#fafafa')
                fig.patch.set_facecolor('#ffffff')
                
                ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
                
                st.pyplot(fig)
                st.caption("The forecast is a mathematical model of risks and does not guarantee the result on the exchange.")
                
        except Exception as e:
            st.error(f"An error occurred while processing the ticker {ticker}: {e}")
