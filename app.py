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
                    st.markdown(f"<div class='metric-box'><b>Risk Minimum (Panic)</b><br><h2>${lower_bound:.2f}</h2></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box' style='background-color: #e3f2fd;'><b>Mathematical Average</b><br><h2>${average_price:.2f}</h2></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='metric-box' style='background-color: #e8f5e9;'><b>Uptrend Maximum</b><br><h2>${upper_bound:.2f}</h2></div>", unsafe_allow_html=True)
                
                st.write("")
                st.info(f"Investor Analytics: In a positive market scenario based on Geometric Brownian Motion, {ticker} stock has a mathematical growth potential of up to {growth_potential:.1f}% from its current price over the next 3 months.")
                
                st.write("")
                st.subheader("Visualization of parallel universes Monte Carlo:")
                
                # Создаем красивый яркий график
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Используем яркую палитру 'plasma' для генерации неоновых цветов линий
                colors = plt.cm.plasma(np.linspace(0.1, 0.9, 150))
                
                # Рисуем первые 150 траекторий ярче и сочнее
                for i in range(150):
                    ax.plot(S[:, i], color=colors[i], alpha=0.5, linewidth=1.2)
                
                # Выделяем ключевые уровни жирными контрастными линиями
                ax.axhline(average_price, color='#007bff', linestyle='--', linewidth=2.5, label=f'Mathematical Average: ${average_price:.2f}')
                ax.axhline(upper_bound, color='#28a745', linestyle='-.', linewidth=2.5, label=f'Uptrend Target (95%): ${upper_bound:.2f}')
                ax.axhline(S0, color='#dc3545', linestyle='-', linewidth=2, label=f'Current Price: ${S0:.2f}')
                
                # Кастомизация внешнего вида (стильная темноватая сетка)
                ax.set_title(f"Bates Model Simulation Trajectories for {ticker}", fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel("Modeling Days (steps)", fontsize=11, labelpad=10)
                ax.set_ylabel("Stock Price ($)", fontsize=11, labelpad=10)
                
                ax.grid(True, linestyle=':', alpha=0.6, color='gray')
                ax.set_facecolor('#fafafa') # Легкий приятный фон для контраста ярких линий
                fig.patch.set_facecolor('#ffffff')
                
                ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
                
                st.pyplot(fig)
                st.caption("The forecast is a mathematical model of risks and does not guarantee the result on the exchange.")
                
        except Exception as e:
            st.error(f"An error occurred while processing the ticker {ticker}: {e}")
