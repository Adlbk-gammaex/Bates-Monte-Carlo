import numpy as np

def simulate_bates(S0, T, r, V0, kappa, theta, xi, rho, lmbda, mu_j, v_j, steps=100, paths=10000):
    dt = T / steps
    S = np.zeros((steps + 1, paths))
    V = np.zeros((steps + 1, paths))
    
    S[0] = S0
    V[0] = V0
    
    for t in range(1, steps + 1):
        Z_S = np.random.normal(0, 1, paths)
        Z_V = np.random.normal(0, 1, paths)
        
        Z_V = rho * Z_S + np.sqrt(1 - rho**2) * Z_V
        V_prev = np.maximum(V[t-1], 0)
        
        V[t] = V_prev + kappa * (theta - V_prev) * dt + xi * np.sqrt(V_prev * dt) * Z_V
        V[t] = np.maximum(V[t], 0)
        
        N = np.random.poisson(lmbda * dt, paths)
        J = np.exp(np.random.normal(mu_j, v_j, paths)) - 1
        
        k_bar = np.exp(mu_j + 0.5 * v_j**2) - 1
        drift = (r - lmbda * k_bar) * dt
        diffusion = np.sqrt(V_prev * dt) * Z_S
        
        S[t] = S[t-1] * np.exp(drift + diffusion) * (1 + N * J)
        
    return S
