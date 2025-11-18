import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def calculate_b(contact_matrix, suscep='constant', inf='constant', r0=2):
    """
    Returns the B parameter for the model.

    Parameters
    -----------
    contact_matrix : numpy.ndarray
        Contact matrix.
    suscep : str
        Type of susceptibility.
    r0 : float
        Basic reproduction number.

    Returns
    --------
    float
        B parameter.
    """
    C = contact_matrix
    if (suscep == 'linear_increase' or inf == 'linear_increase'):
        sigma = np.linspace(0.5, 2.0, 16)
        sigma = sigma / np.sum(sigma) * 16
        if suscep == 'linear_increase':
            for j in range(16):
                C[:, j] = C[:, j] * sigma[j]
        else:
            for j in range(16):
                C[j, :] = C[j, :] * sigma[j]
    if (suscep == 'linear_decrease' or inf == 'linear_decrease'):
        sigma = np.linspace(2.0, 0.5, 16)
        sigma = sigma / np.sum(sigma) * 16
        if suscep == 'linear_decrease':
            for j in range(16):
                C[:, j] = C[:, j] * sigma[j]
        else:
            for j in range(16):
                C[j, :] = C[j, :] * sigma[j]
    if (suscep == 'u_shaped' or inf == 'u_shaped'):
        sigma = np.array([1.5, 1.2, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8,
                  0.8, 0.8, 0.8, 0.8, 1.0, 1.2, 1.5, 1.8])
        sigma = sigma / np.sum(sigma) * 16
        if suscep == 'u_shaped':
            for j in range(16):
                C[:, j] = C[:, j] * sigma[j]
        else:
            for j in range(16):
                C[j, :] = C[j, :] * sigma[j]
    eigenvalues = np.linalg.eigvals(C)
    b = r0/np.real(np.max(eigenvalues))
    return b

def calculate_contact_matrix(contact_matrix, population):
    """
    Returns the contact matrix C.

    Parameters
    -----------
    contact_matrix : numpy.ndarray
        Contact matrix.
    population : numpy.ndarray
        Population in each age group.

    Returns
    --------
    numpy.ndarray
        Adjusted contact matrix.
    """
    C = contact_matrix
    new_pop = population

    pop_data = pd.read_csv('Data/korea_population.csv', header=0, dtype={'Data': str, 'Population': float})
    N = pop_data[(pop_data['Region'] == 'Nationwide') & (pop_data['Year'] == 2020)]['Population'].values.flatten()
    N = pd.read_csv('Data/popage_total2020.csv', usecols=range(2,23), skiprows=1, header=None, dtype=float).values[96]
    N[-6] = np.sum(N[-6:])
    N = N[:-5]
    N = N*1000

    # Density correction
    C_ = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            C_[i,j] = C[i,j]*(np.sum(N)*new_pop[j])/(np.sum(new_pop)*N[j].item())

    return C_

def system_of_equations(q, beta):
    """
    System of equations for the local outbreak probability.

    Parameters
    -----------
    q : numpy.ndarray
        Probability of local outbreak for each age group.
    beta : numpy.ndarray
        Beta parameter.

    Returns
    --------
    numpy.ndarray
        System of equations.
    """
    n = len(q)  # Number of q_i
    equations = np.zeros(n)

    for i in range(n):
        # Compute the sum in the first term of the equation
        sum_beta_ik = np.sum(beta[i, :])  # sum for k=1 to 16
        first_term = 1 / (1 + sum_beta_ik)

        # Compute the sum in the second term of the equation
        second_term = 0
        for j in range(n):
            second_term += (beta[i, j] / (1 + sum_beta_ik)) * q[i] * q[j]
        
        # Equation for q_i
        equations[i] = q[i] - (first_term + second_term)
    
    return equations

def prob_outbreak(year, suscep='constant', inf='constant', r0=2):
    """
    Returns the probability of a local outbreak given an index case in each age group.
    
    Parameters
    -----------
    year : int
        Year for which the probability of a local outbreak is to be calculated.
    suscep : str
        Profile of susceptibility.
    inf : str
        Profile of infectiousness.
    r0 : float
        Basic reproduction number.

    Returns
    --------
    tuple
        - p : numpy.ndarray
            Probability of local outbreak for each age group.
        - PLO : float
            Average outbreak probability.
        - tc_p : float
            Contact-weighted outbreak probability.
    """
    pop_data = pd.read_csv('Data/korea_population.csv', header=0, dtype={'Data': str, 'Population': float})
    # have baseline as 2020 matrix
    contact_matrix = pd.read_csv('Data/2025_contact_matrix_density.csv', header=None, dtype=float).values
    baseline_contact_matrix = pd.read_csv('Data/contact_matrix_2020.csv', header=None, dtype=float).values
    pop = pop_data[(pop_data['Region'] == 'Nationwide') & (pop_data['Year'] == year)]['Population'].values.flatten()
    pop[-2] = pop[-2] + pop[-1]
    pop = pop[:-1]

    C = calculate_contact_matrix(baseline_contact_matrix, pop)

    if (suscep != 'constant' and inf != 'constant'):
        raise ValueError("Only one of suscep or inf can be non-constant.")

    if (suscep == 'linear_increase' or inf == 'linear_increase'):
        sigma = np.linspace(0.5, 2.0, 16)
        sigma = sigma / np.sum(sigma) * 16
        if suscep == 'linear_increase':
            for j in range(16):
                C[:, j] = C[:, j] * sigma[j]
        else:
            for j in range(16):
                C[j, :] = C[j, :] * sigma[j]
    if (suscep == 'linear_decrease' or inf == 'linear_decrease'):
        sigma = np.linspace(2.0, 0.5, 16)
        sigma = sigma / np.sum(sigma) * 16
        if suscep == 'linear_decrease':
            for j in range(16):
                C[:, j] = C[:, j] * sigma[j]
        else:
            for j in range(16):
                C[j, :] = C[j, :] * sigma[j]
    if (suscep == 'u_shaped' or inf == 'u_shaped'):
        sigma = np.array([1.5, 1.2, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8,
                  0.8, 0.8, 0.8, 0.8, 1.0, 1.2, 1.5, 1.8])
        sigma = sigma / np.sum(sigma) * 16
        if suscep == 'u_shaped':
            for j in range(16):
                C[:, j] = C[:, j] * sigma[j]
        else:
            for j in range(16):
                C[j, :] = C[j, :] * sigma[j]

    N_prop = pop/np.sum(pop)

    # Calculate Beta=B*C*sigma*tau using the reference contact matrix for beta
    Beta = calculate_b(contact_matrix, suscep, r0)*C
    eigenvalues = np.linalg.eigvals(Beta)
    print("Maximum eigenvalue of beta:", np.max(eigenvalues))

    initial_guess = np.ones(16)*0.3

    # Solve the system of equations
    solution = fsolve(system_of_equations, initial_guess, args=(Beta,))
    p = 1 - solution

    # total contacts for age group i
    tc = np.zeros(16)
    for i in range(16):
        tc[i] += C[i, :].sum()

    return p, np.sum(p * N_prop), np.sum(tc*pop*p)/np.sum(tc*pop)

p, PLO, tc_p = prob_outbreak(2000, 'constant', 'constant', 2)

fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.bar(range(1, 17), p, color='lightblue', edgecolor='black', linewidth=1)
plt.axhline(y=PLO, color='black', linewidth=2)
plt.axhline(y=tc_p, color='red', linewidth=2, linestyle='--') # Optional
plt.xlabel('Age group of index case', labelpad=10, fontsize=20)
plt.ylabel(r'Probability of local outbreak ($p_k$)', labelpad=10, fontsize=20)
plt.xticks(np.arange(1, 17), ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                              '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+'],
                              rotation=45)
plt.xlim([0.5, 16.5])
plt.ylim([0, 1])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

# The effect of the susceptibility profile over time (can be changed to consider the infectiousness profile instead)

p, PLO_2000, _ = prob_outbreak(2000, 'constant', 'constant', 2)
p2, PLO2_2000, _ = prob_outbreak(2000, 'linear_increase', 'constant', 2)
p3, PLO3_2000, _ = prob_outbreak(2000, 'linear_decrease', 'constant', 2)
p4, PLO4_2000, _ = prob_outbreak(2000, 'u_shaped', 'constant', 2)
p, PLO_2025, _ = prob_outbreak(2025, 'constant', 'constant', 2)
p2, PLO2_2025, _ = prob_outbreak(2025, 'linear_increase', 'constant', 2)
p3, PLO3_2025, _ = prob_outbreak(2025, 'linear_decrease', 'constant', 2)
p4, PLO4_2025, _ = prob_outbreak(2025, 'u_shaped', 'constant', 2)
p, PLO_2050, _ = prob_outbreak(2050, 'constant', 'constant', 2)
p2, PLO2_2050, _ = prob_outbreak(2050, 'linear_increase', 'constant', 2)
p3, PLO3_2050, _ = prob_outbreak(2050, 'linear_decrease', 'constant', 2)
p4, PLO4_2050, _ = prob_outbreak(2050, 'u_shaped', 'constant', 2)

years = [2000, 2025, 2050]
PLO1 = [PLO_2000, PLO_2025, PLO_2050]
PLO2 = [PLO2_2000, PLO2_2025, PLO2_2050]
PLO3 = [PLO3_2000, PLO3_2025, PLO3_2050]
PLO4 = [PLO4_2000, PLO4_2025, PLO4_2050]

PLO_values = [PLO1, PLO2, PLO3, PLO4]
PLO_labels = ['Constant', 'Linear increase', 'Linear decrease', 'U shaped']
colors = ['#386cb0', '#fdc086', '#beaed4', '#7fc97f']

x = np.arange(len(years))
width = 0.15

fig = plt.gcf()
fig.set_size_inches(8, 6)
for i, (plo, label, color) in enumerate(zip(PLO_values, PLO_labels, colors)):
    plt.bar(x + i*width - 1.5*width, plo, width, label=label, color=color, edgecolor='black', linewidth=1)
plt.xticks(x, years)
plt.xlabel('Year', labelpad=10, fontsize=20)
plt.ylabel(r'Average outbreak probability ($P$)', labelpad=10, fontsize=20)
plt.ylim(0, 1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title='Susceptibility profile', fontsize=14, title_fontsize=16)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
