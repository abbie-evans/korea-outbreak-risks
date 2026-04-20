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
    inf : str
        Type of infectiousness.
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
    print(b)
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
    System of equations for the major outbreak probability.

    Parameters
    -----------
    q : numpy.ndarray
        Probability of major outbreak for each age group.
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
    Returns the probability of a major outbreak given an index case in each age group.
    
    Parameters
    -----------
    year : int
        Year for which the probability of a major outbreak is to be calculated.
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
            Probability of major outbreak for each age group.
        - PLO : float
            Population average outbreak probability.
    """
    pop_data = pd.read_csv('Data/korea_population.csv', header=0, dtype={'Data': str, 'Population': float})
    contact_matrix = pd.read_csv('Data/2025_contact_matrix_density.csv', header=None, dtype=float).values
    # have baseline as 2020 matrix
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
    Beta = calculate_b(contact_matrix, suscep, inf, r0)*C

    initial_guess = np.ones(16)*0.3

    # Solve the system of equations
    solution = fsolve(system_of_equations, initial_guess, args=(Beta,))
    p = 1 - solution

    return p, np.sum(p * N_prop)

# Figure 3A

sigma1 = np.ones(16)
sigma = np.linspace(0.5, 2.0, 16)
sigma2 = sigma / np.sum(sigma) * 16
sigma = np.linspace(2.0, 0.5, 16)
sigma3 = sigma / np.sum(sigma) * 16
sigma = np.array([1.5, 1.2, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8,
                  0.8, 0.8, 0.8, 0.8, 1.0, 1.2, 1.5, 1.8])
sigma4 = sigma / np.sum(sigma) * 16

plt.figure(figsize=(8, 6))
plt.plot(sigma1, label='Uniform', linewidth=2, color='#386cb0')
plt.plot(sigma2, label='Increasing', linewidth=2, color='#fdc086')
plt.plot(sigma3, label='Decreasing', linewidth=2, color='#beaed4')
plt.plot(sigma4, label='U-Shaped', linewidth=2, color='#7fc97f')
plt.xticks(np.arange(0, 16), ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                              '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+'],
                              rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r'Age group', labelpad=10, fontsize=24)
plt.ylabel('Susceptibility /\nInfectiousness', labelpad=10, fontsize=24)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('Figures/susc_inf_lines.svg', bbox_inches='tight')
plt.show()

# The effect of the susceptibility profile over time (can be changed to consider the infectiousness profile instead)

p, PLO_2000 = prob_outbreak(2000, 'constant', 'constant', 2)
p2, PLO2_2000 = prob_outbreak(2000, 'constant', 'linear_increase', 2)
p3, PLO3_2000 = prob_outbreak(2000, 'constant', 'linear_decrease', 2)
p4, PLO4_2000 = prob_outbreak(2000, 'constant', 'u_shaped', 2)

p, PLO_2025 = prob_outbreak(2025, 'constant', 'constant', 2)
p2, PLO2_2025 = prob_outbreak(2025, 'constant', 'linear_increase', 2)
p3, PLO3_2025 = prob_outbreak(2025, 'constant', 'linear_decrease', 2)
p4, PLO4_2025 = prob_outbreak(2025, 'constant', 'u_shaped', 2)

p, PLO_2050 = prob_outbreak(2050, 'constant', 'constant', 2)
p2, PLO2_2050 = prob_outbreak(2050, 'constant', 'linear_increase', 2)
p3, PLO3_2050 = prob_outbreak(2050, 'constant', 'linear_decrease', 2)
p4, PLO4_2050 = prob_outbreak(2050, 'constant', 'u_shaped', 2)

years = [2000, 2025, 2050]
PLO1 = [PLO_2000, PLO_2025, PLO_2050]
PLO2 = [PLO2_2000, PLO2_2025, PLO2_2050]
PLO3 = [PLO3_2000, PLO3_2025, PLO3_2050]
PLO4 = [PLO4_2000, PLO4_2025, PLO4_2050]

# Figure 3B-C

PLO_values = [PLO1, PLO2, PLO3, PLO4]
PLO_labels = ['A (Constant)', 'B (Linear increase)', 'C (Linear decrease)', 'D (U-shaped)']
colors = ['#386cb0', '#fdc086', '#beaed4', '#7fc97f']

x = np.arange(len(years))
width = 0.15

fig = plt.gcf()
fig.set_size_inches(8, 6)
for i, (plo, label, color) in enumerate(zip(PLO_values, PLO_labels, colors)):
    plt.bar(x + i*width - 1.5*width, plo, width, label=label, color=color, edgecolor='black', linewidth=1)
plt.xticks(x, years)
plt.xlabel('Year', labelpad=10, fontsize=24)
plt.ylabel(r'Average outbreak probability', labelpad=10, fontsize=24)
plt.ylim(0, 1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(title='Infectiousness profile', fontsize=18, title_fontsize=20)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('Figures/infectiousness_profile_effect.svg', bbox_inches='tight')
plt.show()

# Figure S2

p, PLO_2050 = prob_outbreak(2000, 'constant', 'constant', 2)
p2, PLO2_2050 = prob_outbreak(2000, 'constant', 'linear_increase', 2)
p3, PLO3_2050 = prob_outbreak(2000, 'constant', 'linear_decrease', 2)
p4, PLO4_2050 = prob_outbreak(2000, 'constant', 'u_shaped', 2)

fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.plot(p, label='A (Constant)', linewidth=2, color='#386cb0')
plt.plot(p2, label='B (Linear increase)', linewidth=2, color='#fdc086')
plt.plot(p3, label='C (Linear decrease)', linewidth=2, color='#beaed4')
plt.plot(p4, label='D (U-shaped)', linewidth=2, color='#7fc97f')
plt.xticks(np.arange(0, 16), ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                              '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+'],
                              rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Age group of index case', labelpad=10, fontsize=24)
plt.ylabel('Major outbreak probability', labelpad=10, fontsize=24)
plt.legend(title='Infectiousness profile', fontsize=18, title_fontsize=20)
plt.ylim(0, 1)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('Figures/infectiousness_lines_2000.svg', bbox_inches='tight')
plt.show()
