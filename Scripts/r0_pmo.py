import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def calculate_b(contact_matrix, r0=2):
    """
    Returns the B parameter for the model.

    Parameters
    -----------
    contact_matrix : numpy.ndarray
        Contact matrix.
    r0 : float
        Basic reproduction number.

    Returns
    --------
    float
        B parameter.
    """
    C = contact_matrix
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

def system_of_equations(q, beta, k=1):
    """
    System of equations for the major outbreak probability.

    Parameters
    -----------
    q : numpy.ndarray
        Probability of major outbreak for each age group.
    beta : numpy.ndarray
        Beta parameter.
    k : float
        Dispersion parameter.

    Returns
    --------
    numpy.ndarray
        System of equations.
    """
    n = len(q)
    equations = np.zeros(n)

    for i in range(n):
        first_term = 0.0
        for j in range(n):
            first_term += beta[i, j] * (1 - q[j])

        equations[i] = q[i] - (1 + first_term / k) ** (-k)

    return equations

def prob_outbreak(year, r0=2, k=1):
    """
    Returns the probability of a major outbreak given an index case in each age group.
    
    Parameters
    -----------
    year : int
        Year for which the probability of a major outbreak is to be calculated.
    r0 : float
        Basic reproduction number.
    k : float
        Dispersion parameter.

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

    N_prop = pop/np.sum(pop)

    # Calculate Beta=B*C using the reference contact matrix for beta
    Beta = calculate_b(contact_matrix, r0)*C

    initial_guess = np.ones(16)*0.3

    # Solve the system of equations
    solution = fsolve(system_of_equations, initial_guess, args=(Beta,k))
    p = 1 - solution

    return p, np.sum(p * N_prop)

# For a range of r0 values, plot the average outbreak probability
fig = plt.gcf()
fig.set_size_inches(8, 6)
P_2025 = []
P_2000 = []
P_2050 = []
for r0 in np.arange(1, 6, 0.01):
    p, PLO = prob_outbreak(2025, r0, k=1)
    P_2025.append(PLO)
    p, PLO = prob_outbreak(2000, r0, k=1)
    P_2000.append(PLO)
    p, PLO = prob_outbreak(2050, r0, k=1)
    P_2050.append(PLO)
plt.plot(np.arange(1, 6, 0.01), P_2000, '-', color='black', lw=2, label='2000')
plt.plot(np.arange(1, 6, 0.01), P_2025, '-', color='blue', lw=2, label='2025')
plt.plot(np.arange(1, 6, 0.01), P_2050, '-', color='red', lw=2, label='2050')
plt.xlabel(r'Basic reproduction number ($R^{(2025)}_0$)', labelpad=10, fontsize=20)
plt.ylabel(r'Average outbreak probability', labelpad=10, fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0, 1)
plt.xlim(1, 6)
plt.legend(fontsize=14, title='Year', title_fontsize=16)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('Figures/outbreak_probability_vs_R0.svg', bbox_inches='tight')
plt.show()
