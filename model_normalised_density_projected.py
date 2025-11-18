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

    pop_data = pd.read_csv('Data/korea_population.csv', header=0, dtype={'Data': str, 'Population': float})
    N = pop_data[(pop_data['Region'] == 'Nationwide') & (pop_data['Year'] == 2020)]['Population'].values.flatten()
    N = pd.read_csv('Data/popage_total2020.csv', usecols=range(2,23), skiprows=1, header=None, dtype=float).values[96]
    N[-6] = np.sum(N[-6:])
    N = N[:-5]
    N = N*1000

    # Normalised density correction
    k = 0
    C_sum = 0
    C_hat = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            C_hat[i,j] = C[i,j]*(np.sum(N)*new_pop[j])/(np.sum(new_pop)*N[j].item())
    for i in range(16):
        for j in range(16):
            C_sum += C_hat[i,j]*new_pop[i]
            k += C[i,j]*N[i].item()
    k = k/np.sum(N)
    C_sum = C_sum/np.sum(new_pop)
    return C_hat/C_sum*k

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

def prob_outbreak(year, r0=2):
    """
    Returns the probability of a local outbreak given an index case in each age group.
    
    Parameters
    -----------
    year : int
        Year for which the probability of a local outbreak is to be calculated.
    r0 : float
        Basic reproduction number.

    Returns
    --------
    tuple
        - p : numpy.ndarray
            Probability of local outbreak for each age group.
        - PLO : float
            Average outbreak probability.
    """
    pop_data = pd.read_csv('Data/korea_population.csv', header=0, dtype={'Data': str, 'Population': float})
    # have baseline as 2020 matrix
    contact_matrix = pd.read_csv('Data/2025_contact_matrix_norm_density.csv', header=None, dtype=float).values
    baseline_contact_matrix = pd.read_csv('Data/contact_matrix_2020.csv', header=None, dtype=float).values
    pop = pop_data[(pop_data['Region'] == 'Nationwide') & (pop_data['Year'] == year)]['Population'].values.flatten()
    pop[-2] = pop[-2] + pop[-1]
    pop = pop[:-1]

    C = calculate_contact_matrix(baseline_contact_matrix, pop)

    N_prop = pop/np.sum(pop)

    # Calculate Beta=B*C*sigma using the reference contact matrix for beta
    Beta = calculate_b(contact_matrix, r0)*C
    eigenvalues = np.linalg.eigvals(Beta)
    print("Maximum eigenvalue of beta:", np.max(eigenvalues))

    initial_guess = np.ones(16)*0.3

    # Solve the system of equations
    solution = fsolve(system_of_equations, initial_guess, args=(Beta,))
    p = 1 - solution

    return p, np.sum(p * N_prop)

p, PLO = prob_outbreak(2000, 2)
p2, PLO2 = prob_outbreak(2025, 2)
p3, PLO3 = prob_outbreak(2050, 2)


fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# First figure
axs[0].bar(range(1, 17), p, color='lightblue', edgecolor='black', linewidth=1)
axs[0].axhline(y=PLO, color='black', linewidth=2)
axs[0].set_title('2000', fontsize=18)
axs[0].set_xlabel('Age group of index case', labelpad=10, fontsize=16)
axs[0].set_ylabel(r'Probability of local outbreak ($p_k$)', labelpad=10, fontsize=16)
axs[0].set_xticks(np.arange(1, 17))
axs[0].set_xticklabels(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                        '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+'],
                        rotation=45, fontsize=12)
axs[0].tick_params(axis='y', labelsize=12)
axs[0].set_xlim([0.5, 16.5])
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# Second figure
axs[1].bar(range(1, 17), p2, color='lightblue', edgecolor='black', linewidth=1)
axs[1].axhline(y=PLO2, color='black', linewidth=2)
axs[1].set_title('2025', fontsize=18)
axs[1].set_xlabel('Age group of index case', labelpad=10, fontsize=16)
axs[1].set_xticks(np.arange(1, 17))
axs[1].set_xticklabels(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                        '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+'],
                        rotation=45, fontsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[1].set_ylim([0, 1])
axs[1].set_xlim([0.5, 16.5])
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Third figure
axs[2].bar(range(1, 17), p3, color='lightblue', edgecolor='black', linewidth=1)
axs[2].axhline(y=PLO3, color='black', linewidth=2)
axs[2].set_title('2050', fontsize=18)
axs[2].set_xlabel('Age group of index case', labelpad=10, fontsize=16)
axs[2].set_xticks(np.arange(1, 17))
axs[2].set_xticklabels(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                        '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+'],
                        rotation=45, fontsize=12)
axs[2].tick_params(axis='y', labelsize=12)
axs[2].set_xlim([0.5, 16.5])
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
