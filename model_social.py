import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    pop_data = pd.read_csv('population.csv', header=0, dtype={'Data': str, 'Population': float})
    N = pop_data[(pop_data['Region'] == 'Nationwide') & (pop_data['Year'] == 2020)]['Population'].values.flatten()
    N = pd.read_csv('popage_total2020.csv', usecols=range(2,23), skiprows=1, header=None, dtype=float).values[96]
    N[-6] = np.sum(N[-6:])
    N = N[:-5]
    N = N*1000

    # M2 method
    C_ = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            C_[i,j] = C[i,j]*(np.sum(N)*new_pop[j])/(np.sum(new_pop)*N[j].item())

    for i in range(16):
        for j in range(16):
            C[i,j] = 0.5*(C_[i,j] + (new_pop[j]/new_pop[i])*C_[j,i])

    return C

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

def prob_outbreak(year, r0=2, change=False):
    """
    Returns the probability of a local outbreak given an index case in each age group.
    
    Parameters
    -----------
    year : int
        Year for which the probability of a local outbreak is to be calculated.
    location : str
        Location for which the probability of a local outbreak is to be calculated.

    Returns
    --------
    tuple
        - p : numpy.ndarray
            Probability of local outbreak for each age group.
        - PLO : float
            Average outbreak probability.
    """
    pop_data = pd.read_csv('population.csv', header=0, dtype={'Data': str, 'Population': float})
    # baseline_contact_matrix = pd.read_csv('2025_contact_matrix_v2_pairwise.csv', header=None, dtype=float).values
    # have baseline as 2020 matrix
    contact_matrix = pd.read_csv('2025_contact_matrix_v2_pairwise.csv', header=None, dtype=float).values
    baseline_contact_matrix = pd.read_csv('contact_matrix_2020_new.csv', header=None, dtype=float).values
    pop = pop_data[(pop_data['Region'] == 'Nationwide') & (pop_data['Year'] == year)]['Population'].values.flatten()
    pop[-2] = pop[-2] + pop[-1]
    pop = pop[:-1]

    if change == True:
        C = calculate_contact_matrix(baseline_contact_matrix, pop)
        C_2050 = pd.read_csv('2050_contact_matrix.csv', header=None, dtype=float).values
        C_ = C_2050.copy()
        # calculate average number of contacts in ages 45-60
        avg_contacts = (sum(C[9, :])*pop[9] + sum(C[10, :])*pop[10] + sum(C[11, :])*pop[11])/(np.sum(pop[9:12]))
        avg_contacts_older = (sum(C[12, :])*pop[12] + sum(C[13, :])*pop[13] + sum(C[14, :])*pop[14])/(np.sum(pop[12:15]))
        frac = avg_contacts / avg_contacts_older
        a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, frac, frac, frac, 1])
        C_mod = (a[:, None] * a[None, :]) * C_
        C = C_mod.copy()
    else:
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

p, PLO = prob_outbreak(2050, 2, change=True)
p2, PLO2 = prob_outbreak(2050, 2, change=False)

fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.bar(range(1, 17), p, color='#8da0cb', edgecolor='black', linewidth=1)
plt.bar(range(1, 17), p2, color='lightgrey', edgecolor='black', linewidth=1)
plt.axhline(y=PLO, color='blue', linewidth=2, label='Increased retirement age')
plt.axhline(y=PLO2, color='black', linewidth=2, label='No change')
plt.xlabel('Age group of index case', labelpad=10, fontsize=20)
plt.ylabel(r'Probability of local outbreak ($p_k$)', labelpad=10, fontsize=20)
plt.xticks(np.arange(1, 17), ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                              '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+'],
                              rotation=45)
plt.xlim([0.5, 16.5])
plt.ylim([0, 1])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title='Average outbreak probability (P)', fontsize=14, title_fontsize=16)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('2025-exp.png', dpi=300, bbox_inches='tight')
plt.show()
