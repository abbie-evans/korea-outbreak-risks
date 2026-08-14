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

    N = pd.read_csv('Data/popage_total2020.csv', usecols=range(2,23), skiprows=1, header=None, dtype=float).values[96]
    N[-6] = np.sum(N[-6:])
    N = N[:-5]
    N = N*1000

    # Density correction
    C_ = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            C_[i,j] = C[i,j]*(np.sum(N)*new_pop[j])/(np.sum(new_pop)*N[j].item())

    plt.figure(figsize=(8, 6))
    sns.heatmap(C_.T, yticklabels=range(1, 16), cmap='Blues', vmin=0, vmax=2)
    plt.gca().invert_yaxis()
    colorbar = plt.gcf().axes[-1]
    colorbar.set_ylabel('Daily contacts', labelpad=10, fontsize=20)
    colorbar.tick_params(labelsize=18)
    # have top of colorbar values say 2+
    colorbar.set_yticks([0, 0.5, 1, 1.5, 2])
    colorbar.set_yticklabels(['0', '0.5', '1', '1.5', '2+'])
    plt.xlabel('Age of individual', labelpad=10, fontsize=24)
    plt.ylabel('Age of contact', labelpad=10, fontsize=24)
    plt.xticks(np.arange(0, 16), labels=['0', '', '10', '', '20', '', '30', '', '40', '', '50', '', '60', '', '70', ''], fontsize=20)
    plt.yticks(np.arange(0, 16), labels=['0', '', '10', '', '20', '', '30', '', '40', '', '50', '', '60', '', '70', ''], fontsize=20, rotation=0)
    plt.tight_layout()
    plt.savefig(f'Figures/Contact_matrix_{year}.svg', bbox_inches='tight')
    plt.show()

    total_contacts = np.sum(C_, axis=1)
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, 17), total_contacts, width=1, color='lightblue', edgecolor='black', linewidth=1)
    plt.ylabel('Total daily contacts', labelpad=10, fontsize=24)
    plt.xticks([])
    plt.yticks(fontsize=20)
    plt.xlim([0.5, 16.75])
    plt.ylim(top=35)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'Figures/Total_contacts_{year}.svg', bbox_inches='tight')
    plt.show()

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
        - tc_p : float
            Contact-weighted outbreak probability.
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

    # total contacts for age group i
    tc = np.zeros(16)
    for i in range(16):
        tc[i] += C[i, :].sum()

    return p, np.sum(p * N_prop), np.sum(tc*pop*p)/np.sum(tc*pop)

p, PLO, tc_p = prob_outbreak(2000, r0=2, k=1) # Can change year to 2000, 2025, or 2050

fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.bar(range(1, 17), p, color='#ffc067', edgecolor='black', linewidth=1)
plt.axhline(y=PLO, color='black', linewidth=2)
plt.axhline(y=tc_p, color='red', linewidth=2, linestyle='--')
plt.xlabel('Age group of index case', labelpad=10, fontsize=24)
plt.ylabel('Major outbreak probability', labelpad=10, fontsize=24)
plt.xticks(np.arange(1, 17), ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                              '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+'],
                              rotation=45)
plt.xlim([0.5, 16.5])
plt.ylim([0, 1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('Figures/Outbreak_probability_2000.svg', bbox_inches='tight') # Can change year to 2000, 2025, or 2050
plt.show()
