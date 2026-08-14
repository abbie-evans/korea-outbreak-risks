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

    N = pd.read_csv('Data/popage_total2020.csv', usecols=range(2,23), skiprows=1, header=None, dtype=float).values[46]
    N[-6] = np.sum(N[-6:])
    N = N[:-5]
    N = N*1000

    # Density correction method
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
        - tc_p : float
            Contact-weighted average outbreak probability.
    """
    pop_data = pd.read_csv('Data/Nigeria_population.csv', header=0, dtype={'Data': str, 'Population': float})
    pop_data.set_index('year', inplace=True)
    contact_matrix = pd.read_csv('Data/Nigeria_2025_contact_matrix_density.csv', header=None, dtype=float).values
    baseline_contact_matrix = pd.read_csv('Data/Nigeria_2020_contact_matrix.csv', header=None, dtype=float).values
    pop = pop_data.loc[year].values.flatten()

    C = calculate_contact_matrix(baseline_contact_matrix, pop)

    N_prop = pop/np.sum(pop)

    # Calculate Beta=B*C using the reference contact matrix for beta
    Beta = calculate_b(contact_matrix, r0)*C
    eigenvalues = np.linalg.eigvals(Beta)
    print("Maximum eigenvalue of beta:", np.max(eigenvalues))

    initial_guess = np.ones(16)*0.3

    # Solve the system of equations
    solution = fsolve(system_of_equations, initial_guess, args=(Beta,k))
    p = 1 - solution

    # total contacts for age group i
    tc = np.zeros(16)
    for i in range(16):
        tc[i] += C[i, :].sum()

    return p, np.sum(p * N_prop), np.sum(tc*pop*p)/np.sum(tc*pop)

# Figure 5B-D

p, PLO, tc_p = prob_outbreak(2000, 2, k=1) # Can change year to 2000, 2025, or 2050

fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.bar(range(1, 17), p, color='#ffc067', edgecolor='black', linewidth=1)
plt.axhline(y=PLO, color='black', linewidth=2)
plt.axhline(y=tc_p, color='red', linewidth=2, linestyle='--')
plt.xlabel('Age group of index case', labelpad=10, fontsize=20)
plt.ylabel('Major outbreak probability', labelpad=10, fontsize=20)
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
plt.savefig('Figures/Outbreak_probability_nigeria_2000.svg', bbox_inches='tight') # Can change year to 2000, 2025, or 2050
plt.show()

# Figure 5A

data = pd.read_csv('Data/Nigeria_population.csv', header=0, dtype={'Data': str, 'Population': float})
df = pd.DataFrame(data)
# Set year as index
df.set_index('year', inplace=True)

# Convert counts to proportions (row-wise)
df_prop = df.div(df.sum(axis=1), axis=0)

colors = [plt.cm.Blues(i) for i in np.linspace(0.2, 1, 4)] + \
         [plt.cm.Reds(i) for i in np.linspace(0.2, 1, 6)] + \
         [plt.cm.Greens(i) for i in np.linspace(0.2, 1, 6)]

df_prop.plot(
    kind='bar',
    stacked=True,
    color=colors,
    figsize=(6, 6),
    width=0.8
)

fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.xticks(rotation=0, fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Proportion of population', labelpad=10, fontsize=20)
plt.xlabel('Year', labelpad=10, fontsize=20)
plt.legend(title='Age group', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=14)
plt.ylim(0, 1)
# add total population on top of each bar
totals = df.sum(axis=1)
for i, total in enumerate(totals):
    plt.text(i, 1.02, f'{total:,}', ha='center', va='bottom', fontsize=14)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('Figures/Proportions_nigeria.svg', bbox_inches='tight')
plt.show()
