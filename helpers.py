import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy.stats import mannwhitneyu, ttest_ind, levene, shapiro
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt



## remove outliers
def remove_outliers_iqr(df, cols):
    '''This function removoes outliers in all the columns of the df
    Inputs:
    df = pd.DataFrame
    cols = Columns of the dataframe (list)
    Output:
    df = pd.DataFrame'''
    combined_mask = pd.Series(False, index=df.index)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)

        for idx in df[mask].index:
            print(f"Row {idx} → outlier in column '{col}' (value = {df.at[idx, col]})")

        combined_mask |= mask

    return df[~combined_mask]

def check_normality(col, alpha=0.05):
    '''This function checks the normality of a
    distribution using a Shapiro-Wilk test.
    If p value is bigger than alpha the function is Normal
    (we do not reject null hypotesis), if p value is smaller than
    alpha we reject null hypotesis, so the function is not normal.
    Inputs:
    col = Pandas Series
    alpha = float
    Output:
    str
    '''
    stat, p = shapiro(col)

    if p < alpha:
        return "Not Normal"
    return "Normal"

def ab_test(cleaned_df,col, alpha=0.05, alt="two-sided"):
    '''
    This function performes an A/B test. If the two datasets have
    Normal distribution it will run a T-test, else a Mann-Whitney U test.
    If the two normal distribution have equal variance it will be a Student-T test,
    else a Welch's t-test.
    Input:
    cleaned_df = pd.DataFrame
    col = Column of the df to run the test on (str)
    alpha = Significance Level (float)
    alt = Alternative Hypotesis (str)
    '''
    control = cleaned_df[cleaned_df["campaign_name"] == "Control Campaign"][col]
    test = cleaned_df[cleaned_df["campaign_name"] == "Test Campaign"][col]

    if check_normality(control) == "Normal" and check_normality(test) == "Normal":
        stat, p_lev = levene(control, test)
        equal_var = p_lev > alpha # if eequal variance use Student-T test else use Welch’s t‑test

        stat, p = ttest_ind(control, test, equal_var=equal_var, alternative=alt)
    else:
        stat, p = mannwhitneyu(control, test, alternative=alt)

    conclusion = "Reject H0" if p < 0.05 else "Fail to Reject H0"
    print(f"{col}: {p} and we {conclusion}")

def calculate_cohens_d(cleaned_df, col):
    '''
    This function claculates Cohen's d coefficient, that
    measures the effect size that quantifies the difference
    between two groups.
    Inputs:
    cleaned_df = pd.DataFrame
    col = Column of the df to run the calculation (str)
    Output
    cohens_d = Value of the coefficient (float)
    '''

    control = cleaned_df[cleaned_df["campaign_name"] == "Control Campaign"][col]
    test = cleaned_df[cleaned_df["campaign_name"] == "Test Campaign"][col]

    mean_diff = test.mean() - control.mean() #means diff
    pooled_std = np.sqrt((control.std()**2 + test.std(ddof=1)**2) / 2) #pooled std

    cohens_d = mean_diff / pooled_std
    print(f"Cohen's d: {cohens_d:.2f}")
    return cohens_d

def ideal_sample_size(cohens_d, alpha = 0.05, power = 0.8):
    '''
    This function claculates the ideal sample size for the A/B test
    for a specific significance level and desired power.
    Inputs:
    cohens_d = float
    alpha = significance level (float)
    power = desired power (float)
    '''
    # Parameters
    effect_size = cohens_d
    alpha = 0.05
    power = 0.8

    # Calculate sample size
    analysis = TTestIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)
    print(f"Ideal sample size per group for 0.8 power: {sample_size:.0f}")

def post_hoc_power(cohens_d,test,alternative = 'two-sided'):
    '''
    This function calculates the power of the A/B test
    Inputs:
    cohens_d = float
    test = pd.DataFrame
    alternative = Alternative Hypotesis (str)
    '''
    effect_size = cohens_d
    n =  len(test) #sample size per group
    alpha = 0.05

    power_analysis = TTestIndPower()
    power = power_analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0, alternative=alternative)

    print(f"Power: {power:.3f}")

def plot_numeric(df, func):
    '''
    This function plots the numeric columns of a dataframe.
    Inputs:
    df = pd.DataFrame
    func = function to use for the plot
    '''
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.to_list()

    if "date" in numeric_cols:
        numeric_cols.remove("date")

    n_cols = len(numeric_cols)
    n_per_row = 3
    n_rows = (n_cols + n_per_row - 1) // n_per_row  # Calculate number of rows needed

    fig, axes = plt.subplots(n_rows, n_per_row, figsize=(5 * n_per_row, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        func(data=df, x="campaign_name", y=col, ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_ylabel("")

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()