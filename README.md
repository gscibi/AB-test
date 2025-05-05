# A/B Test Analysis

This repository contains the analysis of an A/B test conducted to measure the effectiveness of a new campaign. The analysis includes various statistical tests, visualizations, and insights aimed at understanding whether the new campaign ("Test") performs better than the existing campaign ("Control") in terms of various key metrics.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Analysis](#analysis)
  - [Statistical Testing](#statistical-testing)
  - [Visualizations](#visualizations)
  - [Power Analysis](#power-analysis)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
- [License](#license)

## Introduction

The purpose of this A/B test is to compare the performance of two campaigns (Test vs. Control) based on key business metrics, including spend, clicks, conversions, and more. The analysis is performed using hypothesis testing and power analysis to evaluate whether the observed differences in performance are statistically significant.

## Data
Source of the data: https://www.kaggle.com/datasets/amirmotefaker/ab-testing-dataset/data  
The dataset used in this analysis contains the following columns:
- `campaign_name`: Name of the campaign (Control or Test).
- `spend_[usd]`: Total spend in USD for each campaign.
- `impressions`: Number of impressions for each campaign.
- `reach`: Reach of the campaign (how many unique users were reached).
- `website_clicks`: Total number of website clicks.
- `searches`: Total number of searches initiated.
- `view_content`: Number of users who viewed content.
- `add_to_cart`: Number of items added to the cart.
- `purchase`: Number of purchases.
- `checkout_conversion_rate`: Conversion rate during checkout.

The data is provided in CSV format and can be found in the `data/raw_data.csv` file.

## Analysis

The analysis in this repository includes the following steps:

### Statistical Testing

- **Shapiro-Wilk Test**: Used to check if the data is normally distributed.
- **T-test**: A statistical test used to determine if there is a significant difference between the means of two groups (Control and Test).
- **Cohen's d**: Measures the effect size to understand the magnitude of the difference between the two groups.
- **Power Analysis**: Performed post-hoc to determine if the sample size was sufficient to detect a meaningful difference.

### Visualizations

- Correlation Matrix: To examine relationships between various metrics such as spend, clicks, and conversions.
- A/B Test Plots: Visualizations comparing the key metrics between the Control and Test campaigns.

### Power Analysis

Post-hoc power analysis is performed to evaluate if the statistical tests conducted had enough power to detect significant differences. This helps to ensure that any failure to reject the null hypothesis is not due to a lack of power (i.e., sample size).

## Requirements

To run the analysis, you will need the following Python libraries:
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- statsmodels

You can install these dependencies using pip:

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels
