import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

def estimate():
    df = pd.read_csv('../data/Standard Report - Imports.csv', header=2)
    gdp = pd.read_csv('../data/GDP.csv')
    gdp['year'] = pd.to_datetime(gdp['observation_date']).dt.year
    gdp = gdp.groupby('year').max()

    #print(df.columns.values)
    df = df.groupby('Time').sum().reset_index()
    df = pd.merge(df, gdp, how='inner', left_on=['Time'], right_on=['year'])
    print(df.head())
    #df = df[df['Time'] == 2024]
    df['Calculated Duty ($US)']=df['Calculated Duty ($US)']/1e13
    df['Customs Value (Gen) ($US)']=df['Customs Value (Gen) ($US)']/1e13
    df['GDP'] = df['GDP']/10000
    df['Tariff'] = df['Calculated Duty ($US)'].div(df['Customs Value (Gen) ($US)'])
    df['Tariff Discount'] = df['GDP']/(1+df['Tariff'])
    df = df.dropna()
    plt.scatter(df['Tariff'], df['Customs Value (Gen) ($US)'],)
    plt.title('Figure 2: Tariffs vs. Imports')
    plt.xlabel('Tariff Rate')
    plt.ylabel('Value of Imports')
    #df['recession'] = [1 if obj in (2007,2008,2009,2020) else 0 for obj in df['Time']]
    #gf = pd.get_dummies(df,columns=['Time'], drop_first=True)
    model = sm.OLS(endog=df['Customs Value (Gen) ($US)'], exog=sm.add_constant(df[df.columns.values[-2:]]).astype('float'))
    result = model.fit()
    print(result.summary())
    plt.savefig('../output/figures/figure2.png')
    plt.close()
    #print(df['Tariff'].mean())
    return result.params['Tariff Discount'],-result.params['Tariff'],result.params['Tariff']-result.params['const']

if __name__=='__main__':
    print(estimate())
