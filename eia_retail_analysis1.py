#!/usr/bin/env python
"""Example Multivariate Regression
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as ml
import statsmodels.formula.api as smf
import statsmodels.api as sm
from datetime import datetime, date
from tabulate import tabulate
import os
import re
import itertools
from pprint import pprint
from pathlib import Path
from pprint import pprint
import glob
from pathlib import Path
from dateutil.parser import parse
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format
#%%
# specific libraries to download from web
from pandas.tseries.offsets import Day
from io import BytesIO
from zipfile import ZipFile
import requests
from jinja2 import Template
import csv
from collections import OrderedDict
#%%
DOWNLOAD_ROOT = "D:\\DKF\\Data\\EIA"  # Put zipfiles in this directory
zipped_dict = {}
zipped_files = glob.glob(str(Path(DOWNLOAD_ROOT, 'f86*.zip')))
pprint(zipped_files)
#%%
file_data_class = dict([('table6', 'residential'), ('table7', 'commercial'), ('table8', 'industrial'),
                        ('table9', 'transportation'), ('table10', 'all')])
header_units_2008_forward = OrderedDict(zip(['Entity', 'State', 'Ownership', 'Customers', 'Sales', 'Rev', 'AvgPrc'],
                                            ['NA', 'NA', 'NA', 'Count', 'MWH',  '$000', 'cPerKWH']))
header_units_2007_back = OrderedDict(zip(['Entity', 'State', 'Ownership', 'Customers', 'Rev', 'Sales', 'AvgPrc'],
                                            ['NA', 'NA', 'NA', 'Count', '$1000',  'MWH', 'cPerKWH']))

header_types = OrderedDict(zip(['Entity', 'State', 'Ownership', 'Customers', 'Sales',  'Rev', 'AvgPrc'],
                               ['category', 'category', 'category', 'int64', 'float64',  'float64', 'float64']))
data_years = []
all_data = None
for zipped_file in zipped_files:
    f = ZipFile(zipped_file)
    data_year = int(Path(zipped_file).stem[-4:])
    data_years.append(data_year)
    print("fetching data year {0:d}".format(data_year))
    for filename in f.namelist():
        table_stem = Path(filename).stem
        if table_stem in file_data_class.keys():
            data_class = file_data_class[table_stem]
            print("fetching {0}".format(data_class))
            with f.open(filename, 'r') as g:
                df = pd.read_excel(g, na_values=['.', '', '-', '*'])
                # print("before reduce")
                # print(df.head(n=8))
                # find the first 'Alaska Entry', (if not zero then there are extraneous rows in df)
                first_valid_index = df.iloc[:, 1].eq('AK').idxmax()
                if first_valid_index == 0:
                    first_valid_index = df.iloc[:, 1].eq('AR').idxmax()
                if first_valid_index == 0:
                    first_valid_index = df.iloc[:, 1].eq('CA').idxmax()
                print("First Valid Index {0:d}".format(first_valid_index))
                if first_valid_index != 0:
                    df = df.iloc[first_valid_index:, :7]
                    # print("after reduce")
                    # print(df.head())
                if data_year >= 2008:
                    df.columns = header_units_2008_forward.keys()
                else:
                    df.columns = header_units_2007_back.keys()
                df = df.convert_dtypes()
                print("after reformat")
                print(df.head())
                convert_specials = ['Customers', 'Sales', 'Rev']  # convert to float
                for convert_special in convert_specials:
                    df[convert_special] = df[convert_special].astype('float64')
                df.insert(loc=0, column='CustClass', value=data_class)
                df.insert(loc=0, column='Year', value=data_year)
                all_data = pd.concat([all_data, df]) if all_data is not None else df.copy()

#%%
# fix AvgPrice
avg_price_cleaned = []
for val in all_data['AvgPrc'].values:
    if isinstance(val, float) or isinstance(val, int):
        avg_price_cleaned.append(float(val))
    else:
        try:
            avg_price_cleaned.append(float(val.strip()))
        except:
            avg_price_cleaned.append(float('nan'))
all_data['AvgPrc'] = avg_price_cleaned
#%%
# make check col
all_data['check_centsPerKwh'] = all_data['Rev'] / all_data['Sales'] * 100.
# make any adjustments here.
# all_data = all_data[all_data['Entity'].str.find('Adjustment') == -1]

#%%
tx_data = all_data[all_data['State'] == 'TX'].copy()
tx_data['OwnershipType'] = tx_data['Ownership'].apply(
    lambda x: 'DeReg' if x in ['Retail Energy Provider', 'Power Marketer'] else 'Reg')
print("Simple Average of Avg Price...")
pd.options.display.float_format = '{:,.4f}'.format
print(tx_data.pivot_table(values='AvgPrc', index=['OwnershipType', 'CustClass'], columns='Year'))
#%%
print("weighted average...")

sums = tx_data.pivot_table(index=['OwnershipType', 'CustClass'], columns='Year', aggfunc='sum')
idx = pd.IndexSlice

tx_revs_per_sales = np.divide(sums.loc[:, idx['Rev', :]], sums.loc[:, idx['Sales', :]]) * 100.
wt_avg = pd.DataFrame(data=tx_revs_per_sales.values, index=sums.index, columns=data_years)
wt_avg.columns.name = 'Year'
#%%
wsj_data_retail_prov = [{"y":0.104512769},{"y":0.119135077},{"y":0.147907235},{"y":0.141452687},{"y":0.145587637},
                        {"y":0.141123651},{"y":0.127802827},{"y":0.11816842},{"y":0.11753115},{"y":0.120762946},
                        {"y":0.125853745},{"y":0.122157},{"y":0.113752593},{"y":0.111016108},{"y":0.115200733},
                        {"y":0.125923234}]
wsj_data_retail_prov = [ele['y'] for ele in wsj_data_retail_prov]
years = [["1/1/04","1/1/05","1/1/06","1/1/07","1/1/08","1/1/09","1/1/10","1/1/11","1/1/12","1/1/13","1/1/14","1/1/15"
             ,"1/1/16","1/1/17","1/1/18","1/1/19"]]
wsj_data_trad_uts = [{"y":0.086349232},{"y":0.094671363},{"y":0.100960378},{"y":0.0985534},{"y":0.109442624},
                     {"y":0.10041566},{"y":0.09992686},{"y":0.10082636},{"y":0.099043415},{"y":0.103632208},
                     {"y":0.108971341},{"y":0.106471592},{"y":0.104522108},{"y":0.108908586},{"y":0.107523383},
                     {"y":0.106381295}]
wsj_data_trad_uts = [ele['y'] for ele in wsj_data_trad_uts]

wsj_data = pd.DataFrame(data=None, index=data_years)
wsj_data['DeReg'] = wsj_data_retail_prov
wsj_data['Reg'] = wsj_data_trad_uts
wsj_data = wsj_data[wsj_data.index >= np.min(data_years)].T.copy()
wsj_index = pd.MultiIndex.from_tuples([('DeReg', 'wsj'), ('Reg', 'wsj')], names=wt_avg.index.names)
wsj_data = pd.DataFrame(index=wsj_index, columns=wt_avg.columns, data=np.multiply(wsj_data.values, 100.))
#%%
print("Sort Weighted Averages, print out some summaries...")
total_wt_avg = pd.concat([wt_avg, wsj_data], axis=0)
total_wt_avg.sort_index(inplace=True)
print(total_wt_avg.loc[idx[:, ['residential', 'wsj']], :])
print(total_wt_avg.loc[idx[:, ['commercial', 'industrial']], :])
#%%
# reconcile downloaded data with WSJ, graph to check
res_reconcile = total_wt_avg.loc[idx[:, ['residential', 'wsj']], :]
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Year')
ax.set_ylabel('Price (cents/kwh)')
ax.set_title('Pricey Power Revisited--WSJ Values vs EIA Download')
ax.grid(True)
# Use linestyle keyword to style our plot
ax.plot(res_reconcile.columns.tolist(), res_reconcile.loc[('DeReg', 'residential'), :].values, color='red', linestyle='--',
        label='Download: Retail Providers')
ax.plot(res_reconcile.columns.tolist(), res_reconcile.loc[('DeReg', 'wsj'), :].values, color='blue', linestyle='--',
        label='WSJ: Retail Providers')
ax.plot(res_reconcile.columns.tolist(), res_reconcile.loc[('Reg', 'residential'), :].values, color='red', linestyle=':',
        label='Download: Retail Providers')
ax.plot(res_reconcile.columns.tolist(), res_reconcile.loc[('Reg', 'wsj'), :].values, color='blue', linestyle=':',
        label='WSJ: Retail Providers')
fig.canvas.draw()
ax.legend(loc='upper right')
# %%
# Plot Commercial vs Industrial Prices
com_ind = total_wt_avg.loc[idx[:, ['commercial', 'industrial']], :]
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Year')
ax.set_ylabel('Price (cents/kwh)')
ax.set_title("'Pricey Power' Revisited--Commercials and Industrials")
ax.grid(True)
# Use linestyle keyword to style our plot
ax.plot(com_ind.columns.tolist(), com_ind.loc[('DeReg', 'commercial'), :].values, color='red', linestyle='--',
        label='Retail Providers: Commercial')
ax.plot(com_ind.columns.tolist(), com_ind.loc[('Reg', 'commercial'), :].values, color='blue', linestyle='--',
        label="'Traditional': Commercial")
ax.plot(com_ind.columns.tolist(), com_ind.loc[('DeReg', 'industrial'), :].values, color='red', linestyle=':',
        label='Retail Providers: Industrial')
ax.plot(com_ind.columns.tolist(), com_ind.loc[('Reg', 'industrial'), :].values, color='blue', linestyle=':',
        label="'Traditional': Industrial")
fig.canvas.draw()
ax.legend(loc='upper right')
# %%
res_tx_2018_dereg = tx_data[(tx_data['Year'] == 2018) & (tx_data['CustClass'] == 'residential')
                            & (tx_data['AvgPrc'].notnull()) & (tx_data['OwnershipType'] == 'DeReg')]
price_dict = dict(zip(res_tx_2018_dereg['AvgPrc'].values, res_tx_2018_dereg['Customers'].values))

from matplotlib.ticker import FuncFormatter
def millions(x, pos):
    'The two args are the value and tick position'
    return "{0:.2f}".format(x*1e-6)

formatter = FuncFormatter(millions)
fig, ax = plt.subplots(figsize=(10, 5)) #10, 6
# fig.subplots_adjust(left=0.09, right=0.95, top=0.85, bottom=0.2) #left=0.075, right=0.95, top=0.9, bottom=0.25
_ = ax.bar(price_dict.keys(), height=price_dict.values(), align='center', width=0.8)
# ax.set_xticks(list(np.arange(len(other_included))))
# ax.set_xticklabels(other_included.index.tolist(), rotation=90)
ax.set_xlim([4., 15.])
ax.yaxis.set_major_formatter(formatter)
ax.set_axisbelow(True)
ax.set_title("'Retail Provider' Customer Prices, Texas, 2018")
ax.set_xlabel('Price (cents/kwh)')
ax.set_ylabel('Number of Customers (Millions)')
print("put in annotations")

reg_res_tx_2018_wtAvg = wt_avg.loc[idx['Reg', 'residential'], 2018]
dereg_received_price_example = 9.1
annotate_data = [(dereg_received_price_example, 'Fixed Price\n Aug 2018'),
                 (reg_res_tx_2018_wtAvg, "Weighted Average\n 'Traditional Utilities'")]
for price, label in annotate_data:
    ax.annotate(label, xy=(price, .8e6), xytext=(price, 1.2e6),
                arrowprops=dict(facecolor='black'), horizontalalignment='center',
                verticalalignment='top', fontsize=10)

plt.show()
# %%
print("Who were the most expensive Retail Providers?")
print(res_tx_2018_dereg.sort_values(by=['Customers', 'AvgPrc'], ascending=False).head(10))
print("What was the average for the highlighted provider from Aug 2008?")
res_tx_2018_dereg[res_tx_2018_dereg['Entity'] == 'Our Energy LLC']
print("Who were the least expensive Retail Providers?")
print(res_tx_2018_dereg.sort_values(by=['Customers', 'AvgPrc'], ascending=False).tail(10))

# %%
res_tx_2018_reg = tx_data[(tx_data['Year'] == 2018) & (tx_data['CustClass'] == 'residential')
                            & (tx_data['AvgPrc'].notnull()) & (tx_data['OwnershipType'] == 'Reg')]
res_tx_2018_reg = res_tx_2018_reg.sort_values(by='AvgPrc')
# %%
# wt_avg1 = pd.concat({'wt_avg':wt_avg.copy()}, names=)
tot_sales = pd.concat([pd.DataFrame(index=wt_avg.index, columns=wt_avg.columns,
                                    data=sums.loc[:, idx['Sales', :]].values)],
                      keys=['totSales'], names=['DataType'])
wt_avg_with_sales = pd.concat([wt_avg.copy()], keys=['avgPrice'], names=['DataType'])
wt_avg_with_sales = pd.concat([wt_avg_with_sales, tot_sales], axis=0)
# %%
# Reconcile to WSJ '28Bn" number
price_diff = wt_avg_with_sales.loc[idx['avgPrice', 'DeReg', 'residential'], :] - \
             wt_avg_with_sales.loc[idx['avgPrice', 'Reg', 'residential'], :]
dereg_sales = wt_avg_with_sales.loc[idx['totSales', 'DeReg', 'residential'], :]
pro_forma_vals = price_diff * dereg_sales * 1.e-8  # adjust to get to $Bn
pro_forma_losses = pd.DataFrame(data=[price_diff, dereg_sales, pro_forma_vals], index=['price_diff', 'totSales', 'revDiffBns'])

print("revs by ownership type, customer class, in billions")
total_revs = tx_data.pivot_table(index=['OwnershipType', 'CustClass'], aggfunc='sum').loc[:, 'Rev']/1.e6
print(total_revs)
# %%
pd.options.display.float_format = '{:,.2f}'.format
rel_columns = ['Entity', 'Customers', 'Rev', 'Sales']
dereg_custs_2019 = tx_data[(tx_data['Year'] == 2019) & (tx_data['OwnershipType'] == 'DeReg')
                           & (tx_data['CustClass'] == 'residential')].copy().loc[:, rel_columns]
dereg_custs_2019.sort_values(by='Customers', ascending=False, inplace=True)

dereg_custs_2019_sums = dereg_custs_2019.sum()
for cat in rel_columns[1:]:
    col = "{0}_pctTot".format(cat)
    tot = dereg_custs_2019_sums[cat]
    dereg_custs_2019[col] = dereg_custs_2019.loc[:, cat] / tot
print(dereg_custs_2019)

# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
