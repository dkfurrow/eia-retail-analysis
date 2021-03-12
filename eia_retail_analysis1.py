#!/usr/bin/env python
"""Analyze EIA retail sales data
   Python Version: 3.83
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as ml
from datetime import datetime, date
from tabulate import tabulate
import os
import re
import itertools
from pprint import pprint
from pathlib import Path
from pprint import pprint
import glob
from dateutil.parser import parse
# specific libraries to download from web
from zipfile import ZipFile
from collections import OrderedDict
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format
idx = pd.IndexSlice
#%%
DOWNLOAD_ROOT = "./data"  # Put zipfiles in this directory
zipped_dict = {}
zipped_files = glob.glob(str(Path(DOWNLOAD_ROOT, 'f86*.zip')))
print("Data will be extracted from the following files")
pprint(zipped_files)
#%%
# We want to extract tables 6-10, which have meanings as follows
file_data_class = dict([('table6', 'residential'), ('table7', 'commercial'), ('table8', 'industrial'),
                        ('table9', 'transportation'), ('table10', 'all')])
# EIA switched the order of revenue and sales columns between 2007 and 2008, hence we have two units dictionaries
header_units_2008_forward = OrderedDict(zip(['Entity', 'State', 'Ownership', 'Customers', 'Sales', 'Rev', 'AvgPrc'],
                                            ['NA', 'NA', 'NA', 'Count', 'MWH',  '$000', 'cPerKWH']))
header_units_2007_back = OrderedDict(zip(['Entity', 'State', 'Ownership', 'Customers', 'Rev', 'Sales', 'AvgPrc'],
                                            ['NA', 'NA', 'NA', 'Count', '$1000',  'MWH', 'cPerKWH']))
# These are the desired header types
header_types = OrderedDict(zip(['Entity', 'State', 'Ownership', 'Customers', 'Sales',  'Rev', 'AvgPrc'],
                               ['category', 'category', 'category', 'int64', 'float64',  'float64', 'float64']))
data_years = []
all_data = None
print("fetching EIA data from zipped files...")
for zipped_file in zipped_files:
    f = ZipFile(zipped_file)
    data_year = int(Path(zipped_file).stem[-4:])
    data_years.append(data_year)
    print("fetching all tablses data year {0:d}".format(data_year))
    for filename in f.namelist():
        table_stem = Path(filename).stem
        if table_stem in file_data_class.keys():
            data_class = file_data_class[table_stem]
            print("fetching {0} category for {1:d}".format(data_class, data_year))
            with f.open(filename, 'r') as g:
                df = pd.read_excel(g, na_values=['.', '', '-', '*'])
                # we want to truncate the dataframe using the first valid state index
                # find the first 'Alaska Entry', (if not zero then there are extraneous rows in df)
                first_valid_index = df.iloc[:, 1].eq('AK').idxmax()
                if first_valid_index == 0:
                # If there's no Alaska, then first row is Arkansas [transportation]
                    first_valid_index = df.iloc[:, 1].eq('AR').idxmax()
                if first_valid_index == 0:
                    first_valid_index = df.iloc[:, 1].eq('CA').idxmax()
                if first_valid_index != 0:
                    df = df.iloc[first_valid_index:, :7]  # some spreadsheets have extraneous columns
                # as indicated above, EIA switched the column order in
                if data_year >= 2008:
                    df.columns = header_units_2008_forward.keys()
                else:
                    df.columns = header_units_2007_back.keys()
                df = df.convert_dtypes()
                print("selected dataframe excerpt")
                print(df.head())
                convert_numerics = df.columns.tolist()[3:]  # convert numeric columns toi float
                for convert_numeric in convert_numerics:
                    df[convert_numeric] = pd.to_numeric(df[convert_numeric], errors='coerce')
                    df[convert_numeric] = df[convert_numeric].astype('float64')
                # insert customer class, data year
                df.insert(loc=0, column='CustClass', value=data_class)
                df.insert(loc=0, column='Year', value=data_year)
                # append to get full dataset
                all_data = pd.concat([all_data, df]) if all_data is not None else df.copy()
print("cleaning extraneous null values...")
all_data = all_data[all_data.State.notnull()]  # these are all null rows, 2004-2009
all_data = all_data[(all_data.Rev != 0.) & (all_data.Sales != 0.) & (all_data.Customers != 0.)]
all_data.reset_index(inplace=True)
print("all eia revenue and sales data cleaned and assembled: {0:,} rows".format(len(all_data)))

#%%
# make check column, compare to extracted average price, print useful data summaries
print("Make check columns, examine reults")
all_data['RevPerSales'] = all_data['Rev'] / all_data['Sales'] * 100.
all_data['ChkAvgPrc'] = np.abs(all_data['AvgPrc'] - all_data['RevPerSales'])
num_cols_not_tying = len(all_data[(all_data.ChkAvgPrc > 0.01)])
print("AvgPrc equal to RevPerSales except in {0:d} instances where AvgPrc was listed as zero"
      .format(num_cols_not_tying))
print("Unique categories of Ownership in data:\n{0}".format("\n".join(list(all_data['Ownership'].unique()))))
print("Create column 'OwnershipType' to capture 'Reg' vs 'DeReg' where\n"
      " 'DReg' => 'Power Marketer' or 'Retail Energy Provider' and 'Reg; => all other")
all_data['OwnershipType'] = all_data['Ownership'].apply(
    lambda x: 'DeReg' if x in ['Retail Energy Provider', 'Power Marketer'] else 'Reg')
#%%
tx_data = all_data[all_data['State'] == 'TX'].copy()
print("extract data from Texas, {0:,} rows".format(len(tx_data)))
#%%
print("Copy WSJ Article Data from Json, load into dataframe")
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
wsj_CustClass = 'wsj________'
wsj_index = pd.MultiIndex.from_tuples([('DeReg', wsj_CustClass), ('Reg', wsj_CustClass)], names=['OwnershipType', 'CustClass'])
wsj_data = pd.DataFrame(index=wsj_index, columns=data_years, data=np.multiply(wsj_data.values, 100.))
print(wsj_data)
#%%
print("Compare to unweighted data EIA, mean and median...")
pd.options.display.float_format = '{:,.2f}'.format
print(wsj_data)
aggfuncs = ['mean', 'median']
for aggfunc in aggfuncs:
    print("Table of {0}".format(aggfunc))
    print(tx_data.pivot_table(values='AvgPrc', index=['OwnershipType', 'CustClass'],
                              columns='Year', aggfunc=aggfunc).loc[idx[:, 'residential'], :])
print("So we observe that...")
print("(1) the 'averages' in the article must be weighted average...")
print("(2) the customer of both the 'average' and 'median' retail provider experienced significantly lower prices\n"
      "than those characterized in the article")

#%%
print("")
print("Sort Weighted Averages, print out some summaries...")
total_wt_avg = pd.concat([wt_avg, wsj_data], axis=0)
total_wt_avg.sort_index(inplace=True)
print(total_wt_avg.loc[idx[:, ['residential', 'wsj']], :])
print(total_wt_avg.loc[idx[:, ['commercial', 'industrial']], :])

#%%
print("weighted average...")

sums = tx_data.pivot_table(index=['OwnershipType', 'CustClass'], columns='Year', aggfunc='sum')


tx_revs_per_sales = np.divide(sums.loc[:, idx['Rev', :]], sums.loc[:, idx['Sales', :]]) * 100.
wt_avg = pd.DataFrame(data=tx_revs_per_sales.values, index=sums.index, columns=data_years)
wt_avg.columns.name = 'Year'
#%%

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
