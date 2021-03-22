#!/usr/bin/env python
"""Analyze EIA retail sales data
   Python Version: 3.83
******
Copyright (c) 2021, Dale Furrow
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import glob
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from zipfile import ZipFile
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# %%
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format
idx = pd.IndexSlice
#%%
DOWNLOAD_ROOT = "./data"  # Put zipfiles in this directory
# Run from this block forward if data has already been cleaned and saved...
records_filename = 'eia_alldata_records.parquet'
print("Loading {0}".format(records_filename))
records_filepath = Path(DOWNLOAD_ROOT, records_filename)
all_records = pd.read_parquet(records_filepath)
print("EIA records retrieved : {0:,}".format(len(all_records)))
data_years = sorted(list(all_records['Year'].unique()))
#%%
tx_records = all_records[all_records['State'] == 'TX'].copy()
print("extract data from Texas, {0:,} rows".format(len(tx_records)))
#%%
# load wall street journal data , calculate simple averages and medians, compare.
wsj_data_retail_prov = [{"y":0.104512769},{"y":0.119135077},{"y":0.147907235},{"y":0.141452687},{"y":0.145587637},
                        {"y":0.141123651},{"y":0.127802827},{"y":0.11816842},{"y":0.11753115},{"y":0.120762946},
                        {"y":0.125853745},{"y":0.122157},{"y":0.113752593},{"y":0.111016108},{"y":0.115200733},
                        {"y":0.125923234}]
wsj_data_retail_prov = [ele['y'] for ele in wsj_data_retail_prov]
wsj_data_trad_uts = [{"y":0.086349232},{"y":0.094671363},{"y":0.100960378},{"y":0.0985534},{"y":0.109442624},
                     {"y":0.10041566},{"y":0.09992686},{"y":0.10082636},{"y":0.099043415},{"y":0.103632208},
                     {"y":0.108971341},{"y":0.106471592},{"y":0.104522108},{"y":0.108908586},{"y":0.107523383},
                     {"y":0.106381295}]
wsj_data_trad_uts = [ele['y'] for ele in wsj_data_trad_uts]
wsj_data = pd.DataFrame(data=None, index=data_years)
wsj_data['DeReg'] = wsj_data_retail_prov
wsj_data['Reg'] = wsj_data_trad_uts
wsj_data = wsj_data[wsj_data.index >= np.min(data_years)].T.copy()
wsj_CustClass = 'wsj_resident'
wsj_index = pd.MultiIndex.from_tuples([('DeReg', wsj_CustClass),
                                       ('Reg', wsj_CustClass)],
                                      names=['OwnershipType', 'CustClass'])
wsj_data = pd.DataFrame(index=wsj_index, columns=data_years, data=np.multiply(wsj_data.values, 100.))
# get unweighted averages from data set, merge wsj data on to it...
cust_subset = ['commercial', 'industrial', 'residential']
avg_price_data = tx_records[(tx_records['ValueType'] == 'AvgPrc') & (tx_records['CustClass'].isin(cust_subset))]
pd.options.display.float_format = '{:,.2f}'.format
aggregate_prices = avg_price_data.pivot_table(values='Value', index=['OwnershipType', 'CustClass'],
                              columns='Year', aggfunc={'Value': [np.mean, np.median]})
aggregate_prices.columns.names = ['Aggregate', 'Year']  # name aggregates column, since there are two elements
aggregate_prices = aggregate_prices.stack(level=0) # unstack to get aggregates in index
aggregate_prices.sort_index(inplace=True)
wsj_data.sort_index(inplace=True)
aggregate_prices.loc[('DeReg', 'residential', 'wsjWtAvg'), :] = wsj_data.loc[idx['DeReg', 'wsj_resident'], :]
aggregate_prices.sort_index(inplace=True)
aggregate_prices.loc[('Reg', 'residential', 'wsjWtAvg'), :] = wsj_data.loc[idx['Reg', 'wsj_resident'], :]
md_str: str = aggregate_prices.loc[idx[:, 'residential', :], :].to_markdown(floatfmt=".2f")
repl_dict = {"('DeReg', 'residential', 'mean')": "Calc DeReg Mean",
             "('DeReg', 'residential', 'median')": "Calc DeReg Median",
             "('DeReg', 'residential', 'wsjWtAvg')": "WSJ DeReg 'Average'",
             "('Reg', 'residential', 'mean')": "Calc Reg Mean",
             "('Reg', 'residential', 'median')": "Calc Reg Median",
             "('Reg', 'residential', 'wsjWtAvg')": "WSJ Reg 'Average'"}
for v1, v2 in repl_dict.items():
    md_str = md_str.replace(v1, v2)
print(md_str)
"""
"So we observe that..."
"(1) the 'averages' in the article must be weighted average, because these values do not tie..."
"(2) the customer of both the 'average' and 'median' retail provider experienced significantly lower prices\n"
"than those characterized as 'average' in the article, and the median retail provider customer has often\n"
"received a lower price than the median regulated customer"
"""
#%%
# block 1 code
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Year')
ax.set_ylabel('Price (\u00A2/kwh)')
ax.set_title("""Pricey Power Revisited--WSJ 'Average' Values vs Median Supplier Prices (\u00A2/kwh)""")
ax.grid(True)
plot_schemes = [{'index_slice': ('DeReg', 'residential', 'mean'), 'color': 'red', 'linestyle': '--', 'marker': 'o',
                 'label': """Median Retail Provider Price"""},
                {'index_slice': ('DeReg', 'residential', 'wsjWtAvg'), 'color': 'blue', 'linestyle': '--', 'marker': 'o',
                 'label': """WSJ: Retail Providers 'Average'"""},
                {'index_slice': ('Reg', 'residential', 'mean'), 'color': 'red', 'linestyle': ':', 'marker': 'x',
                 'label': """Median 'Traditional Utility' Price"""},
                {'index_slice': ('Reg', 'residential', 'wsjWtAvg'), 'color': 'blue', 'linestyle': ':', 'marker': 'x',
                 'label': """WSJ: 'Traditional Utility' 'Average'"""}]
# Use linestyle keyword to style our plot
for plot_scheme in plot_schemes:
    ax.plot(aggregate_prices.columns.tolist(), aggregate_prices.loc[plot_scheme['index_slice'], :].values,
            color=plot_scheme['color'], linestyle=plot_scheme['linestyle'], marker=plot_scheme['marker'],
            label=plot_scheme['label'])
fig.canvas.draw()
ax.legend(loc='upper right')
plt.show()

#%%
# block 2 code
# So now we calculate weighted average
print("So, calculate weighted average price, compare to wsj data...")
print("Note, we take only commercial, industrial and residential categories for comparison,\n"
      " dropping 'all' and 'transport")
aggregate_sums = tx_records.pivot_table(values='Value', index=['ValueType', 'OwnershipType', 'CustClass'],
                                        columns='Year', aggfunc='sum')

aggregate_sums = aggregate_sums.loc[idx[:, :, cust_subset], :]

new_index = pd.MultiIndex.from_product([['calcWtAvg'], aggregate_sums.index.get_level_values(1).unique(),
                                        aggregate_sums.index.get_level_values(2).unique()],
                                       names=['Aggregate', 'OwnershipType', 'CustClass'])
wt_avg_prc_df = pd.DataFrame(data=aggregate_sums.loc[idx['Rev', :, :], :].values * 100. /
                                  aggregate_sums.loc[idx['Sales', :, :], :].values,
                             index=new_index, columns=aggregate_sums.columns)
wt_avg_prc_df = wt_avg_prc_df.reorder_levels(order=[1, 2, 0], axis=0)
aggregate_prices = pd.concat([aggregate_prices, wt_avg_prc_df], axis=0)
aggregate_prices.sort_index(axis=0, inplace=True)
print("compare calculated weighted average with wsj article\n")
md_str = aggregate_prices.loc[idx[:, 'residential', ['calcWtAvg', 'wsjWtAvg']], :].to_markdown(floatfmt=".2f")
repl_dict = {"('DeReg', 'residential', 'calcWtAvg')": "Calc DeReg Median",
             "('DeReg', 'residential', 'wsjWtAvg')": "WSJ DeReg 'Average'",
             "('Reg', 'residential', 'calcWtAvg')": "Calc Reg Median",
             "('Reg', 'residential', 'wsjWtAvg')": "WSJ Reg 'Average'"}
for v1, v2 in repl_dict.items():
    md_str = md_str.replace(v1, v2)
print(md_str)
print("\nand they appear to tie closely")
#%%
# block 3 code
res_tx_2018_dereg = tx_records[(tx_records['Year'] == 2018) & (tx_records['CustClass'] == 'residential') &
                               (tx_records['OwnershipType'] == 'DeReg')]  # 66 records of AvgPrc and Customers
res_tx_2018_dereg[res_tx_2018_dereg.ValueType == 'AvgPrc']['Value'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

price_dict = OrderedDict(zip(res_tx_2018_dereg[res_tx_2018_dereg['ValueType'] == 'AvgPrc']['Value'],
                      res_tx_2018_dereg[res_tx_2018_dereg['ValueType'] == 'Customers']['Value']))

def millions(x, pos):
    'The two args are the value and tick position'
    return "{0:.2f}".format(x*1e-6)

formatter = FuncFormatter(millions)
fig, ax = plt.subplots(figsize=(10, 5)) #10, 6
_ = ax.bar(price_dict.keys(), height=price_dict.values(), align='center', width=0.3)
ax.set_xlim([4., 15.])
ax.yaxis.set_major_formatter(formatter)
ax.set_axisbelow(True)
ax.set_title("'Retail Provider' Residential Customer Prices, Texas, 2018")
ax.set_xlabel('Price (\u00A2/kwh)')
ax.set_ylabel('Number of Customers (Millions)')
print("put in annotations")

reg_res_tx_2018_wtAvg = aggregate_prices.loc[idx['Reg', 'residential', 'calcWtAvg'], 2018]
dereg_res_tx_2018_wtAvg = aggregate_prices.loc[idx['DeReg', 'residential', 'calcWtAvg'], 2018]
dereg_res_tx_2018_median = aggregate_prices.loc[idx['DeReg', 'residential', 'median'], 2018]
dereg_received_price_example = 9.1
annotate_data = [(dereg_received_price_example, 'My Fixed Price\n Aug 2018', 1.2e6),
                 (dereg_res_tx_2018_median, 'Median Supplier\n price', 1.4e6),
                 (reg_res_tx_2018_wtAvg, "Wt Avg\n 'Trad Utilities'", 1.2e6),
                 (dereg_res_tx_2018_wtAvg, "Wt Avg\n 'Retail Provider'", 1.4e6)]
for price, label, text_height in annotate_data:
    ax.annotate(label, xy=(price, .8e6), xytext=(price, text_height),
                arrowprops=dict(facecolor='black'), horizontalalignment='center',
                verticalalignment='top', fontsize=10)
plt.show()
#%%
# block 4 code
print("Who were the top 5 suppliers by price?")
md_str = res_tx_2018_dereg.pivot_table(values='Value', index='Entity', columns='ValueType')\
    .sort_values(by='Customers', ascending=False).head().to_markdown(floatfmt=",.2f")
md_str = md_str.replace('.00', '')
print(md_str)

#%%
# block 5 code
# legacy suppliers vs others
legacy = ['TXU Energy Retail Co, LLC', 'Reliant Energy Retail Services']
res_tx_2018_dereg_pivot = res_tx_2018_dereg.pivot_table(values='Value', index='Entity', columns='ValueType')
print(res_tx_2018_dereg_pivot[res_tx_2018_dereg_pivot.index.isin(legacy)]['Customers'].sum())
print(res_tx_2018_dereg_pivot[~res_tx_2018_dereg_pivot.index.isin(legacy)]['Customers'].sum())
print(res_tx_2018_dereg_pivot[res_tx_2018_dereg_pivot.index.isin(legacy)]['AvgPrc'].mean())
print(res_tx_2018_dereg_pivot[~res_tx_2018_dereg_pivot.index.isin(legacy)]['AvgPrc'].mean())



#%%
prc_diff = aggregate_prices.loc[idx['DeReg', :, 'calcWtAvg'], :].values - \
                     aggregate_prices.loc[idx['Reg', :, 'calcWtAvg'], :].values
new_index = pd.MultiIndex.from_product([['DeRegMinusReg'], cust_subset, ['WtAvgUnitPrcDiff']],
                                       names=aggregate_prices.index.names)
prc_diff = pd.DataFrame(data=prc_diff, index=new_index, columns=aggregate_prices.columns)
print("looking at *all* price differentials we see...")
print(prc_diff)
print("so the experience of the commercial and industrial markets appears to diverge from residential...")

#%%
print("applying that weighted average price difference to Sales, converting to $Bn, we find...")
new_index = pd.MultiIndex.from_product([['WAvgPrcDiff*Sales'], ['DeRegMinusReg'], cust_subset],
                                       names=aggregate_sums.index.names)
wAvgPrcDiffXSales = prc_diff.loc[idx[:, :, :], :].values * \
                    aggregate_sums.loc[idx['Sales', 'DeReg', :], :].values / 1.e8
aggregate_sums = aggregate_sums.append(pd.DataFrame(data=wAvgPrcDiffXSales, index=new_index,
                                                columns=aggregate_sums.columns))
print("looking at *all* price differentials we see...")
wAvgPrcDiffXSales2 = aggregate_sums.loc[idx['WAvgPrcDiff*Sales', :, :], :].copy()
wAvgPrcDiffXSales2.loc[('Total', 'DeRegMinusReg', 'total'), :] = wAvgPrcDiffXSales2.sum(axis=0)
print("Summing across customer categories we see...")
print(wAvgPrcDiffXSales2)
print("So benefits in commercial and industrial in recent years have often more than offset residential")
print("Summing across years to check with article data, we see...")
print(aggregate_sums.loc[idx['WAvgPrcDiff*Sales', :, :], :].sum(axis=1))
print("So we tie to the '$28Bn' calculation in the article...\n"
      "So regardless of the application, the math in the article appears to be correct")
print("for context, total power revenues during the whole period were...")
print(aggregate_sums.loc[idx['Rev', 'DeReg', :], :].sum(axis=1) / 1.e6)
print("over the following customer base...")
dereg_customer_count = aggregate_sums.loc[idx['Customers', 'DeReg', :], :]
print(dereg_customer_count)
print("or in total customer-months...")
print(dereg_customer_count.sum(axis=1) * 12)
# %%
# %%
cust_values_year = aggregate_sums.loc[idx[['WAvgPrcDiff*Sales', 'Customers'], ['DeReg', 'DeRegMinusReg'], :], :].copy()
scaled_prc_times_sales = cust_values_year.loc[idx['WAvgPrcDiff*Sales', :, :], :].values * 1.e9 / \
                     cust_values_year.loc[idx['Customers', :, :], :].values
new_index = pd.MultiIndex.from_product([['PrcDiffPerCustomer'], ['DeReg'], cust_subset],
                                       names=cust_values_year.index.names)
cust_values_year = cust_values_year.append(pd.DataFrame(data=scaled_prc_times_sales, index=new_index,
                                                        columns=cust_values_year.columns))
# sum of customer-months
pd.DataFrame(cust_values_year.sum(axis=1)).loc[idx['Customers', 'DeReg', :], :] * 12
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
