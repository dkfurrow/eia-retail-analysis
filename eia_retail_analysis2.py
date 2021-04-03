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
# get unweighted averages from data set, combine with weighted averages
cust_subset = ['commercial', 'industrial', 'residential']
avg_price_data = tx_records[(tx_records['ValueType'] == 'AvgPrc') & (tx_records['CustClass'].isin(cust_subset))]
pd.options.display.float_format = '{:,.2f}'.format
aggregate_prices = avg_price_data.pivot_table(values='Value', index=['OwnershipType', 'CustClass'],
                              columns='Year', aggfunc={'Value': [np.mean, np.median]})
aggregate_prices.columns.names = ['Aggregate', 'Year']  # name aggregates column, since there are two elements
aggregate_prices = aggregate_prices.stack(level=0) # unstack to get aggregates in index
aggregate_prices.sort_index(inplace=True)
#%%
# get aggregate sums of revenues, sales volumes and customers
aggregate_sums = tx_records.pivot_table(values='Value', index=['ValueType', 'OwnershipType', 'CustClass'],
                                        columns='Year', aggfunc='sum')

aggregate_sums = aggregate_sums.loc[idx[:, :, cust_subset], :]
aggregate_sums = aggregate_sums.loc[idx[['Customers', 'Rev', 'Sales'], :, :], :]
new_index = pd.MultiIndex.from_product([['WtAvg'], aggregate_sums.index.get_level_values(1).unique(),
                                        aggregate_sums.index.get_level_values(2).unique()],
                                       names=['Aggregate', 'OwnershipType', 'CustClass'])
wt_avg_prc_df = pd.DataFrame(data=aggregate_sums.loc[idx['Rev', :, :], :].values * 100. /
                                  aggregate_sums.loc[idx['Sales', :, :], :].values,
                             index=new_index, columns=aggregate_sums.columns)
wt_avg_prc_df = wt_avg_prc_df.reorder_levels(order=[1, 2, 0], axis=0)
aggregate_prices = pd.concat([aggregate_prices, wt_avg_prc_df], axis=0)
aggregate_prices.sort_index(axis=0, inplace=True)
#%%
# block 1 code
# Draw up three graphs, in accordance with customer segments
axes_cust_segments = ['residential', 'commercial', 'industrial']
reg_segments = {'DeReg': 'red', 'Reg': 'blue'}
analysis_segments = {'WtAvg': {'linestyle': '--', 'marker': 'o'}, 'median': {'linestyle': '--', 'marker': 'x'}}
plt.rc('font', size=12)
fig, axes = plt.subplots(figsize=(10, 15), nrows=3, ncols=1, sharex=True) #10, 6
for i, ax in enumerate(axes):
    ax.set_xlabel('Year')
    ax.set_ylabel('Price (\u00A2/kwh)')
    ax.set_title("""{0} Weighted Avg Values vs Median Supplier Prices (\u00A2/kwh)"""
                 .format(axes_cust_segments[i].capitalize()))
    ax.grid(True)
    ax.set_ylim([5., 15.])
#%%
# populate axes and draw
for i, axes_cust_segment in enumerate(axes_cust_segments):
    for j, reg_segment in enumerate(reg_segments.keys()):
        for k, analysis_segment in enumerate(analysis_segments.keys()):
            # print(reg_segment, axes_cust_segment, analysis_segment)
            # print(aggregate_prices.loc[(reg_segment, axes_cust_segment, analysis_segment), :])
            plot_color = reg_segments[reg_segment]
            plot_lineStyle = analysis_segments[analysis_segment]
            axes[i].plot(aggregate_prices.columns.tolist(),
                         aggregate_prices.loc[(reg_segment, axes_cust_segment, analysis_segment), :].values,
                         color=plot_color, linestyle=plot_lineStyle['linestyle'], marker=plot_lineStyle['marker'],
                    label="{0}_{1}".format(reg_segment, analysis_segment))

for i, axes_cust_segment in enumerate(axes_cust_segments):
    axes[i].legend(loc='upper right')
fig.tight_layout()
fig.canvas.draw()
plt.show()
#%%
total_market_sums = aggregate_sums.sum(axis=0, level=['ValueType', 'OwnershipType'])
new_index = pd.MultiIndex.from_product([['WtAvg'], total_market_sums.index.get_level_values(1).unique()],
                                       names=['Aggregate', 'OwnershipType'])
total_avg_prc_df = pd.DataFrame(data=total_market_sums.loc[idx['Rev', :], :].values * 100. /
                                  total_market_sums.loc[idx['Sales', :], :].values,
                             index=new_index, columns=aggregate_sums.columns)
total_avg_prc_df
# totoal_market_sums = aggregate_sums.groupby(level=['ValueType', 'OwnershipType']).sum().sum()
# total_market_sums = aggregate_sums.swaplevel(axis=0, i=0, j=2)
# total_market_sums = total_market_sums.sum(axis=0, level='CustClass')
# total_market_sums
# total_market_sums = aggregate_sums.sum(axis=0, level='CustClass')
#%%
sales = aggregate_sums.loc[idx['Sales', :, :], :].droplevel(level=0, axis=0)
print(sales)
totes = sales.sum(axis=0, level=['OwnershipType'])
print(totes)
for ownership_type in totes.index.tolist():
    sales.loc[idx[ownership_type, :], :] = sales.loc[idx[ownership_type, :], :].div(totes.loc[ownership_type, :])
print(sales)
#%%
fig, ax = plt.subplots(figsize=(10, 10)) #10, 6
reg_segments = {'DeReg': 'red', 'Reg': 'blue'}
cust_segments = ['residential', 'commercial', 'industrial']
markers = {'commercial', "$c$", }
for ownership_type in reg_segments:
    for customer_class in cust_segments:
        ax.plot(sales.columns.tolist(),
                sales.loc[(ownership_type, customer_class), :].values,
                color=reg_segments[ownership_type], linestyle='--', marker="${0}$".format(customer_class[0].capitalize()),
                markersize=12, label="{0}_{1}".format(ownership_type, customer_class))
ax.legend(loc='upper right')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage of Sales')
ax.set_title("""Percentage of Sales, Retail Providers vs 'Traditional Utilities'""")
ax.grid(True)
ax.set_ylim([.2, .5])
fig.tight_layout()
fig.canvas.draw()
plt.show()

#%%

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
repl_dict = {"('DeReg', 'residential', 'calcWtAvg')": "Calc DeReg WtMean",
             "('DeReg', 'residential', 'wsjWtAvg')": "WSJ DeReg 'Average'",
             "('Reg', 'residential', 'calcWtAvg')": "Calc Reg WtMean",
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

res_tx_2018_dereg_price_dict = OrderedDict(zip(res_tx_2018_dereg[res_tx_2018_dereg['ValueType'] == 'AvgPrc']['Value'],
                                               res_tx_2018_dereg[res_tx_2018_dereg['ValueType'] == 'Customers']
                                               ['Value']))

def millions(x, pos):
    'The two args are the value and tick position'
    return "{0:.2f}".format(x*1e-6)

formatter = FuncFormatter(millions)
fig, axes = plt.subplots(figsize=(10, 10), nrows=2, ncols=1, sharex=True) #10, 6
_ = axes[0].bar(res_tx_2018_dereg_price_dict.keys(), height=res_tx_2018_dereg_price_dict.values(),
             align='center', width=0.3)
axes[0].set_xlim([4., 15.])
axes[0].yaxis.set_major_formatter(formatter)
axes[0].set_axisbelow(True)
axes[0].set_title("'Retail Provider' Residential Customer Prices, Texas, 2018")
axes[0].set_xlabel('Price (\u00A2/kwh)')
axes[0].set_ylabel('Number of Customers (Millions)')
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
    axes[0].annotate(label, xy=(price, .8e6), xytext=(price, text_height),
                  arrowprops=dict(facecolor='black'), horizontalalignment='center',
                  verticalalignment='top', fontsize=10)

res_tx_2018_reg = tx_records[(tx_records['Year'] == 2018) & (tx_records['CustClass'] == 'residential') &
                             (tx_records['OwnershipType'] == 'Reg') &
                             (tx_records['Ownership'].isin(['Municipal', 'Cooperative', 'Investor Owned']))]
res_tx_2018_reg_pivot = res_tx_2018_reg.pivot_table(values='Value', index='Entity', columns='ValueType')
res_tx_2018_reg_pivot = res_tx_2018_reg_pivot[res_tx_2018_reg_pivot['AvgPrc'].notnull()]
res_tx_2018_reg_price_dict = OrderedDict(zip(res_tx_2018_reg_pivot['AvgPrc'].values.tolist(),
                                             res_tx_2018_reg_pivot['Customers'].values.tolist()))


_ = axes[1].bar(res_tx_2018_reg_price_dict.keys(), height=res_tx_2018_reg_price_dict.values(),
             align='center', width=0.3)
axes[1].set_xlim([4., 15.])
axes[1].yaxis.set_major_formatter(formatter)
axes[1].set_axisbelow(True)
axes[1].set_title("'Traditional Utility' Residential Customer Prices, Texas, 2018")
axes[1].set_xlabel('Price (\u00A2/kwh)')
axes[1].set_ylabel('Number of Customers (Millions)')

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
