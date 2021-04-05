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
print("aggregate prices...")
print(aggregate_prices.head())
print("aggregate sums...")
print(aggregate_sums.head())
#%%
new_index = pd.MultiIndex.from_product([['RegMinusDeReg'], aggregate_prices.index.get_level_values(1).unique(),
                                        ['median']], names=['LegacyType', 'CustClass',  'Aggregate'])
RegMinusDereg = aggregate_prices.loc[idx['Reg', :, 'median'], :].values - \
                aggregate_prices.loc[idx['DeReg', :, 'median'], :].values
RegMinusDereg_df = pd.DataFrame(data=RegMinusDereg, index=new_index, columns=aggregate_prices.columns)
new_index = pd.MultiIndex.from_product([['DeRegSavings'], aggregate_prices.index.get_level_values(1).unique(),
                                        ['sumProduct']], names=['LegacyType', 'CustClass',  'Aggregate'])
DeRegSavings_df = pd.DataFrame(data=RegMinusDereg_df.values, index=new_index, columns=aggregate_prices.columns)
print(DeRegSavings_df)
for ind, series in DeRegSavings_df.iterrows():
    multiplier = aggregate_sums.loc[idx['Sales', 'DeReg', ind[1]], :].values
    DeRegSavings_df.loc[ind, :] = DeRegSavings_df.loc[ind, :] * multiplier / 1.e8
print(aggregate_sums.loc[idx['Sales', 'DeReg', :], :])
print(DeRegSavings_df)

out_df = pd.DataFrame(DeRegSavings_df.sum(axis=1))
out_df.index = out_df.index.droplevel([0,2])
out_df.columns = ['All']
out_df.loc['Total'] = out_df.sum(axis=0)
print(out_df.to_markdown(floatfmt=".2f"))
#%%

labels = [str(x) for x in DeRegSavings_df.columns]
x = np.arange(len(labels))  # the label locations
width = 0.25   # the width of the bars
# cust_segments = {'residential':-width/3, 'commercial': 0., 'industrial':width/2.}
cust_segments = ['commercial', 'industrial', 'residential']
fig, ax = plt.subplots(figsize=(10, 6))
rects = []
rects1 = ax.bar(x - width, DeRegSavings_df.loc[idx[:, cust_segments[0], :], :].values.flatten(), width, label=cust_segments[0])
rects2 = ax.bar(x, DeRegSavings_df.loc[idx[:, cust_segments[1], :], :].values.flatten(), width, label=cust_segments[1])
rects3 = ax.bar(x + width, DeRegSavings_df.loc[idx[:, cust_segments[2], :], :].values.flatten(), width, label=cust_segments[2])
# for i, year in enumerate(DeRegSavings_df.columns.to_list()):
#     for j, cust_segment in enumerate(cust_segments.keys()):
#         rects.append(ax.bar(i + cust_segments[cust_segment], DeRegSavings_df.loc[idx[:, cust_segment, :], year],
#                             width))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("""Median Provider Unit Price Difference X Sales Volumes ($Bn)""")
ax.set_title("Median Unit Price Difference X Sales Volumes\n['Traditional Utility' minus 'Retail Provider'")
ax.set_xticks(x)
ax.grid(True)
ax.set_yticks(ticks=np.arange(-2., 2.5, 0.25), minor=True)
ax.set_xticklabels(labels)
ax.legend()
# for rect in rects:
#     ax.bar_label(rect, padding=3)
#
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()



#%%
# create total sales visualization for retail providers vs traditional utililities
# one line per customer class
sales = aggregate_sums.loc[idx['Sales', :, :], :].droplevel(level=0, axis=0)
sales = sales / 1.e6
print(sales.head())
fig, ax = plt.subplots(figsize=(10, 6)) #10, 6
reg_segments = {'DeReg': 'red', 'Reg': 'blue'}
cust_segments = ['residential', 'commercial', 'industrial']
markers = {'commercial', "$c$", }
for ownership_type in reg_segments:
    for customer_class in cust_segments:
        ax.plot(sales.columns.tolist(),
                sales.loc[(ownership_type, customer_class), :].values,
                color=reg_segments[ownership_type], linestyle='--', marker="${0}$".format(customer_class[0].capitalize()),
                markersize=12, label="{0}_{1}".format(ownership_type, customer_class))
ax.legend(loc='upper left')
ax.set_xlabel('Year')
ax.set_ylabel('Sales (TWH)')
ax.set_title("""Total Sales, Retail Providers vs 'Traditional Utilities'""")
ax.grid(True)
ax.set_ylim([25., 100.])
fig.tight_layout()
fig.canvas.draw()
plt.show()
#%%
# Examine maxes and mins in order to set ylim range on next graph
axes_cust_segments = ['residential', 'commercial', 'industrial']
for axes_cust_segment in axes_cust_segments:
    print("maxes and mins by customer class, across years and provider types")
    print(axes_cust_segment)
    print(aggregate_prices.T.loc[:, idx[:, axes_cust_segment, ['WtAvg', 'median']]].min().min())
    print(aggregate_prices.T.loc[:, idx[:, axes_cust_segment, ['WtAvg', 'median']]].max().max())
#%%
# block 1 code
# Draw up three graphs, in accordance with customer segments
axes_cust_segments = ['residential', 'commercial', 'industrial']
ylims = [[8., 15.], [6., 13.], [3.5, 10.5]]  # Range is alway 7 cents, window reflects range of prices
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
    ax.set_ylim(ylims[i])
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
print(total_avg_prc_df)
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Year')
ax.set_ylabel('Price (\u00A2/kwh)')
ax.set_title("""Pricey Power Revisited--Weighted Average Prices\n Retail Provider vs 'Traditional Utility', ALL CUSTOMERS (\u00A2/kwh)""")
ax.grid(True)
plot_schemes = [{'slice': ('WtAvg', 'DeReg'), 'plotColor': 'red'}, {'slice': ('WtAvg', 'Reg'), 'plotColor': 'blue'} ]
for plot_scheme in plot_schemes:
    ax.plot(total_avg_prc_df.columns.tolist(),
                             total_avg_prc_df.loc[plot_scheme['slice'], :].values,
                             color=plot_scheme['plotColor'], linestyle='--', marker='o',
                        label="{0}_{1}".format(plot_scheme['slice'][0], plot_scheme['slice'][1]))
fig.canvas.draw()
ax.legend(loc='upper right')
plt.show()
#%%
# 6
# legacy = ['TXU Energy Retail Co, LLC', 'Reliant Energy Retail Services']
# print(tx_records[(tx_records.Entity.str.find('TXU')!=-1) | (tx_records.Entity.str.find('Reliant')!=-1)])
legacy_list = sorted(list(tx_records[(tx_records.Entity.str.find('TXU')!=-1) |
                                     (tx_records.Entity.str.find('Reliant')!=-1)]['Entity'].unique()))
print(pd.DataFrame(data=legacy_list, columns=["Name"]).to_markdown(index=False))
def get_lecacy_type(row: pd.Series):
    if row['OwnershipType'] == 'Reg':
        return 'Reg'
    else:
        entity: str = row['Entity']
        if entity in legacy_list:
            return 'Legacy'
        else:
            return 'DeReg'
legacy = tx_records.apply(func=get_lecacy_type, axis=1)
tx_records.insert(loc=6, column='LegacyType', value=legacy.values)
print(tx_records[tx_records['LegacyType']=='Legacy'].head())
#%%
# get unweighted averages from data set, combine with weighted averages
cust_subset = ['commercial', 'industrial', 'residential']
avg_price_data = tx_records[(tx_records['ValueType'] == 'AvgPrc') & (tx_records['CustClass'].isin(cust_subset))]
pd.options.display.float_format = '{:,.2f}'.format
aggregate_prices = avg_price_data.pivot_table(values='Value', index=['LegacyType', 'CustClass'],
                              columns='Year', aggfunc={'Value': [np.mean, np.median]})
aggregate_prices.columns.names = ['Aggregate', 'Year']  # name aggregates column, since there are two elements
aggregate_prices = aggregate_prices.stack(level=0) # unstack to get aggregates in index
aggregate_prices.sort_index(inplace=True)
#%%
# get aggregate sums of revenues, sales volumes and customers
aggregate_sums = tx_records.pivot_table(values='Value', index=['ValueType', 'LegacyType', 'CustClass'],
                                        columns='Year', aggfunc='sum')

aggregate_sums = aggregate_sums.loc[idx[:, :, cust_subset], :]
aggregate_sums = aggregate_sums.loc[idx[['Customers', 'Rev', 'Sales'], :, :], :]
# %%
# add differences index to aggregate prices
new_index = pd.MultiIndex.from_product([['legacySwitchSave'], aggregate_prices.index.get_level_values(1).unique(),
                                        aggregate_prices.index.get_level_values(2).unique()],
                                       names=['LegacyType', 'CustClass',  'Aggregate'])
legacySwitchSavings = aggregate_prices.loc[idx['Legacy', :, :], :].values - aggregate_prices.loc[idx['DeReg', :, :], :].values
legacySwitchSavings_df = pd.DataFrame(data= legacySwitchSavings, index=new_index, columns=aggregate_prices.columns)
aggregate_prices = pd.concat([aggregate_prices, legacySwitchSavings_df], axis=0)
aggregate_prices.sort_index(axis=0, inplace=True)

#%%
new_index = pd.MultiIndex.from_product([['LegacySwitchBn'], aggregate_prices.index.get_level_values(1).unique(),
                                        aggregate_prices.index.get_level_values(2).unique()],
                                       names=['LegacyType', 'CustClass',  'Aggregate'])
LegacySwitchBn_df = pd.DataFrame(data=aggregate_prices.loc[idx['legacySwitchSave', :, :], :].values, index=new_index,
                                 columns=aggregate_prices.columns)
print(LegacySwitchBn_df)
for ind, series in LegacySwitchBn_df.iterrows():
    multiplier = aggregate_sums.loc[idx['Sales', 'Legacy', ind[1]], :].values
    LegacySwitchBn_df.loc[ind, :] = LegacySwitchBn_df.loc[ind, :] * multiplier / 1.e8
print(aggregate_sums.loc[idx['Sales', 'Legacy', :], :])
print(LegacySwitchBn_df)

out_df = pd.DataFrame(LegacySwitchBn_df.loc[idx[:, :, 'median'], :].sum(axis=1))
out_df.index = out_df.index.droplevel([0,2])
out_df.columns = ['All']
out_df.loc['Total'] = out_df.sum(axis=0)
print(out_df.to_markdown(floatfmt=".2f"))
#%%
new_index = pd.MultiIndex.from_product([['LegacyPct'], ['Legacy'], aggregate_sums.index.get_level_values(2).unique()],
                                       names="ValueType LegacyType CustClass".split())
pct_legacy = aggregate_sums.loc[idx['Sales', 'Legacy', :], :].values / \
             (aggregate_sums.loc[idx['Sales', 'Legacy', :], :].values +
              aggregate_sums.loc[idx['Sales', 'DeReg', :], :].values)
pct_legacy_df = pd.DataFrame(data = pct_legacy * 100., index=new_index, columns=aggregate_sums.columns)
print(pct_legacy_df)

#%%
# Insert graph of Legacy vs DeReg Volumes here
fig, ax = plt.subplots(figsize=(10, 6)) #10, 6
cust_segments = {'residential':'red', 'commercial':'green', 'industrial':'blue'}
markers = {'commercial', "$c$", }

for customer_class in cust_segments.keys():
    ax.plot(pct_legacy_df.columns.tolist(),
            pct_legacy_df.loc[('LegacyPct', 'Legacy', customer_class), :].values,
            color=cust_segments[customer_class], linestyle='--', marker="${0}$".format(customer_class[0].capitalize()),
            markersize=12, label="{0}".format(customer_class))
ax.legend(loc='upper right')
ax.set_xlabel('Year')
ax.set_ylabel('Sales (% of Total)')
ax.set_title("""Texas Retail Electricity\nMarket Share of Legacy Providers""")
ax.grid(True)
fig.tight_layout()
fig.canvas.draw()
plt.show()
# %%
# Insert graph price savings here

# %%
# Insert multi-bar of potential savings here
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

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
