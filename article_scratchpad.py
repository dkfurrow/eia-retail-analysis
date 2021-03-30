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
# this section assumes the tx_records dataset in memory
# total number of dereg names
len(tx_records[tx_records.OwnershipType == 'DeReg']['Entity'].unique())
# total number of dereg names with residential customers
len(tx_records[(tx_records.OwnershipType == 'DeReg') & (tx_records.CustClass == 'residential') &
               (tx_records.ValueType == 'Customers') & (tx_records.Value > 0.)]['Entity'].unique())
# count of rows showing customers
tx_records[(tx_records.OwnershipType == 'DeReg') & (tx_records.ValueType == 'Customers')]\
    .pivot_table(values='Value', columns='Year' ,aggfunc='count')
# count of rows showing customers
tx_records[(tx_records.OwnershipType == 'DeReg') & (tx_records.CustClass == 'residential') &
               (tx_records.ValueType == 'Customers') & (tx_records.Value > 0.)]\
    .pivot_table(values='Value', index='Entity', columns='Year' ,aggfunc='count')
# %%
print(len(tx_records))
print(pd.DataFrame(tx_records[tx_records.ValueType == 'Customers']['CustClass'].value_counts()).to_markdown())

# %%
len_unique_names = len(tx_records[(tx_records.OwnershipType == 'DeReg') & (tx_records.CustClass == 'residential')
                                  & (tx_records.ValueType == 'Customers') & (tx_records.Value > 0.)]['Entity'].unique())
print("Number of unique entity names: {0:,}".format(len_unique_names))
#%%
# characterize unregulated market participants
deregs = tx_records[(tx_records.OwnershipType == 'DeReg') & (tx_records.CustClass == 'residential') &
               (tx_records.ValueType == 'Customers') & (tx_records.Value > 0.)]
print('Unique Dereg Entities over all time: {0:,}'.format(len(deregs['Entity'].unique())))
print('Unique Dereg Entities: {0:,}'.format(len(deregs[deregs['Year'] == 2019]['Entity'].unique())))
print(deregs.pivot_table(values='Entity', columns='Year', aggfunc='count').to_markdown())
#%%
# generate table based on customer count
bin_data = deregs[deregs.Year == 2019]['Value']
print("Total Customers: {0:,.0f}".format(bin_data.sum()))
retailer_size = [1, 10, 100, 1000, 10000, 100000, 500000, 1000000, 2000000]
bin_table = pd.cut(x = bin_data, bins=retailer_size).value_counts().sort_index()
retailer_ind = pd.Index(["{0:,}-{1:,}".format(x.left, x.right) for x in bin_table.index], name='Retailer Size')
bin_table.index = retailer_ind
bin_table.name='Retailer Count'
print(bin_table.to_markdown())
print("Total Retailers: {0:d}".format(bin_table.sum()))
#%%
# characterize regulated entities
res_tx_2019_reg = tx_records[(tx_records['Year'] == 2019) & (tx_records['CustClass'] == 'residential') &
                               (tx_records['OwnershipType'] == 'Reg')]
res_tx_2019_reg_pivot = res_tx_2019_reg.pivot_table(values='Value', index=['Ownership', 'Entity'], columns='ValueType')
res_tx_2019_reg_pivot.drop('AvgPrc', axis=1, inplace=True)
sums = res_tx_2019_reg_pivot.sum(axis=0, level=0)
# print(res_tx_2019_reg_pivot.sum(axis=0, level=0).to_markdown(floatfmt=",.0f"))
entity_count = res_tx_2019_reg_pivot.count(axis=0, level=0)['Customers']
sums.insert(loc=0, column='Count', value=entity_count.values)
total= sums.sum(axis=0)
sums.loc['Total', :] = total
print(sums.to_markdown(floatfmt=",.0f"))

#%%
regs = tx_records[(tx_records.OwnershipType == 'Reg') & (tx_records.CustClass == 'residential') &
               (tx_records.ValueType == 'Customers') & (tx_records.Value > 0.)]
#%%
regs = regs[regs['Entity'].duplicated(keep='last')]
regs = regs[regs['Year'].duplicated(keep='last')]


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
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
