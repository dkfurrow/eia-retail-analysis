#!/usr/bin/env python
"""Scratchpad for article narrative
"""
# %%
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
# %%
