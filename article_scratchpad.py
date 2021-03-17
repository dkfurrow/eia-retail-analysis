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
print(len(tx_records))
print(pd.DataFrame(tx_records[tx_records.ValueType == 'Customers']['CustClass'].value_counts()).to_markdown())

# %%
len(tx_records[(tx_records.OwnershipType == 'DeReg') & (tx_records.CustClass == 'residential') &
               (tx_records.ValueType == 'Customers') & (tx_records.Value > 0.)]['Entity'].unique())
#%%
# characterize unregulated market participants
deregs = tx_records[(tx_records.OwnershipType == 'DeReg') & (tx_records.CustClass == 'residential') &
               (tx_records.ValueType == 'Customers') & (tx_records.Value > 0.)]
print('Unique Dereg Entities over all time: {0:,}'.format(len(deregs['Entity'].unique())))
print('Unique Dereg Entities: {0:,}'.format(len(deregs[deregs['Year'] == 2019]['Entity'].unique())))
print(deregs.pivot_table(values='Entity', columns='Year', aggfunc='count').to_html())

#%%
# characterize regulated entities
regs = tx_records[(tx_records.OwnershipType == 'Reg') & (tx_records.CustClass == 'residential') &
               (tx_records.ValueType == 'Customers') & (tx_records.Value > 0.)]
value_counts = pd.DataFrame(regs[regs.Year == 2019]['Ownership'].value_counts())
total= value_counts.sum(axis=0).astype('int64')
total.name = 'Total'
value_counts = value_counts.append(total)
value_counts.index.name = "Type"
print(value_counts.to_markdown())

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
