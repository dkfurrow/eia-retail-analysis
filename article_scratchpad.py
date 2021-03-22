#!/usr/bin/env python
"""Scratchpad for article narrative, tables and plots
******
Copyright (c) 2021, Dale Furrow
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

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
print(pd.cut(x = hist_data, bins=[10, 100, 1000, 10000, 100000, 1000000, 10000000])
      .value_counts().sort_index().to_markdown())

fig, ax = plt.subplots(figsize=(12, 8)) #10, 6
ax.hist(hist_data.values, bins=30, color='k', alpha=0.5)
# ax.set_xticks(np.arange(-3., 3., 1), minor=False)
ax.set_title("Customer Count of Retail Providers 2019: Historgram")
ax.set_xlabel("Retail Provider")
ax.set_ylabel("Customer Count")
plt.tight_layout(rect=[.10, .10, 1, 0.95])
fig.canvas.draw()
plt.show()
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
