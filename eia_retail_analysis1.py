#!/usr/bin/env python
"""Analyze EIA retail sales data
   Python Version: 3.83
"""
import glob
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
# specific libraries to download from web
from zipfile import ZipFile

import matplotlib.pyplot as plt
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
all_data.reset_index(inplace=True, drop=True)
print("all eia revenue and sales data cleaned and assembled: {0:,} rows".format(len(all_data)))

#%%
# make check column, compare to extracted average price, print useful data summaries
print("Make check columns, examine reults")
all_data['RevPerSales'] = all_data['Rev'] / all_data['Sales'] * 100.
all_data['ChkAvgPrc'] = np.abs(all_data['AvgPrc'] - all_data['RevPerSales'])
num_cols_not_tying = len(all_data[(all_data.ChkAvgPrc > 0.01)])
print("AvgPrc equal to RevPerSales except in {0:d} instances where AvgPrc was listed as zero"
      .format(num_cols_not_tying))
print("It appears that these values are associated with adjustments or behind-meter sales from solar...")
print("Unique categories of Ownership in data:\n{0}".format("\n".join(list(all_data['Ownership'].unique()))))
print("Create column 'OwnershipType' to capture 'Reg' vs 'DeReg' where\n"
      " 'DReg' => 'Power Marketer' or 'Retail Energy Provider' and 'Reg; => all other")
all_data['OwnershipType'] = all_data['Ownership'].apply(
    lambda x: 'DeReg' if x in ['Retail Energy Provider', 'Power Marketer'] else 'Reg')
#%%
# Convert data into records
cats = "Year CustClass Entity State Ownership OwnershipType".split()
value_types = 'Customers Rev Sales AvgPrc'.split()
all_records = pd.DataFrame()
for value_type in value_types:
    print("stacking {0}".format(value_type))
    cols_to_fetch = cats.copy()
    cols_to_fetch.append(value_type)
    df = all_data[cols_to_fetch].copy()
    df.insert(loc=len(df.columns) - 1, column='ValueType', value=value_type)
    df.rename(columns={value_type: 'Value'}, inplace=True)
    if len(all_records) == 0:
        all_records = pd.DataFrame(df)
    else:
        all_records = pd.concat([all_records, df])
all_records.reset_index(drop=True, inplace=True)
#%%
# Save to parquet for convenience
records_filename = 'eia_alldata_records.parquet'
records_filepath = Path(DOWNLOAD_ROOT, records_filename)
all_records.to_parquet(Path(DOWNLOAD_ROOT, records_filename))
#%%
# Run from this block forward if data has already been cleaned and saved...
records_filename = 'eia_alldata_records.parquet'
records_filepath = Path(DOWNLOAD_ROOT, records_filename)
all_records = pd.read_parquet(records_filepath)
print("EIA records retrieved : {0:,}".format(len(all_records)))
data_years = sorted(list(all_records['Year'].unique()))
#%%
tx_records = all_records[all_records['State'] == 'TX'].copy()
print("extract data from Texas, {0:,} rows".format(len(tx_records)))
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
wsj_CustClass = 'wsj_resident'
wsj_index = pd.MultiIndex.from_tuples([('DeReg', wsj_CustClass), ('Reg', wsj_CustClass)], names=['OwnershipType', 'CustClass'])
wsj_data = pd.DataFrame(index=wsj_index, columns=data_years, data=np.multiply(wsj_data.values, 100.))
print(wsj_data)
#%%
print("Compare to unweighted data EIA, mean and median; commercial, industrial, residential...")
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
print(aggregate_prices.loc[idx[:, 'residential', :], :])
print("So we observe that...")
print("(1) the 'averages' in the article must be weighted average, because these values do not tie...")
print("(2) the customer of both the 'average' and 'median' retail provider experienced significantly lower prices\n"
      "than those characterized as 'average' in the article, and the median retail provider customer has often\n"
      "received a lower price than the median regulated customer")
#%%
print("Well, the fact that the weighted average varies so much from the median and mean supplier is interesting...")
print("Let's redo their graph, include the customer of the median supplier")
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Year')
ax.set_ylabel('Price (\u00A2/kwh)')
ax.set_title("""Pricey Power Revisited--WSJ "Average' Values vs Median Supplier Prices (\u00A2/kwh)""")
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
# So now we calculate weighted average
print("So, calculate weighted average price, compare to wsj data...")
print("Note, we take only commercial, industrial and residential categories for comparison,\n"
      " dropping 'all' and 'transport")
aggregates = tx_records.pivot_table(values='Value', index=['ValueType', 'OwnershipType', 'CustClass'],
                                    columns='Year', aggfunc='sum')

aggregates = aggregates.loc[idx[:, :, cust_subset], :]

new_index = pd.MultiIndex.from_product([['calcWtAvg'], aggregates.index.get_level_values(1).unique(),
                                        aggregates.index.get_level_values(2).unique()],
                                       names=['Aggregate', 'OwnershipType', 'CustClass'])
wt_avg_prc_df = pd.DataFrame(data=aggregates.loc[idx['Rev', :, :], :].values * 100. /
                                  aggregates.loc[idx['Sales', :, :], :].values,
                             index=new_index, columns=aggregates.columns)
wt_avg_prc_df = wt_avg_prc_df.reorder_levels(order=[1, 2, 0], axis=0)
aggregate_prices = pd.concat([aggregate_prices, wt_avg_prc_df], axis=0)
aggregate_prices.sort_index(axis=0, inplace=True)
print("compare calculated weighted average with wsj article\n")
print(aggregate_prices.loc[idx[:, 'residential', ['calcWtAvg', 'wsjWtAvg']], :])
print("\nand they appear to tie closely")

#%%
wt_avg_prc_diff = aggregates.loc[idx['WtAvgUnitPrc', 'DeReg', :], :].values - \
                     aggregates.loc[idx['WtAvgUnitPrc', 'Reg', :], :].values
new_index = pd.MultiIndex.from_product([['WtAvgUnitPrcDiff'], ['DeRegMinusReg'], cust_subset],
                                       names=aggregates.index.names)
aggregates = aggregates.append(pd.DataFrame(data=wt_avg_prc_diff, index=new_index, columns=aggregates.columns))
print("looking at *all* price differentials we see...")
print(aggregates.loc[idx['WtAvgUnitPrcDiff', :, :], :])
print("so the experience of the commercial and industrial markets appears to diverge from residential...")

#%%
print("applying that weighted average price difference to Sales, converting to $Bn, we find...")
new_index = pd.MultiIndex.from_product([['WAvgPrcDiff*Sales'], ['DeRegMinusReg'], cust_subset],
                                       names=aggregates.index.names)
wAvgPrcDiffXSales = aggregates.loc[idx['WtAvgUnitPrcDiff', :, :], :].values * \
                    aggregates.loc[idx['Sales', 'DeReg', :], :].values / 1.e8
aggregates = aggregates.append(pd.DataFrame(data=wAvgPrcDiffXSales, index=new_index, columns=aggregates.columns))
print("looking at *all* price differentials we see...")
wAvgPrcDiffXSales2 = aggregates.loc[idx['WAvgPrcDiff*Sales', :, :], :].copy()
wAvgPrcDiffXSales2.loc[('Total', 'DeRegMinusReg', 'total'), :] = wAvgPrcDiffXSales2.sum(axis=0)
print("Summing across customer categories we see...")
print(wAvgPrcDiffXSales2)
print("So benefits in commercial and industrial in recent years have often more than offset residential")
print("Summing across years to check with article data, we see...")
print(aggregates.loc[idx['WAvgPrcDiff*Sales', :, :], :].sum(axis=1))
print("So we tie to the '$28Bn' proxy in the article...\n"
      "So regardless of the application, the math in the article appears to be correct")
print("for context, total power revenues during the whole period were...")
print(aggregates.loc[idx['Rev', 'DeReg', :], :].sum(axis=1) / 1.e6)
print("over the following customer base...")
dereg_customer_count = aggregates.loc[idx['Customers', 'DeReg', :], :]
print(dereg_customer_count)
print("differential per customer-month")
dollarsPerCustMonth = ((aggregates.loc[idx['WAvgPrcDiff*Sales', :, :], :] * 1.e9).values /
                      aggregates.loc[idx['Customers', 'DeReg', :], :].values) / 12.
new_index = pd.MultiIndex.from_product([['AppliedValPerCustMonth'], ['DeRegMinusReg'], cust_subset],
                                       names=aggregates.index.names)
print(pd.DataFrame(data=dollarsPerCustMonth, index=new_index, columns=aggregates.columns))
print()

#%%
sums = pd.DataFrame()
for category in ['Customers', 'Rev', 'Sales']:
    cat_df = tx_data[category]

tx_revs_per_sales = np.divide(sums.loc[:, idx['Rev', :]], sums.loc[:, idx['Sales', :]]) * 100.
wt_avg = pd.DataFrame(data=tx_revs_per_sales.values, index=sums.index, columns=data_years)
wt_avg.columns.name = 'Year'
aggregated = pd.concat({"WeightedAvg": wt_avg}, keys=["WeightedAvg"], names=['Value_Category', 'OwnershipType', 'CustClass'])
print(aggregated)
print("Compare with WJS Values")
print(wsj_data)
print("And those tie quite closely!")
#%%
print("Take weighted average price differentials, apply to sales in MWH, convert to $BN")


aggregated = aggregated.loc[idx[:, :, cust_subset], :]
dereg_minus_reg = np.add(aggregated.loc[idx[:, 'DeReg', :], :], -aggregated.loc[idx[:, 'Reg', :], :])
new_index = pd.MultiIndex.from_product([['WtAvgDiff'], ['DeRegMinusReg'], dereg_minus_reg.index.get_level_values(2)],
                                   names=dereg_minus_reg.index.names)
dereg_minus_reg.index = new_index
print("so the differences in *weighted average* prices are...")
print(dereg_minus_reg)
dereg_sales = sums.loc[idx['DeReg', cust_subset], idx['Sales', :]]
print("against these sales in MWH...")
dereg_sales.columns = dereg_minus_reg.columns
print(dereg_sales)
new_index = pd.MultiIndex.from_product([['WtAvgDiff*Sales'], ['DeRegMinusReg'],
                                        dereg_minus_reg.index.get_level_values(2)], names=dereg_minus_reg.index.names)
diffTimeSales = pd.DataFrame(data=np.multiply(dereg_minus_reg.values, dereg_sales.values) / 1.e8, index = new_index,
                             columns=dereg_minus_reg.columns)
print("Yields this difference in $bn...")
print(diffTimeSales)
print("Compared to these actual costs in $bn")
actual_revs = pd.concat({'Revs_Bn':(sums.loc[idx["DeReg", cust_subset], idx['Rev', :]] / 1.e6)},
                        keys=['Revs_Bn'], names=['Value_Category'])
print(actual_revs)
print("And this number of customers")
total_cust = pd.concat({'Custs_Mn':(sums.loc[idx["DeReg", cust_subset], idx['Customers', :]] / 1.e6)},
                       keys=['Custs_Mn'], names=['Value_Category'])
print("So this dollar figure per customer")
print(diffTimeSales * 1.e9 / (sums.loc[idx["DeReg", cust_subset], idx['Customers', :]]))
#%%

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
