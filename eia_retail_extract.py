#!/usr/bin/env python
"""Analyze EIA retail sales data
   Python Version: 3.83
"""
import glob
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from zipfile import ZipFile
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format
idx = pd.IndexSlice
#%%
# files from https://www.eia.gov/electricity/data/eia861/
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
print("Saving to {0}".format(records_filename))
records_filepath = Path(DOWNLOAD_ROOT, records_filename)
all_records.to_parquet(Path(DOWNLOAD_ROOT, records_filename))
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
# %%
# %%
