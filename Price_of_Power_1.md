<h1 style="text-align: center;">The Price of Power</h1>
<h3 style="text-align: center;">A More Balanced View of Retail Power Prices in Texas</h3>
######

I read with interest the front-page Wall Street Journal article "[Texas Electric Bills Were $28 Billion Higher Under Deregulation](https://www.wsj.com/articles/texas-electric-bills-were-28-billion-higher-under-deregulation-11614162780?st=4669lbei6w8wzq0&reflink=desktopwebshare_permalink)" of February 24th of this year.  It promised an "analysis" of Texas retail power prices.  The headline graph and its title (reproduced here) seemed promising:
> ###Pricey Power###
> The roughly 60% of Texans who must choose a retail electricity provider consistently pay more than customers in the state who buy their power from traditional utilities.
<img src="./files/20210224WSJGraphReproduction.png" alt="Wsj Graph">
>Source: Wall Street Journal analysis of U.S. Energy Information Administration data

The article then takes the price differential in this graph (elsewhere defined as a difference of 'average' prices for *residential customers only*), multiplies it by the amount of energy sold via retail providers, and *voila*: [28 *Billion Dollars*](https://youtu.be/BRAkobf-tVI?t=11).

Well, that seemed clear enough, except, as someone who regularly shops power suppliers as a retail consumer in Texas, the numbers seemed...***suspiciously high***.  For example, rummaging around in my records, I found:
<img src="./files/20180909_EFL.jpg" alt="August 2018 EFL">

And, I have a stack of similar or lower-priced records dating back to 2010. My 2018 contract renewal, and the others, seemed to be considerably below *both* the 'Retail Provider' and 'Traditional Utility' numbers provided in the article.  Well, it's not as if this is some obscure marketing scheme--each year, when my contract expires, I go to the puc-sponsored website [powertochoose.org](http://powertochoose.org), choose "12-month, fixed price, sorted" and *choose the lowest price.*  Not a lot of brainpower involved there.  So I have a stack of these 'EFLs' over the last 10 years with similar results, all lower that this graph.

### Questions raised, ***but not answered***: ###
1. So where is this data?  How can I analyze it myself?
1. Who are the 'Traditional Utilities' and 'Retail Providers'? What can be learned from the price comparison?  What are the limitations of such a comparison?
3. Why are my results so different? What is meant by 'average' price?
1. What has gone on in the (roughly 2/3) of the market for commercial and industrial customers?
1. How can I place this $28 Billion in context?  What are some insights that can be gained from this data?

***We will answer those questions in two parts.  This article will focus on questions 1-3, and a subsequent article will focus on 4-5.***

###Code, Formatting, Conventions###
All of the data and code supporting this article can be found [here](https://github.com/dkfurrow/eia-retail-analysis).  All of the code is in python (I used version 3.83).  The python files are in ordinary *.py format (i.e. they are not jupyter notebooks).  I have separated the code into 'code cells' consistent with those used by the [Spyder IDE](https://docs.spyder-ide.org/current/editor.html). 
###The Data###
The data are from data files the Energy Information Agencies Form EIA-861 data files [here](https://www.eia.gov/electricity/data/eia861/), and copied to github [here](https://github.com/dkfurrow/eia-retail-analysis/tree/master/data).  Those datafiles are in a series of spreadsheets embedded in zipfiles, one file per year.  In the zipfiles, there are spreadsheets, with each spreadsheet representing a customer sector, as follows:

| Table   | Customer Sector   |
|:--------|:------------------|
| table6  | residential       |
| table7  | commercial        |
| table8  | industrial        |
| table9  | transportation    |
| table10 | all               |

The data elements recorded are as follows:

| Columns   | Notes   |
|:----------|:--------|
| Entity    | Name of Company      |
| State     | Two-Character state abbreviation      |
| Ownership | Indicator for type of owner, e.g. 'Cooperative', 'Investor Owned'|
| Customers | Count of Customers   |
| Sales     | Sales volume in MWH     |
| Rev       | Sale revenue in $000    |
| AvgPrc    | Average price in Cents/KWH |

###Conversion and Cleaning###
The script to extract the EIA data is [here](https://github.com/dkfurrow/eia-retail-analysis/blob/master/eia_retail_extract.py).  It's too long to usefully excerpt here, I'll simply note the issues with clear, I'll simply note the process and issues:

1. From 2007 and previous, the EIA reversed the order of Revenue and Sales as noted in the table above.
1. The spreadsheet data started on different lines depending on the year, but always started with Alaska in the case of all sectors *except* Transportation, which started with Arkansas.
1. Some of the early year spreadsheets had an extraneous 8th ('Data Check') column, excluded here.
1. Some of the State values showed as Null, excluded here, likewise all rows where all of Revenue, Sales, and Customers were zero were excluded.

**Whew! Almost done....**

1. Checked that Rev/Sales = AvgPrc, which was true except in some cases where Ownership was 'Other' or 'Behind the Meter' (also Entity was 'adjustment'), in which case AvgPrc was null.  Our focus here isn't on adjustment or behind-meters sales, so I left the data as is.
1. Added feature 'OwneshipType' *['Reg', 'DeReg']* based on whether 'Ownership' is either 'Power Marketer' or 'Retail Provider'...it appears there was a change in terminology over the years.
2. Added columns 'Year' and 'CustClass' (i.e. 'commercial', 'industrial', 'residential', 'transportation', 'all'), converted the whole dataset to a 'records' format with Columns 'ValueType' to indicate Revenues, Sales, Customers or AvgPrc and 'Value' to indicate quantity, and then saved the whole thing to parquet for convenience.

###What can be learned here?###
Well, it's a comprehensive set revenue, cost and customer data, so one can certainly do both horizontal (across customer classes, entities, states) and vertical (across time) comparisons of those data.
###What are the limitations?###
You can most certainly *not* definitively ascertain the effectiveness of either a regulatory regime or market mechanism from this data, as implied in the reviewed article.  Most specifically, there is no data here on:
<p></p>
1. **Wholesale prices:** You can't normalize results with prices available via the (deregulated) wholesale bulk power market--that information isn't here.
1. **Distribution or 'Wires' charges:** retail customers pay a (regulated) fee to access their (unique) distribution system--that value is bundled in the results here.
1. **Generation and Load Characteristics** (Somewhat related to the above) There's no data here on 
	1. Weather/load characteristics [flatter, easier-to-predict loads *should be* cheaper to serve, regardless or regulatory choice]
	1. Local generation asset mix: Does the local utility own cheap hydro generation?  Or natural gas generation [subject to substantial price changes between 2004-2019]?  Are their (inefficient) assets maintained solely for reliability?
	1. Has the customer chosen (more expensive)renewable generation in lieu of fossil power?  That's a choice in Texas, and other states.

**So, it's important to have some humility here, and recognize what the data *can and cannot show.***
