<h1 style="text-align: center;">The Price of Power</h1>
<h3 style="text-align: center;">A More Balanced View of Retail Power Prices in Texas</h3>
######

I read with interest the front-page Wall Street Journal article "[Texas Electric Bills Were $28 Billion Higher Under Deregulation](https://www.wsj.com/articles/texas-electric-bills-were-28-billion-higher-under-deregulation-11614162780?st=4669lbei6w8wzq0&reflink=desktopwebshare_permalink)" of February 24th of this year.  It promised an "analysis" of Texas retail power prices.  The headline graph and its title (reproduced here) seemed promising:
> ###Pricey Power###
> The roughly 60% of Texans who must choose a retail electricity provider consistently pay more than customers in the state who buy their power from traditional utilities.
<img src="./files/20210224WSJGraphReproduction.png" alt="Wsj Graph">
>Source: Wall Street Journal analysis of U.S. Energy Information Administration data

The article then takes the price differential in this graph (elsewhere defined as a difference of 'average' prices for *residential customers only*), multiplies it by the amount of energy sold via retail providers, and *voila*: [28 *Billion Dollars*](https://youtu.be/BRAkobf-tVI?t=11).

Well, that seemed clear enough, except, as someone who regularly shops power suppliers as a retail consumer in Texas, the numbers seemed...suspiciously high.  For example, rummaging around in my records, I found:
<img src="./files/20180909_EFL.jpg" alt="August 2018 EFL">

And, I have a stack of similar or lower-priced records dating back to 2010. My 2018 contract renewal, and the others, seemed to be considerably below *both* the 'Retail Provider' and 'Traditional Utility' numbers provided in the article.  Well, it's not as if this is some obscure marketing scheme--each year, when my contract expires, I go to the puc-sponsored website [powertochoose.org](http://powertochoose.org), choose "12-month, fixed price, sorted" and *choose the lowest price.*  Not a lot of brainpower involved there.  So I have a stack of these 'EFLs' over the last 10 years with similar results, all lower that this graph.

### Questions raised, ***but not answered***: ###
1. So where is this data?  How can I analyze it myself?
1. Who are the 'Traditional Utilities' and 'Retail Providers'? What can be learned from the price comparison?  What are the limitations of such a comparison?
3. Why are my results so different? What is meant by 'average' price?
1. What has gone on in the (roughly 2/3) of the market for commercial and industrial customers?
1. How can I place this $28 Billion in context?  What are some insights that can be gained from this data?

We will answer those questions in two parts.  This article will focus on questions 1-3, and a subsequent article will focus on 4-5.

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
