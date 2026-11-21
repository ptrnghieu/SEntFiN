# Aspect annotation guidelines (taxonomy v1)

Assign exactly one aspect category to each (text, target) pair: the topic of the text **as it concerns that target**.

## Categories


### Corporate

- **Corporate/Financial Results**: Earnings, profit or loss, revenue, sales figures, margins, quarterly or annual results, results previews and guidance, operating metrics reported by the company.
- **Corporate/M&A & Stakes**: Mergers, acquisitions, takeover bids, divestments, buyouts, joint ventures, strategic purchase or sale of a stake in a company.
- **Corporate/Strategy & Operations**: Expansion, new products or launches, capex and investment plans, contracts and orders won or lost, partnerships, restructuring, plant openings or closures, layoffs, the company's own statements about its business outlook.
- **Corporate/Capital & Dividends**: Dividends, buybacks, bonus issues and splits, fund-raising (QIP, rights issue, warrants, bonds or debentures issued by the company, loans), debt levels, credit ratings of the company, promoter share pledges.
- **Corporate/Management & Governance**: Appointments, resignations and exits of executives or directors, board changes, executive pay, governance disputes between promoters, management or shareholders.
- **Corporate/Legal & Regulatory**: Lawsuits, court rulings, arbitration, tax disputes, and actions of regulators or authorities concerning the company: approvals, licences, orders, fines, bans, probes.
- **Corporate/Risks & Reputation**: Adverse events at the company: frauds or accounting irregularities, defaults, bankruptcy, accidents, recalls, strikes, hacks, scandals, reputational damage.

### Stock

- **Stock/Price Movement**: Reported moves of the share price: rises, falls, record highs or lows, top gainer or loser, market-cap changes, when no corporate event is given as the reason.
- **Stock/Outlook & Recommendations**: Forward-looking views on the stock: analyst ratings, upgrades, downgrades, price targets, buy/sell/hold calls and expert advice, technical levels, chart patterns and indicator signals, valuation opinions, and authors or traders disclosing their own trades or positions ("bought more", "long here", "took some puts").
- **Stock/Trading Activity & Ownership**: Market activity in the shares reported as news: trading volumes, block or bulk deals, options and futures activity, short interest, buying or selling by FIIs, mutual funds, insiders, promoters or other institutions, inclusion in or exclusion from an index. An author's own trades are Stock/Outlook & Recommendations.
- **Stock/IPO & Listing**: IPOs and follow-on offers, subscription levels, grey-market premium, listings, listing-day performance, delistings.

### Market

- **Market/Equity Market**: Stock indices (Nifty, Sensex, FTSE, Dow, ...) and the stock market as a whole: moves, trend, outlook, technical levels, volatility, market breadth.
- **Market/Sectors**: A sector, industry or group of stocks (IT stocks, PSU banks, pharma, auto, midcaps, small caps): performance, outlook, recommendations.
- **Market/Commodities**: Gold, silver, crude oil, natural gas, metals, agricultural commodities: prices, demand, supply, outlook.
- **Market/Currencies**: Exchange rates and currency moves or outlook (rupee, dollar, euro, yen, ...).
- **Market/Bonds & Rates**: Government and corporate bond markets, yields, money-market and interest rates as market variables (not a single company raising debt, not a central-bank decision).
- **Market/Funds & Flows**: Mutual funds, ETFs, hedge funds and investor groups (FIIs, DIIs, HNIs, retail) as targets: investment flows, fund performance, fund recommendations, investor positioning.

### Economy

- **Economy/Monetary Policy**: Central banks (RBI, Fed, ECB, ...) and their decisions and stance: policy rates, liquidity measures, inflation targeting, central-bank officials.
- **Economy/Regulation & Government Policy**: Regulators (Sebi, IRDA, TRAI, CCI, ...), stock exchanges acting as market operators, governments and ministries as targets: rules, norms, circulars, budgets, taxation, reforms, disinvestment programmes.
- **Economy/Macroeconomy & Trade**: Countries, regions and economies: GDP, inflation data, industrial output, employment, trade, sanctions, economic outlook.

### Other

- **Other**: The target is only the source of an opinion (a quoted brokerage, analyst or research firm), the text says nothing specific about it, or no category above fits.

## Decision rules

1. Label what the text says about the given target. Two targets in one text can get different aspects ("Infosys drags Nifty lower": Infosys -> Stock/Price Movement, Nifty -> Market/Equity Market).
2. First decide what the target is. A company (including banks, brokerages, fund houses and exchanges acting as companies) -> a Corporate/* or Stock/* label. A stock index or the stock market -> Market/Equity Market. A sector or group of stocks -> Market/Sectors. A commodity -> Market/Commodities. A currency -> Market/Currencies. Bonds or rates -> Market/Bonds & Rates. Funds or investor groups -> Market/Funds & Flows. A central bank -> Economy/Monetary Policy. A regulator, government or an exchange acting as regulator -> Economy/Regulation & Government Policy. A country or economy -> Economy/Macroeconomy & Trade.
3. For a company, business events are Corporate/*; the share price, trading in the shares, opinions about the stock and IPOs are Stock/*.
4. If a price move is explained by a corporate event ("X jumps 5% on strong Q2"), label the event (Corporate/Financial Results), not the move.
5. Stock/Price Movement is for what the price did; Stock/Outlook & Recommendations is for expectations, forecasts, advice, ratings and technical levels. For indices, sectors, commodities and currencies both go into the same Market/* label.
6. When a regulator acts on a company, the company gets Corporate/Legal & Regulatory and the regulator gets Economy/Regulation & Government Policy.
7. A brokerage, analyst or research house that is only quoted as the source of a view about something else gets Other.
8. Use Other only when nothing else fits. Confidence: high = clear-cut, medium = two labels plausible, low = guess.

## Examples

- *MMTC Q2 net loss at Rs 10.4 crore*
  - MMTC → Corporate/Financial Results
- *Pegasystems Inc acquires MeshLabs Software*
  - Pegasystems Inc → Corporate/M&A & Stakes
  - MeshLabs Software → Corporate/M&A & Stakes
- *Hudco raises Rs 279 cr via tax-free bonds*
  - Hudco → Corporate/Capital & Dividends
- *Genpact appoints Edward J Fitzpatrick as CFO*
  - Genpact → Corporate/Management & Governance
- *SEBI slaps Rs 8 lakh penalty on 2 promoters of Blue Blends*
  - Blue Blends → Corporate/Legal & Regulatory
  - SEBI → Economy/Regulation & Government Policy
- *Voltas surges on cooling products-led beat in earnings*
  - Voltas → Corporate/Financial Results
- *Syngenta surges as European shares edge higher*
  - Syngenta → Stock/Price Movement
  - European shares → Market/Equity Market
- *4200 is key level for Titan: Ashwani Gujral*
  - Titan → Stock/Outlook & Recommendations
- *Wonderla Holidays IPO subscribed 42% on Day 1*
  - Wonderla Holidays → Stock/IPO & Listing
- *Bharat Forge to be included in Nifty 50 from April*
  - Bharat Forge → Stock/Trading Activity & Ownership
  - Nifty 50 → Market/Equity Market
- *Be tactful on IT stocks: Deven Choksey*
  - IT stocks → Market/Sectors
- *Gold trades nearly flat on dull demand*
  - Gold → Market/Commodities
- *Rupee ends little changed; fund flows in focus*
  - Rupee → Market/Currencies
- *PSU bonds match equity returns in fourth quarter*
  - PSU bonds → Market/Bonds & Rates
  - equity → Market/Equity Market
- *Mid-cap funds can deliver more, stay put: Experts*
  - Mid-cap funds → Market/Funds & Flows
- *What to expect from RBI policy tomorrow*
  - RBI → Economy/Monetary Policy
- *India's GDP growth slows to 5.7% in April-June quarter*
  - India → Economy/Macroeconomy & Trade
- *All FIIs trying to create some alpha in market: Prateek Agarwal, ASK Investment Manager*
  - FIIs → Market/Funds & Flows
  - ASK Investment Manager → Other
- *Carmaker recalls 1.3 lakh vehicles over faulty airbags; Maruti shares slip*
  - Maruti → Corporate/Risks & Reputation
- *$TSLA adding to my long here, looks ready to break out*
  - TSLA → Stock/Outlook & Recommendations
