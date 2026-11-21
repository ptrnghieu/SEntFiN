"""Unified aspect-category taxonomy for SEntFiN and FiQA (version v1).

Why a new taxonomy instead of FiQA's own:

* FiQA's 28 second-level aspects were designed for UK/US company news and
  stock tweets. SEntFiN is dominated by things FiQA barely covers: stock
  indices (Nifty, Sensex), regulators and central banks (Sebi, RBI),
  commodities (gold, crude) and currencies (rupee). FiQA has 47 Market /
  Economy examples in total and no commodity class.
* FiQA's gold aspects are inconsistent: "Broker tips: RBS, Croda, Sage" is
  Corporate/Rumors, profit announcements are Stock/Fundamentals, and
  Market/Market is used for company targets.

The aspect is assigned per (text, target): what the text says about *that*
target. Labels are "<Level 1>/<Level 2>"; level 1 is one of Corporate, Stock,
Market, Economy, Other.

This module is the single source of truth for the LLM prompt, the human
annotation guidelines and the review spreadsheet's dropdown.
"""

from __future__ import annotations

from dataclasses import dataclass

TAXONOMY_VERSION = "v1"


@dataclass(frozen=True)
class AspectCategory:
    label: str
    definition: str


CATEGORIES: tuple[AspectCategory, ...] = (
    # --- Corporate: business events and fundamentals of a company -------------
    AspectCategory("Corporate/Financial Results",
                   "Earnings, profit or loss, revenue, sales figures, margins, quarterly or annual results, "
                   "results previews and guidance, operating metrics reported by the company."),
    AspectCategory("Corporate/M&A & Stakes",
                   "Mergers, acquisitions, takeover bids, divestments, buyouts, joint ventures, strategic "
                   "purchase or sale of a stake in a company."),
    AspectCategory("Corporate/Strategy & Operations",
                   "Expansion, new products or launches, capex and investment plans, contracts and orders won "
                   "or lost, partnerships, restructuring, plant openings or closures, layoffs, the company's "
                   "own statements about its business outlook."),
    AspectCategory("Corporate/Capital & Dividends",
                   "Dividends, buybacks, bonus issues and splits, fund-raising (QIP, rights issue, warrants, "
                   "bonds or debentures issued by the company, loans), debt levels, credit ratings of the "
                   "company, promoter share pledges."),
    AspectCategory("Corporate/Management & Governance",
                   "Appointments, resignations and exits of executives or directors, board changes, executive "
                   "pay, governance disputes between promoters, management or shareholders."),
    AspectCategory("Corporate/Legal & Regulatory",
                   "Lawsuits, court rulings, arbitration, tax disputes, and actions of regulators or "
                   "authorities concerning the company: approvals, licences, orders, fines, bans, probes."),
    AspectCategory("Corporate/Risks & Reputation",
                   "Adverse events at the company: frauds or accounting irregularities, defaults, bankruptcy, "
                   "accidents, recalls, strikes, hacks, scandals, reputational damage."),
    # --- Stock: the company's shares and how investors treat them ----------------
    AspectCategory("Stock/Price Movement",
                   "Reported moves of the share price: rises, falls, record highs or lows, top gainer or loser, "
                   "market-cap changes, when no corporate event is given as the reason."),
    AspectCategory("Stock/Outlook & Recommendations",
                   "Forward-looking views on the stock: analyst ratings, upgrades, downgrades, price targets, "
                   "buy/sell/hold calls and expert advice, technical levels, chart patterns and indicator "
                   "signals, valuation opinions, and authors or traders disclosing their own trades or "
                   "positions (\"bought more\", \"long here\", \"took some puts\")."),
    AspectCategory("Stock/Trading Activity & Ownership",
                   "Market activity in the shares reported as news: trading volumes, block or bulk deals, "
                   "options and futures activity, short interest, buying or selling by FIIs, mutual funds, "
                   "insiders, promoters or other institutions, inclusion in or exclusion from an index. An "
                   "author's own trades are Stock/Outlook & Recommendations."),
    AspectCategory("Stock/IPO & Listing",
                   "IPOs and follow-on offers, subscription levels, grey-market premium, listings, listing-day "
                   "performance, delistings."),
    # --- Market: the market or an asset class as a whole ------------------------
    AspectCategory("Market/Equity Market",
                   "Stock indices (Nifty, Sensex, FTSE, Dow, ...) and the stock market as a whole: moves, "
                   "trend, outlook, technical levels, volatility, market breadth."),
    AspectCategory("Market/Sectors",
                   "A sector, industry or group of stocks (IT stocks, PSU banks, pharma, auto, midcaps, "
                   "small caps): performance, outlook, recommendations."),
    AspectCategory("Market/Commodities",
                   "Gold, silver, crude oil, natural gas, metals, agricultural commodities: prices, demand, "
                   "supply, outlook."),
    AspectCategory("Market/Currencies",
                   "Exchange rates and currency moves or outlook (rupee, dollar, euro, yen, ...)."),
    AspectCategory("Market/Bonds & Rates",
                   "Government and corporate bond markets, yields, money-market and interest rates as market "
                   "variables (not a single company raising debt, not a central-bank decision)."),
    AspectCategory("Market/Funds & Flows",
                   "Mutual funds, ETFs, hedge funds and investor groups (FIIs, DIIs, HNIs, retail) as targets: "
                   "investment flows, fund performance, fund recommendations, investor positioning."),
    # --- Economy: macroeconomy and policy ---------------------------------------
    AspectCategory("Economy/Monetary Policy",
                   "Central banks (RBI, Fed, ECB, ...) and their decisions and stance: policy rates, "
                   "liquidity measures, inflation targeting, central-bank officials."),
    AspectCategory("Economy/Regulation & Government Policy",
                   "Regulators (Sebi, IRDA, TRAI, CCI, ...), stock exchanges acting as market operators, "
                   "governments and ministries as targets: rules, norms, circulars, budgets, taxation, "
                   "reforms, disinvestment programmes."),
    AspectCategory("Economy/Macroeconomy & Trade",
                   "Countries, regions and economies: GDP, inflation data, industrial output, employment, "
                   "trade, sanctions, economic outlook."),
    # --- Other ----------------------------------------------------------------
    AspectCategory("Other",
                   "The target is only the source of an opinion (a quoted brokerage, analyst or research firm), "
                   "the text says nothing specific about it, or no category above fits."),
)

LABELS: tuple[str, ...] = tuple(c.label for c in CATEGORIES)
L1_LABELS: tuple[str, ...] = tuple(dict.fromkeys(label.split("/")[0] for label in LABELS))
CONFIDENCE = ("high", "medium", "low")


def level1(label: str | None) -> str | None:
    return label.split("/")[0] if label else None


RULES: tuple[str, ...] = (
    "Label what the text says about the given target. Two targets in one text can get different aspects "
    "(\"Infosys drags Nifty lower\": Infosys -> Stock/Price Movement, Nifty -> Market/Equity Market).",
    "First decide what the target is. A company (including banks, brokerages, fund houses and exchanges "
    "acting as companies) -> a Corporate/* or Stock/* label. A stock index or the stock market -> "
    "Market/Equity Market. A sector or group of stocks -> Market/Sectors. A commodity -> Market/Commodities. "
    "A currency -> Market/Currencies. Bonds or rates -> Market/Bonds & Rates. Funds or investor groups -> "
    "Market/Funds & Flows. A central bank -> Economy/Monetary Policy. A regulator, government or an exchange "
    "acting as regulator -> Economy/Regulation & Government Policy. A country or economy -> "
    "Economy/Macroeconomy & Trade.",
    "For a company, business events are Corporate/*; the share price, trading in the shares, opinions about "
    "the stock and IPOs are Stock/*.",
    "If a price move is explained by a corporate event (\"X jumps 5% on strong Q2\"), label the event "
    "(Corporate/Financial Results), not the move.",
    "Stock/Price Movement is for what the price did; Stock/Outlook & Recommendations is for expectations, "
    "forecasts, advice, ratings and technical levels. For indices, sectors, commodities and currencies both "
    "go into the same Market/* label.",
    "When a regulator acts on a company, the company gets Corporate/Legal & Regulatory and the regulator gets "
    "Economy/Regulation & Government Policy.",
    "A brokerage, analyst or research house that is only quoted as the source of a view about something else "
    "gets Other.",
    "Use Other only when nothing else fits. Confidence: high = clear-cut, medium = two labels plausible, "
    "low = guess.",
)

# Few-shot examples. SEntFiN ones are from its *train* split; the rest are invented.
# tests/test_taxonomy.py checks none of them occurs in a valid/test split.
EXAMPLES: tuple[tuple[str, dict[str, str]], ...] = (
    ("MMTC Q2 net loss at Rs 10.4 crore", {"MMTC": "Corporate/Financial Results"}),
    ("Pegasystems Inc acquires MeshLabs Software",
     {"Pegasystems Inc": "Corporate/M&A & Stakes", "MeshLabs Software": "Corporate/M&A & Stakes"}),
    ("Hudco raises Rs 279 cr via tax-free bonds", {"Hudco": "Corporate/Capital & Dividends"}),
    ("Genpact appoints Edward J Fitzpatrick as CFO", {"Genpact": "Corporate/Management & Governance"}),
    ("SEBI slaps Rs 8 lakh penalty on 2 promoters of Blue Blends",
     {"Blue Blends": "Corporate/Legal & Regulatory", "SEBI": "Economy/Regulation & Government Policy"}),
    ("Voltas surges on cooling products-led beat in earnings", {"Voltas": "Corporate/Financial Results"}),
    ("Syngenta surges as European shares edge higher",
     {"Syngenta": "Stock/Price Movement", "European shares": "Market/Equity Market"}),
    ("4200 is key level for Titan: Ashwani Gujral", {"Titan": "Stock/Outlook & Recommendations"}),
    ("Wonderla Holidays IPO subscribed 42% on Day 1", {"Wonderla Holidays": "Stock/IPO & Listing"}),
    ("Bharat Forge to be included in Nifty 50 from April",
     {"Bharat Forge": "Stock/Trading Activity & Ownership", "Nifty 50": "Market/Equity Market"}),
    ("Be tactful on IT stocks: Deven Choksey", {"IT stocks": "Market/Sectors"}),
    ("Gold trades nearly flat on dull demand", {"Gold": "Market/Commodities"}),
    ("Rupee ends little changed; fund flows in focus", {"Rupee": "Market/Currencies"}),
    ("PSU bonds match equity returns in fourth quarter",
     {"PSU bonds": "Market/Bonds & Rates", "equity": "Market/Equity Market"}),
    ("Mid-cap funds can deliver more, stay put: Experts", {"Mid-cap funds": "Market/Funds & Flows"}),
    ("What to expect from RBI policy tomorrow", {"RBI": "Economy/Monetary Policy"}),
    ("India's GDP growth slows to 5.7% in April-June quarter", {"India": "Economy/Macroeconomy & Trade"}),
    ("All FIIs trying to create some alpha in market: Prateek Agarwal, ASK Investment Manager",
     {"FIIs": "Market/Funds & Flows", "ASK Investment Manager": "Other"}),
    ("Carmaker recalls 1.3 lakh vehicles over faulty airbags; Maruti shares slip",
     {"Maruti": "Corporate/Risks & Reputation"}),
    ("$TSLA adding to my long here, looks ready to break out", {"TSLA": "Stock/Outlook & Recommendations"}),
)

# FiQA gold aspect path prefix -> unified label, used only to sanity-check the LLM
# labels against FiQA's annotators. Only prefixes whose FiQA examples consistently
# mean one unified category are mapped; the rest (Rumors, Company Communication,
# Market/*, Stock/Fundamentals, Stock/Buyside, Sales/Deal, Bullish/Bearish Behavior,
# generic Strategy and Risks, ...) mix several meanings and are left unmapped
# (excluded from the comparison). Checked against the disagreements of the
# first pilot run.
FIQA_PREFIX_MAP: tuple[tuple[str, str | None], ...] = (
    ("Corporate/Appointment", "Corporate/Management & Governance"),
    ("Corporate/Dividend Policy", "Corporate/Capital & Dividends"),
    ("Corporate/Financial/Bankruptcy", "Corporate/Risks & Reputation"),
    ("Corporate/Financial/Financial Results", "Corporate/Financial Results"),
    ("Corporate/Financial/Accounting", "Corporate/Financial Results"),
    ("Corporate/Legal", "Corporate/Legal & Regulatory"),
    ("Corporate/Regulatory", "Corporate/Legal & Regulatory"),
    ("Corporate/M&A", "Corporate/M&A & Stakes"),
    ("Corporate/Strategy/Divestment", "Corporate/M&A & Stakes"),
    ("Corporate/Strategy/Buyout", "Corporate/M&A & Stakes"),
    ("Corporate/Strategy/Share Buyback", "Corporate/Capital & Dividends"),
    ("Corporate/Strategy/New Product", "Corporate/Strategy & Operations"),
    ("Corporate/Strategy/Corporate Expansion", "Corporate/Strategy & Operations"),
    ("Corporate/Strategy/Business Outlook", "Corporate/Strategy & Operations"),
    ("Corporate/Strategy/Business Model", "Corporate/Strategy & Operations"),
    ("Corporate/Strategy/Company Agreement", "Corporate/Strategy & Operations"),
    # "Sales/Deal" mixes acquisitions, asset sales and contracts -> not mapped.
    ("Corporate/Sales/Partnership", "Corporate/Strategy & Operations"),
    ("Corporate/Sales/Contract Withdrawn", "Corporate/Strategy & Operations"),
    ("Corporate/Sales/Failed Contract Discussion", "Corporate/Strategy & Operations"),
    ("Corporate/Sales/Sales", "Corporate/Financial Results"),
    # Generic "Corporate/Risks" is a grab bag; only specific risk types are mapped.
    ("Corporate/Risks/Product Recall", "Corporate/Risks & Reputation"),
    ("Corporate/Risks/Product Flaws", "Corporate/Risks & Reputation"),
    ("Corporate/Risks/Hacks", "Corporate/Risks & Reputation"),
    ("Corporate/Risks/Halt in Services", "Corporate/Risks & Reputation"),
    ("Corporate/Risks/Sale Halt", "Corporate/Risks & Reputation"),
    ("Stock/Price Action/Future Price", "Stock/Outlook & Recommendations"),
    ("Stock/Price Action/Bullish/Bull Position", "Stock/Outlook & Recommendations"),
    ("Stock/Price Action/Bearish/Bear Position", "Stock/Outlook & Recommendations"),
    # "Bullish/Bearish Behavior" mixes reported moves with chart analysis -> not mapped.
    ("Stock/Price Action/Current Price/52-Week High", "Stock/Price Movement"),
    ("Stock/Price Action/Current Price/Stock Price", "Stock/Price Movement"),
    ("Stock/Price Action/Current Price/Oversold", "Stock/Outlook & Recommendations"),
    ("Stock/Price Action/Current Price/Overbought", "Stock/Outlook & Recommendations"),
    ("Stock/Price Action/Volatility/Sell-Off", "Stock/Price Movement"),
    ("Stock/Price Action/Volatility/Stock Volatility", "Stock/Price Movement"),
    ("Stock/Price Action/Volatility/Short Selling", "Stock/Trading Activity & Ownership"),
    ("Stock/Price Action/Volume", "Stock/Trading Activity & Ownership"),
    ("Stock/Technical Analysis", "Stock/Outlook & Recommendations"),
    ("Stock/Coverage/AnalystRatings", "Stock/Outlook & Recommendations"),
    ("Stock/Signal", "Stock/Outlook & Recommendations"),
    ("Stock/Options", "Stock/Trading Activity & Ownership"),
    ("Stock/Insider Activity", "Stock/Trading Activity & Ownership"),
    ("Stock/Fundamentals/Short Interest Rate", "Stock/Trading Activity & Ownership"),
    ("Stock/Fundamentals/Share Buyback", "Corporate/Capital & Dividends"),
    ("Stock/IPO", "Stock/IPO & Listing"),
    ("Economy/Central Banks", "Economy/Monetary Policy"),
    ("Economy/Trade", "Economy/Macroeconomy & Trade"),
)


def map_fiqa_aspect(path: str | None) -> str | None:
    """Unified label for a (normalized) FiQA aspect path, or None if ambiguous."""
    if not path:
        return None
    best = None
    for prefix, label in FIQA_PREFIX_MAP:
        if (path == prefix or path.startswith(prefix + "/")) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, label)
    return best[1] if best else None


def guidelines_markdown() -> str:
    """Human-readable guidelines; identical content to what the LLM is told."""
    out = [f"# Aspect annotation guidelines (taxonomy {TAXONOMY_VERSION})\n",
           "Assign exactly one aspect category to each (text, target) pair: the topic of the text "
           "**as it concerns that target**.\n", "## Categories\n"]
    current = None
    for c in CATEGORIES:
        top = level1(c.label)
        if top != current:
            out.append(f"\n### {top}\n")
            current = top
        out.append(f"- **{c.label}**: {c.definition}")
    out.append("\n## Decision rules\n")
    out += [f"{i}. {r}" for i, r in enumerate(RULES, 1)]
    out.append("\n## Examples\n")
    for text, labels in EXAMPLES:
        out.append(f"- *{text}*")
        out += [f"  - {t} → {a}" for t, a in labels.items()]
    return "\n".join(out) + "\n"
