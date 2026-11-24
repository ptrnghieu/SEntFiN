# Vietnam news: LLM triplet pre-labels

Source: `data/sentences_clean.json` of minhhai205/aspect-based-FSA. Labels: gpt-6-luna (targets: target-completion prompt; sentiment: `s1-712b7e41`; aspect: `v1-6a74660b`). **All labels are silver until reviewed.**

| split | sentences | articles | targets | sentences without target |
|---|---|---|---|---|
| train | 1238 | 302 | 2643 | 9 |
| valid | 141 | 25 | 317 | 1 |
| test | 300 | 65 | 663 | 0 |
| all | 1679 | 392 | 3623 | 10 |

## Sentiment

| value | pairs | share |
|---|---|---|
| neutral | 1924 | 53.1% |
| positive | 1188 | 32.8% |
| negative | 511 | 14.1% |

## Sentiment confidence

| value | pairs | share |
|---|---|---|
| high | 2780 | 76.7% |
| medium | 836 | 23.1% |
| low | 7 | 0.2% |

## Target type

| value | pairs | share |
|---|---|---|
| company | 1725 | 47.6% |
| regulator_government | 610 | 16.8% |
| index_market | 409 | 11.3% |
| sector | 254 | 7.0% |
| country_economy | 180 | 5.0% |
| other | 138 | 3.8% |
| currency | 99 | 2.7% |
| fund_investor_group | 79 | 2.2% |
| bonds_rates | 75 | 2.1% |
| commodity | 54 | 1.5% |

## Aspect, level 1

| value | pairs | share |
|---|---|---|
| Corporate | 1230 | 33.9% |
| Market | 799 | 22.1% |
| Economy | 795 | 21.9% |
| Other | 407 | 11.2% |
| Stock | 392 | 10.8% |

## Aspect

| value | pairs | share |
|---|---|---|
| Corporate/Strategy & Operations | 940 | 25.9% |
| Economy/Regulation & Government Policy | 525 | 14.5% |
| Other | 407 | 11.2% |
| Market/Equity Market | 318 | 8.8% |
| Stock/Price Movement | 318 | 8.8% |
| Market/Sectors | 205 | 5.7% |
| Economy/Macroeconomy & Trade | 144 | 4.0% |
| Corporate/Financial Results | 129 | 3.6% |
| Economy/Monetary Policy | 126 | 3.5% |
| Market/Currencies | 99 | 2.7% |
| Corporate/Legal & Regulatory | 71 | 2.0% |
| Market/Funds & Flows | 65 | 1.8% |
| Market/Bonds & Rates | 65 | 1.8% |
| Stock/Trading Activity & Ownership | 58 | 1.6% |
| Market/Commodities | 47 | 1.3% |
| Corporate/Capital & Dividends | 45 | 1.2% |
| Corporate/M&A & Stakes | 34 | 0.9% |
| Stock/IPO & Listing | 10 | 0.3% |
| Stock/Outlook & Recommendations | 6 | 0.2% |
| Corporate/Management & Governance | 6 | 0.2% |
| Corporate/Risks & Reputation | 5 | 0.1% |

## Most frequent targets

| target | pairs |
|---|---|
| VN-Index | 119 |
| SBV | 83 |
| Vietjet | 79 |
| VPBank | 71 |
| Ministry of Industry and Trade | 69 |
| SABECO | 68 |
| Vinamilk | 60 |
| Việt Nam | 56 |
| Petrovietnam | 49 |
| Ministry of Finance | 48 |
| Vietnam | 41 |
| VIFC-HCMC | 41 |
| Techcombank | 40 |
| Viettel | 35 |
| HoSE | 34 |
| Vietnam Airlines | 33 |
| BIDV | 27 |
| State Bank of Vietnam (SBV) | 26 |
| stock market | 26 |
| VIFC | 25 |
| HDBank | 25 |
| HNX | 25 |
| banks | 24 |
| State Bank of Vietnam | 22 |
| VIB | 22 |
