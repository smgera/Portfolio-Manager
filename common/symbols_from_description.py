"""
Utility to extract ticker symbols from descriptions when the Symbol column contains descriptions.
"""

def extract_symbol_from_description(description: str) -> str:
    """Extract ticker symbol from description based on known patterns.
    Args:
        description: String containing the full security description
    Returns:
        Mapped ticker symbol or original description if no pattern matches
    """
    # Known mappings based on the data we've seen
    known_mappings = {
        "CENCORA INC COM": "COR",
        "ELI LILLY &CO COM": "LLY",
        "AMGEN INC": "AMGN",
        "HCA HEALTHCARE INC COM": "HCA",
        "WELLTOWER INC COM": "WELL",
        "REGENERON PHARMACEUTICALS INC": "REGN",
        "LAS VEGAS SANDS CORP COM USD0.001": "LVS",
        "MEDTRONIC PLC": "MDT",
        "EDWARDS LIFESCIENCES CORP": "EW",
        "ONEOK INC COM USD0.01": "OKE",
        "MERCK &CO. INC COM": "MRK",
        "BRITISH AMERICAN TOBACCO LVL II ADR EACH REP 1 ORD GBP0.25 BNY": "BTI",
        "MARRIOTT INTERNATIONAL INC COM USD0.01 CLASS A": "MAR",
        "ISHARES TR GLB INFRASTR ETF": "IGF",
        "DEVON ENERGY CORP NEW": "DVN",
        "SUNCOR ENERGY INC NEW COM ISIN #CA8672241079 SEDOL #B3NB1P2": "SU",
        "APPLE INC": "AAPL",
        "SPDR S&P500 ETF TRUST TRUST UNIT DEPOSITARY RECEIPT": "SPY",
        "S&P 500 EQUITY INDEX": "SPY",
        "Schwab Prime Advantage Money Fund;Inv": "SPAXX",
        "Sprott Gold Miners ETF": "SGDJ",
        "VANECK ETF TRUST JUNIOR GOLD MINE": "GDXJ",
        "VANECK ETF TRUST RARE EARTH AND S": "REMX",
        "NEWMONT CORP COM ISIN #US6516391066 SEDOL #BJYKTV2": "NEM",
        "ISHARES TR MSCI INTL VLU FT": "EFV",
        "iShares MSCI Intl Value Factor ETF": "EFV",
        "ISHARES MSCI INTL VALUE FACTOR ETF": "EFV",
        "ISHARES TR GLB INFRASTR ETF": "IGF",
        "REGENERON PHARMACEUTICALS INC": "REGN",
        "ALPHABET INC": "GOOGL",
        "ROYAL BANK OF CANADA MONTREAL QUE COM NPV ISIN #CA7800871021 SEDOL #2754383": "RY",
        "FIRST TR EXCH TRADED ALPHADEX FD II DEV MKTS EX US ALPHADEX FD": "FDT",
        "J P MORGAN EXCHANGE TRADED FD JPMORGAN INTL VL": "JXI",
        "TARGA RESOURCES CORP": "TRGP",
        "RIO TINTO ADR EACH REP 1 ORD": "RIO",
        "GE HEALTHCARE TECHNOLOGIES INC COMMON STOCK": "GEHC",
        "TEVA PHARMACEUTICAL INDUSTRIES SPON ADS EACH REP 1 ORD SHS": "TEVA",
        "SPROTT PHYSICAL SILVER TRUST USD": "PSLV",
        "SPROTT PHYSICAL SILVER TRUST TRUST UNIT ISIN #CA85207K1075 SEDOL #B5THDS5": "PSLV",
        "ALPHABET INC CAP STK CL C": "GOOG",
        "CUMMINS INC": "CMI",
        "WESTERN DIGITAL CORP- COM": "WDC",
        "CENTENE CORP": "CNC",
        "WORLD GOLD TR SPDR GLD MINIS": "GLDM",
        "Monster Beverage Corp": "MNST",
        "VALE S-A- SPONS ADS REPR 1 COM NPV": "VALE",
        "WARNER BROS DISCOVERY INC COM SER A": "WBD",
        "SPDR GOLD MINISHARES TRUST": "GLDM",
        "COGNIZANT TECHNOLOGY SOLUTIONS CORP COM CL A USD0-01": "CTSH",
        "GENERAL MTRS CO COM": "GM",
        "PGIM ETF TR PGIM ULTRA SH BD": "PRUL",
        "FIDELITY GOVERNMENT CASH RESERVES": "FCASH",
        "FEDEX CORP COM USD0-10": "FDX",
        "ALPHABET INC": "GOOGL"
    }
    
    # Return the mapped symbol if found
    return known_mappings.get(description, description)
