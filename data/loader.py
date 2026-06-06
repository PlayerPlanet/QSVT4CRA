"""
Real data loader stub for future StatFin / Eurostat / Bank of Finland API hookup.

Planned data sources
-------------------
- **StatFin** (stat.fi): Finnish apartment price index, mortgage interest rate,
  household debt-to-income ratio, regional price indices.
- **Eurostat**: Regional unemployment rates, construction permits issued,
  household sector credit, NPL ratios for Finnish banks.
- **Bank of Finland** (suomenpankki.fi): Household credit stock, NPL ratios,
  interest rate statistics, credit register microdata (if accessible).

Attributes
----------
features : float32[N, 4]
    Per-loan feature matrix with columns [PD, LGD, EAD, maturity].
defaults : bool[N]
    Boolean default indicators per loan.
lgd : float32[K]
    Loss-given-default values (unique, used for aggregation).

Notes
-----
This stub raises NotImplementedError on fetch(). Real API integration is
planned as a Phase 1 extension (see architecture.md §D6).
"""
from __future__ import annotations

from typing import Optional


class RealDataLoader:
    """
    Loader for real Finnish apartment loan data.

    Currently a stub; fetch() raises NotImplementedError with clear TODO.

    Parameters
    ----------
    source : str, default "statfin"
        Data source identifier. Reserved for future API selection.
    """

    def __init__(self, source: str = "statfin") -> None:
        self.source = source

    def fetch(self, start: str = "2010", end: str = "2024") -> tuple:
        """
        Fetch real Finnish housing / loan data from the configured source.

        Parameters
        ----------
        start : str, default "2010"
            Start date / year (ISO format or YYYY).
        end : str, default "2024"
            End date / year (ISO format or YYYY).

        Returns
        -------
        tuple
            (features, defaults, lgd) as described in module docstring.

        Raises
        ------
        NotImplementedError
            This stub is not yet implemented.
        """
        raise NotImplementedError(
            "RealDataLoader.fetch() is not yet implemented.\n"
            "\n"
            "TODO (Phase 1 extension):\n"
            "  1. StatFin API (stat.fi) — apartment price index, mortgage rate,\n"
            "     household debt. Requires: register access request + API key.\n"
            "  2. Eurostat — regional unemployment, construction permits.\n"
            "     Endpoint: https://ec.europa.eu/eurostat/api/dissemination\n"
            "  3. Bank of Finland — household credit, NPL ratios.\n"
            "     Endpoint: https://www.suomenpankki.fi/en/statistics/\n"
            "\n"
            "Until real data access is confirmed, use SyntheticPortfolioGenerator\n"
            "as the primary data source (data/synthetic.py)."
        )
