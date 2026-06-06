"""
Generate region-level credit-risk dataset for the QSVT4CRA mortgage
experiment (K = 17 mainland regions of Finland).

Data sources (public, fetched live via the Statistics Finland PxWeb API):

  1. Table 157w -- Indebtedness, "Housing loan debts" by region, 2024
     https://pxdata.stat.fi/PXWeb/pxweb/en/StatFin/StatFin__velk/
     statfin_velk_pxt_157w.px
     Used for: number of indebted households, share of indebted
     households, mean debt, mean interest, all per region.

  2. Table 13al -- Labour Force Survey, regional unemployment rate 2024
     https://pxdata.stat.fi/PXWeb/pxweb/en/StatFin/StatFin__tyti/
     statfin_tyti_pxt_13al.px
     Used for: regional unemployment rate (macroeconomic factor).

External assumptions (documented, conservative):

  * PD base:  0.015 (1.5% annual default rate)  - representative of
    Finnish residential mortgage NPL ratios reported in
    EBA Risk Dashboard (FI residential NPL ~0.5-1.5% over 2022-2024)
    and FIN-FSA "Disclosure requirement for mortgage credit bank
    operations" disclosures.
  * LGD rate: 0.20  - EBA EU-wide stress test baseline residential
    mortgage LGD; FIN-FSA mortgage disclosures show realised LGDs in
    the 10-25% range for Finnish residential mortgages with full
    recourse and conservative LTVs.
  * rho:      0.09 (constant) - value used in the QSVT4CRA paper.
  * F_values: 2 latent factors; loadings derived from standardized
    regional unemployment and standardized regional mean debt
    (z-scores), then scaled so that sum(F^2) < 0.5 for every region
    (the conditional-PD constraint from the GCI model).

Output: writes Code/dataset_regions.py with four Python lists in the
exact format expected by cell 2 of CRA_QSVT.ipynb.
"""

from __future__ import annotations

import json
import math
import statistics
import urllib.request
from pathlib import Path


# ------------------------------------------------------------------ config --

# 17 mainland regions (excludes MK16 Central Ostrobothnia, smallest
# mainland region by population ~68k, and MK21 Åland).
REGIONS = [
    ("MK01", "Uusimaa"),
    ("MK02", "Southwest Finland"),
    ("MK04", "Satakunta"),
    ("MK05", "Kanta-Häme"),
    ("MK06", "Pirkanmaa"),
    ("MK07", "Päijät-Häme"),
    ("MK08", "Kymenlaakso"),
    ("MK09", "South Karelia"),
    ("MK10", "South Savo"),
    ("MK11", "North Savo"),
    ("MK12", "North Karelia"),
    ("MK13", "Central Finland"),
    ("MK14", "South Ostrobothnia"),
    ("MK15", "Ostrobothnia"),
    ("MK17", "North Ostrobothnia"),
    ("MK18", "Kainuu"),
    ("MK19", "Lapland"),
]
K = len(REGIONS)
YEAR = "2024"

# Credit-risk assumptions (see module docstring for sources)
PD_BASE = 0.015        # 1.5% national annual default probability
LGD_RATE = 0.20        # 20% loss given default
RHO = 0.09             # sensitivity to latent factor (paper default)
F_SCALE = 0.30         # max allowed sqrt(sum(F^2)) per region (kept <1)


# --------------------------------------------------------------- PxWeb API --

def pxweb_query(path: str, body: dict) -> list[dict]:
    """POST a PxWeb query, return the list of data rows."""
    url = f"https://pxdata.stat.fi/PXWeb/api/v1/en/{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["data"]


def fetch_157w() -> dict[str, dict[str, float]]:
    """Return per-region dict with: n_indebted, share_pct, mean_debt, mean_interest."""
    body = {
        "query": [
            {"code": "Alue", "selection": {"filter": "item",
                                            "values": [c for c, _ in REGIONS]}},
            {"code": "Asuntokunnan rakenne", "selection": {"filter": "item",
                                                           "values": ["SSS"]}},
            {"code": "Velkatyyppi", "selection": {"filter": "item",
                                                  "values": ["100"]}},  # Housing loans
            {"code": "Vuosi", "selection": {"filter": "item", "values": [YEAR]}},
            {"code": "Tiedot", "selection": {"filter": "item", "values": [
                "velkas_lkm", "velall_pros", "velat_mean", "korot_mean",
            ]}},
        ],
        "response": {"format": "json"},
    }
    rows = pxweb_query("StatFin/velk/statfin_velk_pxt_157w.px", body)
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        region = row["key"][0]
        vals = row["values"]
        out[region] = {
            "n_indebted": float(vals[0]),
            "share_pct":  float(vals[1]),
            "mean_debt":  float(vals[2]),   # EUR per housing-debt household
            "mean_int":   float(vals[3]),   # EUR/year per indebted household
        }
    return out


def fetch_unemployment() -> dict[str, float]:
    """Return per-region unemployment rate (%)."""
    body = {
        "query": [
            {"code": "Vuosi", "selection": {"filter": "item", "values": [YEAR]}},
            {"code": "Maakunta", "selection": {"filter": "item",
                                                "values": [c for c, _ in REGIONS]}},
            {"code": "Tiedot", "selection": {"filter": "item",
                                             "values": ["Tyottomyysaste"]}},
        ],
        "response": {"format": "json"},
    }
    rows = pxweb_query("StatFin/tyti/statfin_tyti_pxt_13al.px", body)
    out: dict[str, float] = {}
    for row in rows:
        region = row["key"][1]
        v = row["values"][0]
        out[region] = float(v) if v != "." else float("nan")
    return out


# ----------------------------------------------------- parameter derivation --

def zscore(xs: list[float]) -> list[float]:
    mu = statistics.mean(xs)
    sd = statistics.pstdev(xs)
    return [(x - mu) / sd if sd > 0 else 0.0 for x in xs]


def compute_parameters(
    debt_data: dict[str, dict[str, float]],
    unemp: dict[str, float],
) -> dict:
    # Fill missing Kainuu unemployment with the national mean of the others
    available = [v for v in unemp.values() if not math.isnan(v)]
    unemp_filled = {r: (v if not math.isnan(v) else statistics.mean(available))
                    for r, v in unemp.items()}

    unemp_z = zscore([unemp_filled[c] for c, _ in REGIONS])
    debt_z  = zscore([debt_data[c]["mean_debt"] for c, _ in REGIONS])

    # Normalize the F_values so max sqrt(sum(F^2)) <= F_SCALE
    # (keeps the conditional PD well-defined for every region)
    norms = [math.hypot(u, d) for u, d in zip(unemp_z, debt_z)]
    max_norm = max(norms) or 1.0
    k_scale = F_SCALE / max_norm

    p_zeros, rhos, lgds, f_vals = [], [], [], []
    for (code, _), uz, dz in zip(REGIONS, unemp_z, debt_z):
        d = debt_data[code]
        u = unemp_filled[code]

        # --- PD: Vasicek-style regionalization of the national base rate ---
        unemp_national = statistics.mean(available)
        beta_unemp = 0.60    # sensitivity: 60% of the unemployment deviation
                             # passes through to the PD (Basel-style)
        pd = PD_BASE * (1.0 + beta_unemp * (u - unemp_national) / unemp_national)
        pd = max(0.0005, min(pd, 0.10))   # clip to [0.05%, 10%]
        p_zeros.append(round(pd, 5))

        # --- rho: constant (paper default 0.09) ---
        rhos.append(RHO)

        # --- LGD (EUR): total regional housing-loan loss exposure ---
        # = mean housing debt per indebted household
        #   * number of housing-indebted households
        #   * LGD rate
        lgd_eur = d["mean_debt"] * d["n_indebted"] * LGD_RATE
        lgds.append(round(lgd_eur, 2))

        # --- F_values: standardized factor loadings ---
        f_vals.append([round(uz * k_scale, 4), round(dz * k_scale, 4)])

    return {
        "regions": [name for _, name in REGIONS],
        "codes":   [code for code, _ in REGIONS],
        "p_zeros": p_zeros,
        "rhos":    rhos,
        "lgd":     lgds,
        "F_values": f_vals,
        "unemp_filled": unemp_filled,
        "debt_data":   debt_data,
    }


# ------------------------------------------------------------ emit module --

def emit_module(out: dict, path: Path) -> None:
    regions = out["regions"]
    codes   = out["codes"]
    lines = [
        "\"\"\"",
        "Region-level credit-risk dataset for the QSVT4CRA mortgage experiment.",
        f"K = {len(regions)} mainland regions of Finland, anchor year {YEAR}.",
        "",
        "Generated by generate_dataset.py from public Statistics Finland data.",
        "Do not edit by hand; re-run the generator to refresh.",
        "",
        "Sources and assumptions:",
        "  - Indebtedness (157w)         pxdata.stat.fi StatFin/velk/157w",
        "  - Unemployment  (13al)        pxdata.stat.fi StatFin/tyti/13al",
        f"  - PD baseline {PD_BASE}        EBA Risk Dashboard + FIN-FSA mortgage",
        f"  - LGD rate    {LGD_RATE}         EBA EU-wide stress test (residential)",
        f"  - rho         {RHO}        QSVT4CRA paper default",
        f"  - F_values    standardized regional unemployment & mean debt,",
        f"                  scaled so max sqrt(sum(F^2)) = {F_SCALE}",
        "\"\"\"",
        "",
        f"K = {len(regions)}",
        f"n_z = 2",
        f"z_max = 2",
        "",
        f"regions = {regions!r}",
        f"region_codes = {codes!r}",
        "",
        f"p_zeros = {out['p_zeros']}",
        f"rhos    = {out['rhos']}",
        f"lgd     = {out['lgd']}",
        f"F_values = {out['F_values']}",
        "",
        f"unemployment_rate_2024_pct = {out['unemp_filled']!r}",
        "",
        "mean_debt_eur = {",
    ]
    for code, name in zip(codes, regions):
        d = out["debt_data"][code]
        lines.append(
            f'    "{code}": {{"name": {name!r}, '
            f'"n_indebted": {int(d["n_indebted"])}, '
            f'"mean_debt_eur": {d["mean_debt"]}, '
            f'"mean_interest_eur": {d["mean_int"]}, '
            f'"share_indebted_pct": {d["share_pct"]}}},'
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------- main ----

def main() -> None:
    print(f"Fetching 157w (indebtedness) for {K} regions, year {YEAR}...")
    debt = fetch_157w()
    print(f"  got data for {len(debt)} regions")

    print("Fetching 13al (unemployment) for the same regions...")
    unemp = fetch_unemployment()
    print(f"  got data for {len(unemp)} regions "
          f"({sum(1 for v in unemp.values() if math.isnan(v))} missing)")

    out = compute_parameters(debt, unemp)

    out_path = Path(__file__).resolve().parent / "dataset_regions.py"
    emit_module(out, out_path)
    print(f"\nWrote {out_path}")
    print(f"K = {K}, sum(lgd) = {sum(out['lgd']):.0f} EUR")
    print("p_zeros:", [f"{x:.4f}" for x in out["p_zeros"]])
    print("lgd (M EUR):", [f"{x/1e6:.2f}" for x in out["lgd"]])


if __name__ == "__main__":
    main()
