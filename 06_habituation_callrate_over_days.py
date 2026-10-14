"""
06_habituation_callrate_over_days.py
====================================
Test for habituation: did the number of calls per trial decline over the
course of the experiment? (Reviewer 2, comment 2.20; Methods, section 5.5)

For each colony (2022, 2023), daytime trials only:
    log(calls per trial + 1) ~ condition ("Net" = 1, "No-Net" = 0) + experimental day
fitted by ordinary least squares (statsmodels). The coefficient of experimental
day is converted to a percentage change in calls per day: 100 * (exp(b) - 1).

Trial boundaries: in each sheet of all-vocalizations.xlsx, the calls of a trial
are followed by a marker row whose first cell is the recording filename
(e.g. "20221002t1n(7)BATS.wav"); the first 8 digits give the recording date.
One 2023 trial whose filename date (2023-10-24) lies outside the experimental
period was excluded because its recording date is uncertain.

INPUT : all-vocalizations.xlsx (same folder)
OUTPUT: outputs/habituation/trials_per_day.csv
        outputs/habituation/habituation_regression.csv

DEPENDENCIES: Python 3.9+, numpy, pandas, statsmodels, openpyxl
AUTHORS: Omer Yinon, Yossi Yovel Lab, Tel Aviv University
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

XLSX_PATH = Path(__file__).resolve().parent / "all-vocalizations.xlsx"
OUTDIR = Path(__file__).resolve().parent / "outputs" / "habituation"
OUTDIR.mkdir(parents=True, exist_ok=True)

DAY_SHEETS = {
    "2022": ["day-net(22)", "day-no-net(22)"],
    "2023": ["day-net(23)", "day-no-net(23)"],
}
EXCLUDE_DATES = {"2023-10-24"}  # filename date outside the experimental period


def trials_from_sheet(xls: pd.ExcelFile, sheet: str, year: str) -> list[dict]:
    """Return one record per trial: filename, date and number of calls."""
    col = xls.parse(sheet).iloc[:, 0].astype(str)
    records, n_calls = [], 0
    for val in col:
        val = val.strip()
        if "xxx.wav" in val:
            continue
        if val.startswith(year):  # marker row closing the preceding trial
            m = re.match(r"(\d{4})(\d{2})(\d{2})", val)
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            records.append({"sheet": sheet, "file": val, "date": date, "calls": n_calls})
            n_calls = 0
        else:
            n_calls += 1
    return records


def main():
    xls = pd.ExcelFile(XLSX_PATH)
    rows, results = [], []
    for year, sheets in DAY_SHEETS.items():
        recs = []
        for sh in sheets:
            recs += trials_from_sheet(xls, sh, year)
        d = pd.DataFrame(recs)
        d = d[~d["date"].isin(EXCLUDE_DATES)].copy()
        d["date"] = pd.to_datetime(d["date"])
        d["day"] = (d["date"] - d["date"].min()).dt.days
        d["net"] = (~d["sheet"].str.contains("no-net")).astype(int)
        d["log_calls"] = np.log(d["calls"] + 1)
        d["colony"] = year
        rows.append(d)

        fit = smf.ols("log_calls ~ net + day", data=d).fit()
        b = fit.params["day"]
        lo, hi = fit.conf_int().loc["day"]
        results.append({
            "colony": year,
            "n_trials": len(d),
            "b_day": b,
            "se": fit.bse["day"],
            "t": fit.tvalues["day"],
            "df_resid": fit.df_resid,
            "p": fit.pvalues["day"],
            "pct_change_per_day": 100 * (np.exp(b) - 1),
            "pct_CI_low": 100 * (np.exp(lo) - 1),
            "pct_CI_high": 100 * (np.exp(hi) - 1),
        })

    pd.concat(rows).to_csv(OUTDIR / "trials_per_day.csv", index=False)
    res = pd.DataFrame(results)
    res.to_csv(OUTDIR / "habituation_regression.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
