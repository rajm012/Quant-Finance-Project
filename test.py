
# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================



# import json
# from pathlib import Path

# base = Path("6-Results-CTAEA")
# files = [
#     base / "all_results-3.json",
#     base / "all_results-5.json",
#     base / "all_results-8.json",
#     base / "all_results-10.json",
#     base / "all_results-15.json",
# ]

# combined = {}

# for fp in files:
#     with open(fp, "r") as f:
#         data = json.load(f)

#     for problem, byM in data.items():
#         combined.setdefault(problem, {})
#         for m, algos in byM.items():
#             combined[problem].setdefault(m, {})
#             combined[problem][m].update(algos)

# with open(base / "all_results.json", "w") as f:
#     json.dump(combined, f, indent=2)
    


# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================



# import json
# from pathlib import Path


# BASEDIR = Path(__file__).resolve().parent
# MODELWISEDIR = BASEDIR / "ModelWise"
# PROBLEMWISEDIR = BASEDIR / "ProblemWise"


# def BuildPbmResults() -> None:
#     pbmResults = {}
#     for JsonPath in sorted(MODELWISEDIR.glob("*.json")):
#         with JsonPath.open("r", encoding="utf-8") as handle:
#             modelData = json.load(handle)

#         for problem, byM in modelData.items():
#             pbmEntry = pbmResults.setdefault(problem, {})

#             for m, byModel in byM.items():
#                 mEntry = pbmEntry.setdefault(m, {})
#                 for modelName, metrics in byModel.items():
#                     mEntry[modelName] = metrics

#     PROBLEMWISEDIR.mkdir(parents=True, exist_ok=True)
#     for problem, data in pbmResults.items():
#         outPath = PROBLEMWISEDIR / f"{problem}.json"
#         with outPath.open("w", encoding="utf-8") as handle:
#             json.dump({problem: data}, handle, indent=2)


# if __name__ == "__main__":
#     BuildPbmResults()



# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================


import json
from pathlib import Path
from typing import Dict, Any

BASEDIR = Path(__file__).resolve().parent
PROBLEMWISEDIR = BASEDIR / "ProblemWise"
OUTPUTDIR = BASEDIR / "Reports"


MVALUES = ["3", "5", "8", "10", "15"]
CPROBLEMS = ["C1DTLZ1", "C1DTLZ3", "C2DTLZ2", "C3DTLZ1", "C3DTLZ4"]
DCPROBLEMS = ["DC1DTLZ1", "DC1DTLZ3", "DC2DTLZ1", "DC2DTLZ3", "DC3DTLZ1", "DC3DTLZ3"]
METRICS = {
    "HV": {"key": "hv_median", "iqr_key": "hv_iqr"},
    "IGD": {"key": "igd_median", "iqr_key": "igd_iqr"},
    "Feasibility-Ratio": {"feas_key": "n_runs_feasible", "total_key": "n_runs_total"},
}



def loadPbmData(pbmName: str) -> Dict[str, Any]:
    if pbmName in CPROBLEMS:
        fp = PROBLEMWISEDIR / "C-Class" / f"{pbmName}.json"
    else:
        fp = PROBLEMWISEDIR / "DC-Class" / f"{pbmName}.json"
    
    with fp.open() as f:
        data = json.load(f)
        
    return data[pbmName]




def getAlgo(pbmData: Dict) -> list:
    firstM = next(iter(pbmData.values()))
    return sorted(firstM.keys())



def formatHVIGD(value: float, iqr: float) -> str:
    if value == 0 and iqr == 0:
        return "0"
    if value == float("inf"):
        return "∞"
    
    return f"{value:.3f} ({iqr:.3f})"




def formatFeasible(nFeasible: int, nTot: int) -> str:
    return f"{nFeasible}/{nTot}"




def AlgoToCall(pbmData: Dict, m: str, metric: str, isMax: bool) -> tuple:
    results = []
    for algo, metrics in pbmData[m].items():
        if metric == "HV":
            val = metrics["hv_median"]
            iqr = metrics["hv_iqr"]
        else:
            val = metrics["igd_median"]
            iqr = metrics["igd_iqr"]
        results.append((algo, val, iqr))
    
    # For HV, maximize; for IGD, minimize
    results.sort(key=lambda x: x[1], reverse=isMax)
    return results[0] if results else (None, 0, 0)




def buildTable(problems: list, metric: str) -> str:
    if metric == "Feasibility-Ratio":
        is_maximizing = True
        metric_key = "Feas"
    elif metric == "HV":
        is_maximizing = True
        metric_key = "HV"
    else:
        is_maximizing = False
        metric_key = "IGD"
    
    # Build table header
    lines = [f"# Results by {metric}\n"]
    lines.append("| Problem |" + " | ".join([f"m={m}" for m in MVALUES]) + " |")
    lines.append("|" + "|".join(["---"] * (len(MVALUES) + 1)) + "|")
    

    for problem in problems:
        pbmData = loadPbmData(problem)
        row = [problem]
        for m in MVALUES:
            if m not in pbmData:
                row.append("—")
                continue
            
            if metric == "Feasibility-Ratio":
                bestAlgo, _, _ = AlgoToCall(pbmData, m, "HV", is_maximizing)
                nFeas = pbmData[m][bestAlgo]["n_runs_feasible"]
                nTot = pbmData[m][bestAlgo]["n_runs_total"]
                CellVal = formatFeasible(nFeas, nTot)
            else:
                bestAlgo, val, iqr = AlgoToCall(pbmData, m, metric, is_maximizing)
                CellVal = formatHVIGD(val, iqr)
            
            row.append(CellVal)
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)



def main():
    OUTPUTDIR.mkdir(exist_ok=True)
    reports = [
        ("C-problems [HV]", CPROBLEMS, "HV"),
        ("C-problems [IGD]", CPROBLEMS, "IGD"),
        ("C-problems [Feasibility-Ratio]", CPROBLEMS, "Feasibility-Ratio"),
        ("DC-problems [HV]", DCPROBLEMS, "HV"),
        ("DC-problems [IGD]", DCPROBLEMS, "IGD"),
        ("DC-problems [Feasibility-Ratio]", DCPROBLEMS, "Feasibility-Ratio"),
    ]
    
    for title, problems, metric in reports:
        content = buildTable(problems, metric)
        className = "C" if problems == CPROBLEMS else "DC"
        MetricName = metric.replace("-", "").lower()
        filename = f"{className}-{MetricName}.md"
        fp = OUTPUTDIR / filename
        with fp.open("w") as f:
            f.write(content)
        
        print(f"Created {filename}")



if __name__ == "__main__":
    main()



# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================



