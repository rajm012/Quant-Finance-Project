from Algorithms.ctaea import CTAEA
from Algorithms.peer_algorithms import CMOEAD, CNSGAIII, CMOEAD_DD, IDBEA, CMOEA

ALL_ALGORITHMS = {
    'C-TAEA':     CTAEA,
    'C-MOEA/D':   CMOEAD,
    'C-NSGA-III': CNSGAIII,
    'C-MOEA/DD':  CMOEAD_DD,
    'I-DBEA':     IDBEA,
    'CMOEA':      CMOEA,
}
