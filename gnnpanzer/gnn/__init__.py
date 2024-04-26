# import gnn

from .gnnm import fit_gnn, predict_gnn, test_gnn
from .gcnm import GCNModel
from .sgcm import SGCModel
from .mlpm import MLPModel
from .graphsagem import GraphSAGEModel
from .gatm import GATModel
from .labelpropagation import LabelPropagation, test_lp
from .gcnguardm import GCNGuard
from .simpgcnm import SimPGCNModel

