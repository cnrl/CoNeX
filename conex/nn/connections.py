"""
This file specifies the connection types for different structures.
"""
distal = "Distal"
apical = "Apical"
proximal = "Proximal"
noconnection = "noconnection"

exc = "exc"
inh = "inh"

l2_3 = "L2_3"
l4 = "L4"
l5 = "L5"
l6 = "L6"

LAYER_CONNECTION_TYPE = {
    (exc, exc): distal,
    (exc, inh): proximal,
    (inh, inh): proximal,
    (inh, exc): proximal,
}

INTRA_COLUMN_CONNECTION_TYPE = {
    # from l2_3
    (l2_3, l4, exc, exc): apical,
    (l2_3, l4, exc, inh): apical,
    (l2_3, l4, inh, inh): apical,
    (l2_3, l4, inh, exc): apical,
    (l2_3, l5, exc, exc): proximal,
    (l2_3, l5, exc, inh): proximal,
    (l2_3, l5, inh, inh): proximal,
    (l2_3, l5, inh, exc): proximal,
    (l2_3, l6, exc, exc): proximal,
    (l2_3, l6, exc, inh): proximal,
    (l2_3, l6, inh, inh): proximal,
    (l2_3, l6, inh, exc): proximal,
    # from l4
    (l4, l2_3, exc, exc): proximal,
    (l4, l2_3, exc, inh): proximal,
    (l4, l2_3, inh, inh): proximal,
    (l4, l2_3, inh, exc): proximal,
    (l4, l5, exc, exc): apical,
    (l4, l5, exc, inh): apical,
    (l4, l5, inh, inh): apical,
    (l4, l5, inh, exc): apical,
    (l4, l6, exc, exc): noconnection,
    (l4, l6, exc, inh): noconnection,
    (l4, l6, inh, inh): noconnection,
    (l4, l6, inh, exc): noconnection,
    # from l5
    (l5, l2_3, exc, exc): apical,
    (l5, l2_3, exc, inh): apical,
    (l5, l2_3, inh, inh): apical,
    (l5, l2_3, inh, exc): apical,
    (l5, l4, exc, exc): apical,
    (l5, l4, exc, inh): apical,
    (l5, l4, inh, inh): apical,
    (l5, l4, inh, exc): apical,
    (l5, l6, exc, exc): proximal,
    (l5, l6, exc, inh): proximal,
    (l5, l6, inh, inh): proximal,
    (l5, l6, inh, exc): proximal,
    # from l6
    (l6, l2_3, exc, exc): noconnection,
    (l6, l2_3, exc, inh): noconnection,
    (l6, l2_3, inh, inh): noconnection,
    (l6, l2_3, inh, exc): noconnection,
    (l6, l4, exc, exc): apical,
    (l6, l4, exc, inh): apical,
    (l6, l4, inh, inh): apical,
    (l6, l4, inh, exc): apical,
    (l6, l5, exc, exc): proximal,
    (l6, l5, exc, inh): proximal,
    (l6, l5, inh, inh): proximal,
    (l6, l5, inh, exc): proximal,
}


INTER_COLUMN_CONNECTION_TYPE = {
    # from l2_3
    (l2_3, l2_3, exc, exc): distal,
    (l2_3, l2_3, exc, inh): distal,
    (l2_3, l2_3, inh, inh): distal,
    (l2_3, l2_3, inh, exc): distal,
    (l2_3, l4, exc, exc): proximal,
    (l2_3, l4, exc, inh): proximal,
    (l2_3, l4, inh, inh): proximal,
    (l2_3, l4, inh, exc): proximal,
    (l2_3, l5, exc, exc): proximal,
    (l2_3, l5, exc, inh): proximal,
    (l2_3, l5, inh, inh): proximal,
    (l2_3, l5, inh, exc): proximal,
    (l2_3, l6, exc, exc): proximal,
    (l2_3, l6, exc, inh): proximal,
    (l2_3, l6, inh, inh): proximal,
    (l2_3, l6, inh, exc): proximal,
    # from l4
    (l4, l2_3, exc, exc): noconnection,
    (l4, l2_3, exc, inh): noconnection,
    (l4, l2_3, inh, inh): noconnection,
    (l4, l2_3, inh, exc): noconnection,
    (l4, l4, exc, exc): distal,
    (l4, l4, exc, inh): distal,
    (l4, l4, inh, inh): distal,
    (l4, l4, inh, exc): distal,
    (l4, l5, exc, exc): noconnection,
    (l4, l5, exc, inh): noconnection,
    (l4, l5, inh, inh): noconnection,
    (l4, l5, inh, exc): noconnection,
    (l4, l6, exc, exc): noconnection,
    (l4, l6, exc, inh): noconnection,
    (l4, l6, inh, inh): noconnection,
    (l4, l6, inh, exc): noconnection,
    # from l5
    (l5, l2_3, exc, exc): apical,
    (l5, l2_3, exc, inh): apical,
    (l5, l2_3, inh, inh): apical,
    (l5, l2_3, inh, exc): apical,
    (l5, l4, exc, exc): apical,
    (l5, l4, exc, inh): apical,
    (l5, l4, inh, inh): apical,
    (l5, l4, inh, exc): apical,
    (l5, l5, exc, exc): distal,
    (l5, l5, exc, inh): distal,
    (l5, l5, inh, inh): distal,
    (l5, l5, inh, exc): distal,
    (l5, l6, exc, exc): distal,
    (l5, l6, exc, inh): distal,
    (l5, l6, inh, inh): distal,
    (l5, l6, inh, exc): distal,
    # from l6
    (l6, l2_3, exc, exc): noconnection,
    (l6, l2_3, exc, inh): noconnection,
    (l6, l2_3, inh, inh): noconnection,
    (l6, l2_3, inh, exc): noconnection,
    (l6, l4, exc, exc): noconnection,
    (l6, l4, exc, inh): noconnection,
    (l6, l4, inh, inh): noconnection,
    (l6, l4, inh, exc): noconnection,
    (l6, l5, exc, exc): distal,
    (l6, l5, exc, inh): distal,
    (l6, l5, inh, inh): distal,
    (l6, l5, inh, exc): distal,
    (l6, l6, exc, exc): distal,
    (l6, l6, exc, inh): distal,
    (l6, l6, inh, inh): distal,
    (l6, l6, inh, exc): distal,
}
