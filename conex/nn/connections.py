"""
This file specifies the connection types for different structures.
"""
distal = "Distal"
apical = "Apical"
proximal = "Proximal"
no_connection = "NoConnection"

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
    # from l2_3 to L2_3
    (l2_3, l4, exc, exc): apical,
    (l2_3, l4, exc, inh): proximal,
    (l2_3, l4, inh, inh): no_connection,
    (l2_3, l4, inh, exc): no_connection,
    # to L5
    (l2_3, l5, exc, exc): proximal,
    (l2_3, l5, exc, inh): proximal,
    (l2_3, l5, inh, inh): no_connection,
    (l2_3, l5, inh, exc): proximal,
    # to 6
    (l2_3, l6, exc, exc): no_connection,
    (l2_3, l6, exc, inh): no_connection,
    (l2_3, l6, inh, inh): no_connection,
    (l2_3, l6, inh, exc): no_connection,
    # from l4 to L2_3
    (l4, l2_3, exc, exc): proximal,
    (l4, l2_3, exc, inh): no_connection,
    (l4, l2_3, inh, inh): proximal,
    (l4, l2_3, inh, exc): proximal,
    # to 5
    (l4, l5, exc, exc): apical,
    (l4, l5, exc, inh): no_connection,
    (l4, l5, inh, inh): no_connection,
    (l4, l5, inh, exc): no_connection,
    # to L6
    (l4, l6, exc, exc): no_connection,
    (l4, l6, exc, inh): no_connection,
    (l4, l6, inh, inh): no_connection,
    (l4, l6, inh, exc): no_connection,
    # from l5 to l2_3
    (l5, l2_3, exc, exc): no_connection,
    (l5, l2_3, exc, inh): proximal,
    (l5, l2_3, inh, inh): no_connection,
    (l5, l2_3, inh, exc): no_connection,
    # to L4
    (l5, l4, exc, exc): no_connection,
    (l5, l4, exc, inh): no_connection,
    (l5, l4, inh, inh): no_connection,
    (l5, l4, inh, exc): no_connection,
    # to L6
    (l5, l6, exc, exc): apical,
    (l5, l6, exc, inh): no_connection,
    (l5, l6, inh, inh): no_connection,
    (l5, l6, inh, exc): no_connection,
    # from l6 to L2_3
    (l6, l2_3, exc, exc): no_connection,
    (l6, l2_3, exc, inh): no_connection,
    (l6, l2_3, inh, inh): no_connection,
    (l6, l2_3, inh, exc): no_connection,
    # to L4
    (l6, l4, exc, exc): distal,
    (l6, l4, exc, inh): proximal,
    (l6, l4, inh, inh): no_connection,
    (l6, l4, inh, exc): no_connection,
    # to L5
    (l6, l5, exc, exc): proximal,
    (l6, l5, exc, inh): no_connection,
    (l6, l5, inh, inh): no_connection,
    (l6, l5, inh, exc): no_connection,
}


INTER_COLUMN_CONNECTION_TYPE = {
    # from l2_3 to L2_3
    (l2_3, l2_3, exc, exc): apical,
    (l2_3, l2_3, exc, inh): no_connection,
    (l2_3, l2_3, inh, inh): no_connection,
    (l2_3, l2_3, inh, exc): no_connection,
    # to L4
    (l2_3, l4, exc, exc): distal,
    (l2_3, l4, exc, inh): no_connection,
    (l2_3, l4, inh, inh): no_connection,
    (l2_3, l4, inh, exc): no_connection,
    # to L5
    (l2_3, l5, exc, exc): no_connection,
    (l2_3, l5, exc, inh): no_connection,
    (l2_3, l5, inh, inh): no_connection,
    (l2_3, l5, inh, exc): no_connection,
    # to L6
    (l2_3, l6, exc, exc): no_connection,
    (l2_3, l6, exc, inh): no_connection,
    (l2_3, l6, inh, inh): no_connection,
    (l2_3, l6, inh, exc): no_connection,
    # from l4 to L2_3
    (l4, l2_3, exc, exc): no_connection,
    (l4, l2_3, exc, inh): no_connection,
    (l4, l2_3, inh, inh): no_connection,
    (l4, l2_3, inh, exc): no_connection,
    # to L4
    (l4, l4, exc, exc): no_connection,
    (l4, l4, exc, inh): no_connection,
    (l4, l4, inh, inh): no_connection,
    (l4, l4, inh, exc): no_connection,
    # to L5
    (l4, l5, exc, exc): no_connection,
    (l4, l5, exc, inh): no_connection,
    (l4, l5, inh, inh): no_connection,
    (l4, l5, inh, exc): no_connection,
    # to L6
    (l4, l6, exc, exc): no_connection,
    (l4, l6, exc, inh): no_connection,
    (l4, l6, inh, inh): no_connection,
    (l4, l6, inh, exc): no_connection,
    # from l5 to L2_3
    (l5, l2_3, exc, exc): no_connection,
    (l5, l2_3, exc, inh): no_connection,
    (l5, l2_3, inh, inh): no_connection,
    (l5, l2_3, inh, exc): no_connection,
    # to l4
    (l5, l4, exc, exc): no_connection,
    (l5, l4, exc, inh): no_connection,
    (l5, l4, inh, inh): no_connection,
    (l5, l4, inh, exc): no_connection,
    # to L5
    (l5, l5, exc, exc): apical,
    (l5, l5, exc, inh): no_connection,
    (l5, l5, inh, inh): no_connection,
    (l5, l5, inh, exc): no_connection,
    # to L6
    (l5, l6, exc, exc): distal,
    (l5, l6, exc, inh): no_connection,
    (l5, l6, inh, inh): no_connection,
    (l5, l6, inh, exc): no_connection,
    # from l6 to l2_3
    (l6, l2_3, exc, exc): apical,
    (l6, l2_3, exc, inh): no_connection,
    (l6, l2_3, inh, inh): no_connection,
    (l6, l2_3, inh, exc): no_connection,
    # to L4
    (l6, l4, exc, exc): no_connection,
    (l6, l4, exc, inh): no_connection,
    (l6, l4, inh, inh): no_connection,
    (l6, l4, inh, exc): no_connection,
    # to L5
    (l6, l5, exc, exc): no_connection,
    (l6, l5, exc, inh): no_connection,
    (l6, l5, inh, inh): no_connection,
    (l6, l5, inh, exc): no_connection,
    # to L6
    (l6, l6, exc, exc): no_connection,
    (l6, l6, exc, inh): no_connection,
    (l6, l6, inh, inh): no_connection,
    (l6, l6, inh, exc): no_connection,
}
