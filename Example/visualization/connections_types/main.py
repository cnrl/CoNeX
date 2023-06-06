import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from conex.nn.connections import *

CONNECTION_COLOR = {
    "Apical": "orange",
    "Proximal": "green",
    "Distal": "purple",
    "NoConnection": "Black",
}


def make_handle():
    apical = mpatches.Patch(color=CONNECTION_COLOR["Apical"], label="Apical")
    proximal = mpatches.Patch(color=CONNECTION_COLOR["Proximal"], label="Proximal")
    distal = mpatches.Patch(color=CONNECTION_COLOR["Distal"], label="Distal")
    exc = mpatches.Patch(color="blue", label="Excitatory")
    inh = mpatches.Patch(color="red", label="Inhibitory")
    return [exc, inh, proximal, apical, distal]


def make_layer(name, bottom_left, width, height):
    radius = height / 4
    layer = mpatches.Rectangle(bottom_left, width, height, fc=(1, 1, 1, 0), ec="black")
    start_x, start_y = bottom_left
    width_sep = (width - 4 * radius) / 3
    height_sep = (height - 2 * radius) / 2
    exc = mpatches.Circle(
        (start_x + width_sep + radius, start_y + height_sep + radius),
        radius=radius,
        fc=(1, 1, 1, 0),
        ec="blue",
    )
    inh = mpatches.Circle(
        (start_x + 2 * width_sep + 3 * radius, start_y + height_sep + radius),
        radius=radius,
        fc=(1, 1, 1, 0),
        ec="red",
    )
    offset_self_connection = radius * math.sqrt(3) / 2
    exc_exc = mpatches.FancyArrowPatch(
        posA=(
            start_x + width_sep + radius - offset_self_connection,
            start_y + height_sep + radius + (radius / 2),
        ),
        posB=(
            start_x + width_sep + radius - offset_self_connection,
            start_y + height_sep + radius - (radius / 2),
        ),
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
        connectionstyle="arc3, rad=2",
        fc=(1, 1, 1, 0),
        ec=CONNECTION_COLOR[LAYER_CONNECTION_TYPE[("exc", "exc")]],
    )
    inh_inh = mpatches.FancyArrowPatch(
        posA=(
            start_x + 2 * width_sep + 3 * radius + offset_self_connection,
            start_y + height_sep + radius + (radius / 2),
        ),
        posB=(
            start_x + 2 * width_sep + 3 * radius + offset_self_connection,
            start_y + height_sep + radius - (radius / 2),
        ),
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
        connectionstyle="arc3, rad=-2",
        fc=(1, 1, 1, 0),
        ec=CONNECTION_COLOR[LAYER_CONNECTION_TYPE[("inh", "inh")]],
    )
    exc_inh = mpatches.FancyArrowPatch(
        posA=(
            start_x + width_sep + radius + offset_self_connection,
            start_y + height_sep + radius + (radius / 2),
        ),
        posB=(
            start_x + 2 * width_sep + 3 * radius - offset_self_connection,
            start_y + height_sep + radius + (radius / 2),
        ),
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
        connectionstyle="arc3, rad=-0.7",
        fc=(1, 1, 1, 0),
        ec=CONNECTION_COLOR[LAYER_CONNECTION_TYPE[("exc", "inh")]],
    )
    inh_exc = mpatches.FancyArrowPatch(
        posA=(
            start_x + 2 * width_sep + 3 * radius - offset_self_connection,
            start_y + height_sep + radius - (radius / 2),
        ),
        posB=(
            start_x + width_sep + radius + offset_self_connection,
            start_y + height_sep + radius - (radius / 2),
        ),
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
        connectionstyle="arc3, rad=-0.7",
        fc=(1, 1, 1, 0),
        ec=CONNECTION_COLOR[LAYER_CONNECTION_TYPE[("inh", "exc")]],
    )
    layer_text = plt.text(x=start_x + 0.01, y=start_y + 0.01, s=name)
    return {
        "layer": layer,
        "exc": exc,
        "inh": inh,
        "exc_exc": exc_exc,
        "inh_inh": inh_inh,
        "exc_inh": exc_inh,
        "inh_exc": inh_exc,
        "layer_text": layer_text,
    }


def make_cortical_column(name, bottom_left, width, height):
    layer_names = ["L6", "L5", "L4", "L2_3"]
    pop_names = ["exc", "inh"]
    height_step = height / len(layer_names)
    start_x, start_y = bottom_left
    layers = {
        name: make_layer(
            name,
            (start_x, start_y + index * height_step),
            width,
            height=height_step,
        )
        for index, name in enumerate(layer_names)
    }

    cc_text = plt.text(x=start_x + 0.01, y=start_y + height + 0.01, s=name)

    intra_column_synapse = {}
    for i, src in enumerate(layer_names):
        for j, dst in enumerate(layer_names):
            if src != dst:
                for src_type in pop_names:
                    for dst_type in pop_names:
                        if (
                            INTRA_COLUMN_CONNECTION_TYPE[(src, dst, src_type, dst_type)]
                            == no_connection
                        ):
                            continue
                        # src_cen_x, src_cen_y = layers[src][src_type].center
                        # dst_cen_x, dst_cen_y = layers[dst][dst_type].center
                        # dx , dy = dst_cen_x-src_cen_x ,dst_cen_y-src_cen_y
                        # distnat = math.sqrt(dx**2 + dy**2)
                        # dxr = radius*dx/distnat
                        # dyr = radius*dy/distnat
                        # posA=(src_cen_x + dxr, src_cen_y + dyr)
                        # posB=(dst_cen_x - dxr, dst_cen_y - dyr)
                        synapse = mpatches.FancyArrowPatch(
                            posA=layers[src][src_type].center,
                            posB=layers[dst][dst_type].center,
                            arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                            connectionstyle=f"arc3, rad=-0.3",
                            fc=(1, 1, 1, 0),
                            ec=CONNECTION_COLOR[
                                INTRA_COLUMN_CONNECTION_TYPE[
                                    (src, dst, src_type, dst_type)
                                ]
                            ],
                        )
                        intra_column_synapse[(src, dst, src_type, dst_type)] = synapse

    return {
        "layers": layers,
        "intra_column_synapse": intra_column_synapse,
        "cc_text": cc_text,
    }


def make_two_cortical_column(names, bottom_left, width, height, distance):
    cc1 = make_cortical_column(names[0], bottom_left, width, height)
    start_x, start_y = bottom_left
    cc2 = make_cortical_column(
        names[1], (start_x + width + distance, start_y), width, height
    )

    layer_names = ["L6", "L5", "L4", "L2_3"]
    pop_names = ["exc", "inh"]

    inter_column_synapse = {}
    for i, src in enumerate(layer_names):
        for j, dst in enumerate(layer_names):
            for src_type in pop_names:
                for dst_type in pop_names:
                    if (
                        INTER_COLUMN_CONNECTION_TYPE[(src, dst, src_type, dst_type)]
                        == no_connection
                    ):
                        continue
                    synapse = mpatches.FancyArrowPatch(
                        posA=cc1["layers"][src][src_type].center,
                        posB=cc2["layers"][dst][dst_type].center,
                        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                        connectionstyle=f"arc3, rad=-0.3",
                        fc=(1, 1, 1, 0),
                        ec=CONNECTION_COLOR[
                            INTER_COLUMN_CONNECTION_TYPE[(src, dst, src_type, dst_type)]
                        ],
                    )
                    inter_column_synapse[(src, dst, src_type, dst_type)] = synapse
    return {"cc1": cc1, "cc2": cc2, "inter_column_synapse": inter_column_synapse}


if __name__ == "__main__":
    # Single layer
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, value in make_layer("L4", (0.3, 0.3), 0.4, 0.2).items():
        ax.add_artist(value)

    ax.set_axis_off()
    ax.legend(handles=make_handle())
    plt.show()

    # Single column
    fig, ax = plt.subplots(figsize=(8, 8))
    cc = make_cortical_column("CC1", (0.1, 0.1), 0.4, 0.8)

    for layer in ["L2_3", "L4", "L5", "L6"]:
        for patch in cc["layers"][layer].values():
            ax.add_artist(patch)

    for syn in cc["intra_column_synapse"].values():
        ax.add_artist(syn)

    ax.set_axis_off()
    ax.legend(handles=make_handle())
    plt.show()

    # single column showing layer by layer connection:
    for src_layer in ["L2_3", "L4", "L5", "L6"]:
        fig, ax = plt.subplots(figsize=(8, 8))
        cc = make_cortical_column("cc1", (0.1, 0.1), 0.4, 0.8)

        for layer in ["L2_3", "L4", "L5", "L6"]:
            for key, patch in cc["layers"][layer].items():
                if key not in ["layer", "layer_text"]:
                    patch.set_alpha(0.3)
                ax.add_artist(patch)

        for key, syn in cc["intra_column_synapse"].items():
            if key[0] == src_layer:
                ax.add_artist(syn)

        ax.set_axis_off()
        ax.legend(handles=make_handle())
        plt.show()

    # two cortical column
    fig, ax = plt.subplots(figsize=(8, 8))
    ccs = make_two_cortical_column(["CC1", "CC2"], (0.05, 0.1), 0.3, 0.6, 0.1)

    for cc in [ccs["cc1"], ccs["cc2"]]:
        for layer in ["L2_3", "L4", "L5", "L6"]:
            for patch in cc["layers"][layer].values():
                ax.add_artist(patch)

        for syn in cc["intra_column_synapse"].values():
            ax.add_artist(syn)

    for syn in ccs["inter_column_synapse"].values():
        ax.add_artist(syn)

    ax.set_axis_off()
    ax.legend(handles=make_handle())
    plt.show()

    # two cortical column showing layer by layer connection
    for src_layer in ["L2_3", "L4", "L5", "L6"]:
        fig, ax = plt.subplots(figsize=(8, 8))
        ccs = make_two_cortical_column(["CC1", "CC2"], (0.05, 0.1), 0.3, 0.6, 0.1)

        for cc in [ccs["cc1"], ccs["cc2"]]:
            for layer in ["L2_3", "L4", "L5", "L6"]:
                for key, patch in cc["layers"][layer].items():
                    if key not in ["layer", "layer_text"]:
                        patch.set_alpha(0.3)
                    ax.add_artist(patch)

            for syn in cc["intra_column_synapse"].values():
                syn.set_alpha(0.1)
                ax.add_artist(syn)

        for key, syn in ccs["inter_column_synapse"].items():
            if key[0] == src_layer:
                ax.add_artist(syn)

        ax.set_axis_off()
        ax.legend(handles=make_handle())
        plt.show()
