#! /usr/bin/env python3

import os
import threading
from functools import partial

#################################################################
# fmt: off
# this is needed to allow for importing of avstack/bridge things
# while also using matplotlib/pyqt5
import cv2  # noqa # pylint: disable=unused-import


os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
# fmt: on
#################################################################

import matplotlib.animation as anim
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.stats import beta

from trust_msgs.msg import TrustArray as TrustArrayRos

from .bridge import TrustBridge


agent_colors = {
    0: "#f9f06b",
    1: "#cdaB8f",
    2: "#dc8add",
    3: "#99c1f1",
}

track_colors = list(mcolors.TABLEAU_COLORS.keys())
# track_colors = list(mcolors.XKCD_COLORS.keys())


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18


plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("lines", linewidth=4)
plt.rc("grid", linestyle="--", color="black", alpha=0.5)


def get_track_color(ID_track):
    return track_colors[ID_track % len(track_colors)]


def rostime_to_time(msg):
    return float(msg.sec + msg.nanosec / 1e9)


class TrustVisualizer(Node):
    """Trust Visualizer for showing how to use matplotlib within ros 2 node

    Inspiration from: https://github.com/timdodge54/matplotlib_ros_tutorials

    Attributes:
        fig: Figure object for matplotlib
        ax: Axes object for matplotlib
        x: x values for matplotlib
        y: y values for matplotlib
        lock: lock for threading
        _sub: Subscriber for node
    """

    def __init__(self):
        """Initialize."""
        super().__init__("trust_visualizer")

        # Initialize figure and axes and save to class
        self.fig, self.axs = plt.subplots(2, 2, figsize=(20, 10))
        for i, txt in enumerate(["Agent", "Track"]):
            # -- attributes for bar plot
            self.axs[i, 0].set_title(f"{txt} Trust Mean")
            self.axs[i, 0].set_xlim([0, 1])
            self.axs[i, 0].set_xlabel("Mean Trust Value")
            self.axs[i, 0].set_ylabel("Identifier")
            self.axs[i, 0].xaxis.grid()

            # -- attributes for distribution plot
            self.axs[i, 1].set_title(f"{txt} Trust Distributions")
            self.axs[i, 1].set_xlim([0, 1])
            self.axs[i, 1].set_ylim([0, 1])
            self.axs[i, 1].set_xlabel("Trust Value")
            self.axs[i, 1].set_ylabel("PDF")
            self.axs[i, 1].grid()

        plt.tight_layout()

        # x axis on the trust distribution
        npts = 1000
        self._trust_x = np.linspace(0, 1, npts)

        # create Thread lock to prevent multiaccess threading errors
        self._lock = threading.Lock()

        # create initial values to plot
        self.agent_ids_active = set()
        self.agent_trust_data = {}
        self.agent_trust_plot = {"bar": {}, "dist": {}}
        self.track_ids_active = set()
        self.track_trust_data = {}
        self.track_trust_plot = {"bar": {}, "dist": {}}

        # create subscriber
        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )
        self.cbg = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.sub_agent_trust = self.create_subscription(
            TrustArrayRos,
            "agent_trust",
            partial(self.trust_callback, self.agent_trust_data, self.agent_ids_active),
            qos_profile=qos,
            callback_group=self.cbg,
        )
        self.sub_track_trust = self.create_subscription(
            TrustArrayRos,
            "track_trust",
            partial(self.trust_callback, self.track_trust_data, self.track_ids_active),
            qos_profile=qos,
            callback_group=self.cbg,
        )

    def trust_callback(self, datastruct: dict, actives: list, msg: TrustArrayRos):
        """Callback for subscriber

        Args:
            msg: message from subscriber
        """
        # lock thread
        with self._lock:
            trust_array = TrustBridge.ros_to_trust_array(msg)

            # update values
            ids_active = set()
            for trust_id in trust_array:
                trust = trust_array[trust_id]
                # add to active list
                ids_active.add(trust.identifier)
                actives.add(trust.identifier)

                # get the distribution from the beta parameters
                distribution = beta.pdf(self._trust_x, trust.alpha, trust.beta)

                # set the fields
                trust_data = {
                    "timestamp": rostime_to_time(msg.header.stamp),
                    "mean": trust.mean,
                    "variance": trust.variance,
                    "distribution": distribution,
                }
                datastruct[trust.identifier] = trust_data

            # remove inplace anything not active
            ids_remove = set()
            for id_candidate in actives:
                if id_candidate not in ids_active:
                    ids_remove.add(id_candidate)
            for id_remove in ids_remove:
                actives.remove(id_remove)

    def plt_func(self, _, dynamic_ylim: bool = False):
        """Function for for adding data to axis.

        Args:
            _ : Dummy variable that is required for matplotlib animation.
            dynamic_ylim: boolean on whether to enable dynamic ylim adjustment

        Returns:
            Axes object for matplotlib
        """
        # lock thread
        with self._lock:
            plot_all = [self.agent_trust_plot, self.track_trust_plot]
            data_all = [self.agent_trust_data, self.track_trust_data]
            active_all = [self.agent_ids_active, self.track_ids_active]
            for i, (plot, data, active) in enumerate(
                zip(plot_all, data_all, active_all)
            ):
                yticks = []
                ytick_labels = []
                ylims = [0, 1]
                ids_remove = []
                for idx, identifier in enumerate(data):
                    # -- check if still actively publishing data
                    if identifier not in active:
                        ids_remove.append(identifier)
                        plot["bar"][identifier].remove()
                        plot["dist"][identifier].remove()
                        continue

                    # -- attributes
                    # id_int = int(identifier.replace("agent", "").replace("track", ""))
                    id_int = identifier
                    color = (
                        agent_colors[identifier] if i == 0 else get_track_color(id_int)
                    )
                    label = f"Agent {id_int}" if i == 0 else f"Track {id_int}"
                    yticks.append(idx - len(ids_remove))
                    ytick_labels.append(label)

                    # -- update bars
                    y = idx - len(ids_remove)
                    w = data[identifier]["mean"]
                    height = 0.6
                    if identifier not in plot["bar"]:
                        (plot["bar"][identifier],) = self.axs[i, 0].barh(
                            y, w, height=height, left=0.0, color=color, label=label
                        )
                    else:
                        plot["bar"][identifier].set_y(y - height / 2)
                        plot["bar"][identifier].set_width(w)

                    # -- update distributions
                    pdfs = data[identifier]["distribution"]
                    if identifier not in plot["dist"]:
                        (plot["dist"][identifier],) = self.axs[i, 1].plot(
                            self._trust_x,
                            pdfs,
                            color=color,
                            label=label,
                        )
                    else:
                        plot["dist"][identifier].set_ydata(pdfs)

                    # -- set ylim
                    if dynamic_ylim:
                        ylims[1] = max(5, min(20, max(pdfs) + 0.1))
                    else:
                        ylims[1] = 5

                # remove things
                for id_remove in ids_remove:
                    del data[id_remove]

                # update labels and such
                self.axs[i, 0].set_yticks(yticks)
                self.axs[i, 0].set_yticklabels(ytick_labels)
                self.axs[i, 1].set_ylim(ylims)

            return self.axs

    def _plt(self, interval_ms=1000 / 10):
        """Function for initializing and showing matplotlib animation."""
        self.ani = anim.FuncAnimation(self.fig, self.plt_func, interval=interval_ms)
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = TrustVisualizer()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    node._plt()


if __name__ == "__main__":
    main()
