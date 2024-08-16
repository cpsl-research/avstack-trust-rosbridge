import rclpy
from avstack_bridge import Bridge
from avstack_msgs.msg import BoxTrackArray
from mate.updater import TrustUpdater
from rclpy.node import Node

from trust_msgs.msg import PsmArray as PsmArrayRos
from trust_msgs.msg import TrustArray as TrustArrayRos

from .bridge import TrustBridge


class TrustUpdaterNode(Node):
    def __init__(self, verbose: bool = False):
        super().__init__("trust_estimator")
        self.verbose = verbose
        self.declare_parameter("n_agents", 4)
        self.n_agents = self.get_parameter("n_agents").value
        self.trust_pub_rate = 100  # Hz

        # initialize model
        self.model = TrustUpdater()

        # initialize trust for the agents
        agent_ids = list(range(self.n_agents))
        self.model.init_new_agents(timestamp=0.0, agent_ids=agent_ids)

        # qos on messages
        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )

        # listen to tracks from cc
        self.subscriber_trks = self.create_subscription(
            BoxTrackArray,
            "/command_center/tracks_3d",
            self.receive_tracks,
            qos_profile=qos,
        )

        # listen to psms
        self.subscriber_agent_psms = self.create_subscription(
            PsmArrayRos,
            "agent_psms",
            self.receive_agent_psms,
            qos_profile=qos,
        )
        self.subscriber_track_psms = self.create_subscription(
            PsmArrayRos,
            "track_psms",
            self.receive_track_psms,
            qos_profile=qos,
        )

        # publish trusts on a timer
        self.publisher_agent_trust = self.create_publisher(
            TrustArrayRos,
            "agent_trust",
            qos_profile=qos,
        )
        self.publisher_track_trust = self.create_publisher(
            TrustArrayRos,
            "track_trust",
            qos_profile=qos,
        )
        self.timer = self.create_timer(1.0 / self.trust_pub_rate, self.send_trust)

    def receive_tracks(self, msg: BoxTrackArray):
        # initialize new tracks
        timestamp = Bridge.rostime_to_time(msg.header.stamp)
        track_ids = [track.identifier for track in msg.tracks]
        self.model.init_new_tracks(timestamp, track_ids)

        # propagate the track trusts to the current time
        self.model.propagate_track_trust(timestamp)

        # propagate the agent trusts to the current time
        self.model.propagate_agent_trust(timestamp)

    def receive_agent_psms(self, msg: PsmArrayRos):
        psms_agents = TrustBridge.ros_to_psm_array(msg)
        self.model.update_agent_trust(psms_agents)

    def receive_track_psms(self, msg: PsmArrayRos):
        psms_tracks = TrustBridge.ros_to_psm_array(msg)
        self.model.update_track_trust(psms_tracks)

    def send_trust(self):
        # package trust
        agent_trust_msg = TrustBridge.trust_array_to_ros(self.model.agent_trust)
        track_trust_msg = TrustBridge.trust_array_to_ros(self.model.track_trust)

        # send trust
        self.publisher_agent_trust.publish(agent_trust_msg)
        self.publisher_track_trust.publish(track_trust_msg)


def main(args=None):
    rclpy.init(args=args)

    trust_updater = TrustUpdaterNode()

    rclpy.spin(trust_updater)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    trust_updater.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
