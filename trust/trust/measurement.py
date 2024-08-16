import rclpy
from avstack_bridge.geometry import GeometryBridge
from avstack_bridge.tracks import TrackBridge
from avstack_bridge.transform import do_transform_boxtrack
from avstack_msgs.msg import BoxTrackArray
from geometry_msgs.msg import PolygonStamped
from mate.measurement import ViewBasedPsm
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from trust_msgs.msg import PsmArray, TrustArray

from .bridge import TrustBridge


class TrustMeasurement(Node):
    def __init__(self, verbose: bool = False):
        super().__init__("trust_psm")
        self.verbose = verbose
        self.declare_parameter("n_agents", 4)
        self.n_agents = self.get_parameter("n_agents").value

        # initialize model
        self.model = ViewBasedPsm(assign_radius=1)

        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )

        # listen to transform information
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, qos=qos)

        # listen to tracks from agents and cc
        self.subscriber_trks = {
            agent_ID: Subscriber(
                self,
                BoxTrackArray,
                f"/agent{agent_ID}/tracks_3d",
                qos_profile=qos,
            )
            for agent_ID in range(self.n_agents)
        }
        self.subscriber_trks["command_center"] = Subscriber(
            self,
            BoxTrackArray,
            "/command_center/tracks_3d",
            qos_profile=qos,
        )

        # listen to fov from agents
        self.subscriber_fovs = {
            agent_ID: Subscriber(
                self,
                PolygonStamped,
                f"/agent{agent_ID}/fov",
                qos_profile=qos,
            )
            for agent_ID in range(self.n_agents)
        }

        # listen to trust messages
        self.subscriber_agent_trust = Subscriber(
            self,
            TrustArray,
            "agent_trust",
            qos_profile=qos,
        )
        self.subscriber_track_trust = Subscriber(
            self,
            TrustArray,
            "track_trust",
            qos_profile=qos,
        )

        # synchronize track messages
        self.synchronizer_trks = ApproximateTimeSynchronizer(
            tuple(self.subscriber_trks.values())
            + tuple(self.subscriber_fovs.values())
            + (self.subscriber_agent_trust, self.subscriber_track_trust),
            queue_size=10,
            slop=0.1,
        )
        self.synchronizer_trks.registerCallback(self.trks_fov_receive)

        # publish PSM messages
        self.publisher_agent_psms = self.create_publisher(
            PsmArray,
            "agent_psms",
            qos_profile=qos,
        )
        self.publisher_track_psms = self.create_publisher(
            PsmArray,
            "track_psms",
            qos_profile=qos,
        )

    def trks_fov_receive(self, *args):
        """Receive approximately synchronized tracks and fovs

        Since we set a dynamic number of agents, we have to use star input
        """
        if self.verbose:
            self.get_logger().info(f"Received {len(args)} track/fov messages!")

        ###################################################
        # Store messages
        ###################################################
        # set up the data structures -- assume things come in order
        # store track messages
        tracks_agents = {}
        for i_agent, msg in enumerate(args[: self.n_agents + 1]):  # first are track messages
            if i_agent < (self.n_agents):
                # convert to global reference frame
                if msg.header.frame_id != "world":
                    tf_world_trk = self.tf_buffer.lookup_transform(
                        "world",
                        msg.header.frame_id,
                        msg.header.stamp,
                    )
                    msg.tracks = [
                        do_transform_boxtrack(trk, tf_world_trk) for trk in msg.tracks
                    ]
                    msg.header = tf_world_trk.header
                tracks_agents[i_agent] = TrackBridge.tracks_to_avstack(msg)
            else:
                tracks_cc = TrackBridge.tracks_to_avstack(msg)

        # store FOV and pose messages
        fov_agents = {}
        position_agents = {}
        for i_agent, msg in enumerate(
            args[self.n_agents + 1 : 2 * self.n_agents + 1]
        ):  # second are fov messages
            # FOV
            agent_frame_id = f"agent{i_agent}"
            fov_agents[i_agent] = GeometryBridge.polygon_to_avstack(msg)
            if msg.header.frame_id != "world":
                raise NotImplementedError("Need to convert to global here")
            # pose
            tf_world_agent = self.tf_buffer.lookup_transform(
                "world",
                agent_frame_id,
                msg.header.stamp,
            )
            position_agents[i_agent] = GeometryBridge.position_to_avstack(
                tf_world_agent.transform.translation,
                header=msg.header,
            )

        # store trust messages
        agent_trust_msg = args[2 * self.n_agents + 1]
        track_trust_msg = args[2 * self.n_agents + 2]
        trust_agents = TrustBridge.ros_to_trust_array(agent_trust_msg)
        trust_tracks = TrustBridge.ros_to_trust_array(track_trust_msg)

        ###################################################
        # Run PSM generation models
        ###################################################

        # psms for agents and tracks
        psms_agents = self.model.psm_agents(
            fov_agents=fov_agents,
            tracks_agents=tracks_agents,
            tracks_cc=tracks_cc,
            trust_tracks=trust_tracks,
        )
        psms_tracks = self.model.psm_tracks(
            position_agents=position_agents,
            fov_agents=fov_agents,
            tracks_agents=tracks_agents,
            tracks_cc=tracks_cc,
            trust_agents=trust_agents,
        )

        # publish outputs
        psms_agents_msg = TrustBridge.psm_array_to_ros(psms_agents)
        psms_tracks_msg = TrustBridge.psm_array_to_ros(psms_tracks)
        self.publisher_agent_psms.publish(psms_agents_msg)
        self.publisher_track_psms.publish(psms_tracks_msg)


def main(args=None):
    rclpy.init(args=args)

    trust_measurement = TrustMeasurement()

    rclpy.spin(trust_measurement)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    trust_measurement.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
