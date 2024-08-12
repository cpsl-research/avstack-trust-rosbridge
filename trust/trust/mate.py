import rclpy
from avstack_bridge import Bridge
from avstack_bridge.geometry import GeometryBridge
from avstack_bridge.tracks import TrackBridge
from avstack_bridge.transform import do_transform_boxtrack
from avstack_msgs.msg import BoxTrackArray
from geometry_msgs.msg import PolygonStamped
from mate.estimator import TrustEstimator
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from std_msgs.msg import Header, String
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from trust_msgs.msg import Trust, TrustArray


class FakeModel:
    def reset(self):
        pass


class MultiAgentTrustEstimator(Node):
    def __init__(self, verbose: bool = True):
        super().__init__("mate")
        self.verbose = verbose
        self.declare_parameter("n_agents", 4)
        self.n_agents = self.get_parameter("n_agents").value

        # innitialize model
        self.model = TrustEstimator()

        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )

        # listen to transform information
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, qos=qos)

        # subscribe to initialization message (optional)
        self.subscriber_init = self.create_subscription(
            String,
            "/initialization",
            self.init_callback,
            qos_profile=qos,
        )

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
            f"/command_center/tracks_3d",
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

        # synchronize track messages
        self.synchronizer_trks = ApproximateTimeSynchronizer(
            tuple(self.subscriber_trks.values()) + tuple(self.subscriber_fovs.values()),
            queue_size=10,
            slop=0.1,
        )
        self.synchronizer_trks.registerCallback(self.trks_fov_receive)

        # publish trust messages
        self.publisher_agent_trust = self.create_publisher(
            TrustArray,
            "agent_trust",
            qos_profile=qos,
        )
        self.publisher_track_trust = self.create_publisher(
            TrustArray,
            "track_trust",
            qos_profile=qos,
        )

    def init_callback(self, init_msg: String) -> None:
        if init_msg.data == "reset":
            self.get_logger().info("Calling reset on trust estimator!")
            self.model.reset()

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
        agent_tracks = {}
        cc_tracks = None
        for i, msg in enumerate(args[: self.n_agents + 1]):  # first are track messages
            if i < (self.n_agents):
                # convert to global reference frame
                agent = f"agent{i}"
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
                agent_tracks[agent] = TrackBridge.tracks_to_avstack(msg)
            else:
                agent = "command_center"
                cc_tracks = TrackBridge.tracks_to_avstack(msg)

        # store FOV and pose messages
        agent_fovs = {}
        agent_poses = {}
        for i, msg in enumerate(args[self.n_agents + 1 :]):  # second are fov messages
            agent = f"agent{i}"
            # FOV
            agent_fovs[agent] = GeometryBridge.polygon_to_avstack(msg)
            if msg.header.frame_id != "world":
                raise NotImplementedError("Need to convert to global here")
            # pose
            tf_world_agent = self.tf_buffer.lookup_transform(
                "world",
                agent,
                msg.header.stamp,
            )
            agent_poses[agent] = GeometryBridge.position_to_avstack(
                tf_world_agent.transform.translation,
                header=msg.header,
            )
            frame = 0
            timestamp = Bridge.rostime_to_time(msg.header.stamp)

        ###################################################
        # Run trust estimation model
        ###################################################

        trust_out = self.model(
            frame=frame,
            timestamp=timestamp,
            agent_poses=agent_poses,
            agent_fovs=agent_fovs,
            agent_tracks=agent_tracks,
            cc_tracks=cc_tracks,
        )
        header = Header(frame_id="world", stamp=Bridge.time_to_rostime(timestamp))

        # convert agent trust messages
        agent_trust = TrustArray()
        agent_trust.header = header
        agent_trust.trusts = [
            Trust(
                identifier=k,
                alpha=v.alpha,
                beta=v.beta,
                mean=v.mean,
                variance=v.variance,
            )
            for k, v in trust_out.agent_trust.items()
        ]

        # convert track trust messages
        track_trust = TrustArray()
        track_trust.header = header
        track_trust.trusts = []

        # publish trust outputs
        self.publisher_agent_trust.publish(agent_trust)
        self.publisher_track_trust.publish(track_trust)


def main(args=None):
    rclpy.init(args=args)

    trust_estimator = MultiAgentTrustEstimator()

    rclpy.spin(trust_estimator)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    trust_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
