import rclpy
from avstack_msgs.msg import BoxTrackArray
from geometry_msgs.msg import PolygonStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from std_msgs.msg import String


class FakeModel:
    def reset(self):
        pass


class MultiAgentTrustEstimator(Node):
    def __init__(self, verbose: bool = True):
        super().__init__("mate")
        self.verbose = verbose
        self.declare_parameter("n_agents", 4)

        # innitialize model
        self.model = FakeModel()

        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )

        # subscribe to initialization message (optional)
        self.subscriber_init = self.create_subscription(
            String,
            "/initialization",
            self.init_callback,
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
            for agent_ID in range(self.get_parameter("n_agents").value)
        }

        # listen to tracks from agents and cc
        self.subscriber_trks = {
            agent_ID: Subscriber(
                self,
                BoxTrackArray,
                f"/agent{agent_ID}/tracks_3d",
                qos_profile=qos,
            )
            for agent_ID in range(self.get_parameter("n_agents").value)
        }
        self.subscriber_trks["command_center"] = Subscriber(
            self,
            BoxTrackArray,
            f"/command_center/tracks_3d",
            qos_profile=qos,
        )

        # synchronize track messages
        self.synchronizer_trks = ApproximateTimeSynchronizer(
            tuple(self.subscriber_trks.values()) + tuple(self.subscriber_fovs.values()),
            queue_size=10,
            slop=0.1,
        )
        self.synchronizer_trks.registerCallback(self.trks_fov_receive)

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

        # run trust estimation model


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
