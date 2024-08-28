import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from avtrust_msgs.msg import Trust, TrustArray


def sample_trust_parameter(scale: float = 20.0):
    return scale * np.random.rand()


class AgentTrustPublisher(Node):
    def __init__(self):
        super().__init__("agent_trust_publisher_sample")
        self._pub = self.create_publisher(TrustArray, "agent_trust", 10)
        self._timer = self.create_timer(0.05, self.pub_sample)
        self.n_agents = 4
        self.alphas = {
            i_agent: sample_trust_parameter() for i_agent in range(self.n_agents)
        }
        self.betas = {
            i_agent: sample_trust_parameter() for i_agent in range(self.n_agents)
        }

    def pub_sample(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"

        # create fake trust messages
        msg = TrustArray()
        msg.header = header
        for agent_id in range(self.n_agents):
            # construct trust values by some moving average
            w = 0.95
            self.alphas[agent_id] = (
                w * self.alphas[agent_id] + (1 - w) * sample_trust_parameter()
            )
            self.betas[agent_id] = (
                w * self.betas[agent_id] + (1 - w) * sample_trust_parameter()
            )

            # populate the trust message
            alpha = self.alphas[agent_id]
            beta = self.betas[agent_id]
            trust = Trust()
            trust.identifier = f"agent{agent_id}"
            trust.alpha = alpha
            trust.beta = beta
            trust.mean = alpha / (alpha + beta)
            trust.variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

            # add to array
            msg.trusts.append(trust)

        # publish output
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    trust_pub = AgentTrustPublisher()

    rclpy.spin(trust_pub)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    trust_pub.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
