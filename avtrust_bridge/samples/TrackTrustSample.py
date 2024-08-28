import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from avtrust_msgs.msg import Trust, TrustArray


def sample_trust_parameter(scale: float = 20.0):
    return scale * np.random.rand()


class TrackTrustPublisher(Node):
    def __init__(self):
        super().__init__("track_trust_publisher_sample")
        self._pub = self.create_publisher(TrustArray, "track_trust", 10)
        self._timer = self.create_timer(0.05, self.pub_sample)
        self.n_tracks_init = 5
        self.id_counter = 0
        self.track_ids_active = []
        self.alphas = {}
        self.betas = {}
        for i in range(self.n_tracks_init):
            self.track_ids_active.append(i)
            self.alphas[i] = sample_trust_parameter()
            self.betas[i] = sample_trust_parameter()
            self.id_counter += 1

    def pub_sample(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"

        # probability of dropping a track
        p_drop = 0.005
        ids_drop = []
        for trk_id in self.track_ids_active:
            if np.random.rand() < p_drop:
                ids_drop.append(trk_id)
        for id_drop in ids_drop:
            self.track_ids_active.remove(id_drop)

        # probability of adding a single rack
        p_add = 0.02
        if np.random.rand() < p_add:
            self.track_ids_active.append(self.id_counter)
            self.alphas[self.id_counter] = sample_trust_parameter()
            self.betas[self.id_counter] = sample_trust_parameter()
            self.id_counter += 1

        # create fake trust messages
        msg = TrustArray()
        msg.header = header
        for track_id in self.track_ids_active:
            # construct trust values by some moving average
            w = 0.95
            self.alphas[track_id] = (
                w * self.alphas[track_id] + (1 - w) * sample_trust_parameter()
            )
            self.betas[track_id] = (
                w * self.betas[track_id] + (1 - w) * sample_trust_parameter()
            )

            # populate the trust message
            alpha = self.alphas[track_id]
            beta = self.betas[track_id]
            trust = Trust()
            trust.identifier = f"track{track_id}"
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

    trust_pub = TrackTrustPublisher()

    rclpy.spin(trust_pub)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    trust_pub.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
