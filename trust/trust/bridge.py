from avstack_bridge import Bridge
from mate.distributions import TrustArray, TrustBetaDistribution
from mate.measurement import Psm, PsmArray
from std_msgs.msg import Header

from trust_msgs.msg import Psm as PsmRos
from trust_msgs.msg import PsmArray as PsmArrayRos
from trust_msgs.msg import Trust as TrustRos
from trust_msgs.msg import TrustArray as TrustArrayRos


class TrustBridge:
    # ----------------------------------
    # singleton methods
    # ----------------------------------
    @staticmethod
    def psm_to_ros(psm: Psm) -> PsmRos:
        return PsmRos(
            header=Header(
                frame_id="world", stamp=Bridge.time_to_rostime(psm.timestamp)
            ),
            target=psm.target,
            value=psm.value,
            confidence=psm.confidence,
            source=psm.source,
        )

    @staticmethod
    def trust_to_ros(trust: TrustBetaDistribution) -> TrustRos:
        return TrustRos(
            header=Header(
                frame_id="world", stamp=Bridge.time_to_rostime(trust.timestamp)
            ),
            identifier=trust.identifier,
            alpha=trust.alpha,
            beta=trust.beta,
        )

    @staticmethod
    def ros_to_psm(msg: PsmRos) -> Psm:
        return Psm(
            timestamp=Bridge.rostime_to_time(msg.header.stamp),
            target=msg.target,
            value=msg.value,
            confidence=msg.confidence,
            source=msg.source,
        )

    @staticmethod
    def ros_to_trust(msg: TrustRos) -> TrustBetaDistribution:
        return TrustBetaDistribution(
            timestamp=Bridge.rostime_to_time(msg.header.stamp),
            identifier=msg.identifier,
            alpha=msg.alpha,
            beta=msg.beta,
        )

    # ----------------------------------
    # array methods
    # ----------------------------------
    @staticmethod
    def psm_array_to_ros(psms: PsmArray) -> PsmArrayRos:
        psms_ros = [TrustBridge.psm_to_ros(psm) for psm in psms]
        header = Header(frame_id="world", stamp=Bridge.time_to_rostime(psms.timestamp))
        return PsmArrayRos(header=header, psms=psms_ros)

    @staticmethod
    def trust_array_to_ros(trusts: TrustArray) -> TrustArrayRos:
        trusts_ros = [TrustBridge.trust_to_ros(trust) for trust in trusts.trusts.values()]
        header = Header(
            frame_id="world", stamp=Bridge.time_to_rostime(trusts.timestamp)
        )
        return TrustArrayRos(header=header, trusts=trusts_ros)

    @staticmethod
    def ros_to_psm_array(msg: PsmArrayRos) -> PsmArray:
        psms = [TrustBridge.ros_to_psm(psm) for psm in msg.psms]
        timestamp = Bridge.rostime_to_time(msg.header.stamp)
        return PsmArray(timestamp=timestamp, psms=psms)

    @staticmethod
    def ros_to_trust_array(msg: TrustArrayRos) -> TrustArray:
        trusts = [TrustBridge.ros_to_trust(trust) for trust in msg.trusts]
        timestamp = Bridge.rostime_to_time(msg.header.stamp)
        return TrustArray(timestamp=timestamp, trusts=trusts)
