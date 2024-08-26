from avstack_bridge import Bridge
from avtrust.distributions import TrustArray, TrustBetaDistribution
from avtrust.measurement import Psm, PsmArray
from std_msgs.msg import Header

from avtrust_msgs.msg import Psm as PsmRos
from avtrust_msgs.msg import PsmArray as PsmArrayRos
from avtrust_msgs.msg import Trust as TrustRos
from avtrust_msgs.msg import TrustArray as TrustArrayRos


class TrustBridge:
    # ----------------------------------
    # singleton methods
    # ----------------------------------
    @staticmethod
    def psm_avstack_to_ros(psm: Psm) -> PsmRos:
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
    def trust_avstack_to_ros(trust: TrustBetaDistribution) -> TrustRos:
        return TrustRos(
            header=Header(
                frame_id="world", stamp=Bridge.time_to_rostime(trust.timestamp)
            ),
            identifier=trust.identifier,
            alpha=trust.alpha,
            beta=trust.beta,
        )

    @staticmethod
    def psm_ros_to_avstack(msg: PsmRos) -> Psm:
        return Psm(
            timestamp=Bridge.rostime_to_time(msg.header.stamp),
            target=msg.target,
            value=msg.value,
            confidence=msg.confidence,
            source=msg.source,
        )

    @staticmethod
    def trust_ros_to_avstack(msg: TrustRos) -> TrustBetaDistribution:
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
    def psm_array_avstack_to_ros(psms: PsmArray) -> PsmArrayRos:
        psms_ros = [TrustBridge.psm_avstack_to_ros(psm) for psm in psms]
        header = Header(frame_id="world", stamp=Bridge.time_to_rostime(psms.timestamp))
        return PsmArrayRos(header=header, psms=psms_ros)

    @staticmethod
    def trust_array_avstack_to_ros(trusts: TrustArray) -> TrustArrayRos:
        trusts_ros = [
            TrustBridge.trust_avstack_to_ros(trust) for trust in trusts.trusts.values()
        ]
        header = Header(
            frame_id="world", stamp=Bridge.time_to_rostime(trusts.timestamp)
        )
        return TrustArrayRos(header=header, trusts=trusts_ros)

    @staticmethod
    def psm_array_ros_to_avstack(msg: PsmArrayRos) -> PsmArray:
        psms = [TrustBridge.psm_ros_to_avstack(psm) for psm in msg.psms]
        timestamp = Bridge.rostime_to_time(msg.header.stamp)
        return PsmArray(timestamp=timestamp, psms=psms)

    @staticmethod
    def trust_array_ros_to_avstack(msg: TrustArrayRos) -> TrustArray:
        trusts = [TrustBridge.trust_ros_to_avstack(trust) for trust in msg.trusts]
        timestamp = Bridge.rostime_to_time(msg.header.stamp)
        return TrustArray(timestamp=timestamp, trusts=trusts)
