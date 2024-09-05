from avstack_bridge import Bridge
from avtrust.distributions import TrustArray, TrustBetaDistribution
from avtrust.measurement import Psm, PsmArray
from avtrust.metrics import (
    AgentTrustMetric,
    AggregateAgentTrustMetric,
    AggregateTrackTrustMetric,
    TrackTrustMetric,
)
from std_msgs.msg import Header

from avtrust_msgs.msg import AgentTrustMetric as AgentTrustMetricRos
from avtrust_msgs.msg import AgentTrustMetricArray as AgentTrustMetricArrayRos
from avtrust_msgs.msg import Psm as PsmRos
from avtrust_msgs.msg import PsmArray as PsmArrayRos
from avtrust_msgs.msg import TrackTrustMetric as TrackTrustMetricRos
from avtrust_msgs.msg import TrackTrustMetricArray as TrackTrustMetricArrayRos
from avtrust_msgs.msg import Trust as TrustRos
from avtrust_msgs.msg import TrustArray as TrustArrayRos


def get_global_header(timestamp: float):
    return Header(
        frame_id="world",
        stamp=Bridge.time_to_rostime(timestamp),
    )


class TrustBridge:
    # ----------------------------------
    # singleton methods for trust
    # ----------------------------------
    @staticmethod
    def psm_avstack_to_ros(psm: Psm) -> PsmRos:
        return PsmRos(
            header=get_global_header(psm.timestamp),
            target=psm.target,
            value=psm.value,
            confidence=psm.confidence,
            source=psm.source,
        )

    @staticmethod
    def trust_avstack_to_ros(trust: TrustBetaDistribution) -> TrustRos:
        return TrustRos(
            header=get_global_header(trust.timestamp),
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
    # singleton methods for metrics
    # ----------------------------------
    @staticmethod
    def agent_trust_metric_avstack_to_ros(
        metric: AgentTrustMetric,
    ) -> AgentTrustMetricRos:
        return AgentTrustMetricRos(
            header=get_global_header(metric.timestamp),
            identifier=metric.identifier,
            agent_is_attacked=metric.agent_is_attacked,
            f1_score=metric.f1_score,
            area_above_cdf=metric.area_above_cdf,
            f1_threshold=metric.f1_threshold,
        )

    @staticmethod
    def track_trust_metric_avstack_to_ros(
        metric: TrackTrustMetric,
    ) -> TrackTrustMetricRos:
        return TrackTrustMetricRos(
            header=get_global_header(metric.timestamp),
            identifier=metric.identifier,
            area_above_cdf=metric.area_above_cdf,
            assigned_to_truth=metric.assigned_to_truth,
        )

    @staticmethod
    def agent_trust_metric_ros_to_avstack(msg: AgentTrustMetricRos) -> AgentTrustMetric:
        return AgentTrustMetric(
            timestamp=Bridge.rostime_to_time(msg.header.stamp),
            identifier=msg.identifier,
            agent_is_attacked=msg.agent_is_attacked,
            f1_score=msg.f1_score,
            area_above_cdf=msg.area_above_cdf,
            f1_threshold=msg.f1_threshold,
        )

    @staticmethod
    def track_trust_metric_ros_to_avstack(msg: TrackTrustMetricRos) -> TrackTrustMetric:
        return TrackTrustMetric(
            timestamp=Bridge.rostime_to_time(msg.header.stamp),
            identifier=msg.identifier,
            area_above_cdf=msg.area_above_cdf,
            assigned_to_truth=msg.assigned_to_truth,
        )

    # ----------------------------------
    # array methods for trust
    # ----------------------------------
    @staticmethod
    def psm_array_avstack_to_ros(psms: PsmArray) -> PsmArrayRos:
        psms_ros = [TrustBridge.psm_avstack_to_ros(psm) for psm in psms]
        header = get_global_header(psms.timestamp)
        return PsmArrayRos(header=header, psms=psms_ros)

    @staticmethod
    def trust_array_avstack_to_ros(trusts: TrustArray) -> TrustArrayRos:
        trusts_ros = [
            TrustBridge.trust_avstack_to_ros(trust) for trust in trusts.trusts.values()
        ]
        header = get_global_header(trusts.timestamp)
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

    # ----------------------------------
    # array methods for metrics
    # ----------------------------------
    @staticmethod
    def agent_trust_metric_array_avstack_to_ros(
        metrics: AggregateAgentTrustMetric,
    ) -> AgentTrustMetricArrayRos:
        return AgentTrustMetricArrayRos(
            header=get_global_header(metrics.timestamp),
            metrics=[
                TrustBridge.agent_trust_metric_avstack_to_ros(metric)
                for metric in metrics.values()
            ],
        )

    @staticmethod
    def track_trust_metric_array_avstack_to_ros(
        metrics: AggregateTrackTrustMetric,
    ) -> TrackTrustMetricArrayRos:
        return TrackTrustMetricArrayRos(
            header=get_global_header(metrics.timestamp),
            metrics=[
                TrustBridge.track_trust_metric_avstack_to_ros(metric)
                for metric in metrics.values()
            ],
        )

    @staticmethod
    def agent_trust_metric_array_ros_to_avstack(
        msg: AgentTrustMetricArrayRos,
    ) -> AggregateAgentTrustMetric:
        agent_metrics = [
            TrustBridge.agent_trust_metric_ros_to_avstack(metric)
            for metric in msg.metrics
        ]
        agent_metrics = {met.identifier: met for met in agent_metrics}
        return AggregateAgentTrustMetric(
            timestamp=Bridge.rostime_to_time(msg.header.stamp),
            agent_metrics=agent_metrics,
        )

    @staticmethod
    def track_trust_metric_array_ros_to_avstack(
        msg: TrackTrustMetricArrayRos,
    ) -> AggregateTrackTrustMetric:
        track_metrics = [
            TrustBridge.track_trust_metric_ros_to_avstack(metric)
            for metric in msg.metrics
        ]
        track_metrics = {met.identifier: met for met in track_metrics}
        return AggregateTrackTrustMetric(
            timestamp=Bridge.rostime_to_time(msg.header.stamp),
            track_metrics=track_metrics,
        )
