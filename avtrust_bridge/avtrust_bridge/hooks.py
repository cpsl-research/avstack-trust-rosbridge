from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.environment.objects import ObjectState
    from avstack.geometry import Polygon

from avstack.config import HOOKS
from avstack.metrics import get_instantaneous_metrics
from avstack_bridge import Bridge, MetricsBridge, TrackBridge
from avstack_rosbag import RosbagHook
from avtrust.metrics import get_trust_agents_metrics, get_trust_tracks_metrics
from std_msgs.msg import Header

from avtrust_bridge import TrustBridge


class _TrustHook(RosbagHook):
    def __call__(
        self,
        agents: Dict[str, "ObjectState"],
        field_of_view_agents: Dict[str, "Polygon"],
        tracks_agents: Dict[str, "DataContainer"],
        tracks_fused: "DataContainer",
        truths: "DataContainer",
        truths_agents: Dict[str, "DataContainer"],
        attacked_agents: set,
        logger=None,
        *args,
        **kwargs,
    ):
        """Base procedure for running hooks"""
        # prepare the data via hook
        (
            agents_wrap,
            attacked_agents_wrap,
            fovs_wrap,
            tracks_agents_wrap,
            truths_agents_wrap,
        ) = self.wrap_inputs(
            agents=agents,
            attacked_agents=attacked_agents,
            field_of_view_agents=field_of_view_agents,
            tracks_agents=tracks_agents,
            truths_agents=truths_agents,
        )

        # run the model
        self.hook(
            agents=agents_wrap,
            fov_agents=fovs_wrap,
            tracks_agents=tracks_agents_wrap,
            tracks_fused=tracks_fused,
            logger=logger,
        )

        # run assignment metrics
        if self.hook.tracks_trusted is not None:
            self._metrics_assignment = get_instantaneous_metrics(
                tracks=self.hook.tracks_trusted,
                truths=truths,
                timestamp=tracks_fused.timestamp,
            )
        else:
            self._metrics_assignment = None

        # run trust metrics
        if (self.hook.trust_agents is not None) and (
            all([v is not None for v in truths_agents_wrap.values()])
        ):
            self._metrics_trust = {
                "agents": get_trust_agents_metrics(
                    attacked_agents=attacked_agents_wrap,
                    truths_agents=truths_agents_wrap,
                    tracks_agents=tracks_agents_wrap,
                    trust_agents=self.hook.trust_agents,
                ),
                "tracks": get_trust_tracks_metrics(
                    truths=truths,
                    tracks_cc=tracks_fused,
                    trust_tracks=self.hook.trust_tracks,
                ),
            }
        else:
            self._metrics_trust = None

        self.save_outputs()

    @staticmethod
    def wrap_inputs(
        agents, attacked_agents, field_of_view_agents, tracks_agents, truths_agents
    ):
        # wrap the agent names to integers
        ds_in = [agents, field_of_view_agents, tracks_agents, truths_agents]
        ds_out = [
            {int(k.replace("agent", "")): v for k, v in d_in.items() if v is not None}
            for d_in in ds_in
        ]
        agents_wrap, fovs_wrap, tracks_agents_wrap, truths_agents_wrap = ds_out
        truths_agents_wrap = {
            k: v["lidar0"] if "lidar0" in v else v
            for k, v in truths_agents_wrap.items()
        }
        attacked_agents_wrap = (
            [int(agent.replace("agent", "")) for agent in attacked_agents]
            if attacked_agents is not None
            else None
        )
        return (
            agents_wrap,
            attacked_agents_wrap,
            fovs_wrap,
            tracks_agents_wrap,
            truths_agents_wrap,
        )

    def save_outputs(self):
        raise NotImplementedError

    def save_trust_to_rosbag(self):
        if self.hook.trust_agents is not None:
            self.ros_topic_write["/trust/trust_agents"] = {
                "type": "avtrust_msgs/msg/TrustArray",
                "data": TrustBridge.trust_array_avstack_to_ros(self.hook.trust_agents),
            }
        if self.hook.trust_tracks is not None:
            self.ros_topic_write["/trust/trust_tracks"] = {
                "type": "avtrust_msgs/msg/TrustArray",
                "data": TrustBridge.trust_array_avstack_to_ros(self.hook.trust_tracks),
            }
        if self.hook.psms_agents is not None:
            self.ros_topic_write["/trust/psms_agents"] = {
                "type": "avtrust_msgs/msg/PsmArray",
                "data": TrustBridge.psm_array_avstack_to_ros(self.hook.psms_agents),
            }
        if self.hook.psms_tracks is not None:
            self.ros_topic_write["/trust/psms_tracks"] = {
                "type": "avtrust_msgs/msg/PsmArray",
                "data": TrustBridge.psm_array_avstack_to_ros(self.hook.psms_tracks),
            }
        if self._metrics_assignment is not None:
            self.ros_topic_write["/metrics/security_aware_fusion/assignment"] = {
                "type": "avstack_msgs/msg/AssignmentMetrics",
                "data": MetricsBridge.assignment_metrics_avstack_to_ros(
                    self._metrics_assignment
                ),
            }
        if self._metrics_trust is not None:
            self.ros_topic_write["/metrics/security_aware_fusion/agent_trust"] = {
                "type": "avtrust_msgs/msg/AgentTrustMetricArray",
                "data": TrustBridge.agent_trust_metric_array_avstack_to_ros(
                    self._metrics_trust["agents"]
                ),
            }
            self.ros_topic_write["/metrics/security_aware_fusion/track_trust"] = {
                "type": "avtrust_msgs/msg/TrackTrustMetricArray",
                "data": TrustBridge.track_trust_metric_array_avstack_to_ros(
                    self._metrics_trust["tracks"]
                ),
            }


@HOOKS.register_module()
class TrustEstimationRosbagHook(_TrustHook):
    def save_outputs(self):
        self.save_trust_to_rosbag()


@HOOKS.register_module()
class TrustFusionRosbagHook(_TrustHook):
    def save_outputs(self):
        self.save_trust_to_rosbag()
        self.save_tracks_to_rosbag()

    def save_tracks_to_rosbag(self):
        if self.hook.tracks_trusted is not None:
            global_header = Header(
                frame_id="world",
                stamp=Bridge.time_to_rostime(self.hook.tracks_trusted.timestamp),
            )
            self.ros_topic_write["/trust/tracks_trusted"] = {
                "type": "avstack_msgs/msg/BoxTrackArray",
                "data": TrackBridge.avstack_to_tracks(
                    tracks=self.hook.tracks_trusted,
                    header=global_header,
                ),
            }
