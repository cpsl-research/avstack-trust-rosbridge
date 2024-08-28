from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.environment.objects import ObjectState
    from avstack.geometry import Polygon

from avstack.config import HOOKS
from avstack_bridge import Bridge, TrackBridge
from avstack_rosbag import RosbagHook
from std_msgs.msg import Header

from avtrust_bridge import TrustBridge


class _TrustHook(RosbagHook):
    def __call__(
        self,
        agents: Dict[str, "ObjectState"],
        field_of_view_agents: Dict[str, "Polygon"],
        tracks_agents: Dict[str, "DataContainer"],
        tracks_fused: "DataContainer",
        logger=None,
        *args,
        **kwargs,
    ):
        """Base procedure for running hooks"""
        # prepare the data via hook
        agents_wrap, fovs_wrap, tracks_agents_wrap = self.wrap_inputs(
            agents=agents,
            field_of_view_agents=field_of_view_agents,
            tracks_agents=tracks_agents,
        )

        # run the model
        self.hook(
            agents=agents_wrap,
            fov_agents=fovs_wrap,
            tracks_agents=tracks_agents_wrap,
            tracks_fused=tracks_fused,
            logger=logger,
        )
        self.save_outputs()

    @staticmethod
    def wrap_inputs(agents, field_of_view_agents, tracks_agents):
        # wrap the agent names to integers
        ds_in = [agents, field_of_view_agents, tracks_agents]
        ds_out = [
            {int(k.replace("agent", "")): v for k, v in d_in.items() if v is not None}
            for d_in in ds_in
        ]
        agents_wrap, fovs_wrap, tracks_agents_wrap = ds_out
        return agents_wrap, fovs_wrap, tracks_agents_wrap

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
        if self.hook.tracks_fused is not None:
            global_header = Header(
                frame_id="world",
                stamp=Bridge.time_to_rostime(self.hook.tracks_fused.timestamp),
            )
            self.ros_topic_write["/trust/tracks_fused"] = {
                "type": "avstack_msgs/msg/BoxTrackArray",
                "data": TrackBridge.avstack_to_tracks(
                    tracks=self.hook.tracks_fused,
                    header=global_header,
                ),
            }
