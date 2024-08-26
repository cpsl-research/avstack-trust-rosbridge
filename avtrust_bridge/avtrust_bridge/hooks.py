from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.environment.objects import ObjectState
    from avstack.geometry import Polygon

from avstack.config import HOOKS
from avstack_rosbag import RosbagHook

from avtrust_bridge import TrustBridge


@HOOKS.register_module()
class TrustFusionRosbagHook(RosbagHook):
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

        # wrap the agent names to integers
        ds_in = [agents, field_of_view_agents, tracks_agents]
        ds_out = [
            {int(k.replace("agent", "")): v for k, v in d_in.items() if v is not None}
            for d_in in ds_in
        ]
        agents_wrap, fovs_wrap, tracks_agents_wrap = ds_out

        # run the model
        self.hook(
            agents=agents_wrap,
            fov_agents=fovs_wrap,
            tracks_agents=tracks_agents_wrap,
            tracks_fused=tracks_fused,
            logger=logger,
        )

        # save the ros topics
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
