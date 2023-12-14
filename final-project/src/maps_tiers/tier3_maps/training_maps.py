# from enum import Enum
#
# class TrainMaps(Enum):
#     ONE = 5

class TrainMaps():

    @staticmethod
    def get_map(map_num, cfg):
        map_num=2 #TODO: remove
        print(f"picking map {map_num}")
        asset_type_to_dict_map = None
        include_env_bound_type = None
        include_asset_type = None
        if map_num == 0:
            asset_type_to_dict_map = {
                "thin": cfg.thin_asset_params,
                "trees": cfg.tree_asset_params,
                "objects": cfg.object_asset_params,
                "back_wall": cfg.back_wall,
                "front_wall": cfg.front_wall,
                "bottom_wall": cfg.bottom_wall,
                "top_wall": cfg.top_wall,
                "left_wall": cfg.left_wall,
                "right_wall": cfg.right_wall,
                "slanted_wall_right": cfg.slanted_wall_right,
                "slanted_wall_left": cfg.slanted_wall_left,
                "thin_wall_front_right": cfg.thin_wall_front_right,
                "thin_wall_front_left": cfg.thin_wall_front_left,
                "thin_wall_back_right": cfg.thin_wall_back_right,
                "thin_wall_back_left": cfg.thin_wall_back_left,
            }
            include_env_bound_type = {
                "front_wall": False,
                "top_wall": False,
                "back_wall": False,
                "bottom_wall": False,
                "left_wall": False,
                "right_wall": False,
                "slanted_wall_right": False,
                "slanted_wall_left": False,
                "thin_wall_front_right": False,
                "thin_wall_front_left": False,
                "thin_wall_back_right": False,
                "thin_wall_back_left": False,
            }
            include_asset_type = {
                "thin": False,
                "trees": False,
                "objects": True
            }
        if map_num == 1:
            asset_type_to_dict_map = {
                "thin": cfg.thin_asset_params,
                "trees": cfg.tree_asset_params,
                "objects": cfg.object_asset_params,
                "back_wall": cfg.back_wall,
                "front_wall": cfg.front_wall,
                "bottom_wall": cfg.bottom_wall,
                "top_wall": cfg.top_wall,
                "left_wall": cfg.left_wall,
                "right_wall": cfg.right_wall,
                "slanted_wall_right": cfg.slanted_wall_right,
                "slanted_wall_left": cfg.slanted_wall_left,
                "thin_wall_front_right": cfg.thin_wall_front_right,
                "thin_wall_front_left": cfg.thin_wall_front_left,
                "thin_wall_back_right": cfg.thin_wall_back_right,
                "thin_wall_back_left": cfg.thin_wall_back_left,
                "thin_wall_obstacle1": cfg.thin_wall_obstacle1,
                "thin_wall_obstacle2": cfg.thin_wall_obstacle2,
                "thin_wall_obstacle3": cfg.thin_wall_obstacle3,
                "thin_wall_obstacle4": cfg.thin_wall_obstacle4,
            }
            include_env_bound_type = {
                "front_wall": False,
                "top_wall": False,
                "back_wall": False,
                "bottom_wall": False,
                "left_wall": True,
                "right_wall": True,
                "slanted_wall_right": False,
                "slanted_wall_left": False,
                "thin_wall_front_right": False,
                "thin_wall_front_left": False,
                "thin_wall_back_right": False,
                "thin_wall_back_left": False,
                "thin_wall_obstacle1": False,
                "thin_wall_obstacle2": False,
                "thin_wall_obstacle3": False,
                "thin_wall_obstacle4": False,
            }
            include_asset_type = {
                "thin": False,
                "trees": False,
                "objects": True
            }
        if map_num == 2:
            asset_type_to_dict_map = {
                "thin": cfg.thin_asset_params,
                "trees": cfg.tree_asset_params,
                "objects": cfg.object_asset_params,
                "back_wall": cfg.back_wall,
                "front_wall": cfg.front_wall,
                "bottom_wall": cfg.bottom_wall,
                "top_wall": cfg.top_wall,
                "left_wall": cfg.left_wall,
                "right_wall": cfg.right_wall,
                "slanted_wall_right": cfg.slanted_wall_right,
                "slanted_wall_left": cfg.slanted_wall_left,
                "thin_wall_front_right": cfg.thin_wall_front_right,
                "thin_wall_front_left": cfg.thin_wall_front_left,
                "thin_wall_back_right": cfg.thin_wall_back_right,
                "thin_wall_back_left": cfg.thin_wall_back_left,
                "thin_wall_obstacle1": cfg.thin_wall_obstacle1,
                "thin_wall_obstacle2": cfg.thin_wall_obstacle2,
                "thin_wall_obstacle3": cfg.thin_wall_obstacle3,
                "thin_wall_obstacle4": cfg.thin_wall_obstacle4,
            }
            include_env_bound_type = {
                "front_wall": False,
                "top_wall": False,
                "back_wall": False,
                "bottom_wall": False,
                "left_wall": False,
                "right_wall": False,
                "slanted_wall_right": True,
                "slanted_wall_left": True,
                "thin_wall_front_right": True,
                "thin_wall_front_left": True,
                "thin_wall_back_right": True,
                "thin_wall_back_left": True,
                "thin_wall_obstacle1": False,
                "thin_wall_obstacle2": False,
                "thin_wall_obstacle3": False,
                "thin_wall_obstacle4": False,
            }
            include_asset_type = {
                "thin": False,
                "trees": False,
                "objects": True
            }
        if map_num == 3:
            asset_type_to_dict_map = {
                "thin": cfg.thin_asset_params,
                "trees": cfg.tree_asset_params,
                "objects": cfg.object_asset_params,
                "back_wall": cfg.back_wall,
                "front_wall": cfg.front_wall,
                "bottom_wall": cfg.bottom_wall,
                "top_wall": cfg.top_wall,
                "left_wall": cfg.left_wall,
                "right_wall": cfg.right_wall,
                "slanted_wall_right": cfg.slanted_wall_right,
                "slanted_wall_left": cfg.slanted_wall_left,
                "thin_wall_front_right": cfg.thin_wall_front_right,
                "thin_wall_front_left": cfg.thin_wall_front_left,
                "thin_wall_back_right": cfg.thin_wall_back_right,
                "thin_wall_back_left": cfg.thin_wall_back_left,
                "thin_wall_obstacle1": cfg.thin_wall_obstacle1,
                "thin_wall_obstacle2": cfg.thin_wall_obstacle2,
                "thin_wall_obstacle3": cfg.thin_wall_obstacle3,
                "thin_wall_obstacle4": cfg.thin_wall_obstacle4,
            }
            include_env_bound_type = {
                "front_wall": False,
                "top_wall": False,
                "back_wall": False,
                "bottom_wall": False,
                "left_wall": True,
                "right_wall": True,
                "slanted_wall_right": False,
                "slanted_wall_left": False,
                "thin_wall_front_right": False,
                "thin_wall_front_left": False,
                "thin_wall_back_right": False,
                "thin_wall_back_left": False,
                "thin_wall_obstacle1": True,
                "thin_wall_obstacle2": True,
                "thin_wall_obstacle3": True,
                "thin_wall_obstacle4": True,
            }
            include_asset_type = {
                "thin": False,
                "trees": False,
                "objects": True
            }

        return asset_type_to_dict_map, include_env_bound_type, include_asset_type

# # from enum import Enum
# #
# # class TrainMaps(Enum):
# #     ONE = 5
#
# class TrainMaps():
#
#     @staticmethod
#     def get_map(map_num, cfg):
#         # map_num=3 #TODO: remove
#         print(f"picking map {map_num}")
#         asset_type_to_dict_map = None
#         include_env_bound_type = None
#         include_asset_type = None
#         if map_num == 0:
#             asset_type_to_dict_map = {
#                 "thin": cfg.thin_asset_params,
#                 "trees": cfg.tree_asset_params,
#                 "objects": cfg.object_asset_params,
#                 "back_wall": cfg.back_wall,
#                 "front_wall": cfg.front_wall,
#                 "bottom_wall": cfg.bottom_wall,
#                 "top_wall": cfg.top_wall,
#                 "left_wall": cfg.left_wall,
#                 "right_wall": cfg.right_wall,
#                 "slanted_wall_right": cfg.slanted_wall_right,
#                 "slanted_wall_left": cfg.slanted_wall_left,
#                 "thin_wall_front_right": cfg.thin_wall_front_right,
#                 "thin_wall_front_left": cfg.thin_wall_front_left,
#                 "thin_wall_back_right": cfg.thin_wall_back_right,
#                 "thin_wall_back_left": cfg.thin_wall_back_left,
#             }
#             include_env_bound_type = {
#                 "front_wall": False,
#                 "top_wall": False,
#                 "back_wall": False,
#                 "bottom_wall": False,
#                 "left_wall": False,
#                 "right_wall": False,
#                 "slanted_wall_right": False,
#                 "slanted_wall_left": False,
#                 "thin_wall_front_right": False,
#                 "thin_wall_front_left": False,
#                 "thin_wall_back_right": False,
#                 "thin_wall_back_left": False,
#             }
#             include_asset_type = {
#                 "thin": False,
#                 "trees": False,
#                 "objects": True
#             }
#         if map_num == 1:
#             asset_type_to_dict_map = {
#                 "thin": cfg.thin_asset_params,
#                 "trees": cfg.tree_asset_params,
#                 "objects": cfg.object_asset_params,
#                 "back_wall": cfg.back_wall,
#                 "front_wall": cfg.front_wall,
#                 "bottom_wall": cfg.bottom_wall,
#                 "top_wall": cfg.top_wall,
#                 "left_wall": cfg.left_wall,
#                 "right_wall": cfg.right_wall,
#                 "slanted_wall_right": cfg.slanted_wall_right,
#                 "slanted_wall_left": cfg.slanted_wall_left,
#                 "thin_wall_front_right": cfg.thin_wall_front_right,
#                 "thin_wall_front_left": cfg.thin_wall_front_left,
#                 "thin_wall_back_right": cfg.thin_wall_back_right,
#                 "thin_wall_back_left": cfg.thin_wall_back_left,
#                 "thin_wall_obstacle1": cfg.thin_wall_obstacle1,
#                 "thin_wall_obstacle2": cfg.thin_wall_obstacle2,
#                 "thin_wall_obstacle3": cfg.thin_wall_obstacle3,
#                 "thin_wall_obstacle4": cfg.thin_wall_obstacle4,
#             }
#             include_env_bound_type = {
#                 "front_wall": False,
#                 "top_wall": False,
#                 "back_wall": False,
#                 "bottom_wall": False,
#                 "left_wall": True,
#                 "right_wall": True,
#                 "slanted_wall_right": False,
#                 "slanted_wall_left": False,
#                 "thin_wall_front_right": False,
#                 "thin_wall_front_left": False,
#                 "thin_wall_back_right": False,
#                 "thin_wall_back_left": False,
#                 "thin_wall_obstacle1": False,
#                 "thin_wall_obstacle2": False,
#                 "thin_wall_obstacle3": False,
#                 "thin_wall_obstacle4": False,
#             }
#             include_asset_type = {
#                 "thin": False,
#                 "trees": False,
#                 "objects": True
#             }
#         if map_num == 2:
#             asset_type_to_dict_map = {
#                 "thin": cfg.thin_asset_params,
#                 "trees": cfg.tree_asset_params,
#                 "objects": cfg.object_asset_params,
#                 "back_wall": cfg.back_wall,
#                 "front_wall": cfg.front_wall,
#                 "bottom_wall": cfg.bottom_wall,
#                 "top_wall": cfg.top_wall,
#                 "left_wall": cfg.left_wall,
#                 "right_wall": cfg.right_wall,
#                 "slanted_wall_right": cfg.slanted_wall_right,
#                 "slanted_wall_left": cfg.slanted_wall_left,
#                 "thin_wall_front_right": cfg.thin_wall_front_right,
#                 "thin_wall_front_left": cfg.thin_wall_front_left,
#                 "thin_wall_back_right": cfg.thin_wall_back_right,
#                 "thin_wall_back_left": cfg.thin_wall_back_left,
#                 "thin_wall_obstacle1": cfg.thin_wall_obstacle1,
#                 "thin_wall_obstacle2": cfg.thin_wall_obstacle2,
#                 "thin_wall_obstacle3": cfg.thin_wall_obstacle3,
#                 "thin_wall_obstacle4": cfg.thin_wall_obstacle4,
#             }
#             include_env_bound_type = {
#                 "front_wall": False,
#                 "top_wall": False,
#                 "back_wall": False,
#                 "bottom_wall": False,
#                 "left_wall": False,
#                 "right_wall": False,
#                 "slanted_wall_right": True,
#                 "slanted_wall_left": True,
#                 "thin_wall_front_right": True,
#                 "thin_wall_front_left": True,
#                 "thin_wall_back_right": True,
#                 "thin_wall_back_left": True,
#                 "thin_wall_obstacle1": False,
#                 "thin_wall_obstacle2": False,
#                 "thin_wall_obstacle3": False,
#                 "thin_wall_obstacle4": False,
#             }
#             include_asset_type = {
#                 "thin": False,
#                 "trees": False,
#                 "objects": True
#             }
#         if map_num == 3:
#             asset_type_to_dict_map = {
#                 "thin": cfg.thin_asset_params,
#                 "trees": cfg.tree_asset_params,
#                 "objects": cfg.object_asset_params,
#                 "back_wall": cfg.back_wall,
#                 "front_wall": cfg.front_wall,
#                 "bottom_wall": cfg.bottom_wall,
#                 "top_wall": cfg.top_wall,
#                 "left_wall": cfg.left_wall,
#                 "right_wall": cfg.right_wall,
#                 "slanted_wall_right": cfg.slanted_wall_right,
#                 "slanted_wall_left": cfg.slanted_wall_left,
#                 "thin_wall_front_right": cfg.thin_wall_front_right,
#                 "thin_wall_front_left": cfg.thin_wall_front_left,
#                 "thin_wall_back_right": cfg.thin_wall_back_right,
#                 "thin_wall_back_left": cfg.thin_wall_back_left,
#                 "thin_wall_obstacle1": cfg.thin_wall_obstacle1,
#                 "thin_wall_obstacle2": cfg.thin_wall_obstacle2,
#                 "thin_wall_obstacle3": cfg.thin_wall_obstacle3,
#                 "thin_wall_obstacle4": cfg.thin_wall_obstacle4,
#             }
#             include_env_bound_type = {
#                 "front_wall": False,
#                 "top_wall": False,
#                 "back_wall": False,
#                 "bottom_wall": False,
#                 "left_wall": True,
#                 "right_wall": True,
#                 "slanted_wall_right": False,
#                 "slanted_wall_left": False,
#                 "thin_wall_front_right": False,
#                 "thin_wall_front_left": False,
#                 "thin_wall_back_right": False,
#                 "thin_wall_back_left": False,
#                 "thin_wall_obstacle1": True,
#                 "thin_wall_obstacle2": True,
#                 "thin_wall_obstacle3": True,
#                 "thin_wall_obstacle4": True,
#             }
#             include_asset_type = {
#                 "thin": False,
#                 "trees": False,
#                 "objects": True
#             }
#
#         return asset_type_to_dict_map, include_env_bound_type, include_asset_type