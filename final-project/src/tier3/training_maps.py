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
        # if map_num == 0:
        #     asset_type_to_dict_map = {
        #         "thin": cfg.thin_asset_params,
        #         "trees": cfg.tree_asset_params,
        #         "objects": cfg.object_asset_params,
        #         "long_left_wall": cfg.left_wall,
        #         "long_right_wall": cfg.right_wall,
        #         "back_wall": cfg.back_wall,
        #         "front_wall": cfg.front_wall,
        #         "bottom_wall": cfg.bottom_wall,
        #         "top_wall": cfg.top_wall}
        #     include_env_bound_type = {
        #         "front_wall": False,
        #         "long_left_wall": False,
        #         "top_wall": False,
        #         "back_wall": False,
        #         "long_right_wall": False,
        #         "bottom_wall": False}
        #     include_asset_type = {
        #         "thin": False,
        #         "trees": False,
        #         "objects": False
        #     }
        if map_num == 1:
            asset_type_to_dict_map = {
                "thin": cfg.thin_asset_params,
                "trees": cfg.tree_asset_params,
                "objects": cfg.object_asset_params,
                "long_left_wall": cfg.left_wall,
                "long_right_wall": cfg.right_wall,
                "back_wall": cfg.back_wall,
                "front_wall": cfg.front_wall,
                "bottom_wall": cfg.bottom_wall,
                "top_wall": cfg.top_wall}
            include_env_bound_type = {
                "front_wall": False,
                "long_left_wall": True,
                "top_wall": False,
                "back_wall": False,
                "long_right_wall": True,
                "bottom_wall": False,
                "squiggle_wall": False
            }
            include_asset_type = {
                "thin": False,
                "trees": False,
                "objects": False
            }

            if map_num == 2:
                asset_type_to_dict_map = {
                    "thin": cfg.thin_asset_params,
                    "trees": cfg.tree_asset_params,
                    "objects": cfg.object_asset_params,
                    "long_left_wall": cfg.left_wall,
                    "long_right_wall": cfg.right_wall,
                    "back_wall": cfg.back_wall,
                    "front_wall": cfg.front_wall,
                    "bottom_wall": cfg.bottom_wall,
                    "top_wall": cfg.top_wall,
                    "squiggle_wall": cfg.squiggle_wall
                }
                include_env_bound_type = {
                    "front_wall": False,
                    "long_left_wall": False,
                    "top_wall": False,
                    "back_wall": False,
                    "long_right_wall": False,
                    "bottom_wall": False,
                    "squiggle_wall": True
                }
                include_asset_type = {
                    "thin": False,
                    "trees": False,
                    "objects": False
                }

        return asset_type_to_dict_map, include_env_bound_type, include_asset_type