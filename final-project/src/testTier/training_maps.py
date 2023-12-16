# from enum import Enum
#
# class TrainMaps(Enum):
#     ONE = 5

class TrainMaps():

    @staticmethod
    def get_map(cfg):

        cfg.longwall1.specified_position = [5, -10, 10]
        cfg.longwall2.specified_position = [15, -10, 10]
        cfg.longwall3.specified_position = [25, -10, 10]
        cfg.longwall4.specified_position = [35, -10, 10]
        cfg.longwall5.specified_position = [5, 0, 5]
        cfg.longwall6.specified_position = [20, 0, 5]

        cfg.longwall7.specified_position = [5, 10, 10]
        cfg.longwall8.specified_position = [15, 10, 10]
        cfg.longwall9.specified_position = [25, 10, 10]
        cfg.longwall10.specified_position = [35, 10, 10]
        cfg.longwall11.specified_position = [10, 0, 15]
        cfg.longwall12.specified_position = [25, 0, 15]

        cfg.longwall1.specified_euler_angle = [0, 0, 0]
        cfg.longwall2.specified_euler_angle = [0, 0, 0]
        cfg.longwall3.specified_euler_angle = [0, 0, 0]
        cfg.longwall4.specified_euler_angle = [0, 0, 0]
        cfg.longwall5.specified_euler_angle = [1.57, 1.57, 0]
        cfg.longwall6.specified_euler_angle = [1.57, 1.57, 0]

        cfg.longwall7.specified_euler_angle = [0, 0, 0]
        cfg.longwall8.specified_euler_angle = [0, 0, 0]
        cfg.longwall9.specified_euler_angle = [0, 0, 0]
        cfg.longwall10.specified_euler_angle = [0, 0, 0]
        cfg.longwall11.specified_euler_angle = [1.57, 1.57, 0]
        cfg.longwall12.specified_euler_angle = [1.57, 1.57, 0]

        asset_type_to_dict_map = {
            "thin": cfg.thin_asset_params,
            "trees": cfg.tree_asset_params,
            "objects": cfg.object_asset_params,
            "longwall1": cfg.longwall1,
            "longwall2": cfg.longwall2,
            "longwall3": cfg.longwall3,
            "longwall4": cfg.longwall4,
            "longwall5": cfg.longwall5,
            "longwall6": cfg.longwall6,
            "longwall7": cfg.longwall7,
            "longwall8": cfg.longwall8,
            "longwall9": cfg.longwall9,
            "longwall10": cfg.longwall10,
            "longwall11": cfg.longwall11,
            "longwall12": cfg.longwall12,
        }
        include_env_bound_type = {
            "longwall1": True,
            "longwall2": True,
            "longwall3": True,
            "longwall4": True,
            "longwall5": True,
            "longwall6": True,
            "longwall7": True,
            "longwall8": True,
            "longwall9": True,
            "longwall10": True,
            "longwall11": True,
            "longwall12": True,
        }
        include_asset_type = {
            "thin": False,
            "trees": False,
            "objects": True
        }

        return asset_type_to_dict_map, include_env_bound_type, include_asset_type