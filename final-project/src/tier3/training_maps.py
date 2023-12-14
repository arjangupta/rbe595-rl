# from enum import Enum
#
# class TrainMaps(Enum):
#     ONE = 5

class TrainMaps():

    @staticmethod
    def get_map(map_num, cfg):
        map_num=6 #TODO: remove
        print(f"picking map {map_num}")

        if map_num == 1:
            cfg.wall1_map1.specified_position = [2.5, -5, 10]
            cfg.wall2_map1.specified_position = [7.5, -5, 10]
            cfg.wall3_map1.specified_position = [12.5, -5, 10]
            cfg.wall4_map1.specified_position = [17.5, -5, 10]
            cfg.wall5_map1.specified_position = [22.5, -5, 10]
            cfg.wall6_map1.specified_position = [27.5, -5, 10]

            cfg.wall7_map1.specified_position = [2.5, 5, 10]
            cfg.wall8_map1.specified_position = [7.5, 5, 10]
            cfg.wall9_map1.specified_position = [12.5, 5, 10]
            cfg.wall10_map1.specified_position = [17.5, 5, 10]
            cfg.wall11_map1.specified_position = [22.5, 5, 10]
            cfg.wall12_map1.specified_position = [27.5, 5, 10]

            cfg.wall1_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall2_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall3_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall4_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall5_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall6_map1.specified_euler_angle = [0, 0, 0]

            cfg.wall7_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall8_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall9_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall10_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall11_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall12_map1.specified_euler_angle = [0, 0, 0]

        if map_num == 2:
            cfg.wall1_map1.specified_position = [2.5, -10, 10]
            cfg.wall2_map1.specified_position = [7.5, -10, 10]
            cfg.wall3_map1.specified_position = [11.78, -8.24, 10]
            cfg.wall4_map1.specified_position = [15.295, -4.71, 10]
            cfg.wall5_map1.specified_position = [19.56, -2.95, 10]
            cfg.wall6_map1.specified_position = [24.56, -2.95, 10]

            cfg.wall7_map1.specified_position = [2.5, 10, 10]
            cfg.wall8_map1.specified_position = [7.5, 10, 10]
            cfg.wall9_map1.specified_position = [11.78, 8.24, 10]
            cfg.wall10_map1.specified_position = [15.295, 4.71, 10]
            cfg.wall11_map1.specified_position = [19.56, 2.95, 10]
            cfg.wall12_map1.specified_position = [24.56, 2.95, 10]

            cfg.wall1_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall2_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall3_map1.specified_euler_angle = [0, 0, 0.785]
            cfg.wall4_map1.specified_euler_angle = [0, 0, 0.785]
            cfg.wall5_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall6_map1.specified_euler_angle = [0, 0, 0]

            cfg.wall7_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall8_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall9_map1.specified_euler_angle = [0, 0, -0.785]
            cfg.wall10_map1.specified_euler_angle = [0, 0, -0.785]
            cfg.wall11_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall12_map1.specified_euler_angle = [0, 0, 0]

        if map_num == 3:
            cfg.wall1_map1.specified_position = [2.5, -5, 10]
            cfg.wall2_map1.specified_position = [7.5, -5, 10]
            cfg.wall3_map1.specified_position = [12.5, -5, 10]
            cfg.wall4_map1.specified_position = [17.5, -5, 10]
            cfg.wall5_map1.specified_position = [5, -2.5, 10]
            cfg.wall6_map1.specified_position = [15, -2.5, 10]

            cfg.wall7_map1.specified_position = [2.5, 5, 10]
            cfg.wall8_map1.specified_position = [7.5, 5, 10]
            cfg.wall9_map1.specified_position = [12.5, 5, 10]
            cfg.wall10_map1.specified_position = [17.5, 5, 10]
            cfg.wall11_map1.specified_position = [22.5, 5, 10]
            cfg.wall12_map1.specified_position = [10, 2.5, 10]

            cfg.wall1_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall2_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall3_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall4_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall5_map1.specified_euler_angle = [0, 0, 1.57]
            cfg.wall6_map1.specified_euler_angle = [0, 0, 1.57]

            cfg.wall7_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall8_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall9_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall10_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall11_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall12_map1.specified_euler_angle = [0, 0, 1.57]

        if map_num == 4:
            cfg.wall1_map1.specified_position = [2.5, -5, 10]
            cfg.wall2_map1.specified_position = [7.5, -5, 10]
            cfg.wall3_map1.specified_position = [12.5, -5, 10]
            cfg.wall4_map1.specified_position = [17.5, -5, 10]
            cfg.wall5_map1.specified_position = [5, -5, 10]
            cfg.wall6_map1.specified_position = [5, 0, 10]

            cfg.wall7_map1.specified_position = [2.5, 5, 10]
            cfg.wall8_map1.specified_position = [7.5, 5, 10]
            cfg.wall9_map1.specified_position = [12.5, 5, 10]
            cfg.wall10_map1.specified_position = [17.5, 5, 10]
            cfg.wall11_map1.specified_position = [10, 5, 10]
            cfg.wall12_map1.specified_position = [10, 0, 10]

            cfg.wall1_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall2_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall3_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall4_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall5_map1.specified_euler_angle = [0, 0, 1.57]
            cfg.wall6_map1.specified_euler_angle = [0, 0, 1.57]

            cfg.wall7_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall8_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall9_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall10_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall11_map1.specified_euler_angle = [0, 0, 1.57]
            cfg.wall12_map1.specified_euler_angle = [0, 0, 1.57]

        if map_num == 5:
            cfg.wall1_map1.specified_position = [5, 0, 2.5]
            cfg.wall2_map1.specified_position = [5, 0, 2.5]
            cfg.wall3_map1.specified_position = [5, 0, 2.5]
            cfg.wall4_map1.specified_position = [5, 0, 2.5]
            cfg.wall5_map1.specified_position = [5, 0, 2.5]
            cfg.wall6_map1.specified_position = [5, 0, 2.5]

            cfg.wall7_map1.specified_position = [5, 0, 2.5]
            cfg.wall8_map1.specified_position = [5, 0, 2.5]
            cfg.wall9_map1.specified_position = [15, 5, 10]
            cfg.wall10_map1.specified_position = [15, -5, 10]
            cfg.wall11_map1.specified_position = [15, 0, 12.5]
            cfg.wall12_map1.specified_position = [15, 0, 17.5]

            cfg.wall1_map1.specified_euler_angle = [1.57, 1.57, 0]
            cfg.wall2_map1.specified_euler_angle = [1.57, 1.57, 0]
            cfg.wall3_map1.specified_euler_angle = [1.57, 1.57, 0]
            cfg.wall4_map1.specified_euler_angle = [1.57, 1.57, 0]
            cfg.wall5_map1.specified_euler_angle = [1.57, 1.57, 0]
            cfg.wall6_map1.specified_euler_angle = [1.57, 1.57, 0]

            cfg.wall7_map1.specified_euler_angle = [1.57, 1.57, 0]
            cfg.wall8_map1.specified_euler_angle = [1.57, 1.57, 0]
            cfg.wall9_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall10_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall11_map1.specified_euler_angle = [1.57, 1.57, 0]
            cfg.wall12_map1.specified_euler_angle = [1.57, 1.57, 0]

        if map_num == 6:
            cfg.wall1_map1.specified_position = [2.5, -5, 10]
            cfg.wall2_map1.specified_position = [7.5, -5, 10]
            cfg.wall3_map1.specified_position = [12.5, -5, 10]
            cfg.wall4_map1.specified_position = [17.5, -5, 10]
            cfg.wall5_map1.specified_position = [22.5, -5, 10]
            cfg.wall6_map1.specified_position = [10, 0, 10]

            cfg.wall7_map1.specified_position = [2.5, 5, 10]
            cfg.wall8_map1.specified_position = [7.5, 5, 10]
            cfg.wall9_map1.specified_position = [12.5, 5, 10]
            cfg.wall10_map1.specified_position = [17.5, 5, 10]
            cfg.wall11_map1.specified_position = [22.5, 5, 10]
            cfg.wall12_map1.specified_position = [20, 0, 2.5]

            cfg.wall1_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall2_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall3_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall4_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall5_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall6_map1.specified_euler_angle = [1.57, 1.57, 0]

            cfg.wall7_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall8_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall9_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall10_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall11_map1.specified_euler_angle = [0, 0, 0]
            cfg.wall12_map1.specified_euler_angle = [1.57, 1.57, 0]


        asset_type_to_dict_map = {
            "thin": cfg.thin_asset_params,
            "trees": cfg.tree_asset_params,
            "objects": cfg.object_asset_params,
            "wall1_map1": cfg.wall1_map1,
            "wall2_map1": cfg.wall2_map1,
            "wall3_map1": cfg.wall3_map1,
            "wall4_map1": cfg.wall4_map1,
            "wall5_map1": cfg.wall5_map1,
            "wall6_map1": cfg.wall6_map1,
            "wall7_map1": cfg.wall7_map1,
            "wall8_map1": cfg.wall8_map1,
            "wall9_map1": cfg.wall9_map1,
            "wall10_map1": cfg.wall10_map1,
            "wall11_map1": cfg.wall11_map1,
            "wall12_map1": cfg.wall12_map1,
        }
        include_env_bound_type = {
            "wall1_map1": True,
            "wall2_map1": True,
            "wall3_map1": True,
            "wall4_map1": True,
            "wall5_map1": True,
            "wall6_map1": True,
            "wall7_map1": True,
            "wall8_map1": True,
            "wall9_map1": True,
            "wall10_map1": True,
            "wall11_map1": True,
            "wall12_map1": True,
        }
        include_asset_type = {
            "thin": False,
            "trees": False,
            "objects": True
        }

        return asset_type_to_dict_map, include_env_bound_type, include_asset_type