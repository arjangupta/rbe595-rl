Creating the Maps Inspired by the Paper:

* `picking map 1
Adding asset type: objects
Adding environment bound type: left_wall
Adding environment bound type: right_wall
...
 CREATING ENVIRONMENT
...
 ENVIRONMENT CREATED
...
 RESETTING ENV 0
...
picking map 2
Adding asset type: objects
Adding environment bound type: slanted_wall_right
Adding environment bound type: slanted_wall_left
Adding environment bound type: thin_wall_front_right
Adding environment bound type: thin_wall_front_left
Adding environment bound type: thin_wall_back_right
Adding environment bound type: thin_wall_back_left
env ids: tensor([0], device='cuda:0')
Traceback (most recent call last):
  File "main.py", line 28, in <module>
    main()
  File "main.py", line 25, in main
    obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)
  File "/home/taylor/RL/rbe595-rl/final-project/src/tier3/aeriel_robot_final_project.py", line 359, in step
    self.reset_idx(reset_env_ids)
  File "/home/taylor/RL/rbe595-rl/final-project/src/tier3/aeriel_robot_final_project.py", line 387, in reset_idx
    self.env_asset_root_states[env_ids, :, 0:3] = self.env_asset_manager.asset_pose_tensor[env_ids, :, 0:3]
RuntimeError: shape mismatch: value tensor of shape [7, 3] cannot be broadcast to indexing result of shape [1, 3, 3]`
    * This issue is created from resetting an environment with a different number of objects. Essentially, without extensive modifications to the sim, the sim can randomize object placement and such, but it can't handle switching between numbers of bounds

* `Env asset has rigid body with more than 1 link`
  * To remedy the above issue, I created a urdf obstacle that contained all wall segments. Therefore, each reset would only contain one object. However, you can't use an object with multiple joints in the project, so a "compound" urdf maze setup won't work.

* `[Error] [carb.gym.plugin] Gym cuda error: invalid argument: ../../../source/plugins/carb/gym/impl/Gym/GymPhysXCuda.cu: 1038
[Error] [carb.gym.plugin] Incompatible shape of force tensor in function GymApplyRigidBodyForceTensors: expected (19, 3), received (1, 12, 3)`
    * This error came after I attemped to reset all the assets in the environment completely by reinstantiating all of the actors/objects. I don't know what on the backend is causing Gym to do this, as everything should be updated.