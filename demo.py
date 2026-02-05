import numpy as np
import torch
import time
import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu, precision="32")

camera_pos = (0.25, 0.5, 0.25)
camera_lookat = (0.125, 0, 0.125)

viewer_options = gs.options.ViewerOptions(
        camera_pos=camera_pos,
        camera_lookat=camera_lookat,
        camera_fov=60,
        max_FPS=60,
    )
########################## create a scene ##########################
scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            viewer_options=viewer_options,
            show_viewer=True,
            vis_options=gs.options.VisOptions(
                show_world_frame=False
            ),
            profiling_options = gs.options.ProfilingOptions(
                show_FPS       = False,
            ),
        )

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

so_101 = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.MJCF(
            file="assets/robots/SO-ARM100/Simulation/SO101/so101_new_calib.xml",
            collision=True,
        ),
)

cam = scene.add_camera(
    pos=camera_pos,
    lookat=camera_lookat,
)

########################## build ##########################
scene.build()

qpos_tensor = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32, device=gs.device)
so_101.control_dofs_position(qpos_tensor)

while True:
    scene.step()
