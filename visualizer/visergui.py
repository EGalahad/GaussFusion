from threading import Thread
import torch
import numpy as np
import time
import viser
import viser.transforms as tf
from collections import deque
from gaussian_model import GaussianModel, HybridGaussianModel


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    # print("c2w:", c2w)
    c2w = c2w
    return c2w


def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    # print("w2c:", w2c)
    return w2c


class ViserViewer:
    def __init__(self, device, viewer_port):
        self.device = device
        self.port = viewer_port
        self.gaussian_model = None

        self.need_update = False
        self.render_times = deque(maxlen=3)

        self.server = viser.ViserServer(port=self.port)
        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )
        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)
        self.camera_pos = self.server.add_gui_text(
            "Camera Position", initial_value="0 0 0", disabled=True
        )
        self.camera_ori = self.server.add_gui_text(
            "Camera Orientation", initial_value="0 0 0 1", disabled=True
        )

        self.c2ws = []
        self.camera_infos = []

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

    def set_renderer(self, gaussian_model: GaussianModel):
        self.gaussian_model = gaussian_model

    @torch.no_grad()
    def update(self):
        if self.need_update:
            for client in self.server.get_clients().values():
                camera = client.camera
                w2c = get_w2c(camera)
                try:
                    W = self.resolution_slider.value
                    H = int(self.resolution_slider.value / camera.aspect)
                    focal_x = W / 2 / np.tan(camera.fov / 2)
                    focal_y = H / 2 / np.tan(camera.fov / 2)

                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()

                    outputs = self.gaussian_model.render(
                        w2c=w2c,
                        intrinsics={
                            "width": W,
                            "height": H,
                            "focal_x": focal_x,
                            "focal_y": focal_y,
                        },
                    )
                    end_cuda.record()
                    torch.cuda.synchronize()
                    interval = start_cuda.elapsed_time(end_cuda) / 1000.0

                    out = (
                        outputs["render"]
                        .mul(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        .permute(1, 2, 0)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )
                except RuntimeError as e:
                    print(e)
                    interval = 1
                    continue
                client.set_background_image(out, format="jpeg")

                self.render_times.append(interval)
                self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
                self.camera_pos.value = f"{camera.position[0]:.3g} {camera.position[1]:.3g} {camera.position[2]:.3g}"
                # use the ray through camera center to represent the orientation
                ray = camera.wxyz[1:] * 2
                self.camera_ori.value = f"{ray[0]:.3g} {ray[1]:.3g} {ray[2]:.3g}"


# position = [1.1999999999999997, 9.992007221626408e-17, 1.6000000000000005]
# rotation = [[6.118467770994851e-17, 0.734803444627488, -0.6782801027330655], [1.0, 6.118467770994851e-17, -5.647816403995245e-17], [-5.647816403995245e-17, -0.6782801027330655, -0.734803444627488]]

# c2w = np.eye(4)
# c2w[:3, :3] = rotation
# c2w[:3, 3] = position

# w2c = np.linalg.inv(c2w)
# view_matrix = w2c.transpose()
# # print(w2c)
# # print(view_matrix)

# fx = fy = 800.
# width, height = 1024, 768

# intrinsics_matrix = np.eye(4)
# intrinsics_matrix[0, 0] = 2 * fx / width
# intrinsics_matrix[1, 1] = 2 * fy / height
# intrinsics_matrix[2, 3] = 1
# intrinsics_matrix[3, 3] = 0

# full_proj_tf = view_matrix @ intrinsics_matrix
# print(full_proj_tf * (full_proj_tf > 1e-3))

# exit(0)

if __name__ == "__main__":
    path_bg = "/home/elijah/Documents/cv_project/weng-wong-project-cv/train.ply"
    path_obj = "/home/elijah/Documents/cv_project/weng-wong-project-cv/object.ply"
    gm = HybridGaussianModel(3)
    gm.load_ply(path_bg, path_obj)
    gm.set_scaling(1.2)
    gm.set_offset([1, 0, 0])
    gui = ViserViewer(device="cuda", viewer_port=6789)
    gui.set_renderer(gm)
    while True:
        gui.update()
