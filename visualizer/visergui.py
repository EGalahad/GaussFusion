"""Credit to https://github.com/WangFeng18/3d-gaussian-splatting"""
import torch
import numpy as np
import viser
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
    def __init__(self, device, viewer_port, init_offset=[0.0, 0.0, 0.0], init_scaling=1.0):
        self.device = device
        self.port = viewer_port
        self.gaussian_model = None

        self.need_update = False
        self.render_times = deque(maxlen=3)

        self.server = viser.ViserServer(port=self.port)
        self.resolution_slider = self.server.gui.add_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )

        self.x_slider_coarse = self.server.gui.add_slider(
            "Object X (Coarse)", min=-5.0, max=5.0, step=0.1, initial_value=0.0
        )
        self.x_slider_fine = self.server.gui.add_slider(
            "Object X (Fine)", min=-0.1, max=0.1, step=0.001, initial_value=0.0
        )

        self.y_slider_coarse = self.server.gui.add_slider(
            "Object Y (Coarse)", min=-5.0, max=5.0, step=0.1, initial_value=0.0
        )
        self.y_slider_fine = self.server.gui.add_slider(
            "Object Y (Fine)", min=-0.1, max=0.1, step=0.001, initial_value=0.0
        )

        self.z_slider_coarse = self.server.gui.add_slider(
            "Object Z (Coarse)", min=-5.0, max=5.0, step=0.1, initial_value=0.0
        )
        self.z_slider_fine = self.server.gui.add_slider(
            "Object Z (Fine)", min=-0.1, max=0.1, step=0.001, initial_value=0.0
        )

        self.scale_slider = self.server.gui.add_slider(
            "Object Scale", min=0.1, max=5.0, step=0.1, initial_value=1.0
        )
        
        self.object_offset = self.server.gui.add_text(
            "Object Offset", initial_value="0.00 0.00 0.00", disabled=True
        )
        
        self.init_offset = init_offset
        self.init_scaling = init_scaling

        self.fps = self.server.gui.add_text("FPS", initial_value="-1", disabled=True)
        self.camera_pos = self.server.gui.add_text(
            "Camera Position", initial_value="0 0 0", disabled=True
        )

        self.c2ws = []
        self.camera_infos = []

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.x_slider_coarse.on_update
        @self.x_slider_fine.on_update
        @self.y_slider_coarse.on_update
        @self.y_slider_fine.on_update
        @self.z_slider_coarse.on_update
        @self.z_slider_fine.on_update
        def _(_):
            self.update_object_position()

        @self.scale_slider.on_update
        def _(_):
            self.update_object_scale()

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

    def update_object_position(self):
        if self.gaussian_model is not None and isinstance(
            self.gaussian_model, HybridGaussianModel
        ):
            x = self.x_slider_coarse.value + self.x_slider_fine.value + self.init_offset[0]
            y = self.y_slider_coarse.value + self.y_slider_fine.value + self.init_offset[1]
            z = self.z_slider_coarse.value + self.z_slider_fine.value + self.init_offset[2]
            self.gaussian_model.set_offset([x, y, z])
            self.object_offset.value = f"{x:.2f} {y:.2f} {z:.2f}"
            self.need_update = True

    def update_object_scale(self):
        if self.gaussian_model is not None and isinstance(
            self.gaussian_model, HybridGaussianModel
        ):
            scale = self.scale_slider.value * self.init_scaling
            self.gaussian_model.set_scaling(scale)
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
                client.scene.set_background_image(out, format="png")

                self.render_times.append(interval)
                self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
                self.camera_pos.value = f"{camera.position[0]:.2f} {camera.position[1]:.2f} {camera.position[2]:.2f}"
                # use the ray through camera center to represent the orientation


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run HybridGaussianModel with Viser GUI")
    
    parser.add_argument("--bg_path", type=str, required=True, help="Path to the background PLY file")
    parser.add_argument("--obj_path", type=str, required=True, help="Path to the object PLY file")
    parser.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], 
                        help="Initial offset for the object (x y z)")
    parser.add_argument("--scaling", type=float, default=1.0, 
                        help="Initial scaling for the object")
    parser.add_argument("--port", type=int, default=6789, 
                        help="Port number for the Viser server")

    args = parser.parse_args()

    gm = HybridGaussianModel(3)
    gm.load_ply(args.bg_path, args.obj_path)

    gui = ViserViewer(device="cuda", viewer_port=args.port, init_offset=args.offset, init_scaling=args.scaling)
    gui.set_renderer(gm)

    try:
        while True:
            gui.update()
    except KeyboardInterrupt:
        pass
