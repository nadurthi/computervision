import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class PointCloudHoverViewer:
    def __init__(self, pcd):
        self.pcd = pcd
        self.app = gui.Application.instance
        self.app.initialize()

        # Create main window and 3D scene widget
        self.window = self.app.create_window("Open3D Point Cloud Hover Viewer", 1024, 768)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d)

        # Create a floating label to show hovered coordinates
        self.info_label = gui.Label("Hover over a point to see (X, Y, Z) coordinates")
        self.info_label.background_color = gui.Color(0.1, 0.1, 0.1, 0.8)
        self.info_label.text_color = gui.Color(1.0, 1.0, 1.0)
        self.window.add_child(self.info_label)

        # Add the Point Cloud
        pcd_mat = rendering.MaterialRecord()
        pcd_mat.shader = "defaultUnlit"
        pcd_mat.point_size = 4.0
        self.widget3d.scene.add_geometry("point_cloud", self.pcd, pcd_mat)

        # Add a Reference Coordinate Frame (Red=X, Green=Y, Blue=Z)
        bounds = self.pcd.get_axis_aligned_bounding_box()
        frame_size = np.linalg.norm(bounds.get_extent()) * 0.25  # Scale relative to cloud size
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(frame_size, 0.1), origin=[0, 0, 0]
        )
        frame_mat = rendering.MaterialRecord()
        frame_mat.shader = "defaultLit"
        self.widget3d.scene.add_geometry("coord_frame", coord_frame, frame_mat)

        # Configure camera and lighting
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        self.widget3d.scene.set_background([0.05, 0.05, 0.05, 1.0])

        # Register layout and mouse callbacks
        self.window.set_on_layout(self._on_layout)
        self.widget3d.set_on_mouse(self._on_mouse)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.widget3d.frame = r
        pref = self.info_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.info_label.frame = gui.Rect(r.x + 15, r.y + 15, pref.width + 20, pref.height + 10)

    def _on_mouse(self, event):
        # Trigger coordinate lookup whenever the mouse moves
        if event.type == gui.MouseEvent.Type.MOVE:
            def depth_callback(depth_image):
                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                depth_array = np.asarray(depth_image)

                if 0 <= y < depth_array.shape[0] and 0 <= x < depth_array.shape[1]:
                    depth = depth_array[y, x]
                    if depth < 1.0:  # 1.0 means background (no geometry hit)
                        world_coord = self.widget3d.scene.camera.unproject(
                            x, y, depth,
                            self.widget3d.frame.width,
                            self.widget3d.frame.height
                        )
                        self.info_label.text = (
                            f" X: {world_coord[0]:.4f} | "
                            f"Y: {world_coord[1]:.4f} | "
                            f"Z: {world_coord[2]:.4f} "
                        )
                    else:
                        self.info_label.text = " Hover over a point to see (X, Y, Z) coordinates "
                self.window.set_needs_layout()

            self.widget3d.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def run(self):
        self.app.run()

if __name__ == "__main__":
    # Load sample data (replace with o3d.io.read_point_cloud("your_file.pcd"))
    pcd = o3d.io.read_point_cloud("ensenso_capture4.ply")

    viewer = PointCloudHoverViewer(pcd)
    viewer.run()
