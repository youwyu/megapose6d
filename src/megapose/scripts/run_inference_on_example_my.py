# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
import cv2

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)


def load_observation(
    example_dir: Path,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    """
    只加载 RGB 和相机参数，强制忽略深度图。
    """
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    # 读取 RGB 图像
    rgb = np.array(Image.open(example_dir / "image_rgb.png"), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    # 强制不加载深度图
    depth = None 

    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir)
    # 这里的 depth 为 None
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(
    example_dir: Path,
) -> DetectionsType:
    input_object_data = load_object_data(example_dir / "inputs/object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    # 这里假设 mesh 单位是 mm，如果你的模型很大或很小，需要检查这里
    mesh_units = "m" 
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_detections_visualization(
    example_dir: Path,
) -> None:
    rgb, _, _ = load_observation(example_dir)
    detections = load_detections(example_dir)
    plotter = BokehPlotter()
    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    output_fn = example_dir / "visualizations" / "detections.png"
    output_fn.parent.mkdir(exist_ok=True)
    export_png(fig_det, filename=output_fn)
    logger.info(f"Wrote detections visualization: {output_fn}")
    return


def save_predictions(
    example_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "outputs" / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")
    return


def run_inference(
    example_dir: Path,
    model_name: str,
) -> None:
    model_info = NAMED_MODELS[model_name]
    
    # 检查一下选用的模型是否强制需要 Depth，如果是，打个警告
    if model_info.get("requires_depth", False):
        logger.warning(f"Model {model_name} usually expects depth, but we are running in RGB-only mode!")

    # 加载观测数据 (仅 RGB)
    observation = load_observation_tensor(example_dir).cuda()
    detections = load_detections(example_dir).cuda()
    object_dataset = make_object_dataset(example_dir)

    logger.info(f"Loading model {model_name}.")

    pose_estimator = load_named_model(model_name, object_dataset).cuda()
    logger.info(f"Running inference (RGB Only).")
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )
    
    save_predictions(example_dir, output)
    return


def make_output_visualization(
    example_dir: Path,
) -> None:
    # 加载 RGB，忽略 Depth
    rgb, _, camera_data = load_observation(example_dir)
    camera_data.TWC = Transform(np.eye(4))
    object_datas = load_object_data(example_dir / "outputs" / "object_data.json")
    object_dataset = make_object_dataset(example_dir)

    renderer = Panda3dSceneRenderer(object_dataset)

    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]

    # --- 使用 OpenCV 进行可视化保存 (比 Bokeh 更稳定且无需 Selenium) ---
    vis_dir = example_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # 1. 转换格式: RGB -> BGR
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    render_bgr = cv2.cvtColor(renderings.rgb, cv2.COLOR_RGB2BGR)

    # 2. 生成 Mesh Overlay (加权融合)
    overlay_img = cv2.addWeighted(img_bgr, 0.6, render_bgr, 0.4, 0)
    cv2.imwrite(str(vis_dir / "mesh_overlay.png"), overlay_img)

    # 3. 生成 Contour Overlay (轮廓图)
    contour_overlay_rgb = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    contour_overlay_bgr = cv2.cvtColor(contour_overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(vis_dir / "contour_overlay.png"), contour_overlay_bgr)

    # 4. 生成 All Results (拼接图)
    combined = np.hstack([img_bgr, contour_overlay_bgr, overlay_img])
    cv2.imwrite(str(vis_dir / "all_results.png"), combined)

    logger.info(f"Wrote visualizations to {vis_dir} (RGB only mode)")
    return


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    # 默认模型保持为 RGB-multi-hypothesis，这是一个不需要 Depth 的模型
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    args = parser.parse_args()

    example_dir = LOCAL_DATA_DIR / "examples" / args.example_name

    if args.vis_detections:
        make_detections_visualization(example_dir)

    if args.run_inference:
        run_inference(example_dir, args.model)

    if args.vis_outputs:
        make_output_visualization(example_dir)