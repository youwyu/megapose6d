# Standard Library
import argparse
import json
import os
import glob
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
import cv2
from PIL import Image

# MegaPose (保持原有 Import)
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
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# 1. 修改：数据加载函数，支持传入具体的 image_path
# -----------------------------------------------------------------------------

def load_observation(
    example_dir: Path,
    image_filename: str = "image_rgb.png" # 默认为单帧模式的文件名
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    """
    加载指定路径的 RGB 图像和通用的相机参数
    """
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    # 修改：支持读取特定的图片文件
    img_path = example_dir / image_filename
    if not img_path.exists():
        # 尝试去 video 子文件夹找 (如果用户整理了数据集结构)
        img_path = example_dir / "video" / image_filename
    
    assert img_path.exists(), f"Image not found: {img_path}"

    rgb = np.array(Image.open(img_path), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution
    depth = None 
    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path,
    image_filename: str = "image_rgb.png"
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, image_filename)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation

# ... (load_object_data, load_detections, make_object_dataset 保持不变) ...
def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data

def load_detections(example_dir: Path) -> DetectionsType:
    input_object_data = load_object_data(example_dir / "inputs/object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections

def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "m" 
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                mesh_path = fn
        if mesh_path:
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

# -----------------------------------------------------------------------------
# 2. 新增：可视化单帧结果并保存 (为了 Tracking 序列)
# -----------------------------------------------------------------------------

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


def save_tracking_predictions(
    example_dir: Path,
    pose_estimates: PoseEstimatesType,
    frame_name: str,
    refiner_iterations: int,
) -> None:
    """
    保存单帧的预测结果，文件名与图像文件名对应。
    例如图像是 00001.png，保存为 outputs/trajectory/00001.json
    """
    labels = pose_estimates[f'iteration={refiner_iterations}'].infos['label']
    poses = pose_estimates[f'iteration={refiner_iterations}'].poses.cpu().numpy()
    
    # 构建 ObjectData
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    
    # 序列化
    object_data_json = json.dumps([x.to_json() for x in object_data], indent=2)
    
    # 建立专门的子目录来存放轨迹，避免混乱
    output_dir = example_dir / "outputs" / "trajectory"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用 frame_name (即图片的文件名，不含后缀) 作为 json 文件名
    # 例如: frame_name="00015" -> output_fn=".../00015.json"
    output_fn = output_dir / f"{frame_name}.json"
    
    output_fn.write_text(object_data_json)
    # logger.info(f"Saved pose to {output_fn}") # 如果觉得刷屏可以注释掉
    return


def viz_one_frame_track(
    example_dir: Path, 
    image_filename: str, 
    pose_estimates: PoseEstimatesType, 
    object_dataset: RigidObjectDataset,
    output_dir: Path,
    refiner_iterations: int = 1,
):
    """
    渲染当前帧的结果并保存到 visualizations 文件夹
    """
    rgb, _, camera_data = load_observation(example_dir, image_filename)
    camera_data.TWC = Transform(np.eye(4))
    
    # 构建 ObjectData 用于渲染
    labels = pose_estimates[f'iteration={refiner_iterations}'].infos['label']
    poses = pose_estimates[f'iteration={refiner_iterations}'].poses.cpu().numpy()
    object_datas = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]

    renderer = Panda3dSceneRenderer(object_dataset)
    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [Panda3dLightData(light_type="ambient", color=((1.0, 1.0, 1.0, 1)))]
    
    renderings = renderer.render_scene(
        object_datas, [camera_data], light_datas,
        render_depth=False, render_binary_mask=False, render_normals=False, copy_arrays=True
    )[0]

    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    render_bgr = cv2.cvtColor(renderings.rgb, cv2.COLOR_RGB2BGR)
    overlay_img = cv2.addWeighted(img_bgr, 0.6, render_bgr, 0.4, 0)
    
    # 保存文件名保持与输入图片一致
    save_path = output_dir / f"track_{image_filename}"
    cv2.imwrite(str(save_path), overlay_img)
    # logger.info(f"Saved visualization: {save_path}")

def viz_one_frame(
    example_dir: Path, 
    image_filename: str, 
    pose_estimates: PoseEstimatesType, 
    object_dataset: RigidObjectDataset,
    output_dir: Path,
):
    """
    渲染当前帧的结果并保存到 visualizations 文件夹
    """
    rgb, _, camera_data = load_observation(example_dir, image_filename)
    camera_data.TWC = Transform(np.eye(4))
    
    # 构建 ObjectData 用于渲染
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_datas = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]

    renderer = Panda3dSceneRenderer(object_dataset)
    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [Panda3dLightData(light_type="ambient", color=((1.0, 1.0, 1.0, 1)))]
    
    renderings = renderer.render_scene(
        object_datas, [camera_data], light_datas,
        render_depth=False, render_binary_mask=False, render_normals=False, copy_arrays=True
    )[0]

    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    render_bgr = cv2.cvtColor(renderings.rgb, cv2.COLOR_RGB2BGR)
    overlay_img = cv2.addWeighted(img_bgr, 0.6, render_bgr, 0.4, 0)
    
    # 保存文件名保持与输入图片一致
    save_path = output_dir / f"track_{image_filename}"
    cv2.imwrite(str(save_path), overlay_img)
    # logger.info(f"Saved visualization: {save_path}")

# -----------------------------------------------------------------------------
# 3. 核心修改：Tracking 逻辑
# -----------------------------------------------------------------------------
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


def run_tracking(
    example_dir: Path,
    model_name: str,
):
    # 1. 准备数据和模型
    model_info = NAMED_MODELS[model_name]
    object_dataset = make_object_dataset(example_dir)
    logger.info(f"Loading model {model_name}...")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    # 2. 获取图片序列
    # 假设图片在 example_dir 根目录或者 video/ 子目录下，按文件名排序
    # 这里你需要根据你的实际文件结构修改 pattern，例如 "*.png" 或 "frame_*.jpg"
    image_extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    
    # 搜索策略：先看根目录，再看 video 子目录
    search_dir = example_dir
    if (example_dir / "video").exists():
        search_dir = example_dir / "video"
    
    for ext in image_extensions:
        image_files.extend(sorted(list(search_dir.glob(ext))))
    
    image_files = sorted(image_files, key=lambda x: x.name) # 确保按帧顺序排列
    
    if not image_files:
        logger.error(f"No images found in {search_dir}")
        return

    logger.info(f"Found {len(image_files)} frames for tracking.")

    # 准备输出目录
    vis_dir = example_dir / "visualizations_tracking"
    vis_dir.mkdir(exist_ok=True)

    # 3. 开始 Tracking 循环
    prev_pose_estimates = None
    
    for i, img_path in enumerate(image_files):
        img_name = img_path.name
        
        # 加载当前帧观测 (Tensor)
        observation = load_observation_tensor(example_dir, image_filename=img_name).cuda()

        current_pose_estimates = None

        if i == 0:
            # === Frame 0: Initialization (Coarse + Refine) ===
            logger.info(f"[Frame {i}] Initializing with Coarse Model...")
            
            # 仅在第一帧加载 Detections (bounding box)
            detections = load_detections(example_dir).cuda()
            
            # 运行完整 Pipeline (Coarse -> Refine)
            # inference_parameters通常包含 n_refiner_iterations
            output, _ = pose_estimator.run_inference_pipeline(
                observation, detections=detections, **model_info["inference_parameters"]
            )
            current_pose_estimates = output

            save_predictions(example_dir, current_pose_estimates)

            viz_one_frame(example_dir, img_name, current_pose_estimates, object_dataset, vis_dir)

            # 2. 更新状态：当前帧的输出变成下一帧的输入
            prev_pose_estimates = current_pose_estimates

        else:
            # === Frame N: Tracking (Refine Only) ===
            # logger.info(f"[Frame {i}] Tracking using Refiner...")
            
            # 使用上一帧的预测结果作为当前帧的初始猜测
            # 注意：Refiner 需要从 PoseEstimatesType 开始
            
            # 这里的关键是调用 refiner_model 而不是 run_inference_pipeline
            # refiner_model 的输入通常是 (observation, pose_estimates)
            # 我们直接复用上一帧的 pose，假设物体移动不大
            
            # 获取 Refiner 需要的迭代次数参数
            refiner_iterations = 1 # model_info["inference_parameters"].get("n_refiner_iterations", 1)
            # import pdb; pdb.set_trace()
            
            # 运行 Refiner
            # 注意：MegaPose 的 refiner_model 会返回 (predictions, debug_dict)
            predictions, _ = pose_estimator.forward_refiner(
                observation=observation,
                data_TCO_input=prev_pose_estimates,
                n_iterations=refiner_iterations
            )
            current_pose_estimates = predictions

            save_tracking_predictions(example_dir, current_pose_estimates, img_path.stem, refiner_iterations)

            # === 结束当前帧处理 ===
            # 1. 可视化
            # viz_one_frame_track(example_dir, img_name, current_pose_estimates, object_dataset, vis_dir, refiner_iterations)

            prev_pose_estimates = predictions[f"iteration={refiner_iterations}"]
        
        
        
        if i % 10 == 0:
            logger.info(f"Processed frame {i}/{len(image_files)}")

    logger.info(f"Tracking completed. Visualizations saved to {vis_dir}")
    
    # 可以在这里加一段代码把图片合成视频 (使用 ffmpeg 或 cv2)
    # ...

if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--run-tracking", action="store_true", help="Run in tracking mode (video)")
    
    args = parser.parse_args()

    example_dir = LOCAL_DATA_DIR / "examples" / args.example_name

    if args.run_tracking:
        run_tracking(example_dir, args.model)
    else:
        # 兼容旧的单帧逻辑
        if os.path.exists(example_dir / "image_rgb.png"):
             run_inference(example_dir, args.model)
        else:
             logger.warning("No standard image_rgb.png found, assuming tracking mode or check path.")
             run_tracking(example_dir, args.model)