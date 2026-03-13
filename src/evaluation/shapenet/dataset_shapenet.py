from src.dataset.shapenet_eval_dataset import (
    ShapeNetcorev1NormalizedValWithNonOptimizedLatentCodes,
)  # test


def setup_dataset(
    mesh_path,
    points_to_sample,
    query_number,
    lmdb_path,
    value_range,
    resolution,
    examples_per_epoch,
):
    val_dataset = ShapeNetcorev1NormalizedValWithNonOptimizedLatentCodes(
        mesh_path,
        points_to_sample,
        query_number,
        lmdb_path,
        value_range,
        resolution,
        examples_per_epoch,
    )

    print(len(val_dataset))
    return val_dataset
