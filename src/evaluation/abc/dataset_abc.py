from src.dataset.abc_eval_dataset import ABCWITHNONOPTIMIZEDLATENTCODESVAL


def setup_dataset(obj_dir: str, val_lmdb_path: str, value_range: int, resolution: int):

    val_dataset = ABCWITHNONOPTIMIZEDLATENTCODESVAL(
        obj_dir, val_lmdb_path, value_range, resolution
    )

    print("\n setup: val_dataset len: ", len(val_dataset))
    return val_dataset
