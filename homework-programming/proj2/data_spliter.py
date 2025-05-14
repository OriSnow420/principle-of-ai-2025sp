"""Splits the given data
Note: It's an AI-generated file
"""

import os
import shutil
import random


def split_images_into_ratios(
    source_dir, output_dir, split_ratio, split_names=None, seed=None
):
    """
    将源文件夹中的图片按比例分割到指定份数的目标文件夹中

    参数:
        source_dir (str): 源文件夹路径，包含多个子文件夹的图片
        output_dir (str): 输出根目录，分割后的文件夹将保存在此
        split_ratio (list): 分割比例列表，如 [7, 2, 1] 表示分成三份比例为7:2:1
        split_names (list): 各份的名称列表，长度需与split_ratio一致（默认为split_0, split_1...）
        seed (int): 随机种子（可选）
    """
    if seed is not None:
        random.seed(seed)

    if split_names is None:
        split_names = [f"split_{i}" for i in range(len(split_ratio))]
    elif len(split_names) != len(split_ratio):
        raise ValueError("split_names的长度必须与split_ratio相同")

    class_dirs = [
        d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))
    ]

    for class_dir in class_dirs:
        class_path = os.path.join(source_dir, class_dir)
        images = []
        for f in os.listdir(class_path):
            file_path = os.path.join(class_path, f)
            if os.path.isfile(file_path) and f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".bmp")
            ):
                images.append(f)

        random.shuffle(images)
        total = len(images)
        if total == 0:
            continue

        sum_ratio = sum(split_ratio)
        if sum_ratio == 0:
            raise ValueError("分割比例总和不能为0")

        split_counts = [(r / sum_ratio) * total for r in split_ratio]
        integer_counts = [int(c) for c in split_counts]
        remainder = total - sum(integer_counts)

        decimal_parts = [
            split_counts[i] - integer_counts[i] for i in range(len(split_ratio))
        ]
        sorted_indices = sorted(
            range(len(decimal_parts)), key=lambda i: decimal_parts[i], reverse=True
        )

        for i in range(remainder):
            integer_counts[sorted_indices[i]] += 1

        splits = []
        current = 0
        for count in integer_counts:
            splits.append(images[current : current + count])
            current += count

        for i, split in enumerate(splits):
            split_dir = os.path.join(output_dir, split_names[i], class_dir)
            os.makedirs(split_dir, exist_ok=True)
            for img in split:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, required=True, help="源文件夹路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出文件夹路径")
    parser.add_argument(
        "--split-ratio", type=int, nargs="+", required=True, help="分割比例（如7 2 1）"
    )
    parser.add_argument(
        "--split-names", type=str, nargs="+", help="各份名称（如train val test）"
    )
    parser.add_argument("--seed", type=int, help="随机种子")
    args = parser.parse_args()

    split_images_into_ratios(
        args.source_dir, args.output_dir, args.split_ratio, args.split_names, args.seed
    )
