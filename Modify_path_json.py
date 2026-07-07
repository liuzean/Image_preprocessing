from __future__ import annotations

import json
from pathlib import Path


def build_new_name(json_path: Path, current_name: object) -> str:
    suffix = ""
    if isinstance(current_name, str):
        suffix = Path(current_name).suffix
    return f"{json_path.stem}{suffix}"


def update_json_file(json_path: Path, target_dir: Path) -> None:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")
    if not isinstance(data.get("info"), dict):
        raise ValueError(f'JSON must contain an "info" object: {json_path}')

    info = data["info"]
    info["folder"] = str(target_dir)
    info["name"] = build_new_name(json_path, info.get("name"))

    data.pop("folder", None)
    data.pop("name", None)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def update_json_files(target_dir: Path) -> int:
    if not target_dir.exists() or not target_dir.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {target_dir}")

    count = 0
    for json_path in sorted(target_dir.glob("*.json")):
        update_json_file(json_path, target_dir)
        count += 1

    return count


def main(target_dir: Path) -> None:
    count = update_json_files(target_dir)
    print(f"Updated {count} JSON files in: {target_dir}")


if __name__ == "__main__":
    TARGET_DIR = Path(r"E:\projects\datasets\Power_box\Power_box_1")
    main(TARGET_DIR)
