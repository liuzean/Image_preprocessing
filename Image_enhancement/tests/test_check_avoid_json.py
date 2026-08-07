from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(r"E:\projects\datasets\Power_box\old\results\results\Defective_free\11")
TARGET_FOLDERS = (
	"Power_box_1short",
	"Power_box_2long",
	"Power_box_3long",
	"Power_box_4long",
	"Power_box_5long",
	"Power_box_6short",
)
AVOIDED_AREA_KEY = "the_avoided_area"


def normalize_category(value: object) -> str:
	text = str(value).strip().lower().replace(" ", "_")
	return "_".join(part for part in text.split("_") if part)


def read_json(json_path: Path) -> dict:
	with json_path.open("r", encoding="utf-8") as file:
		data = json.load(file)
	if not isinstance(data, dict):
		raise TypeError(f"JSON root must be an object: {json_path}")
	if not isinstance(data.get("objects"), list):
		raise TypeError(f'JSON must contain an "objects" list: {json_path}')
	return data


def has_avoided_area(json_path: Path) -> bool:
	data = read_json(json_path)
	for item in data.get("objects", []):
		if not isinstance(item, dict):
			continue
		if normalize_category(item.get("category")) == AVOIDED_AREA_KEY:
			return True
	return False


def collect_json_files(folder_path: Path) -> list[Path]:
	return sorted(path for path in folder_path.rglob("*.json") if path.is_file())


def main() -> None:
	if not ROOT_DIR.is_dir():
		raise NotADirectoryError(f"Root directory does not exist: {ROOT_DIR}")

	missing_files: list[Path] = []
	checked_count = 0

	for folder_name in TARGET_FOLDERS:
		folder_path = ROOT_DIR / folder_name
		if not folder_path.is_dir():
			raise NotADirectoryError(f"Target folder does not exist: {folder_path}")

		for json_path in collect_json_files(folder_path):
			checked_count += 1
			try:
				if not has_avoided_area(json_path):
					missing_files.append(json_path)
			except (ValueError, TypeError, json.JSONDecodeError) as exc:
				print(f"✗ Error reading {json_path}: {exc}")
				missing_files.append(json_path)

	print(f"Checked {checked_count} JSON file(s).")
	if missing_files:
		print("Missing the_avoided_area in:")
		for json_path in missing_files:
			print(json_path)
	else:
		print("All JSON files contain the_avoided_area.")


if __name__ == "__main__":
	main()
