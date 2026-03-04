import csv
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "combined_raw_filtered_tokens.csv"
INCLUDE_SUFFIX = "_filtered.csv"
EXCLUDED_DIR_NAMES = {"old reddit data"}


def _normalize_column_name(name: str) -> str:
	return name.strip().lower().replace("_", " ")


def _find_column_mapping(fieldnames: list[str]) -> dict[str, str] | None:
	normalized_to_original = {
		_normalize_column_name(field): field for field in fieldnames if field
	}

	required = {
		"raw text": normalized_to_original.get("raw text"),
		"filtered text": normalized_to_original.get("filtered text"),
		"tokens": normalized_to_original.get("tokens"),
		"date source": normalized_to_original.get("date") or normalized_to_original.get("published"),
	}

	if required["raw text"] and required["filtered text"] and required["tokens"]:
		return required
	return None


def _normalize_date(value: str) -> str:
	text = (value or "").strip()
	if not text:
		return ""

	try:
		return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
	except ValueError:
		pass

	try:
		return parsedate_to_datetime(text).date().isoformat()
	except (TypeError, ValueError):
		return ""


def combine_csvs() -> None:
	csv_files = sorted(BASE_DIR.rglob("*.csv"))
	combined_rows: list[dict[str, str]] = []
	unique_rows: list[dict[str, str]] = []
	included_files: list[Path] = []
	skipped_files: list[Path] = []

	for csv_path in csv_files:
		if csv_path == OUTPUT_FILE:
			continue

		if any(part.lower() in EXCLUDED_DIR_NAMES for part in csv_path.parts):
			skipped_files.append(csv_path)
			continue

		if not csv_path.name.lower().endswith(INCLUDE_SUFFIX):
			skipped_files.append(csv_path)
			continue

		try:
			with csv_path.open("r", encoding="utf-8-sig", newline="") as infile:
				reader = csv.DictReader(infile)
				if not reader.fieldnames:
					skipped_files.append(csv_path)
					continue

				mapping = _find_column_mapping(reader.fieldnames)
				if mapping is None:
					skipped_files.append(csv_path)
					continue

				for row in reader:
					date_value = ""
					if mapping["date source"]:
						date_value = _normalize_date(row.get(mapping["date source"]) or "")

					combined_rows.append(
						{
							"raw text": (row.get(mapping["raw text"]) or "").strip(),
							"filtered text": (row.get(mapping["filtered text"]) or "").strip(),
							"tokens": (row.get(mapping["tokens"]) or "").strip(),
							"date": date_value,
						}
					)

				included_files.append(csv_path)

		except UnicodeDecodeError:
			skipped_files.append(csv_path)

	seen_rows: set[tuple[str, str, str, str]] = set()
	for row in combined_rows:
		row_key = (row["raw text"], row["filtered text"], row["tokens"], row["date"])
		if row_key in seen_rows:
			continue
		seen_rows.add(row_key)
		unique_rows.append(row)

	with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=["raw text", "filtered text", "tokens", "date"])
		writer.writeheader()
		writer.writerows(unique_rows)

	print(f"Combined rows (before dedupe): {len(combined_rows)}")
	print(f"Unique rows (after dedupe): {len(unique_rows)}")
	print(f"Duplicates removed: {len(combined_rows) - len(unique_rows)}")
	print(f"Output file: {OUTPUT_FILE}")
	print(f"Included files: {len(included_files)}")
	print(f"Skipped files: {len(skipped_files)}")


if __name__ == "__main__":
	combine_csvs()
