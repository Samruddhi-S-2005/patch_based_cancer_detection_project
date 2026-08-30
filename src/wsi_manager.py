"""
WSI Metadata Manager for CAMELYON16 & Compatible Histopathology Datasets
Recursively scans raw Whole Slide Images, extracts dimensional and resolution
metadata via OpenSlide, detects matching XML annotation files, performs reproducible
WSI-level train/validation/test splitting, and exports data/wsi_registry.csv.
"""

import os
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List
import openslide


SUPPORTED_WSI_EXTENSIONS = {
    ".tif", ".tiff", ".svs", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs", ".bif"
}

def find_matching_annotation(wsi_path: Path, raw_root: Path) -> Optional[Path]:
    """
    Search for a matching XML annotation file for a given WSI.
    Checks:
    1. Same directory as WSI (<slide_id>.xml)
    2. Sibling 'annotations' / 'lesion_annotations' / 'xml' directories
    3. Global search within the dataset or raw_root
    """
    slide_id = wsi_path.stem
    xml_name = f"{slide_id}.xml"

    # 1. Direct same folder
    same_dir_xml = wsi_path.with_suffix(".xml")
    if same_dir_xml.exists():
        return same_dir_xml

    # 2. Sibling annotation folders within the same dataset
    parent_dir = wsi_path.parent
    possible_dirs = [
        parent_dir / "annotations",
        parent_dir / "lesion_annotations",
        parent_dir / "xml",
        parent_dir.parent / "annotations",
        parent_dir.parent / "lesion_annotations",
        parent_dir.parent / "xml",
    ]

    for candidate_dir in possible_dirs:
        if candidate_dir.exists() and candidate_dir.is_dir():
            candidate_file = candidate_dir / xml_name
            if candidate_file.exists():
                return candidate_file

    # 3. Recursive search inside raw_root for <slide_id>.xml (case-insensitive)
    for xml_path in raw_root.rglob("*.xml"):
        if xml_path.stem.lower() == slide_id.lower():
            return xml_path

    return None


def extract_wsi_metadata(wsi_path: Path, base_dir: Path, raw_root: Path) -> Dict[str, Any]:
    """
    Open WSI with OpenSlide and extract dimensional, resolution, and level information.
    """
    rel_path = wsi_path.relative_to(base_dir).as_posix()
    slide_id = wsi_path.stem

    # Determine dataset name (first folder under data/RAW)
    rel_to_raw = wsi_path.relative_to(raw_root)
    dataset_name = rel_to_raw.parts[0] if len(rel_to_raw.parts) > 1 else "camelyon16"

    # Determine WSI category / type and class label
    path_str_lower = wsi_path.as_posix().lower()
    if "normal" in path_str_lower:
        class_name = "normal"
        class_label = 0
    elif "tumor" in path_str_lower:
        class_name = "tumor"
        class_label = 1
    elif "test" in path_str_lower:
        class_name = "test"
        class_label = -1
    else:
        class_name = "unknown"
        class_label = -1

    # Find matching annotation XML if available
    annotation_file = find_matching_annotation(wsi_path, raw_root)
    annotation_rel_path = annotation_file.relative_to(base_dir).as_posix() if annotation_file else ""
    has_annotation = bool(annotation_file)

    metadata: Dict[str, Any] = {
        "wsi_id": slide_id,
        "filename": wsi_path.name,
        "path": rel_path,
        "dataset": dataset_name,
        "class_label": class_label,
        "class_name": class_name,
        "annotation_path": annotation_rel_path,
        "annotation_available": has_annotation,
        "split": "unassigned",
        "width_lvl0": 0,
        "height_lvl0": 0,
        "level_count": 0,
        "level_dimensions": "",
        "level_downsamples": "",
        "mpp_x": "",
        "mpp_y": "",
        "objective_power": "",
        "vendor": "",
        "status": "VALID",
        "error_msg": ""
    }

    try:
        slide = openslide.OpenSlide(str(wsi_path))
        width, height = slide.dimensions
        metadata["width_lvl0"] = width
        metadata["height_lvl0"] = height
        metadata["level_count"] = slide.level_count
        metadata["level_dimensions"] = str(slide.level_dimensions)
        metadata["level_downsamples"] = str([round(d, 2) for d in slide.level_downsamples])
        metadata["mpp_x"] = slide.properties.get("openslide.mpp-x", "")
        metadata["mpp_y"] = slide.properties.get("openslide.mpp-y", "")
        metadata["objective_power"] = slide.properties.get("openslide.objective-power", "")
        metadata["vendor"] = slide.properties.get("openslide.vendor", "")
        slide.close()
    except Exception as exc:
        metadata["status"] = "ERROR"
        metadata["error_msg"] = str(exc)

    return metadata


def assign_wsi_splits(results: List[Dict[str, Any]]) -> None:
    """
    Keep WSI splits unassigned until the verification phase is complete.

    This compatibility function intentionally does not split WSIs or patches.
    """
    for result in results:
        result["split"] = "unassigned"


def write_annotation_audit_report(
    results: List[Dict[str, Any]], raw_root: Path, report_path: Path
) -> None:
    """Write an explicit audit of downloaded tumor WSIs and XML annotations."""
    tumor_ids = {
        result["wsi_id"].lower()
        for result in results
        if result["class_name"] == "tumor"
    }
    xml_paths = sorted(raw_root.rglob("*.xml"), key=lambda path: path.as_posix().lower())
    xml_by_id = {path.stem.lower(): path for path in xml_paths}
    matched_ids = sorted(tumor_ids & xml_by_id.keys())
    missing_ids = sorted(tumor_ids - xml_by_id.keys())
    extra_ids = sorted(xml_by_id.keys() - tumor_ids)

    lines = [
        "# CAMELYON16 Annotation Availability Audit",
        "",
        "This report is generated from the current files under `data/RAW`.",
        "No downloads, archive extraction, patch extraction, or model training are performed.",
        "",
        "## Summary",
        "",
        f"- Downloaded WSIs: {len(results)} (30 normal, 30 tumor expected)",
        f"- XML annotation files found: {len(xml_paths)}",
        f"- Downloaded tumor WSIs with matching XML: {len(matched_ids)} of {len(tumor_ids)}",
        f"- Downloaded tumor WSIs missing XML: {len(missing_ids)}",
        f"- XML files not matching a downloaded tumor WSI: {len(extra_ids)}",
        "- Normal WSIs require no lesion XML for slide-level normal labeling.",
        "",
        "## Downloaded Tumor WSIs With Matching Annotations",
        "",
        "| Slide | WSI path | Annotation path |",
        "| --- | --- | --- |",
    ]

    result_by_id = {result["wsi_id"].lower(): result for result in results}
    for slide_id in matched_ids:
        result = result_by_id[slide_id]
        lines.append(
            f"| `{result['wsi_id']}` | `{result['path']}` | `"
            f"{result['annotation_path']}` |"
        )

    lines.extend(["", "## Missing Annotations", ""])
    if missing_ids:
        lines.extend(f"- `{slide_id}`" for slide_id in missing_ids)
    else:
        lines.append("None.")

    lines.extend([
        "",
        "## XML Files For Other Slides",
        "",
        "These XML files are present locally but their slide IDs are not among the 30 downloaded tumor WSIs; they are not treated as matches.",
        "",
    ])
    if extra_ids:
        lines.extend(f"- `{slide_id}.xml`" for slide_id in extra_ids)
    else:
        lines.append("None.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scan_and_register_wsis(raw_dir: Path, base_dir: Path, output_csv: Path) -> List[Dict[str, Any]]:
    """
    Recursively scan raw_dir for WSI files, extract metadata, assign splits, and export CSV registry.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW directory not found at: {raw_dir}")

    # Discover all WSI files matching supported extensions
    wsi_files: List[Path] = []
    for ext in SUPPORTED_WSI_EXTENSIONS:
        wsi_files.extend(raw_dir.rglob(f"*{ext}"))
        wsi_files.extend(raw_dir.rglob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    wsi_files = sorted(list(set(wsi_files)), key=lambda p: p.name)

    results: List[Dict[str, Any]] = []
    for wsi_p in wsi_files:
        meta = extract_wsi_metadata(wsi_p, base_dir=base_dir, raw_root=raw_dir)
        results.append(meta)

    # Keep splits unassigned during WSI and annotation verification.
    assign_wsi_splits(results)

    # Save to CSV registry
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "wsi_id",
        "filename",
        "path",
        "dataset",
        "class_label",
        "class_name",
        "annotation_path",
        "annotation_available",
        "split",
        "width_lvl0",
        "height_lvl0",
        "level_count",
        "level_dimensions",
        "level_downsamples",
        "mpp_x",
        "mpp_y",
        "objective_power",
        "vendor",
        "status",
        "error_msg"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    return results


def print_registry_report(results: List[Dict[str, Any]], output_csv: Path) -> None:
    """Print a clean and informative console summary of the WSI registry."""
    total_wsis = len(results)
    normal_wsis = [r for r in results if r["class_name"] == "normal"]
    tumor_wsis = [r for r in results if r["class_name"] == "tumor"]
    valid_wsis = sum(1 for r in results if r["status"] == "VALID")
    error_wsis = sum(1 for r in results if r["status"] == "ERROR")
    with_annotations = sum(1 for r in results if r["annotation_available"])
    without_annotations = total_wsis - with_annotations

    print("\n" + "=" * 85)
    print("WHOLE SLIDE IMAGE (WSI) REGISTRY & SPLIT SUMMARY")
    print("=" * 85)
    print(f"Total WSIs detected        : {total_wsis}")
    print(f"  - Normal WSIs            : {len(normal_wsis)}")
    print(f"  - Tumor WSIs             : {len(tumor_wsis)}")
    print(f"Successfully validated     : {valid_wsis}")
    print(f"Read errors / Corrupted    : {error_wsis}")
    print(f"Tumor WSIs with XML Annots : {with_annotations} (of {len(tumor_wsis)} tumor slides)")
    print("-" * 85)
    print("WSI-LEVEL SPLIT             : unassigned (verification phase)")
    print(f"Annotation audit report      : {output_csv.parent / 'ANNOTATION_AVAILABILITY_REPORT.md'}")
    print("=" * 85)

    print(f"\n{'WSI ID':<15} {'CLASS':<8} {'SPLIT':<8} {'DIMENSIONS (LVL 0)':<22} {'LEVELS':<8} {'MPP (X)':<10} {'ANNOTATION':<12}")
    print("-" * 85)
    for r in results:
        dims_str = f"{r['width_lvl0']} x {r['height_lvl0']}"
        mpp_str = r['mpp_x'][:7] if r['mpp_x'] else "N/A"
        annot_str = "MATCHED" if r['annotation_available'] else "NONE (Normal)"
        print(f"{r['wsi_id']:<15} {r['class_name']:<8} {r['split']:<8} {dims_str:<22} {r['level_count']:<8} {mpp_str:<10} {annot_str:<12}")

    print("-" * 85)
    print(f"\n[OUTPUT] Updated registry saved successfully to:")
    print(f"  --> {output_csv.resolve()}")
    print("=" * 85 + "\n")


def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "RAW"
    output_csv = base_dir / "data" / "wsi_registry.csv"
    report_path = base_dir / "data" / "ANNOTATION_AVAILABILITY_REPORT.md"

    results = scan_and_register_wsis(raw_dir=raw_dir, base_dir=base_dir, output_csv=output_csv)
    write_annotation_audit_report(results, raw_root=raw_dir, report_path=report_path)
    print_registry_report(results, output_csv)


if __name__ == "__main__":
    main()
