# CAMELYON16 Annotation Availability Audit

This report is generated from the current files under `data/RAW`.
No downloads, archive extraction, patch extraction, or model training are performed.

## Summary

- Downloaded WSIs: 60 (30 normal, 30 tumor expected)
- XML annotation files found: 111
- Downloaded tumor WSIs with matching XML: 30 of 30
- Downloaded tumor WSIs missing XML: 0
- XML files not matching a downloaded tumor WSI: 81
- Normal WSIs require no lesion XML for slide-level normal labeling.

## Downloaded Tumor WSIs With Matching Annotations

| Slide | WSI path | Annotation path |
| --- | --- | --- |
| `tumor_005` | `data/RAW/camelyon16/tumor/tumor_005.tif` | `data/RAW/camelyon16/annotations/tumor_005.xml` |
| `tumor_016` | `data/RAW/camelyon16/tumor/tumor_016.tif` | `data/RAW/camelyon16/annotations/tumor_016.xml` |
| `tumor_019` | `data/RAW/camelyon16/tumor/tumor_019.tif` | `data/RAW/camelyon16/annotations/tumor_019.xml` |
| `tumor_031` | `data/RAW/camelyon16/tumor/tumor_031.tif` | `data/RAW/camelyon16/annotations/tumor_031.xml` |
| `tumor_034` | `data/RAW/camelyon16/tumor/tumor_034.tif` | `data/RAW/camelyon16/annotations/tumor_034.xml` |
| `tumor_035` | `data/RAW/camelyon16/tumor/tumor_035.tif` | `data/RAW/camelyon16/annotations/tumor_035.xml` |
| `tumor_038` | `data/RAW/camelyon16/tumor/tumor_038.tif` | `data/RAW/camelyon16/annotations/tumor_038.xml` |
| `tumor_057` | `data/RAW/camelyon16/tumor/tumor_057.tif` | `data/RAW/camelyon16/annotations/tumor_057.xml` |
| `tumor_058` | `data/RAW/camelyon16/tumor/tumor_058.tif` | `data/RAW/camelyon16/annotations/tumor_058.xml` |
| `tumor_059` | `data/RAW/camelyon16/tumor/tumor_059.tif` | `data/RAW/camelyon16/annotations/tumor_059.xml` |
| `tumor_065` | `data/RAW/camelyon16/tumor/tumor_065.tif` | `data/RAW/camelyon16/annotations/tumor_065.xml` |
| `tumor_067` | `data/RAW/camelyon16/tumor/tumor_067.tif` | `data/RAW/camelyon16/annotations/tumor_067.xml` |
| `tumor_075` | `data/RAW/camelyon16/tumor/tumor_075.tif` | `data/RAW/camelyon16/annotations/tumor_075.xml` |
| `tumor_081` | `data/RAW/camelyon16/tumor/tumor_081.tif` | `data/RAW/camelyon16/annotations/tumor_081.xml` |
| `tumor_082` | `data/RAW/camelyon16/tumor/tumor_082.tif` | `data/RAW/camelyon16/annotations/tumor_082.xml` |
| `tumor_084` | `data/RAW/camelyon16/tumor/tumor_084.tif` | `data/RAW/camelyon16/annotations/tumor_084.xml` |
| `tumor_087` | `data/RAW/camelyon16/tumor/tumor_087.tif` | `data/RAW/camelyon16/annotations/tumor_087.xml` |
| `tumor_091` | `data/RAW/camelyon16/tumor/tumor_091.tif` | `data/RAW/camelyon16/annotations/tumor_091.xml` |
| `tumor_093` | `data/RAW/camelyon16/tumor/tumor_093.tif` | `data/RAW/camelyon16/annotations/tumor_093.xml` |
| `tumor_094` | `data/RAW/camelyon16/tumor/tumor_094.tif` | `data/RAW/camelyon16/annotations/tumor_094.xml` |
| `tumor_096` | `data/RAW/camelyon16/tumor/tumor_096.tif` | `data/RAW/camelyon16/annotations/tumor_096.xml` |
| `tumor_097` | `data/RAW/camelyon16/tumor/tumor_097.tif` | `data/RAW/camelyon16/annotations/tumor_097.xml` |
| `tumor_098` | `data/RAW/camelyon16/tumor/tumor_098.tif` | `data/RAW/camelyon16/annotations/tumor_098.xml` |
| `tumor_099` | `data/RAW/camelyon16/tumor/tumor_099.tif` | `data/RAW/camelyon16/annotations/tumor_099.xml` |
| `tumor_100` | `data/RAW/camelyon16/tumor/tumor_100.tif` | `data/RAW/camelyon16/annotations/tumor_100.xml` |
| `tumor_101` | `data/RAW/camelyon16/tumor/tumor_101.tif` | `data/RAW/camelyon16/annotations/tumor_101.xml` |
| `tumor_105` | `data/RAW/camelyon16/tumor/tumor_105.tif` | `data/RAW/camelyon16/annotations/tumor_105.xml` |
| `tumor_107` | `data/RAW/camelyon16/tumor/tumor_107.tif` | `data/RAW/camelyon16/annotations/tumor_107.xml` |
| `tumor_109` | `data/RAW/camelyon16/tumor/tumor_109.tif` | `data/RAW/camelyon16/annotations/tumor_109.xml` |
| `tumor_110` | `data/RAW/camelyon16/tumor/tumor_110.tif` | `data/RAW/camelyon16/annotations/tumor_110.xml` |

## Missing Annotations

None.

## XML Files For Other Slides

These XML files are present locally but their slide IDs are not among the 30 downloaded tumor WSIs; they are not treated as matches.

- `tumor_001.xml`
- `tumor_002.xml`
- `tumor_003.xml`
- `tumor_004.xml`
- `tumor_006.xml`
- `tumor_007.xml`
- `tumor_008.xml`
- `tumor_009.xml`
- `tumor_010.xml`
- `tumor_011.xml`
- `tumor_012.xml`
- `tumor_013.xml`
- `tumor_014.xml`
- `tumor_015.xml`
- `tumor_017.xml`
- `tumor_018.xml`
- `tumor_020.xml`
- `tumor_021.xml`
- `tumor_022.xml`
- `tumor_023.xml`
- `tumor_024.xml`
- `tumor_025.xml`
- `tumor_026.xml`
- `tumor_027.xml`
- `tumor_028.xml`
- `tumor_029.xml`
- `tumor_030.xml`
- `tumor_032.xml`
- `tumor_033.xml`
- `tumor_036.xml`
- `tumor_037.xml`
- `tumor_039.xml`
- `tumor_040.xml`
- `tumor_041.xml`
- `tumor_042.xml`
- `tumor_043.xml`
- `tumor_044.xml`
- `tumor_045.xml`
- `tumor_046.xml`
- `tumor_047.xml`
- `tumor_048.xml`
- `tumor_049.xml`
- `tumor_050.xml`
- `tumor_051.xml`
- `tumor_052.xml`
- `tumor_053.xml`
- `tumor_054.xml`
- `tumor_055.xml`
- `tumor_056.xml`
- `tumor_060.xml`
- `tumor_061.xml`
- `tumor_062.xml`
- `tumor_063.xml`
- `tumor_064.xml`
- `tumor_066.xml`
- `tumor_068.xml`
- `tumor_069.xml`
- `tumor_070.xml`
- `tumor_071.xml`
- `tumor_072.xml`
- `tumor_073.xml`
- `tumor_074.xml`
- `tumor_076.xml`
- `tumor_077.xml`
- `tumor_078.xml`
- `tumor_079.xml`
- `tumor_080.xml`
- `tumor_083.xml`
- `tumor_085.xml`
- `tumor_086.xml`
- `tumor_088.xml`
- `tumor_089.xml`
- `tumor_090.xml`
- `tumor_092.xml`
- `tumor_095.xml`
- `tumor_102.xml`
- `tumor_103.xml`
- `tumor_104.xml`
- `tumor_106.xml`
- `tumor_108.xml`
- `tumor_111.xml`
