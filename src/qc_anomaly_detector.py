#!/usr/bin/env python3
"""
Objective 2 Part B: Comprehensive Dataset Anomaly Detection & Quality Assessment
File: qc_anomaly_detector.py
Purpose: Identify and categorize all anomalies in CLEANED dataset patches
Author: Project Team
Date: Current Session
"""

import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qc_anomaly_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Comprehensive anomaly detection for medical imaging datasets"""
    
    def __init__(self, base_path: str):
        """Initialize the detector with dataset base path"""
        self.base_path = Path(base_path)
        self.anomalies = {
            'image_quality': [],
            'mask_quality': [],
            'annotation_quality': [],
            'file_integrity': [],
            'metadata_inconsistencies': [],
            'corruption': []
        }
        self.stats = {
            'total_patches': 0,
            'processed': 0,
            'errors': 0,
            'warnings': 0
        }
        
    def detect_blur(self, image_path: str) -> float:
        """
        Calculate image blur score using Laplacian variance
        Higher score = sharper image, Lower score = blurrier image
        """
        try:
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)
            laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
            blurred = np.abs(np.convolve(img_array.flatten(), laplacian.flatten())).reshape(img_array.shape)
            blur_score = np.var(blurred)
            return blur_score
        except Exception as e:
            logger.error(f"Error calculating blur for {image_path}: {e}")
            return -1
    
    def calculate_tissue_coverage(self, mask_path: str) -> float:
        """
        Calculate tissue coverage percentage from mask
        Returns percentage of pixels with nuclei annotation
        """
        try:
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            
            # For instance masks, any non-zero pixel indicates nucleus presence
            if len(mask_array.shape) == 3:
                tissue_pixels = np.any(mask_array != 0, axis=2).sum()
            else:
                tissue_pixels = (mask_array != 0).sum()
            
            total_pixels = mask_array.shape[0] * mask_array.shape[1]
            coverage = (tissue_pixels / total_pixels) * 100
            return coverage
        except Exception as e:
            logger.error(f"Error calculating tissue coverage for {mask_path}: {e}")
            return -1
    
    def check_file_integrity(self, image_path: str, mask_path: str, annotation_path: str) -> Dict:
        """Verify file triplet integrity and consistency"""
        issues = []
        
        # Check file existence
        if not os.path.exists(image_path):
            issues.append("Image file missing")
        if not os.path.exists(mask_path):
            issues.append("Mask file missing")
        if not os.path.exists(annotation_path):
            issues.append("Annotation file missing")
        
        if issues:
            return {'valid': False, 'issues': issues}
        
        try:
            # Check image dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Check mask dimensions
            mask = Image.open(mask_path)
            mask_width, mask_height = mask.size
            
            # Verify image-mask size consistency
            if (img_width, img_height) != (mask_width, mask_height):
                issues.append(f"Image-mask size mismatch: img={img.size}, mask={mask.size}")
            
            # Check image format
            if img.format != 'PNG':
                issues.append(f"Image format error: {img.format} (expected PNG)")
            
            # Check image mode (should be RGB for NuCLS)
            if img.mode not in ['RGB', 'RGBA']:
                issues.append(f"Image color mode error: {img.mode} (expected RGB/RGBA)")
            
            # Check annotation file validity
            try:
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        issues.append("Annotation file is empty")
            except Exception as e:
                issues.append(f"Annotation file read error: {e}")
            
            return {'valid': len(issues) == 0, 'issues': issues}
            
        except Exception as e:
            issues.append(f"File integrity check failed: {e}")
            return {'valid': False, 'issues': issues}
    
    def validate_annotation_format(self, annotation_path: str) -> Dict:
        """Validate CSV annotation format and content"""
        issues = []
        
        try:
            with open(annotation_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if len(rows) == 0:
                return {'valid': False, 'issues': ["Annotation file is empty"]}
            
            # Check expected columns (typical NuCLS format)
            expected_fields = ['nucleus_id', 'nucleus_type', 'x', 'y', 'w', 'h']
            
            for idx, row in enumerate(rows):
                if len(row) == 0:
                    issues.append(f"Empty row at line {idx}")
                    continue
                
                # Check for nucleus_type validity (should match known types)
                if len(row) >= 2:
                    nucleus_type = row[1]
                    valid_types = [
                        'tumor_mitotic', 'tumor_nonMitotic',
                        'lymphocyte', 'macrophage', 'plasma_cell',
                        'nonTILnonMQ_stromal', 'other_nucleus', 'AMBIGUOUS'
                    ]
                    if nucleus_type not in valid_types:
                        issues.append(f"Invalid nucleus type '{nucleus_type}' at line {idx}")
            
            return {'valid': len(issues) == 0, 'issues': issues}
            
        except Exception as e:
            return {'valid': False, 'issues': [f"Annotation validation failed: {e}"]}
    
    def scan_patch(self, fov_id: str, image_path: str, mask_path: str, 
                   annotation_path: str) -> Dict:
        """Comprehensive scan of a single patch"""
        result = {
            'fov_id': fov_id,
            'timestamp': datetime.now().isoformat(),
            'image_quality': {},
            'mask_quality': {},
            'annotation_quality': {},
            'file_integrity': {},
            'anomalies': []
        }
        
        try:
            # File integrity check
            integrity = self.check_file_integrity(image_path, mask_path, annotation_path)
            result['file_integrity'] = integrity
            
            if not integrity['valid']:
                result['anomalies'].extend([('file_integrity', issue) for issue in integrity['issues']])
                self.anomalies['file_integrity'].append(result)
                return result
            
            # Image quality metrics
            blur_score = self.detect_blur(image_path)
            result['image_quality']['blur_score'] = blur_score
            
            if blur_score < 100:
                result['anomalies'].append(('image_quality', f"High blur detected (score={blur_score:.2f})"))
                self.anomalies['image_quality'].append(result)
            
            # Check image metadata
            img = Image.open(image_path)
            result['image_quality']['dimensions'] = img.size
            result['image_quality']['mode'] = img.mode
            
            # Mask quality metrics
            tissue_coverage = self.calculate_tissue_coverage(mask_path)
            result['mask_quality']['tissue_coverage_percent'] = tissue_coverage
            
            if tissue_coverage < 5:
                result['anomalies'].append(('mask_quality', f"Very low tissue coverage ({tissue_coverage:.1f}%)"))
                self.anomalies['mask_quality'].append(result)
            elif tissue_coverage > 95:
                result['anomalies'].append(('mask_quality', f"Unusually high tissue coverage ({tissue_coverage:.1f}%)"))
            
            # Annotation quality check
            annotation_check = self.validate_annotation_format(annotation_path)
            result['annotation_quality'] = annotation_check
            
            if not annotation_check['valid']:
                result['anomalies'].extend([('annotation_quality', issue) 
                                           for issue in annotation_check['issues']])
                self.anomalies['annotation_quality'].append(result)
            
            self.stats['processed'] += 1
            
        except Exception as e:
            result['anomalies'].append(('corruption', str(e)))
            self.anomalies['corruption'].append(result)
            self.stats['errors'] += 1
            logger.error(f"Error scanning patch {fov_id}: {e}")
        
        return result
    
    def scan_dataset(self, cleaned_dataset_path: str) -> Dict:
        """Scan entire CLEANED dataset for anomalies"""
        logger.info("Starting comprehensive dataset scan...")
        
        cleaned_path = Path(cleaned_dataset_path)
        results = []
        
        for category in ['normal', 'tumor']:
            category_path = cleaned_path / category
            
            if not category_path.exists():
                logger.warning(f"Category directory not found: {category_path}")
                continue
            
            logger.info(f"Scanning {category} patches...")
            
            for image_file in sorted(category_path.glob('*.png')):
                fov_id = image_file.stem
                
                # Construct corresponding paths
                mask_file = image_file.parent / f"{fov_id}_mask.png"
                annotation_file = image_file.parent / f"{fov_id}.csv"
                
                self.stats['total_patches'] += 1
                
                if self.stats['total_patches'] % 100 == 0:
                    logger.info(f"Processed {self.stats['total_patches']} patches...")
                
                result = self.scan_patch(
                    fov_id,
                    str(image_file),
                    str(mask_file),
                    str(annotation_file)
                )
                results.append(result)
        
        logger.info(f"Scan complete. Processed {self.stats['processed']} patches with {self.stats['errors']} errors")
        
        return {
            'scan_timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'anomalies_summary': {
                'image_quality': len(self.anomalies['image_quality']),
                'mask_quality': len(self.anomalies['mask_quality']),
                'annotation_quality': len(self.anomalies['annotation_quality']),
                'file_integrity': len(self.anomalies['file_integrity']),
                'metadata_inconsistencies': len(self.anomalies['metadata_inconsistencies']),
                'corruption': len(self.anomalies['corruption'])
            },
            'detailed_results': results,
            'anomalies': self.anomalies
        }
    
    def generate_report(self, scan_results: Dict, output_path: str) -> None:
        """Generate comprehensive anomaly report"""
        logger.info(f"Generating anomaly report to {output_path}")
        
        report = {
            'title': 'Objective 2 Part B: Dataset Anomaly Detection Report',
            'generated': datetime.now().isoformat(),
            'summary': {
                'total_patches_scanned': scan_results['statistics']['total_patches'],
                'patches_processed': scan_results['statistics']['processed'],
                'scan_errors': scan_results['statistics']['errors'],
                'anomalies_found': sum(scan_results['anomalies_summary'].values())
            },
            'anomalies_by_category': scan_results['anomalies_summary'],
            'detailed_anomalies': scan_results['anomalies'],
            'patches_with_issues': [
                r for r in scan_results['detailed_results'] if r['anomalies']
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("ANOMALY DETECTION SUMMARY")
        print("="*60)
        print(f"Total Patches Scanned: {report['summary']['total_patches_scanned']}")
        print(f"Successfully Processed: {report['summary']['patches_processed']}")
        print(f"Scan Errors: {report['summary']['scan_errors']}")
        print(f"Total Anomalies Found: {report['summary']['anomalies_found']}")
        print("\nAnomalies by Category:")
        for category, count in scan_results['anomalies_summary'].items():
            print(f"  - {category}: {count}")
        print("="*60 + "\n")


def main():
    """Main execution"""
    # Define paths
    project_root = Path(__file__).parent.parent
    cleaned_dataset_path = project_root / 'data' / 'PATCHES' / 'CLEANED'
    output_report = project_root / 'result' / 'anomaly_detection_report.json'
    
    # Initialize detector
    detector = AnomalyDetector(str(project_root))
    
    # Run scan
    scan_results = detector.scan_dataset(str(cleaned_dataset_path))
    
    # Generate report
    output_report.parent.mkdir(parents=True, exist_ok=True)
    detector.generate_report(scan_results, str(output_report))
    
    logger.info("Anomaly detection complete!")


if __name__ == '__main__':
    main()
