"""
VOC Dataset Preprocessing Script

This script processes Pascal VOC format datasets, normalizes class labels,
validates images, and creates stratified train/val/test splits.

Usage:
    python preprocess_voc_dataset.py --data_root /path/to/VOCdevkit [--output_dir /path/to/output]

Features:
- Normalizes inconsistent class names
- Validates image integrity
- Handles missing annotations
- Creates stratified dataset splits
- Generates comprehensive reports
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
from PIL import Image
import argparse
import json
import yaml
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "data_root": "VOCdevkit",
    "img_dir": "JPEGImages",
    "ann_dir": "Annotations",
    "output_dir": "output",
    "output_csv": "labels_clean.csv",
    "class_normalization": {
        "no_defect": "normal",
        "good": "normal",
        "ok": "normal",
        "Normal": "normal",
        "Crack": "crack",
        "cracks": "crack",
        "Porosity": "porosity",
        "Pore": "porosity",
        "Undercut": "undercut",
        "Slag": "slag_inclusion",
        "Inclusion": "slag_inclusion",
    },
    "default_label": "normal",
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42,
    "min_samples_per_class": 2
}

def load_config(config_path=None):
    """Load configuration from file or use defaults"""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            if config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported config file format: {config_path}")
                return config
                
            config.update(user_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    return config

def setup_directories(config):
    """Create necessary directories if they don't exist"""
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Set full paths
    config['img_dir'] = os.path.join(config['data_root'], config['img_dir'])
    config['ann_dir'] = os.path.join(config['data_root'], config['ann_dir'])
    config['output_csv'] = os.path.join(config['output_dir'], config['output_csv'])
    
    return config

def normalize_class(label, normalization_map):
    """Normalize class labels using the provided mapping"""
    label = label.strip().lower()
    return normalization_map.get(label, label)

def parse_annotation(xml_path, normalization_map):
    """Parse VOC annotation file and extract normalized class labels"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objs = []
        
        for obj in root.findall("object"):
            cls_element = obj.find("name")
            if cls_element is not None and cls_element.text:
                objs.append(normalize_class(cls_element.text, normalization_map))
        
        return objs
    except ET.ParseError:
        logger.error(f"Malformed XML in {xml_path}")
        return []
    except Exception as e:
        logger.error(f"Error parsing {xml_path}: {e}")
        return []

def is_image_valid(img_path):
    """Check if image file is valid and can be opened"""
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError, OSError) as e:
        logger.warning(f"Invalid image {img_path}: {e}")
        return False

def process_dataset(config):
    """Main function to process the dataset"""
    rows = []
    missing_xml = []
    bad_images = []
    
    # Check if directories exist
    if not os.path.exists(config['img_dir']):
        logger.error(f"Image directory does not exist: {config['img_dir']}")
        return None
    if not os.path.exists(config['ann_dir']):
        logger.error(f"Annotation directory does not exist: {config['ann_dir']}")
        return None
    
    # Get list of image files
    image_files = [f for f in os.listdir(config['img_dir']) 
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if not image_files:
        logger.error("No images found in the image directory")
        return None
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    for fname in tqdm(image_files, desc="Processing images"):
        stem = os.path.splitext(fname)[0]
        xml_path = os.path.join(config['ann_dir'], stem + ".xml")
        img_path = os.path.join(config['img_dir'], fname)

        # Check for missing XML
        if not os.path.exists(xml_path):
            missing_xml.append(fname)
            continue
            
        # Check image validity
        if not is_image_valid(img_path):
            bad_images.append(fname)
            continue

        # Parse annotations
        labels = parse_annotation(xml_path, config['class_normalization'])
        if not labels:
            labels = [config['default_label']]  # Use default if no labels found

        # Store unique sorted labels
        labels = sorted(set(labels))
        rows.append({"filename": fname, "label": ";".join(labels)})

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Remove classes with too few samples
    if config['min_samples_per_class'] > 1:
        initial_count = len(df)
        df = df.groupby('label').filter(lambda x: len(x) >= config['min_samples_per_class'])
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} samples from classes with fewer than {config['min_samples_per_class']} samples")

    # Save clean CSV
    df.to_csv(config['output_csv'], index=False)
    logger.info(f"Clean labels saved to: {config['output_csv']}")
    logger.info(f"Total valid images: {len(df)}")

    # Report issues
    if missing_xml:
        logger.warning(f"Missing annotations for {len(missing_xml)} images. "
                      f"First 5: {missing_xml[:5]}")
        with open(os.path.join(config['output_dir'], 'missing_annotations.txt'), 'w') as f:
            f.write("\n".join(missing_xml))
    
    if bad_images:
        logger.warning(f"Corrupted images detected: {len(bad_images)}. "
                      f"First 5: {bad_images[:5]}")
        with open(os.path.join(config['output_dir'], 'corrupted_images.txt'), 'w') as f:
            f.write("\n".join(bad_images))

    # Class distribution report
    all_labels = []
    for lbls in df["label"]:
        all_labels.extend(lbls.split(";"))
    
    counts = pd.Series(all_labels).value_counts()
    logger.info("\nClass distribution:\n" + str(counts))
    
    # Save class distribution
    counts.to_csv(os.path.join(config['output_dir'], 'class_distribution.csv'))
    
    # Create dataset splits if we have data
    if len(df) > 0:
        # Data splitting
        train_df, test_df = train_test_split(
            df, 
            test_size=config['test_size'], 
            random_state=config['random_state'], 
            stratify=df['label']
        )
        
        train_df, val_df = train_test_split(
            train_df, 
            test_size=config['val_size'], 
            random_state=config['random_state'], 
            stratify=train_df['label']
        )

        logger.info(f"Train size: {len(train_df)}")
        logger.info(f"Validation size: {len(val_df)}")
        logger.info(f"Test size: {len(test_df)}")

        # Save splits
        train_df.to_csv(os.path.join(config['output_dir'], "train.csv"), index=False)
        val_df.to_csv(os.path.join(config['output_dir'], "val.csv"), index=False)
        test_df.to_csv(os.path.join(config['output_dir'], "test.csv"), index=False)
        
        # Save split information
        split_info = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'total_size': len(df)
        }
        
        with open(os.path.join(config['output_dir'], 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=4)
    else:
        logger.error("No valid data found to create splits")
    
    return df

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Preprocess VOC format dataset')
    parser.add_argument('--data_root', type=str, default=DEFAULT_CONFIG['data_root'],
                        help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                        help='Output directory for processed files')
    parser.add_argument('--config', type=str, 
                        help='Path to config file (JSON or YAML)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.data_root:
        config['data_root'] = args.data_root
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Setup directories
    config = setup_directories(config)
    
    logger.info(f"Data root: {config['data_root']}")
    logger.info(f"Image directory: {config['img_dir']}")
    logger.info(f"Annotation directory: {config['ann_dir']}")
    logger.info(f"Output directory: {config['output_dir']}")
    
    # Process dataset
    process_dataset(config)

if __name__ == "__main__":
    main()