import pandas as pd
import os
from datetime import datetime, timedelta
from owl_highlighter import OWLHighlighter
from owl_highlighter.models import Detection, VideoProcessingResult
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import toml
import re

class DetectionEvaluator:
    def __init__(self, annotation_csv: str, video_dir: str, temporal_tolerance: float = 2.0):
        """
        Initialize the evaluator.
        
        Args:
            annotation_csv: Path to CSV file containing ground truth annotations
            video_dir: Directory containing the videos
            temporal_tolerance: Time window (in seconds) for matching detections
        """
        self.temporal_tolerance = temporal_tolerance
        self.video_dir = video_dir
        
        # Initialize OWL Highlighter
        self.detector = OWLHighlighter(score_threshold=0.1)
        
        # Load and preprocess annotations
        self.annotations = self._load_annotations(annotation_csv)
        
    def _load_annotations(self, csv_path: str) -> Dict[str, List[datetime]]:
        """Load and process SeaTube annotation CSV file."""
        print(f"Loading annotations from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Print some sample rows to understand the data better
        print("\nFirst few rows of annotations:")
        print(df[['Dive ID', 'Start Date', 'Comment', 'Taxonomy', 'Taxon', 'Taxon Path']].head(10).to_string())
        
        # Convert 'Start Date' to datetime and ensure UTC timezone
        df['Start Date'] = pd.to_datetime(df['Start Date']).dt.tz_localize(None)
        
        # Map numeric dive ID to EX format if needed
        dive_mapping = {'2853': 'EX2304'}  # Add more mappings as needed
        df['Formatted Dive ID'] = df['Dive ID'].astype(str).map(dive_mapping)
        
        # Group annotations by formatted dive ID
        annotations = {}
        for dive_id, group in df.groupby('Formatted Dive ID'):
            if pd.isna(dive_id):
                continue
            
            # Look for organisms in comments and taxonomy fields
            organism_annotations = group[
                (group['Comment'].str.contains('fish|shark|squid|jellyfish|cephalopod', 
                                             case=False, 
                                             na=False)) |
                (group['Taxonomy'].str.contains('fish|shark|squid|jellyfish|cephalopod', 
                                              case=False, 
                                              na=False)) |
                (group['Taxon Path'].str.contains('fish|shark|squid|jellyfish|cephalopod', 
                                                case=False, 
                                                na=False))
            ]
            
            if not organism_annotations.empty:
                annotations[str(dive_id)] = organism_annotations['Start Date'].tolist()
        
        print(f"\nLoaded annotations for {len(annotations)} dives")
        
        # Debug info
        if not annotations:
            print("\nWarning: No annotations were loaded!")
            print("\nUnique Dive IDs in CSV:", df['Dive ID'].unique().tolist())
            print("\nUnique Comments (sample):", df['Comment'].dropna().head(10).tolist())
        else:
            print("\nExample matched annotations:")
            for dive_id in list(annotations.keys())[:3]:
                print(f"\nDive {dive_id}:")
                print(f"Number of annotations: {len(annotations[dive_id])}")
                print("First few timestamps:", annotations[dive_id][:3])
            
            # Print sample of matched rows
            print("\nSample of matched annotations:")
            for dive_id in list(annotations.keys())[:2]:
                matched = df[df['Formatted Dive ID'] == dive_id].head(3)
                print(f"\nDive {dive_id}:")
                print(matched[['Start Date', 'Comment', 'Taxonomy', 'Taxon Path']].to_string())
        
        return annotations
    
    def evaluate_video(self, video_path: str, dive_id: str) -> Dict[str, float]:
        """Evaluate detector performance on a single video."""
        # Get ground truth annotations for this dive
        if dive_id not in self.annotations:
            print(f"No annotations found for dive {dive_id}")
            return None
        
        ground_truth = self.annotations[dive_id]
        video_timestamp = self._parse_video_timestamp(os.path.basename(video_path))
        
        # Debug timestamp comparison
        print(f"\nComparing timestamps for {os.path.basename(video_path)}:")
        print(f"Video timestamp: {video_timestamp}")
        print(f"First few ground truth timestamps: {ground_truth[:3]}")
        
        # Only evaluate videos within annotation time range
        if not any(abs((gt - video_timestamp).total_seconds()) < 300 for gt in ground_truth):
            print(f"Video {os.path.basename(video_path)} outside annotation timerange")
            return None
        
        # Run detector on video
        result = self.detector.process_video(video_path)
        
        # Calculate metrics with temporal matching
        metrics = self._calculate_temporal_metrics(
            result.detections,
            ground_truth,
            self.temporal_tolerance
        )
        
        return metrics
    
    def evaluate_all_videos(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all videos in the directory."""
        results = {}
        
        # Find all ROVHD videos
        video_files = [f for f in os.listdir(self.video_dir) 
                      if f.endswith(('.mp4', '.mov', '.avi')) and 'ROVHD' in f]
        
        print(f"\nFound {len(video_files)} ROVHD videos to evaluate")
        
        for video_file in tqdm(video_files, desc="Evaluating videos"):
            # Extract dive ID from filename
            dive_id = video_file.split('_')[0]  # Gets EX2304
            
            video_path = os.path.join(self.video_dir, video_file)
            result = self.evaluate_video(video_path, dive_id)
            
            if result is not None:
                results[video_file] = result
        
        return results
    
    def plot_results(self, results: Dict[str, Dict[str, float]], output_dir: str):
        """Generate plots of evaluation results."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Aggregate metrics
        precisions = [r['precision'] for r in results.values()]
        recalls = [r['recall'] for r in results.values()]
        f1_scores = [r['f1_score'] for r in results.values()]
        
        # Plot metrics distribution
        plt.figure(figsize=(10, 6))
        plt.boxplot([precisions, recalls, f1_scores], labels=['Precision', 'Recall', 'F1 Score'])
        plt.title('Distribution of Detection Metrics')
        plt.ylabel('Score')
        plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'))
        plt.close()
        
        # Save numerical results
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write("Overall Evaluation Results\n")
            f.write("========================\n\n")
            f.write(f"Number of videos evaluated: {len(results)}\n\n")
            f.write("Average Metrics:\n")
            f.write(f"Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}\n")
            f.write(f"Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}\n")
            f.write(f"F1 Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}\n\n")
            
            f.write("Per-Video Results:\n")
            for video, metrics in results.items():
                f.write(f"\n{video}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.3f}\n")

    def _calculate_temporal_metrics(
        self,
        detections: List[Detection],
        ground_truth: List[datetime],
        tolerance: float
    ) -> Dict[str, float]:
        """Calculate precision/recall with temporal matching."""
        true_positives = 0
        detection_times = [d.timestamp for d in detections]
        
        # Match detections to ground truth
        matched_gt = set()
        matched_det = set()
        
        for i, det_time in enumerate(detection_times):
            for j, gt_time in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                
                time_diff = abs((gt_time - pd.Timestamp(det_time)).total_seconds())
                if time_diff <= tolerance:
                    true_positives += 1
                    matched_gt.add(j)
                    matched_det.add(i)
                    break
        
        false_positives = len(detection_times) - true_positives
        false_negatives = len(ground_truth) - true_positives
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def _parse_video_timestamp(self, filename: str) -> datetime:
        """Extract timestamp from video filename."""
        # Extract timestamp portion (e.g., 20230715T163244Z)
        timestamp_match = re.search(r'(\d{8}T\d{6}Z)', filename)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            # Parse without timezone info to match annotation timestamps
            return datetime.strptime(timestamp_str, '%Y%m%dT%H%M%SZ')
        return None

def main():
    """Entry point for evaluation script."""
    # Find config file relative to package location
    package_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(package_dir), 'config.toml')
    
    try:
        config = toml.load(config_path)
    except FileNotFoundError:
        # Fallback to current directory
        config_path = os.path.join(os.getcwd(), 'config.toml')
        try:
            config = toml.load(config_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Could not find config.toml in package directory or current directory. "
                f"Looked in:\n1. {os.path.dirname(package_dir)}\n2. {os.getcwd()}"
            )
    
    # Get paths from config
    ANNOTATION_CSV = config['paths']['annotation_csv']
    VIDEO_DIR = config['paths']['video_dir']
    OUTPUT_DIR = config['paths']['evaluation_output_dir']
    
    # Run evaluation with config parameters
    evaluator = DetectionEvaluator(
        ANNOTATION_CSV, 
        VIDEO_DIR,
        temporal_tolerance=config['evaluation']['temporal_tolerance']
    )
    results = evaluator.evaluate_all_videos()
    evaluator.plot_results(results, OUTPUT_DIR)

if __name__ == "__main__":
    main()