import pandas as pd
import os
from datetime import datetime, timedelta
from owl_highlighter import OWLHighlighter
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        """Load and process annotation CSV file."""
        df = pd.read_csv(csv_path)
        
        # Convert 'Start Date' to datetime
        df['Start Date'] = pd.to_datetime(df['Start Date'])
        
        # Group annotations by video/dive
        annotations = {}
        for dive_id, group in df.groupby('Dive ID'):
            # Filter out non-organism annotations (like "ROV powered off")
            organism_annotations = group[group['Taxon'].notna()]
            if not organism_annotations.empty:
                annotations[str(dive_id)] = organism_annotations['Start Date'].tolist()
        
        return annotations
    
    def evaluate_video(self, video_path: str, dive_id: str) -> Dict[str, float]:
        """
        Evaluate detector performance on a single video.
        
        Args:
            video_path: Path to video file
            dive_id: ID of the dive/video for matching with annotations
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get ground truth annotations for this video
        if dive_id not in self.annotations:
            print(f"No annotations found for dive {dive_id}")
            return None
            
        ground_truth = self.annotations[dive_id]
        
        # Run detector on video
        result = self.detector.process_video(video_path)
        
        # Convert detections to timestamps
        detection_times = [d.timestamp for d in result.detections]
        
        # Calculate temporal overlap
        true_positives = 0
        false_positives = 0
        
        # For each detection, check if there's a matching ground truth annotation
        for det_time in detection_times:
            matched = False
            for gt_time in ground_truth:
                time_diff = abs((gt_time - pd.Timestamp(det_time)).total_seconds())
                if time_diff <= self.temporal_tolerance:
                    true_positives += 1
                    matched = True
                    break
            if not matched:
                false_positives += 1
        
        # Calculate false negatives (missed annotations)
        false_negatives = 0
        for gt_time in ground_truth:
            matched = False
            for det_time in detection_times:
                time_diff = abs((gt_time - pd.Timestamp(det_time)).total_seconds())
                if time_diff <= self.temporal_tolerance:
                    matched = True
                    break
            if not matched:
                false_negatives += 1
        
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
    
    def evaluate_all_videos(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all videos in the directory."""
        results = {}
        
        # Find all videos
        video_files = [f for f in os.listdir(self.video_dir) 
                      if f.endswith(('.mp4', '.mov', '.avi'))]
        
        for video_file in tqdm(video_files, desc="Evaluating videos"):
            # Extract dive ID from filename
            dive_id = video_file.split('_')[0]  # Adjust based on your naming convention
            
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

if __name__ == "__main__":
    # Configuration
    ANNOTATION_CSV = "path/to/annotations.csv"
    VIDEO_DIR = "path/to/videos"
    OUTPUT_DIR = "evaluation_results"
    
    # Run evaluation
    evaluator = DetectionEvaluator(ANNOTATION_CSV, VIDEO_DIR)
    results = evaluator.evaluate_all_videos()
    evaluator.plot_results(results, OUTPUT_DIR)