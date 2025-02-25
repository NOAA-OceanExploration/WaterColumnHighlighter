import pandas as pd
import os
from datetime import datetime, timedelta
from .highlighter import CritterDetector
from .models import Detection, VideoProcessingResult
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import toml
import re

class DetectionEvaluator:
    def __init__(self, annotation_csv: str, video_dir: str, config: dict):
        """Initialize evaluator.
        
        Args:
            annotation_csv: Path to SeaTube annotation CSV
            video_dir: Directory containing videos to evaluate
            config: Configuration dictionary from config.toml
        """
        self.video_dir = video_dir
        self.temporal_tolerance = config['evaluation']['temporal_tolerance']
        self.simplified_mode = config['evaluation'].get('simplified_mode', False)
        self.config = config  # Store the config dictionary
        
        # Initialize detector with config settings
        self.detector = CritterDetector(config=config)
        
        # Load and process annotations
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
        
        # Define terms to exclude for operational annotations
        operational_terms = [
            # ROV and vehicle terms
            'rov', 'vehicle', 'submersible', 'platform',
            
            # Camera and imaging equipment
            'camera', 'lens', 'light', 'strobe', 'laser', 'imaging',
            
            # Sampling equipment
            'sampler', 'niskin', 'bottle', 'collection', 'scoop', 'net',
            'trap', 'corer', 'basket', 'container',
            
            # Sensors and instruments
            'ctd', 'sensor', 'probe', 'instrument', 'gauge', 'meter',
            'sonar', 'hydrophone', 'transducer',
            
            # Support equipment
            'cable', 'tether', 'line', 'rope', 'wire', 'chain',
            'manipulator', 'arm', 'gripper', 'tool', 'equipment',
            
            # Ship and vessel terms
            'vessel', 'ship', 'boat', 'hull', 'deck', 'propeller',
            
            # Infrastructure
            'pipeline', 'cable', 'anchor', 'mooring', 'buoy',
            'platform', 'structure', 'debris',
            
            # Operation terms
            'deployment', 'recovery', 'operation', 'mission',
            'survey', 'transect', 'sample',
            
            # Event markers
            'start', 'begin', 'end', 'stop', 'pause', 'resume',
            'initiated', 'completed', 'commenced', 'finished',
            
            # Test and message terms
            'test', 'message', 'powered off', 'launch'
        ]
        
        # Group annotations by formatted dive ID
        annotations = {}
        for dive_id, group in df.groupby('Formatted Dive ID'):
            if pd.isna(dive_id):
                continue
            
            # Look for organisms in comments and taxonomy fields (positive match)
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
            
            # Filter out operational annotations (negative filter)
            operational_pattern = '|'.join(operational_terms)
            operational_mask = organism_annotations['Comment'].str.contains(
                operational_pattern, case=False, na=False)
            organism_annotations = organism_annotations[~operational_mask]
            
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
        print(f"\nDEBUG: Evaluating video: {video_path}")
        print(f"DEBUG: Dive ID: {dive_id}")
        print(f"DEBUG: Using OWL score threshold: {self.detector.threshold}")
        
        if dive_id not in self.annotations:
            print(f"DEBUG: Available dive IDs: {list(self.annotations.keys())}")
            print(f"DEBUG: No annotations found for dive {dive_id}")
            return None
        
        ground_truth = self.annotations[dive_id]
        video_timestamp = self._parse_video_timestamp(os.path.basename(video_path))
        
        if video_timestamp is None:
            print("DEBUG: Failed to parse video timestamp")
            return None
        
        # Debug timestamp comparison
        print(f"\nDEBUG: Timestamp comparisons:")
        print(f"Video timestamp: {video_timestamp}")
        print(f"Ground truth timestamps: {ground_truth}")
        
        # Only evaluate videos within annotation time range
        if not any(abs((gt - video_timestamp).total_seconds()) < 300 for gt in ground_truth):
            print(f"DEBUG: Video outside annotation timerange")
            print(f"DEBUG: Minimum time difference: {min(abs((gt - video_timestamp).total_seconds()) for gt in ground_truth)} seconds")
            return None
        
        # Run detector on video
        result = self.detector.process_video(video_path)
        
        # Debug detection information
        print(f"\nDEBUG: Detection Statistics:")
        print(f"Total detections after OWL threshold ({self.detector.threshold}): {len(result.detections)}")
        
        # Analyze confidence distribution
        if result.detections:
            confidences = [d.confidence for d in result.detections]
            print(f"Confidence stats:")
            print(f"  Min: {min(confidences):.3f}")
            print(f"  Max: {max(confidences):.3f}")
            print(f"  Mean: {sum(confidences)/len(confidences):.3f}")
            print(f"  Quartiles: {np.percentile(confidences, [25, 50, 75])}")
        
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
        
        # Save numerical results with debug information
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write("Overall Evaluation Results\n")
            f.write("========================\n\n")
            f.write(f"Number of videos evaluated: {len(results)}\n\n")
            f.write("Average Metrics:\n")
            f.write(f"Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}\n")
            f.write(f"Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}\n")
            f.write(f"F1 Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}\n\n")
            
            f.write("Detailed Debug Information:\n")
            f.write("=========================\n\n")
            f.write("Available Dive IDs in annotations:\n")
            f.write(f"{list(self.annotations.keys())}\n\n")
            
            f.write("Per-Video Results and Debug Info:\n")
            for video, metrics in results.items():
                f.write(f"\n{video}:\n")
                f.write("-------------\n")
                
                # Parse and log timestamp information
                video_timestamp = self._parse_video_timestamp(video)
                f.write(f"Video timestamp: {video_timestamp}\n")
                
                # Get dive ID from video name
                dive_id = video.split('_')[0]
                f.write(f"Dive ID: {dive_id}\n")
                
                # Log ground truth timestamps if available
                if dive_id in self.annotations:
                    ground_truth = self.annotations[dive_id]
                    f.write(f"Ground truth timestamps: {ground_truth[:5]}\n")  # Show first 5
                    
                    # Log time differences
                    f.write("Time differences with first 5 ground truth annotations:\n")
                    for gt in ground_truth[:5]:
                        time_diff = abs((gt - video_timestamp).total_seconds())
                        f.write(f"  • {gt}: {time_diff:.2f} seconds\n")
                else:
                    f.write(f"No ground truth annotations found for dive {dive_id}\n")
                
                # Log metrics
                f.write("\nMetrics:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.3f}\n")
                f.write("\n")

    def _calculate_temporal_metrics(
        self,
        detections: List[Detection],
        ground_truth: List[datetime],
        tolerance: float
    ) -> Dict[str, float]:
        """Calculate precision/recall with temporal matching."""
        # Filter out non-organism detections
        filtered_detections = []
        for det in detections:
            # Skip detections that are likely ROV/equipment/operations
            if any(term in det.label.lower() for term in [
                # ROV and vehicle terms
                'rov', 'vehicle', 'submersible', 'platform',
                
                # Camera and imaging equipment
                'camera', 'lens', 'light', 'strobe', 'laser', 'imaging',
                
                # Sampling equipment
                'sampler', 'niskin', 'bottle', 'collection', 'scoop', 'net',
                'trap', 'corer', 'basket', 'container',
                
                # Sensors and instruments
                'ctd', 'sensor', 'probe', 'instrument', 'gauge', 'meter',
                'sonar', 'hydrophone', 'transducer',
                
                # Support equipment
                'cable', 'tether', 'line', 'rope', 'wire', 'chain',
                'manipulator', 'arm', 'gripper', 'tool', 'equipment',
                
                # Ship and vessel terms
                'vessel', 'ship', 'boat', 'hull', 'deck', 'propeller',
                
                # Infrastructure
                'pipeline', 'cable', 'anchor', 'mooring', 'buoy',
                'platform', 'structure', 'debris',
                
                # Operation terms
                'deployment', 'recovery', 'operation', 'mission',
                'survey', 'transect', 'sample',
                
                # Event markers
                'start', 'begin', 'end', 'stop', 'pause', 'resume',
                'initiated', 'completed', 'commenced', 'finished'
            ]):
                continue
            
            # Skip detections with very low confidence
            if det.confidence < self.config['evaluation'].get('min_confidence', 0.1):
                continue
            
            filtered_detections.append(det)
        
        true_positives = 0
        detection_times = [d.timestamp for d in filtered_detections]
        
        # Debug information
        print(f"\nDEBUG: Temporal Matching Details")
        print(f"Number of original detections: {len(detections)}")
        print(f"Number of filtered detections: {len(filtered_detections)}")
        print(f"Number of ground truth annotations: {len(ground_truth)}")
        print(f"Temporal tolerance: {tolerance} seconds")
        
        # Sort detections by confidence
        detections_with_conf = [(d.timestamp, d.confidence, d.label) for d in filtered_detections]
        detections_with_conf.sort(key=lambda x: x[1], reverse=True)
        
        # Print top 10 highest confidence detections
        print("\nTop 10 highest confidence detections:")
        for time, conf, label in detections_with_conf[:10]:
            print(f"  Time: {time:.2f}s, Confidence: {conf:.3f}, Label: {label}")
        
        # Match detections to ground truth
        matched_gt = set()
        matched_det = set()
        
        # For each ground truth annotation, find the closest detection
        print("\nMatching ground truth to nearest detections:")
        for j, gt_time in enumerate(ground_truth):
            closest_detection = None
            min_time_diff = float('inf')
            closest_det_info = None
            
            for i, det in enumerate(filtered_detections):
                if i in matched_det:
                    continue
                
                # Convert detection timestamp to datetime
                video_start_time = ground_truth[0].replace(microsecond=0) - timedelta(seconds=det.timestamp)
                det_datetime = video_start_time + timedelta(seconds=det.timestamp)
                time_diff = abs((gt_time - det_datetime).total_seconds())
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_detection = i
                    closest_det_info = (det.timestamp, det.confidence, det.label)
            
            # Print matching information
            print(f"\nGround truth at {gt_time}:")
            if closest_det_info:
                print(f"  Closest detection: {closest_det_info[0]:.2f}s, conf: {closest_det_info[1]:.3f}, label: {closest_det_info[2]}")
                print(f"  Time difference: {min_time_diff:.2f}s")
                print(f"  Within tolerance? {min_time_diff <= tolerance}")
                
                if min_time_diff <= tolerance:
                    true_positives += 1
                    matched_gt.add(j)
                    matched_det.add(closest_detection)
                    print("  ✓ Matched!")
            else:
                print("  No unmatched detection found")
        
        false_positives = len(detection_times) - true_positives
        false_negatives = len(ground_truth) - true_positives
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nFinal matching results:")
        print(f"True positives: {true_positives}")
        print(f"False positives: {false_positives}")
        print(f"False negatives: {false_negatives}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_detections': len(detections),
            'filtered_detections': len(filtered_detections)
        }

    def _parse_video_timestamp(self, filename: str) -> datetime:
        """Extract timestamp from video filename."""
        print(f"\nDEBUG: Parsing timestamp from filename: {filename}")
        # Extract timestamp portion (e.g., 20230715T163244Z)
        timestamp_match = re.search(r'(\d{8}T\d{6}Z)', filename)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            print(f"DEBUG: Found timestamp string: {timestamp_str}")
            # Parse without timezone info to match annotation timestamps
            parsed_time = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%SZ')
            print(f"DEBUG: Parsed timestamp: {parsed_time}")
            return parsed_time
        print(f"DEBUG: No timestamp match found in filename")
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
    
    # Print evaluation mode
    print(f"\nRunning evaluation in {'simplified' if config['evaluation'].get('simplified_mode', False) else 'detailed'} mode")
    
    # Run evaluation with config parameters
    evaluator = DetectionEvaluator(
        ANNOTATION_CSV, 
        VIDEO_DIR,
        config
    )
    results = evaluator.evaluate_all_videos()
    evaluator.plot_results(results, OUTPUT_DIR)

if __name__ == "__main__":
    main()