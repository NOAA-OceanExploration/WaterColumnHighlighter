import os
import csv
import toml
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import re

from owl_highlighter import CritterDetector
from owl_highlighter.utils import parse_video_timestamp # Assuming this utility exists and works

# --- Configuration ---
DEFAULT_CONFIG_PATH = 'config.toml'
OUTPUT_FILENAME_TEMPLATE = "{dive_id}_annotations_CritterDetector.csv"

# --- Target CSV Header (Based on csv_column_template.csv) ---
# Define the full header based on the template
TARGET_HEADER = [
    'Dive ID', 'Dive Name', 'Cruise Name', 'Start Date', 'End Date',
    'Annotation "Location"', 'Annotation "Size"', 'Annotation ID', 'Annotation Source',
    'Modified Date', 'Resource Type ID', 'Resource ID', 'Creator First Name',
    'Creator Last Name', 'Creator Email', 'Modifier First Name', 'Modifier Last Name',
    'Modifier Email', 'To Be Reviewed', 'Total Reviews', 'Percent Positive Reviews',
    'Comment', 'TEMPPROBEDEEPDISCOVERERROV_23974_Temperature',
    'TEMPPROBEDEEPDISCOVERERROV_23974_Temperature Time',
    'DEEPDISCOVERERNAV01_23975_Latitude', 'DEEPDISCOVERERNAV01_23975_Latitude Time',
    'DEEPDISCOVERERNAV01_23975_Pitch', 'DEEPDISCOVERERNAV01_23975_Pitch Time',
    'DEEPDISCOVERERNAV01_23975_Longitude', 'DEEPDISCOVERERNAV01_23975_Longitude Time',
    'DEEPDISCOVERERNAV01_23975_Roll', 'DEEPDISCOVERERNAV01_23975_Roll Time',
    'DEEPDISCOVERERNAV01_23975_Altitude', 'DEEPDISCOVERERNAV01_23975_Altitude Time',
    'DEEPDISCOVERERNAV01_23975_Heading', 'DEEPDISCOVERERNAV01_23975_Heading Time',
    'SBECTD9PLUSDEEPDISCOVERER_23978_Oxidation Reduction Potential',
    'i want', # Placeholder for the column with typo in template
    'SBECTD9PLUSDEEPDISCOVERER_23978_Oxygen Concentration',
    'SBECTD9PLUSDEEPDISCOVERER_23978_Oxygen Concentration Time',
    'SBECTD9PLUSDEEPDISCOVERER_23978_Temperature',
    'SBECTD9PLUSDEEPDISCOVERER_23978_Temperature Time',
    'SBECTD9PLUSDEEPDISCOVERER_23978_Depth', 'SBECTD9PLUSDEEPDISCOVERER_23978_Depth Time',
    'SBECTD9PLUSDEEPDISCOVERER_23978_Practical Salinity',
    'SBECTD9PLUSDEEPDISCOVERER_23978_Practical Salinity Time',
    'SBECTD9PLUSDEEPDISCOVERER_23978_Turbidity', 'SBECTD9PLUSDEEPDISCOVERER_23978_Turbidity Time',
    'Taxonomy', 'Taxon', 'Taxon Common Names', 'Taxon Path',
    'Common/OBIS Count', 'Common/OBIS Test Date', 'Common/Medium', 'Common/Morphotype',
    'Common/Count', 'Common/Mortality', 'Common/Method',
    'Hydrophone (Michael)/Container Ship', 'Hydrophone (Michael)/Chemical Carrier',
    'Biota', 'Kingdom', 'Subkingdom', 'Infrakingdom', 'Phylum', 'Division',
    'Subphylum', 'Subdivision', 'Infraphylum', 'Infradivision', 'Parvphylum',
    'Parvdivision', 'Gigaclass', 'Megaclass', 'Superclass', 'Class', 'Subclass',
    'Infraclass', 'Subterclass', 'Superorder', 'Order', 'Suborder', 'Infraorder',
    'Parvorder', 'Section', 'Subsection', 'Superfamily', 'Epifamily', 'Family',
    'Subfamily', 'Supertribe', 'Tribe', 'Subtribe', 'Genus', 'Subgenus', 'Species',
    'Subspecies', 'Natio', 'Variety', 'Subvariety', 'Forma', 'Subforma', 'Component',
    'Mutatio', 'Subcomponent', 'Water Column Layer', 'Hydroform Class', 'Hydroform',
    'Hydroform Type', 'Salinity', 'Temperature', 'Biogeochemical Feature',
    'Substrate Origin', 'Substrate Class', 'Substrate Subclass', 'Substrate Group',
    'Substrate Subgroup', 'Physiographic Setting', 'Tectonic Setting',
    'Geoform Origin', 'Geoform', 'Geoform Type', 'Unclassified'
]

def get_dive_id_from_filename(filename):
    """Extracts Dive ID (e.g., EX2304) from a standard video filename."""
    match = re.match(r"(EX\d{4})_", filename)
    if match:
        return match.group(1)
    # Add other patterns if needed
    return None # Or raise an error

def format_bbox_location(bbox):
    """Formats bbox [x1, y1, x2, y2] into '[[x1, y1], [x2, y2]]' string."""
    if not bbox or len(bbox) != 4:
        return ""
    x1, y1, x2, y2 = map(int, map(round, bbox)) # Ensure integers
    return f"[[{x1}, {y1}], [{x2}, {y2}]]"

def format_bbox_size(bbox):
    """Formats bbox [x1, y1, x2, y2] into 'width x height' string."""
    if not bbox or len(bbox) != 4:
        return ""
    x1, y1, x2, y2 = map(int, map(round, bbox))
    width = x2 - x1
    height = y2 - y1
    return f"{width} x {height}"

def main(config_path, video_source, output_dir):
    """Main processing function."""

    # --- Load Config ---
    try:
        config = toml.load(config_path)
        print(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # --- Initialize Detector ---
    try:
        detector = CritterDetector(config=config)
        print(f"Initialized CritterDetector ({detector.model_type})")
    except Exception as e:
        print(f"Error initializing CritterDetector: {e}")
        return

    # --- Identify Videos ---
    video_files = []
    if os.path.isdir(video_source):
        video_dir = video_source
        print(f"Scanning video directory: {video_dir}")
        try:
            for filename in os.listdir(video_dir):
                if filename.lower().endswith(('.mp4', '.mov', '.avi')):
                    # Basic check for likely video files
                    # Add more robust checks if needed (e.g., ROVHD pattern)
                    video_files.append(os.path.join(video_dir, filename))
        except Exception as e:
            print(f"Error scanning video directory {video_dir}: {e}")
            return
    elif os.path.isfile(video_source) and video_source.lower().endswith(('.mp4', '.mov', '.avi')):
        video_files.append(video_source)
    else:
        print(f"Error: video_source '{video_source}' is not a valid directory or video file.")
        return

    if not video_files:
        print("No video files found to process.")
        return

    print(f"Found {len(video_files)} video file(s) to process.")

    # --- Prepare Output ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output CSVs will be saved to: {output_dir}")

    # Group videos by Dive ID
    videos_by_dive = {}
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        dive_id = get_dive_id_from_filename(video_name)
        if not dive_id:
            print(f"Warning: Could not determine Dive ID for {video_name}. Skipping.")
            continue
        if dive_id not in videos_by_dive:
            videos_by_dive[dive_id] = []
        videos_by_dive[dive_id].append(video_path)

    # --- Process Videos ---
    for dive_id, dive_video_paths in videos_by_dive.items():
        print(f"\n--- Processing Dive: {dive_id} ({len(dive_video_paths)} videos) ---")
        output_csv_path = os.path.join(output_dir, OUTPUT_FILENAME_TEMPLATE.format(dive_id=dive_id))
        all_dive_detections = []

        for video_path in dive_video_paths:
            video_name = os.path.basename(video_path)
            print(f"Processing video: {video_name}")

            # Try to parse video start time (essential for absolute timestamps)
            video_start_time_naive = parse_video_timestamp(video_name) # Assumes returns naive datetime
            if video_start_time_naive is None:
                 print(f"Warning: Could not parse start timestamp for {video_name}. Absolute timestamps will be missing.")
                 video_start_time_utc = None
            else:
                 # Assume filename timestamp is UTC (adjust if needed)
                 try:
                     video_start_time_utc = video_start_time_naive.replace(tzinfo=datetime.timezone.utc)
                 except Exception as e:
                     print(f"Warning: Could not make timestamp timezone-aware for {video_name}: {e}")
                     video_start_time_utc = None


            try:
                # Run detection
                result = detector.process_video(
                    video_path=video_path,
                    # output_dir=output_dir, # Let process_video handle internal outputs if needed
                    # create_highlight_clips=False, # Optional: configure as needed
                    # frame_interval=config.get('processing', {}).get('frame_interval', 5), # Optional
                    # save_timeline=True # Optional
                    verbose=False # Keep console less cluttered
                )

                print(f"Found {len(result.detections)} detections in {video_name}")

                # Populate Detection objects with additional info
                for det in result.detections:
                    det.dive_id = dive_id
                    det.video_name = video_name
                    det.taxon = det.label # Map label to Taxon

                    # Calculate absolute timestamp if possible
                    if video_start_time_utc:
                        det.absolute_timestamp = video_start_time_utc + timedelta(seconds=det.timestamp)

                    # Format bbox info
                    det.annotation_location = format_bbox_location(det.bbox)
                    det.annotation_size = format_bbox_size(det.bbox)

                    # Add confidence to comment
                    det.comment = f"Confidence: {det.confidence:.3f}"

                    all_dive_detections.append(det)

            except FileNotFoundError:
                print(f"Error: Video file not found during processing: {video_path}")
            except Exception as e:
                print(f"Error processing video {video_name}: {e}")
                # Optionally continue to next video or stop

        # --- Write Output CSV for the Dive ---
        if not all_dive_detections:
            print(f"No detections found for Dive {dive_id}. Skipping CSV output.")
            continue

        print(f"Writing {len(all_dive_detections)} detections to {output_csv_path}")
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=TARGET_HEADER)
                writer.writeheader()

                for det in all_dive_detections:
                    # Create a row dictionary, filling available fields
                    row_data = {hdr: '' for hdr in TARGET_HEADER} # Initialize with blanks

                    row_data['Dive ID'] = det.dive_id
                    # row_data['Dive Name'] = ... # Requires lookup/external data
                    # row_data['Cruise Name'] = ... # Requires lookup/external data
                    row_data['Start Date'] = det.absolute_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + 'Z' if det.absolute_timestamp else '' # Format as ISO 8601 UTC
                    # row_data['End Date'] = ... # Hard to determine end of a single detection event
                    row_data['Annotation "Location"'] = det.annotation_location
                    row_data['Annotation "Size"'] = det.annotation_size
                    # row_data['Annotation ID'] = ... # Generate unique ID? uuid.uuid4()?
                    row_data['Annotation Source'] = det.annotation_source
                    # row_data['Modified Date'] = datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + 'Z' # Set to now?
                    # ... Fill other metadata if available ...
                    row_data['Comment'] = det.comment
                    # ... Sensor data columns are left blank ...
                    row_data['Taxon'] = det.taxon
                    # row_data['Taxonomy'] = ... # Requires mapping Taxon to full path
                    # ... Fill other relevant fields if data exists ...

                    writer.writerow(row_data)

        except Exception as e:
            print(f"Error writing CSV file {output_csv_path}: {e}")

    print("\nProcessing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CritterDetector on videos and generate annotation CSVs.")
    parser.add_argument('video_source',
                        help="Path to a single video file or a directory containing video files.")
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG_PATH,
                        help=f"Path to the configuration file (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument('-o', '--output-dir', default='.',
                        help="Directory to save the output CSV files (default: current directory).")

    args = parser.parse_args()

    main(config_path=args.config, video_source=args.video_source, output_dir=args.output_dir) 