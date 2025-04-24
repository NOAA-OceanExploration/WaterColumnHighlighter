import os
import cv2
import pandas as pd
import toml
from datetime import datetime, timedelta
import re
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import pytz
import argparse

def parse_video_timestamp(filename: str) -> Optional[datetime]:
    """Extract timestamp from video filename."""
    print(f"DEBUG: Parsing timestamp for filename: {filename}") # Added log
    # Look for standard Z format first
    timestamp_match = re.search(r'(\d{8}T\d{6}Z)', filename)
    if timestamp_match:
        timestamp_str = timestamp_match.group(1)
        try:
            # Parse without timezone info first for consistency
            parsed_time = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%SZ')
            print(f"DEBUG: Parsed Z timestamp: {parsed_time} (naive)") # Added log
            
            # Also create a version with explicit UTC timezone for debugging
            utc_time = pytz.utc.localize(parsed_time)
            print(f"DEBUG: Parsed Z timestamp (UTC): {utc_time}")
            
            # For some common timezones, print what this time would be
            timezones = {
                'US Eastern': pytz.timezone('US/Eastern'),
                'US Pacific': pytz.timezone('US/Pacific'),
                'UTC': pytz.utc
            }
            
            print("DEBUG: This timestamp in different timezones:")
            for name, tz in timezones.items():
                tz_time = utc_time.astimezone(tz)
                print(f"  • {name}: {tz_time}")
            
            return parsed_time
        except ValueError as e:
            print(f"Error parsing Z timestamp {timestamp_str} from {filename}: {e}")
            # Continue to try other formats if Z format fails

    # Try alternative common pattern without Z
    timestamp_match_no_z = re.search(r'(\d{8}T\d{6})', filename)
    if timestamp_match_no_z:
        timestamp_str = timestamp_match_no_z.group(1)
        try:
             parsed_time = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
             print(f"DEBUG: Parsed non-Z timestamp: {parsed_time} (naive)") # Added log
             return parsed_time
        except ValueError as e:
            print(f"Error parsing non-Z timestamp {timestamp_str} from {filename}: {e}")

    # Add more specific patterns if needed based on your filenames
    # Example: timestamp_match_custom = re.search(r'pattern', filename) ...

    print(f"Warning: No recognizable timestamp found in filename: {filename}")
    print(f"DEBUG: Failed to parse timestamp for: {filename}") # Added log
    return None


def load_annotations(csv_path: str, timezone_offset_hours: float = 0.0) -> Optional[pd.DataFrame]:
    """Load and preprocess annotations from CSV."""
    print(f"Loading annotations from: {csv_path}")
    print(f"Using timezone offset: {timezone_offset_hours} hours")
    try:
        # Increase robustness for potentially malformed CSVs
        # Try standard engine first
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except pd.errors.ParserError:
            print("Warning: Standard CSV parsing failed. Trying Python engine (slower)...")
            df = pd.read_csv(csv_path, engine='python', on_bad_lines='warn', low_memory=False) # Use python engine for more complex cases

    except FileNotFoundError:
        print(f"Error: Annotation file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None

    # --- Standardize Column Names (similar to evaluate_detections.py) ---
    column_mapping = {}
    # Added 'Annotation ID' to expected columns
    expected_columns = ['Dive ID', 'Start Date', 'Comment', 'Taxonomy', 'Taxon', 'Taxon Path', 'Annotation ID']
    df_columns_lower = {col.lower(): col for col in df.columns}

    for expected_col in expected_columns:
        expected_lower = expected_col.lower()
        original_col_name = df_columns_lower.get(expected_lower)

        if original_col_name:
             # Found exact or case-insensitive match
             if original_col_name != expected_col:
                 column_mapping[original_col_name] = expected_col
        else:
            # Try partial match if exact/case-insensitive fails
            key_part = expected_col.split()[0].lower()
            # Be careful with partial matches, prioritize exact/case-insensitive
            partial_matches = [orig_col for lower_col, orig_col in df_columns_lower.items()
                               if key_part in lower_col and orig_col not in column_mapping.keys()]
            if partial_matches:
                # Use the first partial match found that hasn't been mapped yet
                column_mapping[partial_matches[0]] = expected_col
                print(f"Mapping partial match '{partial_matches[0]}' to '{expected_col}'")
            # Only warn if it's a core required column or frequently used one
            elif expected_col in ['Dive ID', 'Start Date', 'Comment', 'Taxon', 'Annotation ID']:
                 print(f"Warning: Could not find a suitable match for expected column '{expected_col}'")


    if column_mapping:
        print(f"Renaming columns: {column_mapping}")
        df = df.rename(columns=column_mapping)

    # --- Check required columns ---
    required_cols = ['Dive ID', 'Start Date']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns in annotations. Need: {required_cols}. Found: {list(df.columns)}")
        return None

    # Ensure 'Comment', 'Taxon', and 'Annotation ID' exist, even if empty
    if 'Comment' not in df.columns:
        df['Comment'] = ''
    if 'Annotation ID' not in df.columns:
        df['Annotation ID'] = 'N/A' # Add a default value if missing
    if 'Taxon' not in df.columns:
        # Try 'Taxon Path' or 'Taxonomy' as fallbacks before setting to Unknown
        fallback_taxon_col = None
        if 'Taxon Path' in df.columns:
            fallback_taxon_col = 'Taxon Path'
        elif 'Taxonomy' in df.columns:
             fallback_taxon_col = 'Taxonomy'

        if fallback_taxon_col:
            print(f"Warning: 'Taxon' column missing. Using '{fallback_taxon_col}' as Taxon.")
            df['Taxon'] = df[fallback_taxon_col]
        else:
            print("Warning: 'Taxon' column missing and no suitable fallback found. Setting Taxon to 'Unknown'.")
            df['Taxon'] = 'Unknown'

    # Fill NaN comments/taxons/annotation IDs
    df['Comment'] = df['Comment'].fillna('No comment')
    df['Taxon'] = df['Taxon'].fillna('Unknown')
    df['Annotation ID'] = df['Annotation ID'].fillna('N/A')


    # --- Process Timestamps ---\
    try:
        print(f"DEBUG: Original 'Start Date' head:\n{df['Start Date'].head()}") # Added log
        # Attempt to convert 'Start Date', coercing errors to NaT (Not a Time)
        # Ensure UTC is specified during parsing, as 'Z' indicates UTC
        df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce', utc=True)
        print(f"DEBUG: 'Start Date' after parsing (UTC):\n{df['Start Date'].head()}") # Added log
        
        # Apply timezone offset if provided
        if timezone_offset_hours != 0.0:
            print(f"Applying timezone offset of {timezone_offset_hours} hours to all timestamps...")
            df['Start Date'] = df['Start Date'] + pd.Timedelta(hours=timezone_offset_hours)
            print(f"DEBUG: 'Start Date' after timezone offset adjustment:\n{df['Start Date'].head()}")

        # Drop rows where timestamp conversion failed
        original_count = len(df)
        df = df.dropna(subset=['Start Date'])
        if len(df) < original_count:
            print(f"Warning: Dropped {original_count - len(df)} rows due to invalid 'Start Date' format.")
        if df.empty:
             print("Error: No valid 'Start Date' timestamps found after cleaning.")
             return None

        # Convert to timezone-naive AFTER ensuring it was parsed correctly as UTC
        df['Start Date'] = df['Start Date'].dt.tz_localize(None)
        print(f"DEBUG: 'Start Date' after tz_localize(None):\n{df['Start Date'].head()}") # Added log

    except Exception as e:
        print(f"Error processing 'Start Date' column: {e}")
        return None

    # --- Process Dive IDs ---\
    # Ensure Dive ID is string for consistent matching
    df['Dive ID'] = df['Dive ID'].astype(str)

    # --- Map numeric dive ID to EX format if needed (Example) ---\
    # Make this more robust, maybe read from config?
    dive_mapping = {
        '2853': 'EX2304',
        '2673': 'EX2206'
        # Add more mappings as needed
    }
    # Apply mapping, keeping original ID if no mapping exists
    df['Formatted Dive ID'] = df['Dive ID'].map(lambda x: dive_mapping.get(x, x))

    print(f"Loaded {len(df)} valid annotations.")
    return df

def find_video_for_annotation(annotation_time: datetime, dive_id: str, video_files_map: Dict[str, List[Tuple[str, datetime, datetime]]]) -> Optional[str]:
    """Find the video file that contains the annotation timestamp."""
    if dive_id not in video_files_map:
        print(f"DEBUG: Dive ID '{dive_id}' not found in video_files_map keys.")
        return None

    potential_videos = video_files_map[dive_id]
    # Prioritize ROVHD videos if multiple matches exist for the same time
    rovhd_matches = []
    other_matches = []

    tolerance = timedelta(seconds=1) # Keep tolerance for slight edge cases
    
    # Debug information for timestamp comparison
    print(f"\nDEBUG: Searching for video matching annotation at {annotation_time}")
    print(f"DEBUG: Annotation timezone info: {annotation_time.tzinfo}")
    print(f"DEBUG: Number of potential videos to check: {len(potential_videos)}")
    
    # Store all videos and their time differences for debugging
    all_video_time_diffs = []

    for video_path, start_time, end_time in potential_videos:
        # Calculate difference in seconds for debugging
        time_diff_seconds = abs((annotation_time - start_time).total_seconds())
        video_basename = os.path.basename(video_path)
        all_video_time_diffs.append((video_basename, start_time, end_time, time_diff_seconds))
        
        if (start_time - tolerance) <= annotation_time <= (end_time + tolerance):
            if 'ROVHD' in os.path.basename(video_path).upper():
                rovhd_matches.append(video_path)
            else:
                other_matches.append(video_path)

    # Sort and print all videos by time difference for debugging
    all_video_time_diffs.sort(key=lambda x: x[3])
    print(f"\nDEBUG: All videos sorted by time difference to annotation:")
    for i, (vname, vstart, vend, diff) in enumerate(all_video_time_diffs[:10]):  # Show top 10
        duration = (vend - vstart).total_seconds()
        print(f"  {i+1}. {vname}: diff={diff:.1f}s, start={vstart}, end={vend}, duration={duration:.1f}s")
    
    # Check if we should try alternative approaches when no direct match is found
    if not rovhd_matches and not other_matches:
        print(f"\nDEBUG: No standard match found. Trying alternative approaches:")
        
        # Try with a much larger tolerance
        expanded_tolerance = timedelta(minutes=5)  # 5-minute tolerance
        print(f"DEBUG: Trying expanded tolerance of {expanded_tolerance.total_seconds()} seconds")
        
        for video_path, start_time, end_time in potential_videos:
            if (start_time - expanded_tolerance) <= annotation_time <= (end_time + expanded_tolerance):
                video_name = os.path.basename(video_path)
                time_diff = min(abs((annotation_time - start_time).total_seconds()), 
                               abs((annotation_time - end_time).total_seconds()))
                print(f"DEBUG: Expanded match found: {video_name}, diff={time_diff:.1f}s")
                if 'ROVHD' in video_name.upper():
                    rovhd_matches.append(video_path)
                else:
                    other_matches.append(video_path)
        
        # If still no matches, try checking for possible timezone issues
        if not rovhd_matches and not other_matches and len(potential_videos) > 0:
            print(f"DEBUG: Testing for potential timezone offsets...")
            
            # Common timezone hour offsets to check
            timezone_offsets = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 
                              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            
            best_offset = None
            best_diff = float('inf')
            best_video = None
            
            for offset in timezone_offsets:
                adjusted_time = annotation_time + timedelta(hours=offset)
                
                for video_path, start_time, end_time in potential_videos:
                    if (start_time - tolerance) <= adjusted_time <= (end_time + tolerance):
                        diff = abs((adjusted_time - start_time).total_seconds())
                        if diff < best_diff:
                            best_diff = diff
                            best_offset = offset
                            best_video = video_path
            
            if best_video:
                print(f"DEBUG: Found potential timezone match with offset {best_offset} hours")
                print(f"DEBUG: Best match: {os.path.basename(best_video)}, diff={best_diff:.1f}s")
                if 'ROVHD' in os.path.basename(best_video).upper():
                    rovhd_matches.append(best_video)
                else:
                    other_matches.append(best_video)

    if rovhd_matches:
        # If multiple ROVHD match (unlikely unless overlapping recordings), return the first
        print(f"DEBUG: Found {len(rovhd_matches)} ROVHD matches. Using: {os.path.basename(rovhd_matches[0])}")
        return rovhd_matches[0]
    elif other_matches:
        # If no ROVHD match, return the first non-ROVHD match
        print(f"DEBUG: No ROVHD match found. Found {len(other_matches)} other matches. Using: {os.path.basename(other_matches[0])}")
        return other_matches[0]
    else:
        # Debug why no match was found
        print(f"DEBUG: No video found for {annotation_time} in Dive {dive_id}.")
        if potential_videos:
            print("DEBUG: Videos checked for this dive (Times are naive):")
            for vp, vs, ve in potential_videos[:5]: # Print first 5 for brevity
                print(f"  - {os.path.basename(vp)}: Start={vs}, End={ve}")
            if len(potential_videos) > 5: print("  ...")
        else:
            print("DEBUG: No videos listed for this dive in the map.")
        return None


def get_video_metadata(video_path: str) -> Optional[Tuple[float, int]]:
    """Gets FPS and Frame Count using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path} to get metadata.")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps > 0 and frame_count > 0:
            return fps, frame_count
        else:
            print(f"Warning: Invalid metadata (FPS: {fps}, Frames: {frame_count}) for {video_path}")
            return None
    except Exception as e:
        print(f"Error getting metadata for video {video_path}: {e}")
        return None


def scan_video_directory(video_dir: str) -> Dict[str, List[Tuple[str, datetime, datetime]]]:
    """ Scans the video directory (including subdirs) and creates a map of dive IDs to video files and their times."""
    video_files_map = {}
    print(f"Scanning video directory structure starting from: {video_dir}")
    if not os.path.isdir(video_dir):
        print(f"Error: Video directory not found at {video_dir}")
        return {}

    # Use os.walk to traverse directory tree
    processed_videos = 0
    skipped_metadata = 0
    skipped_timestamp = 0
    skipped_no_dive_id = 0
    timezone_warning_shown = False

    # Check for timezone inconsistencies in filenames
    print("\nDEBUG: Checking for timezone information in filenames...")
    try:
        have_pytz = True
    except ImportError:
        have_pytz = False
        print("DEBUG: pytz not installed, detailed timezone analysis will be limited")
    
    # Initialize containers for timezone analysis
    has_z_suffix = []
    missing_z_suffix = []
    
    for root, dirs, files in os.walk(video_dir):
         # Try to extract Dive ID from the current directory path (e.g., .../EX2304/...)
         dive_id_match = re.search(r'(EX\d{4})', root) # Simple pattern, adjust if needed
         current_dive_id = dive_id_match.group(1) if dive_id_match else None

         if not current_dive_id:
             # Also check parent directory structure, e.g., if videos are in .../EX2304/Compressed/
             parent_dir_match = re.search(r'(EX\d{4})', os.path.dirname(root))
             if parent_dir_match:
                 current_dive_id = parent_dir_match.group(1)


         # Prioritize getting dive_id from folder structure
         root_dive_id = current_dive_id

         for filename in files:
             if filename.lower().endswith(('.mp4', '.mov', '.avi')):
                 # Analyze timestamp format for debugging
                 if 'Z' in filename:
                     has_z_suffix.append(filename)
                 else:
                     # Look for format without Z suffix
                     if re.search(r'\d{8}T\d{6}', filename):
                         missing_z_suffix.append(filename)
                 
                 # Try to get Dive ID: Folder > Filename
                 file_dive_id = root_dive_id # Start with folder dive id
                 if not file_dive_id:
                     # If folder ID not found, try from filename
                     dive_id_match_file = re.match(r'(EX\d{4})', filename)
                     if dive_id_match_file:
                         file_dive_id = dive_id_match_file.group(1)

                 if not file_dive_id:
                     # print(f"Warning: Could not determine Dive ID for video: {os.path.join(root, filename)}")
                     skipped_no_dive_id += 1
                     continue # Skip if no Dive ID is found

                 video_path = os.path.join(root, filename)
                 start_time = parse_video_timestamp(filename)

                 if start_time:
                    metadata = get_video_metadata(video_path)
                    if metadata:
                        fps, frame_count = metadata
                        if fps > 0: # Avoid division by zero
                            duration_seconds = frame_count / fps
                            end_time = start_time + timedelta(seconds=duration_seconds)
                            print(f"DEBUG: Video {filename} - Dive: {file_dive_id}, Start: {start_time}, End: {end_time}, FPS: {fps:.2f}, Frames: {frame_count}, Duration: {duration_seconds:.2f}s") # Added log

                            if file_dive_id not in video_files_map:
                                video_files_map[file_dive_id] = []
                            video_files_map[file_dive_id].append((video_path, start_time, end_time))
                            processed_videos += 1
                        else:
                             print(f"Warning: Skipping video {filename} due to zero FPS.") # Added log
                             skipped_metadata += 1
                    else:
                         print(f"Skipping video {filename} due to missing or invalid metadata.") # Modified log
                         skipped_metadata += 1
                 else:
                     # print(f"Skipping video {filename} due to unparsable timestamp.") # Less verbose
                     skipped_timestamp += 1

    # Print timestamp format analysis
    print("\nDEBUG: Timestamp Format Analysis")
    print(f"Files with Z suffix: {len(has_z_suffix)}")
    print(f"Files missing Z suffix but having timestamp format: {len(missing_z_suffix)}")
    
    if len(has_z_suffix) > 0 and len(missing_z_suffix) > 0:
        print("\nWARNING: Mixed timestamp formats detected. This could indicate inconsistent timezone handling.")
        print("Examples with Z suffix:")
        for f in has_z_suffix[:3]:
            print(f"  • {f}")
        print("Examples without Z suffix:")
        for f in missing_z_suffix[:3]:
            print(f"  • {f}")
            
        # Compare parsed times between formats
        if has_z_suffix and missing_z_suffix and have_pytz:
            print("\nDEBUG: Comparing parsed times between formats...")
            z_time = parse_video_timestamp(has_z_suffix[0])
            non_z_time = parse_video_timestamp(missing_z_suffix[0])
            
            if z_time and non_z_time:
                diff = abs((z_time - non_z_time).total_seconds())
                print(f"Time difference between Z and non-Z format: {diff:.1f} seconds")
                
                # Check if difference is close to common timezone offsets
                common_offsets = [0, 3600, 7200, 14400, 18000, 19800, 28800, 32400, 36000, 39600]  # Common offsets in seconds
                for offset in common_offsets:
                    if abs(diff - offset) < 60:  # Within a minute of the offset
                        hours = offset / 3600
                        print(f"This is approximately {hours} hours difference, suggesting a timezone offset.")
    
    # Sort video lists by start time for each dive
    for dive_id in video_files_map:
        video_files_map[dive_id].sort(key=lambda x: x[1])
        
        # Add debug info: Print start/end time ranges per dive
        videos = video_files_map[dive_id]
        if videos:
            earliest = min(videos, key=lambda x: x[1])
            latest = max(videos, key=lambda x: x[2])
            print(f"\nDEBUG: Dive {dive_id} time range:")
            print(f"  Earliest video: {os.path.basename(earliest[0])}")
            print(f"  Time range: {earliest[1]} to {latest[2]}")
            print(f"  Total duration: {(latest[2] - earliest[1]).total_seconds():.1f} seconds")

    print(f"\nVideo Scan Summary:")
    print(f"  Found and processed {processed_videos} videos across {len(video_files_map)} dives.")
    print(f"  Skipped (no dive ID): {skipped_no_dive_id}")
    print(f"  Skipped (unparsable timestamp): {skipped_timestamp}")
    print(f"  Skipped (missing/invalid metadata): {skipped_metadata}")
    return video_files_map

def main():
    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description='Verify annotations by extracting video frames.')
    parser.add_argument('--timezone-offset', type=float, default=0.0,
                        help='Hours to adjust annotation timestamps by (e.g., -4.0 for EDT, -7.0 for PDT)')
    args = parser.parse_args()
    
    # --- Load Configuration ---\
    try:
        # Try loading from standard locations
        config_path = None
        possible_paths = [
            'config.toml',
            '../config.toml', # If script is run from a subdirectory
            os.path.join(os.path.dirname(__file__), '../config.toml') # If script is in a package structure
        ]
        # Add workspace root if __file__ is not reliable
        # Check if __file__ is defined and use it if available
        if '__file__' in locals() or '__file__' in globals():
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths.append(os.path.join(script_dir, 'config.toml'))
            possible_paths.append(os.path.join(script_dir, '../config.toml'))


        for path in possible_paths:
            abs_path = os.path.abspath(path)
            # print(f"Checking for config at: {abs_path}") # Debug print
            if os.path.exists(abs_path):
                config_path = abs_path
                break

        if not config_path:
             # Try relative to current working directory as a last resort
             cwd_config_path = os.path.abspath('config.toml')
             if os.path.exists(cwd_config_path):
                 config_path = cwd_config_path
             else:
                 # Provide more specific paths checked in the error
                 checked_paths_str = "\n".join([os.path.abspath(p) for p in possible_paths] + [cwd_config_path])
                 raise FileNotFoundError(f"config.toml not found. Checked paths:\n{checked_paths_str}")

        config = toml.load(config_path)
        print(f"Loaded configuration from: {config_path}")

        # Get required paths from config
        video_dir = config['paths']['video_dir']
        annotation_csv = config['paths']['annotation_csv']
        output_dir = config['paths'].get('verification_output_dir') # Use .get for safety

        if not output_dir:
             print("Error: 'verification_output_dir' not found in [paths] section of config.toml")
             # Provide default or fallback?
             output_dir = os.path.join(os.path.dirname(config_path), "annotation_verification_frames")
             print(f"Using default verification output directory: {output_dir}")
             config['paths']['verification_output_dir'] = output_dir # Add to config for consistency


    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except KeyError as e:
        print(f"Error: Missing key in config.toml: {e}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Verification frames will be saved to: {output_dir}")

    # --- Load Annotations ---\
    annotations_df = load_annotations(annotation_csv, timezone_offset_hours=args.timezone_offset)
    if annotations_df is None or annotations_df.empty:
        print("Failed to load valid annotations. Exiting.")
        return

    # --- Scan Video Directory ---
    # This function now handles finding videos and their time ranges
    video_files_map = scan_video_directory(video_dir)
    if not video_files_map:
         print("No video files found or processed successfully. Exiting.")
         return

    # --- Process Annotations ---\
    processed_count = 0
    skipped_no_video = 0
    skipped_frame_bounds = 0
    error_count = 0
    video_caps = {} # Cache open video captures {video_path: cv2.VideoCapture}

    print("\nStarting annotation verification process...")
    try:
        # Use tqdm for progress bar
        for index, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Verifying annotations"):
            annotation_time = row['Start Date']
            dive_id = row['Formatted Dive ID'] # Use the potentially mapped Dive ID
            comment = str(row.get('Comment', '')) # Ensure comment is string
            taxon = str(row.get('Taxon','Unknown')) # Ensure taxon is string
            annotation_id_raw = row.get('Annotation ID', 'N/A') # Get Annotation ID if available
            print(f"\nDEBUG: Processing Annotation ID: {annotation_id_raw}, Dive: {dive_id}, Time: {annotation_time}, Taxon: {taxon}") # Added log

            # Find the correct video file using the pre-scanned map
            video_path = find_video_for_annotation(annotation_time, dive_id, video_files_map)

            if video_path is None:
                # Log already happens inside find_video_for_annotation
                skipped_no_video += 1
                continue
            # print(f"DEBUG: Found matching video: {os.path.basename(video_path)}") # Log inside find_video_for_annotation now

            # Get video details (start time needed for frame calculation)
            video_info = next((info for info in video_files_map[dive_id] if info[0] == video_path), None)
            if video_info is None: # Should not happen if find_video_for_annotation worked
                 print(f"Internal Error: Video info mismatch for {video_path}. Skipping.")
                 error_count += 1
                 continue
            _, video_start_time, video_end_time = video_info # Get end time too
            print(f"DEBUG: Video Time Range (Naive): Start={video_start_time}, End={video_end_time}") # Added log

            # --- Open video using cache ---
            cap = video_caps.get(video_path)
            if cap is None or not cap.isOpened():
                # Release old cap if it exists but is closed
                if cap is not None:
                     cap.release()
                # Try to open the video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error: Could not open video file {video_path}. Skipping annotations for this video.")
                    # Remove from cache if failed
                    if video_path in video_caps: del video_caps[video_path]
                    error_count += 1
                    # Maybe try finding an alternative video if one exists? For now, just skip.
                    
                    # Add detailed timestamp debugging information
                    print(f"\nDEBUG: TIMESTAMP MISMATCH DETAILS:")
                    print(f"Failed video path: {video_path}")
                    print(f"Video name: {os.path.basename(video_path)}")
                    
                    # Look for any video files in the same dive directory with similar timestamp
                    video_dir = os.path.dirname(video_path)
                    print(f"Checking for alternative videos in: {video_dir}")
                    
                    all_videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
                    print(f"Found {len(all_videos)} video files in directory")
                    
                    # Try to parse annotation time and find videos with nearby timestamps
                    annotation_times = [annot for annot in annotations_df['Start Date'] if isinstance(annot, datetime) or isinstance(annot, pd.Timestamp)]
                    
                    if annotation_times:
                        print(f"\nAnnotation timestamp samples:")
                        for i, ts in enumerate(annotation_times[:5]):
                            print(f"  {i+1}: {ts} (UTC aware: {ts.tzinfo is not None})")
                            
                        # Try to find videos with timestamps close to annotations
                        closest_videos = []
                        for video_name in all_videos[:10]:  # Check first 10 videos
                            video_ts = parse_video_timestamp(video_name)
                            if video_ts:
                                # Calculate time differences to first few annotations
                                diffs = []
                                for anno_ts in annotation_times[:5]:
                                    # Ensure both timestamps are timezone-naive for comparison
                                    if anno_ts.tzinfo is not None:
                                        anno_ts = anno_ts.replace(tzinfo=None)
                                    if video_ts.tzinfo is not None:
                                        video_ts = video_ts.replace(tzinfo=None)
                                        
                                    diff_seconds = abs((anno_ts - video_ts).total_seconds())
                                    diffs.append(diff_seconds)
                                
                                min_diff = min(diffs) if diffs else float('inf')
                                closest_videos.append((video_name, video_ts, min_diff))
                        
                        # Sort by minimum time difference
                        closest_videos.sort(key=lambda x: x[2])
                        
                        print(f"\nClosest alternative videos by timestamp:")
                        for v_name, v_ts, diff in closest_videos[:5]:
                            print(f"  {v_name}: {v_ts}, diff: {diff:.1f} seconds")
                        
                        # Check if there might be a timezone issue
                        if annotation_times[0].tzinfo != video_start_time.tzinfo:
                            print(f"\nWARNING: Possible timezone mismatch!")
                            print(f"Annotation timezone: {annotation_times[0].tzinfo}")
                            print(f"Video timestamp timezone: {video_start_time.tzinfo}")
                            
                            # Try adjusting for common timezone offsets and check if match improves
                            common_offsets = [-8, -7, -5, -4, 0, 1, 2, 5.5, 8, 9, 10]
                            best_offset = None
                            best_diff = float('inf')
                            
                            for offset in common_offsets:
                                adjusted_ts = video_start_time + timedelta(hours=offset)
                                new_diff = abs((adjusted_ts - annotation_times[0].replace(tzinfo=None)).total_seconds())
                                if new_diff < best_diff:
                                    best_diff = new_diff
                                    best_offset = offset
                            
                            print(f"Best timezone offset to try: {best_offset} hours (diff: {best_diff:.1f} seconds)")
                    
                    continue
                video_caps[video_path] = cap # Add successfully opened cap to cache
            # --- Video is now open ---

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                print(f"Error: Invalid FPS ({fps}) for video {video_path}. Skipping annotation.")
                error_count += 1
                continue # Skip to next annotation

            # Calculate frame number precisely
            time_diff = annotation_time - video_start_time
            time_diff_seconds = time_diff.total_seconds()

            # Check if time difference seems plausible given video duration
            # Add a safety margin (e.g., 5 seconds) for potential small inaccuracies
            video_duration_seconds = (video_end_time - video_start_time).total_seconds()
            if not (-5 <= time_diff_seconds <= video_duration_seconds + 5):
                 print(f"WARNING: Time difference ({time_diff_seconds:.2f}s) is outside the expected video duration range (0 to {video_duration_seconds:.2f}s +/- 5s). AnnotationTime={annotation_time}, VideoStart={video_start_time}, VideoEnd={video_end_time}. Check timestamp alignment.")
                 # Don't necessarily skip, but log prominently.

            # Clamp negative time diffs, but log warning
            if time_diff_seconds < 0:
                 print(f"WARNING: Calculated negative time difference ({time_diff_seconds:.2f}s) for annotation {annotation_time} in video starting at {video_start_time}. Clamping to 0.")
                 time_diff_seconds = 0


            frame_number = int(round(time_diff_seconds * fps)) # Use round for potentially better accuracy
            print(f"DEBUG: TimeDiff={time_diff_seconds:.3f}s, FPS={fps:.2f}, Calculated Frame={frame_number}") # Added log


            # Check if frame number is valid (using cached or re-fetched frame count)
            frame_count_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not (0 <= frame_number < frame_count_video):
                print(f"WARNING: Calculated frame number {frame_number} is out of bounds (0-{frame_count_video-1}) for annotation at {annotation_time} in {video_path}. Skipping.") # Modified log level
                skipped_frame_bounds += 1
                continue # Skip to next annotation

            # --- Extract and Save Frame Sequence (Instead of just one frame) ---
            frames_to_extract = 11 # Extract 11 frames: target -5 to target +5
            half_window = frames_to_extract // 2
            start_frame = max(0, frame_number - half_window)
            # Ensure end_frame does not exceed frame_count_video
            end_frame = min(frame_count_video, frame_number + half_window + 1)

            # Create a subdirectory for this annotation's sequence
            # Sanitize annotation ID for filename
            safe_annotation_id = re.sub(r'[<>:"/\\|?* ]', '_', str(annotation_id_raw))
            sequence_sub_dir_name = f"Dive{dive_id}_Anno{safe_annotation_id}_TargetF{frame_number}_{annotation_time.strftime('%Y%m%dT%H%M%S%f')[:-3]}"
            sequence_output_dir = os.path.join(output_dir, sequence_sub_dir_name)
            os.makedirs(sequence_output_dir, exist_ok=True)

            print(f"DEBUG: Extracting frames {start_frame} to {end_frame-1} into {sequence_output_dir}")

            frames_saved = 0
            for current_frame_num in range(start_frame, end_frame):
                # Optimized seeking: only set if the frame number changes significantly or crosses a threshold
                # For simplicity here, we still set for each frame, but be aware this can be slow.
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
                ret, frame = cap.read()

                if not ret or frame is None:
                    # Attempt to read the *next* frame as a fallback
                    # print(f"Warning: Initial read failed for frame {current_frame_num}. Trying next frame.")
                    # ret, frame = cap.read() # Read next frame
                    # current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1 # Get actual frame number read
                    # if not ret or frame is None:
                         print(f"Error: Failed to read frame {current_frame_num} (or subsequent) from {video_path}. Skipping frame.")
                         continue # Skip this specific frame

                # --- Draw Annotation Info on Frame ---\
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6 # Slightly smaller font
                base_color = (255, 255, 255) # White
                bg_color = (0, 0, 0)       # Black
                thickness = 1
                line_type = cv2.LINE_AA

                # Position for text (top-left corner with padding)
                x_pos = 10
                y_pos = 25
                line_height = int(font_scale * 30) # Adjust based on font size

                # Text lines to draw
                text_lines = [
                    f"Anno Time: {annotation_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}", # Milliseconds
                    f"Target Frame: {frame_number}",
                    f"Current Frame: {current_frame_num}", # Add current frame number
                    f"Dive: {dive_id} | Anno ID: {annotation_id_raw}", # Combine dive/anno id
                    f"Taxon: {taxon}",
                    f"Comment: {comment[:70]}{'...' if len(comment) > 70 else ''}" # Limit comment length
                ]

                # Calculate background size needed
                max_text_width = 0
                for text in text_lines:
                    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    if w > max_text_width:
                        max_text_width = w
                # Adjust background height calculation
                bg_height = (len(text_lines) * line_height) + (10 if len(text_lines) > 0 else 0)

                # Draw background rectangle (semi-transparent perhaps?)
                # Ensure coordinates are valid
                bg_x1 = x_pos - 5
                bg_y1 = y_pos - line_height + 10 # Adjusted start Y
                bg_x2 = x_pos + max_text_width + 5
                bg_y2 = bg_y1 + bg_height -5 # Adjusted end Y

                # Draw solid background rectangle
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)


                # Draw text lines onto the frame
                current_y = y_pos + 5 # Start text lower
                for text in text_lines:
                    cv2.putText(frame, text, (x_pos, current_y), font, font_scale, base_color, thickness, line_type)
                    current_y += line_height


                # --- Save Frame ---\
                frame_filename = f"frame_{current_frame_num:06d}.jpg" # Pad frame number
                frame_output_path = os.path.join(sequence_output_dir, frame_filename)

                try:
                    # Use imwrite with JPEG quality parameter if needed
                    cv2.imwrite(frame_output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    frames_saved += 1
                except Exception as e:
                    print(f"Error saving frame {current_frame_num} to {frame_output_path}: {e}")
                    # Attempt to save with a generic name if filename causes error (less likely now)
                    try:
                         fallback_filename = f"error_frame_{dive_id}_{current_frame_num}.jpg"
                         fallback_path = os.path.join(sequence_output_dir, fallback_filename)
                         cv2.imwrite(fallback_path, frame)
                         print(f"Saved frame with fallback name: {fallback_filename}")
                         # Don't increment error_count here if fallback succeeds, but log it
                    except Exception as fe:
                         print(f"Failed to save frame {current_frame_num} even with fallback name: {fe}")
                         error_count += 1 # Count as error if fallback also fails

            if frames_saved > 0:
                 processed_count += 1 # Count the whole annotation as processed if we saved >= 1 frame
            else:
                 # Only increment error count if no frames could be saved for this annotation
                 if start_frame < end_frame: # Check if there were frames supposed to be extracted
                      print(f"Error: Failed to save ANY frames for annotation ID {annotation_id_raw} at {annotation_time}")
                      error_count += 1


    finally:
        # Release all cached video captures
        print("\nReleasing video captures...")
        for path, cap in video_caps.items():
            if cap and cap.isOpened():
                 cap.release()
                 # print(f"Released: {path}") # Optional debug print
        print("Video captures released.")

    print("\nVerification complete.")
    print(f"  Successfully processed annotations (saved >= 1 frame): {processed_count}")
    print(f"  Annotations skipped (no video match): {skipped_no_video}")
    print(f"  Annotations skipped (frame out of bounds): {skipped_frame_bounds}")
    print(f"  Errors encountered (file open, frame read, save fail): {error_count}")
    print(f"  Total annotations checked: {len(annotations_df)}")
    
    # Generate timestamp alignment visualization
    print("\nGenerating timestamp alignment visualization...")
    generate_timestamp_alignment_visualization(annotations_df, video_files_map, output_dir)


def generate_timestamp_alignment_visualization(annotations_df, video_files_map, output_dir):
    """
    Generate a visualization showing timestamp alignment between videos and annotations.
    This helps diagnose timezone issues and other temporal alignment problems.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        import numpy as np
        
        # Create output directory for visualizations
        viz_dir = os.path.join(output_dir, "timestamp_alignment")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Group annotations by dive
        annotations_by_dive = {}
        for _, row in annotations_df.iterrows():
            dive_id = row['Formatted Dive ID']
            if dive_id not in annotations_by_dive:
                annotations_by_dive[dive_id] = []
            annotations_by_dive[dive_id].append(row['Start Date'])
        
        # Process each dive that has both videos and annotations
        for dive_id in sorted(annotations_by_dive.keys()):
            if dive_id not in video_files_map:
                print(f"No videos found for dive {dive_id}, skipping visualization.")
                continue
                
            annotation_times = annotations_by_dive[dive_id]
            videos = video_files_map[dive_id]
            
            if not videos or not annotation_times:
                continue
                
            # Create a new figure
            plt.figure(figsize=(20, 10))
            plt.title(f"Timestamp Alignment for Dive {dive_id}", fontsize=16)
            
            # Plot video timespans as rectangles
            for i, (video_path, start_time, end_time) in enumerate(videos):
                video_name = os.path.basename(video_path)
                y_pos = i * 0.5
                duration = (end_time - start_time).total_seconds()
                
                # Create rectangle for video timespan
                rect = Rectangle((mdates.date2num(start_time), y_pos), 
                                mdates.date2num(end_time) - mdates.date2num(start_time), 0.3, 
                                color='lightblue', alpha=0.7)
                plt.gca().add_patch(rect)
                
                # Add video name and timestamp
                plt.text(mdates.date2num(start_time), y_pos + 0.35, 
                         f"{video_name}\n{start_time.strftime('%Y-%m-%d %H:%M:%S')}", 
                         fontsize=8, verticalalignment='bottom')
            
            # Plot annotation timestamps as vertical lines
            for anno_time in annotation_times:
                plt.axvline(x=mdates.date2num(anno_time), color='red', linestyle='--', alpha=0.3)
            
            # Plot densities to show clusters of annotations
            if len(annotation_times) > 1:
                anno_times_num = [mdates.date2num(t) for t in annotation_times]
                kde_x = np.linspace(min(anno_times_num), max(anno_times_num), 1000)
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(anno_times_num)
                    kde_y = kde(kde_x) * 3  # Scale for visibility
                    plt.plot(kde_x, kde_y - 0.5, 'r-', alpha=0.7)
                    plt.fill_between(kde_x, -0.5, kde_y - 0.5, color='red', alpha=0.2)
                except Exception as e:
                    print(f"Could not generate KDE plot: {e}")
            
            # Format x-axis as dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            plt.gcf().autofmt_xdate()
            
            # Set axis labels and limits
            plt.xlabel('Time (UTC)', fontsize=12)
            plt.ylabel('Video Index', fontsize=12)
            plt.yticks([])  # Hide y-axis
            
            # Add explanation
            plt.figtext(0.02, 0.02, 
                      "Blue rectangles: Video timespans\nRed vertical lines: Annotation timestamps\n"
                      "Red curve: Density of annotations", 
                      fontsize=10)
            
            # Save the figure
            output_path = os.path.join(viz_dir, f"alignment_{dive_id}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved timestamp alignment visualization to {output_path}")
            
        # BONUS: Generate consolidated visualization across all dives
        try:
            if len(annotations_by_dive) > 1:
                plt.figure(figsize=(20, 15))
                plt.title(f"Timestamp Alignment Across All Dives", fontsize=16)
                
                all_times = []  # Collect all times for axis limits
                y_offset = 0
                
                for i, dive_id in enumerate(sorted(annotations_by_dive.keys())):
                    if dive_id not in video_files_map:
                        continue
                        
                    y_pos = y_offset
                    plt.text(-0.01, y_pos + 0.3, f"Dive {dive_id}", fontsize=10, 
                            transform=plt.gca().transAxes)
                    
                    # Plot videos
                    videos = video_files_map[dive_id]
                    max_videos_height = 0
                    
                    for j, (video_path, start_time, end_time) in enumerate(videos):
                        all_times.extend([start_time, end_time])
                        video_y = y_pos + j * 0.3
                        max_videos_height = max(max_videos_height, j * 0.3 + 0.2)
                        
                        # Create rectangle for video timespan
                        rect = Rectangle((mdates.date2num(start_time), video_y), 
                                        mdates.date2num(end_time) - mdates.date2num(start_time), 0.2, 
                                        color='lightblue', alpha=0.7)
                        plt.gca().add_patch(rect)
                    
                    # Plot annotations for this dive
                    annotation_times = annotations_by_dive[dive_id]
                    all_times.extend(annotation_times)
                    
                    anno_y = y_pos + max_videos_height + 0.2
                    
                    for anno_time in annotation_times:
                        plt.plot([mdates.date2num(anno_time), mdates.date2num(anno_time)], 
                                [anno_y, anno_y + 0.1], 'r-', alpha=0.5)
                    
                    # Move to next dive group
                    y_offset = anno_y + 0.5
                
                # Format x-axis as dates
                if all_times:
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
                    plt.gcf().autofmt_xdate()
                
                # Save the figure
                output_path = os.path.join(viz_dir, "alignment_all_dives.png")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved consolidated timestamp visualization to {output_path}")
                
        except Exception as e:
            print(f"Error generating consolidated visualization: {e}")
            
        # Also generate a time difference histogram
        try:
            plt.figure(figsize=(15, 8))
            plt.title("Histogram of Time Differences Between Annotations and Nearest Video Start", fontsize=16)
            
            all_diffs = []
            
            for dive_id, anno_times in annotations_by_dive.items():
                if dive_id not in video_files_map:
                    continue
                    
                videos = video_files_map[dive_id]
                video_starts = [start for _, start, _ in videos]
                
                for anno_time in anno_times:
                    # Find nearest video start time
                    if video_starts:
                        nearest_diff = min(abs((anno_time - vstart).total_seconds()) for vstart in video_starts)
                        all_diffs.append(nearest_diff)
            
            # Convert to hours for easier reading
            all_diffs_hours = [d/3600 for d in all_diffs]
            
            plt.hist(all_diffs_hours, bins=50, alpha=0.7, color='blue')
            plt.axvline(x=0, color='red', linestyle='--')
            
            # Add vertical lines at common timezone offsets
            common_offsets = [1, 2, 3, 4, 5, 7, 8, 10, 12, 13]
            for offset in common_offsets:
                plt.axvline(x=offset, color='green', linestyle=':', alpha=0.5)
                plt.axvline(x=-offset, color='green', linestyle=':', alpha=0.5)
            
            plt.xlabel('Time Difference (hours)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Save the figure
            output_path = os.path.join(viz_dir, "time_difference_histogram.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved time difference histogram to {output_path}")
            
        except Exception as e:
            print(f"Error generating time difference histogram: {e}")
    
    except ImportError:
        print("Could not generate visualizations. Required libraries (matplotlib, scipy) not installed.")
    except Exception as e:
        print(f"Error generating timestamp visualizations: {e}")

if __name__ == "__main__":
    main()