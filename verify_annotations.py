import os
import cv2
import pandas as pd
import toml
from datetime import datetime, timedelta
import re
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

def parse_video_timestamp(filename: str) -> Optional[datetime]:
    """Extract timestamp from video filename."""
    # Look for standard Z format first
    timestamp_match = re.search(r'(\d{8}T\d{6}Z)', filename)
    if timestamp_match:
        timestamp_str = timestamp_match.group(1)
        try:
            # Parse without timezone info first for consistency
            parsed_time = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%SZ')
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
             return parsed_time
        except ValueError as e:
            print(f"Error parsing non-Z timestamp {timestamp_str} from {filename}: {e}")
    
    # Add more specific patterns if needed based on your filenames
    # Example: timestamp_match_custom = re.search(r'pattern', filename) ...

    print(f"Warning: No recognizable timestamp found in filename: {filename}")
    return None


def load_annotations(csv_path: str) -> Optional[pd.DataFrame]:
    """Load and preprocess annotations from CSV."""
    print(f"Loading annotations from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None

    # --- Standardize Column Names (similar to evaluate_detections.py) ---
    column_mapping = {}
    expected_columns = ['Dive ID', 'Start Date', 'Comment', 'Taxonomy', 'Taxon', 'Taxon Path']
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
            partial_matches = [orig_col for lower_col, orig_col in df_columns_lower.items() if key_part in lower_col]
            if partial_matches:
                # Use the first partial match found
                column_mapping[partial_matches[0]] = expected_col
                print(f"Mapping partial match '{partial_matches[0]}' to '{expected_col}'")
            else:
                 print(f"Warning: Could not find a suitable match for expected column '{expected_col}'")


    if column_mapping:
        print(f"Renaming columns: {column_mapping}")
        df = df.rename(columns=column_mapping)

    # --- Check required columns ---
    required_cols = ['Dive ID', 'Start Date']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns in annotations. Need: {required_cols}. Found: {list(df.columns)}")
        return None

    # Ensure 'Comment' and 'Taxon' exist, even if empty
    if 'Comment' not in df.columns:
        df['Comment'] = ''
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

    # Fill NaN comments/taxons
    df['Comment'] = df['Comment'].fillna('No comment')
    df['Taxon'] = df['Taxon'].fillna('Unknown')


    # --- Process Timestamps ---\
    try:
        # Attempt to convert 'Start Date', coercing errors to NaT (Not a Time)
        df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
        # Drop rows where timestamp conversion failed
        original_count = len(df)
        df = df.dropna(subset=['Start Date'])
        if len(df) < original_count:
            print(f"Warning: Dropped {original_count - len(df)} rows due to invalid 'Start Date' format.")
        if df.empty:
             print("Error: No valid 'Start Date' timestamps found after cleaning.")
             return None
        # Ensure timezone-naive for consistent comparison with video timestamps later
        # Check if tz-aware first before localizing to None
        if pd.api.types.is_datetime64_any_dtype(df['Start Date']) and df['Start Date'].dt.tz is not None:
             df['Start Date'] = df['Start Date'].dt.tz_localize(None)

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
        return None

    # Find potential matches (videos for the correct dive)
    potential_videos = video_files_map[dive_id]

    # Iterate through sorted videos for the dive
    for video_path, start_time, end_time in potential_videos:
        # Check if the annotation time falls within the video's time range
        # Add a small tolerance (e.g., 1 second) to handle potential rounding issues
        tolerance = timedelta(seconds=1)
        if (start_time - tolerance) <= annotation_time <= (end_time + tolerance):
            return video_path

    # If no exact match, maybe print nearby videos for debugging?
    # print(f"Debug: No video found for {annotation_time} in Dive {dive_id}. Videos checked: {potential_videos}")
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
    for root, dirs, files in os.walk(video_dir):
         # Try to extract Dive ID from the current directory path (e.g., .../EX2304/...)
         dive_id_match = re.search(r'(EX\d{4})', root) # Simple pattern, adjust if needed
         current_dive_id = dive_id_match.group(1) if dive_id_match else None

         if not current_dive_id:
             # Maybe check parent dirs if structure is deeper?
             pass # For now, skip files not in a recognized dive folder structure

         for filename in files:
             if filename.lower().endswith(('.mp4', '.mov', '.avi')):
                 # If we couldn't get Dive ID from folder, try from filename
                 file_dive_id = current_dive_id
                 if not file_dive_id:
                     dive_id_match_file = re.match(r'(EX\d{4})', filename)
                     if dive_id_match_file:
                         file_dive_id = dive_id_match_file.group(1)

                 if not file_dive_id:
                     # print(f"Warning: Could not determine Dive ID for video: {os.path.join(root, filename)}")
                     continue # Skip if no Dive ID is found

                 video_path = os.path.join(root, filename)
                 start_time = parse_video_timestamp(filename)

                 if start_time:
                    metadata = get_video_metadata(video_path)
                    if metadata:
                        fps, frame_count = metadata
                        duration_seconds = frame_count / fps
                        end_time = start_time + timedelta(seconds=duration_seconds)

                        if file_dive_id not in video_files_map:
                            video_files_map[file_dive_id] = []
                        video_files_map[file_dive_id].append((video_path, start_time, end_time))
                    else:
                         print(f"Skipping video {filename} due to missing metadata.")
                 else:
                     print(f"Skipping video {filename} due to unparsable timestamp.")

    # Sort video lists by start time for each dive
    for dive_id in video_files_map:
        video_files_map[dive_id].sort(key=lambda x: x[1])

    print(f"Found {sum(len(v) for v in video_files_map.values())} videos across {len(video_files_map)} dives.")
    return video_files_map

def main():
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
        if '__file__' in locals() or '__file__' in globals():
             possible_paths.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml'))


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
                 raise FileNotFoundError(f"config.toml not found. Checked paths: {possible_paths} and {cwd_config_path}")

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
    annotations_df = load_annotations(annotation_csv)
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

            # Find the correct video file using the pre-scanned map
            video_path = find_video_for_annotation(annotation_time, dive_id, video_files_map)

            if video_path is None:
                skipped_no_video += 1
                continue

            # Get video details (start time needed for frame calculation)
            video_info = next((info for info in video_files_map[dive_id] if info[0] == video_path), None)
            if video_info is None: # Should not happen if find_video_for_annotation worked
                 print(f"Internal Error: Video info mismatch for {video_path}. Skipping.")
                 error_count += 1
                 continue
            _, video_start_time, _ = video_info

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
            # Ensure time_diff is non-negative; could happen with tolerance issues
            if time_diff.total_seconds() < 0:
                 time_diff_seconds = 0
            else:
                 time_diff_seconds = time_diff.total_seconds()

            frame_number = int(round(time_diff_seconds * fps)) # Use round for potentially better accuracy


            # Check if frame number is valid (using cached or re-fetched frame count)
            frame_count_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not (0 <= frame_number < frame_count_video):
                # print(f"Warning: Calculated frame number {frame_number} is out of bounds (0-{frame_count_video-1}) for annotation at {annotation_time} in {video_path}. Skipping.")
                skipped_frame_bounds += 1
                continue # Skip to next annotation

            # Seek to frame
            # set() can be slow, only set if needed or check current position? For simplicity, just set.
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            # Check if frame reading was successful
            if not ret or frame is None:
                # Try reading the next frame in case of seeking issues
                # print(f"Warning: Initial read failed for frame {frame_number}. Trying next frame.")
                # ret, frame = cap.read()
                # if not ret or frame is None:
                    print(f"Error: Failed to read frame {frame_number} (or subsequent) from {video_path}. Skipping annotation.")
                    error_count += 1
                    continue # Skip to next annotation


            # --- Draw Annotation Info on Frame ---\
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            base_color = (255, 255, 255) # White
            bg_color = (0, 0, 0)       # Black
            thickness = 1
            line_type = cv2.LINE_AA

            # Position for text (top-left corner with padding)
            x_pos = 15
            y_pos = 30
            line_height = int(font_scale * 30) # Adjust based on font size

            # Text lines to draw
            text_lines = [
                f"Time: {annotation_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}", # Milliseconds
                f"Frame: {frame_number}",
                f"Dive: {dive_id}",
                f"Taxon: {taxon}",
                f"Comment: {comment[:80]}{'...' if len(comment) > 80 else ''}" # Limit comment length
            ]

            # Calculate background size needed
            max_text_width = 0
            for text in text_lines:
                (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                if w > max_text_width:
                    max_text_width = w
            bg_height = len(text_lines) * line_height + 10 # Add padding

            # Draw background rectangle (semi-transparent perhaps?)
            # For solid background:
            cv2.rectangle(frame, (x_pos - 5, y_pos - line_height + 5), (x_pos + max_text_width + 10, y_pos + bg_height - line_height + 5), bg_color, -1)

            # Draw text lines onto the frame
            current_y = y_pos
            for text in text_lines:
                cv2.putText(frame, text, (x_pos, current_y), font, font_scale, base_color, thickness, line_type)
                current_y += line_height


            # --- Save Frame ---\
            # Sanitize taxon name and comment for filename
            safe_taxon = re.sub(r'[<>:"/\\|?* ]', '_', taxon)[:30] # Replace invalid chars with underscore
            safe_comment_part = re.sub(r'[<>:"/\\|?* ]', '_', comment)[:20] # Short part of comment

            output_filename = f"{dive_id}_{annotation_time.strftime('%Y%m%dT%H%M%S%f')[:-3]}_F{frame_number}_{safe_taxon}_{safe_comment_part}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            try:
                # Use imwrite with JPEG quality parameter if needed
                cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                processed_count += 1
            except Exception as e:
                print(f"Error saving frame to {output_path}: {e}")
                # Attempt to save with a generic name if filename causes error
                try:
                     fallback_filename = f"error_frame_{dive_id}_{frame_number}.jpg"
                     fallback_path = os.path.join(output_dir, fallback_filename)
                     cv2.imwrite(fallback_path, frame)
                     print(f"Saved frame with fallback name: {fallback_filename}")
                     error_count += 1 # Still count as error due to naming issue
                except Exception as fe:
                     print(f"Failed to save frame even with fallback name: {fe}")
                     error_count += 1


    finally:
        # Release all cached video captures
        print("\nReleasing video captures...")
        for path, cap in video_caps.items():
            if cap and cap.isOpened():
                 cap.release()
                 # print(f"Released: {path}") # Optional debug print

    print("\nVerification complete.")
    print(f"  Successfully processed and saved frames: {processed_count}")
    print(f"  Annotations skipped (no video match): {skipped_no_video}")
    print(f"  Annotations skipped (frame out of bounds): {skipped_frame_bounds}")
    print(f"  Errors encountered (file access, frame read, save): {error_count}")
    print(f"  Total annotations checked: {len(annotations_df)}")

if __name__ == "__main__":
    main()