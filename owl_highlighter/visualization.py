from PIL import Image, ImageDraw, ImageFont
from .models import VideoProcessingResult
import os
from typing import Dict, Tuple

def create_timeline_visualization(
    result: VideoProcessingResult,
    output_path: str,
    width: int = 2000,
    height: int = 1200
) -> None:
    """
    Create a visual timeline of detections in the video.
    
    Args:
        result: VideoProcessingResult containing detections and video metadata
        output_path: Path where the timeline image should be saved
        width: Width of the timeline image
        height: Height of the timeline image
    """
    # Timeline layout parameters
    padding = 40
    timeline_y = height - 300
    
    # Create a blank image with a light background
    background_color = (245, 245, 250)  # Light blue-gray
    timeline_img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(timeline_img)
    
    # Get taxonomic groups and colors
    taxonomic_groups = _get_taxonomic_groups()
    
    # Font setup
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 12)
        scientific_font = ImageFont.truetype("DejaVuSans-Oblique.ttf", 10)
    except:
        title_font = label_font = scientific_font = ImageFont.load_default()

    # Draw title
    title = f"Timeline: {os.path.splitext(result.video_name)[0]}"
    draw.text((padding, padding), title, fill=(50, 50, 50), font=title_font)
    
    # Draw main timeline
    timeline_color = (100, 100, 100)
    draw.line([(padding, timeline_y), (width - padding, timeline_y)], 
             fill=timeline_color, width=3)
    
    # Calculate scaling factor
    usable_width = width - (2 * padding)
    scale = usable_width / result.frame_count
    
    # Sort detections by frame number
    sorted_detections = sorted(result.detections, key=lambda x: x.frame_number)
    
    # Draw detections
    previous_frame = None
    image_y_offset = timeline_y - 400
    
    for detection in sorted_detections:
        # Skip if too close to previous detection
        if (previous_frame is not None and 
            detection.frame_number - previous_frame <= result.fps * 2):
            continue
            
        x = int(padding + (detection.frame_number * scale))
        y_offset = image_y_offset
        
        # Determine organism type and color
        detected_group = 'other'
        for group, info in taxonomic_groups.items():
            if any(pattern in detection.label.lower() for pattern in info['patterns']):
                detected_group = group
                break
        color = taxonomic_groups[detected_group]['color']
        
        # Resize detection image while maintaining aspect ratio
        image_patch = detection.image_patch.copy()
        image_patch.thumbnail((200, 200))
        
        # Calculate paste position
        paste_x = x - image_patch.width // 2
        paste_y = y_offset
        
        # Create white background for image
        bg = Image.new('RGB', image_patch.size, 'white')
        timeline_img.paste(bg, (paste_x, paste_y))
        timeline_img.paste(image_patch, (paste_x, paste_y))
        
        # Draw connecting line
        draw.line([(x, paste_y + image_patch.height), (x, timeline_y)], 
                 fill=color, width=2)
        
        # Draw timeline dot
        dot_radius = 4
        draw.ellipse([x - dot_radius, timeline_y - dot_radius, 
                     x + dot_radius, timeline_y + dot_radius], 
                    fill=color)
        
        # Draw label
        label = detection.label.capitalize()
        label_width = label_font.getlength(label)
        draw.text((paste_x + (image_patch.width - label_width) // 2, 
                  paste_y + image_patch.height + 5), 
                 label, fill=color, font=label_font)
        
        # Convert timestamp to MM:SS format
        minutes = int(detection.timestamp // 60)
        seconds = int(detection.timestamp % 60)
        timestamp_text = f"{minutes}:{seconds:02d}"  # :02d ensures two digits for seconds
        timestamp_width = label_font.getlength(timestamp_text)
        draw.text((paste_x + (image_patch.width - timestamp_width) // 2, 
                  paste_y + image_patch.height + 20), 
                 timestamp_text, fill=color, font=label_font)
        
        y_offset -= (image_patch.height + 60)
        previous_frame = detection.frame_number

    # Add timestamp markers
    marker_interval = 60  # Change to 1-minute intervals
    for t in range(0, int(result.duration) + 1, marker_interval):
        x = int(padding + ((t * result.fps) * scale))
        draw.line([(x, timeline_y), (x, timeline_y + 10)], 
                 fill=timeline_color, width=2)
        minutes = t // 60
        seconds = t % 60
        draw.text((x - 15, timeline_y + 15), 
                 f"{minutes}:{seconds:02d}", 
                 fill=timeline_color, font=label_font)
    
    # Add legend
    _add_legend(draw, taxonomic_groups, height, padding, label_font, scientific_font)
    
    # Save visualization
    timeline_img.save(output_path)

def _get_taxonomic_groups() -> Dict[str, Dict]:
    """Return dictionary of taxonomic groups with their colors and patterns."""
    return {
        # Chordata (vertebrates)
        'actinopterygii': {  # Ray-finned fishes
            'color': (65, 105, 225),  # Royal Blue
            'patterns': ['fish', 'anchovy', 'barracuda', 'bass', 'blenny', 'butterflyfish', 
                         'cardinalfish', 'clownfish', 'cod', 'damselfish', 'eel', 'flounder', 
                         'goby', 'grouper', 'grunts', 'halibut', 'herring', 'jackfish', 
                         'lionfish', 'mackerel', 'moray eel', 'mullet', 'parrotfish', 
                         'pipefish', 'pufferfish', 'rabbitfish', 'rays', 'scorpionfish', 
                         'seahorse', 'sergeant major', 'snapper', 'sole', 'surgeonfish', 
                         'tang', 'threadfin', 'triggerfish', 'tuna', 'wrasse']
        },
        'chondrichthyes': {  # Cartilaginous fishes
            'color': (220, 20, 60),  # Crimson
            'patterns': ['shark', 'angel shark', 'bamboo shark', 'blacktip reef shark', 
                         'bull shark', 'carpet shark', 'cat shark', 'dogfish', 
                         'great white shark', 'hammerhead shark', 'leopard shark', 
                         'nurse shark', 'reef shark', 'sand tiger shark', 'thresher shark', 
                         'tiger shark', 'whale shark', 'wobbegong']
        },
        'mammalia': {  # Marine mammals
            'color': (75, 0, 130),  # Indigo
            'patterns': ['whale', 'dolphin', 'porpoise', 'seal', 'sea lion', 'dugong', 
                         'manatee', 'orca', 'pilot whale', 'sperm whale', 'humpback whale', 
                         'blue whale', 'minke whale', 'right whale', 'beluga whale', 
                         'narwhal', 'walrus']
        },
        # Mollusca
        'cephalopoda': {  # Cephalopods
            'color': (255, 69, 0),  # Red-Orange
            'patterns': ['octopus', 'squid', 'cuttlefish', 'nautilus', 'bobtail squid', 
                         'giant squid', 'reef octopus', 'blue-ringed octopus', 
                         'mimic octopus', 'dumbo octopus', 'vampire squid']
        },
        'bivalvia': {  # Bivalves
            'color': (255, 165, 0),  # Orange
            'patterns': ['clam', 'mussel', 'oyster', 'scallop', 'giant clam']
        },
        # Cnidaria
        'anthozoa': {  # Corals and anemones
            'color': (255, 127, 80),  # Coral
            'patterns': ['coral', 'anemone', 'sea fan', 'sea whip', 'brain coral', 
                         'staghorn coral', 'elkhorn coral', 'soft coral', 'gorgonian']
        },
        'scyphozoa': {  # True jellyfish
            'color': (147, 112, 219),  # Medium Purple
            'patterns': ['jellyfish', 'moon jellyfish', 'box jellyfish', 
                         'lion\'s mane jellyfish']
        },
        # Echinodermata
        'echinodermata': {  # Echinoderms
            'color': (34, 139, 34),  # Forest Green
            'patterns': ['starfish', 'sea star', 'brittle star', 'basket star', 
                         'sea cucumber', 'sea urchin', 'sand dollar', 'feather star', 
                         'crinoid']
        },
        # Crustacea
        'crustacea': {  # Crustaceans
            'color': (210, 105, 30),  # Chocolate
            'patterns': ['crab', 'lobster', 'shrimp', 'barnacle', 'hermit crab', 
                         'spider crab', 'king crab', 'snow crab', 'mantis shrimp', 
                         'krill', 'copepod', 'amphipod', 'isopod', 'crawfish', 'crayfish']
        },
        # Other groups
        'other': {
            'color': (128, 128, 128),  # Gray
            'patterns': ['sponge', 'tunicate', 'sea squirt', 'salp', 'pyrosome', 
                         'coral polyp', 'hydrozoan', 'bryozoan', 'zoanthid', 
                         'colonial anemone']
        }
    }

def _add_legend(
    draw: ImageDraw.ImageDraw,
    taxonomic_groups: Dict,
    height: int,
    padding: int,
    label_font: ImageFont.FreeTypeFont,
    scientific_font: ImageFont.FreeTypeFont
) -> None:
    """Add legend to the timeline visualization."""
    legend_y = height - 60
    legend_x = padding
    
    for taxon, info in taxonomic_groups.items():
        if taxon != 'other':
            # Draw color sample
            draw.rectangle([legend_x, legend_y, legend_x + 10, legend_y + 10], 
                         fill=info['color'])
            # Add taxonomic name
            draw.text((legend_x + 15, legend_y), taxon.capitalize(), 
                     fill=info['color'], font=label_font)
            # Add example organisms in italics
            if info['patterns']:
                examples = ', '.join(info['patterns'][:2])
                draw.text((legend_x + 15, legend_y + 12), 
                         f"e.g., {examples}", 
                         fill=info['color'], font=scientific_font)
            
            legend_x += 180
            if legend_x > 1800:  # Start new row
                legend_x = padding
                legend_y += 30