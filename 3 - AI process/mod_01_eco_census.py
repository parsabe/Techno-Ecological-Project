import os
import csv
from collections import defaultdict

# --- MODULE 1: ECO TEAM BIOLOGICAL CENSUS & CSV EXPORT ---
species_unique_ids = defaultdict(set)

def reset_census():
    """Resets unique species tracking sets for a new video session."""
    species_unique_ids.clear()

def update_species_census(species_name, track_id):
    """
    Registers a persistent track ID for a detected species to ensure deduplicated counting.
    """
    if species_name and track_id is not None:
        species_unique_ids[species_name].add(track_id)

def get_census_summary():
    """
    Returns total deduplicated counts per species, sorted in descending order,
    along with Top 4 species ranking and primary species information.
    """
    counts = {species: len(uids) for species, uids in species_unique_ids.items()}
    sorted_species = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    
    total_unique = sum(counts.values())
    top_4 = sorted_species[:4]
    
    primary_species = sorted_species[0][0] if sorted_species else "Salmo trutta"
    primary_count = sorted_species[0][1] if sorted_species else 0
    
    return {
        "sorted_species": sorted_species,
        "top_4": top_4,
        "total_unique": total_unique,
        "primary_species": primary_species,
        "primary_count": primary_count
    }

def generate_csv_report(output_dir="csv", output_filename="fish_counts.csv"):
    """
    Writes the deduplicated species census to fish_counts.csv inside the session output directory.
    Appends methodology statement at the bottom of the CSV file.
    """
    summary = get_census_summary()
    sorted_species = summary["sorted_species"]
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.abspath(os.path.join(output_dir, output_filename))
    
    try:
        with open(filepath, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            
            # Data Table Header
            writer.writerow(["Rank", "Species Name", "Unique Count"])
            
            # Data Rows
            if sorted_species:
                for rank, (species_name, count) in enumerate(sorted_species, start=1):
                    writer.writerow([rank, species_name, count])
            else:
                writer.writerow([1, "No Species Detected", 0])
                
            writer.writerow([])
            
            methodology_text = (
                "Methodology: The system utilizes custom YOLO neural weights for frame-by-frame object detection. "
                "To prevent double-counting, BoT-SORT / ByteTrack assigns persistent unique IDs across frames using Kalman filtering. "
                "Bounding box coordinates are stabilized via Exponential Moving Averages (EMA)."
            )
            writer.writerow([methodology_text])
            
        print(f"[Eco Census] Successfully exported biological report to: {filepath}")
        return filepath
    except Exception as e:
        print(f"[Eco Census Notice] Error writing CSV report: {e}")
        return None
