import math
import argparse

def haversine(lat1: float, lon1: float, lat2: float, lon2: float, unit: str = 'km') -> float:
    """
    Calculate the great-circle distance between two points on a sphere
    (e.g., Earth) given their latitudes and longitudes using the Haversine formula.

    The Haversine formula is particularly suitable for short and medium distances,
    and it accounts for the Earth's curvature, providing more accurate results
    than simpler planar distance calculations for geographical points.

    Args:
        lat1 (float): Latitude of the first point in degrees. Must be between -90 and 90.
        lon1 (float): Longitude of the first point in degrees. Must be between -180 and 180.
        lat2 (float): Latitude of the second point in degrees. Must be between -90 and 90.
        lon2 (float): Longitude of the second point in degrees. Must be between -180 and 180.
        unit (str): The desired unit for the returned distance.
                    Accepted values are 'km' for kilometers (default),
                    'miles' for miles, or 'nautical_miles' for nautical miles.

    Returns:
        float: The calculated distance between the two geographical points
               in the specified unit.

    Raises:
        ValueError: If an invalid unit is provided.
    """
    # Earth's mean radius in various units (WGS 84 ellipsoid mean radius)
    R_km = 6371.0  # kilometers
    R_miles = 3958.8  # miles
    R_nautical_miles = 3440.065  # nautical miles (1 nautical mile = 1852 meters)

    # Convert latitudes and longitudes from degrees to radians for trigonometric functions
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate the differences in longitude and latitude
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Apply the Haversine formula
    # a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2

    # c = 2 ⋅ atan2(√a, √(1−a))
    # This step calculates the angular distance in radians
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Determine the Earth's radius based on the requested unit
    if unit.lower() == 'km':
        earth_radius = R_km
    elif unit.lower() == 'miles':
        earth_radius = R_miles
    elif unit.lower() == 'nautical_miles':
        earth_radius = R_nautical_miles
    else:
        raise ValueError("Invalid unit. Please choose 'km', 'miles', or 'nautical_miles'.")

    # Calculate the final distance (distance = radius * angular_distance)
    distance = earth_radius * c

    return distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the great-circle distance between two geographical points using the Haversine formula.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--lat1', '-l1',
        type=float,
        required=True,
        help="Latitude of the first point in degrees (e.g., 40.7128 for NYC)."
    )
    parser.add_argument(
        '--lon1', '-o1',
        type=float,
        required=True,
        help="Longitude of the first point in degrees (e.g., -74.0060 for NYC)."
    )
    parser.add_argument(
        '--lat2', '-l2',
        type=float,
        required=True,
        help="Latitude of the second point in degrees (e.g., 51.5074 for London)."
    )
    parser.add_argument(
        '--lon2', '-o2',
        type=float,
        required=True,
        help="Longitude of the second point in degrees (e.g., -0.1278 for London)."
    )
    parser.add_argument(
        '--unit', '-u',
        type=str,
        default='km',
        choices=['km', 'miles', 'nautical_miles'],
        help="Unit for the returned distance. Choose 'km' (default), 'miles', or 'nautical_miles'."
    )

    args = parser.parse_args()

    try:
        distance = haversine(args.lat1, args.lon1, args.lat2, args.lon2, args.unit)
        print(f"The Haversine distance between ({args.lat1}, {args.lon1}) and ({args.lat2}, {args.lon2}) is:")
        print(f"{distance:.2f} {args.unit}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        parser.print_help(sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
