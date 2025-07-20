import argparse
import sys
from typing import List, Tuple

def is_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Determine if a geographical point lies inside a given polygon using the
    Ray Casting (or Winding Number) algorithm.

    This algorithm draws a horizontal ray from the test point to the right
    and counts the number of times it intersects with the edges of the polygon.
    If the count is odd, the point is inside. If even, the point is outside.
    Special handling is included for cases where the point is on an edge or vertex.

    Args:
        point (Tuple[float, float]): A tuple representing the (latitude, longitude)
                                     coordinates of the point to test.
        polygon (List[Tuple[float, float]]): A list of (latitude, longitude) tuples
                                              representing the vertices of the polygon.
                                              The polygon should be a closed loop, meaning
                                              the first and last points are implicitly
                                              connected if not explicitly duplicated.
                                              Vertices should be ordered either clockwise
                                              or counter-clockwise.

    Returns:
        bool: True if the point is inside or on the boundary of the polygon,
              False otherwise.
    """
    x, y = point[1], point[0]  # Use (longitude, latitude) for calculations
    n = len(polygon)
    inside = False

    # Handle edge cases: polygon must have at least 3 vertices
    if n < 3:
        # A point cannot be "inside" a line or single point.
        # This implementation will return False, which is appropriate.
        return False

    # Convert polygon vertices to (lon, lat) tuples for consistency
    poly_coords = [(v[1], v[0]) for v in polygon]

    # Iterate through each edge of the polygon
    # (i, j) are the indices of the current edge's start and end vertices
    for i in range(n):
        j = (i + 1) % n # Wrap around to the first vertex for the last edge
        xi, yi = poly_coords[i]
        xj, yj = poly_coords[j]

        # Check if point is on a horizontal segment of the edge
        # This handles horizontal edges that might otherwise be missed or double-counted
        if yi == yj and yi == y and ((xi <= x <= xj) or (xj <= x <= xi)):
            return True # Point is on a horizontal edge

        # Check if point is on a vertical segment of the edge (or a vertex)
        # This handles cases where the ray passes directly through a vertex.
        # It ensures that only one intersection is counted for shared vertices.
        if (x == xi and y == yi) or (x == xj and y == yj):
            return True # Point is on a vertex

        # Check for intersection using the ray casting algorithm
        # This condition checks if the ray crosses the horizontal line segment defined by the edge
        intersect = ((yi <= y < yj) or (yj <= y < yi)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi) + xi)

        if intersect:
            inside = not inside

    return inside

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Determine if a geographical point lies inside a given polygon.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--point', '-p',
        type=float,
        nargs=2, # Expect exactly two floats for lat lon
        required=True,
        metavar=('LAT', 'LON'),
        help="The latitude and longitude of the point to test (e.g., '34.0 118.0')."
    )
    parser.add_argument(
        '--polygon', '-P',
        type=str,
        required=True,
        nargs='+', # Expect one or more strings for polygon vertices
        help="List of polygon vertices as 'LAT,LON' strings. "
             "Example: '-P \"34.0,-118.0\" \"34.1,-118.0\" \"34.1,-118.1\" \"34.0,-118.1\"'.\n"
             "Each vertex should be quoted if it contains special characters like commas in shell."
    )

    args = parser.parse_args()

    # Parse polygon vertices from string arguments
    parsed_polygon = []
    try:
        for v_str in args.polygon:
            lat_str, lon_str = v_str.split(',')
            parsed_polygon.append((float(lat_str), float(lon_str)))
    except ValueError:
        print("Error: Polygon vertices must be in 'LAT,LON' format. Example: '34.0,-118.0'", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)
    except IndexError:
        print("Error: Each polygon vertex must contain both latitude and longitude separated by a comma.", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)

    test_point = tuple(args.point)

    is_inside = is_point_in_polygon(test_point, parsed_polygon)
    if is_inside:
        print(f"Point {test_point} is INSIDE or ON THE BOUNDARY of the provided polygon.")
    else:
        print(f"Point {test_point} is OUTSIDE the provided polygon.")

    print(f"\nTested Point: {test_point}")
    print(f"Tested Polygon: {parsed_polygon}")
