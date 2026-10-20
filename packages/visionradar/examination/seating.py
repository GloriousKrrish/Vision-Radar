import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from visionradar.examination.schemas import SeatingZone, ZoneType, PersonDetection

class SeatingZoneManager:
    """
    Manages polygonal seating areas and associations for Examination Intelligence (Layer C).
    Enables authorized users to configure exam desks, aisles, and restricted proctor areas.
    """

    def __init__(self, zones: Optional[List[SeatingZone]] = None):
        self.zones: List[SeatingZone] = zones or []

    def add_zone(self, zone_id: str, name: str, zone_type: ZoneType, polygon: List[List[float]]) -> SeatingZone:
        zone = SeatingZone(zone_id=zone_id, name=name, zone_type=zone_type, polygon=polygon)
        self.zones = [z for z in self.zones if z.zone_id != zone_id]
        self.zones.append(zone)
        return zone

    def remove_zone(self, zone_id: str):
        self.zones = [z for z in self.zones if z.zone_id != zone_id]

    def get_zones(self) -> List[SeatingZone]:
        return list(self.zones)

    def associate_person(self, person: PersonDetection) -> PersonDetection:
        """
        Determines which zone a person occupies based on their ground contact / centroid coordinate.
        """
        if not self.zones:
            return person

        x1, y1, x2, y2 = person.bbox
        # Representative ground/anchor point (bottom-center of bounding box)
        anchor_x = (x1 + x2) / 2.0
        anchor_y = y2

        best_zone = None
        best_dist = -1e9

        for z in self.zones:
            poly_np = np.array(z.polygon, dtype=np.float32)
            if len(poly_np) < 3:
                continue

            # pointPolygonTest: positive = inside, zero = on edge, negative = outside
            dist = cv2.pointPolygonTest(poly_np, (float(anchor_x), float(anchor_y)), measureDist=True)
            if dist >= 0 and dist > best_dist:
                best_dist = dist
                best_zone = z

        if best_zone is not None:
            # Calibrate zone confidence based on distance to nearest edge
            zone_conf = min(1.0, max(0.50, 0.50 + (best_dist / 30.0)))
            person.assigned_zone = best_zone.zone_id
            person.zone_confidence = round(zone_conf, 3)
        else:
            person.assigned_zone = None
            person.zone_confidence = None

        return person

    def check_restricted_zone(self, person: PersonDetection) -> Optional[SeatingZone]:
        """Returns the restricted zone if person is currently inside one."""
        x1, y1, x2, y2 = person.bbox
        anchor_x = (x1 + x2) / 2.0
        anchor_y = y2

        for z in self.zones:
            if z.zone_type == ZoneType.RESTRICTED_ZONE:
                poly_np = np.array(z.polygon, dtype=np.float32)
                if len(poly_np) >= 3:
                    if cv2.pointPolygonTest(poly_np, (float(anchor_x), float(anchor_y)), measureDist=False) >= 0:
                        return z
        return None
