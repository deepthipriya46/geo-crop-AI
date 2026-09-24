from geopy.geocoders import Nominatim


def get_coordinates(location):

    geolocator = Nominatim(user_agent="agri_ai")

    loc = geolocator.geocode(location)

    if loc is None:
        raise ValueError(f"Location '{location}' not found")

    return loc.latitude, loc.longitude