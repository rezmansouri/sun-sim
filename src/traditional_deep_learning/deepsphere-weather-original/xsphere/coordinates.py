import numpy as np

# radius = 6371.0e6


def lonlat2xyz(longitude, latitude, radius=6371.0e6):
    """From 2D geographic coordinates to cartesian geocentric coordinates."""
    ## - lat = phi
    ## - lon = theta
    ## - radius = rho
    lon, lat = np.deg2rad(longitude), np.deg2rad(latitude)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def xyz2lonlat(x, y, z, radius=6371.0e6):
    """From cartesian geocentric coordinates to 2D geographic coordinates."""
    latitude = np.rad2deg(np.arcsin(z / radius))
    longitude = np.rad2deg(np.arctan2(y, x))
    return longitude, latitude


def xyz2polar(x, y, z):
    """From cartesian geocentric coordinates to spherical polar coordinates."""
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


# -----------------------------------------------------------------------------.
## Testing
# x, y, z = xsphere.lonlat2xyz(lon, lat)
# lon1, lat1 = xsphere.xyz2lonlat(x,y,z)
# np.testing.assert_allclose(lon, lon1)
# np.testing.assert_allclose(lat, lat1)

##----------------------------------------------------------------------------.
## Conversion to spherical coordinate is buggy
# def xyz2sph(x,y,z):
#     """From cartesian geocentric coordinates to spherical polar coordinates."""
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(z/r)
#     phi = np.arctan(y, x)
#     return theta, phi, r

# def sph2xyz(theta, phi, radius=1):
#     """From spherical polar coordinates to cartesian geocentric coordinates."""
#     x = radius * np.sin(theta) * np.cos(phi)
#     y = radius * np.sin(theta) * np.sin(phi)
#     z = radius * np.cos(theta)
#     return x, y, z

# def lonlat2sph(longitude, latitude, radius=1):
#     """From 2D geographic coordinates to spherical polar coordinates."""
#     x, y, z = lonlat2xyz(longitude=longitude, latitude=latitude, radius=radius)
#     return xyz2sph(x,y,z)

# def sph2lonlat(theta, phi, radius=1):
#     """From spherical polar coordinates to 2D geographic coordinates."""
#     x, y, z = sph2xyz(theta=theta, phi=phi, radius=1)
#     return xyz2lonlat(x,y,z)

## Testing
# x, y, z = xsphere.lonlat2xyz(lon, lat)
# theta, phi, r = xsphere.xyz2sph(x,y,z)
# x1, y1, z1 = xsphere.sph2xyz(theta, phi, r)
# np.testing.assert_allclose(x, x1)
# np.testing.assert_allclose(y, y1)
# np.testing.assert_allclose(z, z1)

# -----------------------------------------------------------------------------.
