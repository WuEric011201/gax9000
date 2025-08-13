"""
UNUSED: currently just doing linear interpolation of measured
height points.

Generate sampled die height compensation map for a wafer from
measured points. See `wafer_height_model` for description of
wafer heightmap compensation model used.

This scripts fits this model to sampled wafer height offset points
and then saves coefficients and pre-generated sampled height offsets
at each die.
"""

import numpy as np
import tomli
from scipy.optimize import curve_fit

def wafer_height_model(
    xy,
    a: float,
    b: float,
    c0: float,
    c1: float,
    d0: float,
    d1: float,
):
    """
    Our heightmap offset model for the wafer is, where x,y is relative
    to an origin (0, 0) at the center of the wafer:

    h = a*x + b*y + (c0*r + c1*r^2) * (d0 + d1*cos(4*theta))

        |_______|   |______________________________________|
            |                         |
      Linear stage   Stress and curvature of silicon wafer.
      plane tilt     The cosine term is due to difference in
                     Young's modulus of silicon along different
                     crystal planes. The quadratic r terms are
                     for the curvature from material stress
                     (e.g. PECVD films).
                     Note: vertical displacement of force on a 
                     beam is actually cubic, but for simplicity
                     just do a 2nd degree fit...
    """
    # unpack x,y depending on xy input shape
    if isinstance(xy, np.ndarray):
        if len(xy.shape) == 1:
            x = xy[0]
            y = xy[1]
        else:
            x = xy[:, 0]
            y = xy[:, 1]
    else: # try unpacking directly, could be tuple or list
        x = xy[0]
        y = xy[1]

    theta = np.arctan2(x, y)
    r = np.sqrt((x**2) + (y**2))
    return a*x + b*y + c0* r * (d0 + (d1*np.cos(4*theta)))


def run_fit_test():
    import matplotlib.pyplot as plt
    from tabulate import tabulate

    # numbers where x,y are units of um
    die_size_x = 24000
    die_size_y = 28000
    scaling = 10000.0 # per 10000 um
    a = 1.0 / scaling
    b = 1.0 / scaling
    c0 = 10.0 / scaling
    c1 = 1.0 / (scaling**2)
    d0 = 0.8
    d1 = 0.2

    # generate die integer coordinate points we would typically sample from, then check result
    points = np.array([
        # cross on xy axis
        [1.0, 0.0],
        [2.0, 0.0],
        [-1.0, 0.0],
        [-2.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, -1.0],
        [0.0, -2.0],
        # off-cross points for theta parameter
        [2.0, 2.0],
        [2.0, -2.0],
        [-2.0, 2.0],
        [-2.0, -2.0],
    ])
    
    # generate height offsets for each point and add some noise
    points_xy = points * np.array([die_size_x, die_size_y])
    points_h = wafer_height_model(points_xy, a, b, c0, c1, d0, d1)
    points_h += np.random.normal(0.0, 0.1, points_h.shape)
    # append 0 height at 0
    points_xy = np.vstack((points_xy, np.array([[0.0, 0.0]])))
    points_h = np.append(points_h, 0.0)
    print(f"points_xy={points_xy}")
    print(f"points_h={points_h}")

    # fit model to points
    popt, pcov = curve_fit(
        wafer_height_model,
        points_xy,
        points_h,
        p0=[1, 1, 1, 1, 1, 0],
        bounds=[
            [-np.inf, -np.inf,     0.0,    0.0, 0.0, -1.0],
            [ np.inf,  np.inf,  np.inf, np.inf, 1.0, 1.0],
        ],
        )

    # unpack fitted parameters
    a_fit, b_fit, c0_fit, c1_fit, d0_fit, d1_fit = popt

    # generate height from fitted parameters
    points_h_fit = wafer_height_model(points_xy, a_fit, b_fit, c0_fit, c1_fit, d0_fit, d1_fit)

    # compare
    print(points_h_fit)
    print(points_h_fit - points_h)

    # compare coefficients
    print(a_fit, b_fit, c0_fit, c1_fit, d0_fit, d1_fit)
    print(a, b, c0, c1, d0, d1)
    comparison = [
        ["a", a, a_fit],
        ["b", b, b_fit],
        ["c0", c0, c0_fit],
        ["c1", c1, c1_fit],
        ["d0", d0, d0_fit],
        ["d1", d1, d1_fit],
    ]

    print(tabulate(comparison, headers=["param", "original", "estimate"]))


def test_create_heightmap(
    path_measurements: str,
):
    """Load die height offset measurements from file and generate heightmap
    using basic linear interpolation. For interpolation methods, see
    https://stackoverflow.com/questions/54432470/how-to-get-a-non-smoothing-2d-spline-interpolation-with-scipy

    Issue with default scipy.interpolate.interp2d is that it relies on
    fitting coefficients, so the resulting curve is smooth. However, this
    does not guarantee the interpolated values will end up going through
    the input points.
    """
    import numpy as np
    from scipy.interpolate import interp2d
    from scipy.interpolate import CloughTocher2DInterpolator
    from tabulate import tabulate

    with open(path_measurements, "rb") as f:
        toml_dict = tomli.load(f)
        print(toml_dict)
        
        # fill arrays of points and height offsets
        num_points = len(toml_dict["die_height_offset"])
        x_vals = np.full((num_points,), np.nan)
        y_vals = np.full((num_points,), np.nan)
        dz_vals = np.full((num_points,), np.nan)

        # each measurement is in format like {x: -1.0, y: -2.0, dz: -4}
        for i, x_y_dz in enumerate(toml_dict["die_height_offset"]):
            x, y, dz = x_y_dz.values()
            print(x, y, dz)
            x_vals[i] = x
            y_vals[i] = y
            dz_vals[i] = dz
        
        print("x_vals", x_vals)
        print("y_vals", y_vals)
        print("dz_vals", dz_vals)

        # print table of interpolated values using two methods
        x_sample = np.array([-3, -2, -1, 0, 1, 2, 3])
        y_sample = np.array([3, 2, 1, 0, -1, -2, -3])

        # 1. scipy.interpolate.interp2d (fits coefficients)
        die_dz_interp2d = interp2d(x_vals, y_vals, dz_vals, kind="cubic")
        dz_table = []
        for y in y_sample:
            dz = np.minimum(0.0, die_dz_interp2d(x_sample, y)) # force clamp to 0.0 max
            dz_table.append(dz)
        print("USING scipy.interpolate.interp2d (fits coefficients)")
        print(tabulate(dz_table))

        # 2. 
        die_dz_ct_interp2d = CloughTocher2DInterpolator(
            np.concatenate((x_vals[:, None], y_vals[:, None]), axis=1),
            dz_vals,
        )
        dz_table2 = []
        for y in y_sample:
            dz = np.minimum(0.0, die_dz_ct_interp2d(x_sample, y))
            dz_table2.append(dz)
        print("USING scipy.interpolate.CloughTocher2DInterpolator (ensures points match)")
        print(tabulate(dz_table2))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and fit wafer height map.")

    parser.add_argument(
        "-i",
        "--input",
        metavar="WAFER_PARAMS",
        dest="path_in",
        type=str,
        help="Path to wafer parameters (die size, height map, see sample)"
    )
    parser.add_argument(
        "-p",
        metavar="path_out",
        dest="path_out",
        type=str,
        help="Path to put image output",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Run fitting test (unit test)",
    )

    args = parser.parse_args()

    print(args)

    if args.test:
        run_fit_test()
        exit(0)
    else:
        test_create_heightmap(args.path_in)
    