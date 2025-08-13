"""
Script for taking remote die photographs in Cascade system using GPIB.
Support both taking single images or doing an array of images and
auto-stitching to create a die photo. 
"""

import pyvisa
import os
import time
import shutil
import tomli

# Example configuration toml for taking images. Contains paths and calibration
# settings specific to each cascade controller.
SAMPLE_CONFIG_TOML = \
"""
# instrument address on gpib network
gpib_address = "GPIB0::22::INSTR"

# paths where images are stored on the tool, accessed using the
# institution network remote file system access.
# (make sure paths are single quoted, so backslash not escaped)
path_to_img_dir_on_network = '\\\\GAX1-PROBE\shared\gax_tmp_img'
path_to_img_dir_on_remote = 'D:\shared\gax_tmp_img'

# size of output images in pixels
img_size_x = 1024
img_size_y = 768

# user-defined image zoom levels and calibrations.
# names and pixels/um at each zoom level (must be calculated)
[calibration.1]
name = "eVue1"
um_per_pixel = 2.46 # experimentally calibrated

[calibration.2]
name = "eVue2"
um_per_pixel = 0.56603774 # 300 um / 530 px

[calibration.3]
name = "eVue3"
um_per_pixel = 0.22321429 # 100 um / 448 px
"""

def timestamp(format="%Y_%m_%d_%H_%M_%S"):
    """Return detailed timestamp string"""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime(format)

class SimpleCascadeController:
    """Stripped down version of cascade controller in `controller.backend.controller.py`.
    Contain minimal functions needed to move stage and take images.
    """
    def __init__(
        self,
        addr: str,
        invert_direction: bool = True,
    ):
        """Connect to cascade at input gpib address. Return this object mapped to
        the pyvisa gpib controller.
        Inputs:
            - addr: gpib address
            - invert_direction: invert x,y coordinate direction, inverted means
                top-left is origin, bottom-right is positive x,y (default on MIT
                novels group cascades)
        """
        gpib_resource_manager = pyvisa.ResourceManager()
        self.gpib = gpib_resource_manager.open_resource(addr)
        print(self.gpib.query("*IDN?"))
        self.invert_direction = invert_direction

    def close(self):
        """Close gpib connection to cascade instrument."""
        self.gpib.close()
    
    def write(self, cmd: str):
        """Wrapper for pyvisa gpib write."""
        self.gpib.write(cmd)
    
    def read(self):
        """Wrapper for pyvisa gpib read."""
        self.gpib.read()
    
    def query(self, cmd: str):
        """Wrapper for pyvisa gpib query."""
        self.gpib.query(cmd)
    
    def snap_image(self, zoom: str, path_to_img: str):
        """Return cascade GPIB command to snap image."""
        cmd = f"SnapImage {zoom} {path_to_img}"
        self.gpib.write(cmd)
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")

    def set_chuck_home(self):
        """Set cascade autoprobe chuck home to current location.
        This is used in measurements to probe arrays relative to
        starting location.
            SetChuckHome Mode Unit
        Mode:
            0 - use current position
            V - use given value
        Unit
            Y - micron (default)
            I - mils
        """
        self.gpib.write(f"SetChuckHome 0 Y")
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")

    def move_chuck_relative(self, dx, dy):
        """Moves cascade autoprobe chuck relative to current location
        by (dx, dy). Command format is
            MoveChuck X Y PosRef Unit Velocity Compensation
        X: dx
        Y: dy
        PosRef:
            - H - home (default)
            - Z - zero
            - C - center
            - R - current position
        Unit:
            - Y - Micron (default)
            - I - Mils
            - X - Index
            - J - jog
        Velocity: velocity in percent (100% default)
        Compensation:
            - D - default (kernel setup default compensation)
            - T - technology, use prober, offset, and tech compensation
            - O - offset, use prober and offset
            - P - prober, use only prober
            - N - none, no compensation
        """
        if self.invert_direction:
            dx_ = -dx
            dy_ = -dy
        else:
            dx_ = dx
            dy_ = dy
        
        self.gpib.write(f"MoveChuck {dx_} {dy_} R Y 100")
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")

    def move_chuck_relative_to_home(self, x, y):
        """Moves wafer chuck relative to home position. See `move_chuck_relative`
        for MoveChuck command documentation.
        """
        if self.invert_direction:
            x_ = -x
            y_ = -y
        else:
            x_ = x
            y_ = y
        
        self.gpib.write(f"MoveChuck {x_} {y_} H Y 100")
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")

    def move_to_chuck_home(self):
        """Move chuck to previously set home position. See `move_chuck_relative`
        for MoveChuck command documentation.
        """
        self.gpib.write(f"MoveChuck 0 0 H Y 100")
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")


def move_image_from_remote_when_ready(
    path_remote: str,
    path_out: str,
    poll_timeout = 0.5, # timeout between poll checks
    timeout = 30.0,     # max allowed before error 30s
):
    """Issue is there is no way to query when image is ready after
    sending the GPIB command. So instead rely on polling to check
    if image is done. When done, move it from remote file system to
    desired output path.

    From testing, it can take 8-10s for image to be ready and appear, so
    polling is done by default at coarse 0.5s intervals.
    """
    tstart = time.perf_counter()

    while True:
        # print(f"POLLING ({time.perf_counter() - tstart})")
        if os.path.exists(path_remote):
            shutil.move(path_remote, path_out)
            return
        else:
            dt = time.perf_counter() - tstart
            if dt > timeout:
                raise RuntimeError(f"Failed to get output image in time {path_remote} (timeout: {timeout})")
            time.sleep(poll_timeout)


def stitch_images(
    config: dict,
    path_out: str,
    count_x: int = None,     # number of images on x-axis, None -> autosearch
    count_y: int = None,     # number of images on y axis, None -> autosearch
    margin: float = 0.2,     # margin on each side that is ignored
    downsample: int = 2.0,   # downsampling factor for images
    contrast: float = 0.75,  # contrast filtering, helps reduces lighting discontinuity at edge
    format: str = "png",
    show_image: bool = True, # display output image in system image viewer after stitching
):
    """Routine to stitch together a folder of raw images into a single die 
    photo panorama. Each image is as follows:

                ignored outer region
         _____________v____
        |   ____________   |       The border area ignored is the "margin"
        |  |            |  |       input parameter, which is subtracted from
        |  |  use this  |  |       each edge.
        |  |____________|  |
        |__________________|
    
    Each image has outer margin we ignore. This is because images taken have 
    shadowing near borders that causes lighting discontinuities between
    images. What we do:
    1. Calculate merged image size after including downsampling.
    2. Create output merged image buffer.
    3. Foreach image:
        3.1. Downsample (before cropping to reduce aliasing at edges)
        3.2. Crop center region
        3.3. Insert into correct coordinates in output img buffer.
    
    Downsampling then stitching will introduce 
    If input count_x or count_y are None, this must search raw image folder
    to find number of images along each axis.
    """
    from math import floor, ceil
    from PIL import Image, ImageEnhance

    # generate a folder for this run
    path_raw_images = os.path.join(path_out, "raw")
    if not os.path.exists(path_raw_images):
        raise ValueError(f"No raw images to stitch in path {path_raw_images}")

    # if count not specified, search through images to find images taken 
    # on each axis
    if count_x is None or count_y is None:
        print(f"count_x or count_y not specified...searching for images in {path_raw_images}...")
        count_x = 0
        count_y = 0
        for p in os.listdir(path_raw_images):
            if p.endswith("_raw.png"):
                s_chunks = p.split("_")
                count_x = max(count_x, int(s_chunks[0]))
                count_y = max(count_y, int(s_chunks[1]))
        # add 1 to get actual array size
        count_x += 1
        count_y += 1
        print(f"Found count = ({count_x}, {count_y})")

    # calculate um actually used per image. insert margin in images,
    # e.g. stitch with margin overlap between images on each border
    raw_img_margin_x = round(margin * config["img_size_x"])
    raw_img_margin_y = round(margin * config["img_size_y"])

    raw_img_chunk_size_x_px = round(config["img_size_x"] - 2*raw_img_margin_x)
    raw_img_chunk_size_y_px = round(config["img_size_y"] - 2*raw_img_margin_y)

    downsampled_img_full_size_x_px = round(float(config["img_size_x"]) / downsample)
    downsampled_img_full_size_y_px = round(float(config["img_size_y"]) / downsample)

    downsampled_img_chunk_size_x_px = round(float(raw_img_chunk_size_x_px) / downsample)
    downsampled_img_chunk_size_y_px = round(float(raw_img_chunk_size_y_px) / downsample)

    # crop area for each downsampled image, (left, upper, right, lower)
    # on each edge, remove margin of size downsampled_img_margin_x and
    # downsampled_img_margin_y
    # adding 1 to downsample margin seems to improve image stitching...wtf?
    downsampled_img_margin_x =  1 + round(float(raw_img_margin_x) / downsample)
    downsampled_img_margin_y =  1 + round(float(raw_img_margin_y) / downsample)
    croparea = (
        downsampled_img_margin_x,
        downsampled_img_margin_y,
        downsampled_img_margin_x + downsampled_img_chunk_size_x_px,
        downsampled_img_margin_y + downsampled_img_chunk_size_y_px,
    )

    # print(croparea)

    # stitch images together starting from top-left
    img = Image.new(mode="RGB", size=(count_x * downsampled_img_chunk_size_x_px, count_y * downsampled_img_chunk_size_y_px))
    # print(img)

    for nx in range(count_x):
        for ny in range(count_y):
            path_img_chunk = os.path.join(path_raw_images, f"{nx}_{ny}_raw.png")
            img_chunk = Image.open(path_img_chunk)
            img_chunk = img_chunk.resize((downsampled_img_full_size_x_px, downsampled_img_full_size_y_px), resample=Image.Resampling.BILINEAR)
            img_chunk = img_chunk.crop(croparea)

            # TODO: make contrast an input parameter
            filter_contrast = ImageEnhance.Contrast(img_chunk)
            img_chunk = filter_contrast.enhance(contrast)

            x0 = nx * downsampled_img_chunk_size_x_px
            y0 = ny * downsampled_img_chunk_size_y_px
            img.paste(img_chunk, box=(x0, y0))
    
    path_img_combined_out = os.path.join(path_out, "die_photo.png")
    img.save(path_img_combined_out)

    print(f"Saved stitched image to: {path_img_combined_out}")

    if show_image:
        img.show()


def take_die_photo(
    config: dict,
    zoom: str,
    size_x: int,
    size_y: int,
    path_out: str,
    margin: float = 0.2,    # margin on each side that is ignored
    downsample: int = 2.0,  # downsampling factor for images
    format: str = "png",
):
    """Do a sweep to get a die photo over input x, y size range.
    Note: using PIL which uses top-left origin, e.g.
        (0,0) -----> +x
          |
          |
       +y v
    So we will also save images from top-left to bottom-right.

    Another issue is the images taken typically have shadowing at the borders. For higher
    quality we want to use the center region of each image for more similar lighting,
    e.g.
                ignore outer region
         _____________v____
        |   ____________   |            The border area ignored is the "margin"
        |  |            |  |            input parameter, which is subtracted from
        |  |  use this  |  |            each edge.
        |  |____________|  |
        |__________________|
    """
    from math import ceil

    path_to_img_dir_on_network = config["path_to_img_dir_on_network"]
    path_to_img_dir_on_remote = config["path_to_img_dir_on_remote"]

    # calculate um actually used per image. insert margin in images,
    # e.g. stitch with margin overlap between images on each border
    img_margin_x = round(margin * config["img_size_x"])
    img_margin_y = round(margin * config["img_size_y"])

    img_size_x_px = round(config["img_size_x"] - 2*img_margin_x)
    img_size_y_px = round(config["img_size_y"] - 2*img_margin_y)
    um_per_px = config["calibration"][zoom]["um_per_pixel"]
    img_sixe_x_um = um_per_px * img_size_x_px
    img_size_y_um = um_per_px * img_size_y_px

    # origin offset due to the margin
    dx0_px = -img_margin_x
    dy0_px = img_margin_y
    dx0_um = dx0_px * um_per_px
    dy0_um = dy0_px * um_per_px

    # count how many images we need to take to fit image of (size_x, size_y)
    count_x = ceil(size_x / img_sixe_x_um)
    count_y = ceil(size_y / img_size_y_um)
    total_count = count_x * count_y

    print(f"die photo size = ({size_x}, {size_y}) um")
    print(f"per img size px used = ({img_size_x_px}, {img_size_y_px}) px")
    print(f"per img size = ({img_sixe_x_um:.3f}, {img_size_y_um:.3f}) um")
    print(f"img count = ({count_x}, {count_y})")

    # generate a folder for this run
    path_die_photo = os.path.join(path_out, f"die_{timestamp()}")
    path_die_photo_raw = os.path.join(path_die_photo, "raw")
    os.makedirs(path_die_photo_raw, exist_ok=True)

    controller = SimpleCascadeController(config["gpib_address"])

    # mark home
    controller.set_chuck_home()

    # sweep array and take image
    n = 1
    for nx in range(count_x):
        for ny in range(count_y):
            img_name = f"{nx}_{ny}.png"
            img_raw_name = f"{img_name[:-4]}_raw.png"

            # path to save image locally on the remote computer
            path_to_save_img_remote = os.path.join(path_to_img_dir_on_remote, img_name)

            # remote directory paths, e.g. \\GAX1-PROBE\shared\gax_tmp_img
            path_img_remote = os.path.join(path_to_img_dir_on_network, img_name)
            path_img_remote_raw = os.path.join(path_to_img_dir_on_network, img_raw_name)

            # local computer output paths
            path_img_out = os.path.join(path_die_photo_raw, img_name)
            path_img_out_raw = os.path.join(path_die_photo_raw, img_raw_name)

            # offsets relative to home position on top-left
            dx = dx0_um + nx * img_sixe_x_um
            dy = dy0_um + (count_y - 1 - ny) * img_size_y_um
            
            # print(dx, dy)

            controller.move_chuck_relative_to_home(dx, dy)
            controller.snap_image(config["calibration"][zoom]["name"], path_to_save_img_remote)

            print(f"Saving img {n}/{total_count}: {path_img_out_raw}")
            move_image_from_remote_when_ready(path_img_remote, path_img_out)
            move_image_from_remote_when_ready(path_img_remote_raw, path_img_out_raw)

            n += 1
    
    # move back to home position
    controller.move_to_chuck_home()
    controller.close()

    stitch_images(
        config=config,
        path_out=path_die_photo,
        count_x=count_x,
        count_y=count_y,
        margin=margin,
        downsample=downsample,
        format=format,
        show_image=True,
    )


def take_image(
    config: dict,
    zoom: str,
    path_out: str,
    format: str = "png",
):
    """Take single image snapshot and save to output directory."""
    controller = SimpleCascadeController(config["gpib_address"])

    path_to_img_dir_on_network = config["path_to_img_dir_on_network"]
    path_to_img_dir_on_remote = config["path_to_img_dir_on_remote"]

    print(f"path_to_img_on_network: {path_to_img_dir_on_network}")
    print(f"path_to_img_dir_on_remote: {path_to_img_dir_on_remote}")

    img_name = f"snapshot_zoom_{zoom}.{format}"
    img_raw_name = f"snapshot_zoom_{zoom}_raw.{format}" # raw image has no labels

    path_to_save_img_remote = os.path.join(path_to_img_dir_on_remote, img_name)
    controller.snap_image(config["calibration"][zoom]["name"], path_to_save_img_remote)
    controller.close()

    # copy image to local output
    path_img_remote = os.path.join(path_to_img_dir_on_network, img_name)
    path_img_out = os.path.join(path_out, img_name)
    path_img_remote_raw = os.path.join(path_to_img_dir_on_network, img_raw_name)
    path_img_out_raw = os.path.join(path_out, img_raw_name)
    move_image_from_remote_when_ready(path_img_remote, path_img_out)
    move_image_from_remote_when_ready(path_img_remote_raw, path_img_out_raw)
    print(f"Saved image to: {path_img_out}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform die photo using image stitching.")

    parser.add_argument(
        "-c",
        "--config",
        metavar="config",
        type=str,
        help="Path to instrument config/calibration"
    )
    parser.add_argument(
        metavar="path_out",
        dest="path_out",
        type=str,
        help="Path to put image output"
    )
    parser.add_argument(
        "-z",
        "--zoom",
        metavar="ZOOM",
        dest="zoom",
        type=str,
        default="1",
        help="Zoom level for snapshot"
    )
    parser.add_argument(
        "-x",
        metavar="UM",
        dest="size_x",
        type=int,
        help="Total x size in [um] (for creating panorama)"
    )
    parser.add_argument(
        "-y",
        metavar="UM",
        dest="size_y",
        type=int,
        help="Total y size in [um] (for creating panorama)"
    )
    parser.add_argument(
        "-d",
        metavar="DOWNSAMPLE",
        dest="downsample",
        type=float,
        default=2,
        help="Set downsampling factor for stitched panorama. Default downscaling is 2x."
    )
    parser.add_argument(
        "--stitch",
        dest="stitch",
        action="store_true",
        help="If specified, searches path_out for existing images to stitch together instead of taking images."
    )

    args = parser.parse_args()

    print(args)
    
    if args.config is not None:
        with open(args.config, "rb") as f:
            config = tomli.load(f)
    else:
        print("No config file specified...using sample config settings")
        config = tomli.loads(SAMPLE_CONFIG_TOML)
    
    print(config)

    if args.stitch:
        # stitch together images in existing folder without taking new images
        stitch_images(
            config=config,
            path_out=args.path_out,
            downsample=args.downsample,
            show_image=True,
        )
    else: # will take new images and stitch together
        zoom = args.zoom
        if zoom is None or zoom not in config["calibration"]:
            zoom_settings = config["calibration"]
            raise ValueError(f"Invalid zoom setting: {zoom}, must be one of {list(zoom_settings.keys())}")

        # if x, y arguments specified, take a panorama
        x = args.size_x
        y = args.size_y
        if x is None and y is None: # take single image
            take_image(config, zoom, args.path_out)
        else:
            if x is None or y is None: # must both be specified
                raise ValueError(f"For taking panorama, must specify both -x and -y: currently x: {x}, y: {y}")
            
            take_die_photo(
                config=config,
                zoom=zoom,
                size_x=x,
                size_y=y,
                path_out=args.path_out,
                downsample=args.downsample,
            )
