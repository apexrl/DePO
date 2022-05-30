import os

import numpy as np


def _make_dir(filename=None, folder_name=None):
    folder = folder_name
    if folder is None:
        folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_image(video_frames, filedir, start_frame, format="png"):

    _make_dir(folder_name=filedir)
    from PIL import Image, ImageDraw, ImageFont

    for i, frame in enumerate(video_frames):
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im)
        fnt = ImageFont.truetype(
            "/NAS2020/Share/mhliu/liberation/LiberationSans-Regular.ttf", 40
        )

        # draw text, half opacity
        d.text(
            (10, 10), "Step {}".format(start_frame + i), font=fnt, fill=(255, 255, 204)
        )
        im.save("{}/img_{}.{}".format(filedir, i, format))


def save_video(video_frames, filename, fps=60, video_format="mp4"):
    assert fps == int(fps), fps
    import skvideo.io

    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            "-r": str(int(fps)),
        },
        outputdict={
            "-f": video_format,
            "-pix_fmt": "yuv420p",  # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        },
    )


def create_video_grid(col_and_row_frames):
    video_grid_frames = np.concatenate(
        [np.concatenate(row_frames, axis=-2) for row_frames in col_and_row_frames],
        axis=-3,
    )

    return video_grid_frames
