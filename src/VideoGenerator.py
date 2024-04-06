from examples.Demo import DemoImageHolder
import os
from moviepy.editor import *
import glob
from natsort import natsorted
from joblib import Parallel, delayed
from examples.Demo import DemoImageHolder
from examples.SierpinskiesWithLFOs import SierpinskiesWithLFOs
from examples.SWLFOsMA import SWLFOsMA
from examples.PixeF import PixeF

class VideoGenerator:
    """
    A class that generates videos using an image holder.

    Args:
        image_holder: An instance of the ImageHolder class.

    Attributes:
        image_holder: An instance of the ImageHolder class.

    """

    def __init__(self, image_holder):
        self.image_holder = image_holder

    def do_video_with_IH(self, fps=25, duration=10, size=512, name="demo", nb_jobs=-3, output_format="mp4"):
        """
        Generates a video using the image holder. Writes it in the `../images/name` folder.

        Args:
            fps (int, optional): Frames per second of the video. Defaults to 25.
            duration (int, optional): Duration of the video in seconds. Defaults to 10.
            size (int, optional): Size of the images in pixels. Defaults to 512.
            name (str, optional): Name of the video. Defaults to "demo".
            nb_jobs (int, optional): Number of parallel jobs to use. Defaults to -3.
            output_format (str, optional): Output format of the video. Either "gif" or "mp4". Defaults to "mp4".

        """
        n_im = duration * fps

        # create the images
        images_to_generate = [
            self.image_holder(i, n_im, name=name, size=size) for i in range(n_im + 1)
        ]

        # compute the images
        Parallel(n_jobs=nb_jobs)(
            delayed(images_to_generate[i].run)() for i in range(n_im + 1)
        )

        # take all files in the folder and sort them
        base_dir = os.path.realpath(f"../images/{name}/")
        file_list = glob.glob(f"{base_dir}/{name}*.png")
        file_list_sorted = natsorted(file_list, reverse=False)

        # create the video
        clips = [
            ImageClip(m).set_duration(1 / fps) for m in file_list_sorted
        ]
        concat_clip = concatenate_videoclips(clips, method="compose")
        
        if output_format == "gif":
            concat_clip.write_gif(
                f"{base_dir}/{name}.gif", fps=fps, program="ffmpeg"
            )
        elif output_format == "mp4":
            concat_clip.write_videofile(
                f"{base_dir}/{name}.mp4", fps=fps, codec="libx264"
            )
        else:
            raise ValueError("Invalid output format. Please choose either 'gif' or 'mp4'.")


if __name__ == "__main__":
    
    # video_generator = VideoGenerator(SierpinskiesWithLFOs) 
    # video_generator = VideoGenerator(SWLFOsMA)

    video_generator.do_video_with_IH(fps=30, 
                                     duration=20, 
                                     size=512, 
                                     name="SWLFOsMA3", 
                                     nb_jobs=-3,
                                     output_format="gif")