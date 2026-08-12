"""Visualize every camera configured for a task."""

import argparse

import cv2
import numpy as np

from experiments.mappings import CONFIG_MAPPING


WINDOW = "Configured cameras"


def make_view(images):
    if not images:
        raise ValueError("no cameras are configured")

    panels = []
    for name, rgb in images.items():
        panel = cv2.resize(rgb[..., ::-1], (320, 320), interpolation=cv2.INTER_LINEAR)
        cv2.putText(
            panel,
            name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def camera_sources(env):
    env = env.unwrapped
    if hasattr(env, "env_left"):
        return [("left/", env.env_left), ("right/", env.env_right)]
    return [("", env)]


def open_camera(source):
    if hasattr(source, "get_images"):
        source.init_cameras()
        return source.get_images

    source.cap = None
    source.init_cameras(source.config.REALSENSE_CAMERAS)
    return source.get_im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(CONFIG_MAPPING))
    args = parser.parse_args()

    config = CONFIG_MAPPING[args.task]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    sources = camera_sources(env)
    readers = []

    try:
        for prefix, source in sources:
            readers.append((prefix, source, open_camera(source)))

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        while True:
            images = {
                prefix + name: image
                for prefix, _, read in readers
                for name, image in read().items()
            }
            cv2.imshow(WINDOW, make_view(images))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        for _, source, _ in readers:
            source.close_cameras()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
