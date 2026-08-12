"""Browse recorded demo images and every value used by the replay buffer."""

import argparse
import math
import pickle
from pathlib import Path

import cv2
import numpy as np


WINDOW = "Demo data"
PROGRESS = "Frame"
IMAGE_SIZE = 256
IMAGE_HEADER = 28
TOP_HEIGHT = 44
FOOTER_HEIGHT = 52
TEXT_WIDTH = 640
LINE_HEIGHT = 20
BUTTONS = (
    ("previous_file", "< File"),
    ("previous_frame", "< Frame"),
    ("next_frame", "Frame >"),
    ("next_file", "File >"),
)


def load_data(path):
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a transition list")
    return data


def find_files(inputs):
    paths = []
    for item in inputs:
        path = Path(item)
        paths.extend(sorted(path.glob("*.pkl")) if path.is_dir() else [path])
    paths = list(dict.fromkeys(paths))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    if not paths:
        raise FileNotFoundError("no .pkl files found")
    return paths


def load_nonempty(paths, start, step=1):
    for offset in range(len(paths)):
        file_index = (start + offset * step) % len(paths)
        data = load_data(paths[file_index])
        if data:
            return file_index, data
    return None, None


def is_image(value):
    image = np.asarray(value)
    return (
        image.ndim in (3, 4)
        and image.shape[-1] in (1, 3, 4)
        and image.shape[-2] >= 16
        and image.shape[-3] >= 16
    )


def to_bgr(value):
    image = np.asarray(value)
    if image.ndim == 4:
        image = image[-1]
    if image.dtype != np.uint8:
        maximum = np.nanmax(image) if image.size else 0
        scale = 255 if maximum <= 1 else 1
        image = np.nan_to_num(image * scale)
        image = np.clip(image, 0, 255).astype(np.uint8)
    conversion = {
        1: cv2.COLOR_GRAY2BGR,
        3: cv2.COLOR_RGB2BGR,
        4: cv2.COLOR_RGBA2BGR,
    }[image.shape[-1]]
    return cv2.cvtColor(image, conversion)


def image_names(transition):
    names = []
    for section in ("observations", "next_observations"):
        for name, value in transition.get(section, {}).items():
            if is_image(value) and name not in names:
                names.append(name)
    return names


def format_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):+.6g}"
    return repr(value)


def numeric_lines(value, prefix=""):
    lines = []
    if isinstance(value, dict):
        for name, child in value.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            lines.extend(numeric_lines(child, child_prefix))
        return lines
    if is_image(value):
        return lines

    array = np.asarray(value)
    if array.ndim == 0:
        return [f"{prefix}: {format_scalar(array.item())}"]

    flat = array.reshape(-1)
    lines.append(f"{prefix}  shape={array.shape}  dtype={array.dtype}")
    for start in range(0, flat.size, 4):
        values = "  ".join(
            f"[{index}]={format_scalar(flat[index])}"
            for index in range(start, min(start + 4, flat.size))
        )
        lines.append(f"  {values}")
    if not flat.size:
        lines.append("  (empty)")
    return lines


def episode_position(data, index):
    ends = [
        position
        for position, transition in enumerate(data)
        if bool(np.asarray(transition.get("dones", False)).item())
    ]
    previous_end = max((position for position in ends if position < index), default=-1)
    next_end = min((position for position in ends if position >= index), default=len(data) - 1)
    episode = sum(position < index for position in ends) + 1
    total = len(ends) + (not ends or ends[-1] != len(data) - 1)
    return episode, int(total), index - previous_end, next_end - previous_end


def image_panel(value, label):
    panel = np.full((IMAGE_HEADER + IMAGE_SIZE, IMAGE_SIZE, 3), 24, np.uint8)
    if value is not None:
        panel[IMAGE_HEADER:] = cv2.resize(
            to_bgr(value),
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
    cv2.putText(
        panel,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (100, 230, 130),
        1,
        cv2.LINE_AA,
    )
    return panel


def button_rects(width, height):
    gap = 8
    button_width = min(170, (width - gap * 5) // len(BUTTONS))
    start = (width - button_width * len(BUTTONS) - gap * (len(BUTTONS) - 1)) // 2
    return [
        (
            action,
            label,
            (
                start + index * (button_width + gap),
                height - 44,
                start + index * (button_width + gap) + button_width,
                height - 8,
            ),
        )
        for index, (action, label) in enumerate(BUTTONS)
    ]


def render_transition(transition, path, index, data):
    if not isinstance(transition, dict):
        raise ValueError(f"transition {index + 1} in {path} is not a dictionary")
    names = image_names(transition)
    if not names:
        raise ValueError(f"transition {index + 1} in {path} contains no images")

    panel_height = IMAGE_HEADER + IMAGE_SIZE
    content_height = panel_height * 2
    lines = ["TRAINING VALUES", *numeric_lines(transition)]
    lines_per_column = max(1, (content_height - 12) // LINE_HEIGHT)
    text_columns = max(1, math.ceil(len(lines) / lines_per_column))
    image_width = IMAGE_SIZE * len(names)
    width = image_width + TEXT_WIDTH * text_columns
    height = TOP_HEIGHT + content_height + FOOTER_HEIGHT
    frame = np.full((height, width, 3), 20, np.uint8)

    episode, episode_total, step, episode_length = episode_position(data, index)
    title = (
        f"{path.name}   frame {index + 1}/{len(data)}   "
        f"episode {episode}/{episode_total}   step {step}/{episode_length}"
    )
    cv2.putText(
        frame,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for row, section in enumerate(("observations", "next_observations")):
        observations = transition.get(section, {})
        short_name = "obs" if row == 0 else "next"
        for column, name in enumerate(names):
            panel = image_panel(observations.get(name), f"{short_name}.{name}")
            y = TOP_HEIGHT + row * panel_height
            x = column * IMAGE_SIZE
            frame[y : y + panel_height, x : x + IMAGE_SIZE] = panel

    cv2.line(
        frame,
        (image_width, TOP_HEIGHT),
        (image_width, TOP_HEIGHT + content_height),
        (80, 80, 80),
        1,
    )
    for line_index, line in enumerate(lines):
        column = line_index // lines_per_column
        row = line_index % lines_per_column
        color = (100, 230, 130) if line == "TRAINING VALUES" else (225, 225, 225)
        cv2.putText(
            frame,
            line,
            (image_width + column * TEXT_WIDTH + 12, TOP_HEIGHT + 20 + row * LINE_HEIGHT),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )

    for action, label, (x1, y1, x2, y2) in button_rects(width, height):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (65, 65, 65), -1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(
            frame,
            label,
            (x1 + (x2 - x1 - text_size[0]) // 2, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def on_mouse(event, x, y, _flags, state):
    if event != cv2.EVENT_LBUTTONUP:
        return
    for action, _label, (x1, y1, x2, y2) in state["buttons"]:
        if x1 <= x <= x2 and y1 <= y <= y2:
            state["command"] = action
            return


def set_progress(index, total):
    cv2.setTrackbarMax(PROGRESS, WINDOW, total - 1)
    cv2.setTrackbarPos(PROGRESS, WINDOW, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["demo_data"],
        help="demo pickle files or directories",
    )
    args = parser.parse_args()

    paths = find_files(args.paths)
    file_index, data = load_nonempty(paths, 0)
    if data is None:
        raise ValueError("all demo data files are empty")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar(PROGRESS, WINDOW, 0, len(data) - 1, lambda _: None)
    mouse = {"buttons": [], "command": None}
    cv2.setMouseCallback(WINDOW, on_mouse, mouse)
    shown = None
    print("Use the progress bar and window buttons; q/Esc quits")
    try:
        while True:
            index = min(cv2.getTrackbarPos(PROGRESS, WINDOW), len(data) - 1)
            current = (file_index, index)
            if current != shown:
                frame = render_transition(data[index], paths[file_index], index, data)
                mouse["buttons"] = button_rects(frame.shape[1], frame.shape[0])
                cv2.imshow(WINDOW, frame)
                shown = current

            key = cv2.waitKey(30) & 0xFF
            command = mouse["command"]
            mouse["command"] = None
            window_closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            if key in (ord("q"), 27) or window_closed:
                break
            if command == "previous_frame":
                cv2.setTrackbarPos(PROGRESS, WINDOW, max(index - 1, 0))
            elif command == "next_frame":
                cv2.setTrackbarPos(PROGRESS, WINDOW, min(index + 1, len(data) - 1))
            elif command in ("previous_file", "next_file"):
                step = -1 if command == "previous_file" else 1
                file_index, data = load_nonempty(paths, file_index + step, step)
                set_progress(0, len(data))
                shown = None
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
