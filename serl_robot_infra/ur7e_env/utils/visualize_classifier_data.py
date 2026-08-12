"""Browse classifier pickle files and delete mislabeled transitions."""

import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np


WINDOW = "Classifier data"
PROGRESS = "Frame"
BUTTONS = (
    ("previous_file", "< File"),
    ("delete", "Delete"),
    ("next_file", "File >"),
)


def load_data(path):
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a transition list")
    return data


def save_data(path, data):
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as file:
            pickle.dump(data, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_transition(transition, path, index, total):
    observations = transition.get("observations", {})
    panels = []
    for name, value in observations.items():
        image = np.asarray(value)
        if image.ndim == 4:
            image = image[-1]
        if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
            continue
        if image.dtype != np.uint8:
            scale = 255 if image.max() <= 1 else 1
            image = np.clip(image * scale, 0, 255).astype(np.uint8)
        conversion = {
            1: cv2.COLOR_GRAY2BGR,
            3: cv2.COLOR_RGB2BGR,
            4: cv2.COLOR_RGBA2BGR,
        }[image.shape[-1]]
        panel = cv2.cvtColor(image, conversion)
        panel = cv2.resize(panel, (320, 320), interpolation=cv2.INTER_NEAREST)
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

    if not panels:
        raise ValueError(f"transition {index + 1} in {path} contains no images")

    frame = np.concatenate(panels, axis=1)
    frame = cv2.copyMakeBorder(frame, 48, 56, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(
        frame,
        f"{path.name}  {index + 1}/{total}",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    for action, label, (x1, y1, x2, y2) in button_rects(
        frame.shape[1], frame.shape[0]
    ):
        color = (70, 70, 180) if action == "delete" else (70, 70, 70)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.putText(
            frame,
            label,
            (x1 + (x2 - x1 - text_size[0]) // 2, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
    return frame


def button_rects(width, height):
    gap = 8
    button_width = min(180, (width - gap * 4) // 3)
    start = (width - button_width * 3 - gap * 2) // 2
    buttons = []
    for i, (action, label) in enumerate(BUTTONS):
        x1 = start + i * (button_width + gap)
        buttons.append((action, label, (x1, height - 48, x1 + button_width, height - 8)))
    return buttons


def on_mouse(event, x, y, _flags, state):
    if event != cv2.EVENT_LBUTTONUP:
        return
    for action, _label, (x1, y1, x2, y2) in state["buttons"]:
        if x1 <= x <= x2 and y1 <= y <= y2:
            state["command"] = action
            return


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


def set_progress(index, total):
    cv2.setTrackbarMax(PROGRESS, WINDOW, total - 1)
    cv2.setTrackbarPos(PROGRESS, WINDOW, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["classifier_data"],
        help="pickle files or directories",
    )
    args = parser.parse_args()

    paths = find_files(args.paths)
    file_index, data = load_nonempty(paths, 0)
    if data is None:
        raise ValueError("all classifier data files are empty")

    index = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar(PROGRESS, WINDOW, 0, len(data) - 1, lambda _: None)
    mouse = {"buttons": [], "command": None}
    cv2.setMouseCallback(WINDOW, on_mouse, mouse)
    print("Use the progress bar and window buttons; q/Esc quits")
    shown_index = None
    try:
        while True:
            path = paths[file_index]
            index = min(cv2.getTrackbarPos(PROGRESS, WINDOW), len(data) - 1)
            if index != shown_index:
                frame = render_transition(data[index], path, index, len(data))
                mouse["buttons"] = button_rects(frame.shape[1], frame.shape[0])
                cv2.imshow(WINDOW, frame)
                shown_index = index

            key = cv2.waitKey(30) & 0xFF
            command = mouse["command"]
            mouse["command"] = None
            if key == ord("d"):
                command = "delete"
            elif key == ord("["):
                command = "previous_file"
            elif key == ord("]"):
                command = "next_file"
            window_closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            if key in (ord("q"), 27) or window_closed:
                break
            if command == "delete":
                data.pop(index)
                save_data(path, data)
                print(
                    f"deleted transition {index + 1} from {path} "
                    f"({len(data)} remaining)"
                )
                if not data:
                    result = load_nonempty(paths, file_index + 1)
                    if result[1] is None:
                        break
                    file_index, data = result
                    index = 0
                else:
                    index = min(index, len(data) - 1)
                set_progress(index, len(data))
                shown_index = None
            elif key == ord("n"):
                cv2.setTrackbarPos(PROGRESS, WINDOW, min(index + 1, len(data) - 1))
            elif key == ord("p"):
                cv2.setTrackbarPos(PROGRESS, WINDOW, max(index - 1, 0))
            elif command in ("previous_file", "next_file"):
                step = -1 if command == "previous_file" else 1
                file_index, data = load_nonempty(
                    paths,
                    file_index + step,
                    step,
                )
                index = 0
                set_progress(index, len(data))
                shown_index = None
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
