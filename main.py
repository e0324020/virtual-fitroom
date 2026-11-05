"""Virtual Fitting Room - Main Application.

Real-time virtual try-on using MediaPipe pose estimation, OpenCV,
and cvzone. Detects 33 skeletal landmarks, overlays garments on the
upper torso with perspective warp correction, recommends a retail size,
and supports switching garments via keyboard and an on-screen dropdown.

Controls:
  q / Esc  : quit
  G        : toggle the garment dropdown
  1-9      : select garment directly
  [ ]      : previous / next garment
  Up/Down  : scroll the dropdown (when visible)
  M        : toggle perspective warp / rectangle overlay mode
  B        : toggle body-silhouette clipping
  P        : toggle pose skeleton
  W / S    : move garment up / down
  A / D    : move garment left / right
  X / C    : shrink / grow garment
  + / =    : expand garment (larger step)
  - / _    : contract garment (larger step)
  R        : reset garment fit
  K        : save a debug snapshot (frame + mask + info)
  Mouse    : left-click a garment in the dropdown
"""

import argparse
import os
import sys
import traceback

import cv2
import numpy as np

import config
from garment_manager import GarmentManager
from garment_menu import GarmentMenu
from pose_estimator import PoseEstimator
from size_recommender import SizeRecommender

# Base of the project (not the current working directory), so the app runs
# correctly no matter where the editor/Explorer launches it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MENU_HOTKEYS = "123456789"


def fail(message):
    """Print a startup error and keep the window open so it is visible."""
    print(f"[ERROR] {message}")
    try:
        input("Press Enter to exit...")
    except EOFError:
        pass
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Virtual Fitting Room using Deep Learning-Driven Pose Estimation"
    )
    parser.add_argument(
        "--garments-dir",
        type=str,
        default=os.path.join(BASE_DIR, "garments"),
        help="Directory containing garment PNGs (scanned automatically).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task"),
        help="Path to the MediaPipe PoseLandmarker .task model.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help="Webcam device index.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["perspective", "rectangle"],
        default="perspective",
        help="Garment overlay rendering mode.",
    )
    parser.add_argument(
        "--draw-pose",
        action="store_true",
        help="Draw the pose skeleton overlay (default: hidden).",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable body-silhouette clipping of garments.",
    )
    return parser.parse_args()


def draw_status_bar(img, pose_estimator, size, metrics, garment_name, overlay_mode,
                    fit_text="", warning=""):
    """Render FPS, garment name, size recommendation and metrics."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 55), config.COLORS["overlay"], cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    fps_text = f"FPS: {pose_estimator.fps:.1f}"

    if size == "N/A":
        size_text = "No person detected"
    else:
        size_text = f"Size: {size}"
        metric_text = (
            f"Ratio: {metrics['shoulder_ratio']:.2f}  "
            f"S: {metrics['shoulder_px']}px  W: {metrics['waist_px']}px"
        )

    cv2.putText(
        img, fps_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
        config.COLORS["text"], 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, size_text, (170, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
        config.COLORS["text"], 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, f"[{overlay_mode}] {os.path.basename(garment_name)}",
        (330, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        config.COLORS["text"], 1, cv2.LINE_AA,
    )
    if size != "N/A":
        cv2.putText(
            img, metric_text, (img.shape[1] - 460, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, config.COLORS["text"], 1, cv2.LINE_AA,
        )
    if fit_text:
        cv2.putText(
            img, fit_text, (15, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
    if warning:
        (tw, th), _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
        cx = (img.shape[1] - tw) // 2
        cy = img.shape[0] // 2
        cv2.putText(img, warning, (cx + 2, cy + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, warning, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 3, cv2.LINE_AA)
    return img


def handle_key(key, manager, menu, arg_mode, on_mode_toggle, on_mask_toggle, on_pose_toggle):
    """Process a keyboard event. Returns True when the app should quit."""
    if key in (ord("q"), 27):           # q / Esc
        return True

    if key == ord("g"):                 # toggle dropdown
        menu.visible = not menu.visible

    elif key == ord("b"):               # toggle body-mask clipping
        on_mask_toggle()

    elif key == ord("p"):               # toggle pose skeleton
        on_pose_toggle()

    elif key in MENU_HOTKEYS:           # 1-9 direct selection
        menu.select_by_key(int(chr(key)))
        manager.switch(int(chr(key)) - 1)

    elif key == ord("["):               # previous garment
        manager.cycle(-1)
        menu.selected = manager.current

    elif key == ord("]"):               # next garment
        manager.cycle(1)
        menu.selected = manager.current

    elif key == 82:                     # Up arrow
        menu.scroll_up()

    elif key == 84:                     # Down arrow
        menu.scroll_down()

    elif key == ord("w"):               # move garment up
        manager.adjust(dy=-6)

    elif key == ord("s"):               # move garment down
        manager.adjust(dy=6)

    elif key == ord("a"):               # move garment left
        manager.adjust(dx=-6)

    elif key == ord("d"):               # move garment right
        manager.adjust(dx=6)

    elif key == ord("x"):               # shrink garment
        manager.adjust(scale_factor=0.90)

    elif key == ord("c"):               # grow garment
        manager.adjust(scale_factor=1.10)

    elif key in (ord("+"), ord("=")):   # expand garment (bigger step)
        manager.adjust(scale_factor=1.20)

    elif key in (ord("-"), ord("_")):   # contract garment (bigger step)
        manager.adjust(scale_factor=0.83)

    elif key == ord("r"):               # reset manual fit
        manager.reset_active_fit()

    elif key == ord("m"):               # toggle overlay mode
        new_mode = "rectangle" if arg_mode == "perspective" else "perspective"
        on_mode_toggle(new_mode)

    return False


def save_debug_snapshot(raw_img, out_img, lm_list, seg_mask, garment_name):
    """Dump the raw + annotated frames, the body mask and diagnostics."""
    cv2.imwrite(os.path.join(BASE_DIR, "debug_raw.png"), raw_img)
    cv2.imwrite(os.path.join(BASE_DIR, "debug_frame.png"), out_img)
    info = []
    info.append(f"detected={len(lm_list) == 33}")
    info.append(f"garment={os.path.basename(garment_name)}")
    info.append(f"frame={out_img.shape[1]}x{out_img.shape[0]}")
    if len(lm_list) == 33:
        sl, sr = np.array(lm_list[11], float), np.array(lm_list[12], float)
        hl, hr = np.array(lm_list[23], float), np.array(lm_list[24], float)
        shoulder_w = float(np.linalg.norm(sr - sl))
        smid = (sl + sr) / 2.0
        hmid = (hl + hr) / 2.0
        torso = abs(float(hmid[1] - smid[1]))
        info.append(f"shoulder_w={shoulder_w:.1f} torso_len={torso:.1f}")
        info.append(f"shoulder_mid=({smid[0]:.0f},{smid[1]:.0f}) hip_mid=({hmid[0]:.0f},{hmid[1]:.0f})")
        info.append(f"shoulderL={lm_list[11]} shoulderR={lm_list[12]} hipL={lm_list[23]} hipR={lm_list[24]}")
        info.append(f"nose={lm_list[0]}")
    if seg_mask is None:
        info.append("seg_mask=None  <-- body masking is NOT available")
    else:
        m = np.asarray(seg_mask, np.float32)
        info.append(f"seg_mask shape={m.shape} min={m.min():.3f} max={m.max():.3f} "
                    f"mean={m.mean():.3f} coverage(>0.5)={(m > 0.5).mean()*100:.1f}%")
        cv2.imwrite(os.path.join(BASE_DIR, "debug_mask.png"),
                    (np.clip(m, 0, 1) * 255).astype(np.uint8))
    text = "\n".join(info)
    with open(os.path.join(BASE_DIR, "debug_info.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("[CAPTURE]\n" + text)


def main():
    args = parse_args()

    if not os.path.isdir(args.garments_dir):
        fail(f"Garments directory not found: {args.garments_dir}")

    if not os.path.exists(args.model):
        fail(f"Pose model not found: {args.model}")

    try:
        pose_estimator = PoseEstimator(args.model)
    except Exception as exc:
        fail(f"Could not load pose model {args.model}: {exc}")

    try:
        manager = GarmentManager(args.garments_dir, args.mode)
    except Exception as exc:
        pose_estimator.close()
        fail(f"Could not load garments: {exc}")

    size_recommender = SizeRecommender()

    if manager.count == 0:
        pose_estimator.close()
        fail(f"No garment images found in '{args.garments_dir}'.")

    menu = GarmentMenu(manager.names)

    def set_mode(mode):
        manager.mode = mode
        for overlay in manager._overlays.values():
            overlay.mode = mode

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        cap.release()
        pose_estimator.close()
        fail("Could not open webcam.")

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and menu.select_at(x, y):
            manager.switch(menu.selected)

    print("[INFO] Virtual Fitting Room started.")
    print(f"[INFO] Found {manager.count} garment(s). Press 'q' to quit.")

    try:
        window_ready = False
        use_mask = not args.no_mask
        draw_pose = args.draw_pose
        _error_count = 0
        _frame_i = 0

        def toggle_mask():
            nonlocal use_mask
            use_mask = not use_mask

        def toggle_pose():
            nonlocal draw_pose
            draw_pose = not draw_pose

        while True:
            try:
                success, img = cap.read()
                if not success:
                    print("[ERROR] Failed to read frame from webcam.")
                    break

                img = cv2.flip(img, 1)
                img = pose_estimator.find_pose(img, draw=draw_pose)

                lm_list = pose_estimator.get_position(img)

                warning = ""
                if len(lm_list) == 33:
                    fh, fw = img.shape[:2]
                    nose_y = lm_list[0][1]
                    hip_y = (lm_list[23][1] + lm_list[24][1]) / 2.0
                    sw = float(np.hypot(lm_list[11][0] - lm_list[12][0],
                                        lm_list[11][1] - lm_list[12][1]))
                    if nose_y < 0 or hip_y > fh or sw > 0.45 * fw:
                        warning = "STEP BACK - show head to waist in frame"

                if len(lm_list) == 33:
                    raw_img = img.copy()
                    mask = pose_estimator.seg_mask if use_mask else None
                    img = manager.overlay(img, lm_list, mask)
                    size, metrics = size_recommender.recommend(lm_list)
                    _frame_i += 1
                    if _frame_i % 60 == 0:
                        save_debug_snapshot(raw_img, img, lm_list,
                                            pose_estimator.seg_mask, manager.current_name)
                else:
                    size, metrics = "N/A", {}

                img = draw_status_bar(
                    img, pose_estimator, size, metrics,
                    manager.current_name, manager.mode,
                    manager.active_fit_text(), warning,
                )
                img = menu.draw(img)

                cv2.imshow("Virtual Fitting Room", img)
                if not window_ready:
                    cv2.setMouseCallback("Virtual Fitting Room", on_mouse)
                    window_ready = True

                key = cv2.waitKey(1) & 0xFF
                if key != 255 and key != 0:
                    if handle_key(key, manager, menu, args.mode, set_mode,
                                  toggle_mask, toggle_pose):
                        break
                    if key == ord("k"):
                        save_debug_snapshot(img, img, lm_list,
                                            pose_estimator.seg_mask, manager.current_name)
            except Exception:
                # Log and keep the window alive instead of crashing shut.
                _error_count += 1
                tb = traceback.format_exc()
                print(tb)
                if _error_count <= 20:
                    with open(os.path.join(BASE_DIR, "error.log"), "a", encoding="utf-8") as fh:
                        fh.write(tb + "\n")
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_estimator.close()
        print("[INFO] Application closed.")


if __name__ == "__main__":
    main()