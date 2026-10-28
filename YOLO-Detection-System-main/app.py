from flask import Flask, render_template, jsonify, request
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import base64
import io
import threading
import webbrowser
import time


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

DAWN_IMAGES = BASE_DIR / "dawn" / "images"

RESULT_FOLDER = BASE_DIR / "static" / "results"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_PATH = BASE_DIR / "yolov8m.pt"

print("=" * 75)
print("DAWNVision - YOLO OBJECT DETECTION")
print("=" * 75)

print(f"Project Folder : {BASE_DIR}")
print(f"DAWN Images    : {DAWN_IMAGES}")
print(f"YOLO Model     : {MODEL_PATH}")
print("=" * 75)


# ============================================================
# GLOBAL MODELS
# ============================================================

dataset_model = None
camera_model = None

MODEL_LOADED = False
MODEL_LOADING = True
MODEL_ERROR = None

MODEL_CLASSES = {}


# ============================================================
# DATASET STATE
# ============================================================

dataset_state = {
    "success": True,
    "running": False,
    "complete": False,
    "error": None,

    "images_processed": 0,
    "total_images": 0,

    "total_objects": 0,

    "average_confidence": 0.0,

    "results": [],

    "time": 0,

    "model": "Not Loaded",

    "classes": {}
}


# ============================================================
# LIVE CAMERA STATE
# ============================================================

live_state = {
    "active": False,

    "total_detections": 0,

    "average_confidence": 0.0,

    "objects": {},

    "detections": [],

    "suggestion": "CONTINUE",

    "suggestion_reason":
        "No immediate object detected.",

    "last_update": None,

    "saved_result_image": None
}


# ============================================================
# LOCKS
# ============================================================

dataset_lock = threading.Lock()
camera_lock = threading.Lock()


# ============================================================
# GET DAWN IMAGES
# ============================================================

def get_dawn_images():

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]

    images = []

    if not DAWN_IMAGES.exists():
        return images

    for extension in extensions:
        images.extend(
            DAWN_IMAGES.rglob(extension)
        )

    return sorted(list(set(images)))


# ============================================================
# GET CLASS NAME
# ============================================================

def get_class_name(model, class_id):

    try:

        names = model.names

        if isinstance(names, dict):

            return names.get(
                int(class_id),
                f"class_{class_id}"
            )

        if isinstance(names, list):

            return names[int(class_id)]

    except Exception:
        pass

    return f"class_{class_id}"


# ============================================================
# DRIVING SUGGESTION
# ============================================================
# DEMONSTRATION / SIMULATION ONLY.
# This does NOT control a real vehicle.
# ============================================================

def get_driving_suggestion(
    detections,
    image_width,
    image_height
):

    if not detections:

        return (
            "CONTINUE",
            "No detected obstacle in the current frame."
        )

    dangerous_objects = []

    for detection in detections:

        class_name = str(
            detection["class"]
        ).lower()

        confidence = float(
            detection["confidence"]
        )

        x1, y1, x2, y2 = detection["box"]

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        box_width = max(
            1,
            x2 - x1
        )

        box_height = max(
            1,
            y2 - y1
        )

        # Approximate visual size
        area_ratio = (
            box_width * box_height
        ) / max(
            1,
            image_width * image_height
        )

        obstacle_classes = [
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle"
        ]

        if class_name in obstacle_classes:

            dangerous_objects.append({

                "class":
                    class_name,

                "confidence":
                    confidence,

                "center_x":
                    center_x,

                "center_y":
                    center_y,

                "area_ratio":
                    area_ratio
            })

    if not dangerous_objects:

        return (
            "CONTINUE",
            "No major road obstacle detected."
        )

    # Largest / closest-looking object
    main_object = max(
        dangerous_objects,
        key=lambda x: (
            x["area_ratio"],
            x["confidence"]
        )
    )

    center_x = main_object["center_x"]
    center_y = main_object["center_y"]

    confidence = main_object["confidence"]

    class_name = main_object["class"]

    area_ratio = main_object["area_ratio"]

    # ========================================================
    # CENTER DANGER
    # ========================================================

    if (
        area_ratio > 0.18
        and confidence >= 55
    ):

        return (
            "HOLD BRAKE",
            f"Large {class_name} detected "
            f"close to the center of the frame."
        )

    # ========================================================
    # VERY LARGE OBJECT
    # ========================================================

    if (
        area_ratio > 0.28
        and confidence >= 45
    ):

        return (
            "HOLD BRAKE",
            f"{class_name} occupies a large "
            f"portion of the camera view."
        )

    # ========================================================
    # LEFT SIDE OBJECT
    # ========================================================

    if image_width > 0:

        relative_position = (
            center_x / image_width
        )

        if relative_position < 0.38:

            return (
                "TAKE RIGHT",
                f"{class_name} detected toward "
                f"the left side of the frame."
            )

        # ====================================================
        # RIGHT SIDE OBJECT
        # ====================================================

        if relative_position > 0.62:

            return (
                "TAKE LEFT",
                f"{class_name} detected toward "
                f"the right side of the frame."
            )

    # ========================================================
    # LOWER / NEAR OBJECT
    # ========================================================

    if image_height > 0:

        vertical_position = (
            center_y / image_height
        )

        if (
            vertical_position > 0.72
            and confidence >= 50
        ):

            return (
                "SLOW DOWN",
                f"{class_name} detected in "
                f"the lower forward view."
            )

    # ========================================================
    # DEFAULT
    # ========================================================

    return (
        "SLOW DOWN",
        f"{class_name} detected ahead. "
        f"Use caution."
    )


# ============================================================
# DRAW SUGGESTION ON IMAGE
# ============================================================

def add_suggestion_to_image(
    image,
    suggestion,
    reason,
    confidence
):

    result_image = image.copy()

    draw = ImageDraw.Draw(
        result_image
    )

    width, height = result_image.size

    # --------------------------------------------------------
    # Top information panel
    # --------------------------------------------------------

    panel_height = 105

    draw.rectangle(
        [
            0,
            0,
            width,
            panel_height
        ],
        fill=(0, 0, 0)
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    try:

        font_large = ImageFont.truetype(
            "arial.ttf",
            28
        )

        font_small = ImageFont.truetype(
            "arial.ttf",
            17
        )

    except Exception:

        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text(
        (
            15,
            10
        ),
        f"ACTION: {suggestion}",
        fill=(255, 255, 255),
        font=font_large
    )

    draw.text(
        (
            15,
            50
        ),
        f"Average Confidence: "
        f"{confidence:.2f}%",
        fill=(255, 255, 255),
        font=font_small
    )

    # Reason
    draw.text(
        (
            15,
            75
        ),
        reason[:100],
        fill=(255, 255, 255),
        font=font_small
    )

    return result_image


# ============================================================
# LOAD YOLO MODELS
# ============================================================

def load_models_background():

    global dataset_model
    global camera_model

    global MODEL_LOADED
    global MODEL_LOADING
    global MODEL_ERROR

    global MODEL_CLASSES

    try:

        print("\nLoading YOLOv8m...")
        print("Please wait...\n")

        dataset_model = YOLO(
            str(MODEL_PATH)
        )

        camera_model = YOLO(
            str(MODEL_PATH)
        )

        MODEL_CLASSES = dataset_model.names

        MODEL_LOADED = True
        MODEL_LOADING = False
        MODEL_ERROR = None

        dataset_state["model"] = (
            "YOLOv8m COCO"
        )

        dataset_state["classes"] = (
            MODEL_CLASSES
        )

        print("=" * 75)
        print("YOLO MODEL LOADED SUCCESSFULLY")
        print("=" * 75)

        print("\nMODEL CLASSES:")

        if isinstance(
            MODEL_CLASSES,
            dict
        ):

            for class_id, class_name in (
                MODEL_CLASSES.items()
            ):

                print(
                    f"{class_id} -> {class_name}"
                )

        print("=" * 75)

        # Start DAWN dataset
        dataset_thread = threading.Thread(
            target=run_dawn_dataset,
            daemon=True
        )

        dataset_thread.start()

    except Exception as e:

        MODEL_LOADED = False
        MODEL_LOADING = False

        MODEL_ERROR = str(e)

        dataset_state["error"] = str(e)

        print("\n" + "=" * 75)
        print("YOLO MODEL ERROR")
        print("=" * 75)
        print(str(e))
        print("=" * 75)


# ============================================================
# RUN DAWN DATASET
# ============================================================

def run_dawn_dataset():

    with dataset_lock:

        dataset_state["running"] = True
        dataset_state["complete"] = False

        dataset_state["error"] = None

        dataset_state["results"] = []

        dataset_state["images_processed"] = 0
        dataset_state["total_objects"] = 0

        dataset_state[
            "average_confidence"
        ] = 0.0

    start_time = time.time()

    total_objects = 0

    all_confidences = []

    try:

        if (
            not MODEL_LOADED
            or dataset_model is None
        ):

            raise RuntimeError(
                "YOLO model is not loaded."
            )

        images = get_dawn_images()

        with dataset_lock:

            dataset_state[
                "total_images"
            ] = len(images)

        print(
            f"\nDAWN DATASET: "
            f"{len(images)} images found."
        )

        if len(images) == 0:

            with dataset_lock:

                dataset_state["running"] = False
                dataset_state["complete"] = True

                dataset_state["error"] = (
                    f"No images found in "
                    f"{DAWN_IMAGES}"
                )

            return

        # ====================================================
        # PROCESS IMAGES
        # ====================================================

        for index, image_path in enumerate(
            images
        ):

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                results = dataset_model.predict(

                    source=image,

                    imgsz=640,

                    conf=0.25,

                    iou=0.45,

                    verbose=False

                )

                prediction = results[0]

                objects = {}

                image_confidences = []

                # ------------------------------------------------
                # DETECTIONS
                # ------------------------------------------------

                if prediction.boxes is not None:

                    for box in prediction.boxes:

                        class_id = int(
                            box.cls[0].item()
                        )

                        confidence = float(
                            box.conf[0].item()
                        )

                        class_name = (
                            get_class_name(
                                dataset_model,
                                class_id
                            )
                        )

                        if class_name not in objects:

                            objects[class_name] = 0

                        objects[class_name] += 1

                        image_confidences.append(
                            confidence
                        )

                        all_confidences.append(
                            confidence
                        )

                        total_objects += 1

                # ------------------------------------------------
                # IMAGE AVERAGE CONFIDENCE
                # ------------------------------------------------

                if image_confidences:

                    image_average_confidence = (

                        sum(
                            image_confidences
                        )
                        /
                        len(
                            image_confidences
                        )

                    ) * 100

                else:

                    image_average_confidence = 0.0

                # ------------------------------------------------
                # ANNOTATED IMAGE
                # ------------------------------------------------

                annotated = prediction.plot(

                    conf=True,

                    labels=True,

                    boxes=True

                )

                output_name = (
                    f"{index + 1}_"
                    f"{image_path.stem}"
                    f"_result.jpg"
                )

                output_path = (
                    RESULT_FOLDER
                    / output_name
                )

                Image.fromarray(
                    annotated[..., ::-1]
                ).save(
                    output_path
                )

                # ------------------------------------------------
                # DATASET RESULT
                # ------------------------------------------------

                result_data = {

                    "type":
                        "DAWN DATASET",

                    "image":
                        image_path.name,

                    "objects":
                        objects,

                    "total_objects":
                        sum(
                            objects.values()
                        ),

                    "average_confidence":
                        round(
                            image_average_confidence,
                            2
                        ),

                    "confidence":
                        round(
                            image_average_confidence,
                            2
                        ),

                    "suggestion":
                        "DATASET IMAGE",

                    "suggestion_reason":
                        "DAWN dataset detection.",

                    "result_image":
                        f"/static/results/"
                        f"{output_name}"

                }

                with dataset_lock:

                    dataset_state[
                        "results"
                    ].append(
                        result_data
                    )

                    dataset_state[
                        "images_processed"
                    ] = index + 1

                    dataset_state[
                        "total_objects"
                    ] = total_objects

                    if all_confidences:

                        overall_confidence = (

                            sum(
                                all_confidences
                            )
                            /
                            len(
                                all_confidences
                            )

                        ) * 100

                    else:

                        overall_confidence = 0.0

                    dataset_state[
                        "average_confidence"
                    ] = round(
                        overall_confidence,
                        2
                    )

                print(
                    f"[{index + 1}/"
                    f"{len(images)}] "
                    f"{image_path.name} | "
                    f"Objects: "
                    f"{sum(objects.values())} | "
                    f"Confidence: "
                    f"{image_average_confidence:.2f}%"
                )

            except Exception as image_error:

                print(
                    f"Error processing "
                    f"{image_path.name}: "
                    f"{image_error}"
                )

        elapsed = (
            time.time()
            - start_time
        )

        if all_confidences:

            final_average_confidence = (

                sum(
                    all_confidences
                )
                /
                len(
                    all_confidences
                )

            ) * 100

        else:

            final_average_confidence = 0.0

        with dataset_lock:

            dataset_state[
                "running"
            ] = False

            dataset_state[
                "complete"
            ] = True

            dataset_state[
                "time"
            ] = round(
                elapsed,
                2
            )

            dataset_state[
                "total_objects"
            ] = total_objects

            dataset_state[
                "average_confidence"
            ] = round(
                final_average_confidence,
                2
            )

        print("\n")
        print("=" * 75)
        print("DAWN DATASET PROCESSING COMPLETE")
        print("=" * 75)

        print(
            f"Images processed : "
            f"{len(images)}"
        )

        print(
            f"Objects detected : "
            f"{total_objects}"
        )

        print(
            f"Average confidence : "
            f"{final_average_confidence:.2f}%"
        )

        print(
            f"Processing time : "
            f"{elapsed:.2f} seconds"
        )

        print("=" * 75)

    except Exception as e:

        with dataset_lock:

            dataset_state[
                "running"
            ] = False

            dataset_state[
                "complete"
            ] = False

            dataset_state[
                "error"
            ] = str(e)

        print(
            "\nDAWN DATASET ERROR:",
            str(e)
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    with dataset_lock:

        data = dict(
            dataset_state
        )

    return render_template(
        "index.html",
        data=data,
        model_name="YOLOv8m COCO",
        model_loaded=MODEL_LOADED,
        model_error=MODEL_ERROR
    )


# ============================================================
# DAWN API
# ============================================================

@app.route("/api/dawn")
def api_dawn():

    with dataset_lock:

        data = dict(
            dataset_state
        )

    return jsonify(data)


# ============================================================
# LIVE STATUS
# ============================================================

@app.route("/api/live_status")
def api_live_status():

    with camera_lock:

        data = dict(
            live_state
        )

    return jsonify(data)


# ============================================================
# RUN DAWN AGAIN
# ============================================================

@app.route(
    "/run_dawn",
    methods=["POST"]
)
def run_dawn():

    if not MODEL_LOADED:

        return jsonify({

            "success": False,

            "error":
                "YOLO model is not loaded."

        }), 500

    with dataset_lock:

        if dataset_state["running"]:

            return jsonify({

                "success": False,

                "error":
                    "DAWN detection is "
                    "already running."

            })

        dataset_state["running"] = True

        dataset_state["complete"] = False

        dataset_state["error"] = None

        dataset_state["results"] = []

        dataset_state["images_processed"] = 0

        dataset_state["total_objects"] = 0

        dataset_state[
            "average_confidence"
        ] = 0.0

    thread = threading.Thread(
        target=run_dawn_dataset,
        daemon=True
    )

    thread.start()

    return jsonify({

        "success": True,

        "message":
            "DAWN detection started."

    })


# ============================================================
# LIVE CAMERA DETECTION
# ============================================================

@app.route(
    "/api/live_detect",
    methods=["POST"]
)
def live_detect():

    if (
        not MODEL_LOADED
        or camera_model is None
    ):

        return jsonify({

            "success": False,

            "error":
                "YOLO model is not loaded yet."

        }), 503

    try:

        data = request.get_json()

        if (
            not data
            or "image" not in data
        ):

            return jsonify({

                "success": False,

                "error":
                    "Camera image not received."

            }), 400

        image_data = data["image"]

        # --------------------------------------------------------
        # REMOVE BASE64 HEADER
        # --------------------------------------------------------

        if "," in image_data:

            image_data = (
                image_data.split(
                    ",",
                    1
                )[1]
            )

        # --------------------------------------------------------
        # DECODE
        # --------------------------------------------------------

        image_bytes = base64.b64decode(
            image_data
        )

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

        image_width, image_height = (
            image.size
        )

        start_time = time.time()

        # ========================================================
        # YOLO CAMERA PREDICTION
        # ========================================================

        with camera_lock:

            results = camera_model.predict(

                source=image,

                imgsz=640,

                conf=0.30,

                iou=0.45,

                max_det=100,

                verbose=False

            )

        prediction = results[0]

        objects = {}

        detections = []

        confidence_values = []

        # ========================================================
        # READ DETECTIONS
        # ========================================================

        if prediction.boxes is not None:

            for box in prediction.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = (
                    get_class_name(
                        camera_model,
                        class_id
                    )
                )

                if class_name not in objects:

                    objects[class_name] = 0

                objects[class_name] += 1

                confidence_values.append(
                    confidence
                )

                xyxy = box.xyxy[
                    0
                ].tolist()

                detections.append({

                    "class":
                        class_name,

                    "confidence":
                        round(
                            confidence * 100,
                            2
                        ),

                    "box": [
                        round(
                            x,
                            1
                        )
                        for x in xyxy
                    ]

                })

        # ========================================================
        # AVERAGE CAMERA CONFIDENCE
        # ========================================================

        if confidence_values:

            average_confidence = (

                sum(
                    confidence_values
                )
                /
                len(
                    confidence_values
                )

            ) * 100

        else:

            average_confidence = 0.0

        # ========================================================
        # GET NEXT-STEP SUGGESTION
        # ========================================================

        suggestion, suggestion_reason = (
            get_driving_suggestion(

                detections,

                image_width,

                image_height

            )
        )

        # ========================================================
        # YOLO ANNOTATED IMAGE
        # ========================================================

        annotated = prediction.plot(

            conf=True,

            labels=True,

            boxes=True

        )

        annotated_rgb = (
            annotated[..., ::-1]
        )

        detected_image = Image.fromarray(
            annotated_rgb
        )

        # ========================================================
        # ADD SUGGESTION TO IMAGE
        # ========================================================

        final_live_image = (
            add_suggestion_to_image(

                detected_image,

                suggestion,

                suggestion_reason,

                average_confidence

            )
        )

        # ========================================================
        # SAVE LIVE IMAGE INTO DAWN RESULTS
        # ========================================================

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        milliseconds = int(
            (time.time() % 1) * 1000
        )

        live_filename = (
            f"LIVE_"
            f"{timestamp}_"
            f"{milliseconds}.jpg"
        )

        live_output_path = (
            RESULT_FOLDER
            / live_filename
        )

        final_live_image.save(
            live_output_path,
            quality=90
        )

        live_result_url = (
            f"/static/results/"
            f"{live_filename}"
        )

        # ========================================================
        # ADD LIVE IMAGE TO DAWN RESULTS
        # ========================================================

        live_result_data = {

            "type":
                "LIVE CAMERA",

            "image":
                "Live Camera",

            "objects":
                objects,

            "total_objects":
                len(detections),

            "average_confidence":
                round(
                    average_confidence,
                    2
                ),

            "confidence":
                round(
                    average_confidence,
                    2
                ),

            "suggestion":
                suggestion,

            "suggestion_reason":
                suggestion_reason,

            "result_image":
                live_result_url,

            "timestamp":
                datetime_now()

        }

        with dataset_lock:

            # This is the important part:
            # LIVE detection is added to the
            # same DAWN results list.

            dataset_state[
                "results"
            ].append(
                live_result_data
            )

        # ========================================================
        # UPDATE LIVE STATE
        # ========================================================

        with camera_lock:

            live_state[
                "active"
            ] = True

            live_state[
                "total_detections"
            ] = len(
                detections
            )

            live_state[
                "average_confidence"
            ] = round(
                average_confidence,
                2
            )

            live_state[
                "objects"
            ] = objects

            live_state[
                "detections"
            ] = detections

            live_state[
                "suggestion"
            ] = suggestion

            live_state[
                "suggestion_reason"
            ] = suggestion_reason

            live_state[
                "last_update"
            ] = datetime_now()

            live_state[
                "saved_result_image"
            ] = live_result_url

        # ========================================================
        # CONVERT IMAGE TO BASE64
        # ========================================================

        buffer = io.BytesIO()

        final_live_image.save(

            buffer,

            format="JPEG",

            quality=85

        )

        encoded_image = (
            base64.b64encode(
                buffer.getvalue()
            ).decode(
                "utf-8"
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        # ========================================================
        # RETURN LIVE RESULT
        # ========================================================

        return jsonify({

            "success":
                True,

            "image":
                "data:image/jpeg;base64,"
                + encoded_image,

            "objects":
                objects,

            "detections":
                detections,

            "total_objects":
                len(
                    detections
                ),

            "average_confidence":
                round(
                    average_confidence,
                    2
                ),

            "confidence":
                round(
                    average_confidence,
                    2
                ),

            "suggestion":
                suggestion,

            "suggestion_reason":
                suggestion_reason,

            "saved_result_image":
                live_result_url,

            "time":
                round(
                    elapsed,
                    3
                ),

            "model":
                "YOLOv8m COCO"

        })

    except Exception as e:

        print(
            "\nLIVE CAMERA ERROR:",
            str(e)
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# DATETIME
# ============================================================

def datetime_now():

    return time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# MODEL STATUS
# ============================================================

@app.route("/api/model_status")
def model_status():

    return jsonify({

        "loaded":
            MODEL_LOADED,

        "loading":
            MODEL_LOADING,

        "error":
            MODEL_ERROR,

        "model":
            str(MODEL_PATH),

        "classes":
            MODEL_CLASSES

    })


# ============================================================
# OPEN BROWSER
# ============================================================

def open_browser():

    time.sleep(2)

    try:

        webbrowser.open(
            "http://127.0.0.1:5000/"
        )

    except Exception as e:

        print(
            "Could not open browser:",
            e
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "\nStarting DAWNVision...\n"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model_thread = threading.Thread(

        target=load_models_background,

        daemon=True

    )

    model_thread.start()

    # --------------------------------------------------------
    # OPEN BROWSER
    # --------------------------------------------------------

    browser_thread = threading.Thread(

        target=open_browser,

        daemon=True

    )

    browser_thread.start()

    print(
        "Server starting at:"
    )

    print(
        "http://127.0.0.1:5000/"
    )

    print(
        "http://localhost:5000/"
    )

    print("\n")

    # --------------------------------------------------------
    # FLASK SERVER
    # --------------------------------------------------------

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True,

        use_reloader=False

    )