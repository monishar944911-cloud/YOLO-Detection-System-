// @ts-nocheck

"use strict";

/*
    DAWNVision
    Live AI Camera Controller
*/


document.addEventListener("DOMContentLoaded", function () {

    // ==============================
    // DOM ELEMENTS
    // ==============================

    var camera = document.getElementById("camera");
    var canvas = document.getElementById("canvas");

    var startButton = document.getElementById("startButton");
    var stopButton = document.getElementById("stopButton");
    var snapshotButton = document.getElementById("snapshotButton");

    var aiStatus = document.getElementById("aiStatus");

    var alertBox = document.getElementById("alertBox");
    var alertText = document.getElementById("alertText");

    var cameraResult = document.getElementById("cameraResult");

    var detectionInfo = document.getElementById("detectionInfo");

    var fpsElement = document.getElementById("fps");
    var totalObjectsElement = document.getElementById("totalObjects");
    var confidenceElement = document.getElementById("confidence");
    var alertCountElement = document.getElementById("alertCount");


    // ==============================
    // CAMERA VARIABLES
    // ==============================

    var cameraStream = null;

    var detectionRunning = false;

    var latestDetectionImage = null;

    var frameCount = 0;

    var lastFPSUpdate = Date.now();

    var totalAlerts = 0;


    // ==============================
    // SAFETY CHECK
    // ==============================

    if (
        !camera ||
        !canvas ||
        !startButton ||
        !stopButton ||
        !snapshotButton
    ) {

        console.error(
            "DAWNVision: Required camera elements were not found."
        );

        return;
    }


    // ==============================
    // BUTTON EVENTS
    // ==============================

    startButton.addEventListener(
        "click",
        startCamera
    );

    stopButton.addEventListener(
        "click",
        stopCamera
    );

    snapshotButton.addEventListener(
        "click",
        saveSnapshot
    );


    // ==============================
    // START CAMERA
    // ==============================

    async function startCamera() {

        try {

            if (
                !navigator.mediaDevices ||
                !navigator.mediaDevices.getUserMedia
            ) {

                alert(
                    "Camera access is not supported by this browser."
                );

                return;
            }


            // Request camera permission

            cameraStream =
                await navigator.mediaDevices.getUserMedia({

                    video: {
                        width: {
                            ideal: 640
                        },

                        height: {
                            ideal: 480
                        }
                    },

                    audio: false

                });


            // Connect camera to video element

            camera.srcObject = cameraStream;


            // Start camera

            detectionRunning = true;


            // Update buttons

            startButton.disabled = true;

            stopButton.disabled = false;

            snapshotButton.disabled = false;


            // Update status

            aiStatus.textContent =
                "🟢 AI Camera Running";

            aiStatus.style.background =
                "#d1e7dd";


            // Reset FPS

            frameCount = 0;

            lastFPSUpdate = Date.now();


            // Start AI detection

            detectFrame();

        }

        catch (error) {

            console.error(
                "DAWNVision Camera Error:",
                error
            );


            alert(
                "Camera access failed: " +
                error.message
            );

        }

    }


    // ==============================
    // STOP CAMERA
    // ==============================

    function stopCamera() {

        detectionRunning = false;


        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(function (track) {

                    track.stop();

                });

            cameraStream = null;

        }


        camera.srcObject = null;


        // Reset buttons

        startButton.disabled = false;

        stopButton.disabled = true;

        snapshotButton.disabled = true;


        // Reset status

        aiStatus.textContent =
            "⚪ AI Camera Stopped";

        aiStatus.style.background =
            "#e9ecef";


        // Hide alert

        hideAlert();

    }


    // ==============================
    // DETECT CAMERA FRAME
    // ==============================

    async function detectFrame() {

        if (!detectionRunning) {

            return;

        }


        // Wait until camera has video

        if (
            camera.videoWidth === 0 ||
            camera.videoHeight === 0
        ) {

            setTimeout(
                detectFrame,
                500
            );

            return;

        }


        // Set canvas size

        canvas.width =
            camera.videoWidth;

        canvas.height =
            camera.videoHeight;


        // Get canvas context

        var context =
            canvas.getContext("2d");


        if (!context) {

            console.error(
                "DAWNVision: Canvas context unavailable."
            );

            return;

        }


        // Draw camera frame

        context.drawImage(
            camera,
            0,
            0,
            canvas.width,
            canvas.height
        );


        // Convert image to Base64

        var image =
            canvas.toDataURL(
                "image/jpeg",
                0.65
            );


        try {

            // Send image to Flask

            var response =
                await fetch(
                    "/camera_detect",
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            image: image
                        })

                    }
                );


            // Check HTTP response

            if (!response.ok) {

                throw new Error(
                    "Server returned HTTP " +
                    response.status
                );

            }


            // Convert response to JSON

            var data =
                await response.json();


            // ==============================
            // PROCESS SUCCESS
            // ==============================

            if (data.success) {

                // Store latest result

                latestDetectionImage =
                    data.image;


                // Display detection image

                cameraResult.src =
                    data.image;

                cameraResult.style.display =
                    "block";


                // Total objects

                totalObjectsElement.textContent =
                    data.total || 0;


                // Confidence

                confidenceElement.textContent =
                    (data.confidence || 0) +
                    "%";


                // Object list

                updateDetectionInfo(
                    data.objects || {}
                );


                // Alerts

                if (
                    data.alert === true &&
                    data.alerts &&
                    data.alerts.length > 0
                ) {

                    showAlerts(
                        data.alerts
                    );

                }

                else {

                    hideAlert();

                }


                // FPS frame count

                frameCount++;


                var now =
                    Date.now();


                if (
                    now - lastFPSUpdate >=
                    1000
                ) {

                    fpsElement.textContent =
                        frameCount;

                    frameCount = 0;

                    lastFPSUpdate =
                        now;

                }

            }

        }

        catch (error) {

            console.error(
                "DAWNVision Detection Error:",
                error
            );

        }


        // Continue detection

        if (detectionRunning) {

            setTimeout(
                detectFrame,
                300
            );

        }

    }


    // ==============================
    // SHOW ALERTS
    // ==============================

    function showAlerts(alerts) {

        var messages = [];


        alerts.forEach(function (item) {

            messages.push(

                item.object +
                " detected (" +
                item.count +
                ") — " +
                item.confidence +
                "% confidence"

            );

        });


        alertText.textContent =
            messages.join(" | ");


        alertBox.style.display =
            "block";


        totalAlerts +=
            alerts.length;


        alertCountElement.textContent =
            totalAlerts;

    }


    // ==============================
    // HIDE ALERT
    // ==============================

    function hideAlert() {

        alertBox.style.display =
            "none";

    }


    // ==============================
    // UPDATE OBJECT INFORMATION
    // ==============================

    function updateDetectionInfo(objects) {

        var entries =
            Object.entries(objects);


        // No objects

        if (entries.length === 0) {

            detectionInfo.textContent =
                "No objects detected.";

            return;

        }


        // Clear previous information

        detectionInfo.innerHTML =
            "";


        // Add objects

        entries.forEach(
            function (entry) {

                var row =
                    document.createElement(
                        "p"
                    );


                var name =
                    document.createElement(
                        "strong"
                    );


                name.textContent =
                    entry[0];


                row.appendChild(
                    name
                );


                row.appendChild(
                    document.createTextNode(
                        ": " +
                        entry[1]
                    )
                );


                detectionInfo.appendChild(
                    row
                );

            }
        );

    }


    // ==============================
    // SAVE SNAPSHOT
    // ==============================

    async function saveSnapshot() {

        if (!latestDetectionImage) {

            alert(
                "Start the camera first."
            );

            return;

        }


        try {

            var response =
                await fetch(
                    "/save_snapshot",
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            image:
                                latestDetectionImage
                        })

                    }
                );


            if (!response.ok) {

                throw new Error(
                    "Server returned HTTP " +
                    response.status
                );

            }


            var data =
                await response.json();


            if (data.success) {

                alert(
                    "📸 Snapshot saved successfully!"
                );

            }

            else {

                alert(
                    data.error ||
                    "Snapshot could not be saved."
                );

            }

        }

        catch (error) {

            console.error(
                "DAWNVision Snapshot Error:",
                error
            );


            alert(
                "Snapshot error: " +
                error.message
            );

        }

    }

});