import webbrowser
from threading import Timer
from flask import Flask, render_template, request
import os

from orchestrator.agri_orchestrator import AgriOrchestrator
from soil_model.image_validator import validate_image
from soil_model.gps_extractor import extract_gps_data

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

system = AgriOrchestrator()


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        file  = request.files["soil_image"]
        crop  = request.form["crop"]
        water = request.form["water"]

        if file:

            path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )
            file.save(path)

            # URL to display uploaded image in the result page
            image_path = "/" + path.replace("\\", "/")

            # --------------------------------------------------
            # IMAGE VALIDATION PIPELINE
            # Runs 5 checks before any agents are called:
            #   Stage 1 -- File Format Check
            #   Stage 2 -- Image Quality / Resolution Check
            #   Stage 3 -- Blur Detection (Variance of Laplacian)
            #   Stage 4 -- Soil vs Non-Soil Detectors (HSV, Hough, ImageNet, Custom similarity)
            #   Stage 5 -- Confidence Gate (>= 75%)
            # --------------------------------------------------
            validation = validate_image(path)

            if not validation.is_valid:
                # Stop the pipeline immediately.
                # Render the result page with only the error.
                return render_template(
                    "result.html",
                    image_path=image_path,
                    validation_passed=False,
                    error_message=validation.error_message,
                    validation_report=validation.report
                )

            # --------------------------------------------------
            # VALIDATION PASSED -- Continue existing pipeline
            # (unchanged from original system)
            # --------------------------------------------------

            cnn_soil    = validation.cnn_soil
            system_soil = validation.mapped_soil
            confidence  = validation.confidence

            # GPS Extraction
            gps_data  = extract_gps_data(path)
            location  = gps_data["location"]
            latitude  = gps_data["latitude"]
            longitude = gps_data["longitude"]

            # Crop Recommendation (Multi-Agent Orchestrator)
            result = system.run(
                location,
                crop,
                system_soil,
                water
            )

            return render_template(
                "result.html",
                image_path=image_path,
                cnn_soil=cnn_soil,
                mapped_soil=system_soil,
                confidence=round(confidence, 2),
                location=location,
                latitude=latitude,
                longitude=longitude,
                result=result,
                validation_passed=True,
                validation_report=validation.report
            )

    return render_template("index.html")


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001")


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True, port=5001)