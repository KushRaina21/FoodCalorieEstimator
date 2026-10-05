import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from fastai.vision.all import PILImage, load_learner
from PIL import Image, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "food_calorie_model.pkl"
METRICS_PATH = PROJECT_DIR / "models" / "calorie_metrics.json"

st.set_page_config(page_title="Food Calorie Estimator", page_icon="🍽️")


def load_model():
    if not MODEL_PATH.is_file() or not METRICS_PATH.is_file():
        raise FileNotFoundError(
            "The trained model or its metrics are missing. Run all cells in "
            "notebooks/train_calorie_model.ipynb to create both files."
        )

    with MODEL_PATH.open("rb") as model_file:
        file_header = model_file.read(128)
    if file_header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            "The model file is a Git LFS pointer, not the model binary. "
            "Download or commit the actual trained model file."
        )

    import plum._resolver

    if not getattr(plum._resolver.Resolver, "_food_calorie_legacy_compatible", False):
        class LegacyResolver(plum._resolver.Resolver):
            __slots__ = ("__dict__",)

            def __setstate__(self, state):
                state_parts = state if isinstance(state, tuple) else (state,)
                for part in state_parts:
                    if part is None:
                        continue
                    if not isinstance(part, dict):
                        raise TypeError(
                            "Unsupported serialized Resolver state: "
                            f"{type(part).__name__}"
                        )
                    for name, value in part.items():
                        setattr(self, name, value)

        LegacyResolver.__name__ = "Resolver"
        LegacyResolver.__qualname__ = "Resolver"
        LegacyResolver._food_calorie_legacy_compatible = True
        plum._resolver.Resolver = LegacyResolver

    learner = load_learner(MODEL_PATH, cpu=True)
    learner.model.eval()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    error_range = float(metrics["validation_absolute_error_p90_kcal"])
    calibration_slope = float(metrics["calibration_slope"])
    calibration_intercept = float(metrics["calibration_intercept_kcal"])
    if not np.isfinite(
        [error_range, calibration_slope, calibration_intercept]
    ).all() or error_range < 0:
        raise ValueError("The exported calorie metrics contain invalid values.")
    return learner, error_range, calibration_slope, calibration_intercept


@st.cache_resource
def get_model():
    return load_model()


def estimate_calories(image: Image.Image, model_data):
    model, error_range_kcal, calibration_slope, calibration_intercept = model_data
    image_tensor = model.dls.after_item(PILImage.create(image.convert("RGB")))
    image_tensor = model.dls.after_batch.fs[0](image_tensor)
    image_tensor = model.dls.after_batch.fs[-1](image_tensor)
    with torch.inference_mode():
        prediction = model.model(image_tensor)
    raw_prediction_kcal = float(prediction.reshape(-1)[0].item())
    predicted_kcal = calibration_slope * raw_prediction_kcal + calibration_intercept
    if not np.isfinite(predicted_kcal):
        raise ValueError("The model could not produce a valid calorie estimate.")
    predicted_kcal = max(0.0, predicted_kcal)
    return (
        predicted_kcal,
        max(0.0, predicted_kcal - error_range_kcal),
        predicted_kcal + error_range_kcal,
        error_range_kcal,
    )


st.title("Food Calorie Estimator")
st.write(
    "Upload one food-plate photo for an approximate calorie estimate. "
    "The app estimates calories only; it does not identify the dish."
)

uploaded_file = st.file_uploader(
    "Choose a food photo",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("The uploaded file is not a supported image.")
        st.stop()

    st.image(image, caption="Uploaded food photo", width="stretch")

    try:
        with st.spinner("Estimating calories..."):
            estimate, low, high, error_range = estimate_calories(image, get_model())
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    st.metric("Estimated calories", f"{estimate:,.0f} kcal")
    st.caption(
        f"Validation-based rough range: {low:,.0f}–{high:,.0f} kcal "
        f"(approximately ±{error_range:,.0f} kcal)"
    )

st.warning(
    "This is an uncertain estimate from a small California-cafeteria dataset; "
    "accuracy for Indian meals is not established. A photo cannot reveal exact "
    "ingredients, recipe, or serving weight. Do not use this as medical or "
    "dietary advice."
)
