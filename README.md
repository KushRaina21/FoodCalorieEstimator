# Food Calorie Estimator

A separate, local-first experiment for estimating calories from a food-plate photo. The app reports an approximate calorie range only; it does not identify or name the dish.

## Important limitation

An RGB photo cannot reveal ingredients, recipe, hidden oil, or exact food weight. The first training notebook uses the public [Nutrition5k FoodSeg103-filtered dataset](https://huggingface.co/datasets/arunapb/nutrition5k-foodseg103), a small subset derived from Google's [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k). Nutrition5k has plate images with measured total calories and mass, but its plates were collected in California cafeterias. The filtered subset is under 1,000 examples and is **not Indian-food-specific**. Indian-food image datasets found during planning provide dish labels but not measured calorie labels. Therefore this starter cannot claim reliable estimates for Indian meals or arbitrary dishes.

The estimate is for a whole pictured plate, in kcal, and its displayed range is based on validation errors from this limited dataset. Treat it as a learning demo, not dietary or medical advice. For better Indian-food estimates, collect licensed Indian meal photos paired with measured serving weights and recipe-based calorie totals, then retrain and evaluate on held-out Indian meals.

## Setup

Python 3.11 or newer is recommended. Python 3.13 was used to validate this project.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-training.txt
python -m ipykernel install --user --name food-calorie-estimator
jupyter notebook
```

Open `notebooks/train_calorie_model.ipynb` and run its cells from top to bottom. The notebook downloads a pinned revision of the small public training subset, fine-tunes an ImageNet-pretrained ResNet-18 using FastAI, calibrates using validation data, compares held-out mean absolute error against a training-mean baseline, and exports the model and a validation-error range to `models/`.

The pretrained weights download on first training and training can be slow on CPU. Keep the official train and test splits separate; the notebook uses only the source training split for fitting and validation and reports final error on the held-out source test split.

## Run the Streamlit app

After the notebook completes its export cell:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal and upload one food-plate image. The app only displays estimated calories and the approximate range. If the exported model is missing, it explains how to train it first.

## Deploy to Streamlit Community Cloud

1. Push this project, including `models/food_calorie_model.pkl` and `models/calorie_metrics.json`, to its own GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and choose `app.py` as the main file.
3. Select Python 3.13 in the app's advanced settings. Cloud installs the pinned, locally tested serving packages from `requirements.txt` (including FastAI 2.7.19 and PyTorch 2.6.0).

The trained model artifact is approximately 45 MB and is included with the app so predictions are available at startup. The notebook and training dependencies remain in the repository, but are not installed in the Cloud serving environment.

## Data and attribution

- The starter subset card reports CC BY 4.0 and identifies itself as derived from Nutrition5k; check the source card and original [Google dataset terms](https://github.com/google-research-datasets/Nutrition5k) before redistribution.
- Cite the Nutrition5k paper: Thames et al., *Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food*, CVPR 2021, [arXiv:2103.03375](https://arxiv.org/abs/2103.03375).
- Do not redistribute trained weights or data without checking the applicable data and pretrained-weight terms.
