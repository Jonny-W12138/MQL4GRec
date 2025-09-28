from huggingface_hub import HfApi

api = HfApi(token='')
api.upload_folder(
    folder_path="/Users/jonnyw/Documents/personal/internship/recommender-system/MQL4GRec/log/Instruments",
    repo_id="JonnyW/Instruments_rqvae",
    repo_type="model",
)