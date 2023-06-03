import os
import textwrap
import time
from typing import NoReturn

import optuna
from optuna.trial import TrialState
from optuna_dashboard import ChoiceWidget
from optuna_dashboard import register_objective_form_widgets
from optuna_dashboard import save_note
from optuna_dashboard.artifact import get_artifact_path
from optuna_dashboard.artifact import upload_artifact
from optuna_dashboard.artifact.file_system import FileSystemBackend
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor

doravirine = Chem.MolFromSmiles('Cn1c(n[nH]c1=O)Cn2ccc(c(c2=O)Oc3cc(cc(c3)Cl)C#N)C(F)(F)F')


def make_png(mol, r, g, b, legend="", highlightAtoms=[]):
    d2d = Draw.MolDraw2DCairo(350, 300)
    dopts = d2d.drawOptions()
    dopts.setHighlightColour((r, g, b, .4))
    d2d.DrawMolecule(mol, legend=legend, highlightAtoms=highlightAtoms)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def suggest_and_generate_image(study: optuna.Study, artifact_backend: FileSystemBackend) -> None:
    # 1. Ask new parameters
    trial = study.ask()
    r = trial.suggest_float("r", 0, 1)
    g = trial.suggest_float("g", 0, 1)
    b = trial.suggest_float("b", 0, 1)

    # 2. Generate image
    image_path = f"tmp/sample-{trial.number}.png"
    drawingtext = make_png(doravirine, r, g, b, legend=f"trial {trial.number}", highlightAtoms=(0,1,2,3,4,5,6))
    with open(image_path, 'wb') as f:
        f.write(drawingtext)

    # 3. Upload Artifact
    artifact_id = upload_artifact(artifact_backend, trial, image_path)
    print(artifact_id)
    artifact_path = get_artifact_path(trial, artifact_id)
    #artifact_path = os.path.join("artifact", artifact_id)
    print(artifact_path)

    # 4. Save Note
    note = textwrap.dedent(
        f"""\
    ## Trial {trial.number}

    ![generated-image]({artifact_path})
    """
    )
    save_note(trial, note)


def start_optimization(artifact_backend: FileSystemBackend) -> NoReturn:
    # 1. Create Study
    study = optuna.create_study(
        study_name="Human-in-the-loop Optimization",
        storage="sqlite:///db.sqlite3",
        sampler=optuna.samplers.TPESampler(constant_liar=True, n_startup_trials=5),
        load_if_exists=True,
    )

    # 2. Set an objective name
    study.set_metric_names(["Do you like this color?"])

    # 3. Register ChoiceWidget
    register_objective_form_widgets(
        study,
        widgets=[
            ChoiceWidget(
                choices=["Good 👍", "So-so👌", "Bad 👎"],
                values=[-1, 0, 1],
                description="Please input your score!",
            ),
        ],
    )

    # 4. Start Human-in-the-loop Optimization
    n_batch = 4
    while True:
        running_trials = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
        if len(running_trials) >= n_batch:
            time.sleep(1)  # Avoid busy-loop
            continue
        suggest_and_generate_image(study, artifact_backend)


def main() -> NoReturn:
    tmp_path = os.path.join(os.path.dirname(__file__), "tmp")

    # 1. Create Artifact Store
    artifact_path = os.path.join(os.path.dirname(__file__), "artifact")
    artifact_backend = FileSystemBackend(base_path=artifact_path)

    if not os.path.exists(artifact_path):
        os.mkdir(artifact_path)

    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    # 2. Run optimize loop
    start_optimization(artifact_backend)


if __name__ == "__main__":
    main()
