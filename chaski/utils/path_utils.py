from fastcore.basics import Path

def get_project_root():
    """Get the root directory for the project."""
    return Path(__file__).resolve().parent.parent.parent

def get_outputs_dir():
    """Get the output directory for the project.
    
    Mainly used to make sure all example outputs are saved in the same place.
        The top-level `outputs/` folder.
    """
    project_root = get_project_root()
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir