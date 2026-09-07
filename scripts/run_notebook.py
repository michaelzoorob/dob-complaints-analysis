"""Execute a notebook in place with the pinned kernel (nbclient), from the repo root."""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / sys.argv[1]
nb = nbformat.read(nb_path, as_version=4)
t0 = time.time()
NotebookClient(nb, timeout=3600, kernel_name="pyfix", resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(nb, nb_path)
print(f"executed {nb_path.name} in {time.time()-t0:.0f}s", flush=True)
