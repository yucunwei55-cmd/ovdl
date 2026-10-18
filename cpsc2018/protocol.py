"""Generate reference splits in a temporary directory, verify before committing."""
import tempfile
from pathlib import Path
from io_utils import read_json, write_json, sha256
from .data import make_splits


def prepare_splits(a, out):
    # The imported reference writes CSVs unconditionally. Never let it overwrite
    # an existing experiment: compare generated contents before copying anything.
    with tempfile.TemporaryDirectory(prefix="ovdl_cpsc_split_") as temporary:
        _, paths, manifest = make_splits(a.data_dir, a.labels, temporary, a.split_seed,
                                        a.split_mode, a.expected_af_count, a.reference_csv)
        metadata = out / "run_config.json"
        hashes = {name: manifest["sha256"][name + ".csv"] for name in ("train", "val", "test")}
        if metadata.exists():
            previous = read_json(metadata)
            if previous["split_sha256"] != hashes or previous["reference_sha256"] != manifest["label_csv_sha256"]:
                raise ValueError("Labels or split content changed; refusing to overwrite")
            if previous.get("diagnosis_reference_sha256") != manifest["reference_csv_sha256"]:
                raise ValueError("Complete diagnosis reference changed; refusing to overwrite")
        targets = {}
        for name, source in zip(("train", "val", "test"), paths):
            target = out / "splits" / (name + ".csv")
            if target.exists() and sha256(target) != hashes[name]:
                raise ValueError("Saved split was modified: " + str(target))
            targets[name] = target
        for name, source in zip(("train", "val", "test"), paths):
            target = targets[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(Path(source).read_bytes())
        write_json(manifest, out / "split_manifest.json")
    return hashes, manifest
