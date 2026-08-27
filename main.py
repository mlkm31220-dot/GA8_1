from fastapi import FastAPI
from fastapi.responses import JSONResponse
import hashlib
import json
import math
import re


app = FastAPI()


# ============================================================
# Constants
# ============================================================

REQUIRED_FILES = [
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
]

UNSAFE_EXTENSIONS = {
    ".bin",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
}

SHA256_RE = re.compile(r"^[0-9a-f]{40}$")


# ============================================================
# Helpers
# ============================================================

def utf8_bytes(value):
    return value.encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def utf8_sort_key(value):
    return value.encode("utf-8")


def safe_positive_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= (2**53 - 1)
    )


def finite_unit_interval(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def is_valid_sha256(value):
    return (
        isinstance(value, str)
        and SHA256_RE.fullmatch(value) is not None
    )


# ============================================================
# Policy validation
# ============================================================

def validate_policy(policy):
    violations = []

    if not isinstance(policy, dict):
        return ["INVALID_POLICY"]

    required_slices = policy.get("requiredSlices")

    if (
        not isinstance(required_slices, list)
        or len(required_slices) == 0
        or any(
            not isinstance(x, str) or x == ""
            for x in required_slices
        )
        or len(set(required_slices)) != len(required_slices)
    ):
        violations.append("INVALID_POLICY")

    for field in ("license", "intendedUse", "limitations"):
        value = policy.get(field)

        if not isinstance(value, str) or value == "":
            violations.append("INVALID_POLICY")

    return violations


# ============================================================
# JSON parsing
# ============================================================

def parse_json_file(files, name):
    violations = []

    if name not in files:
        return None, violations

    value = files[name]

    if not isinstance(value, str):
        violations.append(f"INVALID_FILE:{name}")
        return None, violations

    try:
        parsed = json.loads(value)
        return parsed, violations
    except (json.JSONDecodeError, TypeError):
        violations.append(f"INVALID_JSON:{name}")
        return None, violations


# ============================================================
# Inventory
# ============================================================

def compute_inventory(files):
    entries = []

    for name, content in files.items():

        if name == "inventory.json":
            continue

        if not isinstance(name, str):
            continue

        if not isinstance(content, str):
            continue

        raw = content.encode("utf-8")

        entries.append(
            {
                "name": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

    entries.sort(
        key=lambda x: x["name"].encode("utf-8")
    )

    return entries


def verify_inventory(files, supplied_inventory):
    violations = []

    expected = compute_inventory(files)

    if not isinstance(supplied_inventory, list):
        return ["INVENTORY_MISMATCH"]

    # Exact compact JSON comparison is important.
    expected_bytes = compact_json(expected).encode("utf-8")
    supplied_bytes = compact_json(supplied_inventory).encode("utf-8")

    if expected_bytes != supplied_bytes:
        violations.append("INVENTORY_MISMATCH")

    return violations


# ============================================================
# Adapter config
# ============================================================

def validate_adapter_config(config):
    if not isinstance(config, dict):
        return ["INVALID_ADAPTER_CONFIG"]

    r = config.get("r")

    if not safe_positive_integer(r):
        return ["INVALID_ADAPTER_CONFIG"]

    target_modules = config.get("target_modules")

    if (
        not isinstance(target_modules, list)
        or len(target_modules) == 0
        or any(
            not isinstance(x, str) or x == ""
            for x in target_modules
        )
        or len(set(target_modules)) != len(target_modules)
    ):
        return ["INVALID_ADAPTER_CONFIG"]

    return []


# ============================================================
# Training manifest
# ============================================================

MANIFEST_REQUIRED_FIELDS = [
    "baseRevision",
    "task",
    "datasetDigest",
    "codeDigest",
    "trainingConfigDigest",
    "modelArtifactDigest",
    "evaluationArtifactDigest",
]


def validate_training_manifest(manifest):
    violations = []

    if not isinstance(manifest, dict):
        return ["INVALID_TRAINING_MANIFEST"]

    base_revision = manifest.get("baseRevision")

    if (
        not isinstance(base_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", base_revision) is None
    ):
        violations.append("MUTABLE_BASE_REVISION")

    for field in MANIFEST_REQUIRED_FIELDS[1:]:
        value = manifest.get(field)

        if not isinstance(value, str) or value == "":
            violations.append(
                f"MISSING_MANIFEST_FIELD:{field}"
            )

    return violations


# ============================================================
# Evaluation
# ============================================================

def validate_evaluation(
    evaluation,
    model_digest,
    required_slices,
):
    violations = []

    if not isinstance(evaluation, dict):
        return ["INVALID_EVALUATION"]

    # The evaluation must bind the model artifact digest.
    evaluation_model_digest = evaluation.get(
        "modelArtifactDigest"
    )

    if evaluation_model_digest != model_digest:
        violations.append(
            "MODEL_ARTIFACT_MISMATCH"
        )

    # Aggregate
    aggregate = evaluation.get("aggregate")

    if aggregate is None:
        violations.append("INVALID_AGGREGATE")
    elif not finite_unit_interval(aggregate):
        violations.append("INVALID_AGGREGATE")

    # Required slices
    slices = evaluation.get("slices")

    if not isinstance(slices, dict):
        for slice_name in required_slices:
            violations.append(
                f"MISSING_SLICE:{slice_name}"
            )
    else:
        for slice_name in required_slices:

            if slice_name not in slices:
                violations.append(
                    f"MISSING_SLICE:{slice_name}"
                )
                continue

            value = slices[slice_name]

            if not finite_unit_interval(value):
                violations.append(
                    f"SLICE_RANGE:{slice_name}"
                )

    return violations


# ============================================================
# Model card
# ============================================================

MODEL_CARD_PREFIX = "<!-- tds-model-card "
MODEL_CARD_SUFFIX = "-->"


def find_model_cards(readme):
    """
    Find exact model-card markers.

    Braces inside JSON strings do not affect parsing because
    we locate the closing '-->' delimiter rather than trying
    to parse braces manually.
    """

    cards = []
    position = 0

    while True:

        start = readme.find(
            MODEL_CARD_PREFIX,
            position,
        )

        if start == -1:
            break

        payload_start = start + len(
            MODEL_CARD_PREFIX
        )

        end = readme.find(
            MODEL_CARD_SUFFIX,
            payload_start,
        )

        if end == -1:
            # Marker exists but has no closing delimiter.
            cards.append(None)
            break

        payload = readme[
            payload_start:end
        ].strip()

        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            parsed = None

        cards.append(parsed)

        position = end + len(
            MODEL_CARD_SUFFIX
        )

    return cards


def validate_model_card(
    readme,
    manifest,
    evaluation,
    policy,
):
    violations = []

    if not isinstance(readme, str):
        return [
            "MODEL_CARD_COUNT",
            "MISSING_MODEL_CARD",
        ]

    cards = find_model_cards(readme)

    if len(cards) == 0:
        return ["MISSING_MODEL_CARD"]

    if len(cards) > 1:
        return ["MODEL_CARD_COUNT"]

    card = cards[0]

    if not isinstance(card, dict):
        return ["INVALID_MODEL_CARD"]

    expected = {
        "task": manifest.get("task"),
        "baseRevision": manifest.get("baseRevision"),
        "datasetDigest": manifest.get("datasetDigest"),
        "modelArtifactDigest": manifest.get(
            "modelArtifactDigest"
        ),
        "license": policy.get("license"),
        "intendedUse": policy.get("intendedUse"),
        "limitations": policy.get("limitations"),
    }

    for field, expected_value in expected.items():

        if card.get(field) != expected_value:
            violations.append(
                "MODEL_CARD_MISMATCH"
            )
            break

    return violations


# ============================================================
# Main endpoint
# ============================================================

@app.post("/verify-bundle")
async def verify_bundle(body: dict):

    # --------------------------------------------------------
    # Top-level input validation
    # --------------------------------------------------------

    if (
        not isinstance(body, dict)
        or "policy" not in body
        or "files" not in body
        or not isinstance(body.get("policy"), dict)
        or not isinstance(body.get("files"), dict)
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    policy = body["policy"]
    files = body["files"]

    violations = []

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    violations.extend(
        validate_policy(policy)
    )

    required_slices = policy.get(
        "requiredSlices",
        [],
    )

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    for name in REQUIRED_FILES:
        if name not in files:
            violations.append(
                f"MISSING_FILE:{name}"
            )

    # --------------------------------------------------------
    # File value validation
    # --------------------------------------------------------

    for name, value in files.items():

        if not isinstance(name, str):
            violations.append(
                "UNTRACKED_FILE"
            )
            continue

        if not isinstance(value, str):
            violations.append(
                f"INVALID_FILE:{name}"
            )

    # --------------------------------------------------------
    # Extra / untracked files
    # --------------------------------------------------------

    for name in files:

        if not isinstance(name, str):
            continue

        if name not in REQUIRED_FILES:
            violations.append(
                "UNTRACKED_FILE"
            )

    # --------------------------------------------------------
    # Unsafe weights
    # --------------------------------------------------------

    for name in files:

        if not isinstance(name, str):
            continue

        lower = name.lower()

        for extension in UNSAFE_EXTENSIONS:

            if lower.endswith(extension):
                violations.append(
                    "UNSAFE_WEIGHTS"
                )
                break

    # --------------------------------------------------------
    # inventory.json
    # --------------------------------------------------------

    inventory = None

    if "inventory.json" in files:

        inventory, inventory_errors = parse_json_file(
            files,
            "inventory.json",
        )

        violations.extend(inventory_errors)

        if inventory_errors == []:
            violations.extend(
                verify_inventory(
                    files,
                    inventory,
                )
            )

    # --------------------------------------------------------
    # Inventory digest
    # --------------------------------------------------------

    recomputed_inventory = compute_inventory(
        files
    )

    inventory_digest = hashlib.sha256(
        compact_json(
            recomputed_inventory
        ).encode("utf-8")
    ).hexdigest()

    # --------------------------------------------------------
    # Adapter config
    # --------------------------------------------------------

    adapter_config = None

    if "adapter_config.json" in files:

        adapter_config, errors = parse_json_file(
            files,
            "adapter_config.json",
        )

        violations.extend(errors)

        if not errors:
            violations.extend(
                validate_adapter_config(
                    adapter_config
                )
            )

    # --------------------------------------------------------
    # Training manifest
    # --------------------------------------------------------

    manifest = None

    if "training_manifest.json" in files:

        manifest, errors = parse_json_file(
            files,
            "training_manifest.json",
        )

        violations.extend(errors)

        if not errors:
            violations.extend(
                validate_training_manifest(
                    manifest
                )
            )

    # --------------------------------------------------------
    # Model artifact digest
    # --------------------------------------------------------

    model_digest = None

    if (
        "adapter_model.safetensors" in files
        and isinstance(
            files["adapter_model.safetensors"],
            str,
        )
    ):
        model_digest = hashlib.sha256(
            files[
                "adapter_model.safetensors"
            ].encode("utf-8")
        ).hexdigest()

    # --------------------------------------------------------
    # Evaluation artifact digest
    # --------------------------------------------------------

    evaluation_digest = None
    evaluation = None

    if "evaluation.json" in files:

        if isinstance(files["evaluation.json"], str):

            evaluation_bytes = files[
                "evaluation.json"
            ].encode("utf-8")

            evaluation_digest = hashlib.sha256(
                evaluation_bytes
            ).hexdigest()

        evaluation, errors = parse_json_file(
            files,
            "evaluation.json",
        )

        violations.extend(errors)

    # --------------------------------------------------------
    # Manifest digest binding
    # --------------------------------------------------------

    if isinstance(manifest, dict):

        expected_model_digest = manifest.get(
            "modelArtifactDigest"
        )

        expected_evaluation_digest = manifest.get(
            "evaluationArtifactDigest"
        )

        if (
            model_digest is not None
            and isinstance(
                expected_model_digest,
                str,
            )
            and expected_model_digest != model_digest
        ):
            violations.append(
                "MODEL_ARTIFACT_MISMATCH"
            )

        if (
            evaluation_digest is not None
            and isinstance(
                expected_evaluation_digest,
                str,
            )
            and expected_evaluation_digest
            != evaluation_digest
        ):
            violations.append(
                "EVALUATION_DIGEST_MISMATCH"
            )

    # --------------------------------------------------------
    # Evaluation binding
    # --------------------------------------------------------

    if (
        evaluation is not None
        and model_digest is not None
    ):
        violations.extend(
            validate_evaluation(
                evaluation,
                model_digest,
                required_slices
                if isinstance(required_slices, list)
                else [],
            )
        )

    elif evaluation is not None:

        # Still validate evaluation structure when the model
        # artifact is unavailable.
        if not isinstance(evaluation, dict):
            violations.append(
                "INVALID_EVALUATION"
            )

    # --------------------------------------------------------
    # Evaluation artifact digest against manifest
    # --------------------------------------------------------

    if (
        isinstance(manifest, dict)
        and evaluation_digest is not None
    ):

        expected = manifest.get(
            "evaluationArtifactDigest"
        )

        if (
            isinstance(expected, str)
            and expected != evaluation_digest
        ):
            violations.append(
                "EVALUATION_ARTIFACT_MISMATCH"
            )

    # --------------------------------------------------------
    # Model card
    # --------------------------------------------------------

    if "README.md" in files:

        readme = files["README.md"]

        violations.extend(
            validate_model_card(
                readme,
                manifest
                if isinstance(manifest, dict)
                else {},
                evaluation
                if isinstance(evaluation, dict)
                else {},
                policy,
            )
        )

    # --------------------------------------------------------
    # Deterministic violations
    # --------------------------------------------------------

    violations = sorted(
        set(violations),
        key=lambda x: x.encode("utf-8"),
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision = (
        "admit"
        if len(violations) == 0
        else "reject"
    )

    return {
        "decision": decision,
        "violations": violations,
        "inventoryDigest": inventory_digest,
    }
