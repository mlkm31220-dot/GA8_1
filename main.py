from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import unicodedata


app = FastAPI()


# ---------------------------------------------------------
# Constants / validators
# ---------------------------------------------------------

URI_RE = re.compile(r"^gs://[^/]+/[^/]+$")
GEN_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")

# YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)
TS_RE = re.compile(
    r"^"
    r"\d{4}-\d{2}-\d{2}"
    r"T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})"
    r"$"
)

SAFE_INT_MAX = (1 << 53) - 1


# ---------------------------------------------------------
# CRC32C / Castagnoli
# ---------------------------------------------------------

CRC32C_POLY = 0x82F63B78


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ CRC32C_POLY
            else:
                crc >>= 1

    return crc ^ 0xFFFFFFFF


def crc32c_hex(text: str) -> str:
    return f"{crc32c(text.encode('utf-8')):08x}"


# ---------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------

def parse_ts(ts):
    if not isinstance(ts, str):
        return None

    if not TS_RE.fullmatch(ts):
        return None

    try:
        # Validate the offset magnitude strictly.
        if ts.endswith("Z"):
            value = ts[:-1] + "+00:00"
        else:
            offset = ts[-6:]
            sign = offset[0]
            hours = int(offset[1:3])
            minutes = int(offset[4:6])

            if minutes > 59:
                return None

            if hours > 14:
                return None

            if hours == 14 and minutes != 0:
                return None

            value = ts

        dt = datetime.fromisoformat(value)

        if dt.tzinfo is None:
            return None

        return dt

    except (ValueError, TypeError, OverflowError):
        return None


def norm_ts(ts):
    dt = parse_ts(ts)

    if dt is None:
        return None

    dt = dt.astimezone(timezone.utc)

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}"
        + "Z"
    )


# ---------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------

def canon(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()
    value = value.strip()

    # Collapse Unicode whitespace to ASCII space.
    chars = []
    in_space = False

    for ch in value:
        if ch.isspace():
            if not in_space:
                chars.append(" ")
            in_space = True
        else:
            chars.append(ch)
            in_space = False

    return "".join(chars)


# ---------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------

def compact(row):
    return json.dumps(
        {
            "id": row["id"],
            "entity": row["entity"],
            "eventTime": row["eventTime"],
            "revision": row["revision"],
            "text": row["text"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def row_sort_key(row):
    return (
        row["id"].encode("utf-8"),
        compact(row).encode("utf-8"),
    )


def digest(rows):
    rows = sorted(rows, key=row_sort_key)

    data = "".join(
        compact(row) + "\n"
        for row in rows
    ).encode("utf-8")

    return hashlib.sha256(data).hexdigest(), rows


# ---------------------------------------------------------
# Unicode letter/number word-set
# ---------------------------------------------------------

def words(text):
    """
    Extract maximal sequences consisting only of Unicode
    letters or Unicode numbers.

    Underscore is NOT considered a word character.
    """

    result = []
    current = []

    for ch in text.lower():
        category = unicodedata.category(ch)

        if category.startswith("L") or category.startswith("N"):
            current.append(ch)
        else:
            if current:
                result.append("".join(current))
                current = []

    if current:
        result.append("".join(current))

    return set(result)


def jaccard(a, b):
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ---------------------------------------------------------
# Deterministic sorting helpers
# ---------------------------------------------------------

def compact_obj(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def reason_codes(codes):
    return sorted(
        set(codes),
        key=lambda x: x.encode("utf-8"),
    )


def sort_rejected_objects(items):
    return sorted(
        items,
        key=lambda x: (
            (x["uri"] if isinstance(x["uri"], str) else "").encode("utf-8"),
            compact_obj(x).encode("utf-8"),
        ),
    )


def sort_rejected_rows(items):
    return sorted(
        items,
        key=lambda x: (
            x["id"].encode("utf-8"),
            compact_obj(x).encode("utf-8"),
        ),
    )


def sort_lineage(items):
    return sorted(
        items,
        key=lambda x: (
            x["uri"].encode("utf-8"),
            compact_obj(x).encode("utf-8"),
        ),
    )


# ---------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------

@app.post("/build-corpus")
async def build_corpus(body: dict):

    # Exact invalid-input condition.
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("policy"), dict)
        or not isinstance(body.get("objects"), list)
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    policy = body["policy"]

    min_t = parse_ts(policy.get("minTime"))
    max_t = parse_ts(policy.get("maxTime"))

    threshold = policy.get("contaminationThreshold")

    policy_valid = (
        min_t is not None
        and max_t is not None
        and isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(threshold)
        and 0 <= threshold <= 1
    )

    retained = []

    rejected_objects = []
    rejected_rows = []
    lineage = []

    # -----------------------------------------------------
    # Validate objects
    # -----------------------------------------------------

    for obj in body["objects"]:

        # A non-object in objects cannot satisfy the required
        # object fields. Treat it as an invalid object.
        if not isinstance(obj, dict):
            rejected_objects.append(
                {
                    "uri": None,
                    "reasonCodes": [
                        "URI_INVALID",
                        "GENERATION_INVALID",
                        "CRC32C_INVALID",
                        "SCHEMA_INVALID",
                    ],
                }
            )
            continue

        codes = []

        uri = obj.get("uri")

        # URI
        if not isinstance(uri, str) or not URI_RE.fullmatch(uri):
            codes.append("URI_INVALID")

        # Generations
        generation = obj.get("generation")
        fetched_generation = obj.get("fetchedGeneration")

        generation_valid = (
            isinstance(generation, str)
            and GEN_RE.fullmatch(generation) is not None
        )

        fetched_generation_valid = (
            isinstance(fetched_generation, str)
            and GEN_RE.fullmatch(fetched_generation) is not None
        )

        if not generation_valid:
            codes.append("GENERATION_INVALID")

        if not fetched_generation_valid:
            codes.append("GENERATION_INVALID")

        if (
            generation_valid
            and fetched_generation_valid
            and generation != fetched_generation
        ):
            codes.append("GENERATION_MISMATCH")

        # CRC
        crc = obj.get("crc32c")

        crc_valid = (
            isinstance(crc, str)
            and CRC_RE.fullmatch(crc) is not None
        )

        if not crc_valid:
            codes.append("CRC32C_INVALID")

        # Schema/content
        content = obj.get("content")
        schema_id = obj.get("schemaId")

        if schema_id != "training-v1" or not isinstance(content, str):
            codes.append("SCHEMA_INVALID")

        # CRC mismatch is checked only when content is a string
        # and CRC syntax is valid.
        if isinstance(content, str) and crc_valid:
            if crc32c_hex(content) != crc:
                codes.append("CRC32C_MISMATCH")

        rows = []

        # Only parse JSONL when the object-level integrity fields
        # are otherwise valid.
        if not codes:

            for line in content.splitlines():

                # Blank lines are ignored.
                if not line.strip():
                    continue

                try:
                    parsed = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    codes.append("JSONL_INVALID")
                    rows = []
                    break

                # JSON must be an object.
                if not isinstance(parsed, dict):
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                expected_keys = {
                    "id",
                    "entity",
                    "eventTime",
                    "revision",
                    "text",
                }

                if set(parsed.keys()) != expected_keys:
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                # Strings.
                if not isinstance(parsed["id"], str):
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                if not isinstance(parsed["entity"], str):
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                if not isinstance(parsed["eventTime"], str):
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                if not isinstance(parsed["text"], str):
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                # Revision:
                # JSON booleans are not accepted as integers.
                revision = parsed["revision"]

                if (
                    not isinstance(revision, int)
                    or isinstance(revision, bool)
                    or revision < 0
                    or revision > SAFE_INT_MAX
                ):
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                # Timestamp
                event_time = norm_ts(parsed["eventTime"])

                if event_time is None:
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                rows.append(
                    {
                        "id": parsed["id"],
                        "entity": canon(parsed["entity"]),
                        "eventTime": event_time,
                        "revision": revision,
                        "text": canon(parsed["text"]),
                    }
                )

            # Empty/blank-only content.
            if (
                not rows
                and "JSONL_INVALID" not in codes
            ):
                codes.append("SCHEMA_INVALID")

        # Object rejected.
        if codes:
            rejected_objects.append(
                {
                    "uri": uri if isinstance(uri, str) else None,
                    "reasonCodes": reason_codes(codes),
                }
            )
            continue

        # Valid object contributes lineage and rows.
        lineage.append(
            {
                "uri": uri,
                "generation": generation,
                "crc32c": crc,
                "schemaId": "training-v1",
            }
        )

        retained.extend(rows)

    # -----------------------------------------------------
    # Deduplication
    # -----------------------------------------------------

    best = {}

    for row in retained:

        key = (
            row["entity"],
            row["eventTime"],
            row["text"],
        )

        existing = best.get(key)

        if existing is None:
            best[key] = row
            continue

        # Higher revision wins.
        # If tied, UTF-8-smallest ID wins.
        new_wins = (
            row["revision"] > existing["revision"]
            or (
                row["revision"] == existing["revision"]
                and row["id"].encode("utf-8")
                < existing["id"].encode("utf-8")
            )
        )

        if new_wins:
            rejected_rows.append(
                {
                    "id": existing["id"],
                    "reasonCodes": ["DUPLICATE"],
                }
            )
            best[key] = row
        else:
            rejected_rows.append(
                {
                    "id": row["id"],
                    "reasonCodes": ["DUPLICATE"],
                }
            )

    retained = list(best.values())

    # -----------------------------------------------------
    # Policy / time window
    # -----------------------------------------------------

    if not policy_valid:

        for row in retained:
            rejected_rows.append(
                {
                    "id": row["id"],
                    "reasonCodes": ["POLICY_INVALID"],
                }
            )

        retained = []

    else:

        kept = []

        for row in retained:

            dt = parse_ts(row["eventTime"])

            if min_t <= dt <= max_t:
                kept.append(row)
            else:
                rejected_rows.append(
                    {
                        "id": row["id"],
                        "reasonCodes": ["OUT_OF_WINDOW"],
                    }
                )

        retained = kept

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------

    train = []
    validation = []
    test = []

    for row in retained:

        bucket = (
            hashlib.sha256(
                row["entity"].encode("utf-8")
            ).digest()[0]
            % 10
        )

        if 0 <= bucket <= 5:
            train.append(row)

        elif 6 <= bucket <= 7:
            validation.append(row)

        else:
            test.append(row)

    # -----------------------------------------------------
    # Contamination
    # -----------------------------------------------------

    train_sets = [
        words(row["text"])
        for row in train
    ]

    def remove_contamination(rows):

        kept = []

        for row in rows:

            current_words = words(row["text"])

            contaminated = False

            for train_words in train_sets:

                similarity = jaccard(
                    current_words,
                    train_words,
                )

                if similarity >= threshold:
                    rejected_rows.append(
                        {
                            "id": row["id"],
                            "reasonCodes": [
                                "TRAIN_CONTAMINATION"
                            ],
                        }
                    )
                    contaminated = True
                    break

            if not contaminated:
                kept.append(row)

        return kept

    validation = remove_contamination(validation)
    test = remove_contamination(test)

    # -----------------------------------------------------
    # Deterministic artifacts
    # -----------------------------------------------------

    train_digest, train = digest(train)
    validation_digest, validation = digest(validation)
    test_digest, test = digest(test)

    # -----------------------------------------------------
    # Final deterministic ordering
    # -----------------------------------------------------

    rejected_objects = sort_rejected_objects(
        rejected_objects
    )

    rejected_rows = sort_rejected_rows(
        rejected_rows
    )

    lineage = sort_lineage(lineage)

    # -----------------------------------------------------
    # Exact response shape
    # -----------------------------------------------------

    return {
        "splits": {
            "train": train,
            "validation": validation,
            "test": test,
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": {
            "train": train_digest,
            "validation": validation_digest,
            "test": test_digest,
        },
        "lineage": lineage,
    }
