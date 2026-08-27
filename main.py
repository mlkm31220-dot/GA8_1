
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import hashlib, json, math, re, unicodedata, zlib

app = FastAPI()

URI_RE = re.compile(r"^gs://[^/]+/[^/].+$")
GEN_RE = re.compile(r"^\d+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")


def parse_ts(ts):
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except:
        return None


def norm_ts(ts):
    dt = parse_ts(ts)
    if dt is None:
        return None
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def canon(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s.strip())


def crc32_hex(text):
    return format(zlib.crc32(text.encode("utf-8")) & 0xffffffff, "08x")


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


def digest(rows):
    rows = sorted(rows, key=lambda r: (r["id"].encode(), compact(r).encode()))
    data = "".join(compact(r) + "\n" for r in rows).encode("utf-8")
    return hashlib.sha256(data).hexdigest(), rows


def words(text):
    return set(re.findall(r"\w+", text, flags=re.UNICODE))


@app.post("/build-corpus")
async def build_corpus(body: dict):
    if not isinstance(body.get("policy"), dict) or not isinstance(body.get("objects"), list):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    policy = body["policy"]
    min_t = parse_ts(policy.get("minTime", ""))
    max_t = parse_ts(policy.get("maxTime", ""))
    th = policy.get("contaminationThreshold")

    policy_valid = (
        min_t is not None
        and max_t is not None
        and isinstance(th, (int, float))
        and math.isfinite(th)
        and 0 <= th <= 1
    )

    retained = []
    rejected_objects = []
    rejected_rows = []
    lineage = []

    for obj in body["objects"]:
        codes = []
        uri = obj.get("uri")

        if not isinstance(uri, str) or not URI_RE.fullmatch(uri):
            codes.append("URI_INVALID")

        g = obj.get("generation")
        fg = obj.get("fetchedGeneration")

        if not isinstance(g, str) or not GEN_RE.fullmatch(g):
            codes.append("GENERATION_INVALID")
        if not isinstance(fg, str) or not GEN_RE.fullmatch(fg):
            if "GENERATION_INVALID" not in codes:
                codes.append("GENERATION_INVALID")
        elif g != fg:
            codes.append("GENERATION_MISMATCH")

        crc = obj.get("crc32c")
        if not isinstance(crc, str) or not CRC_RE.fullmatch(crc):
            codes.append("CRC32C_INVALID")

        content = obj.get("content")
        if obj.get("schemaId") != "training-v1" or not isinstance(content, str):
            codes.append("SCHEMA_INVALID")

        if isinstance(content, str) and isinstance(crc, str) and CRC_RE.fullmatch(crc):
            if crc32_hex(content) != crc:
                codes.append("CRC32C_MISMATCH")

        rows = []
        if not codes:
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except:
                    codes.append("JSONL_INVALID")
                    rows = []
                    break

                if set(r.keys()) != {"id", "entity", "eventTime", "revision", "text"}:
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                if not (
                    isinstance(r["id"], str)
                    and isinstance(r["entity"], str)
                    and isinstance(r["text"], str)
                    and isinstance(r["revision"], int)
                    and r["revision"] >= 0
                ):
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                nt = norm_ts(r["eventTime"])
                if nt is None:
                    codes.append("SCHEMA_INVALID")
                    rows = []
                    break

                rows.append(
                    {
                        "id": r["id"],
                        "entity": canon(r["entity"]),
                        "eventTime": nt,
                        "revision": r["revision"],
                        "text": canon(r["text"]),
                    }
                )

            if not rows and "JSONL_INVALID" not in codes:
                codes.append("SCHEMA_INVALID")

        if codes:
            rejected_objects.append(
                {
                    "uri": uri if isinstance(uri, str) else None,
                    "reasonCodes": sorted(set(codes)),
                }
            )
        else:
            lineage.append(
                {
                    "uri": uri,
                    "generation": g,
                    "crc32c": crc,
                    "schemaId": "training-v1",
                }
            )
            retained.extend(rows)

    best = {}
    for r in retained:
        key = (r["entity"], r["eventTime"], r["text"])
        if key not in best:
            best[key] = r
        else:
            b = best[key]
            if r["revision"] > b["revision"] or (
                r["revision"] == b["revision"] and r["id"].encode() < b["id"].encode()
            ):
                rejected_rows.append({"id": b["id"], "reasonCodes": ["DUPLICATE"]})
                best[key] = r
            else:
                rejected_rows.append({"id": r["id"], "reasonCodes": ["DUPLICATE"]})

    retained = list(best.values())

    if not policy_valid:
        for r in retained:
            rejected_rows.append({"id": r["id"], "reasonCodes": ["POLICY_INVALID"]})
        retained = []
    else:
        keep = []
        for r in retained:
            dt = parse_ts(r["eventTime"])
            if min_t <= dt <= max_t:
                keep.append(r)
            else:
                rejected_rows.append({"id": r["id"], "reasonCodes": ["OUT_OF_WINDOW"]})
        retained = keep

    train, validation, test = [], [], []

    for r in retained:
        bucket = hashlib.sha256(r["entity"].encode("utf-8")).digest()[0] % 10
        if bucket <= 5:
            train.append(r)
        elif bucket <= 7:
            validation.append(r)
        else:
            test.append(r)

    train_sets = [words(x["text"]) for x in train]

    def filter_contamination(rows):
        keep = []
        for r in rows:
            s = words(r["text"])
            contaminated = False
            for t in train_sets:
                j = 1 if (not s and not t) else len(s & t) / len(s | t)
                if j >= th:
                    rejected_rows.append(
                        {"id": r["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]}
                    )
                    contaminated = True
                    break
            if not contaminated:
                keep.append(r)
        return keep

    validation = filter_contamination(validation)
    test = filter_contamination(test)

    dtrain, train = digest(train)
    dval, validation = digest(validation)
    dtest, test = digest(test)

    rejected_objects = sorted(
        rejected_objects,
        key=lambda x: ((x["uri"] or "").encode(), json.dumps(x, separators=(",", ":")).encode()),
    )
    rejected_rows = sorted(
        rejected_rows,
        key=lambda x: (x["id"].encode(), json.dumps(x, separators=(",", ":")).encode()),
    )
    lineage = sorted(lineage, key=lambda x: x["uri"].encode())

    return {
        "splits": {
            "train": train,
            "validation": validation,
            "test": test,
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": {
            "train": dtrain,
            "validation": dval,
            "test": dtest,
        },
        "lineage": lineage,
    }
