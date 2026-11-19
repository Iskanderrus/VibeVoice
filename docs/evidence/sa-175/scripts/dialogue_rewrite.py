"""Prototype (non-CFOS) stage: rewrite the CFOS-068 Black Swan narration as a two-host (RED/BLUE) podcast.

Uses the CFOS Claude Code subscription adapter (API key scrubbed from the child environment, tools/MCP off).
Sends ONLY the four scene narrations (no source excerpts, no policy, no personal data).
Deterministic checks decide acceptance; one retry; no manual editing of the output.
"""
import json, re, sys
from pathlib import Path
from content_factory_os.adapters.claude_code.creative_text import ClaudeCodeSubscriptionCreativeTextGenerator

SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
scenes = [l[5:].strip() for l in SRC.read_text().splitlines() if l.startswith("RED: ")]
assert len(scenes) == 4, scenes

SYSTEM = (
    "You adapt a finished solo documentary narration into a two-host podcast conversation. "
    "Hosts are RED (curious, asks the questions, voices the audience's doubt) and BLUE (explains, precise, a little dry). "
    "Rules: keep every claim, name, number and example from the source narration and add NO new fact, example, name, "
    "number or quotation. Keep the order and the four scenes. Natural spoken English, short turns (under 35 words), "
    "no stage directions, no sound cues, no markdown. Each scene must have between 4 and 9 turns and the hosts must alternate "
    "most of the time. Total length close to the source (within +30%). "
    "Never use the words welcome, hello, hi, hey or however anywhere, and never begin any turn with So, Well, Okay, Now, Right, "
    "Alright, And or But: begin each turn directly with its content."
)
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["scenes"],
    "properties": {"scenes": {"type": "array", "minItems": 4, "maxItems": 4, "items": {
        "type": "object", "additionalProperties": False, "required": ["turns"],
        "properties": {"turns": {"type": "array", "minItems": 4, "maxItems": 9, "items": {
            "type": "object", "additionalProperties": False, "required": ["speaker", "text"],
            "properties": {"speaker": {"enum": ["RED", "BLUE"]}, "text": {"type": "string", "minLength": 3, "maxLength": 400}}}}}}}},
}
words = lambda s: re.findall(r"[A-Za-z0-9']+", s)
nums = lambda s: set(re.findall(r"\d[\d,.]*", s))

BANNED = re.compile(r"\b(welcome|hello|hi|hey|however)\b", re.I)          # documented BGM triggers (README FAQ Q2)
OPENERS = re.compile(r"^(so|well|okay|ok|now|right|alright|and|but)\b", re.I)  # intro-like openers (project hypothesis)

def validate(doc):
    problems = []
    src_words = sum(len(words(s)) for s in scenes)
    out_words = 0
    for i, sc in enumerate(doc["scenes"]):
        t = sc["turns"]
        for turn in t:
            if BANNED.search(turn["text"]): problems.append(f"scene{i+1}: banned trigger word")
            if OPENERS.match(turn["text"].strip()): problems.append(f"scene{i+1}: intro-like opener")
            if len(words(turn["text"])) > 45: problems.append(f"scene{i+1}: turn over 45 words")
            if re.search(r"[\[\]<>*#_]|\(.*?\)", turn["text"]): problems.append(f"scene{i+1}: markup/stage direction")
            out_words += len(words(turn["text"]))
        if len({x["speaker"] for x in t}) != 2: problems.append(f"scene{i+1}: only one speaker")
        text = " ".join(x["text"] for x in t)
        extra = nums(text) - nums(scenes[i])
        if extra: problems.append(f"scene{i+1}: numbers not in source {sorted(extra)}")
    if not (0.9 * src_words <= out_words <= 1.35 * src_words): problems.append(f"length {out_words} vs source {src_words}")
    return problems, src_words, out_words

gen = ClaudeCodeSubscriptionCreativeTextGenerator(model="claude-sonnet-5-5", timeout_seconds=300)
facts = gen.preflight()
print("preflight:", {k: v for k, v in facts.items() if k != "login_expires_at"})
prompt = "SOURCE NARRATION (four scenes):\n" + "\n".join(f"SCENE {i+1}: {s}" for i, s in enumerate(scenes))
doc = None; log = []
for attempt in range(2):
    p = prompt if attempt == 0 else prompt + "\n\nFix these problems from the previous attempt:\n" + "\n".join(log[-1]["problems"])
    r = gen.generate_json(system=SYSTEM, prompt=p, max_output_tokens=3000, schema=SCHEMA, attempt=attempt)
    cand = json.loads(r.text)
    probs, sw, ow = validate(cand)
    log.append({"attempt": attempt, "model": r.model, "model_version": r.model_version, "billed_cost_usd": r.billed_cost_usd,
                "problems": probs, "src_words": sw, "out_words": ow})
    print(f"attempt {attempt}: problems={probs} words {sw}->{ow}")
    if not probs: doc = cand; break
(OUT / "rewrite_log.json").write_text(json.dumps({"calls": gen.calls, "log": log, "facts": {k: v for k, v in facts.items() if k != "login_expires_at"}}, indent=2))
if doc is None:
    print("REJECTED: no acceptable dialogue"); sys.exit(2)
for i, sc in enumerate(doc["scenes"], 1):
    (OUT / f"scene{i}.txt").write_text("".join(f"{t['speaker']}: {t['text'].strip()}\n" for t in sc["turns"]))
(OUT / "dialogue.json").write_text(json.dumps(doc, indent=2))
print("wrote", OUT)
