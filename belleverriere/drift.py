"""Prompt drift: the prompt wanders slowly away from (and sometimes back toward) what you set.

The prompt is kept as named slots. Each mutation rewrites a few slots with a local LLM
(Ollama). How many slots change and how far they jump follow the `drift` setting; the
`tether` setting is the chance that a slot is pulled back toward your original idea.
The style (stained glass) is never part of the drift, so only the content wanders.
Without Ollama, a word-bank fallback still keeps things moving.
"""
from __future__ import annotations

import json
import re
import random
from dataclasses import dataclass, field

import httpx

SLOTS = ["subject", "action", "setting", "palette", "motif", "mood", "details"]  # details never drifts

SLOT_HELP = {
    "subject": "the main figure(s) or thing depicted",
    "action": "what the subject is doing or its pose",
    "setting": "the surroundings and background",
    "palette": "the dominant glass colors",
    "motif": "the decorative border / recurring ornament",
    "mood": "the feeling or atmosphere",
}  # "details" isn't asked of the model: the app fills it with whatever phrases the model left out

FALLBACK_BANK = {
    "subject": ["a heron", "a crowned stag", "a woman in armor", "a fox", "an owl", "a lion", "a mermaid",
                "a knight on horseback", "two cranes", "a peacock", "a saint with a lantern", "a hare"],
    "action": ["standing tall", "mid-leap", "looking upward", "resting", "holding a lantern aloft",
               "turning to look back", "with wings spread", "kneeling", "walking forward"],
    "setting": ["among tall lilies", "under a crescent moon", "in a walled garden", "on a rocky shore",
                "in a forest of birches", "beneath a starry vault", "in a field of poppies", "by a river"],
    "palette": ["cobalt and gold", "emerald and amber", "ruby, violet and pale blue", "teal and copper",
                "rose and sage", "deep blue with lemon highlights", "amber and burgundy"],
    "motif": ["vines and leaves", "fleur-de-lis", "interlaced knots", "stars", "lotus flowers",
              "oak leaves and acorns", "waves", "sunbursts"],
    "mood": ["serene", "triumphant", "mysterious", "joyful", "solemn", "dreamlike", "watchful"],
}


# Random sparks for "Surprise me": the app picks the ingredients so a small LLM can't fall back
# on the same safe idea every time; the LLM's job is to fuse them into one vivid concept.
SPARKS = {
    "tradition": ["Byzantine mosaic", "Japanese ukiyo-e", "Art Nouveau", "Art Deco", "Celtic illuminated manuscript",
                  "Persian miniature", "Mughal painting", "Russian icon", "Mexican folk art", "Aztec codex",
                  "Ottoman Iznik tile", "Tiffany lamp", "Viking carving", "Moorish geometry", "Mayan relief",
                  "Indian Madhubani", "Ethiopian icon", "Chinese paper cut", "Bauhaus", "1960s psychedelic poster",
                  "Soviet constructivism", "Pre-Raphaelite painting", "Klimt's gold period", "Mucha poster",
                  "medieval bestiary", "Polynesian tapa", "Andean textile", "Victorian botanical plate"],
    "subject": ["bioluminescent fungi", "clockwork birds", "a tide pool at dawn", "migrating cranes", "volcanic glass",
                "a forgotten river goddess", "an orchard at harvest", "constellations come alive", "deep-sea creatures",
                "jeweled beetles", "a lighthouse keeper's dream", "an aurora over ice", "alchemy and its symbols",
                "a circus at the edge of night", "a chess game between seasons", "coral cities", "origami animals",
                "a beekeeper and her swarm", "a desert caravan", "a cathedral of trees", "musical instruments that bloom",
                "a fox spirit", "a cartographer's sea monsters", "moths drawn to a lantern", "a dancer mid-turn",
                "a sleeping giantess made of hills", "storm clouds with faces", "a koi pond seen from above",
                "a woman braiding starlight", "the anatomy of a flower", "a mermaid among shipwreck gold",
                "a hummingbird duel", "an astronomer's tower", "snow leopards on a mountain shrine"],
    "twist": ["two worlds meeting at a seam", "frozen at the instant of change", "seen from directly above",
              "at an impossible scale", "told as three moments in one scene", "mirrored in perfect symmetry",
              "seen through rippling water", "half day and half night", "everything made of one material",
              "growing out of something unexpected", "caught in a spiral", "the small thing is the giant"],
    "mood": ["playful", "eerie but beautiful", "triumphant", "tender", "uncanny", "euphoric", "wistful",
             "mischievous", "serene", "electric", "sacred", "dreamlike"],
}


# "Vary the pose": image models fall back on one default pose unless the prompt names one, and
# a fast model barely varies it between seeds. Each image gets one of these, picked by the engine.
POSES = ["standing with both arms relaxed at the sides", "seated sideways with knees drawn up, arms around them",
         "reclining on one elbow", "seen from behind, looking back over the shoulder",
         "kneeling upright, hands resting in the lap", "mid-stride, walking toward the viewer",
         "dancing, arms flung wide", "leaning back against a column, arms folded",
         "holding a small lantern in both hands", "head bowed, hands clasped at the chest",
         "turned away in profile, hair swept sideways by wind", "lying back with hair fanned out around the head",
         "crouching, one hand touching the ground", "sitting on a ledge, legs dangling",
         "twisting at the waist to look upward", "reaching one hand out toward the viewer",
         "both hands cupped around a glowing light", "standing with hands on hips"]
FRAMINGS = ["close-up of the head and shoulders", "framed from the waist up", "full length, head to toe",
            "small within a wide scene", "seen from a low angle", "seen from slightly above"]



@dataclass
class PromptState:
    original: dict[str, str] = field(default_factory=dict)
    current: dict[str, str] = field(default_factory=dict)
    generation: int = 0
    recent_subjects: list[str] = field(default_factory=list)


def describe(slots: dict[str, str]) -> str:
    """Slots back into a readable idea, for the idea box."""
    body = ", ".join(x for x in (slots.get("subject"), slots.get("action"), slots.get("setting")) if x)
    extras = [f"{k.capitalize()}: {slots[k]}" for k in ("palette", "motif", "mood") if slots.get(k)]
    details = (slots.get("details") or "").strip().rstrip(".")
    return ". ".join(x for x in [body, details, *extras] if x) + "."


def verbatim(style_prompt: str, idea: str, pose: str | None = None) -> str:
    """Drift off: the style and the idea exactly as written (plus a pose, if that switch is on)."""
    text = ", ".join(x.strip().rstrip(",") for x in (style_prompt, idea) if x.strip())
    if pose:
        text = text.rstrip(".") + f". Pose: {pose}."
    return text


def compose(slots: dict[str, str], style_prompt: str, time_hint: str | None = None, frame: str = "none",
            pose: str | None = None, details: bool = True) -> str:
    from .frames import FRAME_SHAPES
    shape = FRAME_SHAPES.get(frame)
    parts = [style_prompt.strip().rstrip(".") + "."] if style_prompt.strip() else []
    body = ", ".join(x for x in (slots.get("subject"), slots.get("action"), slots.get("setting")) if x)
    if body:
        parts.append(body[0].upper() + body[1:] + ".")
    if pose:
        parts.append(f"Pose: {pose}.")
    if details and slots.get("details"):  # the rest of what the owner wrote, as written
        parts.append(slots["details"].strip().rstrip(".") + ".")
    if shape:  # the app draws the frame; asking for a border makes the model paint a rectangular one
        parts.append(f"The artwork is shaped as {shape} that fills the entire image edge to edge, "
                     "with no wall or stonework around it.")
        if slots.get("motif"):
            parts.append(f"Ornamental details of {slots['motif']} woven through the glass.")
    else:
        parts.append("The glass fills the entire image edge to edge.")
        if slots.get("motif"):  # not "border": asking for one makes the model paint a frame
            parts.append(f"Ornamental details of {slots['motif']} woven through the glass.")
    pal = slots.get("palette")
    if pal:
        parts.append(f"Glass colors: {pal}" + (f", {time_hint}" if time_hint else "") + ".")
    elif time_hint:
        parts.append(f"Glass colors: {time_hint}.")
    if slots.get("mood"):
        parts.append(f"Mood: {slots['mood']}.")
    return " ".join(parts)


def _with_dropped_phrases(idea: str, slots: dict[str, str]) -> str:
    """Safety net: any phrase of the idea whose words didn't make it into the parts (a model may quietly
    drop words) is added to details exactly as written, so nothing the owner wrote is lost."""
    kept = " ".join(slots.values()).lower()
    details = ""
    for phrase in re.split(r"[,.;\n]+", idea):
        words = [w for w in re.findall(r"[a-z]+", phrase.lower()) if len(w) > 3]
        if phrase.strip() and any(w not in kept for w in words) and phrase.strip().lower() not in details.lower():
            details = f"{details}, {phrase.strip()}" if details else phrase.strip()
    return details


class Drifter:
    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.last_error: str | None = None

    # -- LLM plumbing ------------------------------------------------------------
    def _chat(self, system: str, user: str, temperature: float) -> dict | None:
        try:
            body = {
                "model": self.model, "stream": False, "format": "json",
                "options": {"temperature": temperature},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            r = httpx.post(f"{self.ollama_url}/api/chat", timeout=120, json=body)
            if r.status_code == 404:  # chosen model isn't installed: use whatever is
                installed = self.list_models()
                if installed:
                    print(f"[drift] model '{self.model}' not installed; using '{installed[0]}'")
                    body["model"] = installed[0]
                    r = httpx.post(f"{self.ollama_url}/api/chat", timeout=120, json=body)
            r.raise_for_status()
            data = json.loads(r.json()["message"]["content"])
            self.last_error = None
            return data if isinstance(data, dict) else None
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"[drift] LLM unavailable, using fallback ({self.last_error})")
            return None

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.ollama_url}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    # -- public API ----------------------------------------------------------------
    @staticmethod
    def _rules(guidance: str) -> str:
        return f" Always follow these rules from the window's owner: {guidance.strip()}" if guidance.strip() else ""

    def decompose(self, base_prompt: str, guidance: str = "") -> dict[str, str]:
        """Split the user's idea into slots, filling gaps with fitting choices."""
        system = ("You help design stained glass windows. Split the user's idea into these fields. Copy the "
                  "user's own words into the fields exactly: do not paraphrase, shorten, or drop any words, "
                  "including words you don't recognize such as trigger words or names. Only fill fields the "
                  "user left out with something fitting (2-12 words). "
                  "Reply with JSON only, with exactly these keys: " +
                  ", ".join(f'"{k}" ({v})' for k, v in SLOT_HELP.items()) + self._rules(guidance))
        data = self._chat(system, f"Idea: {base_prompt}", 0.4)
        slots = {k: str(data.get(k, "")).strip() for k in SLOTS if k != "details"} if data else {}
        if slots.get("subject"):
            slots["details"] = _with_dropped_phrases(base_prompt, slots)
        if not slots.get("subject"):
            slots = {k: "" for k in SLOTS}
            slots["subject"] = base_prompt
            rnd = random.Random(base_prompt)
            for k in ("setting", "palette", "motif", "mood"):
                slots[k] = rnd.choice(FALLBACK_BANK[k])
        return slots

    def invent(self, rnd: random.Random, avoid: list[str]) -> dict[str, str]:
        """A fresh, unusual idea plus drift rules for "Surprise me". Returns {"idea", "rules"}."""
        sparks = {k: rnd.choice(v) for k, v in SPARKS.items()}
        system = ("You invent concepts for a backlit stained glass window whose picture slowly drifts over hours. "
                  "Fuse the sparks you are given into ONE specific, vivid, surprising idea: name concrete subjects, "
                  "what they are doing, the setting and the colors. Never be generic or predictable: no plain "
                  "landscapes, sunsets, generic angels, roses or peacocks. The scene must be bright and full of "
                  "light, never dark, dim or at night. Do not mention stained glass, lead or "
                  "windows; that style is applied separately. Also write 1-3 short rules for how the picture may "
                  "drift so it stays coherent and interesting: what must stay, what may wander, and keep it "
                  "luminous and colorful because sunlight shines through it. Reply with JSON only: "
                  '{"idea": "1-2 sentences", "rules": "1-3 short sentences"}')
        user = (f"Sparks: {json.dumps(sparks)}\n" +
                (f"Ideas already used, make something clearly different: {avoid[-8:]}\n" if avoid else ""))
        data = self._chat(system, user, temperature=1.1)
        idea = str((data or {}).get("idea", "")).strip()
        if idea:
            return {"idea": idea, "rules": str(data.get("rules", "")).strip()}
        return {"idea": f"{sparks['subject']}, {sparks['twist']}, in the manner of {sparks['tradition']}, "
                        f"{sparks['mood']}", "rules": ""}

    def mutate(self, state: PromptState, drift: float, tether: float, rnd: random.Random,
               allowed: list[str] | None = None, guidance: str = "") -> dict[str, str]:
        drift = max(0.0, min(1.0, drift))
        n_change = 1 + int(round(drift * 2.5))            # 1..3 slots (sometimes subject)
        candidates = [s for s in SLOTS if s != "details" and (allowed is None or s in allowed)]
        if "subject" in candidates and len(candidates) > 1 and drift < 0.5 and rnd.random() > drift * 1.5:
            candidates.remove("subject")                  # small drift: usually leave the subject alone
        if not candidates:
            return dict(state.current)
        to_change = rnd.sample(candidates, k=min(n_change, len(candidates)))
        pull_back = [s for s in to_change if rnd.random() < tether and state.original.get(s)]

        if drift < 0.25:
            size = "a subtle variation: a related detail, a neighboring color, a slightly different pose"
        elif drift < 0.6:
            size = "a noticeable but natural step: something related by theme, myth, nature or association"
        else:
            size = "a bold, surprising leap: unexpected but still beautiful as a stained glass window"

        system = ("You slowly evolve the content of a stained glass window that changes over hours, like a "
                  "dream drifting from one image to the next. Each step you change only the fields you are "
                  "told to, and copy the others unchanged, word for word. Keep each field short (2-12 words), and "
                  "keep any words you don't recognize, such as trigger words or names, exactly as written. "
                  "Reply with JSON only, keys: " + ", ".join(SLOTS) +
                  self._rules(guidance))
        user = (f"Original idea (the anchor): {json.dumps(state.original)}\n"
                f"Current window: {json.dumps(state.current)}\n"
                f"Recent subjects, avoid repeating them: {state.recent_subjects[-6:]}\n"
                f"Change these fields: {to_change}. Make {size}.\n" +
                (f"For these fields, move back toward the original idea instead: {pull_back}.\n" if pull_back else ""))
        data = self._chat(system, user, temperature=0.6 + drift * 0.6)

        new = dict(state.current)
        if data:
            for k in to_change:
                v = str(data.get(k, "")).strip()
                if v:
                    new[k] = v
        else:  # fallback: word bank + tether
            for k in to_change:
                new[k] = state.original[k] if k in pull_back else rnd.choice(FALLBACK_BANK[k])
        return new
