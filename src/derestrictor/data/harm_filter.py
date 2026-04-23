"""Harm filter: route truly-harmful prompts out of the derestrict pool.

Applied to the ``derestrict`` split of ``RevivifAI/derestriction`` during
build. Rows that match any filter category are *moved* to the ``restrict``
split rather than being silently dropped, so the final dataset preserves a
faithful record of the restricted content too.

Runs two stages per category:

1. Broad keyword regex (word-boundary aware).
2. Semantic cosine similarity against category anchor prompts, using
   ``sentence-transformers/all-MiniLM-L6-v2``.

Five categories are filtered:

- ``child`` -- prompts that mention, target, or involve children, teenagers,
  or students.
- ``sexual`` -- prompts that request graphic sexual acts, pornography, erotica,
  non-consensual / incest / bestial / necrophilic content.
- ``violence_extreme`` -- prompts requesting targeted assassination, murder,
  torture, kidnap-and-harm, or terrorist planning against specific people or
  venues.
- ``weapons_mass_destruction`` -- prompts requesting construction /
  acquisition / deployment of bombs, biological / chemical / nuclear /
  radiological weapons, or mass-casualty devices.
- ``self_harm`` -- prompts encouraging or instructing suicide, self-injury,
  or eating-disorder self-harm.

The filter is intentionally aggressive -- we err heavily on the side of
moving borderline prompts into ``restrict`` rather than letting genuinely
unsafe prompts live on in the ``derestrict`` or ``allow`` pools.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer

_CHILD_KEYWORD_PATTERNS: list[str] = [
    r"child(?:ren)?",
    r"\bkid\b",
    r"\bkids\b",
    r"\bminor\b",
    r"\bminors\b",
    r"underage",
    r"toddler",
    r"toddlers",
    r"infant",
    r"infants",
    r"\bbaby\b",
    r"\bbabies\b",
    r"preschool",
    r"kindergarten",
    r"elementary[- ]school",
    r"\bpupil\b",
    r"\bpupils\b",
    r"\bteen\b",
    r"\bteens\b",
    r"teenager",
    r"teenagers",
    r"\byouth\b",
    r"\byouths\b",
    r"juvenile",
    r"juveniles",
    r"adolescent",
    r"adolescents",
    r"\bstudent\b",
    r"\bstudents\b",
    r"schoolchild(?:ren)?",
    r"\bboy\b",
    r"\bboys\b",
    r"\bgirl\b",
    r"\bgirls\b",
    r"\bson\b",
    r"\bsons\b",
    r"\bdaughter\b",
    r"\bdaughters\b",
    r"pedophil(?:e|es|ia|ic)",
    r"paedophil(?:e|es|ia|ic)",
    r"\bCSAM\b",
    r"grooming",
    r"pre[- ]teen",
    r"pre[- ]teens",
]

_CHILD_SEMANTIC_ANCHORS: list[str] = [
    "harming or endangering a child",
    "sexual content involving minors",
    "targeting schoolchildren or students",
    "exploiting a teenager or young person",
    "abducting a child",
    "grooming a minor online",
    "teaching a child to do something dangerous or illegal",
]


_SEXUAL_KEYWORD_PATTERNS: list[str] = [
    r"pornograph(?:y|ic|ically|er|ers)",
    r"\bporn\b",
    r"\berotica?\b",
    r"\berotically\b",
    r"\bsmut(?:ty)?\b",
    r"\bhentai\b",
    r"\blolicon\b",
    r"\bnude\b",
    r"\bnudity\b",
    r"\bnaked\b",
    r"masturbat(?:e|es|ed|ing|ion|ory)",
    r"\borgasm(?:s|ic|ed|ing)?\b",
    r"\bejaculat(?:e|es|ed|ing|ion)\b",
    r"\bclimax(?:ed|ing|es)?\b",
    r"\bgenital(?:s|ia)?\b",
    r"\bpenis(?:es)?\b",
    r"\bvagina(?:l|s)?\b",
    r"\bvulva(?:s)?\b",
    r"\bclitoris\b",
    r"\btesticl(?:e|es)\b",
    r"\bscrotum\b",
    r"\bnipples?\b",
    r"\bbreasts?\b",
    r"\bbutthole\b",
    r"\banus\b",
    r"\banal\s+(?:sex|intercourse|penetration)\b",
    r"\bintercourse\b",
    r"\bcoitus\b",
    r"\bcopulat(?:e|es|ed|ing|ion)\b",
    r"\bfornicat(?:e|es|ed|ing|ion)\b",
    r"\bfellatio\b",
    r"\bcunnilingus\b",
    r"\banilingus\b",
    r"\bblow[-\s]?job(?:s)?\b",
    r"\bhand[-\s]?job(?:s)?\b",
    r"\bdeepthroat(?:ing|ed)?\b",
    r"\brimming\b",
    r"\brimjob(?:s)?\b",
    r"\bcum(?:ming|med|shot|shots)?\b",
    r"\bsemen\b",
    r"\bsperm\b",
    r"\bBDSM\b",
    r"\bbondage\b",
    r"\bfetish(?:es|istic|ism)?\b",
    r"\bkink(?:y|s)?\b",
    r"\bsex[-\s]?toy(?:s)?\b",
    r"\bdildo(?:s)?\b",
    r"\bvibrator(?:s)?\b",
    r"\bbutt[-\s]?plug(?:s)?\b",
    r"\bsex\s+scene(?:s)?\b",
    r"\bsex(?:ually)?\s+explicit\b",
    r"\bsexually\s+graphic\b",
    r"\bgraphic\s+sex(?:ual)?\b",
    r"\bsex\s+act(?:s)?\b",
    r"\bsexual\s+act(?:s)?\b",
    r"\bsexual\s+encounter(?:s)?\b",
    r"\bsexual\s+fantasy\b",
    r"\bsexual\s+fantasies\b",
    r"\bsexual\s+role[-\s]?play(?:ing)?\b",
    r"\berotic\s+role[-\s]?play(?:ing)?\b",
    r"\bhave\s+sex\b",
    r"\bhaving\s+sex\b",
    r"\bmake\s+love\b",
    r"\bmaking\s+love\b",
    r"\bget\s+laid\b",
    r"\bfuck(?:ed|ing|s|er|ers)?\b",
    r"\brap(?:e|ed|es|ing|ist|ists)\b",
    r"\bmolest(?:er|ers|ed|ing|ation)\b",
    r"\bincest(?:uous)?\b",
    r"\bbestial(?:ity)?\b",
    r"\bzoophil(?:ia|e|es|ic)\b",
    r"\bnecrophil(?:ia|e|es|ic)\b",
    r"\bprostitut(?:e|es|ion|ing)\b",
    r"\bsex[-\s]?work(?:er|ers|ing)\b",
    r"\bescort\s+service(?:s)?\b",
    r"\bstrip(?:per|pers|tease|club|clubs)\b",
    r"\bcamgirl(?:s)?\b",
    r"\bOnlyFans\b",
]

_SEXUAL_SEMANTIC_ANCHORS: list[str] = [
    "write pornographic or erotic fiction",
    "describe a graphic sex scene in explicit detail",
    "produce sexually explicit material",
    "generate adult pornography",
    "describe two people having sex",
    "write a detailed depiction of sexual acts",
    "create non-consensual sexual content",
    "narrate a rape scene",
    "roleplay an explicit sexual encounter",
]


# Targeted interpersonal violence, torture, and terrorist planning.
# Deliberately narrower than "violence" -- hacking, self-defense, weapons
# training, and fictional combat stay in the derestrict pool.
_VIOLENCE_KEYWORD_PATTERNS: list[str] = [
    r"assassinat(?:e|es|ed|ing|ion|or|ors)",
    r"\bmurder(?:s|ed|ing|er|ers)?\b",
    r"\bkill\s+(?:a|the|my|his|her|their|an|someone|people|person)\b",
    r"\bexecut(?:e|ion)\s+(?:a|the|my|his|her|their|someone|the\s+president|prisoner(?:s)?)\b",
    r"\bmass\s+shooting\b",
    r"\bschool\s+shooting\b",
    r"\bmass\s+casualt(?:y|ies)\b",
    r"\bterror(?:ist|ism|istic)\b",
    r"\bhostage(?:s)?\b",
    r"\bkidnap(?:s|ped|ping|per|pers)?\b",
    r"\babduct(?:s|ed|ing|ion|or|ors)?\b",
    r"\btortur(?:e|es|ed|ing|er|ers)\b",
    r"\bdismember(?:s|ed|ing|ment)?\b",
    r"\bbehead(?:s|ed|ing)?\b",
    r"\bdecapitat(?:e|es|ed|ing|ion)\b",
    r"\blynch(?:es|ed|ing|ings)?\b",
    r"\bgenocid(?:e|al)\b",
    r"\bethnic\s+cleansing\b",
    r"\bhit[-\s]?man\b",
    r"\bcontract\s+killing\b",
    r"\bdeath\s+threat(?:s)?\b",
    r"\bhit[-\s]?squad\b",
    r"\bmaim(?:s|ed|ing)?\b",
    r"\bstab\s+(?:a|the|someone|him|her|them)\b",
    r"\bpoison\s+(?:a|the|someone|him|her|them)\b",
    r"\bshoot\s+(?:up|down|a|the|someone|him|her|them)\b",
]

_VIOLENCE_SEMANTIC_ANCHORS: list[str] = [
    "how to assassinate a public figure",
    "plan a murder of a specific person",
    "instructions for carrying out a terrorist attack",
    "how to kidnap someone and hold them hostage",
    "how to torture a captive",
    "plan a mass shooting at a specific venue",
    "execute a targeted killing of a named individual",
    "carry out ethnic cleansing or genocide",
]


# Weapons capable of mass casualties. Conventional firearms ownership,
# legal hunting, and general gun safety questions stay in derestrict.
_WMD_KEYWORD_PATTERNS: list[str] = [
    r"\bbio(?:logical)?\s+(?:weapon|weapons|agent|agents|warfare)\b",
    r"\bchem(?:ical)?\s+(?:weapon|weapons|agent|agents|warfare)\b",
    r"\bnuclear\s+(?:weapon|weapons|warhead|warheads|bomb|bombs|device|device|arsenal)\b",
    r"\bradiological\s+(?:weapon|weapons|device|devices|bomb|bombs|dispersal)\b",
    r"\bdirty\s+bomb(?:s)?\b",
    r"\bnerve\s+agent(?:s)?\b",
    r"\bVX\s+gas\b",
    r"\bsarin\b",
    r"\bnovichok\b",
    r"\btabun\b",
    r"\bsoman\b",
    r"\bmustard\s+gas\b",
    r"\bchlorine\s+gas\s+attack\b",
    r"\bphosgene\b",
    r"\banthrax\b",
    r"\bricin\b",
    r"\bbotulinum(?:\s+toxin)?\b",
    r"\bsmallpox\s+(?:weapon|outbreak|release)\b",
    r"\bweaponiz(?:e|es|ed|ing|ation)\s+(?:a|the|virus|bacteri(?:a|um)|pathogen)\b",
    r"\bIED\b",
    r"\bimprovised\s+explosive\b",
    r"\bpipe\s+bomb(?:s)?\b",
    r"\bpressure[-\s]?cooker\s+bomb(?:s)?\b",
    r"\bcar\s+bomb(?:s)?\b",
    r"\bsuicide\s+(?:vest|belt|bomber|bombing)\b",
    r"\bexplosive\s+device(?:s)?\b",
    r"\bhigh\s+explosive(?:s)?\b",
    r"\bdetonator(?:s)?\b",
    r"\bblasting\s+cap(?:s)?\b",
    r"\bC-?4\s+(?:explosive|charge|charges)\b",
    r"\bTNT\s+(?:charge|charges|explosive)\b",
    r"\bRDX\s+(?:explosive|charge)\b",
    r"\bsemtex\b",
    r"\bTATP\b",
    r"\bHMTD\b",
    r"\bammonium\s+nitrate\s+bomb\b",
    r"\bthermobaric\b",
    r"\bcluster\s+munition(?:s)?\b",
    r"\bland\s+mine(?:s)?\b",
    r"\bgrenade\s+launcher(?:s)?\b",
    r"\bRPG(?:s)?\b",
    r"\bshoulder[-\s]?fired\s+missile(?:s)?\b",
    r"\bman[-\s]?portable\s+air[-\s]?defense\b",
    r"\bMANPADS?\b",
    r"\benrich(?:ed|ing|ment)\s+uranium\b",
    r"\bplutonium\s+(?:core|pit|device)\b",
    r"\bfissile\s+material\b",
    r"\bgun[-\s]?type\s+fission\b",
    r"\bimplosion[-\s]?type\s+(?:device|weapon)\b",
]

_WMD_SEMANTIC_ANCHORS: list[str] = [
    "how to synthesize a nerve agent like sarin",
    "instructions for building a nuclear weapon",
    "step-by-step guide to making a dirty bomb",
    "how to weaponize a biological pathogen",
    "how to build a pipe bomb or improvised explosive device",
    "synthesize ricin or botulinum toxin for an attack",
    "instructions for producing chemical weapons at home",
    "how to acquire enriched uranium or plutonium",
]


# Encouragement or instruction of suicide and self-injury. Clinical
# discussion, prevention, and harm-reduction framing stay in derestrict.
_SELF_HARM_KEYWORD_PATTERNS: list[str] = [
    r"\bsuicid(?:e|es|al|ality)\s+(?:method(?:s)?|technique(?:s)?|instruction(?:s)?|guide(?:s)?|plan(?:s)?|attempt(?:s)?)\b",
    r"\bcommit\s+suicide\b",
    r"\bkill\s+(?:myself|yourself|himself|herself|oneself|themselves)\b",
    r"\bend\s+(?:my|your|his|her|their|one's|ones)\s+(?:own\s+)?life\b",
    r"\bhang\s+(?:myself|yourself|himself|herself|oneself|themselves)\b",
    r"\bslit\s+(?:my|your|his|her|their|one's|ones)\s+(?:own\s+)?(?:wrists?|throat)\b",
    r"\bself[-\s]?harm(?:ing|er|ers)?\b",
    r"\bself[-\s]?injur(?:y|ies|ious)\b",
    r"\bself[-\s]?mutilat(?:e|es|ed|ing|ion)\b",
    r"\bcutt(?:ing|er|ers)\s+(?:myself|yourself|himself|herself|oneself|themselves|my|your|his|her|their)\s+(?:own\s+)?(?:arm|arms|wrist|wrists|leg|legs|skin|thigh|thighs|body)\b",
    r"\bpro[-\s]?ana\b",
    r"\bpro[-\s]?mia\b",
    r"\bthinspo(?:ration)?\b",
    r"\bbinge\s+and\s+purge\b",
    r"\boverdose\s+on\s+(?:pills?|medication|drugs?)\s+to\s+(?:die|kill)\b",
    r"\blethal\s+dose(?:s)?\s+of\s+(?:pills?|medication|drugs?|poison)\b",
]

_SELF_HARM_SEMANTIC_ANCHORS: list[str] = [
    "encourage someone to commit suicide",
    "give step-by-step instructions for killing oneself",
    "describe the most painless way to die by suicide",
    "instruct someone on how to cut themselves without getting caught",
    "promote anorexia or bulimia as a lifestyle",
    "list a lethal dose of medication for a suicide attempt",
    "help someone plan a suicide",
]


# Category registry: each entry is (category_name, keyword_patterns, anchor_texts).
_CATEGORIES: tuple[tuple[str, list[str], list[str]], ...] = (
    ("child", _CHILD_KEYWORD_PATTERNS, _CHILD_SEMANTIC_ANCHORS),
    ("sexual", _SEXUAL_KEYWORD_PATTERNS, _SEXUAL_SEMANTIC_ANCHORS),
    ("violence_extreme", _VIOLENCE_KEYWORD_PATTERNS, _VIOLENCE_SEMANTIC_ANCHORS),
    ("weapons_mass_destruction", _WMD_KEYWORD_PATTERNS, _WMD_SEMANTIC_ANCHORS),
    ("self_harm", _SELF_HARM_KEYWORD_PATTERNS, _SELF_HARM_SEMANTIC_ANCHORS),
)


@dataclass
class FilterDrop:
    """Record of a prompt moved out of a split by the filter, for the audit log."""

    prompt: str
    source: str
    stage: str  # "keyword" or "semantic"
    category: str  # which filter category fired
    trigger: str  # matching keyword or top anchor text
    score: float  # 1.0 for keyword hits, cosine sim for semantic hits


@dataclass
class FilterResult:
    """Return value of :meth:`HarmFilter.filter`.

    - ``kept`` is every input row the filter did not touch.
    - ``restricted`` is every input row the filter matched, each augmented
      with an ``Restriction_Category`` field carrying the category name
      that fired. These rows are intended to be appended to the
      ``restrict`` split of the dataset.
    """

    kept: list[dict]
    restricted: list[dict]


class HarmFilter:
    """Callable filter that separates genuinely-harmful prompts from the pool.

    Runs every configured category in sequence. Each category has its own
    keyword regex and semantic anchors but shares the same embedder for
    efficiency. Prompts that match any category are removed from the kept
    pool and re-emitted as ``restricted`` rows so the caller can route
    them into the ``restrict`` split.
    """

    def __init__(
        self,
        semantic_threshold: float = 0.55,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.semantic_threshold = semantic_threshold
        self.embedder_name = embedder_name
        self._embedder = None
        self._keyword_res: dict[str, re.Pattern[str]] = {
            name: re.compile("|".join(patterns), re.IGNORECASE) for name, patterns, _ in _CATEGORIES
        }
        self._anchor_texts: dict[str, list[str]] = {name: anchors for name, _, anchors in _CATEGORIES}
        self._anchor_embs: dict[str, np.ndarray] = {}
        self.drops: list[FilterDrop] = []
        self.semantic_scores: list[float] = []

    def _load_embedder(self) -> None:
        if self._embedder is not None:
            return
        self._embedder = SentenceTransformer(self.embedder_name)
        for name, anchors in self._anchor_texts.items():
            self._anchor_embs[name] = self._embedder.encode(
                anchors,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

    @staticmethod
    def _keyword_hit(text: str, regex: re.Pattern[str]) -> str | None:
        m = regex.search(text)
        return m.group(0) if m else None

    def _partition_one_category(
        self,
        rows: list[dict],
        category: str,
    ) -> tuple[list[dict], list[dict]]:
        """Partition ``rows`` for a single category into (kept, restricted)."""
        regex = self._keyword_res[category]
        kept: list[dict] = []
        restricted: list[dict] = []
        needs_semantic: list[tuple[int, dict]] = []

        for row in rows:
            prompt = row["Prompt"]
            hit = self._keyword_hit(prompt, regex)
            if hit is not None:
                self.drops.append(
                    FilterDrop(
                        prompt=prompt,
                        source=row["Source"],
                        stage="keyword",
                        category=category,
                        trigger=hit,
                        score=1.0,
                    )
                )
                restricted.append({**row, "Restriction_Category": category})
                continue
            needs_semantic.append((len(kept), row))
            kept.append(row)

        if not kept:
            return kept, restricted

        self._load_embedder()
        anchor_emb = self._anchor_embs[category]
        prompts = [r["Prompt"] for _, r in needs_semantic]
        emb = self._embedder.encode(
            prompts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=True,
        )
        sims = emb @ anchor_emb.T  # (n, n_anchors)
        max_sims = sims.max(axis=1)
        max_idx = sims.argmax(axis=1)
        self.semantic_scores.extend(max_sims.tolist())

        keep_mask = [True] * len(kept)
        anchor_texts = self._anchor_texts[category]
        for (kept_idx, row), top_sim, anchor_idx in zip(needs_semantic, max_sims, max_idx, strict=False):
            if top_sim >= self.semantic_threshold:
                keep_mask[kept_idx] = False
                self.drops.append(
                    FilterDrop(
                        prompt=row["Prompt"],
                        source=row["Source"],
                        stage="semantic",
                        category=category,
                        trigger=anchor_texts[int(anchor_idx)],
                        score=float(top_sim),
                    )
                )
                restricted.append({**row, "Restriction_Category": category})

        kept = [r for r, keep in zip(kept, keep_mask, strict=False) if keep]
        return kept, restricted

    def filter(self, rows: Iterable[dict]) -> FilterResult:
        """Partition a list of ``{'Prompt': str, 'Source': str}`` rows.

        Applies every configured category in order. Matched rows are
        returned in ``result.restricted`` (annotated with the firing
        category) instead of silently dropped. Audit records live on
        ``self.drops`` for :meth:`write_audit_log`.
        """
        kept = list(rows)
        restricted: list[dict] = []
        for category, _, _ in _CATEGORIES:
            kept, moved = self._partition_one_category(kept, category)
            restricted.extend(moved)
        return FilterResult(kept=kept, restricted=restricted)

    def write_audit_log(self, drops_path: Path, histogram_path: Path) -> None:
        """Write per-drop JSONL and a semantic-score histogram for calibration."""
        drops_path.parent.mkdir(parents=True, exist_ok=True)
        with drops_path.open("w", encoding="utf-8") as f:
            for d in self.drops:
                f.write(
                    json.dumps(
                        {
                            "prompt": d.prompt,
                            "source": d.source,
                            "stage": d.stage,
                            "category": d.category,
                            "trigger": d.trigger,
                            "score": d.score,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if self.semantic_scores:
            scores = np.asarray(self.semantic_scores, dtype=float)
            bin_edges = np.linspace(0.0, 1.0, 21)
            counts, _ = np.histogram(scores, bins=bin_edges)
            histogram = {
                "threshold": self.semantic_threshold,
                "n_scored": int(scores.size),
                "mean": float(scores.mean()),
                "max": float(scores.max()),
                "min": float(scores.min()),
                "bins": [
                    {"lo": float(lo), "hi": float(hi), "count": int(c)}
                    for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], counts, strict=False)
                ],
            }
            histogram_path.parent.mkdir(parents=True, exist_ok=True)
            histogram_path.write_text(json.dumps(histogram, indent=2, ensure_ascii=False), encoding="utf-8")
