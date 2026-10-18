"""Prompt construction. Few-shot examples use invented rotas and phrasing that is deliberately
NOT drawn from the visible items (dev/check_example_overlap.py enforces that)."""
import random
import zlib

RULES = """You convert shift-note lines into structured scheduling facts. Reply with one output line per note and nothing else.

Output line format:  <note number>|<RELATION>|<arguments>

Relations (arguments separated by |):
AT|person|time           the person works that time block
NOT_AT|person|time       the person does not work that time block
ON|person|station        the person is the one on that station
NOT_ON|person|station    the person is not the one on that station
BEFORE|x|y               x's block is earlier in the day than y's block (a gap is allowed)
RIGHT_BEFORE|x|y         x's block is the block immediately before y's block (x, then y, nothing in between)
BETWEEN|p|q|r            p's block is strictly between q's block and r's block, in either order
NONE                     the note states no current scheduling fact
UNSURE                   the note asserts a current scheduling fact, but you cannot map it safely

In BEFORE / RIGHT_BEFORE / BETWEEN each of x, y, p, q, r is either a person's name or @station, meaning whoever is on that station (for example @packing).
BEFORE means strictly earlier and permits any gap. RIGHT_BEFORE means adjacent blocks with no block in between; require explicit adjacency wording such as "directly", "immediately", "back to back", "straight after", or "no block in between". Do not infer adjacency from a generic handover, relief, or "takes over" phrase. For an explicitly immediate handover where B follows A, output RIGHT_BEFORE|A|B. For a strict later-than relation where B is later than A but adjacency is not stated, output BEFORE|A|B. In "by the time B starts, A has already worked", output BEFORE|A|B. In "A is after one of B and C and before the other", output BETWEEN|A|B|C.
Direction check: write order constraints in chronological order, earlier endpoint first, even when the sentence names the later endpoint first. Thus "X is later than Y" means BEFORE|Y|X; "Y takes over from X later in the day" means BEFORE|X|Y unless the note explicitly says the transition is immediate; "by the time X starts, Y has already worked" means BEFORE|Y|X. For BETWEEN, the first operand is the middle person/reference; only the final two endpoints may be swapped.
Use names, times and stations exactly as they appear in the rota facts.
Each output line must be supported by its numbered note alone. Ground every operand in that same note: AT/NOT_AT needs its person and time there; ON/NOT_ON needs its person and station there; BEFORE/RIGHT_BEFORE/BETWEEN needs every person or @station endpoint there. In ordering relations the earlier reference is first. In BETWEEN the first person/reference is the middle one and the final two endpoints are interchangeable. For ON/NOT_ON use person then station; use @station in ordering relations only when the note refers to that station's holder. Use rota facts only to validate or normalize entity spelling. Never borrow a missing operand from another note or infer an unstated time, person, or station.

For every numbered note, decide independently: (1) does it assert a current rota fact, rather than a wish, historical fact, hypothetical, uncertainty about whether it was recorded, or unrelated detail? (2) if yes, which single supported relation and correctly typed operands express it? (3) are all operands explicitly grounded in this note? Emit exactly one line for every note, in ascending index order. Use NONE only when there is no supported current scheduling relation. If a current relation is asserted but you cannot identify its relation or operands without guessing, emit UNSURE for that note; UNSURE is an explicit extraction failure, not a constraint and not NONE. Do not omit a note, invent facts, or copy example entities.
Hedges such as "I'm fairly sure", "as far as I recall" or "pretty sure" do NOT weaken a note: the fact still holds, so still output its relation.
A restatement prefix ("worth repeating", "noted twice") is only decoration: read the statement after it."""

FILLER_RULES = """Output NONE for a note that does not assert who works which block or station right now. That includes notes about:
- the past ("last month", "in the spring", "the previous cycle", "the old arrangement");
- wishes, requests or applications that were refused or are still open ("lobbied", "asked to move", "wanted ... and did not get it");
- hypotheticals and opinions ("if ... had", "would have been", "a better fit");
- doubts or open questions ("nobody could remember", "was left open", "some disagreement");
- anything unrelated to the rota (supplies, rooms, deliveries, training, surveys, who keeps a key).
A note that merely mentions a person together with a time or station is NONE unless it clearly says that person works or holds it now."""

COMPLETENESS_RULES = """Completeness self-check before answering: reread every note you marked NONE and check whether it contains an explicit current scheduling cue. In particular, check strict-between language ("between", "sandwiched"), adjacency language ("back to back", "straight after", "directly", "immediately", "no gap"), temporal ordering ("earlier", "later", "before", "after", "precedes"), a person/block exclusion (a named person negated for a time block), and station assignment/exclusion or station-holder ordering. If such a cue states a current rota fact, encode that relation; if its operands cannot be safely mapped from that same note, use UNSURE rather than NONE. This is a check for omissions, not permission to infer unstated relations. Do not treat a wish, historical note, hypothetical, open question, or unrelated mention as a current relation just because it uses one of these words."""

# Each example set: (rota facts, [(note text, expected output tail)]).
_SETS = [
    ("People: Hana, Viktor, Zoe, Mateo, Imani\nBlocks (earliest to latest): 07:00, 09:00, 11:00, 13:00, 15:00\n"
     "Stations: intake, calibration, packing\nPeople who hold a station: Hana, Viktor, Zoe",
     [("Viktor's shift begins at 09:00.", "AT|Viktor|09:00"),
      ("Imani cannot do the 15:00 block.", "NOT_AT|Imani|15:00"),
      ("Zoe runs calibration this week.", "ON|Zoe|calibration"),
      ("Hana will not be handling packing.", "NOT_ON|Hana|packing"),
      ("Mateo takes over straight after Imani finishes.", "RIGHT_BEFORE|Imani|Mateo"),
      ("Viktor starts some time ahead of Zoe.", "BEFORE|Viktor|Zoe"),
      ("Whoever runs intake starts before Mateo.", "BEFORE|@intake|Mateo"),
      ("Zoe's slot is sandwiched between Hana's and Mateo's, whichever order that is.", "BETWEEN|Zoe|Hana|Mateo"),
      ("Pretty sure the 11:00 block belongs to Hana.", "AT|Hana|11:00"),
      ("Mateo asked for the 07:00 block but it was refused.", "NONE"),
      ("Last quarter Imani was on calibration.", "NONE"),
      ("If Viktor had swapped with Zoe, 13:00 would have been easier.", "NONE"),
      ("The printer on the second floor is out of toner.", "NONE")]),
    ("People: Bao, Elif, Casper, Noor, Sven\nBlocks (earliest to latest): 07:00, 09:00, 11:00, 13:00, 15:00\n"
     "Stations: intake, calibration, packing\nPeople who hold a station: Bao, Casper, Noor",
     [("The 13:00 slot is Elif's.", "AT|Elif|13:00"),
      ("Sven is off the rota for 07:00.", "NOT_AT|Sven|07:00"),
      ("Packing is Noor's job.", "ON|Noor|packing"),
      ("Casper is not the intake person.", "NOT_ON|Casper|intake"),
      ("Bao is relieved by Elif with no gap.", "RIGHT_BEFORE|Bao|Elif"),
      ("Noor finishes up before Sven comes in.", "BEFORE|Noor|Sven"),
      ("The calibration person is on later than Elif.", "BEFORE|Elif|@calibration"),
      ("I think Casper's block lies between Sven's and Bao's, order unknown.", "BETWEEN|Casper|Sven|Bao"),
      ("Elif wanted calibration and was turned down.", "NONE"),
      ("Back in January Noor covered intake.", "NONE"),
      ("Nobody has confirmed whether Bao is on 09:00.", "NONE"),
      ("Catering for the Friday meeting is booked.", "NONE")]),
    ("People: Pavel, Yuki, Greta, Dario, Wanjiru\nBlocks (earliest to latest): 07:00, 09:00, 11:00, 13:00, 15:00\n"
     "Stations: intake, calibration, packing\nPeople who hold a station: Yuki, Greta, Dario",
     [("Wanjiru has the 11:00 block.", "AT|Wanjiru|11:00"),
      ("Pavel does not work 13:00.", "NOT_AT|Pavel|13:00"),
      ("Intake falls to Greta.", "ON|Greta|intake"),
      ("Dario has been kept off calibration.", "NOT_ON|Dario|calibration"),
      ("Yuki comes straight after Pavel.", "RIGHT_BEFORE|Pavel|Yuki"),
      ("Greta works earlier than Dario.", "BEFORE|Greta|Dario"),
      ("Wanjiru arrives earlier than whoever has packing.", "BEFORE|Wanjiru|@packing"),
      ("Pavel falls between Yuki and Dario in the day, one way round or the other.", "BETWEEN|Pavel|Yuki|Dario"),
      ("As I recall, Greta is the 09:00 person.", "AT|Greta|09:00"),
      ("Dario applied for 07:00 and was not selected.", "NONE"),
      ("The year before, Yuki was down for packing.", "NONE"),
      ("Had Pavel taken 15:00 things would have run smoother.", "NONE"),
      ("The loading bay door sticks in wet weather.", "NONE")]),
]


def rota_facts(h):
    return ("People: %s\nBlocks (earliest to latest): %s\nStations: %s\nPeople who hold a station: %s"
            % (", ".join(h.staff), ", ".join(h.blocks), ", ".join(h.stations), ", ".join(h.holders)))


def _examples(k, with_filler):
    facts, rows = _SETS[k % len(_SETS)]
    if not with_filler:
        rows = [r for r in rows if r[1] != "NONE"]
    out = ["Worked example.", facts, "NOTES"]
    out += ["[%d] %s" % (i + 1, t) for i, (t, _) in enumerate(rows)]
    out.append("ANSWER")
    out += ["%d|%s" % (i + 1, a) for i, (_, a) in enumerate(rows)]
    return "\n".join(out)


def system_prompt(variant, cfg):
    """variant selects the worked-example set; same_prompt ablation pins it to 0."""
    k = 0 if cfg.off("same_prompt") else variant
    parts = [RULES]
    if not cfg.off("no_guidance"):
        parts.append(FILLER_RULES)
    if not cfg.off("no_completeness"):
        parts.append(COMPLETENESS_RULES)
    parts.append(_examples(k, with_filler=not cfg.off("no_guidance")))
    return "\n\n".join(parts)


def order_for(variant, n, item_id, cfg):
    """Order in which the notes are shown (indices keep their original numbers)."""
    idx = list(range(n))
    if cfg.off("same_prompt") or variant % 3 == 0:
        return idx
    if variant % 3 == 1:
        return idx[::-1]
    rnd = random.Random(zlib.crc32(("%s:%d" % (item_id, variant)).encode()))
    rnd.shuffle(idx)
    return idx


def user_whole(h, body, variant, item_id, cfg):
    order = order_for(variant, len(body), item_id, cfg)
    lines = ["[%d] %s" % (i + 1, body[i]) for i in order]
    return "\n".join([
        "Rota facts", rota_facts(h), "", "NOTES", *lines, "",
        "Write one line for EVERY note above, as <note number>|<RELATION>|<arguments>, sorted by note number. "
        "Output only those lines."])


def user_targeted(h, body, targets, options, item_id, cfg):
    """options: {idx0: [reading strings]} -- candidate readings seen in earlier parses."""
    lines = ["[%d] %s" % (i + 1, t) for i, t in enumerate(body)]
    tgt = sorted(targets)
    opt_lines = ["%d: options: %s" % (i + 1, " || ".join(options.get(i, []))) for i in tgt]
    return "\n".join([
        "Rota facts", rota_facts(h), "", "NOTES (context)", *lines, "",
        "MODE: TARGETED",
        "Only answer for these notes: %s" % ", ".join(str(i + 1) for i in tgt),
        "Earlier readings disagreed. For each, think about what the note really asserts, then output the "
        "correct line as <note number>|<RELATION>|<arguments> -- copying one of the options if it is right, "
        "or writing your own if none is. Output only those lines.",
        *opt_lines])


def user_direct(item_text):
    return ("Shift notes follow. Decide whether the notes admit exactly one valid assignment (unique), several "
            "(ambiguous, give all of them), or none (inconsistent, quote the minimal set of conflicting "
            "statements verbatim).\n\n" + item_text +
            "\n\nReply with JSON only, one of:\n"
            '{"case":"unique","assignment":{"<name>":{"block":"..","station":".."}}}\n'
            '{"case":"ambiguous","assignments":[{...},{...}]}\n'
            '{"case":"inconsistent","conflicts":["<statement>", "..."]}\n'
            'Give "station" only for people the header puts on a station.')
