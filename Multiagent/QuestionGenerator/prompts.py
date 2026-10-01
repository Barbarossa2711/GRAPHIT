from __future__ import annotations

PLAN_SYSTEM = (
    "Du bist ein didaktischer Assistent, der aus Vorlesungsfolien Wiederholungsfragen plant. "
    "Ein 'Stem' ist genau EIN testbarer Kernfakt bzw. ein Lernziel (eine Knowledge Component). "
    "Formuliere prägnante, klar voneinander abgegrenzte Stems zu GENAU DEM angegebenen Konzept. "
    "Vermeide Überschneidungen zwischen den Stems.\n"
    "\n"
    "ZUORDNUNG — die wichtigste Regel: Die Folien sind BELEG für das Konzept, nicht dessen "
    "Themenvorrat. Eine Folie behandelt oft mehrere Konzepte nebeneinander und nennt zusätzlich "
    "Technologien oder Begriffe, die anderswo in der Vorlesung eigene Konzepte sind. Ordne jeden "
    "Fakt dem Konzept zu, ÜBER das er etwas aussagt, und übernimm nur die Fakten, die über das "
    "angegebene Konzept selbst etwas aussagen. Alles andere gehört zu einem anderen Konzept und "
    "wird dort abgefragt — nicht hier.\n"
    "\n"
    "Beispiel: Steht auf der Folie zum Konzept 'Volume' der Satz "
    "'Volume: große Datenmengen — Hadoop, NoSQL-Systeme (z. B. Cassandra, MongoDB)', dann ist der "
    "prüfbare Fakt zu 'Volume', WAS Volume ausmacht (Datenmenge übersteigt einzelne Maschinen). "
    "Hadoop, Cassandra und MongoDB sind an dieser Stelle nur Verweise; ein Stem 'Welche "
    "Technologien bewältigen Volume?' prüft NICHT Volume, sondern Hadoop und NoSQL — und ist "
    "deshalb falsch.\n"
    "\n"
    "Abgrenzende Nennung ist erlaubt: ein Nachbarbegriff darf als Kontrast auftreten "
    "('Volume betrifft die Menge, nicht die Geschwindigkeit'), solange der geprüfte Kern das "
    "angegebene Konzept bleibt und die Antwort ohne Vorwissen über den Nachbarbegriff möglich ist."
)

PLAN_USER = """\
Konzept: {concept_name}{objective_line}

Folieninhalte (Anzahl Folien: {n_slides}):
{slides}
{foreign_block}{avoid_block}
Erzeuge zwischen {min_stems} und {max_stems} Stems – so viele, wie es inhaltlich WIRKLICH
distinkte, testbare Fakten gibt. Lieber weniger, dafür klar abgegrenzte Stems als erzwungene
oder überlappende. Gib für jeden Stem an:
- objective: der konkrete Fakt / das Lernziel (ein Satz).
- question_types: mindestens ZWEI der Typen [single, multiple, cloze, match, order],
  die zu diesem Fakt sachlich passen. Regeln:
  * 'single' und 'multiple' sind DIESELBE Auswahl-Familie. Wähle pro Stem HÖCHSTENS EINEN davon,
    NIE beide. Nimm 'multiple' nur, wenn es fachlich MEHRERE (>= 2) korrekte Antworten gibt;
    hat der Fakt genau eine richtige Antwort, nimm 'single'.
  * 'order' NUR für Abläufe/Sequenzen/Schritte.
  * 'match' NUR wenn es klar zuordenbare Paar-Beziehungen gibt (>= 2 Paare).
  * 'cloze' für Definitionen/Zusammenhänge mit einsetzbaren Fachbegriffen.
  Bevorzuge VIELFALT: kombiniere die Auswahlfrage möglichst mit cloze/match/order (wo sachlich
  sinnvoll), statt immer nur single+cloze zu nutzen.
Ziel: derselbe Kern soll später in verschiedenen Formaten abfragbar sein,
damit die Antwort nicht auswendig gelernt werden kann.
"""

def build_plan_user(
    *,
    concept_name: str,
    objective_line: str,
    n_slides: int,
    slides: str,
    min_stems: int,
    max_stems: int,
    avoid: list[str] | None = None,
    foreign_concepts: list[str] | None = None,
) -> str:
    """
    Builds the user prompt for phase 1, the stem planning, with optional lists of facts and concepts to avoid.

    Naming the :CO_OCCURS neighbour concepts explicitly works much better than the general
    rule in the system prompt, because the model no longer has to guess which terms are
    concepts of their own.

    :param concept_name: Name of the concept being planned for.
    :param objective_line: Pre-formatted objective line, may be empty.
    :param n_slides: Number of slides supplied.
    :param slides: Slide texts serving as the only source.
    :param min_stems: Lower bound of the stem span.
    :param max_stems: Upper bound of the stem span.
    :param avoid: Prompt texts that must not be repeated.
    :param foreign_concepts: Neighbouring concepts to exclude explicitly.
    :return: The user prompt for phase 1.
    """
    if foreign_concepts:
        foreign_block = (
            "\nAuf denselben Folien werden diese EIGENSTÄNDIGEN Konzepte behandelt. Sie haben "
            "ihre eigenen Fragen und dürfen hier NICHT abgefragt werden — sie sind höchstens "
            "zur Abgrenzung erwähnbar:\n"
            + "\n".join(f"- {k}" for k in foreign_concepts)
            + "\n"
        )
    else:
        foreign_block = ""
    if avoid:
        avoid_block = (
            "\nBereits durch Vorlesungsfragen ABGEDECKTE Fakten (NICHT erneut abfragen, "
            "keine Dopplungen erzeugen):\n"
            + "\n".join(f"- {a}" for a in avoid)
            + "\n"
        )
    else:
        avoid_block = ""
    return PLAN_USER.format(
        concept_name=concept_name,
        objective_line=objective_line,
        n_slides=n_slides,
        slides=slides,
        min_stems=min_stems,
        max_stems=max_stems,
        avoid_block=avoid_block,
        foreign_block=foreign_block,
    )


LECTURE_SYSTEM = (
    "Du bist ein didaktischer Assistent. Du erhältst bereits in der Vorlesung gestellte "
    "Wiederholungsfragen. Extrahiere je Vorlesungsfrage GENAU EINEN Stem (den testbaren "
    "Kernfakt / das Lernziel, das die Frage prüft) und schlage passende Fragetypen vor. "
    "Fasse den Kern neutral als Lernziel, nicht als konkrete Fragestellung."
)

LECTURE_USER = """\
Konzept: {concept_name}

Bereits vorhandene Vorlesungsfragen:
{questions}

Erzeuge für JEDE dieser Fragen genau einen Stem. Gib je Stem an:
- objective: der getestete Kernfakt / das Lernziel (ein Satz).
- question_types: mindestens ZWEI passende Typen aus [single, multiple, cloze, match, order]
  (nur sachlich passende, siehe unten), damit der Kern in verschiedenen Formaten abfragbar ist.
  * 'single' und 'multiple' sind DIESELBE Auswahl-Familie. Wähle pro Stem HÖCHSTENS EINEN davon,
    NIE beide. 'multiple' nur bei fachlich MEHREREN (>= 2) korrekten Antworten, sonst 'single'.
  * 'order' NUR für Abläufe/Sequenzen.
  * 'match' NUR bei klar zuordenbaren Paar-Beziehungen (>= 2 Paare).
  * 'cloze' für Definitionen/Zusammenhänge mit einsetzbaren Fachbegriffen.
"""


VARIANT_SYSTEM = (
    "Du bist ein Assistent, der eine einzelne, hochwertige Quizfrage im geforderten Format erzeugt. "
    "Alle Optionen/Begriffe/Elemente tragen kurze IDs (z. B. 'a','b' bzw. 't1','l1','r1','s1'). "
    "Die Bewertung erfolgt rein über ID-Vergleich, daher muss die 'solution' konsistent auf die "
    "vergebenen IDs verweisen. Antworte ausschließlich auf Deutsch und halte dich exakt an die Struktur. "
    "Alle Antwortmöglichkeiten haben etwa dieselbe Länge und denselben Detailgrad — die richtige "
    "Antwort darf nicht schon dadurch auffallen, dass sie länger oder ausführlicher formuliert ist "
    "als die anderen. Die Begründung gehört in 'explanation', nicht in die richtige Option. "
    "Die IDs sind rein technisch: Sie dürfen in KEINEM sichtbaren Text auftauchen — weder in "
    "'prompt' noch in 'explanation' noch in den Antworttexten. Beziehe dich in der Erklärung "
    "immer auf den Wortlaut der Elemente, nie auf 'l1', 'r2' oder 's3'. Schreibe die Erklärung "
    "für den Lernenden; Begriffe der Aufgabenkonstruktion wie 'Distraktor' gehören nicht hinein."
)

_TYPE_RULES = {
    "single": (
        "Fragetyp single (Single Choice): 3-4 Optionen mit IDs 'a','b','c','d'; genau eine ist korrekt. "
        "solution.correct = ID der korrekten Option. Distraktoren sollen plausibel sein."
    ),
    "multiple": (
        "Fragetyp multiple (Multiple Choice): 4-5 Optionen mit IDs 'a','b',...; es MÜSSEN MINDESTENS "
        "ZWEI Antworten korrekt sein (sonst wäre es eine single-Frage). "
        "solution.correct = Liste der korrekten IDs (>= 2)."
    ),
    "cloze": (
        "Fragetyp cloze (Lückentext): 'prompt' ist eine kurze Arbeitsanweisung "
        "(z. B. 'Fülle die Lücken mit den passenden Begriffen aus dem Pool.'). "
        "'text' enthält Platzhalter der Form {{b1}}, {{b2}}, ... "
        "'blanks' listet diese Blank-IDs. 'bank' ist der Begriffs-Pool (IDs 't1','t2',...) und "
        "enthält alle Lösungsbegriffe PLUS mindestens einen Distraktor. "
        "'solution' bildet jede Blank-ID auf die korrekte Begriffs-ID ab, z. B. {{\"b1\":\"t1\"}}."
    ),
    "match": (
        "Fragetyp match (Zuordnung): 'left' (IDs 'l1','l2',...) und 'right' (IDs 'r1','r2',...). "
        "Gern mehr rechte als linke Einträge; die überzähligen rechten sind Distraktoren. "
        "'solution' ist eine Liste von {{\"left\":\"l1\",\"right\":\"r1\"}}-Paaren; "
        "mindestens zwei Paare. "
        "WICHTIG: Jedes linke Element kommt in 'solution' GENAU EINMAL vor — eine 1:n-Zuordnung "
        "(ein linkes Element mit mehreren rechten) ist nicht erlaubt, da die Antwort als Map "
        "left->right zurückkommt und mehrere Partner nicht darstellen kann. Passen mehrere "
        "rechte Einträge zu einem linken, fasse sie zu EINER Aussage zusammen oder mache die "
        "überzähligen zu Distraktoren. Umgekehrt braucht JEDES linke Element einen Partner."
    ),
    "order": (
        "Fragetyp order (Reihenfolge): 'items' (IDs 's1','s2',...) in beliebiger Anzeigereihenfolge. "
        "'solution' ist die Liste der Item-IDs in der KORREKTEN Sequenz (mindestens zwei)."
    ),
}

VARIANT_USER = """\
Konzept: {concept_name}
Abzufragender Kernfakt (Stem): {objective}

Relevante Folieninhalte:
{slides}
{avoid_block}
Erzeuge dazu genau EINE Frage vom {qtype_rule}
Füge eine kurze 'explanation' (Begründung der Lösung) hinzu.
"""

_ISOMORPH_BLOCK = """
Zu diesem Kernfakt existieren BEREITS die folgenden Fragen:
{existing}
Erzeuge eine davon klar UNTERSCHEIDBARE Frage: derselbe Kernfakt, aber andere
Formulierung, anderer Blickwinkel (z. B. Anwendung statt Definition, Negativfall statt
Positivfall) und andere Distraktoren. Die Lösung darf nicht allein durch Wiedererkennen
einer bereits gesehenen Formulierung zu finden sein.
"""


def build_variant_user(
    *,
    concept_name: str,
    objective: str,
    slides: str,
    qtype: str,
    avoid: list[str] | None = None,
) -> str:
    """
    Builds the user prompt for phase 2, one variant per stem and question type, including the type-specific rule.

    A non-empty ``avoid`` produces an isomorphic item: same fact, different surface form;
    without it the model returns nearly identical questions for the same stem and type.

    :param concept_name: Name of the concept being generated for.
    :param objective: Learning objective of this stem.
    :param slides: Slide texts serving as the only source.
    :param qtype: Question type whose rules are appended.
    :param avoid: Prompt texts that must not be repeated.
    :return: The user prompt for phase 2.
    """
    avoid_block = (
        _ISOMORPH_BLOCK.format(existing="\n".join(f"- {t}" for t in avoid)) if avoid else ""
    )
    return VARIANT_USER.format(
        concept_name=concept_name,
        objective=objective,
        slides=slides,
        avoid_block=avoid_block,
        qtype_rule=f"Typ '{qtype}'. {_TYPE_RULES[qtype]}",
    )
