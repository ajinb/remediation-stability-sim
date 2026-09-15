"""Build the reference list from live records, in order of first citation.

Every arXiv entry is re-read from its arXiv abstract page and every DOI entry from
Crossref at build time; the fetched records are cached under paper/refs/ with the
fetch date, so each formatted reference traces to a record, never to memory.

    python paper/build_bib.py --sections <paper>/sections --out <paper>/sections/10-references.tex
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "refs"
UA = "remediation-stability-sim reference builder (mailto:ajinb@arizona.edu)"

ARXIV = {
    "chen2025aiopslab": "2501.06706", "chen2025stratus": "2506.02009", "qi2026r2act": "2607.04623",
    "zhang2025microremed": "2511.01166", "ardebili2025kubeintellect": "2509.02449", "bai2026oracl": "2602.05292",
    "habibi2026arbiter": "2607.19182", "jha2025itbench": "2502.05352", "clark2026sregym": "2605.07161",
    "shetty2024building": "2407.12165", "she2025learning": "2512.16262", "lyu2026coagent": "2606.15376",
    "peng2026stateful": "2608.02764", "prinos2026stable": "2605.03034", "liu2026selfcorrection": "2604.22273",
    "paul2026context": "2607.25408", "chen2026fastslow": "2604.01681", "vyas2025autonomous": "2507.07115",
    "vyas2026detection": "2606.28011", "shaikh2026temporal": "2603.28790", "kumar2026stability": "2609.14894",
    "kumar2026agentic": "2609.14898", "zhang2026rlscale": "2605.26418", "hamzeh2026mas": "2603.07607",
}
DOI = {
    "serracanta2025stability": "10.1109/ACCESS.2025.3526751", "kephart2003vision": "10.1109/MC.2003.1160055",
    "hellerstein2004feedback": "10.1002/047166880X", "padala2007adaptive": "10.1145/1272996.1273026",
    "lim2010automated": "10.1145/1809049.1809051", "alieldin2012adaptive": "10.1109/NOMS.2012.6211900",
    "gandhi2012autoscale": "10.1145/2382553.2382556", "lorido2014review": "10.1007/s10723-014-9314-7",
    "qu2018autoscaling": "10.1145/3148149", "aldhuraibi2018elasticity": "10.1109/TSC.2017.2711009",
    "patikirikorala2012systematic": "10.1109/SEAMS.2012.6224389", "filieri2015software": "10.1109/SEAMS.2015.12",
    "shevtsov2018control": "10.1109/TSE.2017.2704579", "straesser2022why": "10.1145/3489525.3511680",
    "levin1976note": "10.1016/0040-5809(76)90043-5", "kuruklis1994asymptotic": "10.1006/jmaa.1994.1457",
    "erbe1995global": "10.1080/10236199508808016", "yu1998asymptotic": "10.1016/S0898-1221(98)80021-7",
    "berezansky2011new": "10.1080/10236190903146938", "hespanha2007survey": "10.1109/JPROC.2006.887288",
    "normeyrico2007control": "10.1007/978-1-84628-829-6", "astrom2008feedback": "10.1515/9781400828739",
    "marden1949geometry": "10.1090/surv/003", "anderson1979enestrom": "10.1016/0024-3795(79)90113-7",
}
# Crossref's record for Normey-Rico and Camacho carries no author field; authors from the
# IEEE Control Systems review of the book. Crossref spells Astrom without the ring; the
# publisher's author page (press.princeton.edu/our-authors/astrom-karl-johan) gives the name.
AUTHOR_OVERRIDE = {"normeyrico2007control": ["Normey-Rico JE", "Camacho EF"],
                   "astrom2008feedback": ["\\AA{}str\\\"{o}m KJ", "Murray RM"]}
# Crossref flattens the mathematics in this title.
TITLE_OVERRIDE = {"kuruklis1994asymptotic": "The asymptotic stability of $x_{n+1}-ax_n+bx_{n-k}=0$"}
MANUAL = {
    "k8s2026hpa": "Kubernetes Documentation (2026) Horizontal Pod Autoscaling. "
                  "\\url{https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/}. "
                  "Accessed 15 Sep 2026",
    "keda2026redis": "KEDA Documentation (2026) Redis Lists scaler, version 2.20. "
                     "\\url{https://keda.sh/docs/2.20/scalers/redis-lists/}. Accessed 15 Sep 2026",
    "keda2026scaling": "KEDA Documentation (2026) Scaling Deployments, StatefulSets and Custom Resources, version 2.20. "
                       "\\url{https://keda.sh/docs/2.20/concepts/scaling-deployments/}. Accessed 15 Sep 2026",
    "baby2026stable": "Baby A (2026) Stable by design: a control-theoretic account of AI-driven self-healing "
                      "remediation loops. Preprint, version 0.3. "
                      "\\url{https://cloudandsre.com/research/stable-by-design-remediation-loops/}. Accessed 15 Sep 2026",
    "baby2026sim": "Baby A (2026) remediation-stability-sim: simulator, Kubernetes harness and data for "
                   "remediation-loop stability. \\url{https://github.com/ajinb/remediation-stability-sim}. "
                   "Accessed 15 Sep 2026",
}


def curl(url: str) -> str:
    r = subprocess.run(["curl", "-sL", "--max-time", "60", "-A", UA, url], capture_output=True)
    return r.stdout.decode("utf-8", "ignore")


def fetch_arxiv(aid: str) -> dict:
    p = CACHE / "arxiv" / f"{aid}.json"
    if p.exists():
        return json.loads(p.read_text())
    s = curl(f"https://arxiv.org/abs/{aid}")
    meta = lambda n: [html.unescape(m) for m in re.findall(r'<meta name="%s" content="([^"]*)"' % n, s)]
    rec = {"id": aid, "title": (meta("citation_title") or [""])[0], "authors": meta("citation_author"),
           "date": (meta("citation_date") or [""])[0], "fetched": dt.date.today().isoformat()}
    if not rec["title"] or not rec["authors"]:
        raise RuntimeError(f"arXiv {aid}: could not read metadata")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    time.sleep(2)
    return rec


def fetch_doi(doi: str) -> dict:
    p = CACHE / "crossref" / (doi.replace("/", "_") + ".json")
    if p.exists():
        return json.loads(p.read_text())
    s = curl(f"https://api.crossref.org/works/{doi}")
    msg = json.loads(s)["message"]
    msg["fetched"] = dt.date.today().isoformat()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(msg, indent=1))
    time.sleep(1)
    return msg


def initials(given: str) -> str:
    """'Karl Johan' -> 'KJ', 'S.A.' -> 'SA', 'Hans-Arno' -> 'H-A'."""
    out = []
    for part in re.split(r"[\s.]+", given.strip()):
        if not part:
            continue
        if "-" in part:
            out.append("-".join(x[0] for x in part.split("-") if x))
        else:
            out.append(part[0])
    return "".join(out)


def tex(s: str) -> str:
    s = re.sub(r"<sup>.*?</sup>", "", s)          # footnote markers in Crossref titles
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    repl = {"&": "\\&", "%": "\\%", "_": "\\_", "#": "\\#", "$": "\\$",
            "\u2013": "--", "\u2014": "---", "\u2019": "'", "\u2018": "`", "\u201c": "``", "\u201d": "''",
            "\u2212": "-"}
    return "".join(repl.get(ch, ch) for ch in s)


def names_arxiv(authors: list[str]) -> list[str]:
    out = []
    for a in authors:
        if "," in a:
            fam, given = a.split(",", 1)
        else:
            bits = a.split()
            fam, given = bits[-1], " ".join(bits[:-1])
        out.append(f"{fam.strip()} {initials(given)}")
    return out


def join_names(names: list[str], limit: int = 8) -> str:
    if len(names) > limit:
        return ", ".join(names[:limit]) + " et al."
    return ", ".join(names)


def fmt_arxiv(key: str) -> str:
    r = fetch_arxiv(ARXIV[key])
    year = r["date"][:4]
    return f"{tex(join_names(names_arxiv(r['authors'])))} ({year}) {tex(r['title'])}. arXiv:{r['id']}"


def fmt_doi(key: str) -> str:
    m = fetch_doi(DOI[key])
    if key in AUTHOR_OVERRIDE:
        names = AUTHOR_OVERRIDE[key]
    else:
        names = [f"{a.get('family', '').strip()} {initials(a.get('given', ''))}".strip() for a in m.get("author", [])]
    year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    title = tex(" ".join((m.get("title") or [""])[0].split()))
    sub = (m.get("subtitle") or [""])
    if sub and sub[0] and sub[0].lower() not in title.lower():
        sep = " " if title.rstrip()[-1:] in "?!." else ": "
        title = f"{title}{sep}{tex(sub[0])}"
    title = TITLE_OVERRIDE.get(key, title)
    cont = tex((m.get("container-title") or [""])[0])
    typ = m.get("type")
    vol, iss, page = m.get("volume"), m.get("issue"), m.get("page")
    doi = DOI[key]
    head = f"{tex(join_names(names))} ({year}) {title}."
    if typ == "journal-article":
        loc = cont
        if vol:
            loc += f" {vol}"
            if iss:
                loc += f"({iss})"
        if page:
            loc += f":{page.replace('-', '--')}"
        return f"{head} {loc}. doi:{doi}"
    if typ == "proceedings-article":
        loc = f"In: {cont}"
        if page:
            loc += f", pp {page.replace('-', '--')}"
        return f"{head} {loc}. doi:{doi}"
    publisher = tex(m.get("publisher", ""))
    series = f" {cont}." if cont else ""
    return f"{head}{series} {publisher}. doi:{doi}"


def cite_order(sections: Path) -> list[str]:
    order = []
    for f in sorted(sections.glob("*.tex")):
        if f.name.startswith("10-references"):
            continue
        for m in re.finditer(r"\\cite[tp]?\{([^}]*)\}", f.read_text()):
            for k in m.group(1).split(","):
                k = k.strip()
                if k and k not in order:
                    order.append(k)
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    order = cite_order(Path(args.sections))
    known = set(ARXIV) | set(DOI) | set(MANUAL)
    missing = [k for k in order if k not in known]
    if missing:
        raise SystemExit(f"cited but not defined: {missing}")
    unused = sorted(known - set(order))
    lines = ["\\section*{References}",
             f"\\noindent{{\\footnotesize\\itshape Author lists, titles and identifiers generated from arXiv and "
             f"Crossref records fetched {dt.date.today().strftime('%-d %B %Y')}; web pages checked the same day.}}",
             "\\begingroup\\small",
             "\\renewcommand{\\section}[2]{}",
             "\\begin{thebibliography}{99}"]
    for k in order:
        if k in ARXIV:
            body = fmt_arxiv(k)
        elif k in DOI:
            body = fmt_doi(k)
        else:
            body = MANUAL[k]
        lines.append(f"\\bibitem{{{k}}} {body}")
    lines += ["\\end{thebibliography}", "\\endgroup", ""]
    Path(args.out).write_text("\n".join(lines))
    print(f"{len(order)} references written; unused definitions: {unused}")


if __name__ == "__main__":
    main()
