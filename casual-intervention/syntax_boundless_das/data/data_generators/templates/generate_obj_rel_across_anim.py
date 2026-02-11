#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_yaml(path):
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read config.yaml. Install with: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir, maybe_path):
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.normpath(os.path.join(base_dir, maybe_path))


def normalize_pseudo_json(line):
    # Converts {word:"the", category:"ANIMATE", tags:["D"]} -> valid JSON
    line = line.strip()
    if not line:
        return None
    line = re.sub(r'(?<!")\b(word|category|tags)\b\s*:', r'"\1":', line)
    return line


def load_jsonl(path, pseudo_json=False):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            line = normalize_pseudo_json(raw) if pseudo_json else raw
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}: {raw}") from exc
    return data


def filter_by_tag(items, tag):
    return [item for item in items if tag in item.get("tags", [])]


def filter_ev_anim_only(items):
    return [
        item
        for item in items
        if "EV_ANIM" in item.get("tags", []) and "EV_INAN" not in item.get("tags", [])
    ]


def build_sentence(d_word, ms_word, blocks, mv_word):
    sentence = " ".join([d_word, ms_word] + blocks + [mv_word])
    sentence = sentence[0].upper() + sentence[1:] + "."
    return sentence


def choose_attractor_number(use_trick, main_number):
    if use_trick:
        return "plural" if main_number == "singular" else "singular"
    return random.choice(["singular", "plural"])


def validate_config(config):
    gen = config.get("generation", {})
    lex = config.get("lexicon", {})
    paths = config.get("paths", {})
    if "num_attractors" not in gen or "num_pairs" not in gen:
        raise ValueError(
            "Config must include generation.num_attractors and generation.num_pairs."
        )
    if "nouns" not in paths or "verbs" not in paths:
        raise ValueError("Config must include paths.nouns and paths.verbs.")
    if "c_words_anim" not in lex or "c_words_inanim" not in lex:
        raise ValueError(
            "Config must include lexicon.c_words_anim and lexicon.c_words_inanim."
        )
    if "d_words" not in lex:
        raise ValueError("Config must include lexicon.d_words.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate obj_rel_across_anim minimal pairs."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    config = load_yaml(config_path)
    validate_config(config)

    gen_cfg = config["generation"]
    paths_cfg = config["paths"]
    lex_cfg = config["lexicon"]
    out_cfg = config["output"]

    random.seed(gen_cfg.get("seed", 13))

    nouns_path = resolve_path(config_dir, paths_cfg["nouns"])
    verbs_path = resolve_path(config_dir, paths_cfg["verbs"])
    nouns = load_jsonl(nouns_path)
    verbs = load_jsonl(verbs_path)

    ms_list = filter_by_tag(nouns, "MS")
    es_list = filter_by_tag(nouns, "ES")
    mv_list = filter_by_tag(verbs, "MV")
    ev_list = filter_ev_anim_only(verbs)

    if not ms_list:
        raise ValueError("MS_LIST is empty. Check noun tags in data.")
    if not es_list:
        raise ValueError("ES_LIST is empty. Check noun tags in data.")
    if not mv_list:
        raise ValueError("MV_LIST is empty. Check verb tags in data.")
    if not ev_list:
        raise ValueError("EV_LIST is empty. Check verb tags in data.")

    c_mode = (gen_cfg.get("c_mode") or "anim").strip().lower()
    if c_mode not in {"anim", "inanim"}:
        raise ValueError("generation.c_mode must be 'anim' or 'inanim'.")

    if c_mode == "anim":
        c_words = list(lex_cfg.get("c_words_anim") or [])
    else:
        c_words = list(lex_cfg.get("c_words_inanim") or [])
    if not c_words:
        raise ValueError("No complementizers available for selected c_mode.")

    d_words = [str(w).lower() for w in (lex_cfg.get("d_words") or [])]
    if not d_words:
        raise ValueError("No determiners available in D list.")
    if "the" not in d_words:
        raise ValueError("lexicon.d_words must include 'the'.")
    d_word = "the"

    num_attractors = int(gen_cfg["num_attractors"])
    num_pairs = int(gen_cfg["num_pairs"])
    use_trick = bool(gen_cfg.get("use_trick", True))

    if num_attractors < 1:
        raise ValueError("num_attractors must be >= 1.")

    available_es_count = len({n["lemma"] for n in es_list})
    if num_attractors > available_es_count - 1:
        raise ValueError(
            "num_attractors is too large for ES_LIST after excluding MS."
        )
    if num_attractors > len(ev_list):
        raise ValueError("num_attractors is too large for EV_LIST.")

    out_dir = resolve_path(config_dir, out_cfg["paradigm"])
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir, f"obj_rel_across_anim_pairs_{num_attractors}.csv"
    )

    fieldnames = [
        "pair_id",
        # "num_attractors",
        # "use_trick",
        # "attractor_number_mode",
        # "main_subject_lemma",
        # "main_subject_plural",
        # "main_verb_sg",
        # "main_verb_pl",
        # "c_words",
        # "es_lemmas",
        # "es_plurals",
        # "ev_sg",
        # "ev_pl",
        "sentence_singular",
        "sentence_plural",
    ]

    generated = 0
    pair_id = 1
    with open(out_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while generated < num_pairs:
            random.shuffle(ms_list)
            for ms in ms_list:
                if generated >= num_pairs:
                    break

                ms_lemma = ms["lemma"]
                ms_plural = ms["plural"]

                available_es_map = {
                    e["lemma"]: e for e in es_list if e["lemma"] != ms_lemma
                }
                available_es = list(available_es_map.values())
                es_samples = random.sample(available_es, num_attractors)
                ev_samples = random.sample(ev_list, num_attractors)
                c_samples = [random.choice(c_words) for _ in range(num_attractors)]

                mv = random.choice(mv_list)

                attractor_number = choose_attractor_number(use_trick, "singular")
                attractor_number_mode = (
                    "opposite" if use_trick else attractor_number
                )
                blocks_singular = []
                blocks_plural = []
                for idx in range(num_attractors):
                    es = es_samples[idx]
                    ev = ev_samples[idx]
                    c = c_samples[idx]

                    if use_trick:
                        es_sing = es["lemma"]
                        es_plur = es["plural"]
                        ev_sing = ev["sg"]
                        ev_plur = ev["pl"]

                        blocks_singular.append(
                            f"{c} {d_word} {es_plur} {ev_plur}"
                        )
                        blocks_plural.append(
                            f"{c} {d_word} {es_sing} {ev_sing}"
                        )
                    else:
                        if attractor_number == "singular":
                            es_word = es["lemma"]
                            ev_word = ev["sg"]
                        else:
                            es_word = es["plural"]
                            ev_word = ev["pl"]
                        blocks_singular.append(
                            f"{c} {d_word} {es_word} {ev_word}"
                        )
                        blocks_plural.append(
                            f"{c} {d_word} {es_word} {ev_word}"
                        )

                sentence_singular = build_sentence(
                    d_word, ms_lemma, blocks_singular, mv["sg"]
                )
                sentence_plural = build_sentence(
                    d_word, ms_plural, blocks_plural, mv["pl"]
                )

                writer.writerow(
                    {
                        "pair_id": pair_id,
                        # "num_attractors": num_attractors,
                        # "use_trick": use_trick,
                        # "attractor_number_mode": attractor_number_mode,
                        # "main_subject_lemma": ms_lemma,
                        # "main_subject_plural": ms_plural,
                        # "main_verb_sg": mv["sg"],
                        # "main_verb_pl": mv["pl"],
                        # "c_words": "|".join(c_samples),
                        # "es_lemmas": "|".join([e["lemma"] for e in es_samples]),
                        # "es_plurals": "|".join([e["plural"] for e in es_samples]),
                        # "ev_sg": "|".join([e["sg"] for e in ev_samples]),
                        # "ev_pl": "|".join([e["pl"] for e in ev_samples]),
                        "sentence_singular": sentence_singular,
                        "sentence_plural": sentence_plural,
                    }
                )

                generated += 1
                pair_id += 1

    print(f"Wrote {generated} pairs to {out_file}")


if __name__ == "__main__":
    main()
