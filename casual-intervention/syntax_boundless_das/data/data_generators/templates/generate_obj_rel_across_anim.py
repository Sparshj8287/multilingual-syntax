#!/usr/bin/env python3
import argparse
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


def filter_by_tag_and_category(items, tag, category):
    category_norm = str(category).strip().upper()
    return [
        item
        for item in items
        if tag in item.get("tags", [])
        and str(item.get("category", "")).strip().upper() == category_norm
    ]


def build_sentence(d_word, ms_word, blocks, mv_word):
    sentence = " ".join([d_word, ms_word] + blocks + [mv_word])
    sentence = sentence[0].upper() + sentence[1:] + "."
    return sentence


def build_base_attractor_numbers(num_attractors, variation_type):
    if variation_type == "linear":
        return ["singular"] * (num_attractors - 1) + ["plural"]
    if variation_type == "bow":
        return ["plural"] * (num_attractors - 1) + ["singular"]
    if variation_type == "maximal":
        return ["plural"] * num_attractors
    raise ValueError(
        "variation_type must be one of: 'linear', 'bow', or 'maximal'."
    )


def invert_attractor_numbers(attractor_numbers):
    inverted = []
    for number in attractor_numbers:
        if number == "singular":
            inverted.append("plural")
        elif number == "plural":
            inverted.append("singular")
        else:
            raise ValueError(
                f"Invalid attractor number '{number}'. Expected singular/plural."
            )
    return inverted


def build_clause_blocks(es_samples, ev_samples, c_samples, d_word, attractor_numbers):
    blocks = []
    for idx, number in enumerate(attractor_numbers):
        es = es_samples[idx]
        ev = ev_samples[idx]
        c = c_samples[idx]

        if number == "singular":
            es_word = es["lemma"]
            ev_word = ev["sg"]
        elif number == "plural":
            es_word = es["plural"]
            ev_word = ev["pl"]
        else:
            raise ValueError(
                f"Invalid attractor number '{number}'. Expected singular/plural."
            )
        blocks.append(f"{c} {d_word} {es_word} {ev_word}")
    return blocks


def choose_main_verb(main_verb_list, ev_list, num_attractors):
    mv_candidates = main_verb_list[:]
    random.shuffle(mv_candidates)
    for mv_candidate in mv_candidates:
        mv_base = str(mv_candidate.get("pl", "")).strip().lower()
        available_ev = [
            ev for ev in ev_list if str(ev.get("pl", "")).strip().lower() != mv_base
        ]
        if len(available_ev) >= num_attractors:
            return mv_candidate, available_ev
    return None, []


def normalize_variation_types(value):
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]
    normalized = []
    for item in raw_values:
        v = str(item).strip().lower()
        if v not in {"linear", "bow", "maximal"}:
            raise ValueError("variation_type must be one of: linear, bow, maximal.")
        if v not in normalized:
            normalized.append(v)
    if not normalized:
        raise ValueError("At least one variation_type is required.")
    return normalized


def normalize_num_attractors(value):
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]
    normalized = []
    for item in raw_values:
        n = int(item)
        if n < 1:
            raise ValueError("num_attractors must be >= 1.")
        if n not in normalized:
            normalized.append(n)
    if not normalized:
        raise ValueError("At least one num_attractors value is required.")
    return normalized


def normalize_modes(value):
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]
    normalized = []
    for item in raw_values:
        mode = str(item).strip().lower()
        if mode not in {"anim", "inanim"}:
            raise ValueError("mode must only contain: anim, inanim.")
        if mode not in normalized:
            normalized.append(mode)
    if not normalized:
        raise ValueError("At least one generation.mode value is required.")
    return normalized


def build_mode_targets(num_pairs, modes):
    if len(modes) == 1:
        return {modes[0]: num_pairs}
    if set(modes) == {"anim", "inanim"}:
        target_anim = num_pairs // 2
        target_inanim = num_pairs - target_anim
        return {"anim": target_anim, "inanim": target_inanim}
    raise ValueError("mode must be one of: [anim], [inanim], or [anim, inanim].")


def validate_config(config):
    gen = config.get("generation", {})
    lex = config.get("lexicon", {})
    paths = config.get("paths", {})
    if "num_attractors" not in gen or "num_pairs" not in gen:
        raise ValueError(
            "Config must include generation.num_attractors and generation.num_pairs."
        )
    if "mode" not in gen:
        raise ValueError("Config must include generation.mode.")
    if "c_mode" in gen:
        raise ValueError("generation.c_mode is deprecated. Use generation.mode list.")
    if "nouns" not in paths or "verbs" not in paths:
        raise ValueError("Config must include paths.nouns and paths.verbs.")
    if "filtered_verbs" not in paths:
        raise ValueError("Config must include paths.filtered_verbs.")
    if "c_words_anim" not in lex or "c_words_inanim" not in lex:
        raise ValueError(
            "Config must include lexicon.c_words_anim and lexicon.c_words_inanim."
        )
    if "d_words" not in lex:
        raise ValueError("Config must include lexicon.d_words.")
    normalize_variation_types(gen.get("variation_type") or "maximal")
    normalize_num_attractors(gen.get("num_attractors"))
    normalize_modes(gen.get("mode"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate obj_rel_across_anim minimal pairs."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--variation_type",
        choices=["linear", "bow", "maximal"],
        help=(
            "Attractor-number variation. Overrides generation.variation_type in config."
        ),
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
    filtered_verbs_path = resolve_path(config_dir, paths_cfg["filtered_verbs"])
    nouns = load_jsonl(nouns_path)
    verbs = load_jsonl(verbs_path)
    filtered_verbs = load_jsonl(filtered_verbs_path)

    mode_values = normalize_modes(gen_cfg.get("mode"))

    anim_ms_list = filter_by_tag_and_category(nouns, "MS", "ANIMATE")
    inanim_is_list = filter_by_tag_and_category(nouns, "IS", "INANIMATE")
    es_anim_list = filter_by_tag_and_category(nouns, "ES", "ANIMATE")

    ev_anim_list = filter_by_tag(verbs, "EV_ANIM")
    mv_list = filter_by_tag(filtered_verbs, "MV")
    iv_list = filter_by_tag(filtered_verbs, "IV")
    if not mv_list:
        raise ValueError("No verbs with MV tag are available.")
    if not iv_list:
        raise ValueError("No verbs with IV tag are available.")

    c_words_anim = list(lex_cfg.get("c_words_anim") or [])
    c_words_inanim = list(lex_cfg.get("c_words_inanim") or [])

    mode_specs = {
        "anim": {
            "template": ["D", "MS", "C", "D", "ES", "EV", "MV"],
            "main_subjects": anim_ms_list,
            "main_verbs": mv_list,
            "main_verb_tag": "MV",
            "c_words": c_words_anim,
            "category": "ANIMATE",
        },
        "inanim": {
            "template": ["D", "IS", "IC", "D", "ES", "EV", "IV"],
            "main_subjects": inanim_is_list,
            "main_verbs": iv_list,
            "main_verb_tag": "IV",
            "c_words": c_words_inanim,
            "category": "INANIMATE",
        },
    }

    if "anim" in mode_values and not anim_ms_list:
        raise ValueError("ANIMATE mode selected but no nouns with MS tag are available.")
    if "inanim" in mode_values and not inanim_is_list:
        raise ValueError("INANIMATE mode selected but no nouns with IS tag are available.")
    if mode_values and not es_anim_list:
        raise ValueError("No animate nouns with ES tag are available.")
    if mode_values and not ev_anim_list:
        raise ValueError("No verbs with EV_ANIM tag are available.")
    if "anim" in mode_values and not mv_list:
        raise ValueError("ANIMATE mode selected but no verbs with MV tag are available.")
    if "inanim" in mode_values and not iv_list:
        raise ValueError("INANIMATE mode selected but no verbs with IV tag are available.")
    if "anim" in mode_values and not c_words_anim:
        raise ValueError("ANIMATE mode selected but lexicon.c_words_anim is empty.")
    if "inanim" in mode_values and not c_words_inanim:
        raise ValueError("INANIMATE mode selected but lexicon.c_words_inanim is empty.")

    d_words = [str(w).lower() for w in (lex_cfg.get("d_words") or [])]
    if not d_words:
        raise ValueError("No determiners available in D list.")
    if "the" not in d_words:
        raise ValueError("lexicon.d_words must include 'the'.")
    d_word = "the"

    num_pairs = int(gen_cfg["num_pairs"])
    if num_pairs < 1:
        raise ValueError("generation.num_pairs must be >= 1.")
    num_attractor_values = normalize_num_attractors(gen_cfg["num_attractors"])
    if args.variation_type:
        variation_values = [args.variation_type]
    else:
        variation_values = normalize_variation_types(
            gen_cfg.get("variation_type") or "maximal"
        )

    available_es_count = len({n["lemma"] for n in es_anim_list})
    paradigm_root = out_cfg["paradigm"]
    split = out_cfg["split"]
    data_root = resolve_path(config_dir, out_cfg.get("data_dir", "data"))

    for variation_type in variation_values:
        for num_attractors in num_attractor_values:
            if "anim" in mode_values and num_attractors > available_es_count - 1:
                raise ValueError(
                    f"num_attractors={num_attractors} is too large for ES_LIST after excluding MS."
                )
            if "inanim" in mode_values and num_attractors > available_es_count:
                raise ValueError(
                    f"num_attractors={num_attractors} is too large for animate ES_LIST."
                )
            if num_attractors > len(ev_anim_list):
                raise ValueError(
                    f"num_attractors={num_attractors} is too large for EV_LIST."
                )

            base_attractor_numbers = build_base_attractor_numbers(
                num_attractors, variation_type
            )
            source_attractor_numbers = invert_attractor_numbers(base_attractor_numbers)

            jsonl_dir = os.path.join(data_root, paradigm_root, split, variation_type)
            os.makedirs(jsonl_dir, exist_ok=True)
            jsonl_file = os.path.join(
                jsonl_dir,
                f"obj_rel_across_anim_pairs_{variation_type}_{num_attractors}.jsonl",
            )

            mode_targets = build_mode_targets(num_pairs, mode_values)
            generated = 0
            pair_id = 1
            with open(jsonl_file, "w", encoding="utf-8") as jsonl_out:
                for mode in mode_values:
                    mode_target = mode_targets.get(mode, 0)
                    if mode_target <= 0:
                        continue

                    mode_spec = mode_specs[mode]
                    mode_generated = 0
                    no_progress_rounds = 0

                    while mode_generated < mode_target:
                        produced_this_round = False
                        random.shuffle(mode_spec["main_subjects"])

                        for ms in mode_spec["main_subjects"]:
                            if mode_generated >= mode_target:
                                break

                            ms_lemma = ms["lemma"]
                            ms_plural = ms["plural"]

                            available_es_map = {
                                e["lemma"]: e
                                for e in es_anim_list
                                if e["lemma"] != ms_lemma
                            }
                            available_es = list(available_es_map.values())
                            if len(available_es) < num_attractors:
                                continue

                            es_samples = random.sample(available_es, num_attractors)
                            c_samples = [
                                random.choice(mode_spec["c_words"])
                                for _ in range(num_attractors)
                            ]
                            mv, available_ev = choose_main_verb(
                                mode_spec["main_verbs"], ev_anim_list, num_attractors
                            )
                            if mv is None:
                                raise ValueError(
                                    f"Could not sample {mode_spec['main_verb_tag']} verb with EV base-lemma exclusion."
                                )
                            ev_samples = random.sample(available_ev, num_attractors)

                            blocks_singular = build_clause_blocks(
                                es_samples,
                                ev_samples,
                                c_samples,
                                d_word,
                                base_attractor_numbers,
                            )
                            blocks_plural = build_clause_blocks(
                                es_samples,
                                ev_samples,
                                c_samples,
                                d_word,
                                source_attractor_numbers,
                            )

                            sentence_singular = build_sentence(
                                d_word, ms_lemma, blocks_singular, mv["sg"]
                            )
                            sentence_plural = build_sentence(
                                d_word, ms_plural, blocks_plural, mv["pl"]
                            )
                            jsonl_row = {
                                # "pair_id": pair_id,
                                # "variation_type": variation_type,
                                "base_sentence": sentence_singular,
                                "source_sentence": sentence_plural,
                                "NUA": num_attractors,
                                "MS_base": ms_lemma,
                                "MV_base": mv["sg"],
                                "MS_source": ms_plural,
                                "MV_source": mv["pl"],
                                # "base_attractor_numbers": base_attractor_numbers,
                                # "source_attractor_numbers": source_attractor_numbers,
                                "category": mode_spec["category"],
                            }
                            jsonl_out.write(
                                json.dumps(jsonl_row, ensure_ascii=True) + "\n"
                            )

                            generated += 1
                            mode_generated += 1
                            pair_id += 1
                            produced_this_round = True

                        if produced_this_round:
                            no_progress_rounds = 0
                        else:
                            no_progress_rounds += 1
                            if no_progress_rounds >= 3:
                                raise ValueError(
                                    f"Unable to generate enough pairs for mode={mode}, "
                                    f"variation={variation_type}, num_attractors={num_attractors}."
                                )

            print(f"Wrote {generated} pairs to {jsonl_file}")


if __name__ == "__main__":
    main()
