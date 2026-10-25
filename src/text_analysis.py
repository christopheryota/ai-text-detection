"""Plain-English analysis grounded in fitted linear-model measurements."""

import math


RECONSTRUCTION_TOLERANCE = 1e-8


def _feature_explanation(item, total_absolute_impact):
    contribution = item["contribution"]
    direction = "AI-associated" if contribution > 0 else "human-associated"
    effect = "increases" if contribution > 0 else "decreases"
    relative_impact = (
        item["absolute_contribution"] / total_absolute_impact
        if total_absolute_impact > 0 else 0.0
    )
    if item["feature_type"] == "word_ngram":
        observed_feature = f"the word sequence {item['term']!r}"
    elif item["feature_type"] == "character_ngram":
        observed_feature = f"the {len(item['term'])}-character sequence {item['term']!r}"
    else:
        observed_feature = f"the TF-IDF feature {item['term']!r}"
    explanation = (
        f"The model measured {observed_feature} with TF-IDF value "
        f"{item['tfidf_value']:.6g}. Its fitted AI-oriented coefficient is "
        f"{item['ai_oriented_coefficient']:+.6g}; their product "
        f"{item['contribution']:+.6g} {effect} the AI-oriented decision score, "
        f"supporting {direction} patterns. This is a model association, not evidence "
        f"that a person or AI authored the passage."
    )
    return {
        **item,
        "relative_impact": float(relative_impact),
        "plain_english": explanation,
    }


def build_text_analysis(explanation, word_count, thresholds, risk_gate_met):
    """Build directional analysis without converting decision scores to probabilities."""
    score = float(explanation["decision_score"])
    intercept = float(explanation["intercept"])
    contributions = explanation["all_feature_contributions"]
    if not math.isfinite(score) or not math.isfinite(intercept):
        raise ValueError("The model returned a non-finite decision value.")
    if explanation["active_feature_count"] <= 0 or not contributions:
        raise ValueError("No fitted TF-IDF features matched the input text.")

    contribution_sum = float(sum(item["contribution"] for item in contributions))
    reconstructed_score = intercept + contribution_sum
    reconstruction_error = reconstructed_score - score
    if not math.isfinite(reconstruction_error) or abs(reconstruction_error) > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            "Feature contributions do not reconstruct the model score within numerical tolerance."
        )

    human_cutoff = thresholds["human_score_at_or_below"]
    ai_cutoff = thresholds["ai_score_at_or_above"]
    if (human_cutoff is not None and ai_cutoff is not None
            and human_cutoff >= ai_cutoff):
        raise ValueError("The saved human and AI decision cutoffs overlap.")

    if score > 0:
        direction = "AI-associated writing patterns"
        direction_key = "ai_associated"
    elif score < 0:
        direction = "human-associated writing patterns"
        direction_key = "human_associated"
    else:
        direction = "balanced model evidence"
        direction_key = "balanced"

    crosses_ai_cutoff = ai_cutoff is not None and score >= ai_cutoff
    crosses_human_cutoff = human_cutoff is not None and score <= human_cutoff
    if word_count < 80:
        strength = "weak"
        strength_basis = "The passage is short, and evidence strength has not been validated by text-length band."
    elif not risk_gate_met:
        strength = "weak"
        strength_basis = "The deployment risk gate is not satisfied, so no validated strength level is available."
    elif crosses_ai_cutoff or crosses_human_cutoff:
        strength = "moderate"
        strength_basis = "The score crosses a conservative decision cutoff selected on held-out validation data and passed the configured external risk gate."
    else:
        strength = "weak"
        strength_basis = "The score remains between the conservative validation-selected decision cutoffs."

    total_absolute_impact = float(sum(item["absolute_contribution"] for item in contributions))
    explained = [_feature_explanation(item, total_absolute_impact) for item in contributions]
    ai_features = [item for item in explained if item["contribution"] > 0]
    human_features = [item for item in explained if item["contribution"] < 0]
    ai_mass = float(sum(item["contribution"] for item in ai_features))
    human_mass = float(-sum(item["contribution"] for item in human_features))
    net_feature_contribution = ai_mass - human_mass
    technical_evidence = explained[:100]
    omitted_evidence = explained[100:]
    omitted_contribution = float(sum(item["contribution"] for item in omitted_evidence))
    technical_reconstructed_score = float(
        intercept
        + sum(item["contribution"] for item in technical_evidence)
        + omitted_contribution
    )

    if crosses_ai_cutoff and risk_gate_met:
        risk_assessment = "AI-side conservative cutoff crossed"
    elif crosses_human_cutoff and risk_gate_met:
        risk_assessment = "Human-side conservative cutoff crossed"
    else:
        risk_assessment = "No conservative decision cutoff crossed"

    overall = (
        f"The fitted model leans toward {direction} with {strength} evidence. "
        f"Its oriented decision score is {score:+.4f} (not a probability). "
        f"Active features contribute +{ai_mass:.4f} toward AI-associated patterns "
        f"and -{human_mass:.4f} toward human-associated patterns; the model intercept "
        f"is {intercept:+.4f}, producing the reported score."
    )
    if not risk_gate_met:
        overall += " The evaluation risk gate failed, so this direction is for research context only."

    if len(explained) > 1 and ai_features and human_features:
        balance = (
            f"Both directions occur in the measured features. The largest absolute "
            f"contribution is {explained[0]['contribution']:+.4f} from "
            f"{explained[0]['feature_name']!r}. Summing all signed feature contributions "
            f"gives {net_feature_contribution:+.4f}; adding the intercept yields "
            f"{score:+.4f}. The displayed feature-impact share is relative contribution "
            f"magnitude, not a probability or percentage of authorship."
        )
    elif explained:
        balance = (
            f"The active features all push in one direction. Their signed sum is "
            f"{net_feature_contribution:+.4f}; adding the intercept ({intercept:+.4f}) "
            f"yields the decision score ({score:+.4f})."
        )
    else:
        balance = "No active feature contributions were available."

    limitations = [
        "This model was trained on English examples from the reported dataset; non-English text, code, markup, and unfamiliar input formats are outside its validated scope. Language is not automatically detected.",
        "Word and character TF-IDF associations can reflect topic, genre, source, editing, common phrasing, or language proficiency. The benchmark does not isolate these causes.",
        "The external evaluation holds out the HC3 collection within the same dataset; it is not an independent corpus or proof of performance on other populations or generators.",
        "Probability calibration did not pass the available held-out diagnostics. No writing-probability estimate is produced.",
        "No strong-evidence tier was validated. The highest supported level is moderate when the conservative cutoff and external risk gate are both satisfied.",
    ]
    if word_count < 80:
        limitations.append(
            f"This passage has {word_count} words and is short relative to the accepted input range. Length-specific reliability has not been established; a small set of terms can dominate its score."
        )
    else:
        limitations.append(
            f"This passage has {word_count} words. The report does not establish calibrated reliability by text-length band."
        )

    return {
        "direction": direction,
        "direction_key": direction_key,
        "direction_score": score,
        "strength": strength,
        "strength_basis": strength_basis,
        "risk_assessment": risk_assessment,
        "overall_assessment": overall,
        "balance_of_evidence": balance,
        "word_count": int(word_count),
        "active_feature_count": int(explanation["active_feature_count"]),
        "ai_contribution_mass": ai_mass,
        "human_contribution_mass": human_mass,
        "ai_bar_width": float(100 * ai_mass / total_absolute_impact) if total_absolute_impact else 50.0,
        "human_bar_width": float(100 * human_mass / total_absolute_impact) if total_absolute_impact else 50.0,
        "intercept": intercept,
        "feature_contribution_sum": contribution_sum,
        "reconstructed_score": reconstructed_score,
        "reconstruction_error": float(reconstruction_error),
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "contribution_definition": explanation["contribution_definition"],
        "ai_evidence": ai_features,
        "human_evidence": human_features,
        "ranked_evidence": explained,
        "technical_evidence": technical_evidence,
        "technical_omitted_feature_count": len(omitted_evidence),
        "technical_omitted_contribution": omitted_contribution,
        "technical_reconstructed_score": technical_reconstructed_score,
        "limitations": limitations,
        "what_measured": "The model measured word and character TF-IDF values, multiplied them by fitted Logistic Regression coefficients, and added its intercept.",
        "what_inferred": f"The signed decision score leans toward {direction}; strength is limited to the validation-supported scale described above.",
        "what_uncertain": "The score cannot identify an author. Source, genre, topic, editing, and population shifts can change the learned feature associations.",
        "additional_authorship_evidence": "Authorship would require provenance such as drafts, revision history, account or device records, or other independently verified evidence. This text-only model cannot supply that proof.",
        "probability_estimate": None,
        "probability_display_enabled": False,
        "probability_status": "disabled: held-out HC3 Brier score and expected calibration error worsened after Platt scaling",
        "risk_gate_met": bool(risk_gate_met),
    }
