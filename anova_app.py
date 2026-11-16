"""Interactive, manually calculated one-way ANOVA teaching application."""

from __future__ import annotations

import io
import math
import warnings
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import streamlit as st
from scipy import stats


EXAMPLE_GROUPS = {
    "Group A": [4.0, 5.0, 6.0],
    "Group B": [7.0, 8.0, 9.0],
    "Group C": [10.0, 11.0, 12.0],
}
ALPHA_LEVELS = (0.01, 0.05, 0.10)


def parse_values(text: str, group_name: str) -> list[float]:
    """Parse comma-separated, finite numbers or raise an explanatory error."""
    pieces = [piece.strip() for piece in text.split(",")]
    if not any(pieces):
        raise ValueError(f"{group_name}: enter at least two comma-separated numbers.")

    values: list[float] = []
    for position, piece in enumerate(pieces, start=1):
        if not piece:
            raise ValueError(
                f"{group_name}: value {position} is empty. "
                "Remove extra commas or enter a number."
            )
        try:
            value = float(piece)
        except ValueError as error:
            raise ValueError(
                f"{group_name}: {piece!r} is not a valid number."
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"{group_name}: all values must be finite numbers.")
        values.append(value)

    if len(values) < 2:
        raise ValueError(f"{group_name}: enter at least two values.")
    return values


def validate_groups(groups: dict[str, Sequence[float]]) -> dict[str, np.ndarray]:
    """Validate the group names, count, sizes, and finite numeric observations."""
    if not 2 <= len(groups) <= 6:
        raise ValueError("Enter between 2 and 6 groups.")

    names = [name.strip() for name in groups]
    if any(not name for name in names):
        raise ValueError("Every group must have a non-empty name.")
    if len(set(names)) != len(names):
        raise ValueError("Group names must be unique.")

    validated: dict[str, np.ndarray] = {}
    for name, observations in groups.items():
        values = np.asarray(observations, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"{name}: observations must be a one-dimensional list.")
        if values.size < 2:
            raise ValueError(f"{name}: enter at least two values.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}: all values must be finite numbers.")
        validated[name.strip()] = values
    return validated


def calculate_one_way_anova(
    groups: dict[str, Sequence[float]], alpha: float = 0.05
) -> dict[str, Any]:
    """Calculate one-way ANOVA sums of squares and follow-up diagnostics.

    The ANOVA sums of squares, degrees of freedom, mean squares, and F statistic
    are calculated directly from their definitions rather than from an ANOVA
    convenience function. SciPy supplies reference distributions and diagnostics.
    """
    if alpha not in ALPHA_LEVELS:
        raise ValueError("Alpha must be one of 0.01, 0.05, or 0.10.")
    data = validate_groups(groups)
    names = list(data)
    sizes = {name: int(values.size) for name, values in data.items()}
    means = {name: float(np.mean(values)) for name, values in data.items()}
    all_values = np.concatenate(list(data.values()))
    group_count = len(data)
    total_count = int(all_values.size)
    grand_mean = float(np.mean(all_values))

    ssb_terms = {
        name: sizes[name] * (means[name] - grand_mean) ** 2 for name in names
    }
    ssb = float(sum(ssb_terms.values()))
    ssw_terms = {
        name: float(np.sum((data[name] - means[name]) ** 2)) for name in names
    }
    ssw = float(sum(ssw_terms.values()))
    sst = ssb + ssw
    direct_sst = float(np.sum((all_values - grand_mean) ** 2))
    sums_of_squares_match = bool(np.isclose(sst, direct_sst, rtol=1e-10, atol=1e-12))

    df_between = group_count - 1
    df_within = total_count - group_count
    df_total = total_count - 1
    msb = ssb / df_between
    msw = ssw / df_within
    if msw == 0:
        f_statistic = math.nan if msb == 0 else math.inf
    else:
        f_statistic = msb / msw
    f_critical = float(stats.f.ppf(1 - alpha, df_between, df_within))
    p_value = float(stats.f.sf(f_statistic, df_between, df_within))
    eta_squared = ssb / sst if sst > 0 else math.nan

    scipy_result = stats.f_oneway(*(data[name] for name in names))
    scipy_f = float(scipy_result.statistic)
    scipy_p = float(scipy_result.pvalue)
    f_matches = bool(
        np.isclose(f_statistic, scipy_f, rtol=1e-8, atol=1e-12, equal_nan=True)
    )
    p_matches = bool(
        np.isclose(p_value, scipy_p, rtol=1e-8, atol=1e-12, equal_nan=True)
    )

    normality: dict[str, dict[str, Any]] = {}
    for name, values in data.items():
        if values.size < 3:
            normality[name] = {
                "statistic": math.nan,
                "p_value": math.nan,
                "passed": None,
                "note": "Not assessed: Shapiro-Wilk requires at least 3 observations.",
            }
        elif np.ptp(values) == 0:
            normality[name] = {
                "statistic": math.nan,
                "p_value": math.nan,
                "passed": None,
                "note": "Not assessed: a constant group has no distribution to assess.",
            }
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = stats.shapiro(values)
            normality[name] = {
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "passed": bool(result.pvalue > alpha),
                "note": (
                    "For samples larger than 5,000, the Shapiro-Wilk statistic "
                    "is accurate but its p-value may not be."
                    if values.size > 5000
                    else "; ".join(str(item.message) for item in caught)
                ),
            }

    with warnings.catch_warnings(record=True) as levene_warnings:
        warnings.simplefilter("always")
        levene_result = stats.levene(*(data[name] for name in names), center="median")
    levene_p = float(levene_result.pvalue)
    levene_tested = math.isfinite(levene_p) and math.isfinite(
        float(levene_result.statistic)
    )
    levene_passed = bool(levene_p > alpha) if levene_tested else None
    levene_note = (
        "A small p-value is evidence against equal group variances."
        if levene_tested
        else "Not assessed: Levene's test returned an undefined result for these data."
    )
    if levene_warnings and not levene_tested:
        levene_note += " " + str(levene_warnings[0].message)

    decision_defined = not (math.isnan(f_statistic) or math.isnan(p_value))
    reject_null = bool(decision_defined and p_value < alpha)
    assumption_issues: list[str] = []
    for name, result in normality.items():
        if result["passed"] is False:
            assumption_issues.append(f"Normality was not supported for {name}.")
        elif result["passed"] is None:
            assumption_issues.append(f"Normality could not be assessed for {name}.")
    if levene_passed is False:
        assumption_issues.append("Levene's test indicates unequal variances.")
    elif levene_passed is None:
        assumption_issues.append("Equal variances could not be assessed with Levene's test.")

    tukey_result = None
    if reject_null:
        tukey_result = stats.tukey_hsd(*(data[name] for name in names))
        confidence = tukey_result.confidence_interval(confidence_level=1 - alpha)
        pairwise: list[dict[str, Any]] = []
        for first in range(group_count):
            for second in range(first + 1, group_count):
                pairwise.append(
                    {
                        "Group 1": names[first],
                        "Group 2": names[second],
                        "Mean difference (1 - 2)": float(
                            tukey_result.statistic[first, second]
                        ),
                        "Adjusted p-value": float(tukey_result.pvalue[first, second]),
                        f"{100 * (1 - alpha):g}% CI lower": float(
                            confidence.low[first, second]
                        ),
                        f"{100 * (1 - alpha):g}% CI upper": float(
                            confidence.high[first, second]
                        ),
                        "Significant": bool(
                            tukey_result.pvalue[first, second] < alpha
                        ),
                    }
                )
    else:
        pairwise = []

    return {
        "groups": data,
        "names": names,
        "sizes": sizes,
        "means": means,
        "grand_mean": grand_mean,
        "group_count": group_count,
        "total_count": total_count,
        "ssb_terms": ssb_terms,
        "ssw_terms": ssw_terms,
        "ssb": ssb,
        "ssw": ssw,
        "sst": sst,
        "direct_sst": direct_sst,
        "sums_of_squares_match": sums_of_squares_match,
        "df_between": df_between,
        "df_within": df_within,
        "df_total": df_total,
        "msb": msb,
        "msw": msw,
        "f_statistic": f_statistic,
        "f_critical": f_critical,
        "p_value": p_value,
        "alpha": alpha,
        "eta_squared": eta_squared,
        "scipy_f": scipy_f,
        "scipy_p": scipy_p,
        "verification_matches": f_matches and p_matches,
        "normality": normality,
        "levene_statistic": float(levene_result.statistic),
        "levene_p": levene_p,
        "levene_passed": levene_passed,
        "levene_note": levene_note,
        "assumption_issues": assumption_issues,
        "decision_defined": decision_defined,
        "reject_null": reject_null,
        "pairwise": pairwise,
        "scipy_version": scipy.__version__,
    }


def parse_csv_upload(contents: bytes) -> dict[str, list[float]]:
    """Read one group per CSV column, allowing blanks for unequal group sizes."""
    try:
        frame = pd.read_csv(io.BytesIO(contents), skipinitialspace=True)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Could not read the CSV file: {error}") from error

    if not 2 <= len(frame.columns) <= 6:
        raise ValueError("The CSV must have between 2 and 6 group columns.")
    if len(set(str(column).strip() for column in frame.columns)) != len(frame.columns):
        raise ValueError("CSV group column names must be unique.")

    groups: dict[str, list[float]] = {}
    for column in frame.columns:
        name = str(column).strip()
        if not name:
            raise ValueError("Each CSV column must have a non-empty group name.")
        raw_values = frame[column]
        blank = raw_values.isna() | raw_values.astype(str).str.strip().eq("")
        numeric = pd.to_numeric(raw_values, errors="coerce")
        invalid = numeric.isna() & ~blank
        if bool(invalid.any()):
            bad_value = raw_values[invalid].iloc[0]
            raise ValueError(f"{name}: {bad_value!r} is not a valid number.")
        values = numeric[~blank].to_numpy(dtype=float)
        groups[name] = values.tolist()
    return validate_groups(groups)


def _format_number(value: float) -> str:
    """Render finite numbers compactly while preserving useful precision."""
    if math.isnan(value):
        return "undefined"
    if math.isinf(value):
        return "infinity"
    return f"{value:.6g}"


def _decision_text(result: dict[str, Any]) -> str:
    names = ", ".join(result["names"])
    if not result["decision_defined"]:
        return (
            "The F ratio is undefined because both between-group and within-group "
            "variation are zero. No decision about the population means can be "
            "made from this data."
        )
    if result["reject_null"]:
        return (
            f"At alpha = {result['alpha']:.2f}, reject H0. There is evidence that "
            f"at least one population mean differs among {names}. Tukey's HSD "
            "identifies which pairs differ."
        )
    return (
        f"At alpha = {result['alpha']:.2f}, fail to reject H0. The data do not "
        f"provide sufficient evidence that the population means differ among "
        f"{names}."
    )


def _effect_interpretation(eta_squared: float) -> str:
    if math.isnan(eta_squared):
        return "The effect size is undefined because the total variation is zero."
    if eta_squared < 0.01:
        strength = "very small"
    elif eta_squared < 0.06:
        strength = "small"
    elif eta_squared < 0.14:
        strength = "medium"
    else:
        strength = "large"
    return (
        f"Group membership accounts for {_format_number(eta_squared * 100)}% "
        f"of the observed total variation (eta squared = {_format_number(eta_squared)}; "
        f"a {strength} effect by common Cohen-style guidelines)."
    )


def _show_working(result: dict[str, Any]) -> None:
    """Display formulas with the current data substituted into each expression."""
    with st.expander("1. Hypotheses, groups, and sample sizes", expanded=False):
        st.markdown(
            "**H0:** All population group means are equal.  \n"
            "**H1:** At least one population group mean differs."
        )
        st.write(f"Groups (k): {result['names']}")
        st.write(f"Group sizes (n): {result['sizes']}")
        st.write(f"Group means: {result['means']}")
        st.write(
            f"Total sample size (N) = {' + '.join(map(str, result['sizes'].values()))} "
            f"= {result['total_count']}"
        )
        st.write(
            "Grand mean = sum of all observations / N = "
            f"{_format_number(result['grand_mean'])}"
        )

    with st.expander("2. Between-group sum of squares (SSB)", expanded=False):
        terms = [
            f"{result['sizes'][name]} x "
            f"({_format_number(result['means'][name])} - "
            f"{_format_number(result['grand_mean'])})^2"
            for name in result["names"]
        ]
        st.latex(r"SSB = \sum_i n_i(\bar{x}_i-\bar{x})^2")
        st.write("Substitution: " + " + ".join(terms))
        st.write(f"SSB = {_format_number(result['ssb'])}")

    with st.expander("3. Within-group sum of squares (SSW)", expanded=False):
        terms = [
            f"{name}: sum((x - {_format_number(result['means'][name])})^2) "
            f"= {_format_number(result['ssw_terms'][name])}"
            for name in result["names"]
        ]
        st.latex(r"SSW = \sum_i \sum_j (x_{ij}-\bar{x}_i)^2")
        st.write("; ".join(terms))
        st.write(f"SSW = {_format_number(result['ssw'])}")

    with st.expander("4. Total sum of squares and verification", expanded=False):
        st.latex(r"SST = SSB + SSW = \sum_i \sum_j (x_{ij}-\bar{x})^2")
        st.write(
            f"SST = {_format_number(result['ssb'])} + "
            f"{_format_number(result['ssw'])} = {_format_number(result['sst'])}"
        )
        st.write(
            f"Direct check: sum((x - grand mean)^2) = "
            f"{_format_number(result['direct_sst'])}"
        )
        if result["sums_of_squares_match"]:
            st.success("The two SST calculations agree.")
        else:
            st.error("The two SST calculations do not agree.")

    with st.expander("5. Degrees of freedom and mean squares", expanded=False):
        st.write(
            f"df_between = k - 1 = {result['group_count']} - 1 "
            f"= {result['df_between']}"
        )
        st.write(
            f"df_within = N - k = {result['total_count']} - "
            f"{result['group_count']} = {result['df_within']}"
        )
        st.write(
            f"df_total = N - 1 = {result['total_count']} - 1 "
            f"= {result['df_total']}"
        )
        st.write(
            f"MSB = SSB / df_between = {_format_number(result['ssb'])} / "
            f"{result['df_between']} = {_format_number(result['msb'])}"
        )
        st.write(
            f"MSW = SSW / df_within = {_format_number(result['ssw'])} / "
            f"{result['df_within']} = {_format_number(result['msw'])}"
        )

    with st.expander("6. F statistic, reference values, and decision", expanded=False):
        st.latex(r"F = \frac{MSB}{MSW}")
        st.write(
            f"F = {_format_number(result['msb'])} / "
            f"{_format_number(result['msw'])} = "
            f"{_format_number(result['f_statistic'])}"
        )
        st.write(
            f"F critical = F inverse CDF(1 - alpha; "
            f"{result['df_between']}, {result['df_within']}) = "
            f"{_format_number(result['f_critical'])}"
        )
        st.write(
            f"p-value = upper-tail F probability = "
            f"{_format_number(result['p_value'])}"
        )
        st.write(_decision_text(result))

    with st.expander("7. Effect size", expanded=False):
        st.latex(r"\eta^2 = \frac{SSB}{SST}")
        st.write(
            f"eta squared = {_format_number(result['ssb'])} / "
            f"{_format_number(result['sst'])} = "
            f"{_format_number(result['eta_squared'])}"
        )
        st.write(_effect_interpretation(result["eta_squared"]))


def _show_assumptions(result: dict[str, Any]) -> None:
    st.subheader("Assumption checks")
    st.caption(
        "Normality and equal-variance tests are diagnostics, not proof that an "
        "assumption is true. Shapiro-Wilk has limited power with small samples, "
        "so a non-significant result does not prove normality. Interpret these "
        "checks alongside the study design and plots."
    )
    rows = []
    for name, check in result["normality"].items():
        if check["passed"] is None:
            status = "Not assessed"
        else:
            status = "Pass" if check["passed"] else "Warning"
        rows.append(
            {
                "Check": f"Shapiro-Wilk: {name}",
                "Statistic": check["statistic"],
                "p-value": check["p_value"],
                "Result": status,
                "Note": check["note"],
            }
        )
    rows.append(
        {
            "Check": "Levene (median-centered)",
            "Statistic": result["levene_statistic"],
            "p-value": result["levene_p"],
            "Result": (
                "Not assessed"
                if result["levene_passed"] is None
                else "Pass"
                if result["levene_passed"]
                else "Warning"
            ),
            "Note": result["levene_note"],
        }
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info(
        "**Independence:** This cannot be checked from the values alone. "
        "It depends on how observations were sampled and whether one observation "
        "influences another."
    )
    if result["assumption_issues"]:
        st.warning(
            "At least one assumption needs attention: "
            + " ".join(result["assumption_issues"])
            + " Review the study design and distributions. If the observations "
            "are independent but the normality or equal-variance conditions are "
            "not credible, Kruskal-Wallis is a possible nonparametric alternative; "
            "it tests distributions/ranks, not equality of means."
        )


def _show_charts(result: dict[str, Any]) -> None:
    st.subheader("Charts")
    data = result["groups"]
    names = result["names"]

    box_figure, box_axis = plt.subplots()
    box_axis.boxplot([data[name] for name in names], tick_labels=names)
    box_axis.set_title("Observed values by group")
    box_axis.set_ylabel("Value")
    box_axis.tick_params(axis="x", rotation=20)
    box_figure.tight_layout()
    st.pyplot(box_figure)
    plt.close(box_figure)

    means = [result["means"][name] for name in names]
    standard_errors = [
        float(np.std(data[name], ddof=1) / math.sqrt(result["sizes"][name]))
        for name in names
    ]
    bar_figure, bar_axis = plt.subplots()
    bar_axis.bar(names, means, yerr=standard_errors, capsize=5)
    bar_axis.set_title("Group means with standard-error bars")
    bar_axis.set_ylabel("Mean (error bars: +/- 1 standard error)")
    bar_axis.tick_params(axis="x", rotation=20)
    bar_figure.tight_layout()
    st.pyplot(bar_figure)
    plt.close(bar_figure)

    critical = result["f_critical"]
    observed = result["f_statistic"]
    curve_end = max(critical * 1.5, 1.0)
    if math.isfinite(observed):
        curve_end = max(curve_end, observed * 1.15)
    else:
        curve_end *= 2
    x_values = np.linspace(1e-8, curve_end, 600)
    y_values = stats.f.pdf(x_values, result["df_between"], result["df_within"])
    f_figure, f_axis = plt.subplots()
    f_axis.plot(x_values, y_values, color="navy", label="F distribution under H0")
    tail = x_values >= critical
    f_axis.fill_between(
        x_values,
        y_values,
        where=tail,
        color="tomato",
        alpha=0.35,
        label=f"Critical region (alpha = {result['alpha']:.2f})",
    )
    f_axis.axvline(critical, color="tomato", linestyle="--", label="Critical F")
    if math.isfinite(observed):
        f_axis.axvline(
            observed, color="darkgreen", linestyle="-.", label="Observed F"
        )
    else:
        f_axis.annotate(
            "Observed F = infinity",
            xy=(curve_end * 0.96, max(y_values) * 0.8),
            ha="right",
        )
    f_axis.set_title("F distribution and upper-tail critical region")
    f_axis.set_xlabel("F")
    f_axis.set_ylabel("Density")
    f_axis.legend()
    f_figure.tight_layout()
    st.pyplot(f_figure)
    plt.close(f_figure)


def main() -> None:
    """Build the Streamlit interface and display a one-way ANOVA analysis."""
    st.set_page_config(page_title="One-Way ANOVA Lab", page_icon="📊", layout="wide")
    st.title("One-Way ANOVA: a worked hypothesis test")
    st.write(
        "Compare the means of 2 to 6 independent groups. Enter observations "
        "directly or upload a CSV with one group per column."
    )
    st.caption(
        "The ANOVA arithmetic is computed directly from the formulas below; "
        "SciPy is used for reference distributions, diagnostic tests, and an "
        "independent verification."
    )

    if "group_count" not in st.session_state:
        st.session_state["group_count"] = 3
    for index in range(6):
        st.session_state.setdefault(f"group_name_{index}", f"Group {index + 1}")
        st.session_state.setdefault(f"group_values_{index}", "")

    if st.button("Load example data", type="secondary"):
        st.session_state["group_count"] = len(EXAMPLE_GROUPS)
        for index, (name, values) in enumerate(EXAMPLE_GROUPS.items()):
            st.session_state[f"group_name_{index}"] = name
            st.session_state[f"group_values_{index}"] = ", ".join(
                _format_number(value) for value in values
            )

    alpha = st.selectbox("Significance level (alpha)", ALPHA_LEVELS, index=1)
    source = st.radio(
        "Choose how to enter data",
        ("Enter values", "Upload CSV"),
        horizontal=True,
    )

    submitted_groups: dict[str, Sequence[float]] | None = None
    if source == "Enter values":
        group_count = st.slider("Number of groups", 2, 6, key="group_count")
        with st.form("manual_data_form"):
            entered: list[tuple[str, str]] = []
            columns = st.columns(2)
            for index in range(group_count):
                with columns[index % 2]:
                    name = st.text_input(
                        f"Group {index + 1} name",
                        key=f"group_name_{index}",
                    )
                    values = st.text_input(
                        f"Group {index + 1} values",
                        key=f"group_values_{index}",
                        placeholder="e.g. 4, 5, 6",
                    )
                    entered.append((name, values))
            run_analysis = st.form_submit_button("Calculate ANOVA", type="primary")
        if run_analysis:
            try:
                if len({name.strip() for name, _ in entered}) != len(entered):
                    raise ValueError("Group names must be unique.")
                submitted_groups = {}
                for name, text in entered:
                    if not name.strip():
                        raise ValueError("Every group must have a non-empty name.")
                    submitted_groups[name] = parse_values(text, name)
            except ValueError as error:
                st.error(str(error))
                return
    else:
        uploaded_file = st.file_uploader(
            "Upload a CSV file (one group per column; blank cells are ignored)",
            type=["csv"],
        )
        run_analysis = st.button("Calculate ANOVA", type="primary")
        if run_analysis:
            if uploaded_file is None:
                st.error("Upload a CSV file before calculating the ANOVA.")
                return
            try:
                submitted_groups = parse_csv_upload(uploaded_file.getvalue())
            except ValueError as error:
                st.error(str(error))
                return

    if submitted_groups is None:
        st.info(
            "Load the example data or enter observations, then select "
            "**Calculate ANOVA**."
        )
        return

    try:
        result = calculate_one_way_anova(submitted_groups, float(alpha))
    except ValueError as error:
        st.error(str(error))
        return

    st.subheader("Hypotheses")
    st.markdown(
        "**H0:** All group population means are equal.  \n"
        "**H1:** At least one group population mean differs."
    )
    st.write(f"Groups: {', '.join(result['names'])} | k = {result['group_count']} "
             f"| N = {result['total_count']} | alpha = {result['alpha']:.2f}")

    st.subheader("ANOVA table")
    anova_table = pd.DataFrame(
        [
            {
                "Source": "Between groups",
                "SS": result["ssb"],
                "df": result["df_between"],
                "MS": result["msb"],
                "F": result["f_statistic"],
                "p-value": result["p_value"],
            },
            {
                "Source": "Within groups (error)",
                "SS": result["ssw"],
                "df": result["df_within"],
                "MS": result["msw"],
                "F": np.nan,
                "p-value": np.nan,
            },
            {
                "Source": "Total",
                "SS": result["sst"],
                "df": result["df_total"],
                "MS": np.nan,
                "F": np.nan,
                "p-value": np.nan,
            },
        ]
    )
    st.dataframe(anova_table, hide_index=True, width="stretch")

    if result["sums_of_squares_match"]:
        st.success(
            f"SST check passed: SSB + SSW = {_format_number(result['sst'])} "
            f"and direct SST = {_format_number(result['direct_sst'])}."
        )
    else:
        st.error("SST verification failed; inspect the calculation before interpreting.")

    st.subheader("Manual calculation vs. SciPy")
    verification = pd.DataFrame(
        [
            {
                "Quantity": "F statistic",
                "Manual calculation": result["f_statistic"],
                "scipy.stats.f_oneway": result["scipy_f"],
            },
            {
                "Quantity": "p-value",
                "Manual calculation": result["p_value"],
                "scipy.stats.f_oneway": result["scipy_p"],
            },
        ]
    )
    st.dataframe(verification, hide_index=True, width="stretch")
    if result["verification_matches"]:
        st.success(
            f"Results match (within numerical tolerance). Reference: "
            f"SciPy {result['scipy_version']}."
        )
    else:
        st.error(
            "Results do not match the independent SciPy calculation. "
            "Do not interpret the result until this discrepancy is resolved."
        )

    st.subheader("Hypothesis-test decision")
    if not result["decision_defined"]:
        st.warning(_decision_text(result))
    elif result["reject_null"]:
        st.success(_decision_text(result))
    else:
        st.info(_decision_text(result))
    st.write(
        f"Compare F = {_format_number(result['f_statistic'])} with critical F = "
        f"{_format_number(result['f_critical'])}; equivalently, compare p = "
        f"{_format_number(result['p_value'])} with alpha = {result['alpha']:.2f}."
    )
    st.subheader("Effect size")
    st.write(_effect_interpretation(result["eta_squared"]))

    if result["reject_null"]:
        st.subheader("Tukey HSD post-hoc comparisons")
        st.caption(
            "Tukey's honestly significant difference test uses the family-wise "
            f"confidence level {100 * (1 - result['alpha']):g}%."
        )
        st.dataframe(
            pd.DataFrame(result["pairwise"]),
            hide_index=True,
            width="stretch",
        )
        if result["pairwise"]:
            significant_pairs = [
                f"{pair['Group 1']} vs {pair['Group 2']}"
                for pair in result["pairwise"]
                if pair["Significant"]
            ]
            if significant_pairs:
                st.write("Pairs with significant differences: " + "; ".join(significant_pairs))
            else:
                st.write("No pair is significant after Tukey's multiple-comparison adjustment.")

    _show_assumptions(result)
    _show_charts(result)
    st.subheader("Show the working")
    _show_working(result)

    if result["assumption_issues"]:
        if st.checkbox("Run the optional Kruskal-Wallis alternative"):
            kruskal = stats.kruskal(*(result["groups"][name] for name in result["names"]))
            st.write(
                f"Kruskal-Wallis H = {_format_number(float(kruskal.statistic))}; "
                f"p = {_format_number(float(kruskal.pvalue))}. "
                "This rank-based test does not test equality of means."
            )

    st.subheader("Final summary")
    summary = (
        f"**F({result['df_between']}, {result['df_within']}) = "
        f"{_format_number(result['f_statistic'])}; p = "
        f"{_format_number(result['p_value'])}; alpha = {result['alpha']:.2f}.**  \n"
        f"{_decision_text(result)}  \n"
        f"{_effect_interpretation(result['eta_squared'])}"
    )
    if not result["decision_defined"]:
        st.warning(summary)
    elif result["reject_null"]:
        st.success(summary)
    else:
        st.info(summary)


if __name__ == "__main__":
    main()
