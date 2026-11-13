"""BANKING77 support-manager workspace. Loads frozen results and models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_data import (category_counts, csv_bytes, probability_table, representatives,
                      route_export, threshold_table)
from common import ROOT, read_json
from models import Router

TEAL = "#087f8c"
NAVY = "#243d5a"
METHOD_NAMES = {"kmeans": "K-means", "gmm": "Gaussian mixture"}
SPLIT_NAMES = {"fit": "Fit", "selection": "Selection", "calibration": "Calibration",
               "test": "Official test"}
MODEL_NAMES = {"frequency": "Category-frequency baseline", "tfidf": "TF-IDF classifier",
               "encoder": "Encoder classifier", "route": "Route classifier"}


def table(path):
    return pd.read_csv(ROOT / path)


def chart(figure):
    figure.update_layout(
        font={"family": "Inter, sans-serif", "color": "#172b42"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 15, "r": 20, "t": 25, "b": 20},
        legend_title_text="", colorway=[TEAL, NAVY],
    )
    st.plotly_chart(figure, use_container_width=True)


def download(label, frame, filename, key):
    st.download_button(label, csv_bytes(frame), file_name=filename,
                       mime="text/csv", key=key, on_click="ignore")


def percent(value):
    return "Undefined" if pd.isna(value) else f"{value:.2%}"


def rule_text(cfg):
    threshold = cfg["review_threshold"]
    if threshold is None:
        return "Review every enquiry"
    return f"Automatic route at confidence ≥ {threshold:.0%}"


def message_table(rows):
    columns = [column for column in [
        "source_id", "text", "category", "true_intent", "predicted_intent", "confidence",
        "source_file", "source_csv_line",
    ] if column in rows]
    st.dataframe(rows[columns], hide_index=True, width="stretch", row_height=65,
                 column_config={
                     "source_id": "Source ID", "text": st.column_config.TextColumn("Enquiry", width="large"),
                     "category": "Recorded category", "true_intent": "Recorded category",
                     "predicted_intent": "Predicted category",
                     "confidence": st.column_config.NumberColumn("Confidence", format="percent"),
                     "source_file": "Source file", "source_csv_line": "Source CSV line",
                 })


def data_overview(cfg, names):
    st.title("Data overview")
    st.caption("Browse the development messages and their original source references.")
    left, right = st.columns(2)
    split = left.selectbox("Development subset", ["fit", "selection", "calibration"],
                           format_func=SPLIT_NAMES.get, key="data_split")
    category = right.selectbox("Category", ["all", *cfg["category_order"]],
                               format_func=lambda label: "All categories" if label == "all" else names[label],
                               key="data_category")
    subset = table(f"data/development/{split}.csv")
    rows = subset if category == "all" else subset[subset.category.eq(category)]
    first, second, third = st.columns(3)
    first.metric("Messages", f"{len(rows):,}")
    second.metric("Categories", rows.category.nunique())
    third.metric("Median words", f"{rows.word_count.median():.0f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Category counts")
        counts = category_counts(rows, cfg["category_order"], names)
        figure = px.bar(counts, x="count", y="display_name", orientation="h",
                        labels={"count": "Messages", "display_name": ""}, color_discrete_sequence=[TEAL])
        figure.update_layout(height=420, yaxis={"autorange": "reversed"})
        chart(figure)
    with right:
        st.subheader("Text length")
        unit = st.radio("Measure length in", ["Words", "Characters"], horizontal=True, key="length_unit")
        column = "word_count" if unit == "Words" else "char_length"
        figure = px.histogram(rows, x=column, nbins=25, labels={column: unit},
                              color_discrete_sequence=[TEAL])
        figure.update_layout(height=360, yaxis_title="Messages", bargap=.08)
        chart(figure)
    st.subheader("Source enquiries")
    st.caption(f"{SPLIT_NAMES[split]} · {len(rows):,} messages · original text and recorded labels")
    message_table(rows)
    download("Download displayed messages", rows, f"{split}_{category}_messages.csv", "data_download")
    with st.expander("Split and duplicate audit"):
        audit = read_json(ROOT / "data/development_summary.json")
        st.write(f"{audit['duplicate_groups_retained']} duplicate groups containing "
                 f"{audit['messages_in_duplicate_groups']} messages are kept together. "
                 f"{audit['conflicting_rows_excluded']} conflicting rows were excluded. Split seed is 42.")
        st.dataframe(pd.DataFrame(audit["counts"]).rename(index=names), width="stretch")


def route_enquiry(cfg, names):
    st.title("Route an enquiry")
    st.caption("Classify a new message using the saved model, adjusted probabilities and review rule.")
    with st.form("route_form"):
        text = st.text_area("Customer enquiry", height=150, max_chars=20000,
                            placeholder="My card has not arrived.", key="route_text")
        submitted = st.form_submit_button("Route enquiry", type="primary")
    if submitted:
        st.session_state.pop("route_result", None)
        try:
            with st.spinner("Classifying the enquiry…"):
                st.session_state.route_result = Router().predict(text)
        except ValueError as error:
            st.error(str(error))
        except (OSError, RuntimeError) as error:
            st.error(f"Routing is unavailable. {error}")
    result = st.session_state.get("route_result")
    if result is None:
        st.caption(f"{MODEL_NAMES[cfg['chosen_method']]} · T = {cfg['temperature']:.6f} · {rule_text(cfg)}")
        return
    category = names[result["suggested_category"]]
    if result["status"] == "automatic route":
        st.success(f"Automatic route · {category}")
    else:
        st.warning(f"Needs human review · Suggested category is {category}")
    st.subheader("Top three categories")
    for column, entry in zip(st.columns(3), result["top_three"]):
        column.metric(names[entry["category"]], percent(entry["probability"]))
    st.caption(f"Confidence {percent(result['confidence'])} · T = {result['temperature']:.6f} · {result['review_rule']}")
    st.caption(f"Model {result['model_id']} · Run {result['run_id']}")
    with st.expander("Submitted enquiry", expanded=True):
        st.text(result["text"])
    st.subheader("All category probabilities")
    probabilities = probability_table(result, names)
    figure = px.bar(probabilities, x="probability", y="display_name", orientation="h",
                    color_discrete_sequence=[TEAL], labels={"probability": "Adjusted probability", "display_name": ""})
    figure.update_layout(height=440, xaxis={"range": [0, 1], "tickformat": ".0%"},
                         yaxis={"autorange": "reversed"})
    chart(figure)
    st.dataframe(probabilities, hide_index=True, width="stretch", column_config={
        "category": "Category ID", "display_name": "Category",
        "probability": st.column_config.NumberColumn("Adjusted probability", format="percent"),
    })
    download("Download routing result", route_export(result), "routing_result.csv", "route_download")


def explore_groups(cfg, names):
    st.title("Explore request groups")
    st.caption("Inspect saved seed-0 groups built from fit messages in the shared PCA space.")
    default = cfg["clustering"]["default"]
    first, second, third = st.columns(3)
    methods = cfg["clustering"]["methods"]
    counts = cfg["clustering"]["counts"]
    method = first.selectbox("Method", methods, index=methods.index(default["method"]),
                             format_func=METHOD_NAMES.get, key="group_method")
    k = second.selectbox("Number of groups", counts, index=counts.index(default["k"]), key="group_k")
    group = third.selectbox("Group", range(k), format_func=lambda value: f"Group {value}", key="group_id")
    fit = table("data/development/fit.csv")
    assignments = table("results/clusters.csv")
    run = assignments[assignments.method.eq(method) & assignments.k.eq(k) & assignments.seed.eq(0)]
    selected = run[run.cluster_id.eq(group)]
    st.caption(f"Run {method}_k{k}_seed0 · {len(fit):,} fit messages · 20 PCA dimensions")
    members_tab, quality_tab = st.tabs(["Group contents", "Group quality"])
    with members_tab:
        left, right = st.columns([1, 2])
        with left:
            st.subheader("Groups and sizes")
            sizes = run.cluster_id.value_counts().reindex(range(k), fill_value=0)
            st.dataframe(pd.DataFrame({"Group": sizes.index, "Messages": sizes.values}),
                         hide_index=True, width="stretch", height=350)
        with right:
            st.subheader(f"Group {group}")
            st.metric("Messages in selected group", len(selected))
            categories = category_counts(selected, cfg["category_order"], names)
            figure = px.bar(categories, x="count", y="display_name", orientation="h",
                            labels={"count": "Messages", "display_name": ""}, color_discrete_sequence=[TEAL])
            figure.update_layout(height=340, yaxis={"autorange": "reversed"})
            chart(figure)
        st.subheader("Representative enquiries")
        if selected.empty:
            st.info("This group is empty.")
        else:
            st.caption("Five assigned messages nearest the centre in PCA space." if method == "kmeans"
                       else "Five assigned messages with the highest membership probability.")
            examples = representatives(fit, assignments, method, k, group)
            message_table(examples)
            download("Download representatives", examples, f"{method}_k{k}_group{group}_representatives.csv", "reps_download")
        download("Download group assignments", run, f"{method}_k{k}_seed0_assignments.csv", "assignments_download")
    with quality_tab:
        quality = table("results/clustering_results.csv")
        row = quality[quality.method.eq(method) & quality.k.eq(k)].iloc[0]
        first, second, third = st.columns(3)
        first.metric("Mean stability (ARI)", f"{row.mean_ari:.4f}")
        second.metric("Mutual information", f"{row.mutual_information:.4f} nats")
        third.metric("Category entropy", f"{row.entropy_y:.4f} nats")
        st.subheader("All grouping comparisons")
        st.dataframe(quality, hide_index=True, width="stretch")
        shuffled = table("results/clustering_shuffles.csv")
        shuffled = shuffled[shuffled.method.eq(method) & shuffled.k.eq(k)]
        st.subheader("Association against shuffled labels")
        frequencies, edges = np.histogram(shuffled.mutual_information, bins=20)
        figure = go.Figure(go.Bar(
            x=((edges[:-1] + edges[1:]) / 2).tolist(), y=frequencies.tolist(),
            width=np.diff(edges).tolist(), marker_color=TEAL, name="Shuffled MI",
        ))
        figure.add_trace(go.Scatter(
            x=[row.mutual_information, row.mutual_information],
            y=[0, int(frequencies.max()) + 1], mode="lines",
            line={"color": NAVY}, name=f"Observed MI ({row.mutual_information:.4f})",
        ))
        figure.update_layout(yaxis_title="Shuffles", xaxis_type="log",
                             xaxis_title="Mutual information (nats, log scale)",
                             xaxis_tickvals=[.01, .02, .05, .1, .2, .5, 1, 2],
                             xaxis_ticktext=["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"],
                             legend={"orientation": "h", "y": 1.1, "x": 0})
        chart(figure)
        st.caption(f"100 shuffles · Seed 42 · Mean {row.shuffled_mean:.4f} · "
                   f"95% reference interval [{row.shuffled_low:.4f}, {row.shuffled_high:.4f}] nats")
        download("Download group quality", quality, "clustering_results.csv", "quality_download")
        download("Download selected shuffle results", shuffled, f"{method}_k{k}_shuffles.csv", "shuffles_download")
        with st.expander("Saved default grouping"):
            st.write(f"{METHOD_NAMES[default['method']]} with {default['k']} groups, seed 0.")
            st.caption(default["criterion"])


def scores_panel(cfg, names, model_id, split, metrics):
    row = metrics[metrics.model_id.eq(model_id) & metrics.split.eq(split)].iloc[0]
    for column, label, value in zip(st.columns(4), ["Messages", "Accuracy", "Macro-F1", "Mean log-loss"],
                                    [str(row.n), percent(row.accuracy), f"{row.macro_f1:.4f}", f"{row.mean_log_loss:.6f}"]):
        column.metric(label, value)
    st.caption(f"Model {model_id} · Run {row.run_id}")
    per_category = table("results/per_category.csv")
    per_category = per_category[per_category.model_id.eq(model_id) & per_category.split.eq(split)]
    st.subheader("Per-category scores")
    st.dataframe(per_category[["category", "precision", "recall", "f1", "support"]],
                 hide_index=True, width="stretch")
    confusion = table("results/confusion.csv")
    confusion = confusion[confusion.model_id.eq(model_id) & confusion.split.eq(split)]
    matrix = confusion.pivot(index="true_intent", columns="predicted_intent", values="count").reindex(
        index=cfg["category_order"], columns=cfg["category_order"])
    st.subheader("Confusion counts")
    figure = px.imshow(matrix.to_numpy(), x=[names[label] for label in matrix.columns],
                       y=[names[label] for label in matrix.index], text_auto=True,
                       color_continuous_scale=["#edf5f5", TEAL], aspect="auto",
                       labels={"x": "Predicted category", "y": "Recorded category", "color": "Messages"})
    figure.update_layout(height=610, xaxis={"tickangle": -40})
    chart(figure)
    predictions = table(f"results/{'official_test' if split == 'test' else 'development/' + split}/predictions.csv")
    predictions = predictions[predictions.model_id.eq(model_id)]
    mistakes = predictions[predictions.true_intent.ne(predictions.predicted_intent)].head(10)
    st.subheader("Misclassified enquiries")
    if mistakes.empty:
        st.success("There are no misclassified messages for this model and subset.")
    else:
        st.caption("Up to ten actual mistakes from the selected model and subset.")
        message_table(mistakes)
    with st.expander("Download comparison tables"):
        download("Download scores", pd.DataFrame([row]), f"{model_id}_{split}_scores.csv", "scores_download")
        download("Download per-category scores", per_category, f"{model_id}_{split}_categories.csv", "per_category_download")
        download("Download confusion counts", confusion, f"{model_id}_{split}_confusion.csv", "confusion_download")
        download("Download predictions", predictions, f"{model_id}_{split}_predictions.csv", "predictions_download")
        download("Download displayed mistakes", mistakes, f"{model_id}_{split}_mistakes.csv", "mistakes_download")


def selection_panel(cfg):
    st.subheader("Training and selection")
    st.dataframe(table("results/selection_results.csv"), hide_index=True, width="stretch")
    first, second, third, fourth = st.columns(4)
    first.metric("Chosen method", "Encoder" if cfg["chosen_method"] == "encoder" else "TF-IDF")
    second.metric("Selected C", cfg["selected_C"][cfg["chosen_method"]])
    third.metric("Temperature", f"{cfg['temperature']:.6f}")
    fourth.metric("Review threshold", "Review all" if cfg["review_threshold"] is None else percent(cfg["review_threshold"]))
    st.caption(f"Tie rule · {cfg['selection_ties']}")
    st.caption(f"Temperature optimiser · bounded minimisation of mean calibration loss over log(T) "
               f"in {cfg['temperature_log_bounds']}")
    download("Download selection results", table("results/selection_results.csv"), "selection_results.csv", "selection_download")


def confidence_panel(cfg, split, metrics):
    st.subheader(f"{MODEL_NAMES[cfg['chosen_method']]} confidence")
    original_id = cfg["model_ids"][cfg["chosen_method"]]
    route_id = cfg["route_model_id"]
    selected_metrics = metrics[metrics.split.eq(split)].set_index("model_id")
    first, second, third = st.columns(3)
    first.metric("Original mean log-loss", f"{selected_metrics.loc[original_id, 'mean_log_loss']:.6f}")
    second.metric("Adjusted mean log-loss", f"{selected_metrics.loc[route_id, 'mean_log_loss']:.6f}")
    third.metric("Saved temperature", f"{cfg['temperature']:.6f}")
    calibration = table("results/calibration.csv")
    bins = calibration[calibration.kind.eq("confidence_bin") & calibration.split.eq(split)]
    bins = bins[bins.model_id.isin([original_id, route_id])].copy()
    bins["Probabilities"] = bins.model_id.map({original_id: "Original", route_id: "Adjusted"})
    figure = px.line(bins[bins["count"].gt(0)], x="mean_confidence", y="fraction_correct",
                     color="Probabilities", markers=True,
                     color_discrete_map={"Original": NAVY, "Adjusted": TEAL},
                     labels={"mean_confidence": "Mean confidence", "fraction_correct": "Fraction correct"})
    figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                     line={"color": "#a5b2bf", "dash": "dash"})
    figure.update_layout(xaxis={"range": [0, 1], "tickformat": ".0%"},
                         yaxis={"range": [0, 1.02], "tickformat": ".0%"})
    chart(figure)
    st.dataframe(bins[["Probabilities", "bin", "lower", "upper", "count", "mean_confidence", "fraction_correct"]],
                 hide_index=True, width="stretch")
    st.caption("All ten bins are included for each version. Empty-bin means and fractions are undefined.")
    download("Download reliability bins", bins.drop(columns="Probabilities"), f"{split}_reliability.csv", "bins_download")
    st.subheader("Calibration rule selection")
    evidence = threshold_table(calibration, "calibration", route_id, cfg["review_threshold"])
    st.dataframe(evidence, hide_index=True, width="stretch")
    st.caption(f"Smallest threshold with at least 20 accepted and at most 5% accepted error. {rule_text(cfg)}.")
    trials = threshold_table(calibration, split, route_id, cfg["review_threshold"])
    if split != "calibration":
        st.subheader(f"{SPLIT_NAMES[split]} threshold outcomes")
        st.dataframe(trials, hide_index=True, width="stretch")
    figure = px.line(trials, x="coverage", y="error", text="threshold", markers=True,
                     labels={"coverage": "Coverage", "error": "Accepted error rate"})
    chosen = trials[trials.saved_rule]
    figure.add_trace(go.Scatter(x=chosen.coverage, y=chosen.error, mode="markers", name="Saved threshold",
                               marker={"size": 16, "symbol": "diamond-open", "color": NAVY}))
    figure.update_traces(textposition="top center", selector={"mode": "markers+lines+text"})
    figure.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".1%")
    chart(figure)
    fixed = calibration[calibration.kind.eq("saved_rule") & calibration.split.eq(split)
                        & calibration.model_id.eq(route_id)].iloc[0]
    st.subheader(f"Saved-rule outcome · {SPLIT_NAMES[split]}")
    for column, label, value in zip(st.columns(5),
                                    ["Accepted", "Needs review", "Wrong accepted", "Coverage", "Accepted error"],
                                    [str(int(fixed.accepted)), str(int(fixed.review_required)),
                                     str(int(fixed.wrong_accepted)), percent(fixed.coverage), percent(fixed.error)]):
        column.metric(label, value)
    download("Download review outcomes", trials, f"{split}_review_thresholds.csv", "review_download")
    download("Download calibration evidence", evidence, "calibration_review_selection.csv", "evidence_download")
    download("Download saved-rule outcome", pd.DataFrame([fixed]), f"{split}_saved_rule.csv", "fixed_download")


def bootstrap_panel():
    st.subheader("TF-IDF versus encoder")
    st.caption("Fixed official-test comparison · Original probabilities · TF-IDF loss minus encoder loss")
    results = table("results/bootstrap.csv")
    row = results.iloc[0]
    first, second, third = st.columns(3)
    first.metric("TF-IDF mean test loss", f"{row.tfidf_loss:.6f}")
    second.metric("Encoder mean test loss", f"{row.encoder_loss:.6f}")
    third.metric("Mean difference Δ", f"{row.delta:.6f}")
    st.write(f"95% paired-bootstrap interval [{row.ci_low:.6f}, {row.ci_high:.6f}].")
    if row.ci_low > 0:
        st.info("The interval is wholly above zero, supporting lower encoder loss on comparable enquiries.")
    elif row.ci_high < 0:
        st.info("The interval is wholly below zero, supporting lower TF-IDF loss on comparable enquiries.")
    else:
        st.info("The interval crosses zero. The direction is uncertain under this bootstrap procedure.")
    st.caption(f"{row.tfidf_model_id} (C={row.tfidf_C:g}) · {row.encoder_model_id} (C={row.encoder_C:g}) · "
               f"n={row.n} · {row.resamples} resamples · Seed {row.seed}")
    st.caption("Each resample uses the saved test losses. The interval measures test-sample variation and assumes independent, comparable enquiries.")
    download("Download bootstrap comparison", results, "bootstrap.csv", "bootstrap_download")


def compare_models(cfg, names):
    st.title("Compare models")
    first, second = st.columns([2, 1])
    method = first.selectbox("Model", list(MODEL_NAMES), index=3, format_func=MODEL_NAMES.get, key="compare_model")
    split = second.selectbox("Subset", list(SPLIT_NAMES), index=3, format_func=SPLIT_NAMES.get, key="compare_split")
    st.caption(f"{MODEL_NAMES[method]} · {SPLIT_NAMES[split]} · Saved evaluation results")
    metrics = table("results/metrics.csv")
    scores, selection, confidence, bootstrap = st.tabs([
        "Scores and mistakes", "Training and selection", "Confidence and review", "TF-IDF versus encoder",
    ])
    with scores:
        scores_panel(cfg, names, cfg["model_ids"][method], split, metrics)
    with selection:
        selection_panel(cfg)
    with confidence:
        confidence_panel(cfg, split, metrics)
    with bootstrap:
        bootstrap_panel()


def analysis_agent(cfg, names):
    st.title("Ask the analysis agent")
    st.info("The API analysis agent will be connected in checkpoint 3. Live agent evaluation is pending your API setup.")
    st.text_area("Analysis question", height=130, key="analysis_question",
                 placeholder="On selection data, which category pair has the most mistakes? Show three enquiries.")
    st.button("Ask agent", disabled=True, help="Available after the API tools and controller are integrated.")
    st.caption("The connected agent will show its answer, supporting source/run IDs, and an expandable tool-call trace.")


def main():
    st.set_page_config(page_title="Enquiry desk · BANKING77", page_icon="🏦", layout="wide")
    st.html("""<style>
        .stMainBlockContainer {max-width: 1360px; padding-top: 2.5rem;}
        h1 {letter-spacing: -.035em; font-weight: 700;}
        h2, h3 {letter-spacing: -.02em;}
        [data-testid="stMetric"] {background: #fff; padding: 1rem 1.2rem;
            border: 1px solid #e3e9ef; border-radius: 10px;}
        [data-testid="stSidebar"] {border-right: 1px solid #e3e9ef;}
        [data-testid="stSidebar"] h1 {font-size: 1.5rem;}
        button[kind="primary"] {border-radius: 7px;}
    </style>""")
    cfg = read_json(ROOT / "config.json")
    names = {entry["label"]: entry["display_name"] for entry in read_json(ROOT / "intents.json")["intents"]}
    pages = {"Data overview": data_overview, "Route an enquiry": route_enquiry,
             "Explore request groups": explore_groups, "Compare models": compare_models,
             "Ask the analysis agent": analysis_agent}
    st.sidebar.title("Enquiry desk")
    st.sidebar.caption("BANKING77 · Support operations")
    screen = st.sidebar.radio("Workspace", list(pages), key="screen", label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Saved Route model")
    st.sidebar.write(cfg["route_model_id"])
    st.sidebar.caption(rule_text(cfg))
    try:
        pages[screen](cfg, names)
    except FileNotFoundError as error:
        st.error(f"A required saved project file is missing. {error}")


if __name__ == "__main__":
    main()
