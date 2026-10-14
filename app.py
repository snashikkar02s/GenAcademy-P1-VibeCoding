"""Jenkins Deployment Audit Dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import calendar
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_FILE = Path(__file__).parent / "jenkins-data.csv"
REQUIRED_COLUMNS = {"Build #", "Result", "Start Time (UTC)", "User"}
MODEL = "claude-sonnet-5-5"

# Status colors (good / critical / neutral) and a single-hue blue ramp for the heat map.
COLOR = {"Success": "#0ca30c", "Failure": "#d03b3b", "Other": "#8a8984"}
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
             "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab"]
OUTCOMES = ["Success", "Failure", "Other"]

st.set_page_config(page_title="Jenkins Deployment Audit", page_icon="🛠️", layout="wide")


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_data(raw: bytes) -> pd.DataFrame:
    """Parse a Jenkins CSV export into a clean frame. Raises ValueError on bad input."""
    if not raw.strip():
        raise ValueError("The CSV file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except Exception as exc:  # malformed CSV
        raise ValueError(f"Could not parse the CSV: {exc}") from exc

    df.columns = [c.strip() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    df["Start Time (UTC)"] = pd.to_datetime(df["Start Time (UTC)"], errors="coerce")
    df = df.dropna(subset=["Start Time (UTC)"])
    if df.empty:
        raise ValueError("No rows with a valid 'Start Time (UTC)' were found.")

    df["User"] = df["User"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    df["Result"] = df["Result"].fillna("").astype(str).str.strip().str.upper()
    df["Result"] = df["Result"].replace("", "UNKNOWN")
    df["Outcome"] = df["Result"].map({"SUCCESS": "Success", "FAILURE": "Failure"}).fillna("Other")

    if "Building" in df.columns:
        df["Building"] = df["Building"].astype(str).str.strip().str.upper().eq("TRUE")
    else:
        df["Building"] = False

    if "Duration (min)" not in df.columns and "Duration (ms)" in df.columns:
        df["Duration (min)"] = df["Duration (ms)"] / 60000
    if "Duration (min)" in df.columns:
        df["Duration (min)"] = pd.to_numeric(df["Duration (min)"], errors="coerce")
    else:
        df["Duration (min)"] = float("nan")

    df["Year"] = df["Start Time (UTC)"].dt.year
    df["Month"] = df["Start Time (UTC)"].dt.month
    return df.sort_values("Start Time (UTC)", ascending=False).reset_index(drop=True)


def compute_kpis(df: pd.DataFrame) -> dict:
    """Headline counts and rates over completed (not still-building) deployments."""
    done = df[~df["Building"]]
    total = len(done)
    counts = done["Outcome"].value_counts().reindex(OUTCOMES, fill_value=0)

    def pct(n: int) -> float:
        return round(100 * float(n) / total, 1) if total else 0.0

    return {
        "total": total,
        "success": int(counts["Success"]),
        "failure": int(counts["Failure"]),
        "other": int(counts["Other"]),
        "success_pct": pct(counts["Success"]),
        "failure_pct": pct(counts["Failure"]),
        "other_pct": pct(counts["Other"]),
        "in_progress": int(df["Building"].sum()),
        "avg_duration": round(float(done["Duration (min)"].mean()), 1) if total else 0.0,
        "max_duration": round(float(done["Duration (min)"].max()), 1) if total else 0.0,
        "start": df["Start Time (UTC)"].min(),
        "end": df["Start Time (UTC)"].max(),
    }


def monthly_pivot(df: pd.DataFrame, year: int, mode: str) -> pd.DataFrame:
    """User x month table (Jan..Oct, extended if later months exist) with totals.

    mode: "Deployments" (counts), "Failures" (counts), or "Success %".
    """
    done = df[(~df["Building"]) & (df["Year"] == year)]
    last_month = max(10, int(done["Month"].max()) if not done.empty else 10)
    months = list(range(1, last_month + 1))
    labels = [calendar.month_abbr[m] for m in months]

    def count_table(frame: pd.DataFrame) -> pd.DataFrame:
        t = pd.crosstab(frame["User"], frame["Month"]).reindex(columns=months, fill_value=0)
        t.columns = labels
        t["Total"] = t.sum(axis=1)
        t = t.sort_values("Total", ascending=False)
        t.loc["Grand Total"] = t.sum()
        return t.astype(int)

    all_counts = count_table(done) if not done.empty else pd.DataFrame(columns=labels + ["Total"])
    if mode == "Deployments" or done.empty:
        table = all_counts
    elif mode == "Failures":
        table = count_table(done[done["Outcome"] == "Failure"]) if (done["Outcome"] == "Failure").any() \
            else all_counts * 0
        table = table.reindex(index=all_counts.index, fill_value=0)
    else:  # Success %
        ok = count_table(done[done["Outcome"] == "Success"]).reindex(
            index=all_counts.index, columns=all_counts.columns, fill_value=0)
        table = (100 * ok / all_counts.where(all_counts > 0)).round(1)

    table.index.name = "Started by"
    return table


# --------------------------------------------------------------------------- charts
def _layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      bargap=0.35, hoverlabel=dict(namelength=-1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def donut_chart(k: dict) -> go.Figure:
    values = [k["success"], k["failure"], k["other"]]
    labels = [o for o, v in zip(OUTCOMES, values) if v]
    vals = [v for v in values if v]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.62, sort=False,
        marker=dict(colors=[COLOR[l] for l in labels], line=dict(color="rgba(255,255,255,0.9)", width=2)),
        textinfo="percent", texttemplate="%{percent:.1%}",
        hovertemplate="<b>%{label}</b><br>%{value} deployments<br>%{percent:.1%}<extra></extra>",
    ))
    fig.update_layout(annotations=[dict(text=f"<b>{k['success_pct']}%</b><br>success",
                                        showarrow=False, font=dict(size=18))])
    return _layout(fig)


def user_chart(df: pd.DataFrame) -> go.Figure:
    done = df[~df["Building"]]
    t = pd.crosstab(done["User"], done["Outcome"]).reindex(columns=OUTCOMES, fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values().index]
    totals = t.sum(axis=1)
    fig = go.Figure()
    for outcome in OUTCOMES:
        if t[outcome].sum() == 0:
            continue
        pct = (100 * t[outcome] / totals).round(1)
        fig.add_bar(y=t.index, x=t[outcome], name=outcome, orientation="h",
                    marker=dict(color=COLOR[outcome], line=dict(color="rgba(255,255,255,0.9)", width=2)),
                    customdata=pct,
                    hovertemplate="<b>%{y}</b><br>" + outcome + ": %{x} (%{customdata}%)<extra></extra>")
    fig.update_layout(barmode="stack", xaxis_title="Deployments")
    return _layout(fig)


def monthly_chart(df: pd.DataFrame) -> go.Figure:
    done = df[~df["Building"]].copy()
    done["Period"] = done["Start Time (UTC)"].dt.to_period("M")
    t = pd.crosstab(done["Period"], done["Outcome"]).reindex(columns=OUTCOMES, fill_value=0).sort_index()
    x = [p.strftime("%b %Y") for p in t.index]
    totals = t.sum(axis=1)
    fig = go.Figure()
    for outcome in OUTCOMES:
        if t[outcome].sum() == 0:
            continue
        fig.add_bar(x=x, y=t[outcome], name=outcome,
                    marker=dict(color=COLOR[outcome], line=dict(color="rgba(255,255,255,0.9)", width=2)),
                    hovertemplate="<b>%{x}</b><br>" + outcome + ": %{y}<extra></extra>")
    rate = (100 * t["Success"] / totals).round(1)
    for xi, tot, r in zip(x, totals, rate):
        fig.add_annotation(x=xi, y=tot, text=f"{r}% success", showarrow=False, yshift=12,
                           font=dict(size=12))
    fig.update_layout(barmode="stack", yaxis_title="Deployments")
    return _layout(fig)


def render_charts(df: pd.DataFrame, k: dict) -> None:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("Outcome share")
        st.plotly_chart(donut_chart(k), width="stretch")
    with c2:
        st.subheader("Outcomes by user")
        st.plotly_chart(user_chart(df), width="stretch")
    st.subheader("Monthly trend")
    st.plotly_chart(monthly_chart(df), width="stretch")


def heat_style(table: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Single-hue gradient on the user x month cells; totals row/column stay plain."""
    body = table.drop(index="Grand Total", errors="ignore").drop(columns="Total", errors="ignore")
    vmax = body.max().max() if body.size else 0

    def css(v, row, col):
        if row == "Grand Total" or col == "Total":
            return "font-weight: 600;"
        if pd.isna(v) or not vmax or v == 0:
            return ""
        step = min(len(BLUE_RAMP) - 1, int((v / vmax) * (len(BLUE_RAMP) - 1)))
        ink = "#0b0b0b" if step < 6 else "#ffffff"
        return f"background-color: {BLUE_RAMP[step]}; color: {ink};"

    styles = pd.DataFrame([[css(table.at[r, c], r, c) for c in table.columns] for r in table.index],
                          index=table.index, columns=table.columns)
    fmt = "{:.1f}%" if table.dtypes.eq(float).any() else "{:,.0f}"
    return table.style.apply(lambda _: styles, axis=None).format(fmt, na_rep="–")


def to_excel(table: pd.DataFrame, failures: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name="Monthly by user")
        failures.to_excel(xw, sheet_name="Failed builds", index=False)
    return buf.getvalue()


# --------------------------------------------------------------------------- AI
def _secrets_key() -> str | None:
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets.toml
        return None


def _has_credentials() -> bool:
    return bool(_secrets_key() or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def audit_stats(df: pd.DataFrame, k: dict) -> dict:
    """Aggregated (not row-level) figures sent to the model."""
    done = df[~df["Building"]]
    per_user = done.groupby("User").agg(
        deployments=("Outcome", "size"),
        failures=("Outcome", lambda s: int((s == "Failure").sum())),
        avg_duration_min=("Duration (min)", "mean"),
        max_duration_min=("Duration (min)", "max"),
    ).round(1)
    per_user["failure_pct"] = (100 * per_user["failures"] / per_user["deployments"]).round(1)
    monthly = done.groupby(done["Start Time (UTC)"].dt.strftime("%Y-%m")).agg(
        deployments=("Outcome", "size"),
        failures=("Outcome", lambda s: int((s == "Failure").sum())),
    )
    dur = done["Duration (min)"].dropna()
    p95 = float(dur.quantile(0.95)) if not dur.empty else 0.0
    return {
        "period_utc": [str(k["start"]), str(k["end"])],
        "totals": {key: k[key] for key in ("total", "success", "failure", "other",
                                           "success_pct", "failure_pct", "in_progress")},
        "duration_min": {"avg": k["avg_duration"], "max": k["max_duration"], "p95": round(p95, 1),
                         "builds_over_p95": int((dur > p95).sum())},
        "per_user": per_user.reset_index().to_dict(orient="records"),
        "per_month": monthly.reset_index(names="month").to_dict(orient="records"),
    }


def ai_summary(stats: dict) -> str:
    import anthropic

    key = _secrets_key()
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    prompt = (
        "You are a release-management auditor. Using only the aggregated Jenkins deployment "
        "statistics below, write a concise audit summary in Markdown with these sections: "
        "**Overall reliability**, **Users to review** (high failure rates or unusual volume), "
        "**Monthly volume** (spikes or gaps), **Build duration outliers**, and "
        "**Recommendations** (3-5 bullets). Quote exact figures. Under 300 words.\n\n"
        f"```json\n{json.dumps(stats, default=str, indent=2)}\n```"
    )
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        return "The model declined to produce a summary for this data."
    return "".join(b.text for b in response.content if b.type == "text").strip()


def render_ai(df: pd.DataFrame, k: dict) -> None:
    st.subheader("🤖 AI audit summary")
    if not _has_credentials():
        st.info("Set `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml` or as an environment "
                "variable to enable AI-generated audit summaries.")
        return
    if st.button("Generate Audit Summary", type="primary"):
        import anthropic

        with st.spinner("Analyzing deployment statistics…"):
            try:
                st.session_state["ai_summary"] = ai_summary(audit_stats(df, k))
            except anthropic.AuthenticationError:
                st.error("The Anthropic API key was rejected. Check that it's valid.")
            except anthropic.RateLimitError:
                st.warning("Rate limited by the Anthropic API. Please try again shortly.")
            except anthropic.APIStatusError as exc:
                st.error(f"Anthropic API error ({exc.status_code}): {exc.message}")
            except anthropic.APIConnectionError:
                st.error("Could not reach the Anthropic API. Check your network connection.")
    if st.session_state.get("ai_summary"):
        st.markdown(st.session_state["ai_summary"])


# --------------------------------------------------------------------------- page
def main() -> None:
    st.title("🛠️ Jenkins Deployment Audit Dashboard")

    with st.sidebar:
        st.header("Data source")
        upload = st.file_uploader("Upload a newer Jenkins CSV export", type="csv")
        if upload is not None:
            raw, source = upload.getvalue(), upload.name
        elif DATA_FILE.exists():
            raw, source = DATA_FILE.read_bytes(), DATA_FILE.name
        else:
            st.error(f"{DATA_FILE.name} not found. Upload a CSV to continue.")
            st.stop()

    try:
        data = load_data(raw)
    except ValueError as exc:
        st.error(f"**{source}**: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Filters")
        lo, hi = data["Start Time (UTC)"].min().date(), data["Start Time (UTC)"].max().date()
        dates = st.date_input("Date range (UTC)", value=(lo, hi), min_value=lo, max_value=hi)
        users = sorted(data["User"].unique())
        sel_users = st.multiselect("Users", users, default=users)
        sel_outcomes = st.multiselect("Result", OUTCOMES, default=OUTCOMES)
        st.caption(f"Source: `{source}`  \nLast refreshed: "
                   f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")

    start, end = (dates if isinstance(dates, tuple) and len(dates) == 2 else (lo, hi))
    day = data["Start Time (UTC)"].dt.date
    df = data[(day >= start) & (day <= end)
              & data["User"].isin(sel_users) & data["Outcome"].isin(sel_outcomes)]

    if df.empty:
        st.warning("No deployments match the current filters.")
        st.stop()

    k = compute_kpis(df)
    st.caption(f"Covering **{k['start']:%d %b %Y}** to **{k['end']:%d %b %Y}** (UTC)"
               + (f" · {k['in_progress']} build(s) still in progress, excluded from counts"
                  if k["in_progress"] else ""))

    # KPI row
    cols = st.columns(6)
    cols[0].metric("Total deployments", f"{k['total']:,}")
    cols[1].metric("✅ Successful", f"{k['success']:,}", f"{k['success_pct']}%", delta_color="off")
    cols[2].metric("❌ Failed", f"{k['failure']:,}", f"{k['failure_pct']}%", delta_color="off")
    cols[3].metric("⚪ Other", f"{k['other']:,}", f"{k['other_pct']}%", delta_color="off")
    cols[4].metric("Success rate", f"{k['success_pct']}%")
    cols[5].metric("Avg duration", f"{k['avg_duration']} min")
    st.progress(k["success_pct"] / 100, text=f"Success rate {k['success_pct']}% · "
                                              f"Failure rate {k['failure_pct']}%")

    st.divider()
    render_charts(df, k)

    # Monthly audit table
    st.divider()
    st.subheader("Monthly deployment summary by user")
    years = sorted(df["Year"].unique(), reverse=True)
    c1, c2 = st.columns([1, 3])
    year = c1.selectbox("Year", years) if len(years) > 1 else years[0]
    mode = c2.radio("Show", ["Deployments", "Success %", "Failures"], horizontal=True)
    table = monthly_pivot(df, int(year), mode)
    st.dataframe(heat_style(table), width="stretch")

    failures = df[(df["Outcome"] == "Failure") & ~df["Building"]][
        ["Build #", "User", "Start Time (UTC)", "Duration (min)", "Result"]]
    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("⬇️ Download CSV", table.to_csv().encode("utf-8"),
                       file_name=f"jenkins_monthly_{mode.replace(' %', '_pct').lower()}_{year}.csv",
                       mime="text/csv")
    d2.download_button("⬇️ Download Excel", to_excel(table, failures),
                       file_name=f"jenkins_audit_{year}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Failure detail
    st.divider()
    st.subheader("Failure audit detail")
    f1, f2 = st.columns([1, 2])
    with f1:
        done = df[~df["Building"]]
        per_user = done.groupby("User").agg(
            Deployments=("Outcome", "size"),
            Failures=("Outcome", lambda s: int((s == "Failure").sum())))
        per_user["Failure %"] = (100 * per_user["Failures"] / per_user["Deployments"]).round(1)
        st.dataframe(per_user.sort_values("Failure %", ascending=False),
                     width="stretch",
                     column_config={"Failure %": st.column_config.ProgressColumn(
                         "Failure %", format="%.1f%%", min_value=0, max_value=100)})
    with f2:
        if failures.empty:
            st.success("No failed builds in the selected range.")
        else:
            st.dataframe(failures, hide_index=True, width="stretch",
                         column_config={"Duration (min)": st.column_config.NumberColumn(format="%.1f")})

    st.divider()
    render_ai(df, k)


if __name__ == "__main__":
    main()
