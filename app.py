import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Spain Top 50: Lifecycle Dashboard", layout="wide")
BASE = Path(__file__).parent
GREEN, RED = "#1DB954", "#e74c3c"
STAGES = ["New Entry", "Growth Phase", "Peak Phase", "Mature Phase", "Decline Phase"]
DUR = ["Shortest 25%", "Short-mid", "Long-mid", "Longest 25%"]
SIZE = ["1 track", "2-5 tracks", "6-12 tracks", "13+ tracks"]


@st.cache_data
def load():
    d = pd.read_csv(BASE / "spain_stage.csv", parse_dates=["date"])
    life = pd.read_csv(BASE / "song_lifecycle.csv")
    d = d.sort_values(["song_id", "date"])
    prev_date = d.groupby("song_id")["date"].shift(1)
    prev_pos = d.groupby("song_id")["position"].shift(1)
    d["daily_move"] = np.where((d["date"] - prev_date).dt.days == 1,
                               (d["position"] - prev_pos).abs(), np.nan)
    return d, life


@st.cache_data
def churn_table(f):
    """Songs entering / leaving each day (only days next to a data day)."""
    by_day = f.groupby("date")["song_id"].apply(set)
    rows, events = [], []
    for prev, cur in zip(by_day.index[:-1], by_day.index[1:]):
        if (cur - prev).days != 1:
            continue
        new = by_day[cur] - by_day[prev]
        rows.append({"date": cur, "entries": len(new),
                     "exits": len(by_day[prev] - by_day[cur])})
        events += [(cur, sid) for sid in new]
    c = pd.DataFrame(rows, columns=["date", "entries", "exits"])
    c["churn_pct"] = c["entries"] / 50 * 100
    c["month"] = c["date"].dt.to_period("M").astype(str)
    e = pd.DataFrame(events, columns=["date", "song_id"])
    return c, e


def songs_view(f, life):
    """One row per song, computed only from the rows you selected."""
    v = f.groupby("song_id").agg(days=("date", "nunique"), first=("date", "min"),
                                 best=("position", "min"),
                                 avg_pop=("popularity", "mean"))
    peak_date = f.loc[f.groupby("song_id")["position"].idxmin()].set_index("song_id")["date"]
    v["to_peak"] = (peak_date - v["first"]).dt.days
    extra = life.set_index("song_id")[["song", "artist", "is_explicit", "album_type",
                                       "duration_group", "album_size"]]
    v = v.join(extra)
    v["label"] = v["song"] + " - " + v["artist"]
    v["content"] = np.where(v["is_explicit"], "Explicit", "Clean")
    return v


def ratio(series, a, b):
    if a in series.index and b in series.index and series[b] > 0:
        return f"{series[a] / series[b]:.2f}"
    return "n/a"


def bar(ax, s, title, color=GREEN):
    ax.bar([str(i) for i in s.index], s.values, color=color)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)


d, life = load()

# ------------------------------ sidebar ------------------------------
st.sidebar.title("Filters")
dmin, dmax = d["date"].min().date(), d["date"].max().date()
dr = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
if not isinstance(dr, (tuple, list)):
    dr = (dr,)
start, end = dr[0], dr[-1]
stage_sel = st.sidebar.multiselect("Lifecycle stage", STAGES, default=STAGES)
expl = st.sidebar.radio("Explicit content", ["All", "Explicit only", "Clean only"])
album_sel = st.sidebar.multiselect("Album type", ["single", "album"], default=["single", "album"])

mask = (d["date"].dt.date >= start) & (d["date"].dt.date <= end) & d["album_type"].isin(album_sel)
if expl == "Explicit only":
    mask &= d["is_explicit"]
elif expl == "Clean only":
    mask &= ~d["is_explicit"]
f_content = d[mask]                          # date + explicit + album filters
f = f_content[f_content["stage"].isin(stage_sel)]   # + stage filter

st.title("Spain Top 50: Content Maturity, Lifecycle & Playlist Rotation")
st.caption("Atlantic Recording Corporation | daily Top 50 playlist, 18 May 2024 to 27 Nov 2025")

if f.empty:
    st.warning("No songs match these filters. Widen the date range or select more options.")
else:
    sv = songs_view(f, life)
    ch, ev = churn_table(f_content)

    rsi_song = f.groupby("song_id")["daily_move"].mean()
    rsi_song = rsi_song[sv.loc[rsi_song.index, "days"] >= 7]
    rsi = (1 - rsi_song / 49).mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Songs in view", f"{len(sv):,}")
    c2.metric("Average days on playlist", f"{sv['days'].mean():.1f}")
    c3.metric("Entry-to-peak time (days)", f"{sv['to_peak'].mean():.1f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Playlist churn rate (% of 50 slots/day)",
              f"{ch['churn_pct'].mean():.1f}" if len(ch) else "n/a")
    c5.metric("Retention Stability Index (0-100)", f"{rsi:.1f}" if rsi == rsi else "n/a")
    c6.metric("Single vs album longevity ratio",
              ratio(sv.groupby("album_type")["days"].mean(), "single", "album"))
    st.metric("Explicit content lifecycle score (explicit days / clean days)",
              ratio(sv.groupby("is_explicit")["days"].mean(), True, False))

    t1, t2, t3, t4, t5 = st.tabs(["1. Song timeline", "2. Entry vs exit",
                                  "3. Lifecycle stages", "4. Content maturity", "5. Churn analytics"])

    # ------------------- 1. song timeline -------------------
    with t1:
        st.subheader("Song lifecycle timeline")
        top = sv.sort_values("days", ascending=False)
        pick = st.multiselect("Choose songs (type to search)", list(top["label"]),
                              default=list(top["label"][:3]))
        if pick:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            for lab in pick:
                sid = sv.index[sv["label"] == lab][0]
                one = f[f["song_id"] == sid]
                ax.plot(one["date"], one["position"], marker=".", label=lab)
            ax.set_ylim(51, 0)
            ax.set_ylabel("Playlist position (1 = top)")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
            sids = sv.index[sv["label"].isin(pick)]
            st.write("Days spent in each lifecycle stage")
            tab = pd.crosstab(f[f["song_id"].isin(sids)]["song"], f["stage"])
            st.dataframe(tab.reindex(columns=[s for s in STAGES if s in tab.columns]))

    # ------------------- 2. entry vs exit -------------------
    with t2:
        st.subheader("Entry vs exit flow")
        if len(ch):
            if (ch["entries"] == ch["exits"]).all():
                st.info("The Top 50 always holds exactly 50 songs, so with every song included, "
                        "songs entering = songs leaving each day (the two lines are identical). "
                        "The useful question is WHO enters, shown in the charts below. "
                        "Pick 'Explicit only' or 'Single' in the filters to see entries and exits separate.")
            fig, ax = plt.subplots(figsize=(11, 3.8))
            ax.plot(ch["date"], ch["entries"].rolling(7, min_periods=1).mean(), color=GREEN, lw=2, label="Entries")
            ax.plot(ch["date"], ch["exits"].rolling(7, min_periods=1).mean(), color=RED, lw=1.5, ls="--", label="Exits")
            ax.set_ylabel("Songs per day (7-day average)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            if len(ev):
                ev2 = ev.merge(life[["song_id", "is_explicit", "album_type"]], on="song_id")
                ev2["month"] = ev2["date"].dt.to_period("M").astype(str)
                ev2["content"] = np.where(ev2["is_explicit"], "Explicit", "Clean")
                a_, b_ = st.columns(2)
                fig, ax = plt.subplots(figsize=(5.5, 4))
                pd.crosstab(ev2["month"], ev2["album_type"]).plot(kind="bar", stacked=True, ax=ax,
                                                                   color=[GREEN, "#145a32"])
                ax.set_title("New entries per month: single vs album")
                ax.legend(fontsize=8)
                a_.pyplot(fig)
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(5.5, 4))
                pd.crosstab(ev2["month"], ev2["content"]).plot(kind="bar", stacked=True, ax=ax,
                                                               color=["#145a32", GREEN])
                ax.set_title("New entries per month: clean vs explicit")
                ax.legend(fontsize=8)
                b_.pyplot(fig)
                plt.close(fig)
            m = ch.groupby("month")[["entries", "exits"]].sum()
            m["net"] = m["entries"] - m["exits"]
            st.write("Monthly totals")
            st.dataframe(m)
        else:
            st.info("Pick a wider date range to see entries and exits.")

    # ------------------- 3. lifecycle stages -------------------
    with t3:
        st.subheader("Lifecycle stage distribution")
        share = (f["stage"].value_counts(normalize=True) * 100).reindex(STAGES).fillna(0)
        pop = f.groupby("stage")["popularity"].mean().reindex(STAGES)
        a, b = st.columns(2)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bar(ax, share, "% of song-days in each stage")
        a.pyplot(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bar(ax, pop.fillna(0), "Average popularity by stage")
        b.pyplot(fig)
        plt.close(fig)
        mo = f.assign(month=f["date"].dt.to_period("M").astype(str))
        mix = (pd.crosstab(mo["month"], mo["stage"], normalize="index") * 100)
        mix = mix.reindex(columns=[s for s in STAGES if s in mix.columns])
        fig, ax = plt.subplots(figsize=(11, 4))
        mix.plot(kind="bar", stacked=True, ax=ax, colormap="Greens")
        ax.set_ylabel("% of song-days")
        ax.set_title("Stage mix by month")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
        st.pyplot(fig)
        plt.close(fig)

    # ------------------- 4. content maturity -------------------
    with t4:
        st.subheader("Content maturity comparisons (average days on playlist)")
        fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
        for a_, col, order in zip(ax, ["content", "album_type", "duration_group", "album_size"],
                                  [None, None, DUR, SIZE]):
            s = sv.groupby(col)["days"].mean()
            if order:
                s = s.reindex([o for o in order if o in s.index])
            bar(a_, s, col)
        ax[0].set_ylabel("Avg days")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.write("Details")
        for col in ["content", "album_type", "duration_group", "album_size"]:
            st.dataframe(sv.groupby(col).agg(songs=("days", "size"),
                                             avg_days=("days", "mean"),
                                             median_days=("days", "median"),
                                             avg_days_to_peak=("to_peak", "mean"),
                                             avg_best_position=("best", "mean")).round(1))

    # ------------------- 5. churn analytics -------------------
    with t5:
        st.subheader("Playlist churn analytics")
        if len(ch):
            moves = f_content.groupby("date")["daily_move"].mean().rename("avg_position_move")
            ch2 = ch.merge(moves.reset_index(), on="date", how="left")
            k1, k2, k3 = st.columns(3)
            k1.metric("Avg songs entering per day", f"{ch['entries'].mean():.2f}")
            k2.metric("Avg churn rate (%)", f"{ch['churn_pct'].mean():.1f}")
            k3.metric("Volatility (std of daily churn %)", f"{ch['churn_pct'].std():.1f}")
            mon = ch2.groupby("month").agg(churn_pct=("churn_pct", "mean"),
                                           avg_position_move=("avg_position_move", "mean")).round(2)
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            bar(ax[0], mon["churn_pct"], "Average churn % by month")
            ax[0].tick_params(axis="x", rotation=90)
            bar(ax[1], mon["avg_position_move"], "Avg daily position move (places)", RED)
            ax[1].tick_params(axis="x", rotation=90)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(mon)
        else:
            st.info("Pick a wider date range to see churn.")

    st.caption("Notes: the data only covers 18 May 2024 to 27 Nov 2025, so songs already on the "
               "playlist on day 1 or still there on the last day have unseen parts of their life. "
               "Four days are missing from the source (14 Mar, 25 Mar, 11 Jul, 13 Aug 2025). "
               "Churn ignores the Stage filter; all other numbers follow every filter.")
