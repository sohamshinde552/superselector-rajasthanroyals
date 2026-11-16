import os
import sys
import subprocess
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SupeRR Selector Season 2 | Pressure Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom RR-inspired CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 20px;
    }
    .rr-card {
        background-color: #f8f9fa;
        border-left: 5px solid #e91e63;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    .rr-metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #718096;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .rr-metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0b132b;
    }
</style>
""", unsafe_allow_html=True)

# Auto-run script.py if output CSVs are not present
required_files = ["output_batters.csv", "output_bowlers.csv", "output_validation.csv"]
if not all(os.path.exists(f) for f in required_files):
    with st.spinner("Executing script.py on ball-by-ball dataset..."):
        subprocess.run([sys.executable, "script.py"], check=True)

@st.cache_data
def load_data():
    batters_df = pd.read_csv("output_batters.csv")
    bowlers_df = pd.read_csv("output_bowlers.csv")
    val_df = pd.read_csv("output_validation.csv")
    return batters_df, bowlers_df, val_df

batters, bowlers, validation = load_data()

# Header banner
st.markdown('<div class="main-header">🏏 Rajasthan Royals SupeRR Selector — Season 2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-Driven Identification & Ranking of the Most Reliable Players Under Pressure (IPL & T20I, 2022–2026)</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/6/60/Rajasthan_Royals_Logo.svg/300px-Rajasthan_Royals_Logo.svg.png", width=120)
st.sidebar.title("Filter & Tuning Controls")
st.sidebar.markdown("---")

min_hp_balls = st.sidebar.slider("Minimum High-Pressure Balls", min_value=100, max_value=400, value=150, step=25)
min_pos_rate = st.sidebar.slider("Minimum Positive Match Rate (%)", min_value=30, max_value=75, value=40, step=5)
search_player = st.sidebar.text_input("🔍 Search Player Name", "")

# Apply sidebar filters
filt_bat = batters[(batters["hp_balls"] >= min_hp_balls) & (batters["pos_match_rate"] >= min_pos_rate)].copy()
filt_bowl = bowlers[(bowlers["hp_balls"] >= min_hp_balls) & (bowlers["pos_match_rate"] >= min_pos_rate)].copy()

if search_player.strip():
    filt_bat = filt_bat[filt_bat["player_name"].str.contains(search_player, case=False, na=False)]
    filt_bowl = filt_bowl[filt_bowl["player_name"].str.contains(search_player, case=False, na=False)]

filt_bat["filtered_rank"] = range(1, len(filt_bat) + 1)
filt_bowl["filtered_rank"] = range(1, len(filt_bowl) + 1)

# Tab layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 1. Unstop Top 10 Rankings",
    "📈 2. Quadrant Analysis (Pressure vs. Par)",
    "🔬 3. Pressure Index Model & Validation",
    "👤 4. Player Deep-Dive Inspector",
    "📋 5. Full Qualified Pool & Data Export"
])

# ----------------- TAB 1: TOP 10 RANKINGS -----------------
with tab1:
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("🏏 Top 10 Batters Under Pressure")
        top10_bat = filt_bat.head(10)
        
        if not top10_bat.empty:
            leader_bat = top10_bat.iloc[0]
            st.info(f"🥇 **Rank 1 Batter:** **{leader_bat['player_name']}** | Final Rating: **{leader_bat['final_rating']:.2f}** | HP Impact: **+{leader_bat['hp_impact']:.2f}** RVA/100 | Match Consistency: **{leader_bat['pos_match_rate']:.1f}%**")
            
            fig_bat = px.bar(
                top10_bat.iloc[::-1],
                x="final_rating",
                y="player_name",
                orientation="h",
                text="final_rating",
                color="hp_impact",
                color_continuous_scale="Reds",
                labels={"final_rating": "Final Rating", "player_name": "", "hp_impact": "HP Impact (RVA/100)"},
                title="Top 10 Clutch Batters (Final Rating)"
            )
            fig_bat.update_traces(texttemplate='%{text:.2f}', textposition="outside")
            fig_bat.update_layout(height=400, margin=dict(l=10, r=40, t=40, b=20))
            st.plotly_chart(fig_bat, use_container_width=True)
            
            st.dataframe(
                top10_bat[[
                    "filtered_rank", "player_name", "hp_balls", "hp_matches", "hp_impact", "ord_baseline", "clutch_delta", "pos_match_rate", "final_rating"
                ]].rename(columns={
                    "filtered_rank": "Rank",
                    "player_name": "Batter",
                    "hp_balls": "HP Balls",
                    "hp_matches": "HP Matches",
                    "hp_impact": "HP Impact",
                    "ord_baseline": "Ord Base",
                    "clutch_delta": "Delta",
                    "pos_match_rate": "Consistency %",
                    "final_rating": "Final Rating"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No batters match the selected filter criteria.")

    with col_t2:
        st.subheader("🎯 Top 10 Bowlers Under Pressure")
        top10_bowl = filt_bowl.head(10)
        
        if not top10_bowl.empty:
            leader_bowl = top10_bowl.iloc[0]
            st.info(f"🥇 **Rank 1 Bowler:** **{leader_bowl['player_name']}** | Final Rating: **{leader_bowl['final_rating']:.2f}** | HP Impact: **+{leader_bowl['hp_impact']:.2f}** RVS/24 | Match Consistency: **{leader_bowl['pos_match_rate']:.1f}%**")
            
            fig_bowl = px.bar(
                top10_bowl.iloc[::-1],
                x="final_rating",
                y="player_name",
                orientation="h",
                text="final_rating",
                color="hp_impact",
                color_continuous_scale="Blues",
                labels={"final_rating": "Final Rating", "player_name": "", "hp_impact": "HP Impact (RVS/24)"},
                title="Top 10 Clutch Bowlers (Final Rating)"
            )
            fig_bowl.update_traces(texttemplate='%{text:.2f}', textposition="outside")
            fig_bowl.update_layout(height=400, margin=dict(l=10, r=40, t=40, b=20))
            st.plotly_chart(fig_bowl, use_container_width=True)
            
            st.dataframe(
                top10_bowl[[
                    "filtered_rank", "player_name", "hp_balls", "hp_matches", "hp_impact", "ord_baseline", "clutch_delta", "pos_match_rate", "final_rating"
                ]].rename(columns={
                    "filtered_rank": "Rank",
                    "player_name": "Bowler",
                    "hp_balls": "HP Balls",
                    "hp_matches": "HP Matches",
                    "hp_impact": "HP Impact",
                    "ord_baseline": "Ord Base",
                    "clutch_delta": "Delta",
                    "pos_match_rate": "Consistency %",
                    "final_rating": "Final Rating"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No bowlers match the selected filter criteria.")

# ----------------- TAB 2: QUADRANTS -----------------
with tab2:
    st.subheader("Ordinary Baseline vs. High-Pressure Impact")
    st.markdown(
        "Players in the **top-right quadrant** perform above par in both ordinary and high-pressure situations. "
        "Players in the **top-left quadrant** are true pressure specialists who gear up significantly when the game is on the line."
    )
    
    disc_quad = st.radio("Select Discipline for Quadrant View", ["Batters (RVA per 100 Balls)", "Bowlers (RVS per 24 Balls)"], horizontal=True)
    df_q = filt_bat if "Batters" in disc_quad else filt_bowl
    df_q = df_q.copy()
    df_q["Status"] = ["Top 10 Elite" if i < 10 else "Qualified Pool" for i in range(len(df_q))]
    
    fig_scatter = px.scatter(
        df_q,
        x="ord_baseline",
        y="hp_impact",
        size="hp_balls",
        color="Status",
        hover_name="player_name",
        hover_data=["hp_balls", "hp_matches", "pos_match_rate", "final_rating", "clutch_delta"],
        color_discrete_map={"Top 10 Elite": "#E91E63", "Qualified Pool": "#004BA0"},
        labels={
            "ord_baseline": "Ordinary Situation Baseline (Impact vs Par)",
            "hp_impact": "High-Pressure Impact (Impact vs Par)",
            "hp_balls": "HP Balls"
        },
        title=f"{disc_quad}: Baseline vs. Pressure Performance"
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="#718096")
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="#718096")
    fig_scatter.update_layout(height=560)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------- TAB 3: VALIDATION -----------------
with tab3:
    st.subheader("Delivery-Level Pressure Index (PI) Formulation & Empirical Validation")
    
    st.markdown("""
    ### Mathematical Definition of Pressure
    The framework establishes a delivery-level **Pressure Index ($PI \in [0, 1]$)** evaluated at the exact micro-state before each ball is bowled:
    
    - **Innings 2 (Chasing):**
      $$PI = 0.45 \cdot P_{\\text{RRR}} + 0.30 \cdot P_{\\text{Wickets}} + 0.25 \cdot P_{\\text{Endgame}}$$
      Where $P_{\\text{RRR}} = \\frac{1}{1 + e^{-3.2(RRR/Par - 1.22)}}$, $P_{\\text{Wickets}} = (Wickets/10)^{1.3}$, and $P_{\\text{Endgame}} = \\max(0, \\frac{60 - BallsLeft}{60})$.
      *Garbage-time dampeners* suppress dead-rubber states ($RRR > 24$ with 8+ down or $RRR < 4.5$ with $\le 3$ down).
    
    - **Innings 1 (Defending/Setting):**
      Captures top-order collapses ($\le 30\%$ balls & $\ge 2$ wickets down $\implies PI=0.75$), death phase acceleration ($\ge 75\%$ balls completed), and pacing deficits relative to venue par.
    
    - **High-Pressure Threshold:** $PI \ge 0.65$ isolates the top **24.1%** most critical match situations.
    """)
    
    st.markdown("---")
    st.markdown("### Empirical Validation of Pressure Regime Separation")
    st.dataframe(validation, use_container_width=True, hide_index=True)
    
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        fig_v1 = px.bar(
            validation,
            x="situation",
            y="wicket_rate_pct",
            color="situation",
            color_discrete_map={"Ordinary (PI < 0.65)": "#607D8B", "High Pressure (PI >= 0.65)": "#E91E63"},
            text="wicket_rate_pct",
            title="Wicket Hazard Rate Surge (+44.8% under Pressure)",
            labels={"wicket_rate_pct": "Wicket Rate (%)", "situation": ""}
        )
        fig_v1.update_traces(texttemplate='%{text:.2f}%', textposition="outside")
        fig_v1.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_v1, use_container_width=True)
        
    with vcol2:
        fig_v2 = px.bar(
            validation,
            x="situation",
            y="implied_run_rate",
            color="situation",
            color_discrete_map={"Ordinary (PI < 0.65)": "#607D8B", "High Pressure (PI >= 0.65)": "#004BA0"},
            text="implied_run_rate",
            title="Implied Run Rate (RPO) Escalation",
            labels={"implied_run_rate": "Run Rate (Runs/Over)", "situation": ""}
        )
        fig_v2.update_traces(texttemplate='%{text:.2f} RPO', textposition="outside")
        fig_v2.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_v2, use_container_width=True)

# ----------------- TAB 4: PLAYER DEEP-DIVE -----------------
with tab4:
    st.subheader("Player Performance & Consistency Inspector")
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        sel_role = st.selectbox("Select Discipline", ["Batters", "Bowlers"])
        target_df = batters if sel_role == "Batters" else bowlers
        player_list = target_df["player_name"].tolist()
        selected_player = st.selectbox("Select Player", player_list)
        
    p_data = target_df[target_df["player_name"] == selected_player].iloc[0]
    
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Overall Rank", f"#{int(p_data['rank'])}")
    m2.metric("Final Clutch Rating", f"{p_data['final_rating']:.2f}")
    m3.metric("High-Pressure Impact", f"{p_data['hp_impact']:+.2f}")
    m4.metric("Ordinary Baseline", f"{p_data['ord_baseline']:+.2f}")
    m5.metric("Positive Match %", f"{p_data['pos_match_rate']:.1f}%")
    
    st.markdown("---")
    c_chart, c_details = st.columns([3, 2])
    with c_chart:
        comp_bar = go.Figure(data=[
            go.Bar(name="Ordinary Baseline", x=["Performance vs Par"], y=[p_data["ord_baseline"]], marker_color="#718096"),
            go.Bar(name="High-Pressure Impact", x=["Performance vs Par"], y=[p_data["hp_impact"]], marker_color="#E91E63")
        ])
        comp_bar.update_layout(
            barmode="group",
            title=f"{selected_player}: Ordinary vs High-Pressure Impact",
            height=360,
            yaxis_title="Value Added / Saved vs Par"
        )
        st.plotly_chart(comp_bar, use_container_width=True)
        
    with c_details:
        st.markdown(f"#### 📊 Sample & Competition Breakdown")
        st.markdown(f"""
        - **Total High-Pressure Deliveries:** **{int(p_data['hp_balls'])}**
          - 🏆 **IPL High-Pressure Balls:** {int(p_data['ipl_hp_balls'])}
          - 🌍 **T20I High-Pressure Balls:** {int(p_data['t20i_hp_balls'])}
        - **Ordinary Situation Deliveries:** **{int(p_data['ord_balls'])}**
        - **High-Pressure Matches:** **{int(p_data['hp_matches'])}** matches
        - **Pressure Gear-Up Delta:** **{p_data['clutch_delta']:+.2f}**
        - **Match Consistency:** Delivered positive net contribution in **{p_data['pos_match_rate']:.1f}%** of pressure matches.
        """)

# ----------------- TAB 5: EXPORT & FULL POOL -----------------
with tab5:
    st.subheader("Full Qualified Pool & Export Tables")
    st.markdown("Download full ranked datasets for external reporting or submission verification.")
    
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(f"**Qualified Batters Pool ({len(batters)} players)**")
        st.download_button(
            label="📥 Download Batters CSV (output_batters.csv)",
            data=batters.to_csv(index=False),
            file_name="output_batters.csv",
            mime="text/csv"
        )
        st.dataframe(batters, use_container_width=True, hide_index=True)
        
    with e2:
        st.markdown(f"**Qualified Bowlers Pool ({len(bowlers)} players)**")
        st.download_button(
            label="📥 Download Bowlers CSV (output_bowlers.csv)",
            data=bowlers.to_csv(index=False),
            file_name="output_bowlers.csv",
            mime="text/csv"
        )
        st.dataframe(bowlers, use_container_width=True, hide_index=True)
