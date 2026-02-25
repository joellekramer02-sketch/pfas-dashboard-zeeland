import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="PFAS Dashboard Zeeland", layout="wide")

DATA_PATH = "Data/ultieme_master_coords_slim_genormaliseerd.csv"


# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Numeriek maken
    for col in ["Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix: als coords geschaald zijn (bv 514425), schaal terug
    # Alleen op rijen waar lat/lon buiten WGS84 bereik vallen
    if "Latitude" in df.columns and "Longitude" in df.columns:
        mask_scaled = (df["Latitude"].abs() > 90) | (df["Longitude"].abs() > 180)
        df.loc[mask_scaled, "Latitude"] = df.loc[mask_scaled, "Latitude"] / 10000
        df.loc[mask_scaled, "Longitude"] = df.loc[mask_scaled, "Longitude"] / 10000

    # Jaar/Waarde types
    if "Jaar" in df.columns:
        df["Jaar"] = pd.to_numeric(df["Jaar"], errors="coerce").astype("Int64")
    if "Waarde" in df.columns:
        df["Waarde"] = pd.to_numeric(df["Waarde"], errors="coerce")

    # Strings strippen
    for col in ["Locatie", "PFAS", "Bron", "Medium", "Sampletype", "Eenheid"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================
# Plots
# =========================
def make_bar_by_location(df: pd.DataFrame, max_locations: int = 25):
    plot_df = df.copy()

    # Verwijder lege of ongeldige locaties
    plot_df = plot_df[
        plot_df["Locatie"].notna()
        & (plot_df["Locatie"].astype(str).str.strip() != "")
        & (plot_df["Locatie"].astype(str).str.lower() != "nan")
    ]

    # ---- Automatische eenheid-normalisatie (ug/L -> ng/L) ----
    if "Eenheid" in plot_df.columns:
        plot_df["Waarde_plot"] = plot_df["Waarde"]
        mask_ug = plot_df["Eenheid"].str.lower().isin(["ug/l", "µg/l"])
        if mask_ug.any():
            plot_df.loc[mask_ug, "Waarde_plot"] = plot_df.loc[mask_ug, "Waarde"] * 1000
    else:
        plot_df["Waarde_plot"] = plot_df["Waarde"]

    tmp = (
        plot_df.groupby("Locatie", dropna=False)["Waarde_plot"]
        .median()
        .sort_values(ascending=False)
        .head(max_locations)
        .index
    )

    plot_df = plot_df[plot_df["Locatie"].isin(tmp)]
    agg = plot_df.groupby("Locatie")["Waarde_plot"].median().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(agg.index.astype(str), agg.values)
    ax.set_title(f"Mediaan PFAS-waarde per locatie (top {len(agg)})")
    ax.set_ylabel("Concentratie (ng/L indien van toepassing)")
    ax.set_xlabel("Locatie")

    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    return fig


# =========================
# Map
# =========================
def make_map(map_df: pd.DataFrame, kaarttype: str) -> folium.Map:
    tiles = {
        "Normaal": ("OpenStreetMap", "© OpenStreetMap contributors"),
        "Satelliet": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        ),
    }
    tile_url, attr = tiles[kaarttype]

    center_lat, center_lon = 51.45, 3.80

    nl = map_df[
        map_df["Latitude"].between(50, 54) & map_df["Longitude"].between(3, 8)
    ]
    if not nl.empty:
        center_lat = float(nl["Latitude"].mean())
        center_lon = float(nl["Longitude"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles=None)
    folium.TileLayer(tiles=tile_url, attr=attr, name=kaarttype).add_to(m)

    pfas_priority = [
        "PFOS",
        "PFOA",
        "PFHxS",
        "PFNA",
        "PFHxA",
        "PFBS",
        "PFPeA",
        "PFPeS",
        "PFHpA",
        "PFHpS",
        "PFDA",
        "PFUnDA",
        "PFDoDA",
        "GenX",
        "HFPO-DA",
    ]
    priority_rank = {p: i for i, p in enumerate(pfas_priority)}
    default_rank = 9999

    map_df = map_df.copy()
    if "Locatie" in map_df.columns:
        map_df["Locatie"] = map_df["Locatie"].fillna("(onbekend)").astype(str)

    group_cols = ["Locatie", "Latitude", "Longitude"]
    grouped = map_df.dropna(subset=["Latitude", "Longitude"]).groupby(
        group_cols, dropna=False
    )

    for (loc, lat, lon), g in grouped:
        n = len(g)

        # LOQ statistiek per locatie
        if "LOQ_flag" in g.columns:
            n_loq = int(g["LOQ_flag"].fillna(False).sum())
        else:
            n_loq = 0
        perc_loq = round((n_loq / n) * 100, 1) if n > 0 else 0

        tmp = g.copy()
        tmp["__pfas_rank"] = tmp["PFAS"].map(priority_rank).fillna(default_rank).astype(int)
        tmp["__jaar_sort"] = pd.to_numeric(tmp.get("Jaar", None), errors="coerce").fillna(-1)
        tmp["__waarde_sort"] = pd.to_numeric(tmp.get("Waarde", None), errors="coerce").fillna(-1)

        tmp = tmp.sort_values(
            by=["__pfas_rank", "__jaar_sort", "__waarde_sort"],
            ascending=[True, False, False],
        )

        rows_html = []
        max_rows = 80
        show_tmp = tmp.head(max_rows)

        for _, row in show_tmp.iterrows():
            rows_html.append(
                "<tr>"
                f"<td>{row.get('PFAS','')}</td>"
                f"<td>{row.get('Jaar','')}</td>"
                f"<td>{row.get('Waarde','')}</td>"
                f"<td>{row.get('Eenheid','')}</td>"
                f"<td>{row.get('Bron','')}</td>"
                f"<td>{row.get('Medium','')}</td>"
                f"<td>{row.get('Sampletype','')}</td>"
                "</tr>"
            )

        more_note = ""
        if len(tmp) > max_rows:
            more_note = (
                f"<div style='margin-top:6px; font-size:12px;'><i>"
                f"Toont {max_rows} van {len(tmp)} metingen. Filter verder om minder te tonen."
                f"</i></div>"
            )

        popup_html = f"""
        <div style="width: 520px;">
          <div style="font-weight:700; font-size:14px; margin-bottom:6px;">
            {loc}
          </div>

          <div style="margin-bottom:6px; font-size:13px;">
            Aantal metingen op deze locatie: <b>{n}</b>
          </div>

          <div style="margin-bottom:8px; font-size:12px; color:#b45309;">
            <b>{n_loq}</b> van {n} metingen zijn &lt;LOQ ({perc_loq}%).
          </div>

          <div style="max-height: 240px; overflow-y: auto; border: 1px solid #ddd; padding: 6px;">
            <table style="width:100%; border-collapse: collapse; font-size:12px;">
              <thead>
                <tr>
                  <th style="text-align:left; border-bottom:1px solid #ddd;">PFAS</th>
                  <th style="text-align:left; border-bottom:1px solid #ddd;">Jaar</th>
                  <th style="text-align:left; border-bottom:1px solid #ddd;">Waarde</th>
                  <th style="text-align:left; border-bottom:1px solid #ddd;">Eenheid</th>
                  <th style="text-align:left; border-bottom:1px solid #ddd;">Bron</th>
                  <th style="text-align:left; border-bottom:1px solid #ddd;">Medium</th>
                  <th style="text-align:left; border-bottom:1px solid #ddd;">Sampletype</th>
                </tr>
              </thead>
              <tbody>
                {''.join(rows_html)}
              </tbody>
            </table>
          </div>
          {more_note}
        </div>
        """

        icon_html = f"""
        <div style="
            background: #2b6cb0;
            color: white;
            border-radius: 999px;
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            border: 2px solid white;
            box-shadow: 0 1px 4px rgba(0,0,0,0.35);
            ">
            {n}
        </div>
        """

        folium.Marker(
            location=[float(lat), float(lon)],
            icon=folium.DivIcon(html=icon_html),
            popup=folium.Popup(popup_html, max_width=600),
            tooltip=f"{loc} ({n} metingen)",
        ).add_to(m)

    return m


# =========================
# App
# =========================
st.title("🌍 PFAS Dashboard — Zeeland")
import os
st.write("Working dir:", os.getcwd())
st.write("Files in root:", os.listdir("."))
st.write("Files in Data:", os.listdir("Data") if os.path.exists("Data") else "Data folder not found")
st.write("DATA_PATH:", DATA_PATH, "Exists?", os.path.exists(DATA_PATH))
df = load_data(DATA_PATH)

# -------------------------
# Sidebar: Filters
# -------------------------
with st.sidebar:
    st.header("🔎 Filters")

    filters_aan = st.toggle("Filters gebruiken", value=True)

    working_df = df.copy()

    if not filters_aan:
        st.caption("Filters staan uit: je ziet alle data binnen de dataset.")
    else:
        bron_options = sorted(working_df["Bron"].dropna().unique())
        bron_filter = st.multiselect("Bron", options=bron_options)
        if bron_filter:
            working_df = working_df[working_df["Bron"].isin(bron_filter)]

        medium_options = sorted(working_df["Medium"].dropna().unique())
        medium_filter = st.multiselect("Medium", options=medium_options)
        if medium_filter:
            working_df = working_df[working_df["Medium"].isin(medium_filter)]

        pfas_all = working_df["PFAS"].dropna().unique()
        top_pfas = ["PFOS", "PFOA", "PFHxS", "PFNA", "PFHxA"]
        top_present = [p for p in top_pfas if p in pfas_all]
        rest = sorted([p for p in pfas_all if p not in top_present])
        pfas_options = top_present + rest

        pfas_filter = st.multiselect("PFAS", options=pfas_options)
        if pfas_filter:
            working_df = working_df[working_df["PFAS"].isin(pfas_filter)]

        if "Jaar" in working_df.columns:
            jaar_options = sorted(
                [int(x) for x in working_df["Jaar"].dropna().unique()]
            )
            jaar_filter = st.multiselect("Jaar", options=jaar_options)
            if jaar_filter:
                working_df = working_df[working_df["Jaar"].isin(jaar_filter)]

        sampletype_options = sorted(working_df["Sampletype"].dropna().unique())
        sampletype_filter = st.multiselect("Sampletype", options=sampletype_options)
        if sampletype_filter:
            working_df = working_df[working_df["Sampletype"].isin(sampletype_filter)]

        locatie_options = sorted(working_df["Locatie"].dropna().unique())
        locatie_filter = st.multiselect("Locatie (optioneel)", options=locatie_options)
        if locatie_filter:
            working_df = working_df[working_df["Locatie"].isin(locatie_filter)]

    kaarttype = st.selectbox("Kaart type", ["Normaal", "Satelliet"])

# -------------------------
# Filter logic
# -------------------------
subset = working_df.copy()
if "Waarde" in subset.columns:
    subset = subset.dropna(subset=["Waarde"])

# -------------------------
# Sidebar: Info
# -------------------------
with st.sidebar:
    st.divider()
    st.header("ℹ️ Informatie")

    n_rows = len(subset)
    n_loc = subset["Locatie"].nunique(dropna=True) if "Locatie" in subset.columns else 0
    n_map = (
        subset.dropna(subset=["Latitude", "Longitude"]).shape[0]
        if ("Latitude" in subset.columns and "Longitude" in subset.columns)
        else 0
    )

    st.write(f"**Rijen (na filters):** {n_rows:,}".replace(",", "."))
    st.write(f"**Unieke locaties:** {n_loc:,}".replace(",", "."))
    st.write(f"**Rijen met coördinaten:** {n_map:,}".replace(",", "."))

    if "Eenheid" in subset.columns and not subset.empty:
        units = sorted([u for u in subset["Eenheid"].dropna().unique() if str(u).strip() != ""])
        if len(units) > 1:
            st.warning("Meerdere eenheden in selectie: " + ", ".join(units[:6]) + (" ..." if len(units) > 6 else ""))
        elif len(units) == 1:
            st.caption(f"Eenheid: {units[0]}")

    if "LOQ_flag" in subset.columns and not subset.empty:
        n_loq = int(subset["LOQ_flag"].fillna(False).sum())
        if n_loq > 0:
            with st.expander("ℹ️ Informatie over <LOQ waarden"):
                st.markdown(f"""
**{n_loq} metingen** in deze selectie liggen onder de rapportage-/detectiegrens (<LOQ).

- In de **kaart en grafieken** worden deze weergegeven als numerieke waarde.
- In de **tabel** kun je via de kolom `LOQ_flag` zien of een meting <LOQ is.
- Interpretatie van trends met <LOQ waarden vereist voorzichtigheid.
""")

    st.caption("Kaart toont alleen rijen met Latitude/Longitude. Alle metingen blijven wel in de tabel/grafieken.")

# -------------------------
# Main: Tabs
# -------------------------
tab_kaart, tab_tabel, tab_lijn, tab_staaf, tab_info = st.tabs(
    ["🗺️ Kaart", "📄 Tabel", "📈 Lijngrafiek", "📊 Staafgrafiek", "ℹ️ Over PFAS"]
)

with tab_kaart:
    st.subheader("🗺️ Kaart")
    if subset.empty:
        st.info("Geen data om te plotten.")
    elif ("Latitude" not in subset.columns) or ("Longitude" not in subset.columns):
        st.error("Latitude/Longitude ontbreken.")
    else:
        map_df = subset.dropna(subset=["Latitude", "Longitude"]).copy()
        st.caption(
            f"Op kaart: {len(map_df):,} rijen met coords (van {len(subset):,} gefilterde rijen)".replace(",", ".")
        )
        if map_df.empty:
            st.info("Binnen deze filters zijn er geen rijen met coördinaten.")
        else:
            m = make_map(map_df, kaarttype=kaarttype)
            st_folium(m, width=1100, height=600)

with tab_tabel:
    st.subheader("📄 Gefilterde data")
    st.caption(f"Rijen: {len(subset):,}".replace(",", "."))
    st.dataframe(subset, use_container_width=True)

    st.download_button(
        label="📥 Download gefilterde dataset (CSV)",
        data=download_csv(subset),
        file_name="pfas_subset.csv",
        mime="text/csv",
    )

with tab_lijn:
    st.subheader("📈 Tijdreeksanalyse (wetenschappelijk verantwoord)")

    if subset.empty:
        st.info("Geen data beschikbaar binnen huidige filters.")
    else:
        valid_groups = (
            subset.groupby(["PFAS", "Locatie", "Medium", "Bron"])["Jaar"]
            .nunique()
            .reset_index()
        )
        valid_groups = valid_groups[valid_groups["Jaar"] >= 2]

        if valid_groups.empty:
            st.warning(
                "Binnen de huidige selectie zijn geen combinaties met minimaal 2 verschillende jaren. "
                "Een tijdreeksanalyse is daarom niet mogelijk."
            )
        else:
            st.markdown("Selecteer een geldige combinatie voor tijdreeksanalyse:")

            top_pfas = ["PFOS", "PFOA", "PFHxS", "PFNA", "PFHxA"]
            pfas_all = valid_groups["PFAS"].unique()
            top_present = [p for p in top_pfas if p in pfas_all]
            rest = sorted([p for p in pfas_all if p not in top_present])
            pfas_options = top_present + rest

            default_index = pfas_options.index("PFOS") if "PFOS" in pfas_options else 0
            gekozen_pfas = st.selectbox("PFAS", pfas_options, index=default_index)

            vg_pfas = valid_groups[valid_groups["PFAS"] == gekozen_pfas]
            locatie_options = sorted(vg_pfas["Locatie"].unique())
            gekozen_locatie = st.selectbox("Locatie", locatie_options)

            vg_loc = vg_pfas[vg_pfas["Locatie"] == gekozen_locatie]
            medium_options = sorted(vg_loc["Medium"].unique())
            gekozen_medium = st.selectbox("Medium", medium_options)

            vg_med = vg_loc[vg_loc["Medium"] == gekozen_medium]
            bron_options = sorted(vg_med["Bron"].unique())
            gekozen_bron = st.selectbox("Bron", bron_options)

            lijn_df = subset[
                (subset["PFAS"] == gekozen_pfas)
                & (subset["Locatie"] == gekozen_locatie)
                & (subset["Medium"] == gekozen_medium)
                & (subset["Bron"] == gekozen_bron)
            ].copy()

            # Eenheid-normalisatie (ug/L -> ng/L)
            lijn_df["Waarde_plot"] = lijn_df["Waarde"]
            if "Eenheid" in lijn_df.columns:
                mask_ug = lijn_df["Eenheid"].str.lower().isin(["ug/l", "µg/l"])
                lijn_df.loc[mask_ug, "Waarde_plot"] = lijn_df.loc[mask_ug, "Waarde"] * 1000

            agg = lijn_df.groupby("Jaar")["Waarde_plot"].median().sort_index()

            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            ax.plot(agg.index, agg.values, marker="o")

            ax.set_title(
                f"Tijdreeks {gekozen_pfas} – {gekozen_locatie}\n{gekozen_medium} | {gekozen_bron}"
            )
            ax.set_xlabel("Jaar")
            ax.set_ylabel("Concentratie (ng/L indien van toepassing)")

            ax.set_xticks(list(agg.index.astype(int)))
            ax.set_xticklabels([str(int(x)) for x in agg.index])

            ax.grid(True)
            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            st.caption("Jaren in deze tijdreeks: " + ", ".join([str(int(x)) for x in agg.index]))

with tab_staaf:
    st.subheader("📊 Staafgrafiek per PFAS")

    geselecteerde_pfas = subset["PFAS"].unique()

    if len(geselecteerde_pfas) == 0:
        st.info("Selecteer minimaal één PFAS om de grafiek te tonen.")
    elif len(geselecteerde_pfas) > 1:
        st.warning("Selecteer één PFAS voor een vergelijking per locatie.")
    else:
        gekozen_pfas = geselecteerde_pfas[0]
        st.markdown(f"**PFAS:** {gekozen_pfas}")

        unieke_eenheden = subset["Eenheid"].dropna().unique()
        if len(unieke_eenheden) > 1:
            st.warning(
                "Let op: meerdere eenheden aanwezig in deze selectie. "
                "Controleer of vergelijking inhoudelijk correct is."
            )
            st.caption("Gevonden eenheden: " + ", ".join(unieke_eenheden))

        fig = make_bar_by_location(subset, max_locations=25)
        st.pyplot(fig, use_container_width=True)

with tab_info:
    st.subheader("ℹ️ Over PFAS")
    st.markdown("""
**PFAS** staat voor *Per- en PolyFluorAlkylStoffen*: een grote groep door mensen gemaakte chemische stoffen.
Ze worden al decennia gebruikt omdat ze water-, vet- en vuilafstotend zijn.

### Waarom zijn PFAS een aandachtspunt?
- PFAS kunnen **lang in het milieu** aanwezig blijven (sommige breken nauwelijks af).
- Sommige PFAS kunnen zich **ophopen** in organismen.
- Er is aandacht voor mogelijke effecten op **mens en ecosysteem**.

### Wat laat dit dashboard zien?
Dit dashboard toont metingen van PFAS in Zeeland uit verschillende bronnen (o.a. RWS/WUR/RWZI/VWS).
Je kunt filteren op PFAS, bron, medium en locatie.

### Let op bij interpretatie
- **Eenheden kunnen verschillen** (bijv. ng/L of µg/L). Vergelijk alleen waarden die inhoudelijk vergelijkbaar zijn.
- **<LOQ**: sommige waarden zijn onder de rapportage-/detectiegrens. In de tabel kun je dit herkennen via `LOQ_flag`.
- Meetmethodes en monsters kunnen verschillen per bron en jaar.

""")
