#!/usr/bin/env python3
"""
Figura de Ubicacion — estilo IJRS/MDPI, 3 paneles:
  (A) America del Sur   (B) Colombia   (C) Finca (6 lotes caucho) satelital

Requisitos:
    pip install matplotlib cartopy contextily geopandas shapely pyproj rasterio
"""

import os
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from shapely.geometry import Polygon
from shapely.ops import unary_union
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("cartopy no instalado -- mapa simplificado")

try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print("contextily no instalado -- sin fondo satelital")

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    8,
    "axes.linewidth":     0.6,
    "figure.dpi":         150,
    "savefig.dpi":        300,
})

OUT_DIR = os.path.expanduser("~/analisis_gulupa/06_figuras_publicacion")
os.makedirs(OUT_DIR, exist_ok=True)

LOTES_COORDS = {
1: [(-73.3221749623743,3.80921164319661),(-73.3219419547879,3.80930006236061),(-73.3217183693707,3.80933236850439),(-73.3212291975254,3.80944367456596),(-73.3208098762671,3.80955511695782),(-73.3203207134674,3.80966174480087),(-73.3196917859725,3.80980084193472),(-73.3195567596827,3.80979122233639),(-73.3192931975467,3.80882711907622),(-73.3189876254866,3.80791438740676),(-73.3188860495402,3.80746513829542),(-73.3187473017745,3.80697371798973),(-73.3187381154504,3.80690821351704),(-73.3193949715423,3.80677384846949),(-73.3203499826821,3.80657457939159),(-73.3207879381105,3.80645849542328),(-73.3212024288525,3.80643591756702),(-73.3213375181866,3.80641279351812),(-73.3216893521755,3.80748465450873),(-73.3219992036877,3.80859385456974),(-73.3220502462091,3.80868750661993),(-73.3221749623743,3.80921164319661)],
2: [(-73.3194215609175,3.80987047692765),(-73.3185178061253,3.81005113326707),(-73.3174928316614,3.81030639246708),(-73.3168080327838,3.81044069991719),(-73.3166040928329,3.80994915173867),(-73.3163730221507,3.80904124394343),(-73.3161091001631,3.80826424465387),(-73.3160030601421,3.80771675749792),(-73.3159287259301,3.80762773732084),(-73.3160871441172,3.80758127199624),(-73.316492412047,3.80751190261359),(-73.3170980628531,3.80736808409411),(-73.3177362287111,3.80726642718706),(-73.3183512201159,3.80710859309445),(-73.3185934767939,3.80705293615372),(-73.3187737534551,3.80774089722816),(-73.31882477738,3.80784390447223),(-73.3188943100886,3.80800775684388),(-73.3190148395373,3.8082886492416),(-73.3190978249832,3.80871915157239),(-73.3192042845556,3.80905147014081),(-73.3192921445853,3.80936971948633),(-73.3193569753016,3.80955695073891),(-73.3194215609175,3.80987047692765)],
3: [(-73.321230658925,3.80628628914341),(-73.3206297284904,3.80639737717537),(-73.3198608852652,3.80664846456263),(-73.3192507230765,3.80671743491087),(-73.3187148728519,3.80688711878935),(-73.3186593193136,3.80671861621302),(-73.318543600953,3.80635821401981),(-73.3183400227525,3.80567956268645),(-73.3181597098889,3.80501031218332),(-73.3180023266019,3.80452353322293),(-73.3178817612368,3.80426135138614),(-73.3179285275523,3.80415853561183),(-73.3183198058413,3.80409849241842),(-73.31877640451,3.80397309029981),(-73.31917236669,3.80389902299273),(-73.3196195975954,3.8038016676585),(-73.3200900665643,3.80372774551192),(-73.3203556147334,3.80366745551004),(-73.3204766791981,3.80367236967485),(-73.3206571935275,3.80423871332529),(-73.3207173589516,3.80443061295119),(-73.3207588795484,3.80463183137429),(-73.3208282949378,3.80485649236979),(-73.3208933344816,3.8049361388595),(-73.3208884513287,3.805053069611),(-73.3209255689063,3.80512330631916),(-73.3209579845098,3.80521692189958),(-73.3210459723233,3.80546968488515),(-73.3210968606754,3.80564285597703),(-73.3211570714678,3.80581136767977),(-73.3211939716511,3.80599386667128),(-73.3212355738219,3.80615298679413),(-73.321230658925,3.80628628914341)],
4: [(-73.3185425209787,3.80691484701579),(-73.3177038725579,3.80714240730019),(-73.316869953406,3.80733255502219),(-73.3162130155022,3.80750901601522),(-73.3158543599098,3.8075550886796),(-73.3157714851689,3.80706845581363),(-73.3156556863782,3.806750151979),(-73.3155029705832,3.80625870461511),(-73.3153133014552,3.80561282408746),(-73.315174337642,3.80523366614402),(-73.3150587481229,3.80480777808346),(-73.3152638232627,3.80471462819817),(-73.3156225865715,3.80461242429118),(-73.3160325277837,3.80453370848109),(-73.3162141741786,3.80451535399491),(-73.3164517732605,3.80445968826997),(-73.3166101908078,3.80441322257502),(-73.3170340657322,3.80435324390705),(-73.3172949209319,3.80431165625509),(-73.3175790141175,3.80429350202227),(-73.3177186870146,3.80430780824347),(-73.3179919878986,3.80505208233415),(-73.3182046251124,3.80586172425359),(-73.3184824741598,3.80666213874093),(-73.3185473864119,3.80680727160429),(-73.3185425209787,3.80691484701579)],
5: [(-73.3150636502108,3.804681492554),(-73.3149570559106,3.80441933808291),(-73.3148876967731,3.8041666118261),(-73.3147209567671,3.80370320278119),(-73.3145729707151,3.80317434419124),(-73.3144527337881,3.80274376957688),(-73.3142951800612,3.80234586501404),(-73.3141424920797,3.80184038531618),(-73.3140824372903,3.80159235501167),(-73.3139993355399,3.80122266205386),(-73.3139439192311,3.80098399605626),(-73.3138744057857,3.80081078877138),(-73.3139257197213,3.80076411339541),(-73.3140095483788,3.80075960010121),(-73.3140980336821,3.8007550959246),(-73.3142470735216,3.80074135521293),(-73.3142983874412,3.80069467980385),(-73.3143589328887,3.80069012084688),(-73.3146243706639,3.80068596332745),(-73.3146757481417,3.80060654482868),(-73.3148247789088,3.80059748158121),(-73.3149597761892,3.80062113400479),(-73.3151368104019,3.800579382365),(-73.3152161095449,3.80050937369068),(-73.3153837759878,3.80049566923281),(-73.3155374634274,3.80048661496851),(-73.3156445662691,3.80048682466921),(-73.3157284221919,3.80046827844521),(-73.3157890311882,3.80043097630154),(-73.3159006817841,3.80048732607339),(-73.3159565252387,3.80050614579103),(-73.316007739274,3.80051092364265),(-73.3160263658598,3.80051096010334),(-73.3161055288513,3.80051111505681),(-73.3161661197,3.80048316804696),(-73.3162917675079,3.80052551237479),(-73.3164126226005,3.80063801133223),(-73.3165334232923,3.80077857579031),(-73.3165983351473,3.80092370847832),(-73.3166911234168,3.80110163893859),(-73.3167978635259,3.80128895191275),(-73.3168864215688,3.80124702678855),(-73.3170308322251,3.8012192437448),(-73.3171797634879,3.80126163356564),(-73.3173847015863,3.80123864648118),(-73.3175895218157,3.80127646796731),(-73.317831559058,3.80133307261984),(-73.3179758247167,3.80138013080877),(-73.3179801549936,3.80154853306198),(-73.3179706875509,3.80162803383313),(-73.3180217022534,3.80173571852369),(-73.3180633764473,3.80185741776478),(-73.3181191928869,3.80189027016469),(-73.3181888523837,3.80198863611331),(-73.3183052416802,3.80200289654373),(-73.3185285619508,3.80210624057059),(-73.3187102170581,3.80208320771647),(-73.3189243600545,3.80211636961766),(-73.319175982921,3.80203266453402),(-73.3193995751974,3.80199568066128),(-73.3196092880754,3.80191189351155),(-73.3197024666186,3.80188868753062),(-73.3198557108027,3.8021088345861),(-73.3199301268603,3.80215575607474),(-73.3199998409776,3.80222605642379),(-73.320074057805,3.80237588493973),(-73.3201526596043,3.80266605033344),(-73.3201989364547,3.80281582427537),(-73.320259156148,3.80297965829399),(-73.3202867882138,3.80313875103726),(-73.3203192128156,3.80322768898939),(-73.3203608420715,3.80337277625517),(-73.3203839623927,3.8034570184212),(-73.3203744316535,3.80356926245494),(-73.3195220703026,3.8036658263431),(-73.3191354668656,3.8037165241348),(-73.3188511470061,3.80385161871266),(-73.3183153668606,3.80398622119671),(-73.3176679899413,3.80403640796772),(-73.3173371667792,3.80413866764482),(-73.3169784035771,3.80424087244656),(-73.3150636502108,3.804681492554)],
6: [(-73.3220070218149,3.80936567616293),(-73.3214339883753,3.80949786764739),(-73.3197847612414,3.80988288211925),(-73.3184150270637,3.8102216644672),(-73.3173295788958,3.81044406163334),(-73.316747281946,3.81055050437549),(-73.3166312009077,3.81037720540896),(-73.3164508264875,3.80974069784874),(-73.3159694949975,3.80821485603622),(-73.3154604240919,3.80658605308328),(-73.3150256787571,3.80505094832266),(-73.3143918428192,3.80292607646275),(-73.3140630836648,3.80196652473105),(-73.3138317144883,3.80121297828196),(-73.3137068211739,3.8007823947026),(-73.3140236636776,3.8006847862342),(-73.3145362395667,3.80050804197467),(-73.3149235581415,3.80008781689876),(-73.315505692879,3.80006089122231),(-73.3157757963295,3.80005206479356),(-73.3160969142188,3.800150922938),(-73.3162411522691,3.80021201403966),(-73.3165574866979,3.80037634912554),(-73.3174605228383,3.80056054222979),(-73.3181163855348,3.80093603315935),(-73.3186842164812,3.80108214902504),(-73.3194293404445,3.80105320069869),(-73.3196433836915,3.8011378158265),(-73.3198012754114,3.801362649519),(-73.3198101178058,3.80160590248688),(-73.3198002610662,3.80188653979818),(-73.3199677016953,3.80198977433668),(-73.3200979252795,3.80207422571613),(-73.320180649426,3.80264037816144),(-73.3202967038611,3.80282770931098),(-73.3203846824592,3.80308514977718),(-73.3204723803034,3.80348759564544),(-73.3206114917907,3.80379191215777),(-73.3208105966944,3.80437700259816),(-73.3208472251244,3.80469982937005),(-73.3210046555209,3.80516322056068),(-73.3211713993618,3.80562662999154),(-73.3220134547257,3.80844887642097),(-73.3222405278032,3.80901998916014),(-73.3222213391559,3.80930996375386),(-73.3220070218149,3.80936567616293)],
}

polys = [Polygon(v) for v in LOTES_COORDS.values()]
FINCA_POLY = unary_union(polys)
FINCA_GDF = gpd.GeoDataFrame(geometry=[FINCA_POLY], crs="EPSG:4326")
LOTES_GDF = gpd.GeoDataFrame(
    {"lote_id": list(LOTES_COORDS.keys())},
    geometry=[Polygon(v) for v in LOTES_COORDS.values()], crs="EPSG:4326")
FINCA_LON, FINCA_LAT = FINCA_POLY.centroid.x, FINCA_POLY.centroid.y

LETTERS = list(string.ascii_uppercase)

fig = plt.figure(figsize=(12, 4.6))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.10, left=0.02, right=0.98, top=0.86, bottom=0.06)

# ── Panel A: America del Sur ──────────────────────────────
if HAS_CARTOPY:
    prj = ccrs.PlateCarree()
    ax_sa = fig.add_subplot(gs[0], projection=prj)
    ax_sa.set_extent([-85, -32, -58, 15], crs=prj)
    ax_sa.add_feature(cfeature.LAND, facecolor="#e8e0d0", edgecolor="0.5", linewidth=0.3)
    ax_sa.add_feature(cfeature.OCEAN, facecolor="#c9dff0")
    ax_sa.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.4")
    ax_sa.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.4")

    col_ext = [-82, -66, -5, 13]
    rect_col = mpatches.Rectangle((col_ext[0], col_ext[2]), col_ext[1]-col_ext[0], col_ext[3]-col_ext[2],
                                   linewidth=1.2, edgecolor="red", facecolor="none", transform=prj, zorder=5)
    ax_sa.add_patch(rect_col)
    ax_sa.plot(FINCA_LON, FINCA_LAT, "r*", ms=6, transform=prj, zorder=6)

    gl = ax_sa.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 6}; gl.ylabel_style = {"size": 6}
else:
    ax_sa = fig.add_subplot(gs[0])
    ax_sa.set_facecolor("#c9dff0")
    ax_sa.text(0.5, 0.5, "América del Sur\n(instalar cartopy)", transform=ax_sa.transAxes, ha="center")

ax_sa.set_title("(A) América del Sur", fontsize=10, fontweight="bold", pad=4)
ax_sa.text(0.02, 0.97, "(A)", transform=ax_sa.transAxes, fontsize=11, fontweight="bold",
           va="top", ha="left", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

# ── Panel B: Colombia ─────────────────────────────────────
if HAS_CARTOPY:
    ax_col = fig.add_subplot(gs[1], projection=prj)
    ax_col.set_extent([-82, -66, -5, 13], crs=prj)
    ax_col.add_feature(cfeature.LAND, facecolor="#e8e0d0", edgecolor="0.5", linewidth=0.3)
    ax_col.add_feature(cfeature.OCEAN, facecolor="#c9dff0")
    ax_col.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="0.4")
    ax_col.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.4")
    ax_col.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="#7ecbdb", alpha=0.6)

    buf = 1.2
    rect_region = mpatches.Rectangle((FINCA_LON - buf, FINCA_LAT - buf), buf*2, buf*2,
                                      linewidth=1.2, edgecolor="red", facecolor="#f4b942",
                                      alpha=0.35, transform=prj, zorder=4)
    ax_col.add_patch(rect_region)
    ax_col.plot(FINCA_LON, FINCA_LAT, "r*", ms=7, transform=prj, zorder=6)

    gl2 = ax_col.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
    gl2.top_labels = False; gl2.right_labels = False
    gl2.xlabel_style = {"size": 6}; gl2.ylabel_style = {"size": 6}
    ax_col.text(FINCA_LON + 0.3, FINCA_LAT - 0.8, "Meta", transform=prj, fontsize=7.5,
                color="#8B0000", fontweight="bold", zorder=7)
else:
    ax_col = fig.add_subplot(gs[1])
    ax_col.set_facecolor("#c9dff0")
    ax_col.text(0.5, 0.5, "Colombia\n(instalar cartopy)", transform=ax_col.transAxes, ha="center")

ax_col.set_title("(B) Colombia", fontsize=10, fontweight="bold", pad=4)
ax_col.text(0.02, 0.97, "(B)", transform=ax_col.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

# ── Panel C: Finca (6 lotes de caucho) satelital ──────────
ax_d = fig.add_subplot(gs[2])

lotes_web = LOTES_GDF.to_crs(epsg=3857)
finca_web = FINCA_GDF.to_crs(epsg=3857)
bounds_w = finca_web.total_bounds
pad_m = 150
ax_d.set_xlim(bounds_w[0] - pad_m, bounds_w[2] + pad_m)
ax_d.set_ylim(bounds_w[1] - pad_m, bounds_w[3] + pad_m)

if HAS_CTX:
    try:
        ctx.add_basemap(ax_d, crs=lotes_web.crs.to_string(),
                         source=ctx.providers.Esri.WorldImagery, zoom=17, attribution=False)
    except Exception:
        try:
            ctx.add_basemap(ax_d, crs=lotes_web.crs.to_string(),
                             source=ctx.providers.OpenStreetMap.Mapnik, zoom=16, attribution=False)
        except Exception:
            ax_d.set_facecolor("#4a7c59")
else:
    ax_d.set_facecolor("#4a7c59")

lotes_web.boundary.plot(ax=ax_d, color="red", linewidth=1.5, zorder=6)
lotes_web.plot(ax=ax_d, facecolor="none", edgecolor="red", linewidth=1.2, zorder=6)

for idx, row in lotes_web.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax_d.text(cx, cy, str(row["lote_id"]), color="white", fontsize=8, fontweight="bold",
              ha="center", va="center", zorder=8,
              path_effects=[pe.withStroke(linewidth=2, foreground="black")])

from pyproj import Transformer
tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
xlim = ax_d.get_xlim(); ylim = ax_d.get_ylim()
xticks_m = np.linspace(xlim[0], xlim[1], 4)
yticks_m = np.linspace(ylim[0], ylim[1], 4)
xlabels = [f"{tr.transform(x, ylim[0])[0]:.4f}°O" for x in xticks_m]
ylabels = [f"{tr.transform(xlim[0], y)[1]:.4f}°N" for y in yticks_m]
ax_d.set_xticks(xticks_m); ax_d.set_xticklabels(xlabels, fontsize=6, rotation=30, ha="right", color="black")
ax_d.set_yticks(yticks_m); ax_d.set_yticklabels(ylabels, fontsize=6, color="black")
ax_d.tick_params(axis="both", direction="out", length=3, width=0.5, color="black", labelcolor="black")
for spine in ax_d.spines.values():
    spine.set_edgecolor("black"); spine.set_linewidth(0.7)

ax_d.annotate("", xy=(0.93, 0.93), xytext=(0.93, 0.80), xycoords="axes fraction",
              arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8, mutation_scale=14), zorder=15)
ax_d.text(0.93, 0.96, "N", transform=ax_d.transAxes, ha="center", va="bottom",
          fontsize=9, fontweight="bold", color="white", zorder=15,
          path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

legend_elements = [mpatches.Patch(facecolor="none", edgecolor="red", linewidth=1.5, label="Lotes de caucho (1-6)")]
ax_d.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.20),
            ncol=1, fontsize=7, framealpha=0.95, edgecolor="0.5", facecolor="white", labelcolor="black")

ax_d.set_title("(C) Sitio de estudio (Meta, Colombia)", fontsize=10, fontweight="bold", pad=4)
ax_d.text(0.02, 0.97, "(C)", transform=ax_d.transAxes, fontsize=11, fontweight="bold",
          va="top", ha="left", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

fig.suptitle("Ubicación del área de estudio — plantación de caucho, departamento del Meta, Colombia",
             fontsize=12, fontweight="bold")

out = os.path.join(OUT_DIR, "FigLoc_location_map_caucho.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Guardada: {out}")
