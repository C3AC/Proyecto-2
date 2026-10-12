"""App local para clasificar mosquitos con los modelos entrenados. Desde la raíz del repo:
    conda activate mosquito-ml
    streamlit run app/streamlit_app.py
"""
import io
import json
import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError
from sklearn.metrics import average_precision_score, precision_recall_curve

import inference as inf
import summary
from common import CLASSES

ROOT = inf.ROOT
NAMES = {"aegypti": "Aedes aegypti", "albopictus": "Aedes albopictus", "anopheles": "Anopheles",
         "culex": "Culex", "culiseta": "Culiseta", "japonicus/koreicus": "Aedes japonicus/koreicus"}
DISEASES = {"aegypti": "dengue, zika, chikungunya, fiebre amarilla", "albopictus": "dengue, zika, chikungunya",
            "anopheles": "malaria, filariasis linfática", "culex": "virus del Nilo Occidental, encefalitis japonesa",
            "culiseta": "arbovirus, filariasis linfática", "japonicus/koreicus": "virus del Nilo Occidental, La Crosse"}
# Paleta categórica validada para daltonismo (orden fijo; el color sigue a la clase, nunca al rango)
CLASS_COLORS = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # secuencial: un solo tono, claro -> oscuro
ACCENT, MUTED, HIGHLIGHT = "#2a78d6", "#c3c2b7", "#eb6834"
ENSEMBLE = "Promedio de todos (ensamble)"

pio.templates["mosquito"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, sans-serif", color="#0b0b0b", size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", barcornerradius=4,
    xaxis=dict(gridcolor="#e8e7e3", zeroline=False), yaxis=dict(gridcolor="#e8e7e3", zeroline=False),
    margin=dict(l=10, r=10, t=50, b=10), hoverlabel=dict(bgcolor="white")))
pio.templates.default = "plotly_white+mosquito"

st.set_page_config(page_title="Identificador de mosquitos", page_icon="🦟", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos…")
def pipeline():
    return inf.Pipeline()


@st.cache_data
def dataset():
    df = pd.read_csv(ROOT / "data" / "processed" / "splits.csv")
    df["stem"] = df["img_fName"].str.replace(r"\.[^.]+$", "", regex=True)
    df["bbx_w"], df["bbx_h"] = df.bbx_xbr - df.bbx_xtl, df.bbx_ybr - df.bbx_ytl
    df["area_ratio"] = df.bbx_w * df.bbx_h / (df.img_w * df.img_h)
    df["aspect_ratio"] = df.img_w / df.img_h
    df["clase"] = df["class_label"].map(NAMES)
    return df


@st.cache_data
def test_preds(names):
    return {n: pd.read_csv(ROOT / "results" / f"test_preds_{n}.csv") for n in names}


@st.cache_data(ttl=60, show_spinner=False)
def summary_zip():
    return summary.zip_bytes()


@st.cache_data(show_spinner="Localizando y clasificando el mosquito…", max_entries=16)
def analyze(data, names):
    """Bytes de la imagen -> (vista con la caja dibujada, confianza del detector, recorte, predicciones)."""
    im, box, conf, crop = pipe.preprocess(io.BytesIO(data))
    scale = min(1.0, 1100 / max(im.size))  # fotos de hasta 12000 px: se muestran reducidas
    view = im.resize((round(im.width * scale), round(im.height * scale)))
    ImageDraw.Draw(view).rectangle([v * scale for v in box], outline=HIGHLIGHT, width=max(3, view.width // 150))
    return view, conf, crop, pipe.predict(crop, list(names))


def label(name):
    return inf.LABELS.get(name, name)


def short(c):
    return NAMES[c].replace("Aedes ", "A. ")


pipe = pipeline()
df = dataset()
metrics = pipe.metrics
ALL = list(pipe.models)
PREDS = test_preds(tuple(ALL))
ens = summary.ensemble_and_agreement(list(metrics.values()))
train_counts = df[df.split == "train"].class_label.value_counts()

# ---------------- Barra lateral ----------------
with st.sidebar:
    st.title("🦟 Identificador de mosquitos")
    st.caption("Proyecto 2 · CC3084 Data Science · UVG 2026 · MosquitoAlert Challenge 2023")
    mode = st.radio("Modelos a usar", ["Todos", "Elegir"], horizontal=True)
    selected = ALL if mode == "Todos" else st.multiselect("Modelos", ALL, default=ALL[:1], format_func=label)
    show_perf = st.toggle("Mostrar rendimiento de los modelos", value=True,
                          help="Oculta o muestra las métricas y gráficas de desempeño en toda la app.")
    st.divider()
    st.caption(f"Detector YOLOv8n: {'activo' if pipe.detector else 'no disponible, se usa la imagen completa'}")
    st.caption(f"{len(ALL)} clasificadores cargados · inferencia en CPU")
    st.download_button("Descargar resumen de resultados (.zip)", summary_zip(), "resumen_resultados.zip",
                       "application/zip", use_container_width=True,
                       help="Resumen en Markdown de datos, entrenamiento y prueba + métricas y figuras, para el informe.")

tab_story, tab_pred, tab_data, tab_perf = st.tabs(
    ["Cómo funciona", "Clasificar una imagen", "Explorar los datos", "Rendimiento de los modelos"])

# ---------------- 0. Cómo funciona ----------------
with tab_story:
    st.subheader("Identificar mosquitos vectores en fotos de ciencia ciudadana")
    st.write("Miles de personas envían fotos de mosquitos a Mosquito Alert. Revisarlas a mano no escala, y saber "
             "qué especie es importa: cada una transmite enfermedades distintas (dengue, zika, malaria, Nilo Occidental). "
             "Esta app automatiza la identificación de seis clases con modelos de visión por computadora.")
    steps = [("1 · Foto", "Tomada con un celular; el mosquito ocupa en mediana "
              f"{df.area_ratio.median():.0%} de la imagen."),
             ("2 · Detector", "YOLOv8n de una sola clase encuentra al mosquito (la caja naranja)."),
             ("3 · Recorte", "Cuadrado de 256 px con 15 % de margen: el mismo que se usó al entrenar."),
             ("4 · Clasificadores", f"{len(ALL)} modelos: DINOv2 + clasificadores clásicos y CNN con ajuste fino."),
             ("5 · Consenso", "Cada modelo vota; el desacuerdo indica una foto dudosa.")]
    for col, (title, text) in zip(st.columns(5), steps):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)

    if show_perf:
        best = next(iter(metrics))
        k = st.columns(4)
        k[0].metric("Imágenes limpias", f"{len(df):,}")
        k[1].metric("Mejor modelo (F1 macro)", f"{metrics[best]['macro_f1']:.3f}", label(best), delta_color="off", delta_arrow="off")
        if ens:
            k[2].metric("Promedio de todos los modelos", f"{ens[0]['macro_f1']:.3f}", "F1 macro", delta_color="off", delta_arrow="off")
            k[3].metric("Fotos de prueba con acuerdo total", f"{ens[2]:.0%}", f"aciertan el {ens[3]:.0%}", delta_color="off", delta_arrow="off")

        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            for n, m in metrics.items():
                is_best = n == best
                fig.add_scatter(
                    x=[train_counts[c] for c in CLASSES], y=[m["f1_per_class"][c] for c in CLASSES],
                    mode="markers+text" if is_best else "markers", name=label(n),
                    text=[short(c) for c in CLASSES] if is_best else None,
                    # albopictus y Culex caen casi en el mismo punto: etiquetas a lados opuestos
                    textposition=["middle right", "bottom left", "middle right", "top left", "middle right", "middle right"],
                    marker=dict(size=13 if is_best else 8, color=ACCENT if is_best else MUTED,
                                line=dict(width=2, color="white")),
                    customdata=[NAMES[c] for c in CLASSES],
                    hovertemplate=f"{label(n)}<br>%{{customdata}}<br>%{{x}} fotos de entrenamiento<br>F1 %{{y:.2f}}<extra></extra>")
            fig.update_layout(title="Hallazgo 1 · Las clases con menos fotos son las que fallan",
                              xaxis=dict(type="log", range=[1.2, 3.85], title="fotos de entrenamiento de la clase (escala log)"),
                              yaxis=dict(title="F1 en test", range=[-0.05, 1.12], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1]),
                              showlegend=False, height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"En azul el mejor modelo, en gris los demás. Con {train_counts['aegypti']} fotos de entrenamiento, "
                       "*A. aegypti* queda por debajo de lo que cualquier modelo puede aprender, y además se parece a "
                       "*A. albopictus*, que tiene ~100 veces más ejemplos.")
        with c2:
            if ens:
                fig = go.Figure(go.Bar(x=["Todos coinciden", "Discrepan"], y=[ens[3], ens[4]],
                                       marker_color=[ACCENT, HIGHLIGHT], text=[f"{ens[3]:.0%}", f"{ens[4]:.0%}"],
                                       textposition="outside", width=0.5,
                                       hovertemplate="%{x}: aciertan el %{y:.1%}<extra></extra>"))
                fig.update_layout(title="Hallazgo 2 · El consenso es una señal de confianza",
                                  yaxis=dict(range=[0, 1.12], tickformat=".0%", title="exactitud en test"), height=420)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Los modelos coinciden en el {ens[2]:.0%} de las fotos de prueba y entonces casi nunca se "
                           "equivocan. Cuando discrepan, conviene que un entomólogo revise la foto.")
        st.info("Hallazgo 3 · Con este desbalance la exactitud engaña: todos los modelos superan el 87 % de exactitud "
                "solo por acertar *albopictus* y *Culex*. Por eso se comparan con **F1 macro**, la métrica oficial del reto. "
                "Detalle en la pestaña «Rendimiento de los modelos».")

# ---------------- 1. Clasificar ----------------
with tab_pred:
    st.subheader("¿Qué mosquito es?")
    st.write("Sube una fotografía como las que envían los ciudadanos. La app localiza el mosquito, lo recorta "
             "y lo clasifica con los modelos seleccionados en la barra lateral.")
    c1, c2 = st.columns([3, 2])
    upload = c1.file_uploader("Fotografía", type=["jpg", "jpeg", "png", "webp", "bmp"])
    with c2:
        st.write("¿No tienes una foto? Usa una del conjunto de prueba (ningún modelo la vio al entrenar).")
        cls = st.selectbox("Clase de la imagen de prueba", CLASSES, format_func=NAMES.get)
        if st.button("Imagen de prueba aleatoria", use_container_width=True):
            pool = df[(df.split == "test") & (df.class_label == cls)]
            st.session_state.sample = pool.iloc[random.randrange(len(pool))].to_dict()
    sample = None if upload else st.session_state.get("sample")

    if upload is None and sample is None:
        st.info("Sube una imagen o elige una de prueba para empezar.")
    elif not selected:
        st.warning("Elige al menos un modelo en la barra lateral.")
    else:
        data = upload.getvalue() if upload else (ROOT / "data" / "raw" / "images" / sample["img_fName"]).read_bytes()
        try:
            view, conf, crop, results = analyze(data, tuple(selected))
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
            st.error("No se pudo leer el archivo como imagen. Prueba con otra foto en JPG o PNG.")
            st.stop()
        votes = pd.Series([CLASSES[int(p.argmax())] for p, _ in results.values()]).value_counts()
        winner = votes.index[0]
        avg = np.mean([p for p, _ in results.values()], 0)

        left, right = st.columns([2, 3])
        with left:
            st.image(view, caption="Mosquito localizado" + (f" (confianza del detector {conf:.0%})" if conf
                                                             else " (sin detector: se usa la foto completa)"))
            with st.expander("Ver la entrada de los modelos"):
                st.image(crop, width=200, caption="Recorte cuadrado de 256 px con 15 % de margen")
        with right:
            if len(results) == 1:
                name, (proba, _) = next(iter(results.items()))
                st.metric(label(name), NAMES[winner], f"{proba.max():.0%} de probabilidad", delta_color="off", delta_arrow="off")
            else:
                st.metric("Consenso de los modelos", NAMES[winner],
                          f"{votes.iloc[0]} de {len(results)} modelos coinciden", delta_color="off", delta_arrow="off")
                if votes.iloc[0] < len(results):
                    st.warning("Los modelos no coinciden: la foto es dudosa. En test, cuando discrepan aciertan "
                               f"solo el {ens[4]:.0%} de las veces." if ens else "Los modelos no coinciden.")
            if sample:
                ok = sample["class_label"] == winner
                st.markdown(f"Etiqueta real (entomólogos): **{NAMES[sample['class_label']]}** {'✅' if ok else '❌'}")
            st.caption(f"Enfermedades asociadas a *{NAMES[winner]}*: {DISEASES[winner]}.")
            if show_perf:
                best = next(n for n in metrics if n in results)
                st.caption(f"Qué tan fiable es: en el conjunto de prueba, {label(best)} reconoce el "
                           f"{metrics[best]['recall_per_class'][winner]:.0%} de los *{NAMES[winner]}* reales.")

            if len(results) == 1:
                s = pd.Series(proba, index=CLASSES).sort_values()
                fig = go.Figure(go.Bar(x=s.values, y=[NAMES[c] for c in s.index], orientation="h",
                                       marker_color=[CLASS_COLORS[c] for c in s.index],
                                       text=[f"{v:.1%}" for v in s.values], textposition="outside",
                                       hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
                fig.update_layout(title="Probabilidad por clase", xaxis=dict(range=[0, 1.15], tickformat=".0%"), height=330)
            else:
                probs = pd.DataFrame({**{label(n): p for n, (p, _) in results.items()}, "Promedio": avg},
                                     index=[short(c) for c in CLASSES]).T
                fig = px.imshow(probs, text_auto=".0%", color_continuous_scale=BLUES, zmin=0, zmax=1, aspect="auto")
                fig.update_layout(title="Probabilidad que asigna cada modelo a cada clase", coloraxis_showscale=False,
                                  height=130 + 42 * len(probs), xaxis_title="", yaxis_title="")
                fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:.1%}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

        table = pd.DataFrame([{
            "Modelo": label(n), "Predicción": NAMES[CLASSES[int(p.argmax())]], "Confianza": float(p.max()),
            "Tiempo (ms)": round(ms), **({"F1 macro en test": metrics[n]["macro_f1"]} if show_perf else {}),
        } for n, (p, ms) in results.items()])
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={
            "Confianza": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "F1 macro en test": st.column_config.NumberColumn(format="%.3f")})

# ---------------- 2. Explorar datos ----------------
with tab_data:
    st.subheader("El conjunto de datos después de la limpieza")
    counts = df.class_label.value_counts()
    k = st.columns(4)
    k[0].metric("Imágenes", f"{len(df):,}")
    k[1].metric("Clases", len(CLASSES))
    k[2].metric("Desbalance mayor/menor", f"{counts.max() / counts.min():.0f} : 1")
    k[3].metric("Área mediana del mosquito", f"{df.area_ratio.median():.1%} de la foto")
    st.write("Las fotos vienen de ciencia ciudadana y cada una tiene una caja anotada por entomólogos. "
             "Se descartaron las cajas geométricamente inválidas y se partió 70/15/15 estratificado por clase.")

    c1, c2 = st.columns(2)
    with c1:
        log = st.toggle("Escala logarítmica", value=False)
        cd = counts.rename_axis("class_label").reset_index(name="imágenes")
        cd["clase"] = cd.class_label.map(NAMES)
        fig = px.bar(cd, x="imágenes", y="clase", color="class_label", color_discrete_map=CLASS_COLORS,
                     orientation="h", log_x=log, text="imágenes")
        fig.update_layout(title=f"Dos clases concentran el {counts.iloc[:2].sum() / len(df):.0%} de las fotos",
                          showlegend=False, yaxis=dict(categoryorder="total ascending", title=""), height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"*A. aegypti* ({counts['aegypti']}) y *Anopheles* ({counts['anopheles']}) tienen tan pocas fotos "
                   "que la exactitud global engaña: por eso los modelos se comparan con F1 macro, que pesa igual a cada clase.")
    with c2:
        var = st.selectbox("Variable por clase", ["area_ratio", "aspect_ratio", "img_w", "img_h"], format_func={
            "area_ratio": "Proporción de la foto que ocupa el mosquito", "aspect_ratio": "Relación de aspecto (ancho/alto)",
            "img_w": "Ancho de la imagen (px)", "img_h": "Alto de la imagen (px)"}.get)
        fig = px.box(df, x="clase", y=var, color="class_label", color_discrete_map=CLASS_COLORS,
                     category_orders={"clase": [NAMES[c] for c in CLASSES]}, points=False)
        fig.update_layout(title="Distribución por clase", showlegend=False, xaxis_title="", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"El mosquito suele ser un objeto pequeño (mediana {df.area_ratio.median():.0%} de la foto): por eso "
                   "primero se detecta y recorta. *A. aegypti* aparece mucho más grande: varias fotos se tomaron con lupa.")

    st.markdown("##### Así ve el modelo cada clase")
    gcls = st.radio("Clase", CLASSES, format_func=NAMES.get, horizontal=True, label_visibility="collapsed")
    stems = df[df.class_label == gcls].stem.sample(6, random_state=1)
    for col, s in zip(st.columns(6), stems):
        col.image(str(ROOT / "data" / "processed" / "crops" / f"{s}.jpg"), use_container_width=True)
    st.caption("Recortes de entrada (caja anotada + 15 % de margen, 256 px). Fondo, luz y pose cambian mucho entre fotos.")

    with st.expander("Tabla de la partición entrenamiento / validación / prueba"):
        st.dataframe(pd.crosstab(df.clase, df.split, margins=True, margins_name="Total")[["train", "val", "test", "Total"]],
                     use_container_width=True)

# ---------------- 3. Rendimiento ----------------
with tab_perf:
    if not show_perf:
        st.info("El rendimiento está oculto. Actívalo en la barra lateral con «Mostrar rendimiento de los modelos».")
    else:
        st.subheader("¿Qué tan bien funciona cada modelo en el conjunto de prueba?")
        n_test = int(np.sum(next(iter(metrics.values()))["confusion_matrix"]))
        st.write(f"Evaluación sobre {n_test:,} imágenes de prueba que ningún modelo vio al entrenar.")
        rows = pd.DataFrame([{"modelo": label(n), "F1 macro": m["macro_f1"], "Exactitud balanceada": m["balanced_accuracy"],
                              "Exactitud": m["accuracy"], "ms por imagen": m.get("infer_ms_per_img"),
                              "familia": m.get("family", "")} for n, m in metrics.items()])
        if ens:
            rows.loc[len(rows)] = {"modelo": ENSEMBLE, "F1 macro": ens[0]["macro_f1"], "Exactitud": ens[0]["accuracy"],
                                   "Exactitud balanceada": ens[0]["balanced_accuracy"], "familia": "combinación"}
        met = st.radio("Métrica", ["F1 macro", "Exactitud balanceada", "Exactitud"], horizontal=True)
        r = rows.sort_values(met)
        top = r[r.modelo != ENSEMBLE][met].max()
        colors = [HIGHLIGHT if m == ENSEMBLE else ACCENT if v == top else MUTED for m, v in zip(r.modelo, r[met])]
        fig = go.Figure(go.Bar(x=r[met], y=r.modelo, orientation="h", text=[f"{v:.3f}" for v in r[met]],
                               textposition="outside", marker_color=colors,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title=f"{met} por modelo (azul: mejor modelo individual · naranja: ensamble)",
                          xaxis=dict(range=[0, 1.08]), height=100 + 45 * len(r))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Con la exactitud todos parecen excelentes porque aciertan las dos clases grandes; el F1 macro revela "
                   "la diferencia en las clases raras. El ensamble promedia las probabilidades de todos los modelos.")

        c1, c2 = st.columns(2)
        with c1:
            f1 = pd.DataFrame({label(n): m["f1_per_class"] for n, m in metrics.items()}).T[CLASSES]
            f1.columns = [short(c) for c in CLASSES]
            fig = px.imshow(f1, text_auto=".2f", color_continuous_scale=BLUES, zmin=0, zmax=1, aspect="auto")
            fig.update_layout(title="F1 por clase: dónde falla cada modelo", coloraxis_showscale=False, height=440,
                              xaxis_title="", yaxis_title="")
            fig.update_traces(hovertemplate="%{y}<br>%{x}: F1 %{z:.2f}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cm_model = st.selectbox("Modelo a inspeccionar", ALL, format_func=label)
            norm = st.toggle("Normalizar por fila (recall)", value=True)
            cm = np.array(metrics[cm_model]["confusion_matrix"], dtype=float)
            z = cm / cm.sum(1, keepdims=True) if norm else cm
            fig = px.imshow(z, x=[short(c) for c in CLASSES], y=[short(c) for c in CLASSES],
                            text_auto=".0%" if norm else ".0f", color_continuous_scale=BLUES, aspect="auto",
                            labels=dict(x="Predicción", y="Real", color="valor"))
            fig.update_layout(coloraxis_showscale=False, height=360, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("*A. aegypti* casi siempre se confunde con *A. albopictus* (mismo género, y la segunda tiene "
                   "~100 veces más fotos); *japonicus/koreicus* también cae a menudo en *albopictus*.")

        c1, c2 = st.columns(2)
        with c1:
            p = PREDS[cm_model]
            fig = go.Figure()
            for c in CLASSES:
                y = (p.y_true == c).to_numpy()
                prec, rec, _ = precision_recall_curve(y, p[c])
                ap = average_precision_score(y, p[c])
                fig.add_scatter(x=rec, y=prec, name=f"{short(c)} (AP {ap:.2f})", mode="lines",
                                line=dict(width=2, color=CLASS_COLORS[c]),
                                hovertemplate=f"{NAMES[c]}<br>recall %{{x:.2f}}<br>precisión %{{y:.2f}}<extra></extra>")
            fig.update_layout(title=f"Precisión-recall por clase · {label(cm_model)}", height=420,
                              xaxis=dict(title="recall", range=[0, 1.02]), yaxis=dict(title="precisión", range=[0, 1.02]),
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(rows.dropna(subset=["ms por imagen"]), x="ms por imagen", y="F1 macro", text="modelo",
                             symbol="familia", log_x=True)
            fig.update_traces(marker=dict(size=13, color=ACCENT, line=dict(width=2, color="white")),
                              textposition="top center")
            fig.update_layout(title="Desempeño vs costo de inferencia (GTX 1650)", height=420,
                              legend=dict(orientation="h", y=-0.25, title=""))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"##### Explorar aciertos y errores de {label(cm_model)}")
        c1, c2 = st.columns(2)
        real = c1.selectbox("Clase real", CLASSES, format_func=NAMES.get)
        pred = c2.selectbox("Clase predicha", CLASSES, index=CLASSES.index("albopictus"), format_func=NAMES.get)
        hits = p[(p.y_true == real) & (p.y_pred == pred)].sort_values(pred, ascending=False)
        st.caption(f"{len(hits)} imágenes de prueba de *{NAMES[real]}* clasificadas como *{NAMES[pred]}*"
                   + (" (aciertos)." if real == pred else " (errores)."))
        for col, (_, h) in zip(st.columns(6), hits.head(12).iterrows()):
            col.image(str(ROOT / "data" / "processed" / "crops" / f"{h.stem}.jpg"),
                      caption=f"p({short(pred)}) = {h[pred]:.0%}", use_container_width=True)

        hist = {n: pd.read_csv(f) for n in ALL if (f := ROOT / "results" / f"history_{n}.csv").exists()}
        if hist:
            fig = go.Figure()
            for (n, h), color in zip(hist.items(), [ACCENT, HIGHLIGHT]):
                fig.add_scatter(x=h.epoch, y=h.val_macro_f1, name=label(n), mode="lines+markers",
                                line=dict(width=2, color=color), marker=dict(size=8))
            fig.update_layout(title="CNN: F1 macro en validación por época", yaxis=dict(range=[0, 1]),
                              xaxis=dict(title="época", dtick=1), height=360, legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)

        det_json, det_csv = ROOT / "results" / "detector_summary.json", ROOT / "models" / "detector" / "det_test_boxes.csv"
        if det_json.exists() and det_csv.exists():
            st.markdown("##### Detector YOLOv8n (localización del mosquito)")
            d, boxes = json.loads(det_json.read_text()), pd.read_csv(det_csv)
            k = st.columns(4)
            k[0].metric("mAP@50", f"{d.get('map50', float('nan')):.3f}")
            k[1].metric("IoU medio", f"{d['mean_iou']:.3f}")
            k[2].metric("Fotos con IoU ≥ 0.75", f"{d['iou_ge_075']:.1%}")
            k[3].metric("ms por imagen", f"{d.get('infer_ms_per_img', float('nan')):.0f}")
            fig = px.histogram(boxes, x="iou", nbins=40, color_discrete_sequence=[ACCENT])
            fig.add_vline(x=0.75, line_dash="dash", line_color="#0b0b0b", annotation_text="umbral del reto 0.75")
            fig.update_layout(title="IoU entre la caja predicha y la anotada (test)", height=330, bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
