# -*- coding: utf-8 -*-
"""Entrena los experimentos de la propuesta RGB-D en NYUv2 (40 clases) y los evalúa en cada condición de la
profundidad (`rgbd.CONDICIONES`).

    python src/entrenar_rgbd.py --humo [E2u E9]              # unos pasos y 20 imágenes: que todo corre
    python src/entrenar_rgbd.py --barrido E1                 # lr en la validación por escenas, semilla 0
    python src/entrenar_rgbd.py --exp E1 --lr 1e-3 --semilla 0   # con las 795, evaluación en el test
    python src/entrenar_rgbd.py --completo E8                # barrido, elección y 3 semillas
    python src/entrenar_rgbd.py --derivar E4c E3c            # E3c a partir de los modelos de E4c, sin entrenar (D-06)
    python src/entrenar_rgbd.py --reevaluar                  # condiciones nuevas en modelos ya entrenados
    python src/entrenar_rgbd.py --listar                     # los experimentos y su configuración

Protocolo, igual para todos los experimentos:
- Encoders congelados; se entrenan la fusión y la cabeza de EfficientViT (`rgbd.RGBD`). En E1u, E7, E2u, E9,
  E10 y E11 también la última etapa del codificador de color, con una lr 10 veces menor (D-03, D-04, D-05). En E9
  el experto de color es el E1u ya entrenado de la misma semilla (en la validación, el de la validación),
  congelado (D-04). E10 es E2u con rasgos de profundidad que dependen de la entrada en las tres escalas, y E11,
  E10 con la profundidad también a la entrada de la última etapa del color (D-05).
- Familia corregida (D-06, sufijo «c»): rasgos de DeFM que dependen de la entrada en las tres escalas
  (`c3p4c5`) y el codificador de color congelado, como E1-E5. E2c (suma 1×1), y el factorial 2×2 de E5:
  E3c (sin máscara), E3cd (sin máscara + dropout 0.3), E4c (máscara binaria) y E4cd (máscara binaria + dropout
  0.3); E4cf y E4cfd, con máscara fraccional, son la ablación del tipo de máscara. Dentro de cada par solo cambia
  la máscara. El dropout y los huecos usan un generador propio, así que el orden de los datos y el aumento son
  los mismos con y sin dropout.
- Entrenamiento con la profundidad rellenada (`depths`); en E5, E3cd, E4cd y E4cfd, el 30 % de las muestras sin
  profundidad; en E8, E2h y E9, huecos por muestra (`rgbd.huecos_entrenamiento`). Aumento en la GPU: escala
  U(1, 1.5) con recorte aleatorio de 480×640 y volteo horizontal.
- AdamW (decaimiento 0.01), lote 8, 100 pasos de calentamiento y caída polinómica (0.9); bf16 en la GPU.
- Sin selección de época: se evalúa el modelo final. La lr se elige en una validación del 15 % de las escenas de
  entrenamiento (`nyuv2.validacion_por_escenas`), nunca en el test. En E7 se usa la lr elegida para E1u y se
  elige así el peso de la pérdida auxiliar. En E2u, E10, E11 y en toda la familia «c», si la mejor lr queda en un
  extremo del barrido, se prueba la siguiente de `ESCALA_LR` hacia ese lado, hasta que la mejor quede dentro.
- mIoU sobre las 40 clases a 480×640, ignorando «sin etiqueta».

Cada corrida deja en runs/rgbd/<nombre>/ resultado.json (con la huella de su configuración, el entorno y el
SHA-256 de los pesos), cabeza.pt (solo lo entrenado), estado.pt (para reanudar; nunca se borra) y, en el test,
por_imagen.npz (intersección y unión por imagen y clase, para los intervalos por remuestreo de imágenes).
Una corrida interrumpida se reanuda sola al relanzar el mismo comando. Si la carpeta tiene una corrida de otra
configuración, se detiene con ConfigConflict y no toca nada.

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyuv2  # noqa: E402
import procedencia  # noqa: E402
import rgbd  # noqa: E402
import efficientvit_seg as sg  # noqa: E402

PROTOCOLO = "v4-2026-10-29"     # cambia si cambia algo que afecte a los resultados de una configuración dada


@dataclasses.dataclass(frozen=True)
class Exp:
    """Configuración de un experimento. Lo que no está aquí es común a todos (ver la cabecera)."""
    fusion: str
    entrenable: tuple = ()
    aux: tuple = ()                 # pesos de la pérdida auxiliar que se barren (E7)
    fijo: str = None                # experto de color ya entrenado (E9)
    rasgos: str = "p3p4p5"
    codificador: bool = False
    dropout: float = 0.0            # modality dropout: proporción de muestras sin ninguna profundidad
    huecos: bool = False            # rgbd.huecos_entrenamiento
    mascara: str = "fraccional"     # solo «mlca»
    umbral: float = 0.5             # máscara binaria: celda válida si su fracción válida supera el umbral
    gen_fallos: bool = False        # dropout y huecos con un generador propio (D-06)
    borde: bool = False             # la mejor lr no puede quedar en un extremo del barrido
    historico: bool = False         # entrenado antes de D-06: sus resultados no tienen huella
    familia: str = ""
    que: str = ""


_C = dict(rasgos="c3p4c5", gen_fallos=True, borde=True)          # familia corregida (D-06)
EXPERIMENTOS = {
    # Diseño principal de la v3 (rasgos P3-P5 de la BiFPN: en 1/8 y 1/32 son constantes, D-05)
    "E1": Exp("rgb", historico=True, familia="principal v3", que="solo color"),
    "E2": Exp("concat", historico=True, familia="principal v3", que="f + W·f_prof (1×1, W=0 al inicio)"),
    "E4": Exp("mlca", historico=True, familia="principal v3", que="MLCA, máscara fraccional"),
    "E5": Exp("mlca", dropout=0.3, historico=True, familia="principal v3", que="E4 + dropout 0.3"),
    # Familia con la última etapa del color ajustada y experimentos complementarios (D-03 a D-05)
    "E1u": Exp("rgb", ("stage4",), historico=True, familia="ajuste u", que="E1 + última etapa ajustada"),
    "E7": Exp("rgb", ("stage4",), aux=(1.0, 3.0, 10.0), historico=True, familia="complementario",
              que="E1u + profundidad como maestra"),
    "E8": Exp("respaldo", huecos=True, historico=True, familia="complementario", que="dos expertos + compuerta"),
    "E2h": Exp("concat", huecos=True, historico=True, familia="complementario", que="E2 + huecos al entrenar"),
    "E2u": Exp("concat", ("stage4",), borde=True, historico=True, familia="ajuste u", que="E2 + última etapa"),
    "E9": Exp("respaldo", ("stage4",), fijo="E1u", huecos=True, historico=True, familia="complementario",
              que="E8 con el experto de color = E1u"),
    "E10": Exp("concat", ("stage4",), rasgos="c3p4c5", borde=True, historico=True, familia="ajuste u",
               que="E2u con C3/P4/C5"),
    "E11": Exp("concat", ("stage4",), rasgos="c3p4c5", codificador=True, borde=True, historico=True,
               familia="ajuste u", que="E10 + profundidad en el codificador"),
    # Familia corregida D-06: rasgos que dependen de la entrada en las tres escalas, color congelado
    "E2c": Exp("concat", **_C, familia="principal c", que="E2 con C3/P4/C5"),
    "E3c": Exp("mlca", mascara="ninguna", **_C, familia="principal c", que="atención lineal sin máscara"),
    "E3cd": Exp("mlca", mascara="ninguna", dropout=0.3, **_C, familia="principal c", que="E3c + dropout 0.3"),
    "E4c": Exp("mlca", mascara="binaria", **_C, familia="principal c", que="MLCA, máscara binaria (resumen)"),
    "E4cd": Exp("mlca", mascara="binaria", dropout=0.3, **_C, familia="principal c", que="E4c + dropout 0.3"),
    "E4cf": Exp("mlca", mascara="fraccional", **_C, familia="ablación c", que="MLCA, máscara fraccional"),
    "E4cfd": Exp("mlca", mascara="fraccional", dropout=0.3, **_C, familia="ablación c",
                 que="E4cf + dropout 0.3"),
    # D-07: la suma con el mismo entrenamiento robusto que E4cd (cierra «MLCA + dropout frente a suma»)
    "E2cd": Exp("concat", dropout=0.3, **_C, familia="robusto D-07", que="E2c + dropout 0.3"),
    # D-07: E11 robusto en un solo modelo (generador propio de fallos: mismo orden de datos que E11 por semilla)
    "E11d": Exp("concat", ("stage4",), rasgos="c3p4c5", codificador=True, dropout=0.3, gen_fallos=True, borde=True,
                familia="robusto D-07", que="E11 + dropout 0.3"),
}
# Diccionarios de la v3, derivados del registro (los usan los demás scripts)
FUSION = {e: c.fusion for e, c in EXPERIMENTOS.items()}
DROPOUT = {e: c.dropout for e, c in EXPERIMENTOS.items() if c.dropout}
HUECOS = {e for e, c in EXPERIMENTOS.items() if c.huecos}
ENTRENABLE = {e: c.entrenable for e, c in EXPERIMENTOS.items() if c.entrenable}
FIJO = {e: c.fijo for e, c in EXPERIMENTOS.items() if c.fijo}
RASGOS = {e: c.rasgos for e, c in EXPERIMENTOS.items() if c.rasgos != "p3p4p5"}
CODIFICADOR = {e for e, c in EXPERIMENTOS.items() if c.codificador}
AUX = {e: c.aux for e, c in EXPERIMENTOS.items() if c.aux}
BORDE = {e for e, c in EXPERIMENTOS.items() if c.borde}
FACTOR_COD = 0.1            # lr de esas etapas = 0.1 × la de la cabeza
LRS = (3e-4, 1e-3, 3e-3)
ESCALA_LR = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
DEV = os.environ.get("RGBD_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
AMP = os.environ.get("RGBD_AMP", "1" if DEV.startswith("cuda") else "0") not in ("", "0")   # bf16
IGNORAR = 255
_PRUEBA = nyuv2.sinteticos_activos() or procedencia.sin_pesos()
RUNS = Path(os.environ["RGBD_RUNS"]) if os.environ.get("RGBD_RUNS") else (
    sg.RAIZ / "runs" / ("_prueba" if _PRUEBA else "") / "rgbd")


class ConfigConflict(RuntimeError):
    """La carpeta de una corrida ya tiene resultados o un estado de otra configuración."""


def autocast():
    return torch.autocast(DEV.split(":")[0], dtype=torch.bfloat16, enabled=AMP)


def lote_a_gpu(datos, idx):
    """Arrays del memmap -> tensores en el dispositivo: rgb (B,3,H,W) uint8, prof en mm, etiqueta 0..39 / 255."""
    idx = np.sort(idx)
    rgb = torch.from_numpy(np.ascontiguousarray(datos["rgb"][idx])).to(DEV).permute(0, 3, 1, 2)
    prof = torch.from_numpy(datos["prof"][idx].astype(np.float32)).to(DEV)[:, None]
    et = torch.from_numpy(datos["etiqueta"][idx].astype(np.int64)).to(DEV)
    et = torch.where(et == 0, torch.full_like(et, IGNORAR), et - 1)
    return rgb, prof, et


def aumentar(rgb, prof, et, gen):
    b, _, h, w = rgb.shape
    s = 1 + 0.5 * torch.rand(1, generator=gen).item()
    H, W = round(h * s), round(w * s)
    rgb = F.interpolate(rgb.float(), (H, W), mode="bilinear", align_corners=False)
    prof = F.interpolate(prof, (H, W), mode="nearest")
    et = F.interpolate(et[:, None].float(), (H, W), mode="nearest")[:, 0].long()
    sal = [], [], []
    for i in range(b):
        y = torch.randint(0, H - h + 1, (1,), generator=gen).item()
        x = torch.randint(0, W - w + 1, (1,), generator=gen).item()
        t = [rgb[i, :, y:y + h, x:x + w], prof[i, :, y:y + h, x:x + w], et[i, y:y + h, x:x + w]]
        if torch.rand(1, generator=gen).item() < 0.5:
            t = [z.flip(-1) for z in t]
        for lista, z in zip(sal, t):
            lista.append(z)
    return [torch.stack(z) for z in sal]


def normalizar_rgb(rgb):
    return (rgb.float() / 255 - sg.MEAN.to(rgb.device)) / sg.STD.to(rgb.device)


def ce(lg, et):
    """Entropía cruzada a la resolución de la etiqueta; 0 si el lote no tiene ningún píxel etiquetado."""
    if not (et != IGNORAR).any():
        return lg.sum() * 0
    lg = F.interpolate(lg.float(), et.shape[-2:], mode="bilinear", align_corners=False)
    return F.cross_entropy(lg, et, ignore_index=IGNORAR)


def perdida_prof(pred, prof):
    """Error cuadrático del log de la profundidad, invariante a la escala de cada imagen (Eigen et al., 2014):
    var(log p̂ − log p) en las celdas de 1/8 con más de la mitad de los píxeles medidos. Invariante a la escala
    porque el aumento amplía la imagen sin cambiar la profundidad. pred (B,1,h,w); prof (B,1,H,W) en mm."""
    tam = pred.shape[-2:]
    v = (prof > 0).float()
    num, den = F.adaptive_avg_pool2d(prof * v, tam), F.adaptive_avg_pool2d(v, tam)
    ok = den > 0.5
    obj = torch.log((num / den.clamp_min(1e-6)).clamp_min(1.0) / 1e3)
    g = torch.where(ok, pred.float() - obj, torch.zeros_like(obj))
    n = ok.sum((1, 2, 3)).clamp_min(1)
    m1, m2 = g.sum((1, 2, 3)) / n, (g ** 2).sum((1, 2, 3)) / n
    return (m2 - m1 ** 2).mean()


def construir(exp):
    c = EXPERIMENTOS[exp]
    return rgbd.RGBD(c.fusion, entrenable=c.entrenable, aux_prof=bool(c.aux), experto_fijo=c.fijo is not None,
                     rasgos_prof=c.rasgos, codificador=c.codificador, mascara=c.mascara,
                     umbral=c.umbral).to(DEV)


def huella(exp, lr, semilla, epocas, lote, validacion, humo, aux, derivado_de=None):
    """Identifica todo lo que determina el resultado de una corrida (salvo el no determinismo de la GPU)."""
    c = dataclasses.asdict(EXPERIMENTOS[exp])
    for k in ("familia", "que", "historico", "borde"):
        c.pop(k)
    clave = {"exp": exp, "config": c, "lr": lr, "semilla": semilla, "epocas": epocas, "lote": lote,
             "validacion": validacion, "humo": humo, "aux": aux, "protocolo": PROTOCOLO,
             "datos": "sinteticos" if nyuv2.sinteticos_activos() else "nyuv2", "sin_pesos": procedencia.sin_pesos(),
             "derivado_de": derivado_de}
    return hashlib.sha256(json.dumps(clave, sort_keys=True).encode()).hexdigest()[:16]


def resultado_valido(carpeta, exp, h):
    """True si la carpeta ya tiene el resultado de esta configuración; ConfigConflict si tiene el de otra. Los
    resultados de los experimentos históricos no tienen huella y se aceptan tal como están."""
    ruta = carpeta / "resultado.json"
    if not ruta.exists():
        return False
    out = json.loads(ruta.read_text(encoding="utf-8"))
    if "huella" not in out:
        if EXPERIMENTOS[exp].historico:
            return True
        raise ConfigConflict(f"{ruta}: resultado sin huella para un experimento de D-06")
    if out["huella"] != h:
        raise ConfigConflict(f"{ruta}: huella {out['huella']} distinta de la esperada {h}")
    return True


def experto_de(exp, semilla, validacion):
    """Carpeta del modelo que hace de experto de color fijo (E9): el de `FIJO[exp]` con su lr elegida y la misma
    semilla; en la validación, su corrida de validación, que no vio esas escenas."""
    base = FIJO[exp]
    (lr, _), _ = elegida(base)
    return RUNS / nombre_corrida(base, lr, 0 if validacion else semilla, validacion)


def nombre_corrida(exp, lr, semilla, validacion=False, aux=None, humo=False):
    return (f"{'val_' if validacion else ''}{exp}_lr{lr:g}{'' if aux is None else f'_aux{aux:g}'}"
            f"_s{semilla}{'_humo' if humo else ''}")


def cargar(carpeta, exp=None):
    """Modelo final de una corrida (runs/rgbd/<nombre>/): se construye según su resultado.json y carga cabeza.pt."""
    out = json.loads((carpeta / "resultado.json").read_text(encoding="utf-8"))
    modelo = construir(exp or out["exp"])
    origen = carpeta / "cabeza.pt" if (carpeta / "cabeza.pt").exists() else RUNS / out["derivado_de"] / "cabeza.pt"
    info = modelo.load_state_dict(torch.load(origen, map_location=DEV, weights_only=True), strict=False)
    assert not info.unexpected_keys and all(k.startswith(("evit.backbone", "defm")) for k in info.missing_keys)
    return modelo, out


def _guardables(modelo, nombres):
    return {k: v for k, v in modelo.state_dict().items()
            if not k.startswith(("evit.backbone", "defm")) or k in nombres}


def evaluar(modelo, datos, posiciones, condiciones, lote=4, por_imagen=None):
    """mIoU por condición. Las condiciones son deterministas por imagen (semilla = índice en el split). Con
    `por_imagen` (un dict), guarda ahí la intersección y la unión por imagen y clase de cada condición."""
    modelo.eval()
    res = {}
    for cond in condiciones:
        conf = torch.zeros(40, 40, dtype=torch.long, device=DEV)
        inter, union = [], []
        for k in range(0, len(posiciones), lote):
            idx = np.sort(posiciones[k:k + lote])
            rgb, _, et = lote_a_gpu(datos, idx)
            pv = [rgbd.condicion(cond, datos["prof"][i], datos["cruda"][i], semilla=int(i)) for i in idx]
            prof = torch.from_numpy(np.stack([p for p, _ in pv])).to(DEV)[:, None]
            valida = torch.from_numpy(np.stack([v for _, v in pv])).to(DEV)[:, None]
            with torch.no_grad(), autocast():
                lg = modelo(normalizar_rgb(rgb), prof, valida)
            pred = F.interpolate(lg.float(), et.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
            for j in range(len(idx)):
                m = et[j] != IGNORAR
                c = torch.bincount(40 * et[j][m] + pred[j][m], minlength=1600).view(40, 40)
                conf += c
                if por_imagen is not None:
                    d = c.diag()
                    inter.append(d.cpu().numpy())
                    union.append((c.sum(0) + c.sum(1) - d).cpu().numpy())
        conf = conf.double()
        iou = (conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp(min=1)).cpu().numpy()
        res[cond] = {"miou": float(100 * iou.mean()), "acc_pixel": float(100 * conf.diag().sum() / conf.sum()),
                     "iou_clase": [round(float(100 * x), 2) for x in iou]}
        if por_imagen is not None:
            por_imagen[f"{cond}_inter"] = np.stack(inter).astype(np.int32)
            por_imagen[f"{cond}_union"] = np.stack(union).astype(np.int32)
        if modelo.fusion == "rgb":                       # sin profundidad, todas las condiciones dan lo mismo
            res = {c: res[cond] for c in condiciones}
            if por_imagen is not None:
                por_imagen["nota"] = np.array("modelo sin profundidad: una sola condición, igual en todas")
            break
    return res


def _escribir(ruta, datos):
    """Escritura atómica: un corte no deja un archivo a medias."""
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    if isinstance(datos, (dict, list)) and ruta.suffix == ".json":
        tmp.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    else:
        torch.save(datos, tmp)
    os.replace(tmp, ruta)


def entrenar(exp, lr, semilla, epocas=60, lote=8, validacion=False, humo=False, nombre=None, aux=None,
             guardar_cada=500, pasos_max=None, cortar_en=None):
    """Entrena y evalúa una corrida. `pasos_max` acorta el entrenamiento (pruebas); `cortar_en` simula un corte
    después de ese paso (pruebas de reanudación)."""
    cfg = EXPERIMENTOS[exp]
    if cfg.aux and aux is None:
        aux = cfg.aux[0]
    nombre = nombre or nombre_corrida(exp, lr, semilla, validacion, aux, humo)
    salida = RUNS / nombre
    h = huella(exp, lr, semilla, epocas, lote, validacion, humo, aux)
    if resultado_valido(salida, exp, h):
        print(f"{nombre}: ya está (huella {h})", flush=True)
        return json.loads((salida / "resultado.json").read_text(encoding="utf-8"))
    salida.mkdir(parents=True, exist_ok=True)
    try:
        return _entrenar(exp, cfg, lr, semilla, epocas, lote, validacion, humo, aux, nombre, salida, h,
                         guardar_cada, pasos_max, cortar_en)
    except Exception:
        (salida / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _entrenar(exp, cfg, lr, semilla, epocas, lote, validacion, humo, aux, nombre, salida, h, guardar_cada,
              pasos_max, cortar_en):
    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    # D-06: dropout y huecos con su propio generador, para que no cambien el orden de los datos ni el aumento
    gen_f = torch.Generator().manual_seed(semilla + 1_000_003) if cfg.gen_fallos else gen
    tr_datos = nyuv2.cache("entrenamiento")
    if validacion:
        pos_tr, pos_ev = nyuv2.validacion_por_escenas()
        ev_datos, condiciones = tr_datos, ["completa"]
    else:
        pos_tr, ev_datos = np.arange(len(tr_datos["rgb"])), nyuv2.cache("test")
        pos_ev = np.arange(len(ev_datos["rgb"]))
        condiciones = rgbd.CONDICIONES + ([] if cfg.historico else rgbd.CONDICIONES_EXTRA)
    if humo:
        epocas, pos_ev = 1, pos_ev[:20]

    modelo = construir(exp)
    experto = None
    if exp in FIJO:
        experto = experto_de(exp, semilla, validacion)
        modelo.fijar_experto(torch.load(experto / "cabeza.pt", map_location=DEV, weights_only=True))
    nombres = [n for n, p in modelo.named_parameters() if p.requires_grad]
    cod = [p for n, p in modelo.named_parameters() if p.requires_grad and n.startswith("evit.backbone")]
    resto = [p for n, p in modelo.named_parameters() if p.requires_grad and not n.startswith("evit.backbone")]
    grupos = [{"params": resto}] + ([{"params": cod, "lr": lr * FACTOR_COD}] if cod else [])
    opt = torch.optim.AdamW(grupos, lr=lr, weight_decay=0.01)
    pasos_ep = len(pos_tr) // lote
    total = 5 if humo else epocas * pasos_ep
    if pasos_max:
        total = min(total, pasos_max)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / 100) * (1 - min(t, total - 1) / total) ** 0.9)
    print(f"{nombre}: {sum(p.numel() for p in modelo.entrenables()) / 1e6:.2f} M entrenables "
          f"({sum(p.numel() for p in cod) / 1e6:.2f} M del codificador), {total} pasos, huella {h}", flush=True)

    paso, historial, orden, k0, t_previo = 0, [], None, 0, 0.0
    ruta_estado = salida / "estado.pt"
    if ruta_estado.exists():                                # reanudar (D-06)
        e = torch.load(ruta_estado, map_location="cpu", weights_only=False)
        if e["huella"] != h:
            raise ConfigConflict(f"{ruta_estado}: huella {e['huella']} distinta de {h}")
        info = modelo.load_state_dict({k: v.to(DEV) for k, v in e["modelo"].items()}, strict=False)
        assert not info.unexpected_keys
        opt.load_state_dict(e["opt"]); sched.load_state_dict(e["sched"])
        paso, historial, orden, k0, t_previo = e["paso"], e["historial"], e["orden"], e["k"], e["segundos"]
        gen.set_state(e["gen"]); gen_f.set_state(e["gen_f"])
        torch.set_rng_state(e["torch"]); np.random.set_state(e["numpy"]); random.setstate(e["random"])
        if torch.cuda.is_available() and e.get("cuda") is not None:
            torch.cuda.set_rng_state_all(e["cuda"])
        print(f"  reanudada en el paso {paso}/{total}", flush=True)

    def guardar_estado():
        _escribir(ruta_estado, {
            "huella": h, "modelo": {k: v.cpu() for k, v in _guardables(modelo, nombres).items()},
            "opt": opt.state_dict(), "sched": sched.state_dict(), "paso": paso, "historial": historial,
            "orden": orden, "k": k_sig, "segundos": t_previo + time.time() - t0, "gen": gen.get_state(),
            "gen_f": gen_f.get_state(), "torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "random": random.getstate(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})

    t0 = time.time()
    modelo.train()
    while paso < total:
        if orden is None:
            orden, k0 = pos_tr[torch.randperm(len(pos_tr), generator=gen).numpy()], 0
        for k in range(k0, len(orden) - lote + 1, lote):
            rgb, prof, et = aumentar(*lote_a_gpu(tr_datos, orden[k:k + lote]), gen)
            if cfg.dropout:                         # la muestra pierde toda la profundidad, como «sin»
                quita = torch.rand(len(prof), generator=gen_f) < cfg.dropout
                prof = torch.where(quita.to(DEV)[:, None, None, None], torch.zeros_like(prof), prof)
            if cfg.huecos:
                prof = rgbd.huecos_entrenamiento(prof, gen_f)
            with autocast():
                sal = modelo(normalizar_rgb(rgb), prof / 1e3, prof > 0, todo=True)
            perdida = ce(sal["seg"], et)
            if "rgb" in sal:      # E8: cada experto con su pérdida; el de profundidad, solo donde hay medida
                et_fus = torch.where(prof[:, 0] > 0, et, torch.full_like(et, IGNORAR))
                if exp in FIJO:   # E9: el experto de color no se entrena
                    perdida = perdida + ce(sal["fus"], et_fus)
                else:
                    perdida = perdida + ce(sal["rgb"], et) + ce(sal["fus"], et_fus)
            if "prof" in sal:     # E7: la profundidad solo enseña
                perdida = perdida + aux * perdida_prof(sal["prof"], prof)
            if not torch.isfinite(perdida):
                raise FloatingPointError(f"pérdida no finita en el paso {paso}")
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            opt.step(); sched.step(); paso += 1
            historial.append(float(perdida))
            k_sig = k + lote
            if paso % pasos_ep == 0 or paso == total:
                print(f"  paso {paso}/{total}  pérdida {np.mean(historial[-pasos_ep:]):.3f}  "
                      f"{(t_previo + time.time() - t0) / 60:.1f} min", flush=True)
            if guardar_cada and (paso % guardar_cada == 0 or paso == cortar_en) and paso < total:
                guardar_estado()
            if cortar_en is not None and paso == cortar_en:
                raise KeyboardInterrupt(f"corte simulado en el paso {paso}")
            if paso == total:
                break
        orden, k0 = None, 0

    t_ent = t_previo + time.time() - t0
    por_imagen = None if validacion else {}
    res = evaluar(modelo, ev_datos, pos_ev, condiciones, por_imagen=por_imagen)
    out = {"exp": exp, "fusion": cfg.fusion, "lr": lr, "semilla": semilla, "epocas": epocas, "lote": lote,
           "pasos": total, "validacion": validacion, "humo": humo, "minutos_entrenamiento": t_ent / 60,
           "perdida_final": float(np.mean(historial[-pasos_ep:])), "clases": nyuv2.clases40()[1:],
           "evaluado_en": "validación por escenas" if validacion else "test oficial (654)", "resultados": res,
           "huella": h, "config": dataclasses.asdict(cfg), "protocolo": PROTOCOLO, "dispositivo": DEV, "bf16": AMP,
           "entorno": procedencia.entorno()}
    if aux is not None:
        out["aux"] = aux
    if cfg.entrenable:
        out["entrenable"] = list(cfg.entrenable)
    if experto is not None:
        out["experto"] = experto.name
    torch.save(_guardables(modelo, nombres), salida / "cabeza.pt")
    if por_imagen:
        np.savez_compressed(salida / "por_imagen.npz", indices=np.asarray(pos_ev), **por_imagen)
    _escribir(salida / "resultado.json", out)       # el último: su presencia marca la corrida como terminada
    print("  " + "  ".join(f"{c} {r['miou']:.2f}" for c, r in res.items()), flush=True)
    return out


def elegida(exp):
    """(lr, aux) con mayor mIoU en la validación por escenas (profundidad completa), y todas las cifras."""
    val = {}
    for ruta in RUNS.glob(f"val_{exp}_lr*_s0/resultado.json"):
        out = json.loads(ruta.read_text(encoding="utf-8"))
        if out["exp"] != exp:
            continue
        if not EXPERIMENTOS[exp].historico and "huella" not in out:
            raise ConfigConflict(f"{ruta}: validación sin huella")
        val[(out["lr"], out.get("aux"))] = out["resultados"]["completa"]["miou"]
    if not val:
        raise FileNotFoundError(f"no hay barrido de {exp} en {RUNS}")
    # D-06: con empates, la menor lr (antes decidía el orden en que el sistema de archivos lista las carpetas)
    claves = sorted(val, key=lambda k: (k[0], -1 if k[1] is None else k[1]))
    return max(claves, key=val.get), val


def completo(exp, **kw):
    """Barrido en la validación por escenas, elección y 3 semillas en el test. En E7 la lr es la elegida para E1u
    y lo que se barre es el peso de la pérdida auxiliar."""
    if exp in AUX:
        (lr, _), _ = elegida("E1u")
        opciones = [(lr, a) for a in AUX[exp]]
    else:
        opciones = [(lr, None) for lr in LRS]
    for lr, a in opciones:
        entrenar(exp, lr, 0, validacion=True, aux=a, **kw)
    while exp in BORDE:                 # D-05: la mejor lr no puede quedar en un extremo del barrido
        (lr, _), val = elegida(exp)
        probadas, i = sorted(l for l, _ in val), ESCALA_LR.index(lr)
        if lr == probadas[-1] and i + 1 < len(ESCALA_LR):
            entrenar(exp, ESCALA_LR[i + 1], 0, validacion=True, **kw)
        elif lr == probadas[0] and i > 0:
            entrenar(exp, ESCALA_LR[i - 1], 0, validacion=True, **kw)
        else:
            break
    (lr, a), val = elegida(exp)
    print(f"{exp}: validación " + ", ".join(f"lr {l:g}{'' if x is None else f' aux {x:g}'}: {v:.2f}"
                                            for (l, x), v in sorted(val.items(), key=str)), flush=True)
    print(f"{exp}: elegida lr {lr:g}" + ("" if a is None else f", aux {a:g}"), flush=True)
    for s in range(3):
        entrenar(exp, lr, s, aux=a, **kw)


# ------------------------------------------------------------------------- derivar sin entrenar (D-06)
def equivalentes_en_entrenamiento(origen, destino):
    """¿Dan la misma trayectoria de entrenamiento? Solo si difieren únicamente en la máscara y la máscara no
    puede actuar al entrenar: sin dropout ni huecos todas las celdas valen 1 (profundidad rellenada sin ceros)
    con cualquier modo; con dropout, las muestras completas valen 1 y las quitadas 0 en «binaria» y en
    «fraccional», pero no en «ninguna»."""
    a, b = EXPERIMENTOS[origen], EXPERIMENTOS[destino]
    if a.fusion != "mlca" or dataclasses.replace(a, mascara=b.mascara, familia=b.familia, que=b.que) != b:
        return False
    if a.huecos:
        return False
    if a.dropout:
        return {a.mascara, b.mascara} <= {"binaria", "fraccional"}
    return True


def ceros_en_entrenamiento(datos=None):
    """Fracción de píxeles sin profundidad en la profundidad rellenada del entrenamiento. Debe ser 0 para que la
    máscara no actúe al entrenar."""
    datos = datos or nyuv2.cache("entrenamiento")
    n = sum(int((datos["prof"][i] == 0).sum()) for i in range(len(datos["prof"])))
    return n / datos["prof"].size


def derivar(origen, destino):
    """Evalúa los modelos ya entrenados de `origen` con la máscara de `destino` (mismas lr y semillas, también en
    la validación) y guarda los resultados como corridas de `destino` con «derivado_de». Solo si las dos
    configuraciones tienen la misma trayectoria de entrenamiento."""
    if not equivalentes_en_entrenamiento(origen, destino):
        raise ValueError(f"{origen} y {destino} no se entrenan igual: hay que entrenar {destino}")
    ceros = ceros_en_entrenamiento()
    if ceros > 0:
        raise ValueError(f"la profundidad rellenada tiene {100 * ceros:.4f} % de ceros: la máscara actúa al entrenar")
    cfg = EXPERIMENTOS[destino]
    (lr, _), _ = elegida(origen)
    tr, te = nyuv2.cache("entrenamiento"), nyuv2.cache("test")
    _, pos_val = nyuv2.validacion_por_escenas()
    corridas = [(l, 0, True) for (l, _) in sorted(elegida(origen)[1])] + [(lr, s, False) for s in range(3)]
    for l, s, val in corridas:
        src = nombre_corrida(origen, l, s, val)
        dst = RUNS / nombre_corrida(destino, l, s, val)
        base = json.loads((RUNS / src / "resultado.json").read_text(encoding="utf-8"))
        h = huella(destino, l, s, base["epocas"], base["lote"], val, base["humo"], None, derivado_de=src)
        if resultado_valido(dst, destino, h):
            continue
        modelo, _ = cargar(RUNS / src, exp=destino)
        assert modelo.mascara == cfg.mascara
        por_imagen = None if val else {}
        res = evaluar(modelo, tr if val else te, pos_val if val else np.arange(len(te["rgb"])),
                      ["completa"] if val else rgbd.CONDICIONES + rgbd.CONDICIONES_EXTRA, por_imagen=por_imagen)
        out = {**{k: base[k] for k in ("lr", "semilla", "epocas", "lote", "pasos", "validacion", "humo",
                                         "minutos_entrenamiento", "perdida_final", "clases", "evaluado_en")},
               "exp": destino, "fusion": cfg.fusion, "resultados": res, "huella": h, "derivado_de": src,
               "nota": f"pesos de {src}, evaluados con la máscara «{cfg.mascara}» (D-06)",
               "ceros_en_entrenamiento": ceros, "config": dataclasses.asdict(cfg), "protocolo": PROTOCOLO,
               "dispositivo": DEV, "bf16": AMP, "entorno": procedencia.entorno()}
        dst.mkdir(parents=True, exist_ok=True)
        if por_imagen:
            np.savez_compressed(dst / "por_imagen.npz", indices=np.arange(len(te["rgb"])), **por_imagen)
        _escribir(dst / "resultado.json", out)
        print(f"{dst.name} (de {src}): " + "  ".join(f"{c} {r['miou']:.2f}" for c, r in res.items()), flush=True)


def reevaluar():
    """Evalúa los modelos finales ya entrenados (runs/rgbd/E*_s*) en las condiciones que les falten, sin
    reentrenar: carga cabeza.pt y añade los resultados a su resultado.json. No cambia las cifras existentes."""
    for ruta in sorted(RUNS.glob("E*_lr*_s[0-9]/resultado.json")):
        out = json.loads(ruta.read_text(encoding="utf-8"))
        conds = rgbd.CONDICIONES + ([] if EXPERIMENTOS[out["exp"]].historico else rgbd.CONDICIONES_EXTRA)
        faltan = [c for c in conds if c not in out["resultados"]]
        if not faltan:
            continue
        modelo, _ = cargar(ruta.parent)
        datos = nyuv2.cache("test")
        nuevos = evaluar(modelo, datos, np.arange(len(datos["rgb"])), faltan)
        out["resultados"].update({c: nuevos[c] for c in faltan})
        _escribir(ruta, out)
        print(ruta.parent.name, "  ".join(f"{c} {nuevos[c]['miou']:.2f}" for c in faltan), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(FUSION))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--aux", type=float, default=None)
    ap.add_argument("--barrido", choices=list(FUSION))
    ap.add_argument("--completo", choices=list(FUSION), nargs="+")
    ap.add_argument("--derivar", nargs=2, metavar=("ORIGEN", "DESTINO"))
    ap.add_argument("--humo", choices=list(FUSION), nargs="*", help="sin nombres, todos los experimentos")
    ap.add_argument("--reevaluar", action="store_true")
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--guardar-cada", type=int, default=500)
    a = ap.parse_args(argv)
    kw = {"guardar_cada": a.guardar_cada}
    if a.listar:
        for e, c in EXPERIMENTOS.items():
            print(f"{e:6s} {c.familia:14s} {c.que}")
    elif a.reevaluar:
        reevaluar()
    elif a.derivar:
        derivar(*a.derivar)
    elif a.humo is not None:
        for exp in a.humo or [e for e in FUSION if e not in FIJO]:
            entrenar(exp, 1e-3, 0, humo=True, **kw)
    elif a.completo:
        for exp in a.completo:
            completo(exp, **kw)
    elif a.barrido:
        for lr in LRS:
            entrenar(a.barrido, lr, 0, epocas=a.epocas, validacion=True, **kw)
    elif a.exp:
        entrenar(a.exp, a.lr, a.semilla, epocas=a.epocas, aux=a.aux, **kw)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
