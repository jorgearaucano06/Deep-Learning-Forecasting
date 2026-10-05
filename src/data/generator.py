"""
Generador de Datos Sinteticos - Red de Fibra Optica GPON
==========================================================
Simula una red GPON (Gigabit Passive Optical Network) con 200 segmentos
generando lecturas de sensores cada 15 minutos durante 7 dias.

=====================================================================
FUNDAMENTO DE LOS DATOS GENERADOS
=====================================================================

1. QUE EQUIPO GENERA ESTOS DATOS EN LA VIDA REAL?
   - OTDR (Optical Time Domain Reflectometer): Equipos de EXFO, VIAVI, Anritsu.
     Envian un pulso de luz y miden lo que rebota. Genera archivos .SOR
     (Standard OTDR Record, estandar Bellcore SR-4731).
     Columnas del .SOR: distancia (m) y potencia reflejada (dBm).
   - OLT/ONT (equipos de red GPON): Reportan en tiempo real via SNMP/OMCI:
     potencia optica recibida (dBm), BER, estado del enlace.
   - Sistemas NMS (Network Management System): Agregan todas las metricas
     y generan alarmas.

2. DE DONDE VIENEN LOS VALORES QUE USAMOS?
   - ITU-T G.652 (2024): Define la fibra monomodo estandar.
     Atenuacion maxima: 0.25 dB/km @ 1550nm, 0.40 dB/km @ 1310nm.
   - ITU-T G.984.2: Define GPON.
     Potencia OLT: +3 a +7 dBm. Sensibilidad ONT: -28 dBm (APD).
     BER objetivo: <1e-10 post-FEC.
   - TIA/EIA-568: Define perdidas maximas aceptables.
     Conector: max 0.75 dB. Empalme mecanico: max 0.3 dB.
   - FOA (Fiber Optic Association): Valores tipicos de campo.
     Empalme fusion: 0.05-0.15 dB. Conector tipico: 0.3 dB.

3. EXISTE DATA PUBLICA REAL?
   - IEEE DataPort: "OTDR dataset for optical fiber monitoring" (Abdelli et al.)
     Trazas OTDR reales con eventos de reflectancia y fallas.
   - IEEE DataPort: "Dataset for Optical fiber faults"
     Datos de corte, empalme malo, conector sucio, fiber tapping.
   - Ambos requieren suscripcion IEEE pero confirman nuestros rangos.

4. COMO VALIDAMOS QUE NUESTROS DATOS SON REALISTAS?
   - Los rangos normales coinciden con los estandares ITU-T y FOA.
   - Los valores de falla reproducen las firmas documentadas en papers
     de fault detection en PON (arxiv 2202.11756, 2204.07059, 2203.11727).
   - Las transiciones pre-falla modelan fenomenos fisicos reales:
     microcurvaturas (pre-corte), envejecimiento (degradacion),
     inestabilidad termica de empalme (bad splice).

5. QUE FALTA PARA SER 100% REAL?
   - No incluimos trazas OTDR completas (distancia vs potencia), solo
     metricas agregadas por segmento. Un sistema real tendria ambas.
   - No modelamos el splitter 1:32 explicitamente (asumimos la perdida
     ya esta incluida en la potencia recibida).
   - No incluimos PMD (Polarization Mode Dispersion), que es relevante
     solo en enlaces >80 km.
=====================================================================

Metricas generadas por segmento cada 15 minutos:
- optical_power_dbm: Potencia optica recibida [-18, -8] dBm (ITU-T G.984.2)
- attenuation_db_km: Atenuacion de la fibra [0.18, 0.25] dB/km (ITU-T G.652)
- ber: Bit Error Rate [1e-12, 1e-9] (ITU-T G.984.2)
- osnr_db: Optical Signal-to-Noise Ratio [20, 35] dB (OPM research)
- chromatic_dispersion: Coeficiente CD [15, 18] ps/(nm*km) (ITU-T G.652)
- temperature: Temperatura ambiental del ducto (SENAMHI Lima)
- humidity: Humedad relativa (SENAMHI Lima)
- segment_length_km: Longitud del segmento [5, 20] km

Tipos de falla y sus firmas fisicas:
1. Corte fisico (25%): Potencia cae a <-40 dBm, BER sube a 0.5.
   Causa real: excavaciones, roedores, vandalismo.
   Firma pre-falla: microcurvaturas causan caidas de 1-3 dB.
2. Degradacion gradual (50%): Atenuacion sube gradualmente.
   Causa real: envejecimiento, humedad en empalmes, estres mecanico.
   Firma pre-falla: deterioro lento de todas las metricas.
3. Empalme defectuoso (25%): Oscilaciones de +/-2 dB.
   Causa real: fusion incompleta, alineacion incorrecta, contaminacion.
   Firma pre-falla: inestabilidad ciclica en potencia optica.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple

from src.utils.config_loader import load_config, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FiberNetworkGenerator:
    """
    Genera datos sinteticos de una red GPON basados en estandares ITU-T.

    Cada segmento tiene propiedades fisicas individuales (longitud, numero
    de empalmes, tipo de conector) que afectan sus metricas base.
    Esto es mas realista que asumir que todos los segmentos son identicos.
    """

    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        gen_cfg = self.config["data_generation"]

        self.n_segments = gen_cfg["n_segments"]
        self.days = gen_cfg["days"]
        self.interval_min = gen_cfg["reading_interval_min"]
        self.seed = gen_cfg["random_seed"]
        self.rng = np.random.default_rng(self.seed)

        self.fault_dist = gen_cfg["fault_distribution"]
        self.pre_fault = gen_cfg["pre_fault_signature"]
        self.normal_ranges = gen_cfg["normal_ranges"]
        self.fault_values = gen_cfg["fault_values"]
        self.network = gen_cfg["network"]
        self.system_losses = gen_cfg["system_losses"]

        self.n_timestamps = self.days * 24 * (60 // self.interval_min)
        logger.info(
            "Generador GPON inicializado: %d segmentos x %d timestamps = %d lecturas",
            self.n_segments, self.n_timestamps,
            self.n_segments * self.n_timestamps,
        )

    def _generate_segment_properties(self) -> pd.DataFrame:
        """
        Genera propiedades fisicas individuales para cada segmento.

        En una red real, cada segmento de fibra tiene:
        - Diferente longitud (5-20 km en GPON urbano)
        - Diferente numero de empalmes (1 cada ~2 km aprox)
        - Diferente numero de conectores (2-4 por segmento)

        Estas propiedades determinan la potencia optica base de cada segmento:
        Potencia recibida = Potencia OLT - (atenuacion_fibra * km)
                           - (perdida_empalme * n_empalmes)
                           - (perdida_conector * n_conectores)
                           - perdida_splitter

        Fuente: FOA link loss budget formula
        """
        segment_ids = [f"SEG-{i:03d}" for i in range(1, self.n_segments + 1)]

        # Longitud de cada segmento (km)
        min_km, max_km = self.network["segment_length_km"]
        lengths = self.rng.uniform(min_km, max_km, self.n_segments)

        # Numero de empalmes (aprox 1 cada 2-4 km)
        n_splices = np.maximum(1, (lengths / self.rng.uniform(2, 4, self.n_segments)).astype(int))

        # Numero de conectores (2-4 por segmento: OLT + splitter + ONT + posible patch panel)
        n_connectors = self.rng.integers(2, 5, self.n_segments)

        # Calcular potencia base recibida para cada segmento
        # OLT transmite entre +3 y +7 dBm (ITU-T G.984.2)
        olt_power = self.rng.uniform(3.0, 7.0, self.n_segments)

        # Perdida total del enlace
        fiber_loss = lengths * self.rng.uniform(0.18, 0.22, self.n_segments)  # dB (G.652 @ 1550nm)
        splice_loss = n_splices * self.system_losses["fusion_splice_loss_db"]
        connector_loss = n_connectors * self.system_losses["connector_loss_db"]
        splitter_loss = self.system_losses["splitter_1x32_loss_db"]

        total_loss = fiber_loss + splice_loss + connector_loss + splitter_loss
        base_power = olt_power - total_loss  # Potencia en el receptor ONT

        # Clipear al rango de operacion del receptor APD (-28 a -8 dBm)
        base_power = np.clip(base_power, -28.0, -8.0)

        props = pd.DataFrame({
            "segment_id": segment_ids,
            "length_km": np.round(lengths, 2),
            "n_splices": n_splices,
            "n_connectors": n_connectors,
            "olt_power_dbm": np.round(olt_power, 2),
            "total_loss_db": np.round(total_loss, 2),
            "base_power_dbm": np.round(base_power, 2),
        })

        logger.info(
            "Propiedades de segmento - Long: %.1f-%.1f km, Potencia base: %.1f a %.1f dBm",
            lengths.min(), lengths.max(), base_power.min(), base_power.max(),
        )
        return props

    def generate(self) -> pd.DataFrame:
        """
        Pipeline principal de generacion.

        Returns:
            DataFrame con ~134,400 filas (200 seg * 672 timestamps)
            y columnas de metricas, ambiente, falla, y propiedades del segmento.
        """
        logger.info("=== Iniciando generacion de datos GPON ===")

        # Paso 1: Propiedades fisicas de cada segmento
        segment_props = self._generate_segment_properties()

        # Paso 2: Estructura temporal
        df = self._create_base_structure(segment_props)
        logger.info("Estructura base: %s", df.shape)

        # Paso 3: Lecturas normales (basadas en propiedades fisicas)
        df = self._add_normal_readings(df)
        logger.info("Lecturas normales generadas")

        # Paso 4: Variables ambientales
        df = self._add_environmental(df)

        # Paso 5: Inyectar fallas con firmas pre-falla
        df = self._inject_faults(df)
        n_faults = df["is_fault"].sum()
        logger.info("Fallas inyectadas: %d registros en falla", n_faults)

        # Paso 6: Ruido de medicion
        df = self._add_measurement_noise(df)

        logger.info("=== Generacion completada: %d filas, %d columnas ===", len(df), len(df.columns))
        return df

    def _create_base_structure(self, segment_props: pd.DataFrame) -> pd.DataFrame:
        """Crea DataFrame base con timestamp x segment_id y propiedades del segmento."""
        start_date = datetime(2024, 1, 1)
        timestamps = pd.date_range(start=start_date, periods=self.n_timestamps, freq=f"{self.interval_min}min")
        segment_ids = segment_props["segment_id"].tolist()

        idx = pd.MultiIndex.from_product([timestamps, segment_ids], names=["timestamp", "segment_id"])
        df = pd.DataFrame(index=idx).reset_index()

        # Merge propiedades del segmento
        df = df.merge(segment_props, on="segment_id", how="left")

        df["fault_type"] = "normal"
        df["is_fault"] = 0
        df["pre_fault_hours"] = 0.0
        return df

    def _add_normal_readings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera lecturas normales basadas en las propiedades fisicas de cada segmento.

        La potencia optica no es un valor aleatorio: depende de la longitud
        del segmento, sus empalmes y conectores. Cada segmento tiene un
        "base_power_dbm" calculado en _generate_segment_properties.

        Las variaciones normales vienen de:
        - Patron diurno: temperatura sube de dia -> fibra se expande ligeramente
          -> atenuacion cambia ~0.002 dB/km/C (efecto real documentado).
        - Ruido del receptor: el APD tiene ruido termico y shot noise.
        """
        n = len(df)
        ranges = self.normal_ranges
        hour_of_day = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

        # Patron diurno: efecto real de temperatura en la fibra
        # La atenuacion de la fibra cambia ~0.001 dB/km por grado C
        diurnal = np.sin(2 * np.pi * hour_of_day / 24.0)

        # --- Potencia optica (dBm) ---
        # Basada en la potencia calculada por link budget de cada segmento
        # Variacion diurna: ~0.5 dB por expansion termica de la fibra
        df["optical_power_dbm"] = (
            df["base_power_dbm"]
            + 0.5 * diurnal                     # Expansion termica
            + self.rng.normal(0, 0.3, n)        # Ruido del receptor APD
        )

        # --- Atenuacion (dB/km) ---
        # Fuente: ITU-T G.652 - tipico 0.20 dB/km @ 1550nm
        center_atten = (ranges["attenuation_db_km"][0] + ranges["attenuation_db_km"][1]) / 2
        df["attenuation_db_km"] = (
            center_atten
            + 0.003 * diurnal                   # Efecto termico en la fibra
            + self.rng.normal(0, 0.008, n)      # Variabilidad de medicion
        )

        # --- BER ---
        # Fuente: ITU-T G.984.2 - objetivo post-FEC: <1e-12
        # BER depende fuertemente de la potencia recibida:
        # menor potencia = peor BER (mas errores)
        # Modelamos esta correlacion: segmentos mas largos tienen peor BER
        power_factor = (df["optical_power_dbm"] - (-8)) / ((-28) - (-8))  # 0 (bueno) a 1 (malo)
        log_ber_base = -11.0 + power_factor * 2.0  # -11 a -9 en escala log
        df["ber"] = 10 ** (
            log_ber_base
            + 0.2 * diurnal
            + self.rng.normal(0, 0.3, n)
        )

        # --- OSNR (dB) ---
        # Fuente: OPM papers - rango tipico 20-35 dB en GPON
        # OSNR correlaciona positivamente con potencia optica
        osnr_base = 27.5 - power_factor * 10.0  # 27.5 (bueno) a 17.5 (malo)
        df["osnr_db"] = (
            osnr_base
            - 0.5 * diurnal                     # Mas trafico de dia -> mas crosstalk
            + self.rng.normal(0, 1.0, n)
        )

        # --- Dispersion cromatica (ps/(nm*km)) ---
        # Fuente: ITU-T G.652 - coeficiente tipico ~17 ps/(nm*km) @ 1550nm
        cd_range = ranges["chromatic_dispersion_ps_nm_km"]
        center_cd = (cd_range[0] + cd_range[1]) / 2
        df["chromatic_dispersion"] = (
            center_cd
            + 0.1 * diurnal                     # Efecto termico
            + self.rng.normal(0, 0.3, n)
        )

        return df

    def _add_environmental(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Variables ambientales basadas en clima de Lima, Peru.

        Fuente: SENAMHI (Servicio Nacional de Meteorologia e Hidrologia del Peru)
        - Lima: temperatura media anual 18-22 C, humedad media 80-85%
        - En verano (ene-mar): 22-28 C, humedad 65-75%
        - En invierno (jun-ago): 14-18 C, humedad 85-95%

        La temperatura afecta la fibra optica:
        - Expansion termica cambia la atenuacion (~0.001 dB/km/C)
        - La humedad puede degradar conectores expuestos
        """
        n = len(df)
        hour_of_day = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

        # Temperatura de Lima (simulacion enero = verano)
        df["temperature"] = (
            23.0
            + 4.0 * np.sin(2 * np.pi * (hour_of_day - 14) / 24.0)  # Pico ~2pm
            + self.rng.normal(0, 0.8, n)
        )

        # Humedad (inversa a temperatura)
        df["humidity"] = (
            78.0
            - 8.0 * np.sin(2 * np.pi * (hour_of_day - 14) / 24.0)
            + self.rng.normal(0, 2.5, n)
        )
        df["humidity"] = df["humidity"].clip(40, 100)

        return df

    def _inject_faults(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inyecta fallas con firmas pre-falla fisicamente fundamentadas.

        Selecciona ~30% de los segmentos para tener una falla.
        Cada falla tiene un periodo pre-falla (12-72h) donde las metricas
        se degradan gradualmente - esta es la senal que el modelo debe aprender.
        """
        segments = df["segment_id"].unique()
        n_faulty = int(len(segments) * 0.3)
        faulty_segments = self.rng.choice(segments, size=n_faulty, replace=False)

        fault_types = self.rng.choice(
            ["physical_cut", "degradation", "bad_splice"],
            size=n_faulty,
            p=[self.fault_dist["physical_cut"], self.fault_dist["degradation"], self.fault_dist["bad_splice"]],
        )

        logger.info("Inyectando fallas en %d de %d segmentos", n_faulty, len(segments))

        for segment, fault_type in zip(faulty_segments, fault_types):
            mask = df["segment_id"] == segment
            seg_indices = df.index[mask]
            n_readings = len(seg_indices)

            fault_idx_local = self.rng.integers(int(n_readings * 0.4), int(n_readings * 0.8))

            pre_fault_hours = self.rng.uniform(self.pre_fault["min_hours"], self.pre_fault["max_hours"])
            pre_fault_steps = int(pre_fault_hours * 60 / self.interval_min)

            # --- Firma PRE-FALLA ---
            pre_start = max(0, fault_idx_local - pre_fault_steps)
            pre_indices = seg_indices[pre_start:fault_idx_local]

            if len(pre_indices) > 0:
                progress = np.linspace(0, 1, len(pre_indices))
                self._apply_pre_fault_signature(df, pre_indices, fault_type, progress)
                hours_to_fault = np.linspace(pre_fault_hours, 0, len(pre_indices))
                df.loc[pre_indices, "pre_fault_hours"] = hours_to_fault

            # --- FALLA ACTIVA ---
            fault_indices = seg_indices[fault_idx_local:]
            df.loc[fault_indices, "fault_type"] = fault_type
            df.loc[fault_indices, "is_fault"] = 1
            self._apply_fault(df, fault_indices, fault_type)

        return df

    def _apply_pre_fault_signature(self, df, indices, fault_type, progress):
        """
        Firma pre-falla basada en fenomenos fisicos reales.

        Cada tipo de falla tiene precursores distintos documentados
        en la literatura de fiber optic monitoring:

        Corte fisico: Antes de un corte total, la fibra sufre
        microcurvaturas por estres mecanico (ej: excavacion cercana).
        Las microcurvaturas causan perdida de 0.5-3 dB gradual.
        Ref: OTDR fault diagnosis, FOA technical bulletins.

        Degradacion: El envejecimiento de la fibra o la intrusion
        de humedad causan un aumento lento de la atenuacion.
        La fibra pasa de 0.20 dB/km a 0.35+ dB/km progresivamente.
        Ref: ITU-T G.652 aging specifications.

        Empalme defectuoso: Un empalme con fusion incompleta es
        sensible a la temperatura. Al expandirse/contraerse genera
        oscilaciones periodicas en la potencia (+/- 0.5-2 dB).
        Ref: FOA splice loss characterization.
        """
        n = len(indices)
        fv = self.fault_values

        if fault_type == "physical_cut":
            # Microcurvaturas: caida gradual de potencia 0-3 dB
            df.loc[indices, "optical_power_dbm"] -= progress * 3.0
            df.loc[indices, "osnr_db"] -= progress * 3.0
            df.loc[indices, "ber"] *= (1 + progress * 10)
            df.loc[indices, "attenuation_db_km"] += progress * 0.03

        elif fault_type == "degradation":
            # Envejecimiento: todas las metricas empeoran progresivamente
            power_drop = fv["degradation"]["optical_power_drop_db"]
            atten_increase = fv["degradation"]["attenuation_increase"]
            osnr_drop = fv["degradation"]["osnr_drop_db"]
            df.loc[indices, "optical_power_dbm"] -= progress * power_drop * 0.5
            df.loc[indices, "attenuation_db_km"] += progress * atten_increase * 0.5
            df.loc[indices, "osnr_db"] -= progress * osnr_drop * 0.5
            df.loc[indices, "ber"] *= (1 + progress * 50)
            df.loc[indices, "chromatic_dispersion"] += progress * 0.5

        elif fault_type == "bad_splice":
            # Empalme inestable: oscilaciones termicas
            osc_amp = fv["bad_splice"]["oscillation_amplitude_db"]
            oscillation = np.sin(np.linspace(0, 8 * np.pi, n)) * progress
            df.loc[indices, "optical_power_dbm"] -= progress * 1.0 + osc_amp * 0.3 * oscillation
            df.loc[indices, "attenuation_db_km"] += progress * 0.05
            df.loc[indices, "osnr_db"] -= progress * 2.0

    def _apply_fault(self, df, indices, fault_type):
        """
        Aplica valores de falla activa basados en estandares.

        Los valores aqui representan lo que mediria un OTDR o el
        sistema NMS cuando la falla ya ocurrio.
        """
        n = len(indices)
        fv = self.fault_values

        if fault_type == "physical_cut":
            # Corte total: senal perdida
            # Fuente: En un corte real, la reflectancia es muy alta (pico en OTDR)
            # y la potencia despues del corte cae a nivel de ruido (-40 dBm o menos)
            cut_power = fv["physical_cut"]["optical_power_dbm"]
            df.loc[indices, "optical_power_dbm"] = cut_power + self.rng.normal(0, 1.5, n)
            df.loc[indices, "attenuation_db_km"] = 5.0 + self.rng.normal(0, 0.5, n)
            df.loc[indices, "ber"] = fv["physical_cut"]["ber"] + self.rng.normal(0, 0.1, n)
            df.loc[indices, "ber"] = df.loc[indices, "ber"].clip(0.01, 1.0)
            df.loc[indices, "osnr_db"] = fv["physical_cut"]["osnr_db"] + self.rng.normal(0, 1, n)
            df.loc[indices, "osnr_db"] = df.loc[indices, "osnr_db"].clip(lower=0)

        elif fault_type == "degradation":
            # Degradacion continua: metricas empeoran con el tiempo
            progress = np.linspace(0, 1, n)
            power_drop = fv["degradation"]["optical_power_drop_db"]
            atten_inc = fv["degradation"]["attenuation_increase"]
            osnr_drop = fv["degradation"]["osnr_drop_db"]
            ber_mult = fv["degradation"]["ber_multiplier"]

            df.loc[indices, "optical_power_dbm"] -= power_drop * 0.5 + progress * power_drop * 0.5
            df.loc[indices, "attenuation_db_km"] += atten_inc * 0.5 + progress * atten_inc * 0.5
            df.loc[indices, "ber"] *= ber_mult ** (0.5 + progress * 0.5)
            df.loc[indices, "osnr_db"] -= osnr_drop * 0.5 + progress * osnr_drop * 0.5
            df.loc[indices, "chromatic_dispersion"] += 0.5 + progress * 1.5

        elif fault_type == "bad_splice":
            # Empalme defectuoso: intermitencia severa
            # Fuente: FOA - un empalme malo puede perder 0.5-3 dB
            # con oscilaciones de +/- 2 dB por expansion termica
            osc_amp = fv["bad_splice"]["oscillation_amplitude_db"]
            splice_loss_range = fv["bad_splice"]["splice_loss_db"]
            base_loss = self.rng.uniform(splice_loss_range[0], splice_loss_range[1])

            oscillation = np.sin(np.linspace(0, 20 * np.pi, n))
            df.loc[indices, "optical_power_dbm"] -= base_loss + osc_amp * np.abs(oscillation)
            df.loc[indices, "attenuation_db_km"] += 0.05 + 0.03 * np.abs(oscillation)
            df.loc[indices, "ber"] *= 10 ** (0.5 + np.abs(oscillation))
            df.loc[indices, "osnr_db"] -= 3.0 + 2.0 * np.abs(oscillation)

    def _add_measurement_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega ruido de medicion realista.

        Todo instrumento de medicion tiene un margen de error:
        - OTDR: precision tipica +/- 0.05 dB (EXFO FTB-7200E specs)
        - Medidor de potencia: precision +/- 0.2 dB
        - BER tester: variacion estadistica inherente
        """
        n = len(df)

        # Potencia: +/- 0.2 dB (precision del medidor)
        df["optical_power_dbm"] += self.rng.normal(0, 0.15, n)

        # Atenuacion: +/- 0.01 dB/km
        df["attenuation_db_km"] += self.rng.normal(0, 0.005, n)
        df["attenuation_db_km"] = df["attenuation_db_km"].clip(lower=0.01)

        # BER: variacion multiplicativa (naturaleza estadistica del BER)
        df["ber"] *= 10 ** self.rng.normal(0, 0.08, n)
        df["ber"] = df["ber"].clip(lower=1e-15, upper=1.0)

        # OSNR: +/- 0.5 dB
        df["osnr_db"] += self.rng.normal(0, 0.3, n)
        df["osnr_db"] = df["osnr_db"].clip(lower=0)

        # CD: +/- 0.2 ps/(nm*km)
        df["chromatic_dispersion"] += self.rng.normal(0, 0.15, n)

        return df

    def save(self, df: pd.DataFrame, filename: str = "fiber_readings.parquet") -> str:
        """Guarda el DataFrame en formato Parquet."""
        output_path = get_path("raw_data") / filename
        df.to_parquet(output_path, index=False, engine="pyarrow")
        size_mb = output_path.stat().st_size / 1e6
        logger.info("Datos guardados: %s (%.2f MB)", output_path, size_mb)
        return str(output_path)

    def generate_and_save(self) -> Tuple[pd.DataFrame, str]:
        """Genera datos y los guarda. Retorna (DataFrame, ruta_archivo)."""
        df = self.generate()
        path = self.save(df)
        return df, path
