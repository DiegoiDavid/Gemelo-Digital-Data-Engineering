import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_replace, to_date, coalesce, trim, upper, when,
    sum as _sum, round as _round, date_format
)

# ---------------------------------------------------------------------------
# Columnas esperadas en Bronze (evita "magic strings" repetidos y valida
# el esquema de entrada antes de procesar).
# ---------------------------------------------------------------------------
COLUMNAS_BRONZE_REQUERIDAS = {
    "id_transaccion", "id_usuario", "monto_str",
    "fecha_operacion", "categoria_raw", "tipo_movimiento"
}

# Meses en español -> abreviatura en inglés, para parsear fechas tipo
# "15-Ene-2026" sin depender del locale del JVM/cluster donde corra Spark.
MESES_ES_A_EN = {
    "ENE": "JAN", "FEB": "FEB", "MAR": "MAR", "ABR": "APR",
    "MAY": "MAY", "JUN": "JUN", "JUL": "JUL", "AGO": "AUG",
    "SEP": "SEP", "OCT": "OCT", "NOV": "NOV", "DIC": "DEC",
}


def crear_sesion_spark():
    """Inicializa la sesión de Spark configurando el entorno local de forma óptima."""
    spark = (
        SparkSession.builder
        .appName("BBVA_Financial_Digital_Twin_Medallion")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def procesar_capa_bronze(spark, ruta_origen):
    """
    Capa Bronze (Staging): Ingesta inmutable de transacciones financieras
    preservando su formato crudo con caracteres especiales e inconsistencias de origen.
    """
    print("📥 Cargando Capa Bronze...")
    if not os.path.exists(ruta_origen):
        raise FileNotFoundError(f"No se encontró el archivo de origen en: {ruta_origen}")

    df_bronze = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(ruta_origen)
    )

    # Validación temprana de esquema: falla rápido y con mensaje claro en vez
    # de un AnalysisException críptico más adelante en Silver.
    faltantes = COLUMNAS_BRONZE_REQUERIDAS - set(df_bronze.columns)
    if faltantes:
        raise ValueError(
            f"❌ El CSV de origen no tiene las columnas esperadas. Faltan: {faltantes}"
        )

    print(f"✅ Capa Bronze completada. Registros leídos: {df_bronze.count()}")
    return df_bronze


def _normalizar_mes_espanol(columna):
    """Reemplaza abreviaturas de mes en español por su equivalente en inglés
    para que to_date() las pueda parsear sin depender del locale del sistema."""
    columna_normalizada = columna
    for es, en in MESES_ES_A_EN.items():
        columna_normalizada = regexp_replace(columna_normalizada, f"(?i){es}", en)
    return columna_normalizada


def procesar_capa_silver(df_bronze, ruta_salida_parquet):
    """
    Capa Silver (Cleansing): Tratamiento y tipado estricto de datos con PySpark.
    Limpia monedas, formatea fechas redundantes y estandariza categorías.
    """
    print("🧹 Iniciando procesamiento de Capa Silver...")
    registros_bronze = df_bronze.count()

    # 1. Deduplicación por llave de negocio (evita reprocesos duplicando transacciones)
    df_bronze = df_bronze.dropDuplicates(["id_transaccion"])

    # 2. Limpieza de monto (quitar $, comas, espacios y letras "MXN", casteo a double)
    df_silver = df_bronze.withColumn(
        "monto",
        regexp_replace(
            regexp_replace(col("monto_str"), r"[\$,]", ""),
            r"\s*MXN\s*", ""
        ).cast("double")
    )

    # 3. Homologación de múltiples formatos de fecha en origen.
    #    Normalizamos el mes en texto (Ene, Feb, ...) antes de parsear con
    #    dd-MMM-yyyy para no depender del locale del cluster.
    fecha_normalizada = _normalizar_mes_espanol(upper(col("fecha_operacion")))
    df_silver = df_silver.withColumn(
        "fecha",
        coalesce(
            to_date(col("fecha_operacion"), "yyyy-MM-dd"),
            to_date(col("fecha_operacion"), "dd/MM/yyyy"),
            to_date(col("fecha_operacion"), "MM-dd-yyyy"),
            to_date(fecha_normalizada, "dd-MMM-yyyy"),
            to_date(col("fecha_operacion"), "yyyy/MM/dd"),
        )
    )

    # 4. Limpieza de categorías (remover caracteres especiales o acentos corruptos)
    df_silver = df_silver.withColumn(
        "categoria",
        trim(regexp_replace(col("categoria_raw"), r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ ]", ""))
    )

    # 5. Normalización del tipo de movimiento financiero
    df_silver = df_silver.withColumn(
        "tipo_transaccion",
        when(upper(col("tipo_movimiento")).isin("ABONO", "INGRESO", "CREDITO", "CR"), "INGRESO")
        .when(upper(col("tipo_movimiento")).isin("CARGO", "EGRESO", "DEBITO", "DB"), "EGRESO")
        .otherwise(None)
    )

    # 6. Selección de columnas limpias y remoción de registros corruptos críticos
    df_silver_final = df_silver.select(
        col("id_transaccion"),
        col("id_usuario"),
        col("monto"),
        col("fecha"),
        col("categoria"),
        col("tipo_transaccion")
    ).na.drop(subset=["monto", "fecha", "tipo_transaccion"])

    # 7. Auditoría: cuántos registros se perdieron y por qué causa probable
    #    (visibilidad clave para debugging en producción).
    registros_limpios = df_silver_final.count()
    descartados = registros_bronze - registros_limpios
    if descartados > 0:
        print(f"⚠️ Registros descartados en Silver (duplicados o monto/fecha/tipo inválidos): {descartados}")

    df_silver_final.write.mode("overwrite").parquet(ruta_salida_parquet)
    print(f"✅ Capa Silver procesada y exportada en Parquet. Registros limpios: {registros_limpios}")
    return df_silver_final


def procesar_capa_gold(df_silver, ruta_salida_parquet, granularidad_mensual=True):
    """
    Capa Gold (Master): Agregaciones dimensionales analíticas por cliente.
    Consolida variables clave de negocio listas para alimentar al Asistente de IA (Text-to-SQL).

    granularidad_mensual:
        True  -> agrupa por id_usuario + mes (yyyy-MM). Úsalo si el CSV trae
                 transacciones de varios meses y quieres KPIs mes a mes.
        False -> agrupa solo por id_usuario, sumando el histórico completo.
    """
    print("🥇 Consolidando variables analíticas en la Capa Gold...")

    llaves_agrupacion = ["id_usuario"]
    if granularidad_mensual:
        df_silver = df_silver.withColumn("mes", date_format(col("fecha"), "yyyy-MM"))
        llaves_agrupacion.append("mes")

    df_kpis = df_silver.groupBy(*llaves_agrupacion).agg(
        _round(_sum(when(col("tipo_transaccion") == "INGRESO", col("monto")).otherwise(0)), 2).alias("total_ingresos"),
        _round(_sum(when(col("tipo_transaccion") == "EGRESO", col("monto")).otherwise(0)), 2).alias("total_egresos")
    )

    # Cálculo de métricas de negocio derivadas (Capacidad de Ahorro y % de Endeudamiento)
    df_gold_final = df_kpis.withColumn(
        "capacidad_ahorro",
        _round(col("total_ingresos") - col("total_egresos"), 2)
    ).withColumn(
        "ratio_endeudamiento",
        when(col("total_ingresos") > 0,
             _round((col("total_egresos") / col("total_ingresos")) * 100, 2))
        .otherwise(100.0)
    )

    df_gold_final.write.mode("overwrite").parquet(ruta_salida_parquet)

    print("✅ Capa Gold generada exitosamente en formato Parquet.")
    df_gold_final.show(5)
    return df_gold_final


if __name__ == "__main__":
    # Definición de rutas consistentes de acuerdo a la estructura de delegación
    path_bronze_csv = "./data/bronze/transacciones_bronze.csv"
    path_silver_parquet = "./data/silver/t_mx_clean_financial_transactions.parquet"
    path_gold_parquet = "./data/gold/t_mx_master_financial_health.parquet"

    spark_session = crear_sesion_spark()

    try:
        df_b = procesar_capa_bronze(spark_session, path_bronze_csv)
        df_s = procesar_capa_silver(df_b, path_silver_parquet)
        # Cambia a granularidad_mensual=False si prefieres el total histórico por usuario
        df_g = procesar_capa_gold(df_s, path_gold_parquet, granularidad_mensual=True)

    finally:
        # Cerrar el contexto de Spark de manera segura
        spark_session.stop()
        print("🔌 Sesión de Spark cerrada de forma segura.")
