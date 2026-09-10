import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_replace, to_date, coalesce, trim, upper, when,
    sum as _sum, round as _round, date_format
)

# ---------------------------------------------------------------------------
# CHANGELOG v2:
#   - Silver y Gold ya no encadenan withColumn(): ahora usan un solo select()
#     por capa, lo que reduce el número de nodos "Project" en el plan lógico
#     de Spark y baja el overhead de análisis de Catalyst.
#   - Se agregó validar_calidad_gold(): reglas de Data Quality sobre la salida
#     final, con severidad diferenciada (críticas detienen el pipeline,
#     advertencias solo se registran).
#   - Se cachea df_gold_final antes de correr las reglas de calidad, porque
#     la validación hace varias acciones (count/filter) sobre el mismo
#     DataFrame; sin cache, Spark recalcularía todo el pipeline en cada una.
# ---------------------------------------------------------------------------

COLUMNAS_BRONZE_REQUERIDAS = {
    "id_transaccion", "id_usuario", "monto_str",
    "fecha_operacion", "categoria_raw", "tipo_movimiento"
}

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

    OPTIMIZACIÓN: todas las columnas limpias se calculan en un solo select(),
    en vez de encadenar varios withColumn(). Cada columna nueva depende
    únicamente de columnas originales de Bronze (no una de otra), así que
    pueden resolverse todas en una sola proyección.
    """
    print("🧹 Iniciando procesamiento de Capa Silver...")
    registros_bronze = df_bronze.count()

    # 1. Deduplicación por llave de negocio (evita reprocesos duplicando transacciones)
    df_bronze = df_bronze.dropDuplicates(["id_transaccion"])

    # Expresión de fecha con el mes en español ya normalizado a inglés,
    # para el patrón dd-MMM-yyyy (ej. "15-Ene-2026")
    fecha_normalizada = _normalizar_mes_espanol(upper(col("fecha_operacion")))

    # 2. Una sola proyección con las 4 columnas limpias + las 2 llaves
    df_silver_final = df_bronze.select(
        col("id_transaccion"),
        col("id_usuario"),

        # Monto: quita $, comas y "MXN", castea a double
        regexp_replace(
            regexp_replace(col("monto_str"), r"[\$,]", ""),
            r"\s*MXN\s*", ""
        ).cast("double").alias("monto"),

        # Fecha: homologa 5 formatos distintos de origen
        coalesce(
            to_date(col("fecha_operacion"), "yyyy-MM-dd"),
            to_date(col("fecha_operacion"), "dd/MM/yyyy"),
            to_date(col("fecha_operacion"), "MM-dd-yyyy"),
            to_date(fecha_normalizada, "dd-MMM-yyyy"),
            to_date(col("fecha_operacion"), "yyyy/MM/dd"),
        ).alias("fecha"),

        # Categoría: remueve símbolos raros conservando acentos válidos
        trim(regexp_replace(col("categoria_raw"), r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ ]", "")).alias("categoria"),

        # Tipo de transacción: normaliza sinónimos a INGRESO/EGRESO
        when(upper(col("tipo_movimiento")).isin("ABONO", "INGRESO", "CREDITO", "CR"), "INGRESO")
        .when(upper(col("tipo_movimiento")).isin("CARGO", "EGRESO", "DEBITO", "DB"), "EGRESO")
        .otherwise(None).alias("tipo_transaccion")

    ).na.drop(subset=["monto", "fecha", "tipo_transaccion"])

    registros_limpios = df_silver_final.count()
    descartados = registros_bronze - registros_limpios
    if descartados > 0:
        print(f"⚠️ Registros descartados en Silver (duplicados o monto/fecha/tipo inválidos): {descartados}")

    df_silver_final.write.mode("overwrite").parquet(ruta_salida_parquet)
    print(f"✅ Capa Silver procesada y exportada en Parquet. Registros limpios: {registros_limpios}")
    return df_silver_final


def validar_calidad_gold(df_gold):
    """
    Reglas de calidad de datos (Data Quality) sobre la Capa Gold, antes de
    publicarla. Se dividen en dos niveles de severidad:

      - CRÍTICAS: si fallan, se detiene el pipeline (raise). Indican un bug
        en el pipeline o datos de origen gravemente corruptos.
      - ADVERTENCIAS: se registran en consola pero no detienen nada. Son
        casos de negocio legítimos que solo requieren revisión humana.
    """
    print("🔍 Ejecutando reglas de calidad sobre Capa Gold...")
    total_filas = df_gold.count()
    errores_criticos = []
    advertencias = []

    # Regla 1 (CRÍTICA) — llaves de negocio no pueden ser nulas
    nulos_llave = df_gold.filter(col("id_usuario").isNull() | col("mes").isNull()).count()
    if nulos_llave > 0:
        errores_criticos.append(f"{nulos_llave} fila(s) con id_usuario o mes nulo")

    # Regla 2 (CRÍTICA) — no debe haber filas duplicadas por (id_usuario, mes)
    duplicados = total_filas - df_gold.dropDuplicates(["id_usuario", "mes"]).count()
    if duplicados > 0:
        errores_criticos.append(f"{duplicados} fila(s) duplicada(s) en (id_usuario, mes)")

    # Regla 3 (CRÍTICA) — ingresos/egresos negativos indicarían un bug de Silver
    montos_negativos = df_gold.filter((col("total_ingresos") < 0) | (col("total_egresos") < 0)).count()
    if montos_negativos > 0:
        errores_criticos.append(f"{montos_negativos} fila(s) con ingresos o egresos negativos")

    # Regla 4 (ADVERTENCIA) — ratio de endeudamiento anormalmente alto
    ratio_extremo = df_gold.filter(col("ratio_endeudamiento") > 300).count()
    if ratio_extremo > 0:
        advertencias.append(f"{ratio_extremo} usuario(s)-mes con ratio_endeudamiento > 300% (revisar manualmente)")

    # Regla 5 (ADVERTENCIA) — usuarios sin ningún ingreso registrado en el mes
    sin_ingresos = df_gold.filter(col("total_ingresos") == 0).count()
    if sin_ingresos > 0:
        advertencias.append(f"{sin_ingresos} fila(s) con total_ingresos en 0")

    for a in advertencias:
        print(f"   ⚠️ ADVERTENCIA: {a}")

    if errores_criticos:
        for e in errores_criticos:
            print(f"   ❌ CRÍTICO: {e}")
        raise ValueError("La Capa Gold no pasó las reglas de calidad críticas. Pipeline detenido antes de publicar.")

    print(f"✅ Calidad verificada sobre {total_filas} fila(s). {len(advertencias)} advertencia(s), 0 errores críticos.")


def procesar_capa_gold(df_silver, ruta_salida_parquet, granularidad_mensual=True):
    """
    Capa Gold (Master): Agregaciones dimensionales analíticas por cliente.
    Consolida variables clave de negocio, las valida con reglas de calidad,
    y las deja listas para alimentar al Asistente de IA (Text-to-SQL).
    """
    print("🥇 Consolidando variables analíticas en la Capa Gold...")

    llaves_agrupacion = ["id_usuario"]
    if granularidad_mensual:
        # select("*", ...) agrega la columna "mes" sin encadenar withColumn
        df_silver = df_silver.select("*", date_format(col("fecha"), "yyyy-MM").alias("mes"))
        llaves_agrupacion.append("mes")

    df_kpis = df_silver.groupBy(*llaves_agrupacion).agg(
        _round(_sum(when(col("tipo_transaccion") == "INGRESO", col("monto")).otherwise(0)), 2).alias("total_ingresos"),
        _round(_sum(when(col("tipo_transaccion") == "EGRESO", col("monto")).otherwise(0)), 2).alias("total_egresos")
    )

    # Ambas métricas derivadas en una sola proyección (no dos withColumn encadenados).
    # Ninguna depende de la otra: las dos se calculan solo a partir de
    # total_ingresos/total_egresos, así que es seguro resolverlas juntas.
    df_gold_final = df_kpis.select(
        "*",
        _round(col("total_ingresos") - col("total_egresos"), 2).alias("capacidad_ahorro"),
        when(col("total_ingresos") > 0,
             _round((col("total_egresos") / col("total_ingresos")) * 100, 2))
        .otherwise(100.0).alias("ratio_endeudamiento")
    )

    # Cacheamos porque validar_calidad_gold() va a correr varias acciones
    # (count/filter) sobre este mismo DataFrame; sin cache, Spark repetiría
    # todo el cálculo de Bronze+Silver+Gold en cada una.
    df_gold_final = df_gold_final.cache()

    validar_calidad_gold(df_gold_final)

    df_gold_final.write.mode("overwrite").parquet(ruta_salida_parquet)
    print("✅ Capa Gold generada exitosamente en formato Parquet.")
    df_gold_final.show(5)

    df_gold_final.unpersist()
    return df_gold_final


if __name__ == "__main__":
    path_bronze_csv = "v2/transacciones_bronze.csv"
    path_silver_parquet = "v2/data/silver/t_mx_clean_financial_transactions.parquet"
    path_gold_parquet = "v2/data/gold/t_mx_master_financial_health.parquet"

    spark_session = crear_sesion_spark()

    try:
        df_b = procesar_capa_bronze(spark_session, path_bronze_csv)
        df_s = procesar_capa_silver(df_b, path_silver_parquet)
        df_g = procesar_capa_gold(df_s, path_gold_parquet, granularidad_mensual=True)

    finally:
        spark_session.stop()
        print("🔌 Sesión de Spark cerrada de forma segura.")
