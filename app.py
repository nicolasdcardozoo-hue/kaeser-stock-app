import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from io import BytesIO

st.set_page_config(page_title="Kaeser - IA & Dashboard", layout="wide")

st.title("⚙️ Sistema Integral de Abastecimiento - Kaeser Medellín")
st.markdown("Plataforma de optimización de Stock de Seguridad con IA, basada en el Maestro de Materiales y reglas logísticas corporativas.")

if 'df_procesado' not in st.session_state:
    st.session_state.df_procesado = None

# 1. Zona de carga de archivos (Arquitectura Limpia v4)
with st.expander("📂 Haz clic aquí para subir los archivos fuente", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Datos de SAP (Operativos)**")
        file_vl06o = st.file_uploader("1. Archivo VL06O (Movimientos)", type=['xlsx', 'csv'])
        file_zmd04 = st.file_uploader("2. Archivo ZMD04 (Stock Tenjo y Status)", type=['xlsx', 'csv'])
        file_rmm = st.file_uploader("3. Archivo RMMDMDMA (Lotes y Stock Actual)", type=['xlsx', 'csv'])
        file_mcbe = st.file_uploader("4. Archivo MCBE (Cantidades y Valor)", type=['xlsx', 'csv'])
    with col2:
        st.markdown("**Diccionarios (Maestros)**")
        file_diccionario = st.file_uploader("5. Diccionario General (Tipo, Procedencia)", type=['xlsx', 'csv'])
        file_finanzas = st.file_uploader("6. Reporte Finanzas (Clasificación ABC)", type=['xlsx', 'csv'])

st.markdown("---")

# Función auxiliar para leer excel o csv
def leer_archivo(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    return pd.read_excel(file)

if st.button("🚀 Procesar Datos y Generar Dashboard", type="primary"):
    if file_vl06o and file_zmd04 and file_rmm and file_mcbe and file_diccionario and file_finanzas:
        with st.spinner("Construyendo el modelo de datos y ejecutando IA..."):
            try:
                # --- PASO 1: MCBE (Base principal) ---
                df_mcbe = leer_archivo(file_mcbe)
                col_mat_mcbe = 'P/N' if 'P/N' in df_mcbe.columns else 'Material'
                df_base = df_mcbe[[col_mat_mcbe, 'CtdStkTot.', 'ValStkVal']].copy()
                df_base.columns = ['codigo_material', 'stock_actual_3420', 'valor_total']
                df_base = df_base.dropna(subset=['codigo_material'])
                df_base['codigo_material'] = df_base['codigo_material'].astype(str).str.strip()
                df_base = df_base.drop_duplicates(subset=['codigo_material'])
                
                df_base['stock_actual_3420'] = df_base['stock_actual_3420'].fillna(0)
                df_base['costo_unitario'] = np.where(df_base['stock_actual_3420'] > 0, df_base['valor_total'] / df_base['stock_actual_3420'], 0)
                df_base = df_base.drop(columns=['valor_total'])

                # --- PASO 2: RMMDMDMA (Lotes y Stock Seguridad Actual Medellín) ---
                df_rmm = leer_archivo(file_rmm)
                df_lotes = df_rmm[['Material', 'Valor de redondeo', 'Tamaño lote mínimo', 'Stock de seguridad']].copy()
                df_lotes.columns = ['codigo_material', 'valor_redondeo', 'lote_minimo', 'stock_seguridad_actual_3420']
                df_lotes['codigo_material'] = df_lotes['codigo_material'].astype(str).str.strip()
                
                df_cruce = pd.merge(df_base, df_lotes, on='codigo_material', how='left').fillna(0)

                # --- PASO 3: ZMD04 (Tenjo y Status Alemania) ---
                df_zmd04 = leer_archivo(file_zmd04)
                col_status = [c for c in df_zmd04.columns if 'STATUS' in str(c).upper()][0]
                df_tenjo = df_zmd04[['Material', 'Stock de seguridad', col_status]].copy()
                df_tenjo.columns = ['codigo_material', 'stock_seguridad_3400', 'status_alemania']
                df_tenjo['codigo_material'] = df_tenjo['codigo_material'].astype(str).str.strip()
                df_tenjo['status_alemania'] = df_tenjo['status_alemania'].fillna('')
                
                df_cruce = pd.merge(df_cruce, df_tenjo, on='codigo_material', how='left')
                df_cruce['stock_seguridad_3400'] = df_cruce['stock_seguridad_3400'].fillna(0)
                df_cruce['status_alemania'] = df_cruce['status_alemania'].fillna('')

                def calcular_lead_time(fila):
                    if fila['stock_seguridad_3400'] > 0: return 7
                    elif fila['stock_seguridad_3400'] == 0 and fila['stock_seguridad_actual_3420'] > 0: return 62
                    else: return 55
                df_cruce['lead_time_dias'] = df_cruce.apply(calcular_lead_time, axis=1)

                # --- PASO 4: DICCIONARIO LOGÍSTICA ---
                df_dicc = leer_archivo(file_diccionario)
                df_dicc = df_dicc[['Material', 'Descripción', 'Tipo de Mercancia', 'Tipo de Material', 'Procedencia']].copy()
                df_dicc.columns = ['codigo_material', 'descripcion', 'tipo_mercancia', 'tipo_material', 'procedencia']
                df_dicc['codigo_material'] = df_dicc['codigo_material'].astype(str).str.strip()
                
                df_cruce = pd.merge(df_cruce, df_dicc, on='codigo_material', how='left')
                df_cruce['tipo_mercancia'] = df_cruce['tipo_mercancia'].fillna('Desconocido')
                df_cruce['tipo_material'] = df_cruce['tipo_material'].fillna('Otros')
                df_cruce['procedencia'] = df_cruce['procedencia'].fillna('Desconocido')
                df_cruce['descripcion'] = df_cruce['descripcion'].fillna('Sin descripción')

                # Filtro: Solo analizar Repuestos
                df_cruce = df_cruce[df_cruce['tipo_mercancia'].str.upper().str.contains('REPUESTO')].copy()

                # --- PASO 5: REPORTE FINANZAS (ABC) CORREGIDO ---
                df_fin = leer_archivo(file_finanzas)
                
                # Buscamos la columna inteligentemente
                col_mat_fin = 'Partenúmero' if 'Partenúmero' in df_fin.columns else ('P/N' if 'P/N' in df_fin.columns else 'Material')
                
                df_fin = df_fin[[col_mat_fin, 'ABC']].copy()
                df_fin.columns = ['codigo_material', 'clasificacion_abc']
                df_fin['codigo_material'] = df_fin['codigo_material'].astype(str).str.strip()
                df_fin = df_fin.drop_duplicates(subset=['codigo_material']) # Por si acaso Finanzas trae repetidos
                
                df_cruce = pd.merge(df_cruce, df_fin, on='codigo_material', how='left')
                df_cruce['clasificacion_abc'] = df_cruce['clasificacion_abc'].fillna('Sin Clasificar')

                # --- PASO 6: VL06O (Consumos) ---
                df_vl06o = leer_archivo(file_vl06o)
                df_mov = df_vl06o[['Entrega', 'Cantidad entrega', 'Material']].copy()
                df_mov.columns = ['documento_entrega', 'cantidad', 'codigo_material']
                df_mov = df_mov.dropna(subset=['documento_entrega', 'codigo_material'])
                df_mov['documento_entrega'] = df_mov['documento_entrega'].astype(str).str.strip()
                df_mov['codigo_material'] = df_mov['codigo_material'].astype(str).str.strip()
                df_mov = df_mov[df_mov['documento_entrega'].str.startswith('801') | df_mov['documento_entrega'].str.startswith('84')].copy()
                df_mov['cantidad_consumo_real'] = df_mov.apply(lambda row: abs(row['cantidad']) if row['documento_entrega'].startswith('801') else -abs(row['cantidad']), axis=1)
                
                df_consumo = df_mov.groupby('codigo_material').agg(
                    consumo_total_historico=('cantidad_consumo_real', 'sum'),
                    frecuencia_consumo=('documento_entrega', 'nunique')
                ).reset_index()
                df_consumo['promedio_consumo_mensual'] = df_consumo['consumo_total_historico'] / 12

                df_cruce = pd.merge(df_cruce, df_consumo, on='codigo_material', how='left')
                df_cruce['promedio_consumo_mensual'] = df_cruce['promedio_consumo_mensual'].fillna(0)
                df_cruce['frecuencia_consumo'] = df_cruce['frecuencia_consumo'].fillna(0)

                # --- MACHINE LEARNING ---
                df_ml = df_cruce.copy()
                diccionario_abc = {'A': 1, 'B': 2, 'C': 3, 'Sin Clasificar': 4}
                df_ml['abc_numerico'] = df_ml['clasificacion_abc'].map(diccionario_abc).fillna(4)

                df_ml['demanda_durante_lead_time'] = (df_ml['promedio_consumo_mensual'] / 30) * df_ml['lead_time_dias']
                df_ml['objetivo_stock_realista'] = df_ml['demanda_durante_lead_time'] * 1.15

                columnas_x = ['lead_time_dias', 'costo_unitario', 'abc_numerico', 'promedio_consumo_mensual']
                X = df_ml[columnas_x]
                y = df_ml['objetivo_stock_realista']

                modelo_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                modelo_xgb.fit(X, y)

                df_ml['stock_sugerido_ia'] = modelo_xgb.predict(X)
                df_ml['stock_sugerido_ia'] = np.ceil(df_ml['stock_sugerido_ia']).clip(lower=0)

                # --- ESCUDOS PROTECTORES ---
                df_ml['stock_sugerido_ia'] = np.where(df_ml['promedio_consumo_mensual'] == 0, 0, df_ml['stock_sugerido_ia'])
                cond_esporadico = (~df_ml['tipo_material'].str.upper().str.contains('CRITICO|CRÍTICO')) & (df_ml['stock_seguridad_actual_3420'] == 0) & (df_ml['stock_seguridad_3400'] == 0) & (df_ml['frecuencia_consumo'] <= 1)
                df_ml['stock_sugerido_ia'] = np.where(cond_esporadico, 0, df_ml['stock_sugerido_ia'])
                df_ml['stock_sugerido_ia'] = np.where(df_ml['status_alemania'] != '', 0, df_ml['stock_sugerido_ia'])

                df_ml['stock_sugerido_ia'] = np.where(df_ml['valor_redondeo'] > 0, np.ceil(df_ml['stock_sugerido_ia'] / df_ml['valor_redondeo']) * df_ml['valor_redondeo'], df_ml['stock_sugerido_ia'])
                df_ml['stock_seguridad_FINAL_Kaeser'] = np.minimum(df_ml['stock_sugerido_ia'], df_ml['lote_minimo'].replace(0, 999999))

                df_ml['stock_seguridad_FINAL_Kaeser'] = np.where(
                    (df_ml['tipo_material'].str.upper().str.contains('CRITICO|CRÍTICO')) & 
                    (df_ml['stock_seguridad_FINAL_Kaeser'] < df_ml['stock_seguridad_actual_3420']) &
                    (df_ml['status_alemania'] == ''),
                    df_ml['stock_seguridad_actual_3420'],
                    df_ml['stock_seguridad_FINAL_Kaeser']
                )

                # --- DASHBOARD & ALERTAS ---
                denominador = df_ml['stock_seguridad_3400'] + df_ml['stock_seguridad_FINAL_Kaeser']
                df_ml['nivel_abastecimiento_pct'] = np.where(denominador > 0, (df_ml['stock_actual_3420'] / denominador) * 100, 100)
                
                def clasificar_alerta(pct):
                    if pct <= 25: return '🔴 Crítico'
                    elif pct <= 50: return '🟠 Bajo'
                    elif pct <= 80: return '🟡 Medio'
                    else: return '🟢 Alto'
                
                df_ml['alerta'] = df_ml['nivel_abastecimiento_pct'].apply(clasificar_alerta)
                
                df_ml['faltante'] = df_ml['stock_seguridad_FINAL_Kaeser'] - df_ml['stock_actual_3420']
                df_ml['sugerencia_pedido_urgente'] = np.where(
                    df_ml['faltante'] > 0,
                    np.ceil(df_ml['faltante'] / df_ml['valor_redondeo'].replace(0, 1)) * df_ml['valor_redondeo'].replace(0, 1),
                    0
                )

                st.session_state.df_procesado = df_ml
                st.success("¡Análisis completado! Modelo de datos 100% integrado.")

            except Exception as e:
                st.error(f"❌ Error al procesar: {e}.")
    else:
        st.info("⚠️ Sube los 6 archivos requeridos para iniciar el sistema.")

if st.session_state.df_procesado is not None:
    df_ml = st.session_state.df_procesado

    tab1, tab2 = st.tabs(["🚨 Dashboard Semanal", "🤖 Optimización IA (Maestro Completo)"])

    with tab1:
        st.subheader("Monitoreo de Nivel de Abastecimiento (Solo Repuestos)")
        c1, c2, c3, c4 = st.columns(4)
        c1.error(f"🔴 Críticos (<25%): {len(df_ml[df_ml['alerta'] == '🔴 Crítico'])}")
        c2.warning(f"🟠 Bajos (26-50%): {len(df_ml[df_ml['alerta'] == '🟠 Bajo'])}")
        c3.info(f"🟡 Medios (51-80%): {len(df_ml[df_ml['alerta'] == '🟡 Medio'])}")
        c4.success(f"🟢 Altos (>80%): {len(df_ml[df_ml['alerta'] == '🟢 Alto'])}")
        
        filtro_alerta = st.selectbox("Filtro de acción rápida:", ['Todos', '🔴 Crítico', '🟠 Bajo', '🟡 Medio', '🟢 Alto'])
        
        df_dashboard = df_ml[['codigo_material', 'descripcion', 'tipo_material', 'procedencia', 'stock_actual_3420', 'stock_seguridad_FINAL_Kaeser', 'nivel_abastecimiento_pct', 'alerta', 'sugerencia_pedido_urgente']].copy()
        df_dashboard['nivel_abastecimiento_pct'] = df_dashboard['nivel_abastecimiento_pct'].round(1).astype(str) + "%"
        
        if filtro_alerta != 'Todos':
            df_dashboard = df_dashboard[df_dashboard['alerta'] == filtro_alerta]
        
        st.dataframe(df_dashboard.sort_values(by='sugerencia_pedido_urgente', ascending=False), use_container_width=True)

    with tab2:
        st.subheader("Resultados Completos de la Inteligencia Artificial")
        columnas_finales = [
            'codigo_material', 'descripcion', 'tipo_material', 'procedencia', 'status_alemania', 'clasificacion_abc', 
            'promedio_consumo_mensual', 'frecuencia_consumo', 'lead_time_dias', 'stock_seguridad_actual_3420', 
            'lote_minimo', 'valor_redondeo', 'stock_seguridad_FINAL_Kaeser'
        ]
        df_resultado = df_ml[columnas_finales]
        st.dataframe(df_resultado, use_container_width=True)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_ml.to_excel(writer, index=False, sheet_name='Base Completa')
        
        st.download_button("📥 Descargar Excel Ejecutivo", data=output.getvalue(), file_name="Optimizacion_Kaeser_V4.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
