import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from io import BytesIO

st.set_page_config(page_title="Kaeser - IA Stock", layout="wide")

st.title("⚙️ Optimizador de Stock de Seguridad - Kaeser Medellín")
st.markdown("Sube los reportes crudos de SAP. El sistema cruzará las bases de datos, entrenará la IA y aplicará las reglas de negocio de Kaeser de forma automática.")

# 1. Zona de carga de archivos
col1, col2 = st.columns(2)
with col1:
    file_md07 = st.file_uploader("1. Archivo MD07 (Stock Medellín)", type=['xlsx'])
    file_vl06o = st.file_uploader("2. Archivo VL06O (Movimientos)", type=['xlsx'])
    file_zmd04 = st.file_uploader("3. Archivo ZMD04 (Stock Tenjo)", type=['xlsx'])
with col2:
    file_rmm = st.file_uploader("4. Archivo RMMDMDMA (Lotes Mínimos)", type=['xlsx'])
    file_mcbe = st.file_uploader("5. Archivo MCBE (Valor Monetario)", type=['xlsx'])
    file_plantilla = st.file_uploader("6. Plantilla K.MEDELLIN (Tipo y ABC)", type=['xlsx'])

st.markdown("---")

# 2. Botón de Ejecución Maestro
if st.button("🚀 Procesar Datos y Ejecutar Modelo IA", type="primary"):
    if file_md07 and file_vl06o and file_zmd04 and file_rmm and file_mcbe and file_plantilla:
        with st.spinner("Construyendo el Pipeline de Datos y entrenando la IA (Esto puede tomar un minuto)..."):
            try:
                # --- PASO 1: MD07 (Stock Medellín) ---
                df_md07 = pd.read_excel(file_md07)
                df_md07 = df_md07[['Material', 'Descripción del material', 'Stock de seguridad', 'Stock de centro']].copy()
                df_md07.columns = ['codigo_material', 'descripcion', 'stock_seguridad_actual_3420', 'stock_actual_3420']
                df_md07 = df_md07.dropna(subset=['codigo_material'])
                df_md07['codigo_material'] = df_md07['codigo_material'].astype(str).str.strip()

                # --- PASO 2: ZMD04 (Tenjo y Lead Time) ---
                df_zmd04 = pd.read_excel(file_zmd04)
                df_tenjo = df_zmd04[['Material', 'Stock de seguridad']].copy()
                df_tenjo.columns = ['codigo_material', 'stock_seguridad_3400']
                df_tenjo = df_tenjo.dropna(subset=['codigo_material'])
                df_tenjo['codigo_material'] = df_tenjo['codigo_material'].astype(str).str.strip()
                df_tenjo = df_tenjo[df_tenjo['codigo_material'].str.match(r'^[1-9]')].copy() # Solo importados

                df_cruce = pd.merge(df_md07, df_tenjo, on='codigo_material', how='left')
                df_cruce['stock_seguridad_3400'] = df_cruce['stock_seguridad_3400'].fillna(0)

                def calcular_lead_time(fila):
                    ss_medellin = fila['stock_seguridad_actual_3420']
                    ss_tenjo = fila['stock_seguridad_3400']
                    if ss_tenjo > 0: return 7
                    elif ss_tenjo == 0 and ss_medellin > 0: return 62
                    else: return 55

                df_cruce['lead_time_dias'] = df_cruce.apply(calcular_lead_time, axis=1)

                # --- PASO 3: VL06O (Movimientos y Consumos) ---
                df_vl06o = pd.read_excel(file_vl06o)
                df_mov = df_vl06o[['Entrega', 'Cantidad entrega', 'Material']].copy()
                df_mov.columns = ['documento_entrega', 'cantidad', 'codigo_material']
                df_mov = df_mov.dropna(subset=['documento_entrega', 'codigo_material'])
                df_mov['documento_entrega'] = df_mov['documento_entrega'].astype(str).str.strip()
                df_mov['codigo_material'] = df_mov['codigo_material'].astype(str).str.strip()
                
                # Reglas Kaeser para salidas (801) y devoluciones (84)
                df_mov = df_mov[df_mov['documento_entrega'].str.startswith('801') | df_mov['documento_entrega'].str.startswith('84')].copy()
                df_mov['cantidad_consumo_real'] = df_mov.apply(lambda row: abs(row['cantidad']) if row['documento_entrega'].startswith('801') else -abs(row['cantidad']), axis=1)
                
                df_consumo_total = df_mov.groupby('codigo_material')['cantidad_consumo_real'].sum().reset_index()
                df_consumo_total.columns = ['codigo_material', 'consumo_total_historico']
                df_consumo_total['promedio_consumo_mensual'] = df_consumo_total['consumo_total_historico'] / 12

                df_cruce = pd.merge(df_cruce, df_consumo_total, on='codigo_material', how='left')
                df_cruce['promedio_consumo_mensual'] = df_cruce['promedio_consumo_mensual'].fillna(0)

                # --- PASO 4: RMMDMDMA (Lotes y Redondeo) ---
                df_rmm = pd.read_excel(file_rmm)
                df_lotes = df_rmm[['Material', 'Valor de redondeo', 'Tamaño lote mínimo']].copy()
                df_lotes.columns = ['codigo_material', 'valor_redondeo', 'lote_minimo']
                df_lotes = df_lotes.dropna(subset=['codigo_material'])
                df_lotes['codigo_material'] = df_lotes['codigo_material'].astype(str).str.strip()
                
                df_cruce = pd.merge(df_cruce, df_lotes, on='codigo_material', how='left')
                df_cruce['valor_redondeo'] = df_cruce['valor_redondeo'].fillna(0)
                df_cruce['lote_minimo'] = df_cruce['lote_minimo'].fillna(0)

                # --- PASO 5: MCBE (Costo Unitario) ---
                df_mcbe = pd.read_excel(file_mcbe)
                df_valor = df_mcbe[['P/N', 'CtdStkTot.', 'ValStkVal']].copy()
                df_valor.columns = ['codigo_material', 'cantidad_total', 'valor_total']
                df_valor = df_valor.dropna(subset=['codigo_material'])
                df_valor['codigo_material'] = df_valor['codigo_material'].astype(str).str.strip()
                df_valor = df_valor.drop_duplicates(subset=['codigo_material'])
                
                df_valor['cantidad_total'] = df_valor['cantidad_total'].fillna(0)
                df_valor['valor_total'] = df_valor['valor_total'].fillna(0)
                df_valor['costo_unitario'] = np.where(df_valor['cantidad_total'] > 0, df_valor['valor_total'] / df_valor['cantidad_total'], 0)
                
                df_cruce = pd.merge(df_cruce, df_valor[['codigo_material', 'costo_unitario']], on='codigo_material', how='left')
                df_cruce['costo_unitario'] = df_cruce['costo_unitario'].fillna(0)

                # --- PASO 6: PLANTILLA (ABC y TIPO) ---
                df_plantilla_k = pd.read_excel(file_plantilla, sheet_name='3420', skiprows=8)
                df_tipo = df_plantilla_k[['NUMERO DE PARTE', 'ABC', 'TIPO']].copy()
                df_tipo.columns = ['codigo_material', 'clasificacion_abc', 'tipo_material']
                df_tipo = df_tipo.dropna(subset=['codigo_material'])
                df_tipo['codigo_material'] = df_tipo['codigo_material'].astype(str).str.strip()
                
                df_cruce = pd.merge(df_cruce, df_tipo, on='codigo_material', how='left')
                df_cruce['clasificacion_abc'] = df_cruce['clasificacion_abc'].fillna('Sin Clasificar')
                df_cruce['tipo_material'] = df_cruce['tipo_material'].fillna('Otros')

                # --- PASO 7: MACHINE LEARNING (XGBOOST) ---
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

                # --- PASO 8: REGLAS LOGÍSTICAS FINALES KAESER ---
                df_ml['stock_sugerido_ia'] = np.where(
                    df_ml['valor_redondeo'] > 0,
                    np.ceil(df_ml['stock_sugerido_ia'] / df_ml['valor_redondeo']) * df_ml['valor_redondeo'],
                    df_ml['stock_sugerido_ia']
                )

                df_ml['stock_seguridad_FINAL_Kaeser'] = np.minimum(
                    df_ml['stock_sugerido_ia'], 
                    df_ml['lote_minimo'].replace(0, 999999)
                )

                df_ml['stock_seguridad_FINAL_Kaeser'] = np.where(
                    (df_ml['tipo_material'].str.upper().str.contains('CRITICO|CRÍTICO')) & 
                    (df_ml['stock_seguridad_FINAL_Kaeser'] < df_ml['stock_seguridad_actual_3420']),
                    df_ml['stock_seguridad_actual_3420'],
                    df_ml['stock_seguridad_FINAL_Kaeser']
                )

                # --- PASO 9: PINTAR RESULTADOS ---
                st.success("¡Análisis completado con éxito! 🎉")
                
                columnas_finales = [
                    'codigo_material', 'descripcion', 'tipo_material', 'clasificacion_abc', 
                    'promedio_consumo_mensual', 'lead_time_dias', 'stock_seguridad_actual_3420', 
                    'lote_minimo', 'valor_redondeo', 'stock_sugerido_ia', 'stock_seguridad_FINAL_Kaeser'
                ]
                df_resultado = df_ml[columnas_finales]
                
                st.subheader("📊 Resultados de la Inteligencia Artificial")
                st.dataframe(df_resultado, use_container_width=True)

                # Botón mágico para descargar todo a Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_resultado.to_excel(writer, index=False, sheet_name='Sugerencias IA')
                
                st.download_button(
                    label="📥 Descargar Excel con Resultados Finales",
                    data=output.getvalue(),
                    file_name="Optimizacion_Stock_Kaeser.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"❌ Ocurrió un error al procesar los archivos: {e}")
                st.info("Revisa que los archivos sean exactamente los exportados de SAP y correspondan a cada cajita.")
    else:
        st.warning("⚠️ Por favor, sube los 6 archivos antes de iniciar el motor de IA.")
